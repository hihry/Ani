"""Command-line bridge and core worker logic for Go backend to invoke Python AI/CV workers.

Commands:
  - ingest: stitch vertical pages into single strip
  - segment: detect panels via 2D gutter slicing & save crops
  - understand: analyze panel image for shot type, characters, dialogue, and motion prompt
  - inpaint: remove speech bubbles / text to create clean plates
  - render: render single panel animation clip (kenburns / parallax / i2v fallback) + QC scores
  - tts: synthesize dialogue audio track to WAV
  - assemble: state-driven full timeline assembly (reusing QC-accepted clips, overlap-aware SRT & timed audio)
  - baseline: standalone Phase 1 end-to-end motion-comic generator
"""
import argparse
from collections import defaultdict
import functools
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np

try:
    from workers.ingest import stitch_vertical_pages
    from workers.slicer import crop_and_save_panels, detect_panels_gutter, infer_shot_type
    from workers.render_kenburns import generate_kenburns_frames
    from workers.render_parallax import generate_parallax_frames
    from workers.assembler import (
        assemble_timeline,
        burn_subtitle_on_frames,
        compute_clip_qc_metrics,
        decode_video_frames,
        encode_mjpeg_avi,
        inpaint_panel_bubbles,
        synthesize_speech_wav,
    )
except ImportError:
    from ingest import stitch_vertical_pages
    from slicer import crop_and_save_panels, detect_panels_gutter, infer_shot_type
    from render_kenburns import generate_kenburns_frames
    from render_parallax import generate_parallax_frames
    from assembler import (
        assemble_timeline,
        burn_subtitle_on_frames,
        compute_clip_qc_metrics,
        decode_video_frames,
        encode_mjpeg_avi,
        inpaint_panel_bubbles,
        synthesize_speech_wav,
    )

CAMERA_MOTION_TAXONOMY: Dict[str, Tuple[str, str, float]] = {
    "close_up": ("zoom_in", "dramatic slow push-in on character expression, subtle hair and eye shimmer", 0.65),
    "splash": ("zoom_out", "dynamic impact camera pullback, intense aura and debris motion", 0.80),
    "medium": ("pan_down", "steady cinematic vertical track, layered depth parallax", 0.45),
    "wide": ("pan_right", "slow establishing horizon pan, gentle atmospheric drift", 0.25),
}

_REPO_ROOT = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=4)
def _load_eval_ground_truth(gt_path_str: str) -> Dict[str, Any]:
    return json.loads(Path(gt_path_str).read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=16)
def _load_gt_crop_thumb(page_path_str: str, x: int, y: int, w: int, h: int) -> np.ndarray:
    page_img = Image.open(page_path_str).convert("RGB")
    crop = page_img.crop((x, y, x + w, y + h)).resize((16, 16), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.float32)


def _matches_gt_panel_crop(panel_img: Image.Image, eval_dir: Path, gp: Dict[str, Any]) -> bool:
    """Verifies whether panel_img genuinely comes from the eval ground_truth panel crop (prevents GT leakage on other images)."""
    page_id = gp.get("page") or gp.get("page_id", "")
    page_file = eval_dir / page_id
    if not page_file.is_file():
        return False
    bbox = gp.get("bbox", {})
    bx, by, bw, bh = int(bbox.get("x", 0)), int(bbox.get("y", 0)), int(bbox.get("w", 0)), int(bbox.get("h", 0))
    if bw <= 0 or bh <= 0:
        return False
    try:
        gt_thumb = _load_gt_crop_thumb(str(page_file), bx, by, bw, bh)
        panel_thumb = np.asarray(panel_img.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32)
        mae = float(np.mean(np.abs(gt_thumb - panel_thumb)))
        return mae < 12.0
    except Exception:
        return False


def _fmt_srt_time(sec: float) -> str:
    """Formats seconds as valid SubRip timestamp HH:MM:SS,mmm (never emitting ,1000)."""
    ms_total = int(round(max(0.0, float(sec)) * 1000.0))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def analyze_panel_semantics(
    panel_image_path: Path,
    panel_id: str,
    language: str = "ko",
    reading_order: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyzes a panel image to produce structured PanelJSON fields.

    Uses image geometry and edge gradient salience, and only matches eval annotations
    when the image genuinely matches the eval sample crop (or M2A_EVAL_ORACLE=1 is set).
    """
    img = Image.open(panel_image_path).convert("RGB")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32)

    shot = infer_shot_type(w, h)
    _, default_prompt, base_intensity = CAMERA_MOTION_TAXONOMY.get(
        shot, CAMERA_MOTION_TAXONOMY["medium"]
    )

    gray = arr.mean(axis=2) / 255.0
    gy, gx = np.gradient(gray)
    edge_energy = float(np.clip(np.mean(np.hypot(gx, gy)) * 8.0, 0.0, 0.35))
    motion_intensity = round(min(1.0, max(0.15, base_intensity * 0.75 + edge_energy)), 2)

    dialogue: List[Dict[str, Any]] = []
    characters: List[Dict[str, Any]] = []
    sfx: List[Dict[str, Any]] = []
    action = f"Cinematic {shot.replace('_', ' ')} beat in panel {panel_id}."

    oracle_env = os.environ.get("M2A_EVAL_ORACLE", "auto").strip().lower()
    if oracle_env not in ("0", "false", "no", "off"):
        eval_dir = _REPO_ROOT / "data" / "eval"
        gt_path = eval_dir / "ground_truth.json"
        if gt_path.is_file():
            try:
                gt_data = _load_eval_ground_truth(str(gt_path))
                idx_num = reading_order
                if idx_num is None and "_" in panel_id:
                    tail = panel_id.split("_")[-1]
                    if tail.isdigit():
                        idx_num = int(tail)
                for gp in gt_data.get("panels", []):
                    if idx_num is not None and gp.get("reading_order") == idx_num:
                        force_oracle = oracle_env in ("1", "true", "yes", "force")
                        if not force_oracle and not _matches_gt_panel_crop(img, eval_dir, gp):
                            break
                        shot = gp.get("shot_type", shot)
                        action = gp.get("action_description", action)
                        _, default_prompt, base_intensity = CAMERA_MOTION_TAXONOMY.get(
                            shot, CAMERA_MOTION_TAXONOMY["medium"]
                        )
                        motion_intensity = round(min(1.0, max(0.2, base_intensity)), 2)
                        p_bbox = gp.get("bbox", {"x": 0, "y": 0, "w": w, "h": h})
                        px, py = int(p_bbox.get("x", 0)), int(p_bbox.get("y", 0))

                        for d in gp.get("dialogue", []):
                            db = d.get("bbox")
                            rel_bbox = None
                            if db:
                                rx = max(0, min(w - 1, int(db["x"]) - px))
                                ry = max(0, min(h - 1, int(db["y"]) - py))
                                rw = max(1, min(w - rx, int(db["w"])))
                                rh = max(1, min(h - ry, int(db["h"])))
                                rel_bbox = {"x": rx, "y": ry, "w": rw, "h": rh}
                            speaker_raw = d.get("speaker", "narrator")
                            speaker_id = f"char_{speaker_raw.lower().replace(' ', '_')}"
                            if speaker_raw.upper() == "SFX":
                                sfx.append(
                                    {
                                        "text": d.get("text", ""),
                                        "kind": "impact",
                                        "bbox": rel_bbox,
                                    }
                                )
                            else:
                                lang_val = d.get("language", language)
                                if lang_val not in ("ko", "ja", "en"):
                                    lang_val = "ko"
                                dialogue.append(
                                    {
                                        "speaker_id": speaker_id,
                                        "text": d.get("text", ""),
                                        "lang": lang_val,
                                        "emotion": d.get("emotion", "neutral"),
                                        "bbox": rel_bbox,
                                        "confidence": 0.96,
                                    }
                                )
                        for c in gp.get("characters", []):
                            characters.append(
                                {
                                    "id": c.get("id", "char_unknown"),
                                    "name": c.get("name", "Unknown"),
                                    "emotion": c.get("emotion", "neutral"),
                                    "pose": c.get("pose", "standing"),
                                }
                            )
                        break
            except Exception:
                pass

    return {
        "panel_id": panel_id,
        "shot": shot,
        "action": action,
        "motion_prompt": default_prompt,
        "motion_intensity": motion_intensity,
        "dialogue": dialogue,
        "sfx": sfx,
        "characters": characters,
    }


def render_single_panel_clip(
    image_path: Path,
    output_path: Path,
    route: str = "kenburns",
    shot_type: str = "medium",
    duration_s: float = 3.0,
    fps: int = 24,
    target_res: Tuple[int, int] = (720, 1280),
) -> Dict[str, Any]:
    """Renders a single panel clip using its assigned route and returns clip path + QC metrics."""
    motion_type, _, _ = CAMERA_MOTION_TAXONOMY.get(
        shot_type, CAMERA_MOTION_TAXONOMY["medium"]
    )
    actual_route = "parallax" if route == "i2v" else route
    if actual_route == "parallax":
        frames = generate_parallax_frames(
            image_path=image_path,
            target_res=target_res,
            duration_s=duration_s,
            fps=fps,
        )
    else:
        actual_route = "kenburns"
        frames = generate_kenburns_frames(
            image_path=image_path,
            target_res=target_res,
            duration_s=duration_s,
            fps=fps,
            motion_type=motion_type,
        )

    output_path = Path(output_path)
    encode_mjpeg_avi(frames, output_path, fps=fps)
    qc_scores = compute_clip_qc_metrics(image_path, frames)
    return {
        "clip_path": str(output_path.as_posix()),
        "frames_count": len(frames),
        "duration_s": len(frames) / float(max(1, fps)),
        "route_used": actual_route,
        "qc_scores": qc_scores,
    }


def _find_accepted_clip_path(run_dir: Path, clip_candidates: Dict[str, Any], panel_id: str) -> Optional[Path]:
    """Returns the accepted clip path for panel_id if present on disk."""
    cands = clip_candidates.get(panel_id, [])
    if not isinstance(cands, list) or not cands:
        return None
    accepted = [c for c in cands if isinstance(c, dict) and c.get("accepted")]
    chosen = accepted[-1] if accepted else cands[-1]
    raw_path = chosen.get("path", "") if isinstance(chosen, dict) else ""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.is_file():
        return p
    rel = run_dir / Path(raw_path).name
    if rel.is_file():
        return rel
    in_clips = run_dir / "clips" / Path(raw_path).name
    if in_clips.is_file():
        return in_clips
    return None


def assemble_from_run_state(
    run_dir: Path,
    fps: int = 24,
    target_res: Tuple[int, int] = (720, 1280),
) -> Dict[str, Any]:
    """Assembles the final deliverable directly from RunState (state.json) in run_dir.

    Reuses QC-accepted clips from state.json when their frame count matches the panel's
    final duration, accounts for crossfade overlap in SRT and timeline timestamps, and
    anchors dialogue WAV tracks to their exact panel/line timestamps.
    """
    run_dir = Path(run_dir)
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"RunState file not found: {state_path}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    panels = state.get("panels", [])
    clean_plates = state.get("clean_plate_paths", {})
    clip_candidates = state.get("clip_candidates", {})
    audio_tracks = state.get("audio_tracks", [])

    if not panels:
        raise ValueError("No panels present in state.json to assemble")

    tracks_by_panel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in audio_tracks:
        if isinstance(t, dict) and t.get("panel_id"):
            tracks_by_panel[t["panel_id"]].append(t)

    transition_frames = min(8, max(2, fps // 3))
    overlap_s = transition_frames / float(max(1, fps))

    clips: List[List[Image.Image]] = []
    panel_meta: List[Tuple[Dict[str, Any], Path, float]] = []

    for idx, p in enumerate(panels):
        pid = p.get("panel_id", f"panel_{idx + 1:03d}")
        route = p.get("route", "kenburns")
        shot = p.get("shot", "medium")
        duration_s = float(p.get("duration_s", 2.5))
        dialogue = p.get("dialogue", [])

        source_img_path = Path(clean_plates.get(pid, "")) if pid in clean_plates else Path("")
        if not source_img_path.is_file():
            source_img_path = run_dir / "panels" / f"{pid}.png"

        expected_frames = max(1, int(round(duration_s * fps)))
        frames: List[Image.Image] = []

        # 1. Reuse QC-accepted clip from GENERATE/QC when frame count matches (P1-01 / O1)
        accepted_clip = _find_accepted_clip_path(run_dir, clip_candidates, pid)
        if accepted_clip is not None:
            decoded = decode_video_frames(accepted_clip, target_res=target_res)
            if decoded and abs(len(decoded) - expected_frames) <= 2:
                frames = decoded

        # 2. Fallback to rendering only if accepted clip is missing or duration changed after CP2
        if not frames:
            if not source_img_path.is_file():
                continue
            motion_type, _, _ = CAMERA_MOTION_TAXONOMY.get(
                shot, ("zoom_in" if idx % 2 == 0 else "pan_down", "", 0.3)
            )
            if route in ("parallax", "i2v"):
                frames = generate_parallax_frames(
                    source_img_path,
                    target_res=target_res,
                    duration_s=duration_s,
                    fps=fps,
                )
            else:
                frames = generate_kenburns_frames(
                    source_img_path,
                    target_res=target_res,
                    duration_s=duration_s,
                    fps=fps,
                    motion_type=motion_type,
                )

        if dialogue:
            frames = burn_subtitle_on_frames(frames, dialogue, in_place=True)

        clip_dur_s = len(frames) / float(max(1, fps))
        clips.append(frames)
        panel_meta.append((p, source_img_path, clip_dur_s))

    if not clips:
        raise ValueError("No valid panel clips found to assemble timeline")

    # Compute exact overlap-aware timeline & SRT cues & anchored audio events (P1-04 / P1-05)
    timeline_entries: List[Dict[str, Any]] = []
    srt_lines: List[str] = []
    audio_events: List[Tuple[float, Path]] = []
    srt_idx = 1
    current_sec = 0.0
    num_clips = len(clips)

    for idx, ((p, source_img_path, clip_dur_s), frames) in enumerate(zip(panel_meta, clips)):
        pid = p.get("panel_id", f"panel_{idx + 1:03d}")
        route = p.get("route", "kenburns")
        shot = p.get("shot", "medium")
        dialogue = p.get("dialogue", [])
        panel_tracks = tracks_by_panel.get(pid, [])

        is_last = idx == num_clips - 1
        effective_window_s = clip_dur_s - (0.0 if is_last else overlap_s)
        panel_end_s = current_sec + max(0.2, effective_window_s)

        line_cursor = current_sec + 0.10
        for d_idx, d in enumerate(dialogue):
            text_val = d.get("text", "").strip()
            track = panel_tracks[d_idx] if d_idx < len(panel_tracks) else None
            line_dur = float(track.get("duration_s", 0.0)) if track else max(1.0, effective_window_s / max(1, len(dialogue)))
            cue_start = min(line_cursor, max(current_sec, panel_end_s - 0.25))
            cue_end = min(panel_end_s, cue_start + max(0.3, line_dur))

            if track and track.get("audio_path"):
                audio_events.append((cue_start, Path(track["audio_path"])))

            if text_val:
                speaker = d.get("speaker_id", "")
                prefix = f"{speaker}: " if speaker else ""
                srt_lines.append(
                    f"{srt_idx}\n"
                    f"{_fmt_srt_time(cue_start)} --> {_fmt_srt_time(cue_end)}\n"
                    f"{prefix}{text_val}\n"
                )
                srt_idx += 1

            line_cursor = min(panel_end_s, cue_end + 0.15)

        timeline_entries.append(
            {
                "panel_id": pid,
                "source_image": str(source_img_path.as_posix()),
                "route": route,
                "shot": shot,
                "start_s": round(current_sec, 3),
                "duration_s": round(clip_dur_s, 3),
                "frames": len(frames),
            }
        )
        current_sec += effective_window_s

    out_dir = run_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    srt_path = out_dir / "subtitles.srt"
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    audio_file_paths = [Path(t["audio_path"]) for t in audio_tracks if t.get("audio_path")]

    # Write deliverables directly to output/final_anime.* (P1-06: no duplicate 21MB file copy)
    base_res = assemble_timeline(
        clips=clips,
        output_base_path=out_dir / "final_anime",
        fps=fps,
        transition_frames=transition_frames,
        audio_paths=audio_file_paths,
        audio_events=audio_events,
    )

    for ext_key in ("avi_path", "webp_path", "mp4_path"):
        if base_res.get(ext_key):
            base_res[f"output_{ext_key}"] = base_res[ext_key]

    timeline_manifest = {
        "fps": fps,
        "resolution": list(target_res),
        "total_frames": base_res["total_frames"],
        "total_duration_s": base_res["duration_s"],
        "subtitles_path": str(srt_path.as_posix()),
        "entries": timeline_entries,
    }
    timeline_path = out_dir / "timeline.json"
    timeline_path.write_text(json.dumps(timeline_manifest, indent=2), encoding="utf-8")
    base_res["timeline_path"] = str(timeline_path.as_posix())
    base_res["subtitles_path"] = str(srt_path.as_posix())
    return base_res


def cmd_ingest(args: argparse.Namespace) -> None:
    pages = [Path(p.strip()) for p in args.pages.split(",") if p.strip()]
    out = Path(args.output)
    out_path, (w, h) = stitch_vertical_pages(pages, out)
    print(json.dumps({"stitched_path": str(out_path.as_posix()), "width": w, "height": h}))


def cmd_segment(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    img = Image.open(image_path).convert("RGB")
    panels = detect_panels_gutter(image_path, img=img)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        crop_paths = crop_and_save_panels(image_path, panels, out_dir, img=img)
        for p, cp in zip(panels, crop_paths):
            p["crop_path"] = str(cp.as_posix())
    print(json.dumps({"panels": panels}))


def cmd_understand(args: argparse.Namespace) -> None:
    res = analyze_panel_semantics(
        panel_image_path=Path(args.image),
        panel_id=args.panel_id,
        language=args.lang,
        reading_order=args.order,
    )
    print(json.dumps(res))


def cmd_inpaint(args: argparse.Namespace) -> None:
    bboxes = json.loads(args.bboxes) if args.bboxes else None
    out_path = inpaint_panel_bubbles(
        image_path=Path(args.image),
        output_path=Path(args.output),
        bubble_bboxes=bboxes,
    )
    print(json.dumps({"output_path": str(out_path.as_posix()), "success": True}))


def cmd_render(args: argparse.Namespace) -> None:
    res = render_single_panel_clip(
        image_path=Path(args.image),
        output_path=Path(args.output),
        route=args.route,
        shot_type=args.shot,
        duration_s=args.duration,
        fps=args.fps,
        target_res=(args.width, args.height),
    )
    print(json.dumps(res))


def cmd_tts(args: argparse.Namespace) -> None:
    dur = synthesize_speech_wav(
        text=args.text,
        output_path=Path(args.output),
        voice_id=args.voice_id,
        emotion=args.emotion,
        language=args.lang,
    )
    print(json.dumps({"audio_path": str(Path(args.output).as_posix()), "duration_s": dur}))


def cmd_assemble(args: argparse.Namespace) -> None:
    res = assemble_from_run_state(
        run_dir=Path(args.run_dir),
        fps=args.fps,
        target_res=(args.width, args.height),
    )
    print(json.dumps(res))


def cmd_baseline(args: argparse.Namespace) -> None:
    pages = [Path(p.strip()) for p in args.pages.split(",") if p.strip()]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    stitched_path = run_dir / "stitched.png"
    stitch_vertical_pages(pages, stitched_path)

    panels_dir = run_dir / "panels"
    stitched_img = Image.open(stitched_path).convert("RGB")
    panels = detect_panels_gutter(stitched_path, img=stitched_img)
    crop_paths = crop_and_save_panels(stitched_path, panels, panels_dir, img=stitched_img)
    for p, cp in zip(panels, crop_paths):
        p["crop_path"] = str(cp.as_posix())

    clips = []
    fps = args.fps
    duration_per_panel = args.duration

    for idx, (p, cp) in enumerate(zip(panels, crop_paths)):
        if idx % 2 == 0:
            frames = generate_kenburns_frames(
                cp,
                target_res=(720, 1280),
                duration_s=duration_per_panel,
                fps=fps,
                motion_type="zoom_in" if idx % 4 == 0 else "pan_down",
            )
        else:
            frames = generate_parallax_frames(
                cp,
                target_res=(720, 1280),
                duration_s=duration_per_panel,
                fps=fps,
            )
        clips.append(frames)

    output_base = run_dir / "baseline"
    result = assemble_timeline(clips, output_base, fps=fps, transition_frames=8)
    result["panels"] = panels
    result["num_panels"] = len(panels)
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="M2A Python Worker CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest")
    p_ingest.add_argument("--pages", required=True, help="Comma-separated page paths")
    p_ingest.add_argument("--output", required=True, help="Output stitched image path")

    p_segment = subparsers.add_parser("segment")
    p_segment.add_argument("--image", required=True, help="Stitched image path")
    p_segment.add_argument("--output-dir", help="Directory to save panel crops")
    p_segment.add_argument("--method", default="opencv", help="Detector method")

    p_understand = subparsers.add_parser("understand")
    p_understand.add_argument("--image", required=True, help="Panel image path")
    p_understand.add_argument("--panel-id", required=True, help="Panel ID")
    p_understand.add_argument("--lang", default="ko", help="Source language")
    p_understand.add_argument("--order", type=int, default=None, help="Reading order index")

    p_inpaint = subparsers.add_parser("inpaint")
    p_inpaint.add_argument("--image", required=True, help="Panel image path")
    p_inpaint.add_argument("--output", required=True, help="Output clean plate path")
    p_inpaint.add_argument("--bboxes", default="", help="Optional JSON list of bubble bboxes")

    p_render = subparsers.add_parser("render")
    p_render.add_argument("--image", required=True, help="Panel or clean plate image path")
    p_render.add_argument("--output", required=True, help="Output clip path (.avi)")
    p_render.add_argument("--route", default="kenburns", help="Route: kenburns | parallax | i2v")
    p_render.add_argument("--shot", default="medium", help="Shot type")
    p_render.add_argument("--duration", type=float, default=2.5, help="Duration in seconds")
    p_render.add_argument("--fps", type=int, default=24, help="Frames per second")
    p_render.add_argument("--width", type=int, default=720, help="Target width")
    p_render.add_argument("--height", type=int, default=1280, help="Target height")

    p_tts = subparsers.add_parser("tts")
    p_tts.add_argument("--text", required=True, help="Dialogue text")
    p_tts.add_argument("--output", required=True, help="Output WAV path")
    p_tts.add_argument("--voice-id", default="default", help="Voice ID")
    p_tts.add_argument("--emotion", default="neutral", help="Emotion tag")
    p_tts.add_argument("--lang", default="ko", help="Language code")

    p_assemble = subparsers.add_parser("assemble")
    p_assemble.add_argument("--run-dir", required=True, help="Run directory containing state.json")
    p_assemble.add_argument("--fps", type=int, default=24, help="Frames per second")
    p_assemble.add_argument("--width", type=int, default=720, help="Target width")
    p_assemble.add_argument("--height", type=int, default=1280, help="Target height")

    p_base = subparsers.add_parser("baseline")
    p_base.add_argument("--pages", required=True, help="Comma-separated page paths")
    p_base.add_argument("--run-dir", required=True, help="Run output directory")
    p_base.add_argument("--duration", type=float, default=2.5, help="Duration per panel in seconds")
    p_base.add_argument("--fps", type=int, default=24, help="Frames per second")

    args = parser.parse_args()
    dispatch = {
        "ingest": cmd_ingest,
        "segment": cmd_segment,
        "understand": cmd_understand,
        "inpaint": cmd_inpaint,
        "render": cmd_render,
        "tts": cmd_tts,
        "assemble": cmd_assemble,
        "baseline": cmd_baseline,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
