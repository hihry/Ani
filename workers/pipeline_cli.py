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
    from workers.understanding import (
        CAMERA_MOTION_TAXONOMY,
        analyze_panel,
    )
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
    from understanding import (
        CAMERA_MOTION_TAXONOMY,
        analyze_panel,
    )
    from assembler import (
        assemble_timeline,
        burn_subtitle_on_frames,
        compute_clip_qc_metrics,
        decode_video_frames,
        encode_mjpeg_avi,
        inpaint_panel_bubbles,
        synthesize_speech_wav,
    )


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
    crops_dir: Optional[Path] = None,
    annotated_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyzes a panel image to produce structured PanelJSON fields via Phase 2 understanding engine."""
    return analyze_panel(
        panel_image_path=Path(panel_image_path),
        panel_id=panel_id,
        language=language,
        reading_order=reading_order,
        crops_dir=crops_dir,
        annotated_path=annotated_path,
    )


def render_single_panel_clip(
    image_path: Path,
    output_path: Path,
    route: str = "kenburns",
    shot_type: str = "medium",
    duration_s: float = 3.0,
    fps: int = 24,
    target_res: Tuple[int, int] = (720, 1280),
    panel_id: str = "",
    model_id: str = "",
    seed: int = 42,
    action_prompt: str = "",
    motion_type_override: str = "",
    emotion: str = "neutral",
    comfy_url: str = "",
    cancel_event: Optional[Any] = None,
) -> Dict[str, Any]:
    """Renders a single panel clip using its assigned route and returns clip path + QC metrics."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pid = panel_id or image_path.stem

    default_motion, _, _ = CAMERA_MOTION_TAXONOMY.get(
        shot_type, CAMERA_MOTION_TAXONOMY["medium"]
    )
    motion_type = motion_type_override if motion_type_override in (
        "zoom_in", "zoom_out", "pan_up", "pan_down", "pan_left", "pan_right"
    ) else default_motion

    resolved_comfy = (comfy_url or os.environ.get("M2A_COMFY_URL", "")).strip()
    if route == "i2v" and resolved_comfy:
        from workers.video.comfy_client import ComfyClient
        from workers.video.provider import ComfyVideoProvider, load_model_specs

        specs = load_model_specs()
        eff_model = model_id if model_id in specs else "wan22-ti2v-5b"
        provider = ComfyVideoProvider(
            spec=specs[eff_model],
            client=ComfyClient(base_url=resolved_comfy),
        )
        gen_res = provider.generate(
            panel={
                "panel_id": pid,
                "action_prompt": action_prompt,
                "motion_type": motion_type,
                "emotion": emotion,
            },
            clean_plate=image_path,
            duration_s=duration_s,
            seed=seed,
            out_dir=output_path.parent,
            target_res=target_res,
            fps=fps,
            cancel_event=cancel_event,
        )
        if gen_res.clip_path != output_path:
            import shutil
            shutil.copy2(gen_res.clip_path, output_path)
        frames = decode_video_frames(output_path, target_res=target_res)
        qc_scores = compute_clip_qc_metrics(image_path, frames, route="i2v")
        return {
            "clip_path": str(output_path.as_posix()),
            "raw_clip_path": str(gen_res.raw_clip_path.as_posix()),
            "frames_count": len(frames),
            "duration_s": len(frames) / float(max(1, fps)),
            "route_used": "i2v",
            "model_id": gen_res.model_id,
            "seed": gen_res.seed,
            "positive_prompt": gen_res.positive_prompt,
            "negative_prompt": gen_res.negative_prompt,
            "native_width": gen_res.native_width,
            "native_height": gen_res.native_height,
            "native_frames": gen_res.native_frames,
            "native_fps": gen_res.native_fps,
            "gen_seconds": gen_res.gen_seconds,
            "peak_vram_gb": gen_res.peak_vram_gb,
            "cached": gen_res.cached,
            "qc_scores": qc_scores,
        }

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

    encode_mjpeg_avi(frames, output_path, fps=fps)
    qc_scores = compute_clip_qc_metrics(image_path, frames, route=actual_route)
    return {
        "clip_path": str(output_path.as_posix()),
        "frames_count": len(frames),
        "duration_s": len(frames) / float(max(1, fps)),
        "route_used": actual_route,
        "model_id": model_id or actual_route,
        "seed": int(seed),
        "qc_scores": qc_scores,
    }


def _find_accepted_clip_candidate(
    run_dir: Path,
    clip_candidates: Dict[str, Any],
    panel_id: str,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Returns `(clip_path, candidate_dict)` for panel_id if present on disk."""
    cands = clip_candidates.get(panel_id, [])
    if not isinstance(cands, list) or not cands:
        return None, None
    accepted = [c for c in cands if isinstance(c, dict) and c.get("accepted")]
    chosen = accepted[-1] if accepted else cands[-1]
    if not isinstance(chosen, dict):
        return None, None
    raw_path = chosen.get("path") or chosen.get("clip_path") or ""
    if not raw_path:
        return None, chosen
    p = Path(raw_path)
    if p.is_file():
        return p, chosen
    rel = run_dir / Path(raw_path).name
    if rel.is_file():
        return rel, chosen
    in_clips = run_dir / "clips" / Path(raw_path).name
    if in_clips.is_file():
        return in_clips, chosen
    return None, chosen


def _find_accepted_clip_path(run_dir: Path, clip_candidates: Dict[str, Any], panel_id: str) -> Optional[Path]:
    """Returns the accepted clip path for panel_id if present on disk."""
    clip_p, _ = _find_accepted_clip_candidate(run_dir, clip_candidates, panel_id)
    return clip_p


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
        #    For accepted I2V clips (M0-2 / E7), a frame count mismatch (> 2 frames) is a hard error
        #    and MUST NOT silently re-render as parallax.
        accepted_clip, chosen_cand = _find_accepted_clip_candidate(run_dir, clip_candidates, pid)
        cand_route = str((chosen_cand or {}).get("route") or route).lower()
        if accepted_clip is not None:
            decoded = decode_video_frames(accepted_clip, target_res=target_res)
            if decoded and abs(len(decoded) - expected_frames) <= 2:
                frames = decoded
            elif cand_route == "i2v" and (chosen_cand or {}).get("accepted"):
                raise ValueError(
                    f"I2V clip frame mismatch for {pid}: decoded {len(decoded)} frames, "
                    f"expected {expected_frames} (E7)"
                )
        elif cand_route == "i2v" and (chosen_cand or {}).get("accepted"):
            raise ValueError(f"Accepted I2V clip file missing for {pid}: {chosen_cand} (E7)")

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

        num_lines = len(dialogue)
        if num_lines > 0:
            raw_durs: List[float] = []
            for d_idx in range(num_lines):
                track = panel_tracks[d_idx] if d_idx < len(panel_tracks) else None
                d_val = float(track.get("duration_s", 0.0)) if track else max(0.8, effective_window_s / num_lines)
                raw_durs.append(max(0.25, d_val))

            lead_pad = min(0.10, max(0.0, effective_window_s * 0.05))
            avail_s = max(0.15 * num_lines, (panel_end_s - current_sec) - lead_pad)
            gap_s = min(0.12, avail_s * 0.08 / max(1, num_lines - 1)) if num_lines > 1 else 0.0
            speech_budget = max(0.12 * num_lines, avail_s - gap_s * (num_lines - 1))
            raw_sum = sum(raw_durs)
            scale = min(1.0, speech_budget / max(1e-6, raw_sum))

            line_cursor = current_sec + lead_pad
            for d_idx, d in enumerate(dialogue):
                text_val = d.get("text", "").strip()
                track = panel_tracks[d_idx] if d_idx < len(panel_tracks) else None
                scaled_dur = max(0.12, raw_durs[d_idx] * scale)
                cue_start = min(line_cursor, max(current_sec, panel_end_s - 0.12 * (num_lines - d_idx)))
                cue_end = min(panel_end_s, max(cue_start + 0.10, cue_start + scaled_dur))

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

                line_cursor = min(panel_end_s, cue_end + gap_s)

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
        crops_dir=Path(args.crops_dir) if getattr(args, "crops_dir", None) else None,
        annotated_path=Path(args.annotated) if getattr(args, "annotated", None) else None,
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
        model_id=getattr(args, "model_id", "") or "",
        seed=int(getattr(args, "seed", 42) or 42),
        action_prompt=getattr(args, "prompt", "") or "",
        comfy_url=getattr(args, "comfy_url", "") or "",
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
    p_understand.add_argument("--crops-dir", default=None, help="Directory to save character crops")
    p_understand.add_argument("--annotated", default=None, help="Output path for annotated understanding overlay")

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
    p_render.add_argument("--model-id", default="", help="Video model ID for I2V")
    p_render.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    p_render.add_argument("--prompt", default="", help="Positive action prompt")
    p_render.add_argument("--comfy-url", default="", help="ComfyUI HTTP URL")

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
