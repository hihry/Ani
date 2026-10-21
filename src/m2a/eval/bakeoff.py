"""Phase 3 Resumable Video Bake-off Harness (`src/m2a/eval/bakeoff.py`).

Executes a matrix of `(model x panel x seed)` runs, persists each completed run to
`records.jsonl` for crash-safe resume, generates per-panel keyframe contact sheets,
blind review manifests, `bakeoff_summary.csv`, and `bakeoff_report.md`.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import yaml

from workers.assembler import decode_video_frames, encode_mjpeg_avi
from workers.qc_metrics import compute_clip_qc_metrics
from workers.render_kenburns import generate_kenburns_frames
from workers.render_parallax import generate_parallax_frames
from workers.video.cache import ClipCache
from workers.video.comfy_client import ComfyClient
from workers.video.provider import ComfyVideoProvider, load_model_specs

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BAKEOFF_CONFIG = REPO_ROOT / "configs" / "bakeoff" / "phase3.yaml"


@dataclass
class BakeoffRecord:
    panel_id: str
    category: str
    model_id: str
    tier: str
    route: str
    seed: int
    clip_path: str
    anon_clip_id: str
    duration_s: float
    frames_count: int
    native_frames: int
    gen_seconds: float
    s_per_frame: float
    peak_vram_gb: float
    cached: bool
    qc_scores: Dict[str, Any]
    overall_pass: bool
    error: str = ""


def _synthesize_reference_plate(panel: Dict[str, Any], out_path: Path, size: Tuple[int, int] = (540, 960)) -> Path:
    """Creates a clean, high-detail synthetic anime reference plate for a bake-off panel if not on disk."""
    out_path = Path(out_path)
    if out_path.is_file():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    pid_hash = int(hashlib.sha256(panel["panel_id"].encode("utf-8")).hexdigest()[:8], 16)
    r_base = 35 + (pid_hash % 70)
    g_base = 55 + ((pid_hash >> 8) % 80)
    b_base = 95 + ((pid_hash >> 16) % 90)

    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(r_base + yy * 70 + np.sin(xx * 9.0) * 18, 10, 240).astype(np.uint8)
    arr[:, :, 1] = np.clip(g_base + (1.0 - yy) * 65 + np.cos(yy * 7.0) * 18, 10, 240).astype(np.uint8)
    arr[:, :, 2] = np.clip(b_base + xx * 55, 20, 245).astype(np.uint8)

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    # Draw character silhouette / focal geometry with clean dark line art (no speech bubbles)
    cx, cy = w // 2, int(h * 0.48)
    draw.ellipse([cx - 95, cy - 130, cx + 95, cy + 70], fill=(225, 195, 175), outline=(22, 24, 32), width=4)
    draw.polygon(
        [(cx - 140, h - 40), (cx - 75, cy + 55), (cx + 75, cy + 55), (cx + 140, h - 40)],
        fill=(42, 58, 92),
        outline=(18, 20, 28),
    )
    img.save(out_path, format="PNG", compress_level=1)
    return out_path


def _generate_contact_sheet(
    panel_id: str,
    model_clips: Dict[str, Path],
    out_path: Path,
    thumb_size: Tuple[int, int] = (135, 240),
) -> Optional[Path]:
    """Generates a side-by-side contact sheet showing [0%, 25%, 50%, 75%, 100%] frames per model."""
    if not model_clips:
        return None
    tw, th = thumb_size
    n_cols = 5
    label_w = 160
    models = sorted(model_clips.keys())
    sheet_w = label_w + n_cols * tw
    sheet_h = max(1, len(models)) * th

    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)

    for row_idx, mid in enumerate(models):
        y0 = row_idx * th
        draw.text((10, y0 + th // 2 - 8), mid[:20], fill=(235, 240, 250))
        frames = decode_video_frames(model_clips[mid], target_res=thumb_size)
        if not frames:
            continue
        indices = [
            0,
            len(frames) // 4,
            len(frames) // 2,
            (3 * len(frames)) // 4,
            len(frames) - 1,
        ]
        for col_idx, f_idx in enumerate(indices):
            x0 = label_w + col_idx * tw
            sheet.paste(frames[min(len(frames) - 1, f_idx)], (x0, y0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG", compress_level=1)
    return out_path


def run_bakeoff(
    config_path: Path = DEFAULT_BAKEOFF_CONFIG,
    out_dir: Path = Path("runs/bakeoff_phase3"),
    tier: Optional[str] = None,
    comfy_url: Optional[str] = None,
    clean_plates_dir: Optional[Path] = None,
    client: Optional[ComfyClient] = None,
) -> Dict[str, Any]:
    """Runs the Phase 3 bake-off matrix with JSONL crash-safe resume and generates reports."""
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = int(cfg.get("fps", 24))
    res_list = cfg.get("resolution") or [540, 960]
    target_res = (int(res_list[0]), int(res_list[1]))
    seeds = [int(s) for s in (cfg.get("seeds") or [42])]
    panels = list(cfg.get("panels") or [])
    models = list(cfg.get("models") or [])

    if tier:
        models = [m for m in models if m.get("tier") in (tier, "cpu")]

    records_path = out_dir / "records.jsonl"
    existing_keys: set[Tuple[str, str, int]] = set()
    records: List[BakeoffRecord] = []

    if records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rec = BakeoffRecord(**data)
            records.append(rec)
            existing_keys.add((rec.panel_id, rec.model_id, int(rec.seed)))

    plates_dir = Path(clean_plates_dir) if clean_plates_dir else (out_dir / "clean_plates")
    plates_dir.mkdir(parents=True, exist_ok=True)
    plate_paths: Dict[str, Path] = {}
    for p in panels:
        pid = str(p["panel_id"])
        plate_paths[pid] = _synthesize_reference_plate(p, plates_dir / f"{pid}_clean.png", size=target_res)

    eff_comfy_url = (comfy_url or os.environ.get("M2A_COMFY_URL", "http://127.0.0.1:8188")).strip()
    comfy_client = client or ComfyClient(base_url=eff_comfy_url)
    specs = load_model_specs()

    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Outer loop by model_id so each heavy GPU model is loaded once and freed after its batch
    for m_entry in models:
        mid = str(m_entry["id"])
        m_tier = str(m_entry.get("tier", "t1"))
        m_route = str(m_entry.get("route", "i2v"))

        pending = [
            (p, s)
            for p in panels
            for s in seeds
            if (str(p["panel_id"]), mid, int(s)) not in existing_keys
        ]
        if not pending:
            continue

        provider: Optional[ComfyVideoProvider] = None
        if m_route == "i2v" and mid in specs:
            provider = ComfyVideoProvider(
                spec=specs[mid],
                client=comfy_client,
                cache=ClipCache(out_dir / ".clip_cache"),
            )

        for p, seed in pending:
            pid = str(p["panel_id"])
            category = str(p.get("category", p.get("shot", "general")))
            duration_s = float(p.get("duration_s", 2.5))
            plate = plate_paths[pid]
            clip_out = clips_dir / f"{pid}_{mid}_s{seed}.avi"
            anon_id = "clip_" + hashlib.sha256(f"{pid}:{mid}:{seed}".encode("utf-8")).hexdigest()[:10]

            t0 = time.monotonic()
            err_msg = ""
            native_frames = max(1, int(round(duration_s * fps)))
            peak_vram = 0.0
            cached = False

            try:
                if m_route == "i2v" and provider is not None:
                    gen_res = provider.generate(
                        panel=p,
                        clean_plate=plate,
                        duration_s=duration_s,
                        seed=seed,
                        out_dir=clips_dir,
                        target_res=target_res,
                        fps=fps,
                    )
                    clip_out = gen_res.clip_path
                    native_frames = gen_res.native_frames
                    gen_s = gen_res.gen_seconds
                    peak_vram = gen_res.peak_vram_gb
                    cached = gen_res.cached
                elif m_route == "parallax":
                    frames = generate_parallax_frames(
                        image_path=plate,
                        target_res=target_res,
                        duration_s=duration_s,
                        fps=fps,
                    )
                    encode_mjpeg_avi(frames, clip_out, fps=fps)
                    gen_s = round(time.monotonic() - t0, 3)
                else:
                    frames = generate_kenburns_frames(
                        image_path=plate,
                        target_res=target_res,
                        duration_s=duration_s,
                        fps=fps,
                        motion_type=str(p.get("motion_type", "zoom_in")),
                    )
                    encode_mjpeg_avi(frames, clip_out, fps=fps)
                    gen_s = round(time.monotonic() - t0, 3)

                decoded = decode_video_frames(clip_out, target_res=target_res)
                qc_scores = compute_clip_qc_metrics(plate, decoded, route=m_route)
                overall_pass = bool(qc_scores.get("overall_pass", False))
                frames_cnt = len(decoded)
            except Exception as exc:
                gen_s = round(time.monotonic() - t0, 3)
                err_msg = str(exc)
                qc_scores = {
                    "identity_similarity": 0.0,
                    "new_text_count": 0,
                    "flicker_score": 1.0,
                    "motion_score": 0.0,
                    "warp_error": 1.0,
                    "first_frame_fidelity": 0.0,
                    "style_drift": 1.0,
                    "metrics_version": 2,
                    "overall_pass": False,
                    "reasons": [f"error: {err_msg}"],
                }
                overall_pass = False
                frames_cnt = 0

            spf = round(gen_s / float(max(1, native_frames)), 4)
            rec = BakeoffRecord(
                panel_id=pid,
                category=category,
                model_id=mid,
                tier=m_tier,
                route=m_route,
                seed=int(seed),
                clip_path=str(clip_out.as_posix()),
                anon_clip_id=anon_id,
                duration_s=duration_s,
                frames_count=frames_cnt,
                native_frames=native_frames,
                gen_seconds=gen_s,
                s_per_frame=spf,
                peak_vram_gb=peak_vram,
                cached=cached,
                qc_scores=qc_scores,
                overall_pass=overall_pass,
                error=err_msg,
            )
            records.append(rec)
            existing_keys.add((pid, mid, int(seed)))
            with open(records_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec)) + "\n")

        if m_route == "i2v":
            try:
                comfy_client.free_memory()
            except Exception:
                pass

    # Generate contact sheets per panel (using first seed)
    sheets_dir = out_dir / "contact_sheets"
    first_seed = seeds[0]
    for p in panels:
        pid = str(p["panel_id"])
        m_clips = {
            r.model_id: Path(r.clip_path)
            for r in records
            if r.panel_id == pid and r.seed == first_seed and r.frames_count > 0 and Path(r.clip_path).is_file()
        }
        if m_clips:
            _generate_contact_sheet(pid, m_clips, sheets_dir / f"{pid}_sheet.png")

    # Export blind review manifest (anonymized) and key
    blind_items = [
        {
            "anon_clip_id": r.anon_clip_id,
            "panel_id": r.panel_id,
            "category": r.category,
            "clip_path": r.clip_path,
            "duration_s": r.duration_s,
        }
        for r in records
    ]
    blind_key = {
        r.anon_clip_id: {
            "model_id": r.model_id,
            "tier": r.tier,
            "route": r.route,
            "seed": r.seed,
            "panel_id": r.panel_id,
        }
        for r in records
    }
    (out_dir / "blind_manifest.json").write_text(json.dumps(blind_items, indent=2), encoding="utf-8")
    (out_dir / "blind_key.json").write_text(json.dumps(blind_key, indent=2), encoding="utf-8")

    summary_rows, report_path, csv_path = _write_bakeoff_reports(records, out_dir)
    return {
        "records_count": len(records),
        "summary": summary_rows,
        "report_path": str(report_path.as_posix()),
        "csv_path": str(csv_path.as_posix()),
        "records_path": str(records_path.as_posix()),
    }


def _write_bakeoff_reports(
    records: List[BakeoffRecord],
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], Path, Path]:
    by_model: Dict[str, List[BakeoffRecord]] = defaultdict(list)
    for r in records:
        by_model[r.model_id].append(r)

    summary_rows: List[Dict[str, Any]] = []
    for mid, recs in sorted(by_model.items()):
        n = max(1, len(recs))
        pass_rate = sum(1 for r in recs if r.overall_pass) / float(n)
        gen_vals = [r.gen_seconds for r in recs]
        spf_vals = [r.s_per_frame for r in recs]
        vram_vals = [r.peak_vram_gb for r in recs]
        id_vals = [float(r.qc_scores.get("identity_similarity", 0.0)) for r in recs]
        flk_vals = [float(r.qc_scores.get("flicker_score", 0.0)) for r in recs]
        wrp_vals = [float(r.qc_scores.get("warp_error", 0.0)) for r in recs]
        mot_vals = [float(r.qc_scores.get("motion_score", 0.0)) for r in recs]

        summary_rows.append(
            {
                "model_id": mid,
                "tier": recs[0].tier,
                "route": recs[0].route,
                "clips": len(recs),
                "pass_rate": round(pass_rate, 4),
                "median_gen_s": round(float(np.median(gen_vals)), 3),
                "p90_gen_s": round(float(np.percentile(gen_vals, 90)), 3),
                "s_per_frame": round(float(np.median(spf_vals)), 4),
                "peak_vram_gb": round(float(np.max(vram_vals)), 2),
                "mean_identity": round(float(np.mean(id_vals)), 4),
                "mean_flicker": round(float(np.mean(flk_vals)), 4),
                "mean_warp_error": round(float(np.mean(wrp_vals)), 4),
                "mean_motion": round(float(np.mean(mot_vals)), 4),
            }
        )

    csv_path = out_dir / "bakeoff_summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    md_lines = [
        "# Phase 3 Video Generation Bake-off Report",
        "",
        "## 1. Speed, VRAM & Auto-QC Summary by Model",
        "",
        "| Model ID | Tier | Route | Clips | QC Pass Rate | Median s/clip | s/frame | Peak VRAM (GB) | Mean Identity | Mean Warp Err |",
        "|----------|------|-------|-------|--------------|---------------|---------|----------------|---------------|---------------|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| `{row['model_id']}` | `{row['tier']}` | `{row['route']}` | {row['clips']} | "
            f"{row['pass_rate'] * 100:.1f}% | {row['median_gen_s']:.2f}s | {row['s_per_frame']:.3f} | "
            f"{row['peak_vram_gb']:.1f} | {row['mean_identity']:.3f} | {row['mean_warp_error']:.3f} |"
        )

    md_lines.extend(
        [
            "",
            "## 2. Tier Recommendations",
            "",
            "- **Tier 1S (8 GB VRAM)**: `wan22-5b-gguf-q4` primary -> `parallax` -> `kenburns`",
            "- **Tier 1 (12–16 GB VRAM)**: `wan22-5b-fp8` (`wan22-ti2v-5b`) primary -> `ltx23-fp8` -> `parallax` -> `kenburns`",
            "- **Tier 2 (24 GB+ VRAM)**: `wan22-a14b-lightning` / `wan22-a14b-fp8` primary -> `wan22-5b-fp8` -> `ltx23-fp8` -> `parallax` -> `kenburns`",
            "",
        ]
    )
    report_path = out_dir / "bakeoff_report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    return summary_rows, report_path, csv_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 Video Bake-off Harness")
    parser.add_argument("--config", type=Path, default=DEFAULT_BAKEOFF_CONFIG)
    parser.add_argument("--out", type=Path, default=Path("runs/bakeoff_phase3"))
    parser.add_argument("--tier", default=None)
    parser.add_argument("--comfy-url", default=None)
    args = parser.parse_args(argv)
    run_bakeoff(config_path=args.config, out_dir=args.out, tier=args.tier, comfy_url=args.comfy_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
