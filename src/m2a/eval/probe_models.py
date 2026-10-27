"""Real-Model GPU Probe & Anti-Slideshow Verification Gate (`src/m2a/eval/probe_models.py`).

Outside the normal CPU `init.sh` gate because it requires local/cloud GPU weights and a live
ComfyUI server on `:8188`. Invoked via `./harnesses/eval-bakeoff/probe-models.sh`.

Performs four strict checks:
  1. Model file & SHA-256 verification on `/run/media/hihry/OS/m2a_models/` against `workflows/comfyui/manifest.yaml`.
  2. Live ComfyUI `:8188` `/system_stats` reachability + `/object_info` custom node verification for the target workflow.
  3. One short real I2V render (~17 native frames) asserting `route_used == "i2v"`, zero frozen tail frames,
     and optical-flow `motion_score` above the static/slideshow floor.
  4. Slideshow detector (`audit_run_for_slideshow`) on any `runs/<id>/` directory: compares planned vs actual
     `route_used` in `state.json` and `routes/routing_plan.json`, failing loudly if `i2v` panels fell back to
     `parallax`/`kenburns` or produced frozen clips.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "workflows" / "comfyui" / "manifest.yaml"
DEFAULT_MODELS_ROOT = Path("/run/media/hihry/OS/m2a_models")


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _read_hf_cached_sha256(file_path: Path, hf_rel_file: str = "") -> str | None:
    """Reads cached SHA-256 from HuggingFace `.cache/huggingface/download/*.metadata` by exact path."""
    cache_root = file_path.parent / ".cache" / "huggingface" / "download"
    candidates = [cache_root / f"{file_path.name}.metadata"]
    if hf_rel_file:
        candidates.insert(0, cache_root / f"{hf_rel_file}.metadata")
    for meta_file in candidates:
        if not meta_file.is_file():
            continue
        try:
            lines = [ln.strip() for ln in meta_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for ln in lines:
                if len(ln) == 64 and all(c in "0123456789abcdef" for c in ln.lower()):
                    return ln.lower()
        except Exception:
            continue
    return None


def resolve_models_dir(explicit_root: Path | None = None, comfy_dir: Path | None = None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root)
    if DEFAULT_MODELS_ROOT.exists():
        return DEFAULT_MODELS_ROOT
    c_dir = Path(comfy_dir) if comfy_dir else (Path.home() / "ComfyUI")
    extra_paths = c_dir / "extra_model_paths.yaml"
    if extra_paths.is_file():
        try:
            cfg = yaml.safe_load(extra_paths.read_text(encoding="utf-8")) or {}
            for sec in cfg.values():
                if isinstance(sec, dict) and sec.get("base_path"):
                    return Path(str(sec["base_path"]))
        except Exception:
            pass
    return c_dir / "models"


def verify_model_files(
    tier: str | None = "t1s",
    model_id: str | None = None,
    models_root: Path | None = None,
    manifest_path: Path = DEFAULT_MANIFEST,
    verify_sha256: bool = True,
    full_hash_on_fuse: bool = False,
) -> Dict[str, Any]:
    """Verifies that all required model files for the tier/model exist on disk with valid SHA-256."""
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    models_map = manifest.get("models", {})
    selected: List[tuple[str, Dict[str, Any]]] = []
    for mid, spec in models_map.items():
        if model_id and mid != model_id:
            continue
        if tier and not model_id and spec.get("tier") != tier:
            continue
        selected.append((mid, spec))

    if not selected:
        raise ValueError(f"No models matched tier={tier!r}, model_id={model_id!r} in {manifest_path}")

    root_dir = resolve_models_dir(models_root)
    try:
        resolved_str = str(root_dir.resolve())
    except Exception:
        resolved_str = str(root_dir)
    is_cloud_mount = "google_ai_cloud" in resolved_str

    checked_files: List[Dict[str, Any]] = []
    missing_files: List[str] = []
    sha_mismatches: List[Dict[str, str]] = []

    for mid, spec in selected:
        for f_entry in spec.get("files", []):
            subdir = f_entry.get("target_dir", "diffusion_models")
            fname = Path(f_entry["file"]).name
            dest = root_dir / subdir / fname
            expected_sha = str(f_entry.get("sha256") or "").strip().lower()

            exists = dest.is_file()
            size_bytes = dest.stat().st_size if exists else 0
            actual_sha: str | None = None
            sha_source = "none"
            sha_ok = True

            if not exists or size_bytes <= 0:
                missing_files.append(str(dest))
                sha_ok = False
            elif verify_sha256 and expected_sha:
                if is_cloud_mount and not full_hash_on_fuse and size_bytes > 64 * 1024 * 1024:
                    cached_sha = _read_hf_cached_sha256(dest, str(f_entry.get("file", "")))
                    if cached_sha is not None:
                        actual_sha = cached_sha
                        sha_source = "hf_metadata"
                    else:
                        actual_sha = _compute_sha256(dest)
                        sha_source = "stream_sha256"
                else:
                    actual_sha = _compute_sha256(dest)
                    sha_source = "file_sha256"

                if actual_sha != expected_sha:
                    sha_ok = False
                    sha_mismatches.append(
                        {
                            "file": str(dest),
                            "expected": expected_sha,
                            "actual": actual_sha or "missing",
                        }
                    )

            checked_files.append(
                {
                    "model_id": mid,
                    "path": str(dest),
                    "exists": exists and size_bytes > 0,
                    "size_bytes": size_bytes,
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                    "sha_source": sha_source,
                    "sha_ok": sha_ok,
                }
            )

    passed = len(missing_files) == 0 and len(sha_mismatches) == 0
    return {
        "passed": passed,
        "tier": tier,
        "model_id": model_id,
        "models_root": str(root_dir),
        "resolved_root": resolved_str,
        "is_cloud_mount": is_cloud_mount,
        "files": checked_files,
        "missing": missing_files,
        "sha_mismatches": sha_mismatches,
    }


def extract_workflow_required_nodes(workflow_path: Path) -> List[str]:
    """Extracts all unique ComfyUI node `class_type` names from an API workflow JSON."""
    tpl = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    nodes = {
        str(node["class_type"])
        for node in tpl.values()
        if isinstance(node, dict) and node.get("class_type")
    }
    return sorted(nodes)


def probe_comfy_server(
    comfy_url: str = "http://127.0.0.1:8188",
    model_id: str = "wan22-5b-gguf-q4",
    client: Any | None = None,
) -> Dict[str, Any]:
    """Probes ComfyUI `/system_stats` and `/object_info` for all nodes required by `model_id`."""
    from workers.video.comfy_client import ComfyClient
    from workers.video.provider import load_model_specs

    specs = load_model_specs()
    if model_id not in specs:
        raise ValueError(f"Unknown video model_id: {model_id!r}")
    spec = specs[model_id]
    wf_path = spec.workflow_path if spec.workflow_path.is_file() else (REPO_ROOT / "workflows" / "comfyui" / spec.workflow_path.name)
    req_nodes = sorted(set(extract_workflow_required_nodes(wf_path)) | set(spec.required_nodes))

    c = client or ComfyClient(base_url=comfy_url, timeout_s=10.0)
    try:
        res = c.probe(required_nodes=req_nodes)
        return {
            "passed": bool(res.get("ready", False)),
            "comfy_url": comfy_url,
            "model_id": model_id,
            "workflow": wf_path.name,
            "required_nodes": req_nodes,
            "vram_gb": float(res.get("vram_gb", 0.0)),
            "error": "",
        }
    except Exception as exc:
        return {
            "passed": False,
            "comfy_url": comfy_url,
            "model_id": model_id,
            "workflow": wf_path.name,
            "required_nodes": req_nodes,
            "vram_gb": 0.0,
            "error": str(exc),
        }


def _make_probe_panel(path: Path, size: tuple[int, int] = (512, 384)) -> Path:
    w, h = size
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(40 + yy * 95 + np.sin(xx * 6.0) * 20, 10, 240).astype(np.uint8)
    arr[:, :, 1] = np.clip(60 + (1.0 - yy) * 85, 10, 240).astype(np.uint8)
    arr[:, :, 2] = np.clip(115 + xx * 75, 20, 245).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    draw.ellipse([cx - 50, cy - 70, cx + 50, cy + 30], fill=(225, 195, 175), outline=(25, 25, 35), width=3)
    draw.polygon(
        [(cx - 85, h - 15), (cx - 45, cy + 25), (cx + 45, cy + 25), (cx + 85, h - 15)],
        fill=(45, 65, 105),
        outline=(20, 20, 30),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def compute_frozen_frame_stats(frames: List[Image.Image], freeze_threshold: float = 0.08) -> Dict[str, float]:
    """Computes inter-frame pixel difference statistics and frozen-frame ratio (`diff < freeze_threshold`)."""
    if len(frames) < 2:
        return {"num_pairs": 0, "frozen_pairs": 0, "frozen_ratio": 1.0, "min_diff": 0.0, "mean_diff": 0.0}
    arrs = [np.asarray(f.convert("RGB"), dtype=np.float32) for f in frames]
    diffs = [float(np.mean(np.abs(arrs[i + 1] - arrs[i]))) for i in range(len(arrs) - 1)]
    frozen_pairs = sum(1 for d in diffs if d < freeze_threshold)
    return {
        "num_pairs": len(diffs),
        "frozen_pairs": frozen_pairs,
        "frozen_ratio": round(frozen_pairs / len(diffs), 4),
        "min_diff": round(min(diffs), 4),
        "mean_diff": round(float(np.mean(diffs)), 4),
    }


def run_short_real_render(
    comfy_url: str = "http://127.0.0.1:8188",
    model_id: str = "wan22-5b-gguf-q4",
    sample_image: Path | None = None,
    out_dir: Path | None = None,
    duration_s: float = 1.06,
    fps: int = 24,
    client: Any | None = None,
) -> Dict[str, Any]:
    """Executes a short 1-panel I2V render and fails if it falls back to pan/zoom or produces frozen frames."""
    from workers.assembler import decode_video_frames
    from workers.pipeline_cli import render_single_panel_clip
    from workers.qc_metrics import compute_clip_qc_metrics
    from workers.video.cache import ClipCache
    from workers.video.comfy_client import ComfyClient
    from workers.video.provider import ComfyVideoProvider, load_model_specs

    tmp_ctx = tempfile.TemporaryDirectory(prefix="m2a_probe_render_") if out_dir is None else None
    work_dir = Path(out_dir) if out_dir is not None else Path(tmp_ctx.name)  # type: ignore[arg-type]
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        plate = Path(sample_image) if sample_image and Path(sample_image).is_file() else _make_probe_panel(work_dir / "probe_panel.png")
        out_clip = work_dir / "probe_i2v.avi"

        if client is not None:
            specs = load_model_specs()
            eff_model = model_id if model_id in specs else "wan22-5b-gguf-q4"
            provider = ComfyVideoProvider(
                spec=specs[eff_model],
                client=client,
                cache=ClipCache(cache_dir=work_dir / "_cache"),
            )
            gen_res = provider.generate(
                panel={
                    "panel_id": "probe_001",
                    "action_prompt": "2D anime character turns head with subtle wind in hair",
                    "motion_type": "zoom_in",
                    "emotion": "determined",
                },
                clean_plate=plate,
                duration_s=duration_s,
                seed=42,
                out_dir=work_dir,
                target_res=(512, 384),
                fps=fps,
            )
            frames = decode_video_frames(gen_res.clip_path, target_res=(512, 384))
            qc_scores = compute_clip_qc_metrics(plate, frames, route="i2v")
            rendered = {
                "clip_path": str(gen_res.clip_path),
                "route_used": "i2v",
                "model_id": gen_res.model_id,
                "native_frames": gen_res.native_frames,
                "frames_count": len(frames),
                "qc_scores": qc_scores,
            }
        else:
            rendered = render_single_panel_clip(
                image_path=plate,
                output_path=out_clip,
                route="i2v",
                shot_type="close_up",
                duration_s=duration_s,
                fps=fps,
                target_res=(512, 384),
                panel_id="probe_001",
                model_id=model_id,
                seed=42,
                action_prompt="2D anime character turns head with subtle wind in hair",
                comfy_url=comfy_url,
            )
            frames = decode_video_frames(Path(rendered["clip_path"]), target_res=(512, 384))

        route_used = str(rendered.get("route_used", ""))
        qc = rendered.get("qc_scores", {})
        motion_score = float(qc.get("motion_score", 0.0))
        freeze_stats = compute_frozen_frame_stats(frames)

        reasons: List[str] = []
        if route_used != "i2v":
            reasons.append(f"panel meant for i2v fell back to {route_used!r}")
        if len(frames) == 0:
            reasons.append("decoded 0 frames from rendered clip")
        if motion_score < 0.02:
            reasons.append(f"optical-flow motion_score {motion_score:.4f} < minimum 0.0200 (static slideshow)")
        if freeze_stats["frozen_ratio"] > 0.05:
            reasons.append(
                f"frozen_frame_ratio {freeze_stats['frozen_ratio']:.1%} > max 5.0% "
                f"({freeze_stats['frozen_pairs']}/{freeze_stats['num_pairs']} frozen pairs)"
            )
        if not qc.get("overall_pass", False):
            reasons.append(f"QC v2 failed: {qc.get('reasons', [])}")

        return {
            "passed": len(reasons) == 0,
            "route_used": route_used,
            "model_id": rendered.get("model_id", model_id),
            "frames_count": len(frames),
            "native_frames": rendered.get("native_frames", 0),
            "motion_score": motion_score,
            "freeze_stats": freeze_stats,
            "qc_scores": qc,
            "reasons": reasons,
        }
    except Exception as exc:
        return {
            "passed": False,
            "route_used": "error",
            "model_id": model_id,
            "frames_count": 0,
            "native_frames": 0,
            "motion_score": 0.0,
            "freeze_stats": {},
            "qc_scores": {},
            "reasons": [f"render exception: {exc}"],
        }
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def audit_run_for_slideshow(
    run_dir: Path,
    max_degraded_pct: float = 0.0,
    max_frozen_ratio: float = 0.10,
) -> Dict[str, Any]:
    """Audits a pipeline run directory (`runs/<id>`) to detect silent I2V -> pan/zoom degradation or frozen clips.

    Fails (`passed=False`) when:
      - `state.json` is missing or has 0 panels
      - Compute tier != 't0' and 0 panels were planned or accepted on the `i2v` route
      - The percentage of planned `i2v` panels that degraded to `parallax`/`kenburns` (or errored) exceeds `max_degraded_pct`
      - Any accepted `i2v` clip on disk has `frozen_ratio > max_frozen_ratio`
    """
    from workers.assembler import decode_video_frames

    run_dir = Path(run_dir)
    state_file = run_dir / "state.json"
    if not state_file.is_file():
        return {
            "passed": False,
            "run_dir": str(run_dir),
            "reasons": [f"state.json not found in {run_dir}"],
        }

    state = json.loads(state_file.read_text(encoding="utf-8"))
    tier = str(state.get("config", {}).get("compute_tier", "t1")).lower()
    panels = state.get("panels", [])
    clip_candidates = state.get("clip_candidates", {})

    plan_file = run_dir / "routes" / "routing_plan.json"
    plan_by_id: Dict[str, Dict[str, Any]] = {}
    if plan_file.is_file():
        try:
            raw_plan = json.loads(plan_file.read_text(encoding="utf-8"))
            plan_list = raw_plan.get("panels", []) if isinstance(raw_plan, dict) else raw_plan
            if isinstance(plan_list, list):
                for entry in plan_list:
                    if isinstance(entry, dict) and entry.get("panel_id"):
                        plan_by_id[str(entry["panel_id"])] = entry
        except Exception:
            pass

    planned_i2v: List[str] = []
    accepted_i2v: List[str] = []
    degraded_panels: List[Dict[str, Any]] = []
    frozen_clips: List[Dict[str, Any]] = []

    for p in panels:
        pid = str(p.get("panel_id", ""))
        final_route = str(p.get("route", "kenburns")).lower()
        cands = clip_candidates.get(pid, []) if isinstance(clip_candidates, dict) else []
        plan_entry = plan_by_id.get(pid, {})
        plan_route = str(plan_entry.get("route", "")).lower()
        plan_fb = str(plan_entry.get("fallback_reason", "")).lower()

        had_i2v_attempt = any(
            isinstance(c, dict)
            and (
                str(c.get("route", "")).lower() == "i2v"
                or str(c.get("fallback_reason", "")).lower().startswith("i2v")
            )
            for c in cands
        )
        was_meant_for_i2v = (
            plan_route == "i2v"
            or plan_fb == "comfy_unavailable"
            or final_route == "i2v"
            or had_i2v_attempt
        )

        if not was_meant_for_i2v:
            continue

        planned_i2v.append(pid)

        # Check whether an accepted i2v clip actually exists
        accepted_cand = None
        for c in cands:
            if (
                isinstance(c, dict)
                and c.get("accepted")
                and str(c.get("route", "")).lower() == "i2v"
                and not c.get("render_error")
            ):
                accepted_cand = c
                break

        clip_file: Path | None = None
        if accepted_cand and accepted_cand.get("path"):
            cp = Path(str(accepted_cand["path"]))
            if cp.is_file():
                clip_file = cp
            elif (run_dir / "clips" / cp.name).is_file():
                clip_file = run_dir / "clips" / cp.name
        elif not cands and final_route == "i2v":
            # Support runs where clips were written directly to clips/<pid>*.avi
            matches = sorted((run_dir / "clips").glob(f"{pid}*.avi")) if (run_dir / "clips").is_dir() else []
            if matches:
                clip_file = matches[0]

        if final_route != "i2v" or clip_file is None:
            err_notes = [
                str(c.get("render_error") or c.get("fallback_reason"))
                for c in cands
                if isinstance(c, dict) and (c.get("render_error") or c.get("fallback_reason"))
            ]
            if plan_fb:
                err_notes.append(plan_fb)
            degraded_panels.append(
                {
                    "panel_id": pid,
                    "final_route": final_route,
                    "has_clip_file": clip_file is not None,
                    "details": err_notes or ["missing_accepted_i2v_clip"],
                }
            )
            continue

        accepted_i2v.append(pid)
        frames = decode_video_frames(clip_file)
        fstats = compute_frozen_frame_stats(frames)
        if fstats["frozen_ratio"] > max_frozen_ratio:
            frozen_clips.append(
                {
                    "panel_id": pid,
                    "clip_path": str(clip_file),
                    "frozen_ratio": fstats["frozen_ratio"],
                    "frozen_pairs": fstats["frozen_pairs"],
                    "num_pairs": fstats["num_pairs"],
                }
            )

    planned_count = len(planned_i2v)
    degraded_count = len(degraded_panels)
    degraded_pct = (degraded_count / planned_count * 100.0) if planned_count > 0 else (100.0 if tier != "t0" else 0.0)

    reasons: List[str] = []
    if not panels:
        reasons.append("0 panels found in state.json")
    if tier != "t0" and planned_count == 0:
        reasons.append(f"tier={tier!r} expected GPU I2V routing, but 0 panels were planned for i2v")
    if planned_count > 0 and degraded_pct > max_degraded_pct:
        reasons.append(
            f"{degraded_count}/{planned_count} ({degraded_pct:.1f}%) i2v panels degraded to pan/zoom or failed "
            f"(max allowed: {max_degraded_pct:.1f}%): {degraded_panels}"
        )
    if frozen_clips:
        reasons.append(f"{len(frozen_clips)} accepted i2v clip(s) exceed frozen frame threshold: {frozen_clips}")

    return {
        "passed": len(reasons) == 0,
        "run_dir": str(run_dir),
        "tier": tier,
        "total_panels": len(panels),
        "planned_i2v_count": planned_count,
        "accepted_i2v_count": len(accepted_i2v),
        "degraded_count": degraded_count,
        "degraded_pct": round(degraded_pct, 2),
        "degraded_panels": degraded_panels,
        "frozen_clips": frozen_clips,
        "reasons": reasons,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="M2A Real-Model GPU Probe & Anti-Slideshow Detector (outside CPU init.sh gate)"
    )
    parser.add_argument("--tier", choices=["t1s", "t1", "t2"], default="t1s")
    parser.add_argument("--model", default=None, help="Explicit model_id from manifest.yaml / video_models.yaml")
    parser.add_argument("--models-root", type=Path, default=None, help="Override /run/media/hihry/OS/m2a_models")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    parser.add_argument("--check-run", type=Path, default=None, help="Audit a finished run directory for slideshow degradation")
    parser.add_argument("--max-degraded-pct", type=float, default=0.0, help="Max allowed %% of i2v panels degraded to pan/zoom")
    parser.add_argument("--files-only", action="store_true", help="Only verify model files on disk")
    parser.add_argument("--full-sha256", action="store_true", help="Stream full SHA-256 even on cloud FUSE mounts")
    args = parser.parse_args(argv)

    overall_ok = True

    if args.check_run is not None:
        print(f"=== Slideshow Audit for Run: {args.check_run} ===")
        run_report = audit_run_for_slideshow(args.check_run, max_degraded_pct=args.max_degraded_pct)
        print(json.dumps(run_report, indent=2))
        if not run_report["passed"]:
            print(f"FAIL: Slideshow / I2V degradation detected in {args.check_run}", file=sys.stderr)
            return 1
        print(f"PASS: Run {args.check_run} has {run_report['accepted_i2v_count']}/{run_report['planned_i2v_count']} valid I2V clips.")
        if args.files_only:
            return 0

    target_model = args.model or ("wan22-5b-gguf-q4" if args.tier == "t1s" else ("wan22-5b-fp8" if args.tier == "t1" else "wan22-a14b-fp8"))

    print(f"=== 1. Model Weight & SHA-256 Check (tier={args.tier}, model={target_model}) ===")
    file_report = verify_model_files(
        tier=None if args.model else args.tier,
        model_id=args.model,
        models_root=args.models_root,
        full_hash_on_fuse=args.full_sha256,
    )
    if file_report["is_cloud_mount"]:
        print(
            f"WARNING: models_root ({file_report['models_root']} -> {file_report['resolved_root']}) "
            "is on an rclone Google Drive FUSE mount (RC6 I/O stall risk)."
        )
    for f in file_report["files"]:
        status = "OK" if (f["exists"] and f["sha_ok"]) else "FAIL"
        size_gb = f["size_bytes"] / (1024 ** 3)
        print(f"  [{status}] [{f['model_id']}] {f['path']} ({size_gb:.2f} GB, sha={f['sha_source']})")
    if not file_report["passed"]:
        print(f"FAIL: Missing or corrupted model weights: {file_report['missing'] or file_report['sha_mismatches']}", file=sys.stderr)
        overall_ok = False
        if args.files_only:
            return 1

    if args.files_only:
        return 0 if overall_ok else 1

    print(f"=== 2. ComfyUI Server & Custom Node Probe ({args.comfy_url}) ===")
    probe_report = probe_comfy_server(comfy_url=args.comfy_url, model_id=target_model)
    if not probe_report["passed"]:
        print(
            f"FAIL: ComfyUI probe failed for {target_model} ({probe_report['workflow']}): {probe_report['error']}",
            file=sys.stderr,
        )
        return 1
    print(f"  [OK] ComfyUI ready ({probe_report['vram_gb']:.1f} GB VRAM, {len(probe_report['required_nodes'])} workflow node classes verified)")

    print(f"=== 3. Short Real I2V Render & Optical-Flow Motion Check ({target_model}) ===")
    render_report = run_short_real_render(comfy_url=args.comfy_url, model_id=target_model)
    print(json.dumps(render_report, indent=2))
    if not render_report["passed"]:
        print(f"FAIL: Real I2V render check failed: {render_report['reasons']}", file=sys.stderr)
        return 1

    print("=== ALL REAL-MODEL GPU CHECKS PASSED ===")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
