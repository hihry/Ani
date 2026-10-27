"""Comprehensive Phase 3 test suite (`tests/test_phase3.py`).

Covers:
  - M0: models.yaml licence/repo audit, requirements split, ASSEMBLE I2V frame mismatch hard-fail (E7)
  - M1 & M2: All 6 tagged ComfyUI API workflow JSONs, parameter injection, ComfyClient against FakeComfyUI
    (normal, cached, oom, node_error, timeout), conform (4k+1 / 8k+1 frame snapping, 1:4 vertical strip
    canvas preparation, 1.25x slow-mo + hold-and-drift tail fill, exact frame count), ComfyVideoProvider + ClipCache
  - M3: Go/Python ClipCandidate & QCScores golden contract parity, deterministic seed parity,
    workers/server.py async `/render/jobs`, `/gpu/free`, and `/health` `video_providers`
  - M4: QC Metrics v2 optical flow motion_score, warp_error, first_frame_fidelity, style_drift,
    multi-frame new_text_count, and 100% rejection of all 6 adversarial fixture types (E6)
  - M5, M6, M7: Bake-off harness resumability & contact sheets/reports, rubric Krippendorff's alpha
    & Bradley-Terry ranking, ROC/AUC + Youden's J + 5-fold CV calibration, and tier configs (t1s, t1, t2)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient
import httpx
import numpy as np
from PIL import Image, ImageDraw
import pytest
import yaml

from m2a.config import load_config
from m2a.eval.bakeoff import run_bakeoff
from m2a.eval.calibrate_qc import calibrate_thresholds, compute_roc_auc_youden
from m2a.eval.rubric import (
    RubricRating,
    bradley_terry_ranks,
    compute_quality_score,
    krippendorff_alpha_ordinal,
)
from m2a.state import ClipCandidate, QCScores, compute_deterministic_seed
from scripts.comfyui.download_models import download_manifest_models
from tests.fakes.fake_comfyui import create_fake_comfy_app
from tests.fixtures.qc_adversarial import generate_all_adversarial_clips
from workers.assembler import decode_video_frames, encode_mjpeg_avi
from workers.pipeline_cli import assemble_from_run_state
from workers.qc_metrics import compute_clip_qc_metrics
from workers.render_kenburns import generate_kenburns_frames
from workers.server import app as worker_app
from workers.video.cache import ClipCache
from workers.video.comfy_client import (
    ComfyClient,
    ComfyExecutionError,
    ComfyNodeMissing,
    ComfyOOM,
    ComfyTimeout,
)
from workers.video.conform import conform, frames_for, pick_bucket, prepare_input
from workers.video.prompts import build_prompt
from workers.video.provider import ComfyVideoProvider, load_model_specs
from workers.video.workflows import REQUIRED_TAGS, inject, load_template

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_clean_anime_panel(path: Path, size: tuple[int, int] = (540, 960)) -> Path:
    w, h = size
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, h), np.linspace(0.0, 1.0, w), indexing="ij")
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(45 + yy * 90 + np.sin(xx * 8.0) * 20, 10, 240).astype(np.uint8)
    arr[:, :, 1] = np.clip(65 + (1.0 - yy) * 80, 10, 240).astype(np.uint8)
    arr[:, :, 2] = np.clip(110 + xx * 70, 20, 245).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    draw.ellipse([cx - 80, cy - 110, cx + 80, cy + 60], fill=(220, 190, 170), outline=(25, 25, 35), width=4)
    draw.polygon(
        [(cx - 120, h - 30), (cx - 65, cy + 50), (cx + 65, cy + 50), (cx + 120, h - 30)],
        fill=(40, 60, 95),
        outline=(20, 20, 30),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def _make_inprocess_comfy_client(tmp_path: Path, mode: str = "normal") -> tuple[ComfyClient, Any]:
    fake_app, fake_state = create_fake_comfy_app(tmp_path / f"comfy_{mode}", mode=mode)
    tc = TestClient(fake_app)
    client = ComfyClient(base_url="http://testserver", timeout_s=2.0, poll_interval_s=0.01, http_client=tc)
    return client, fake_state


@pytest.mark.contract
@pytest.mark.worker
@pytest.mark.eval_qc
@pytest.mark.eval
def test_m0_models_yaml_and_manifest_audit(tmp_path: Path) -> None:
    """Verifies M0-4 and M1-4: models.yaml licences & repo IDs, and download_models.py --dry-run."""
    models_data = yaml.safe_load((REPO_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert models_data["ltx23"]["repo"] == "Lightricks/LTX-Video"
    assert models_data["ltx23"]["license"] == "LTX-2-Community"
    assert models_data["hunyuan15"]["repo"] == "tencent/HunyuanVideo-I2V"
    assert models_data["hunyuan15"]["license"] == "Tencent-Hunyuan-Community"
    assert models_data["hunyuan15"]["enabled"] is False
    assert "EU" in models_data["hunyuan15"]["excluded_regions"]
    assert "wan22-5b-gguf-q4" in models_data
    assert "wan22-a14b-lightning" in models_data

    plan = download_manifest_models(comfy_dir=tmp_path / "ComfyUI", tier="t1", dry_run=True)
    assert len(plan) >= 3
    assert all(item["sha256"] for item in plan)


@pytest.mark.worker
@pytest.mark.contract
def test_m2_all_6_workflow_templates_and_injection() -> None:
    """Verifies M2-1 & M2-2: all 6 API workflow templates contain REQUIRED_TAGS and inject cleanly."""
    wf_dir = REPO_ROOT / "workflows" / "comfyui"
    expected_files = [
        "wan22_ti2v5b_gguf_api.json",
        "wan22_ti2v5b_fp8_api.json",
        "wan22_i2v_a14b_api.json",
        "wan22_i2v_a14b_lightning_api.json",
        "ltx23_i2v_api.json",
        "hunyuan15_i2v_api.json",
    ]
    for fname in expected_files:
        path = wf_dir / fname
        assert path.is_file(), f"Missing workflow file: {path}"
        tpl = load_template(path)
        injected = inject(
            tpl,
            image_name="panel_001_clean.png",
            pos_prompt="heroic dash forward, 2D anime style",
            neg_prompt="text, speech bubble, watermark",
            seed=1337,
            width=544,
            height=960,
            frames=49,
            steps=16,
            cfg=4.5,
            fps=16,
            filename_prefix="test_prefix",
            sampler="euler",
            scheduler="simple",
        )
        found_tags = {n.get("_meta", {}).get("title") for n in injected.values() if isinstance(n, dict)}
        assert REQUIRED_TAGS.issubset(found_tags)
        if fname in ("wan22_ti2v5b_gguf_api.json", "wan22_ti2v5b_fp8_api.json"):
            class_types = {n.get("class_type") for n in injected.values() if isinstance(n, dict)}
            assert "Wan22ImageToVideoLatent" in class_types
            assert "ModelSamplingSD3" in class_types
        for node in injected.values():
            tag = node.get("_meta", {}).get("title")
            inp = node.get("inputs", {})
            if tag == "M2A_IMAGE":
                assert inp["image"] == "panel_001_clean.png"
            elif tag == "M2A_POS":
                assert inp["text"] == "heroic dash forward, 2D anime style"
            elif tag == "M2A_SEED":
                assert (inp.get("noise_seed") or inp.get("seed")) == 1337
            elif tag == "M2A_LATENT":
                assert inp["width"] == 544 and inp["height"] == 960
                assert (inp.get("length") or inp.get("num_frames")) == 49
            elif tag == "M2A_SAMPLER":
                if "sampler_name" in inp:
                    assert inp["sampler_name"] == "euler"
                if "scheduler" in inp:
                    assert inp["scheduler"] == "simple"


@pytest.mark.worker
def test_m2_comfy_client_fault_modes(tmp_path: Path) -> None:
    """Verifies M1-5 & M2-3: ComfyClient against FakeComfyUI in normal, cached, oom, node_error, and timeout modes."""
    wf = load_template(REPO_ROOT / "workflows" / "comfyui" / "wan22_ti2v5b_fp8_api.json")
    plate = _make_clean_anime_panel(tmp_path / "plate.png", size=(480, 832))

    # 1. Normal mode
    client, state = _make_inprocess_comfy_client(tmp_path, mode="normal")
    probe = client.probe(required_nodes=["UNETLoader", "Wan22ImageToVideoLatent", "ModelSamplingSD3"])
    assert probe["ready"] is True and probe["vram_gb"] == 24.0
    uploaded = client.upload_image(plate)
    assert uploaded == "plate.png"
    wf_inj = inject(
        wf,
        image_name=uploaded,
        pos_prompt="subtle motion",
        neg_prompt="text",
        seed=42,
        width=480,
        height=832,
        frames=17,
        steps=10,
        cfg=5.0,
        fps=16,
    )
    clip_path, meta = client.submit_and_wait(wf_inj, output_dir=tmp_path / "out_normal")
    assert clip_path.is_file()
    assert len(decode_video_frames(clip_path)) == 17
    client.free_memory()
    assert len(state.free_calls) == 1

    # 2. Missing node check
    with pytest.raises(ComfyNodeMissing):
        client.probe(required_nodes=["NonExistentCustomNode"])

    # 3. Cached execution mode
    c_cached, _ = _make_inprocess_comfy_client(tmp_path, mode="cached")
    c_cached.upload_image(plate)
    _, meta_cached = c_cached.submit_and_wait(wf_inj, output_dir=tmp_path / "out_cached")
    assert meta_cached["cached"] is True

    # 4. OOM mode
    c_oom, _ = _make_inprocess_comfy_client(tmp_path, mode="oom")
    with pytest.raises(ComfyOOM):
        c_oom.submit_and_wait(wf_inj, output_dir=tmp_path / "out_oom")

    # 5. Node error mode
    c_err, _ = _make_inprocess_comfy_client(tmp_path, mode="node_error")
    with pytest.raises(ComfyExecutionError):
        c_err.submit_and_wait(wf_inj, output_dir=tmp_path / "out_err")

    # 6. Timeout mode (verifies /interrupt is called)
    c_to, st_to = _make_inprocess_comfy_client(tmp_path, mode="timeout")
    c_to.timeout_s = 0.08
    with pytest.raises(ComfyTimeout):
        c_to.submit_and_wait(wf_inj, output_dir=tmp_path / "out_to")
    assert st_to.interrupted >= 1


@pytest.mark.worker
def test_m2_conform_frame_snapping_and_long_duration_tail(tmp_path: Path) -> None:
    """Verifies M2-6 & E2/E7: frame snapping (4k+1 / 8k+1), 1:4 vertical strip canvas preparation,
    exact frame count `round(duration_s * 24)` even when `duration_s > max_frames / native_fps`,
    and zero frozen frames / zero jump discontinuity across the slow-mo-to-tail transition (RC3).
    """
    specs = load_model_specs()
    wan_spec = specs["wan22-5b-fp8"]
    ltx_spec = specs["ltx23-fp8"]

    for d in [1.0, 2.3, 3.7, 5.0, 7.5]:
        f_wan = frames_for(d, wan_spec)
        assert (f_wan - 1) % 4 == 0 and f_wan <= wan_spec.max_frames
        f_ltx = frames_for(d, ltx_spec)
        assert (f_ltx - 1) % 8 == 0 and f_ltx <= ltx_spec.max_frames

    # Extreme 1:4 vertical scroll strip -> prepared into 9:16 bucket without distortion
    tall_plate = _make_clean_anime_panel(tmp_path / "tall_strip.png", size=(400, 1600))
    bw, bh = pick_bucket(400, 1600, wan_spec.resolution_buckets)
    prep = prepare_input(tall_plate, tmp_path / "prep", target_w=bw, target_h=bh)
    assert Image.open(prep).size == (bw, bh)

    # Generate a short 25-frame @ 16fps raw clip (~1.56s) and conform to 6.5s @ 24fps (156 frames)
    raw_frames = generate_kenburns_frames(tall_plate, target_res=(480, 832), duration_s=1.56, fps=16)
    raw_clip = encode_mjpeg_avi(raw_frames, tmp_path / "raw_16fps.avi", fps=16)
    conformed_path = conform(
        raw_clip=raw_clip,
        clean_plate=tall_plate,
        duration_s=6.5,
        out_path=tmp_path / "conformed_6p5s.avi",
        fps=24,
        width=540,
        height=960,
        native_fps=16,
    )
    decoded = decode_video_frames(conformed_path)
    assert len(decoded) == round(6.5 * 24)
    assert decoded[0].size == (540, 960)
    arrs = [np.asarray(f, dtype=np.float32) for f in decoded]
    diffs = [float(np.mean(np.abs(arrs[i + 1] - arrs[i]))) for i in range(len(arrs) - 1)]
    assert min(diffs) > 0.05, f"Expected zero bit-identical frames in conformed clip, got min_diff={min(diffs)}"
    assert min(diffs[47:]) > 0.25, f"Expected continuous motion across conformed tail, got tail min_diff={min(diffs[47:])}"
    assert max(diffs[46:]) < 2.0, f"Expected smooth slow-mo-to-tail transition without jump pop, got tail max_diff={max(diffs[46:])}"


@pytest.mark.worker
def test_m2_video_provider_and_cache_hit(tmp_path: Path) -> None:
    """Verifies M2-4 & M2-7: ComfyVideoProvider end-to-end generation + content-addressed ClipCache hit."""
    client, state = _make_inprocess_comfy_client(tmp_path, mode="normal")
    specs = load_model_specs()
    cache = ClipCache(cache_dir=tmp_path / "_cache")
    provider = ComfyVideoProvider(spec=specs["wan22-5b-fp8"], client=client, cache=cache)

    assert provider.health()["ready"] is True
    plate = _make_clean_anime_panel(tmp_path / "panel_001_clean.png", size=(540, 960))
    panel = {"panel_id": "panel_001", "action_prompt": "protagonist turns head", "motion_type": "zoom_in"}

    res1 = provider.generate(
        panel=panel,
        clean_plate=plate,
        duration_s=2.0,
        seed=42,
        out_dir=tmp_path / "run1",
        target_res=(540, 960),
        fps=24,
    )
    assert res1.cached is False
    assert len(decode_video_frames(res1.clip_path)) == 48
    assert len(state.prompts_submitted) == 1

    # Second call with identical inputs hits ClipCache without submitting a new prompt to ComfyUI
    res2 = provider.generate(
        panel=panel,
        clean_plate=plate,
        duration_s=2.0,
        seed=42,
        out_dir=tmp_path / "run2",
        target_res=(540, 960),
        fps=24,
    )
    assert res2.cached is True
    assert len(state.prompts_submitted) == 1
    assert len(decode_video_frames(res2.clip_path)) == 48


@pytest.mark.eval_qc
@pytest.mark.eval
@pytest.mark.worker
def test_m4_qc_metrics_v2_rejects_all_6_adversarial_clips(tmp_path: Path) -> None:
    """Verifies M4-1, M4-2 & E6: all 6 adversarial clip classes are 100% rejected by QC v2 while clean clips pass."""
    plate = _make_clean_anime_panel(tmp_path / "clean_plate.png", size=(540, 960))
    clean_frames = generate_kenburns_frames(plate, target_res=(540, 960), duration_s=1.5, fps=24, motion_type="zoom_in")

    clean_qc = compute_clip_qc_metrics(plate, clean_frames, route="i2v")
    assert clean_qc["metrics_version"] == 2
    assert clean_qc["overall_pass"] is True, f"Expected clean clip to pass QC v2, got: {clean_qc}"

    adv_clips = generate_all_adversarial_clips(clean_frames, num_frames=len(clean_frames), size=(540, 960))
    assert set(adv_clips.keys()) == {"frozen", "shuffled", "flicker", "text_overlay", "hue_shift", "face_swap"}

    for defect_name, frames in adv_clips.items():
        qc = compute_clip_qc_metrics(plate, frames, route="i2v")
        assert qc["overall_pass"] is False, f"Adversarial clip {defect_name!r} was not rejected! Scores: {qc}"
        assert len(qc["reasons"]) >= 1


@pytest.mark.worker
@pytest.mark.contract
def test_m0_assemble_i2v_frame_mismatch_hard_fails(tmp_path: Path) -> None:
    """Verifies M0-2 & E7: assemble_from_run_state raises ValueError when an accepted I2V clip has wrong frame count."""
    run_dir = tmp_path / "run_e7"
    (run_dir / "panels").mkdir(parents=True)
    (run_dir / "clips").mkdir(parents=True)
    plate = _make_clean_anime_panel(run_dir / "panels" / "panel_001.png", size=(540, 960))

    # Render a 1.0s clip (24 frames), but state.json says panel_001 has duration_s=4.0s (96 frames) and route="i2v"
    short_frames = generate_kenburns_frames(plate, target_res=(540, 960), duration_s=1.0, fps=24)
    clip_path = encode_mjpeg_avi(short_frames, run_dir / "clips" / "panel_001_clip_01.avi", fps=24)

    state_doc = {
        "config": {"run_id": "test_e7", "source_language": "ko", "compute_tier": "t1"},
        "current_stage": "ASSEMBLE",
        "panels": [
            {
                "panel_id": "panel_001",
                "bbox": {"x": 0, "y": 0, "w": 540, "h": 960},
                "shot": "close_up",
                "route": "i2v",
                "duration_s": 4.0,
                "dialogue": [],
                "sfx": [],
                "characters": [],
            }
        ],
        "clean_plate_paths": {"panel_001": str(plate.as_posix())},
        "clip_candidates": {
            "panel_001": [
                {
                    "panel_id": "panel_001",
                    "path": str(clip_path.as_posix()),
                    "take_number": 1,
                    "route": "i2v",
                    "accepted": True,
                    "qc_scores": {"identity_similarity": 0.9, "overall_pass": True},
                }
            ]
        },
        "audio_tracks": [],
    }
    (run_dir / "state.json").write_text(json.dumps(state_doc), encoding="utf-8")

    with pytest.raises(ValueError, match="I2V clip frame mismatch"):
        assemble_from_run_state(run_dir, fps=24, target_res=(540, 960))


@pytest.mark.worker
@pytest.mark.contract
@pytest.mark.api_ui
@pytest.mark.api
def test_m3_worker_server_async_jobs_and_health(tmp_path: Path) -> None:
    """Verifies M3-7: workers/server.py `/health` video_providers, `/gpu/free`, and `/render/jobs` lifecycle."""
    tc = TestClient(worker_app)
    r_health = tc.get("/health")
    assert r_health.status_code == 200
    h_data = r_health.json()
    assert "video_providers" in h_data
    assert h_data["video_providers"]["kenburns"] == "ready"
    assert h_data["video_providers"]["hunyuan15"] == "disabled"

    plate = _make_clean_anime_panel(tmp_path / "srv_panel.png", size=(360, 640))
    out_clip = tmp_path / "srv_out.avi"
    r_job = tc.post(
        "/render/jobs",
        json={
            "image_path": str(plate),
            "output_path": str(out_clip),
            "route": "parallax",
            "shot_type": "close_up",
            "duration_s": 1.0,
            "fps": 24,
            "width": 360,
            "height": 640,
            "panel_id": "panel_001",
            "seed": 99,
        },
    )
    assert r_job.status_code == 200
    job_id = r_job.json()["job_id"]

    # Poll until done
    import time
    status_data = {}
    for _ in range(100):
        status_data = tc.get(f"/render/jobs/{job_id}").json()
        if status_data["status"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert status_data["status"] == "done"
    assert status_data["result"]["frames_count"] == 24

    r_free = tc.post("/gpu/free", json={})
    assert r_free.status_code == 200


@pytest.mark.eval_qc
@pytest.mark.eval
def test_m5_m6_rubric_calibration_and_bakeoff(tmp_path: Path) -> None:
    """Verifies M5 & M6: rubric Krippendorff's alpha, Bradley-Terry ranks, ROC/Youden/5-fold CV calibration,
    and resumable bake-off runner against FakeComfyUI.
    """
    # 1. Rubric quality score & Krippendorff's alpha
    assert compute_quality_score(motion=5, fidelity=5, stability=5, prompt=5, style=5) == 1.0
    assert compute_quality_score(motion=1, fidelity=1, stability=1, prompt=1, style=1) == 0.0

    identical_units = [[5, 5, 5], [4, 4, 4], [2, 2, 2], [1, 1, 1], [3, 3, 3]]
    assert krippendorff_alpha_ordinal(identical_units) == 1.0

    bt = bradley_terry_ranks(
        {
            "p1": {"wan22-a14b-fp8": 0.92, "wan22-5b-fp8": 0.80, "ltx23-fp8": 0.68},
            "p2": {"wan22-a14b-fp8": 0.89, "wan22-5b-fp8": 0.82, "ltx23-fp8": 0.65},
            "p3": {"wan22-a14b-fp8": 0.94, "wan22-5b-fp8": 0.78, "ltx23-fp8": 0.70},
        }
    )
    assert abs(sum(bt.values()) - 1.0) < 1e-3
    assert bt["wan22-a14b-fp8"] > bt["wan22-5b-fp8"] > bt["ltx23-fp8"]

    # 2. Calibration ROC/Youden + 5-fold CV
    synthetic_samples: List[Dict[str, Any]] = []
    for i in range(15):
        synthetic_samples.append(
            {
                "human_pass": True,
                "qc_scores": {
                    "identity_similarity": 0.86 + (i % 5) * 0.02,
                    "flicker_score": 0.03 + (i % 4) * 0.015,
                    "motion_score": 0.12 + (i % 5) * 0.04,
                    "warp_error": 0.03 + (i % 4) * 0.01,
                    "first_frame_fidelity": 0.88,
                    "style_drift": 0.09,
                    "new_text_count": 0,
                },
            }
        )
        synthetic_samples.append(
            {
                "human_pass": False,
                "qc_scores": {
                    "identity_similarity": 0.55 + (i % 4) * 0.03,
                    "flicker_score": 0.22 + (i % 4) * 0.02,
                    "motion_score": 0.001 if i % 2 == 0 else 0.20,
                    "warp_error": 0.18 + (i % 3) * 0.03,
                    "first_frame_fidelity": 0.62,
                    "style_drift": 0.35,
                    "new_text_count": 1 if i % 3 == 0 else 0,
                },
            }
        )
    cal = calibrate_thresholds(synthetic_samples, route="i2v", min_precision=0.85, n_folds=5)
    assert cal["cv_precision"] >= 0.85
    assert cal["cv_agreement"] >= 0.80

    # 3. Resumable bake-off harness against FakeComfyUI
    mini_cfg = {
        "bakeoff_id": "mini_test",
        "fps": 24,
        "resolution": [360, 640],
        "seeds": [42],
        "panels": [
            {
                "panel_id": "panel_001",
                "category": "closeup",
                "shot": "close_up",
                "duration_s": 1.0,
                "action_prompt": "subtle motion",
                "motion_type": "zoom_in",
            }
        ],
        "models": [
            {"id": "wan22-5b-fp8", "tier": "t1", "route": "i2v"},
            {"id": "parallax", "tier": "cpu", "route": "parallax"},
        ],
    }
    cfg_file = tmp_path / "mini_bakeoff.yaml"
    cfg_file.write_text(yaml.safe_dump(mini_cfg), encoding="utf-8")

    comfy_client, comfy_state = _make_inprocess_comfy_client(tmp_path, mode="normal")
    bake_out = tmp_path / "bake_out"
    res1 = run_bakeoff(config_path=cfg_file, out_dir=bake_out, client=comfy_client)
    assert res1["records_count"] == 2
    assert (bake_out / "bakeoff_report.md").is_file()
    assert (bake_out / "bakeoff_summary.csv").is_file()
    assert (bake_out / "blind_manifest.json").is_file()
    assert (bake_out / "contact_sheets" / "panel_001_sheet.png").is_file()
    submitted_after_first = len(comfy_state.prompts_submitted)
    assert submitted_after_first == 1

    # Re-running bake-off on the same out_dir resumes from records.jsonl with 0 new ComfyUI prompts
    res2 = run_bakeoff(config_path=cfg_file, out_dir=bake_out, client=comfy_client)
    assert res2["records_count"] == 2
    assert len(comfy_state.prompts_submitted) == submitted_after_first


@pytest.mark.contract
@pytest.mark.worker
def test_m3_m7_contract_and_tier_configs() -> None:
    """Verifies M3-1 & M7: Python config loader for t0/t1s/t1/t2 and ClipCandidate/QCScores contract."""
    for tier in ("t0", "t1s", "t1", "t2"):
        cfg = load_config(tier=tier, config_dir=REPO_ROOT / "configs")
        assert cfg.tier == tier

    seed = compute_deterministic_seed("abc123hash", "panel_001", 1)
    assert seed > 0
    cand = ClipCandidate(
        panel_id="panel_001",
        clip_path="runs/r1/clips/p1.avi",
        take=2,
        seed=seed,
        route="i2v",
        model_id="wan22-ti2v-5b",
        gen_seconds=18.5,
        peak_vram_gb=11.2,
        qc_scores=QCScores(
            identity_similarity=0.88,
            flicker_score=0.04,
            motion_score=0.21,
            warp_error=0.03,
            first_frame_fidelity=0.90,
            style_drift=0.09,
            metrics_version=2,
            overall_pass=True,
        ),
        accepted=True,
    )
    dumped = json.loads(cand.model_dump_json())
    assert dumped["path"] == "runs/r1/clips/p1.avi"
    assert dumped["take_number"] == 2
    assert dumped["model_id"] == "wan22-ti2v-5b"
    assert dumped["qc_scores"]["metrics_version"] == 2


@pytest.mark.worker
@pytest.mark.eval_qc
@pytest.mark.eval
def test_real_world_webtoon_qc_and_ocr_regressions(tmp_path: Path) -> None:
    """Verifies real-world webtoon regressions:
    1) Full-width white top/bottom fade bands are not misdetected as speech bubbles.
    2) Non-PIL-bitmap noise strokes do not decode as punctuation gibberish.
    3) Tall webtoon strip (800x1956) rendered via parallax/kenburns passes first_frame_fidelity >= 0.85.
    4) Wan2.2 A14B dual-expert workflows split steps across M2A_SAMPLER and M2A_SAMPLER_LOW.
    """
    from workers.render_parallax import generate_parallax_frames
    from workers.understanding import _decode_pil_bitmap_text, detect_speech_bubbles_and_text

    # 1. Tall panel with a full-width top white fade band (y=0..150, w=800) + stray dark flecks
    tall_img = _make_clean_anime_panel(tmp_path / "tall_fade.png", size=(800, 1956))
    im = Image.open(tall_img).convert("RGB")
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, 800, 150], fill=(250, 250, 250))
    # Add a few clipped dark line fragments along the bottom of the top white fade band
    for x in range(120, 680, 90):
        draw.line([(x, 140), (x + 12, 148)], fill=(30, 30, 30), width=2)
    im.save(tall_img)

    bubbles = detect_speech_bubbles_and_text(im)
    for b in bubbles:
        assert not (b["bbox"]["x"] <= 10 and b["bbox"]["w"] >= int(800 * 0.85))

    # 2. Stray non-bitmap strokes inside a white box must not decode into ASCII symbol gibberish
    noise_crop = Image.new("RGB", (220, 60), (252, 252, 252))
    ndraw = ImageDraw.Draw(noise_crop)
    ndraw.line([(20, 28), (35, 25), (50, 31), (85, 27)], fill=(40, 40, 40), width=2)
    ndraw.point([(110, 29), (130, 27), (155, 30)], fill=(20, 20, 20))
    assert _decode_pil_bitmap_text(noise_crop) is None

    # 3. Tall strip rendered with parallax and kenburns passes aligned first_frame_fidelity >= 0.85
    p_frames = generate_parallax_frames(tall_img, target_res=(576, 1024), duration_s=1.0, fps=16)
    p_qc = compute_clip_qc_metrics(tall_img, p_frames, route="parallax")
    assert p_qc["first_frame_fidelity"] >= 0.85, f"Expected aligned first_frame_fidelity >= 0.85, got {p_qc}"
    assert p_qc["new_text_count"] == 0, f"Expected new_text_count == 0 on deterministic parallax, got {p_qc}"

    # 4. Dual-expert Wan2.2 A14B workflow splits steps across high-noise and low-noise samplers
    a14b_tpl = load_template(REPO_ROOT / "workflows" / "comfyui" / "wan22_i2v_a14b_api.json")
    a14b_inj = inject(
        a14b_tpl,
        image_name="in.png",
        pos_prompt="pos",
        neg_prompt="neg",
        seed=99,
        width=720,
        height=1280,
        frames=49,
        steps=20,
        cfg=5.0,
        fps=16,
    )
    high_nodes = [n for n in a14b_inj.values() if isinstance(n, dict) and n.get("_meta", {}).get("title") == "M2A_SAMPLER"]
    low_nodes = [n for n in a14b_inj.values() if isinstance(n, dict) and n.get("_meta", {}).get("title") == "M2A_SAMPLER_LOW"]
    assert len(high_nodes) == 1 and len(low_nodes) == 1
    assert high_nodes[0]["inputs"]["start_at_step"] == 0
    assert high_nodes[0]["inputs"]["end_at_step"] == 10
    assert low_nodes[0]["inputs"]["start_at_step"] == 10
    assert low_nodes[0]["inputs"]["end_at_step"] == 20


@pytest.mark.eval_qc
@pytest.mark.eval
@pytest.mark.worker
def test_probe_models_and_slideshow_detector(tmp_path: Path) -> None:
    """Verifies C1: `src/m2a/eval/probe_models.py` model file + SHA-256 verification, ComfyUI probe,
    short real-render motion/route gate, and slideshow detector (`audit_run_for_slideshow`) that fails
    loudly when `i2v` panels degrade to `parallax`/`kenburns` or produce frozen frames.
    """
    import hashlib
    from m2a.eval.probe_models import (
        audit_run_for_slideshow,
        probe_comfy_server,
        run_short_real_render,
        verify_model_files,
    )

    # 1. Model file + SHA-256 verification
    models_root = tmp_path / "m2a_models"
    manifest_data = yaml.safe_load((REPO_ROOT / "workflows" / "comfyui" / "manifest.yaml").read_text(encoding="utf-8"))
    t1s_spec = manifest_data["models"]["wan22-5b-gguf-q4"]
    first_dest: Path | None = None
    for f_entry in t1s_spec["files"]:
        subdir = f_entry.get("target_dir", "diffusion_models")
        fname = Path(f_entry["file"]).name
        dest = models_root / subdir / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = f"model-bytes-{fname}".encode("utf-8")
        dest.write_bytes(payload)
        f_entry["sha256"] = hashlib.sha256(payload).hexdigest()
        if first_dest is None:
            first_dest = dest

    temp_manifest = tmp_path / "manifest.yaml"
    temp_manifest.write_text(yaml.safe_dump(manifest_data), encoding="utf-8")

    ok_res = verify_model_files(models_root=models_root, tier="t1s", manifest_path=temp_manifest, verify_sha256=True)
    assert ok_res["passed"] is True
    assert len(ok_res["files"]) == 3
    assert ok_res["missing"] == []
    assert ok_res["sha_mismatches"] == []

    # Corrupt one file -> must fail SHA-256 verification
    assert first_dest is not None
    first_dest.write_bytes(b"corrupted-bytes")
    bad_res = verify_model_files(models_root=models_root, tier="t1s", manifest_path=temp_manifest, verify_sha256=True)
    assert bad_res["passed"] is False
    assert len(bad_res["sha_mismatches"]) == 1

    # 2. ComfyUI server probe & short real render
    client, _ = _make_inprocess_comfy_client(tmp_path, mode="normal")
    srv_probe = probe_comfy_server(comfy_url="http://testserver", model_id="wan22-5b-fp8", client=client)
    assert srv_probe["passed"] is True

    plate = _make_clean_anime_panel(tmp_path / "probe_plate.png", size=(512, 384))
    render_res = run_short_real_render(
        comfy_url="http://testserver",
        model_id="wan22-5b-fp8",
        sample_image=plate,
        out_dir=tmp_path / "probe_render",
        client=client,
    )
    assert render_res["passed"] is True, f"Expected probe render pass, got: {render_res}"
    assert render_res["route_used"] == "i2v"
    assert render_res["motion_score"] >= 0.02

    # 3. Slideshow detector: fail when an `i2v`-routed panel degrades to `parallax`/`kenburns`
    degraded_run = tmp_path / "run_degraded"
    (degraded_run / "clips").mkdir(parents=True)
    mov_frames = generate_kenburns_frames(plate, target_res=(360, 640), duration_s=1.0, fps=16)
    clip_ok = encode_mjpeg_avi(mov_frames, degraded_run / "clips" / "p1.avi", fps=16)
    state_degraded = {
        "config": {"run_id": "deg_01", "compute_tier": "t1s"},
        "panels": [
            {"panel_id": "panel_001", "route": "kenburns", "duration_s": 1.0},
            {"panel_id": "panel_002", "route": "parallax", "duration_s": 1.0},
        ],
        "clip_candidates": {
            "panel_001": [
                {
                    "panel_id": "panel_001",
                    "path": "",
                    "take_number": 1,
                    "route": "i2v",
                    "accepted": False,
                    "render_error": "timeout after 900s",
                },
                {
                    "panel_id": "panel_001",
                    "path": str(clip_ok),
                    "take_number": 2,
                    "route": "kenburns",
                    "accepted": True,
                    "qc_scores": {"motion_score": 0.12, "overall_pass": True},
                },
            ],
            "panel_002": [
                {
                    "panel_id": "panel_002",
                    "path": "",
                    "take_number": 1,
                    "route": "i2v",
                    "accepted": False,
                    "render_error": "timeout after 900s",
                },
                {
                    "panel_id": "panel_002",
                    "path": str(clip_ok),
                    "take_number": 2,
                    "route": "parallax",
                    "accepted": True,
                    "qc_scores": {"motion_score": 0.15, "overall_pass": True},
                },
            ],
        },
    }
    (degraded_run / "state.json").write_text(json.dumps(state_degraded), encoding="utf-8")
    audit_deg = audit_run_for_slideshow(degraded_run, max_degraded_pct=0.0)
    assert audit_deg["passed"] is False
    assert audit_deg["degraded_count"] == 2
    assert len(audit_deg["degraded_panels"]) == 2

    # 4. Slideshow detector: pass when `i2v` panels genuinely render via `i2v` with real motion
    real_run = tmp_path / "run_real_i2v"
    (real_run / "clips").mkdir(parents=True)
    clip_real = encode_mjpeg_avi(mov_frames, real_run / "clips" / "p1_i2v.avi", fps=16)
    state_real = {
        "config": {"run_id": "real_01", "compute_tier": "t1s"},
        "panels": [{"panel_id": "panel_001", "route": "i2v", "duration_s": 1.0}],
        "clip_candidates": {
            "panel_001": [
                {
                    "panel_id": "panel_001",
                    "path": str(clip_real),
                    "take_number": 1,
                    "route": "i2v",
                    "accepted": True,
                    "qc_scores": {"motion_score": 0.18, "overall_pass": True},
                }
            ]
        },
    }
    (real_run / "state.json").write_text(json.dumps(state_real), encoding="utf-8")
    audit_ok = audit_run_for_slideshow(real_run, max_degraded_pct=0.0)
    assert audit_ok["passed"] is True, f"Expected real I2V audit pass, got: {audit_ok}"

    # 5. Slideshow detector: fail when an `i2v` clip consists of frozen/repeated frames
    frozen_run = tmp_path / "run_frozen_i2v"
    (frozen_run / "clips").mkdir(parents=True)
    frozen_frames = [Image.open(plate).convert("RGB") for _ in range(16)]
    clip_frozen = encode_mjpeg_avi(frozen_frames, frozen_run / "clips" / "p1_frozen.avi", fps=16)
    state_frozen = {
        "config": {"run_id": "froz_01", "compute_tier": "t1s"},
        "panels": [{"panel_id": "panel_001", "route": "i2v", "duration_s": 1.0}],
        "clip_candidates": {
            "panel_001": [
                {
                    "panel_id": "panel_001",
                    "path": str(clip_frozen),
                    "take_number": 1,
                    "route": "i2v",
                    "accepted": True,
                    "qc_scores": {"motion_score": 0.0, "overall_pass": True},
                }
            ]
        },
    }
    (frozen_run / "state.json").write_text(json.dumps(state_frozen), encoding="utf-8")
    audit_froz = audit_run_for_slideshow(frozen_run, max_degraded_pct=0.0)
    assert audit_froz["passed"] is False
    assert len(audit_froz["frozen_clips"]) == 1



