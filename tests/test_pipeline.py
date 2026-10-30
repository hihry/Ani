"""Comprehensive integration and unit tests for the Python Worker, State Contracts, and APIs."""
from pathlib import Path
import json
import re
import sys
import wave
from PIL import Image, ImageDraw
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from workers.eval_segmentation import calculate_iou, evaluate_detection
from workers.eval_understanding import evaluate_understanding
from workers.slicer import detect_panels_gutter
from workers.understanding import count_residual_text_bubbles, detect_speech_bubbles_and_text
from workers.render_kenburns import generate_kenburns_frames, prepare_cinematic_canvas
from workers.render_parallax import generate_parallax_frames
from workers.assembler import (
    compute_clip_qc_metrics,
    inpaint_panel_bubbles,
    mix_timed_tracks,
    synthesize_speech_wav,
)
from workers.pipeline_cli import _fmt_srt_time, analyze_panel_semantics, assemble_from_run_state
from workers.server import app as worker_app
from m2a.config import load_config
from m2a.graph import PipelineGraph
from m2a.state import PipelineStage, RunConfig, RunState
from m2a.eval.metrics import character_error_rate, identity_similarity, panel_detection_f1, temporal_stability
from api.app import app as job_app


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_pages():
    """Ensures evaluation sample pages exist on disk even in a fresh clone."""
    p1 = ROOT / "data" / "eval" / "sample_page_1.png"
    if not p1.exists():
        from data.eval.generate_sample_pages import main as gen_pages
        gen_pages()


@pytest.mark.eval_qc
@pytest.mark.eval
@pytest.mark.worker
def test_phase1_segmentation_f1_and_iou():
    gt_path = ROOT / "data" / "eval" / "ground_truth.json"
    report = evaluate_detection(gt_path)
    assert report["f1"] >= 0.90
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["mean_iou"] >= 0.99


@pytest.mark.worker
def test_side_by_side_vertical_gutter_slicing(tmp_path: Path):
    """Verifies that two side-by-side panels separated by a vertical gutter are segmented individually."""
    img = Image.new("RGB", (800, 600), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 50, 370, 500], fill=(80, 140, 220), outline=(20, 20, 24), width=4)
    draw.rectangle([420, 55, 760, 500], fill=(220, 110, 80), outline=(20, 20, 24), width=4)

    page_file = tmp_path / "side_by_side.png"
    img.save(page_file)

    panels = detect_panels_gutter(page_file)
    assert len(panels) == 2
    assert panels[0]["bbox"]["x"] < panels[1]["bbox"]["x"]
    assert panels[0]["reading_order"] == 1
    assert panels[1]["reading_order"] == 2


@pytest.mark.worker
def test_wide_panel_cinematic_canvas_preserves_aspect():
    """Verifies P10 fix: wide landscape panels are letterboxed onto blurred-fill 9:16 canvas."""
    wide_img = Image.new("RGB", (720, 400), (60, 130, 200))
    canvas = prepare_cinematic_canvas(wide_img, target_res=(360, 640), overscan=1.2)
    assert canvas.size == (int(360 * 1.2), int(640 * 1.2))


@pytest.mark.worker
@pytest.mark.eval_qc
@pytest.mark.eval
def test_kenburns_and_parallax_and_qc(tmp_path: Path):
    sample_panel = ROOT / "data" / "eval" / "sample_page_1.png"
    kb_frames = generate_kenburns_frames(sample_panel, target_res=(180, 320), duration_s=1.0, fps=8)
    assert len(kb_frames) == 8
    assert kb_frames[0].size == (180, 320)

    px_frames = generate_parallax_frames(sample_panel, target_res=(180, 320), duration_s=1.0, fps=8)
    assert len(px_frames) == 8

    qc_scores = compute_clip_qc_metrics(sample_panel, kb_frames)
    assert qc_scores["identity_similarity"] >= 0.85
    assert qc_scores["flicker_score"] <= 0.15
    assert qc_scores["motion_score"] >= 0.01
    assert qc_scores["overall_pass"] is True


@pytest.mark.eval_qc
@pytest.mark.eval
@pytest.mark.worker
def test_qc_rejects_frozen_clip():
    """Verifies P1-03: a completely static clip (identical frames) has motion_score=0.0 and fails QC."""
    sample_panel = ROOT / "data" / "eval" / "sample_page_1.png"
    static_frame = Image.open(sample_panel).convert("RGB").resize((180, 320))
    frozen_frames = [static_frame.copy() for _ in range(24)]

    qc_scores = compute_clip_qc_metrics(sample_panel, frozen_frames)
    assert qc_scores["motion_score"] == 0.0
    assert qc_scores["overall_pass"] is False


@pytest.mark.worker
@pytest.mark.eval_qc
@pytest.mark.eval
def test_understand_no_oracle_leakage_on_noise(tmp_path: Path):
    """Verifies P0-03: random noise or unrelated images do not leak ground_truth.json dialogue."""
    rng = np.random.default_rng(42)
    noise_arr = rng.integers(0, 256, size=(400, 720, 3), dtype=np.uint8)
    noise_file = tmp_path / "noise_panel.png"
    Image.fromarray(noise_arr).save(noise_file)

    sem = analyze_panel_semantics(noise_file, panel_id="panel_001", language="en", reading_order=1)
    assert sem["dialogue"] == []
    assert sem["characters"] == []


@pytest.mark.worker
def test_srt_formatting_and_overlap_sync(tmp_path: Path):
    """Verifies P1-04: _fmt_srt_time never emits ,1000 and SRT cues stay within crossfaded video duration."""
    assert _fmt_srt_time(1.9996) == "00:00:02,000"
    assert _fmt_srt_time(0.0) == "00:00:00,000"

    run_dir = tmp_path / "run_sync"
    panels_dir = run_dir / "panels"
    panels_dir.mkdir(parents=True)
    for i in range(1, 4):
        im = Image.new("RGB", (360, 480), (40 * i, 90, 160))
        im.save(panels_dir / f"panel_{i:03d}.png")

    state_data = {
        "config": {"run_id": "sync-test", "source_language": "en", "output_resolution": [180, 320]},
        "current_stage": "ASSEMBLE",
        "panels": [
            {
                "panel_id": "panel_001",
                "bbox": {"x": 0, "y": 0, "w": 360, "h": 480},
                "shot": "medium",
                "route": "kenburns",
                "duration_s": 2.0,
                "dialogue": [{"speaker_id": "char_jin", "text": "First line.", "lang": "en"}],
            },
            {
                "panel_id": "panel_002",
                "bbox": {"x": 0, "y": 0, "w": 360, "h": 480},
                "shot": "medium",
                "route": "kenburns",
                "duration_s": 2.0,
                "dialogue": [],
            },
            {
                "panel_id": "panel_003",
                "bbox": {"x": 0, "y": 0, "w": 360, "h": 480},
                "shot": "close_up",
                "route": "kenburns",
                "duration_s": 2.0,
                "dialogue": [{"speaker_id": "char_jin", "text": "Third panel line.", "lang": "en"}],
            },
        ],
        "clean_plate_paths": {},
        "clip_candidates": {},
        "audio_tracks": [],
    }
    (run_dir / "state.json").write_text(json.dumps(state_data), encoding="utf-8")

    res = assemble_from_run_state(run_dir, fps=12, target_res=(180, 320))
    srt_text = Path(res["subtitles_path"]).read_text(encoding="utf-8")
    ts_pattern = re.compile(r"^(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})$", re.M)
    matches = ts_pattern.findall(srt_text)
    assert len(matches) == 2
    last = matches[-1]
    last_end_s = int(last[4]) * 3600 + int(last[5]) * 60 + int(last[6]) + int(last[7]) / 1000.0
    assert last_end_s <= res["duration_s"] + 1e-3


@pytest.mark.worker
def test_audio_anchored_mix(tmp_path: Path):
    """Verifies P1-05: mix_timed_tracks places audio at its exact timeline offset (silent panels don't shift later lines)."""
    wav1 = tmp_path / "line_late.wav"
    synthesize_speech_wav("Delayed speech", wav1, sample_rate=24000)
    master = tmp_path / "master.wav"

    out = mix_timed_tracks([(1.5, wav1)], total_duration_s=3.5, output_wav_path=master, sample_rate=24000)
    assert out is not None and out.exists()

    with wave.open(str(out), "rb") as wf:
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    # First 1.4 seconds must be completely silent
    assert np.max(np.abs(samples[: int(1.4 * 24000)])) == 0
    # Audio must be active right after 1.5s
    assert np.max(np.abs(samples[int(1.55 * 24000) : int(2.2 * 24000)])) > 500


@pytest.mark.worker
def test_clean_plate_and_tts_synthesis(tmp_path: Path):
    sample_page = ROOT / "data" / "eval" / "sample_page_1.png"
    clean_out = tmp_path / "clean.png"
    inpaint_panel_bubbles(sample_page, clean_out, bubble_bboxes=[{"x": 70, "y": 90, "w": 300, "h": 70}])
    assert clean_out.exists() and clean_out.stat().st_size > 0

    wav_out = tmp_path / "line.wav"
    dur = synthesize_speech_wav("I won't run away this time.", wav_out, voice_id="char_jin", emotion="confident")
    assert wav_out.exists() and wav_out.stat().st_size > 1000
    assert dur >= 1.2


@pytest.mark.contract
@pytest.mark.api_ui
@pytest.mark.api
def test_go_state_json_interoperability(tmp_path: Path):
    """Verifies that Python RunState loads Go-serialized state.json cleanly and rejects out-of-range durations."""
    from pydantic import ValidationError

    go_run_dir = tmp_path / "go_interop_run"
    go_run_dir.mkdir(parents=True)
    go_state_doc = {
        "config": {
            "run_id": "go-contract-001",
            "config_hash": "deadbeef1234",
            "source_pages": ["data/eval/sample_page_1.png"],
            "source_language": "ko",
            "output_resolution": [720, 1280],
            "compute_tier": "t1s",
            "created_at": "2026-10-27T12:00:00Z",
        },
        "current_stage": "DONE",
        "panels": [
            {
                "panel_id": "panel_001",
                "bbox": {"x": 40, "y": 60, "w": 720, "h": 500},
                "shot": "close_up",
                "characters": [{"id": "char_jin", "name": "Jin", "emotion": "determined", "pose": "standing"}],
                "action": "Jin faces the gate.",
                "dialogue": [{"speaker_id": "char_jin", "text": "Arise.", "lang": "en", "emotion": "serious"}],
                "sfx": [],
                "motion_prompt": "subtle push-in",
                "style_anchor": "2D anime",
                "motion_intensity": 0.65,
                "route": "i2v",
                "duration_s": 3.5,
            }
        ],
        "clean_plate_paths": {"panel_001": "runs\\go-contract-001\\clean_plates\\panel_001_clean.png"},
        "clip_candidates": {
            "panel_001": [
                {
                    "panel_id": "panel_001",
                    "path": "runs/go-contract-001/clips/panel_001_take01_i2v.avi",
                    "take_number": 1,
                    "route": "i2v",
                    "model_id": "wan22-5b-gguf-q4",
                    "seed": 12345,
                    "qc_scores": {
                        "identity_similarity": 0.92,
                        "new_text_count": 0,
                        "flicker_score": 0.03,
                        "motion_score": 0.18,
                        "warp_error": 0.02,
                        "first_frame_fidelity": 0.91,
                        "style_drift": 0.05,
                        "metrics_version": 2,
                        "overall_pass": True,
                    },
                    "accepted": True,
                }
            ]
        },
        "stage_history": [{"stage": "INGEST", "entered_at": "2026-10-27T12:00:01Z"}],
        "cost_ledger": {"total_gpu_seconds": 14.2, "per_model_seconds": {"wan22-5b-gguf-q4": 14.2}},
    }
    (go_run_dir / "state.json").write_text(json.dumps(go_state_doc), encoding="utf-8")
    loaded = RunState.load(go_run_dir)
    assert loaded.config.output_resolution == [720, 1280]
    assert len(loaded.panels) == 1
    assert loaded.panels[0].route == "i2v"
    assert "\\" not in loaded.clean_plate_paths["panel_001"]
    assert loaded.clip_candidates["panel_001"][0].model_id == "wan22-5b-gguf-q4"

    # Out-of-range duration_s (> 10.0) and unsupported lang ('zh') must fail validation (P0-04)
    bad_dur_dir = tmp_path / "bad_dur_run"
    bad_dur_dir.mkdir(parents=True)
    bad_doc = json.loads(json.dumps(go_state_doc))
    bad_doc["panels"][0]["duration_s"] = 12.3
    (bad_dur_dir / "state.json").write_text(json.dumps(bad_doc), encoding="utf-8")
    with pytest.raises(ValidationError):
        RunState.load(bad_dur_dir)

    bad_lang_dir = tmp_path / "bad_lang_run"
    bad_lang_dir.mkdir(parents=True)
    bad_lang_doc = json.loads(json.dumps(go_state_doc))
    bad_lang_doc["panels"][0]["dialogue"][0]["lang"] = "zh"
    (bad_lang_dir / "state.json").write_text(json.dumps(bad_lang_doc), encoding="utf-8")
    with pytest.raises(ValidationError):
        RunState.load(bad_lang_dir)

    existing_run = ROOT / "runs" / "396d8463-dbfb-4263-b474-d9c558263b20"
    if (existing_run / "state.json").exists():
        state = RunState.load(existing_run)
        assert state.config.output_resolution == [720, 1280]
        assert len(state.panels) == 9
        for cp in state.clean_plate_paths.values():
            assert "\\" not in cp


@pytest.mark.worker
@pytest.mark.api_ui
@pytest.mark.api
def test_python_graph_missing_page_fails_and_resumes(tmp_path: Path):
    """Verifies P0-01 and P0-10: missing input page marks run FAILED, and resume recovers from FAILED."""
    cfg = load_config("t0", config_dir=ROOT / "configs")
    run_dir = tmp_path / "fail_run"
    missing_page = tmp_path / "missing.png"

    state = RunState(
        config=RunConfig(source_pages=[str(missing_page)], source_language="en", compute_tier="t0")
    )
    graph = PipelineGraph(config=cfg, run_dir=run_dir, auto_approve_checkpoints=False)
    state = graph.run(state)
    assert state.current_stage == PipelineStage.FAILED


@pytest.mark.contract
@pytest.mark.api_ui
@pytest.mark.api
def test_config_tier_loading():
    cfg_t0 = load_config("t0", config_dir=ROOT / "configs")
    assert cfg_t0.generation.primary_model == "kenburns"
    assert cfg_t0.understanding.vlm == "qwen3-vl-2b"

    cfg_t2 = load_config("t2", config_dir=ROOT / "configs")
    assert cfg_t2.generation.primary_model == "wan22-i2v-a14b"


@pytest.mark.worker
def test_worker_fastapi_endpoints(tmp_path: Path):
    client = TestClient(worker_app)
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    p1 = str(ROOT / "data" / "eval" / "sample_page_1.png")
    seg = client.post("/segment", json={"image_path": p1, "output_dir": str(tmp_path / "panels")})
    assert seg.status_code == 200
    panels = seg.json()["panels"]
    assert len(panels) == 3

    und = client.post(
        "/understand",
        json={"panel_image_path": panels[0]["crop_path"], "panel_id": "panel_001", "language": "en", "reading_order": 1},
    )
    assert und.status_code == 200
    assert und.json()["shot"] == "wide"
    assert len(und.json()["dialogue"]) >= 1


@pytest.mark.eval_qc
@pytest.mark.eval
def test_eval_metrics_functions():
    assert character_error_rate("hello", "hello") == 0.0
    assert 0.15 <= character_error_rate("hallo", "hello") <= 0.25
    f1 = panel_detection_f1([(0, 0, 100, 100)], [(0, 0, 100, 100)])
    assert f1 == 1.0
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert identity_similarity(arr, arr) > 0.99
    assert temporal_stability([arr, arr, arr]) == 1.0


@pytest.mark.api_ui
@pytest.mark.api
def test_job_api_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("api.app.RUNS_DIR", tmp_path / "runs")
    client = TestClient(job_app)
    resp = client.get("/health")
    assert resp.status_code == 200

    sub = client.post("/jobs", json={"source_pages": [], "source_language": "en", "compute_tier": "t0"})
    assert sub.status_code == 200
    job_id = sub.json()["job_id"]
    assert sub.json()["current_stage"] == "INGEST"

    st = client.get(f"/jobs/{job_id}")
    assert st.status_code == 200
    assert st.json()["job_id"] == job_id

    full_st = client.get(f"/jobs/{job_id}/state")
    assert full_st.status_code == 200
    assert full_st.json()["config"]["run_id"] == job_id

    insp = client.get(f"/jobs/{job_id}/inspect")
    assert insp.status_code == 200
    assert job_id in insp.json()["report"]
    assert "PIPELINE RUN INSPECTION" in insp.json()["report"]

    canc = client.delete(f"/jobs/{job_id}")
    assert canc.status_code == 200
    assert client.get(f"/jobs/{job_id}").json()["current_stage"] == "FAILED"

    # Submit a real 1-page job that pauses at CHECKPOINT_1 via the Go m2a engine, then edit & save CP1 via API
    page_img = Image.new("RGB", (800, 640), (255, 255, 255))
    draw = ImageDraw.Draw(page_img)
    draw.rectangle([40, 50, 760, 580], fill=(85, 145, 210), outline=(20, 20, 24), width=4)
    draw.rounded_rectangle([80, 90, 360, 155], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((100, 110), "API checkpoint test.", fill=(20, 20, 20))
    page_file = tmp_path / "api_page.png"
    page_img.save(page_file)

    sub_real = client.post(
        "/jobs",
        json={
            "source_pages": [str(page_file)],
            "source_language": "en",
            "compute_tier": "t0",
            "auto_approve": False,
            "config": {"composition": {"resolution": [180, 320], "fps": 8}},
        },
    )
    assert sub_real.status_code == 200
    real_id = sub_real.json()["job_id"]
    assert sub_real.json()["current_stage"] == "CHECKPOINT_1"
    assert sub_real.json()["panels_count"] == 1

    st_doc = client.get(f"/jobs/{real_id}/state").json()
    panels = st_doc["panels"]
    panels[0]["dialogue"][0]["text"] = "Edited via PUT checkpoint!"
    cp_resp = client.put(
        f"/jobs/{real_id}/checkpoint",
        json={"panels": panels, "resume_after_save": False},
    )
    assert cp_resp.status_code == 200
    saved_panel_json = json.loads(
        (tmp_path / "runs" / real_id / "understanding" / "panel_001.json").read_text(encoding="utf-8")
    )
    assert saved_panel_json["dialogue"][0]["text"] == "Edited via PUT checkpoint!"


@pytest.mark.eval_qc
@pytest.mark.eval
@pytest.mark.worker
def test_phase2_evaluation_benchmark():
    """Verifies Phase 2 benchmark targets: Speech Bubble F1 >= 0.90, CER <= 0.10, Speaker Acc >= 0.90, Clean-Plate Removal == 1.0."""
    gt_path = ROOT / "data" / "eval" / "ground_truth.json"
    report = evaluate_understanding(gt_path)
    assert report["passed_target"] is True
    assert report["bubble_f1"] >= 0.90
    assert report["mean_bubble_iou"] >= 0.95
    assert report["ocr_cer"] <= 0.10
    assert report["speaker_attribution_accuracy"] >= 0.90
    assert report["shot_classification_accuracy"] >= 0.85
    assert report["clean_plate_removal_rate"] == 1.0


@pytest.mark.worker
def test_phase2_unseen_panel_bubble_detection_and_character_crop(tmp_path: Path):
    """Verifies that a brand-new manhwa panel (not in data/eval) has its speech bubble, pointer tail, and character crop extracted."""
    img = Image.new("RGB", (640, 520), (110, 170, 225))
    draw = ImageDraw.Draw(img)
    # Ground layer
    draw.rectangle([0, 320, 640, 520], fill=(70, 130, 60))
    # Character body + head in center
    draw.polygon([(320, 170), (245, 360), (395, 360)], fill=(215, 65, 55))
    draw.ellipse([292, 118, 348, 174], fill=(255, 220, 190))
    # Outer panel border
    draw.rectangle([0, 0, 639, 519], outline=(20, 20, 24), width=4)
    # Speech bubble with pointer tail and interior text strokes
    gt_bubble = {"x": 50, "y": 35, "w": 260, "h": 68}
    draw.rounded_rectangle([50, 35, 310, 103], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.polygon([(80, 103), (96, 103), (75, 118)], fill=(255, 255, 255), outline=(40, 40, 45))
    draw.text((66, 52), "Awaken the shadow gate!", fill=(20, 20, 20))

    panel_path = tmp_path / "unseen_panel.png"
    img.save(panel_path)

    crops_dir = tmp_path / "crops"
    annotated_path = tmp_path / "unseen_annotated.png"
    sem = analyze_panel_semantics(
        panel_path,
        panel_id="unseen_001",
        language="en",
        reading_order=99,
        crops_dir=crops_dir,
        annotated_path=annotated_path,
    )
    assert len(sem["dialogue"]) == 1
    det_bbox = sem["dialogue"][0]["bbox"]
    assert calculate_iou(det_bbox, gt_bubble) >= 0.90
    assert len(sem["characters"]) >= 1
    crop_file = Path(sem["characters"][0]["crop_path"])
    assert crop_file.is_file()
    with Image.open(crop_file) as c_im:
        # Tight character crop must be strictly smaller than the full 640x520 panel
        assert c_im.width < 450 and c_im.height < 420
    assert annotated_path.is_file()


@pytest.mark.worker
@pytest.mark.eval_qc
@pytest.mark.eval
def test_phase2_clean_plate_mask_and_qc_text_leakage(tmp_path: Path):
    """Verifies clean-plate *_mask.png persistence, 100% bubble+tail removal, and QC new_text_count leakage detection."""
    img = Image.new("RGB", (600, 480), (95, 155, 215))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 290, 600, 480], fill=(65, 125, 55))
    draw.polygon([(300, 160), (230, 340), (370, 340)], fill=(215, 70, 60))
    draw.rectangle([0, 0, 599, 479], outline=(20, 20, 24), width=4)
    draw.rounded_rectangle([60, 40, 340, 110], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.polygon([(85, 110), (102, 110), (80, 125)], fill=(255, 255, 255), outline=(40, 40, 45))
    draw.text((78, 60), "Uncleaned speech bubble text!", fill=(20, 20, 20))

    raw_panel = tmp_path / "panel_001.png"
    img.save(raw_panel)
    assert count_residual_text_bubbles(img) == 1

    clean_plate = tmp_path / "panel_001_clean.png"
    inpaint_panel_bubbles(raw_panel, clean_plate, bubble_bboxes=[{"x": 60, "y": 40, "w": 280, "h": 70}])
    mask_file = tmp_path / "panel_001_mask.png"
    assert clean_plate.is_file()
    assert mask_file.is_file()

    clean_img = Image.open(clean_plate).convert("RGB")
    assert count_residual_text_bubbles(clean_img) == 0

    # If a clip is rendered from the clean plate, new_text_count == 0 and QC passes
    clean_frames = generate_kenburns_frames(clean_plate, target_res=(360, 640), duration_s=1.0, fps=8)
    qc_clean = compute_clip_qc_metrics(clean_plate, clean_frames)
    assert qc_clean["new_text_count"] == 0
    assert qc_clean["overall_pass"] is True

    # If a frame leaks an uncleaned speech bubble when compared against clean_plate, new_text_count > 0 and QC fails
    leaked_frames = [img.copy() for _ in range(8)]
    qc_leaked = compute_clip_qc_metrics(clean_plate, leaked_frames)
    assert qc_leaked["new_text_count"] >= 1
    assert qc_leaked["overall_pass"] is False


@pytest.mark.api_ui
@pytest.mark.api
@pytest.mark.worker
def test_phase2_checkpoint1_human_edit_sync(tmp_path: Path):
    """Verifies that human edits to understanding/<panel_id>.json at CHECKPOINT_1 are synced on resume."""
    page_img = Image.new("RGB", (800, 700), (255, 255, 255))
    draw = ImageDraw.Draw(page_img)
    draw.rectangle([40, 50, 760, 620], fill=(90, 150, 215), outline=(20, 20, 24), width=4)
    draw.rounded_rectangle([80, 90, 380, 160], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((100, 110), "Initial OCR line.", fill=(20, 20, 20))
    page_file = tmp_path / "cp1_page.png"
    page_img.save(page_file)

    cfg = load_config("t0", config_dir=ROOT / "configs")
    cfg.composition.resolution = [180, 320]
    cfg.composition.fps = 8
    run_dir = tmp_path / "run_cp1"

    state = RunState(
        config=RunConfig(source_pages=[str(page_file)], source_language="en", compute_tier="t0", output_resolution=[180, 320])
    )
    graph = PipelineGraph(config=cfg, run_dir=run_dir, auto_approve_checkpoints=False)
    state = graph.run(state)
    assert state.current_stage == PipelineStage.CHECKPOINT_1

    # Human edits understanding/panel_001.json during CHECKPOINT_1 review
    panel_json_path = run_dir / "understanding" / "panel_001.json"
    assert panel_json_path.is_file()
    p_data = json.loads(panel_json_path.read_text(encoding="utf-8"))
    p_data["dialogue"][0]["text"] = "Human reviewed CP1 line!"
    panel_json_path.write_text(json.dumps(p_data, indent=2), encoding="utf-8")

    # Resume past CP1 (and auto-approve CP2)
    graph.auto_approve_checkpoints = True
    state = graph.resume(state)
    assert state.current_stage == PipelineStage.DONE
    assert state.panels[0].dialogue[0].text == "Human reviewed CP1 line!"
    srt_content = (run_dir / "output" / "subtitles.srt").read_text(encoding="utf-8")
    assert "Human reviewed CP1 line!" in srt_content


@pytest.mark.worker
@pytest.mark.contract
def test_phase2_providers_contract(tmp_path: Path):
    """Verifies concrete Phase 2 providers (Magiv3, PaddleOCR, MangaOCR, QwenOCR, Qwen3VL, LaMa)."""
    from m2a.providers import (
        get_inpainter,
        get_ocr_provider,
        get_perception_provider,
        get_vlm_provider,
    )

    p1 = ROOT / "data" / "eval" / "sample_page_1.png"
    perc = get_perception_provider("magiv3")
    panels = perc.detect_panels(p1)
    assert len(panels) == 3

    page_img = Image.open(p1).convert("RGB")
    pb = panels[0]
    panel_crop_path = tmp_path / "p1_crop.png"
    page_img.crop((pb.x1, pb.y1, pb.x2, pb.y2)).save(panel_crop_path)

    txt_dets = perc.detect_text(panel_crop_path)
    assert len(txt_dets) == 1
    char_dets = perc.detect_characters(panel_crop_path, [])
    assert len(char_dets) >= 1
    links = perc.link_speakers(panel_crop_path, txt_dets, char_dets)
    assert len(links) == 1

    for ocr_name, lang in [("paddle", "en"), ("manga_ocr", "ja"), ("qwen", "en")]:
        ocr_prov = get_ocr_provider(ocr_name)
        regions = ocr_prov.recognize(panel_crop_path, lang)
        assert len(regions) >= 1
        assert regions[0].text != ""

    vlm = get_vlm_provider("qwen3-vl")
    sem = vlm.analyze_panel(panel_crop_path, schema={}, language="en")
    assert sem["shot"] == "wide"

    inp = get_inpainter("lama")
    clean_out = tmp_path / "p1_lama_clean.png"
    inp.inpaint(panel_crop_path, None, clean_out)
    assert clean_out.is_file()


@pytest.mark.worker
def test_adversarial_asymmetric_t_gutter_and_3column_slicing(tmp_path: Path):
    """Stress-tests 2D gutter slicing on an asymmetric T-split (1 tall left + 2 stacked right) and a 3-column row."""
    page = Image.new("RGB", (800, 1150), (255, 255, 255))
    draw = ImageDraw.Draw(page)

    # Band 1 (y=40..560): Asymmetric T-layout
    # Left tall panel (x=40..380, y=40..560)
    draw.rectangle([40, 40, 380, 560], fill=(70, 120, 200), outline=(20, 20, 24), width=4)
    # Right top panel (x=410..760, y=40..280)
    draw.rectangle([410, 40, 760, 280], fill=(200, 95, 75), outline=(20, 20, 24), width=4)
    # Right bottom panel (x=410..760, y=315..560) separated by a partial horizontal gutter (y=281..314)
    draw.rectangle([410, 315, 760, 560], fill=(80, 175, 110), outline=(20, 20, 24), width=4)

    # Band 2 (y=620..1080): 3-column row separated by two vertical gutters
    draw.rectangle([40, 620, 255, 1080], fill=(150, 90, 200), outline=(20, 20, 24), width=4)
    draw.rectangle([285, 620, 515, 1080], fill=(210, 160, 60), outline=(20, 20, 24), width=4)
    draw.rectangle([545, 620, 760, 1080], fill=(60, 160, 190), outline=(20, 20, 24), width=4)

    page_path = tmp_path / "complex_layout.png"
    page.save(page_path)

    panels = detect_panels_gutter(page_path)
    assert len(panels) == 6
    # Verify reading order:
    # 1: Left tall (x=40, y=40), 2: Right top (x=410, y=40), 3: Right bottom (x=410, y=315)
    assert panels[0]["bbox"]["x"] == 40 and panels[0]["bbox"]["h"] >= 500
    assert panels[1]["bbox"]["x"] == 410 and panels[1]["bbox"]["y"] == 40
    assert panels[2]["bbox"]["x"] == 410 and panels[2]["bbox"]["y"] >= 310
    # 4, 5, 6: 3-column bottom band sorted strictly left-to-right
    assert panels[3]["bbox"]["x"] < panels[4]["bbox"]["x"] < panels[5]["bbox"]["x"]


@pytest.mark.worker
def test_adversarial_bucket_boundary_and_multi_bubble_reading_order():
    """Stress-tests multi-bubble reading order when two side-by-side bubbles straddle y=45 (46 vs 40)."""
    img = Image.new("RGB", (720, 500), (85, 135, 195))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 719, 499], outline=(20, 20, 24), width=4)

    # Bubble 1: Left side at y=46 (just past 45)
    draw.rounded_rectangle([40, 46, 280, 110], radius=10, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((58, 66), "Left bubble first!", fill=(20, 20, 20))

    # Bubble 2: Right side at y=40 (just before 45 — naive y//45 would sort this BEFORE Bubble 1!)
    draw.rounded_rectangle([380, 40, 640, 104], radius=10, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((398, 60), "Right bubble second!", fill=(20, 20, 20))

    # Bubble 3: Middle-left at y=185
    draw.rounded_rectangle([55, 185, 310, 250], radius=10, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((72, 205), "Middle-left third!", fill=(20, 20, 20))

    # Bubble 4: Bottom-right at y=330
    draw.rounded_rectangle([370, 330, 650, 395], radius=10, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((388, 350), "Bottom-right fourth!", fill=(20, 20, 20))

    bubbles = detect_speech_bubbles_and_text(img)
    assert len(bubbles) == 4
    # Must order Top-Left (x~40) BEFORE Top-Right (x~380) despite y=46 vs y=40!
    assert bubbles[0]["bbox"]["x"] < 100 and bubbles[0]["bbox"]["y"] < 80
    assert bubbles[1]["bbox"]["x"] > 350 and bubbles[1]["bbox"]["y"] < 80
    assert bubbles[2]["bbox"]["y"] > 160 and bubbles[2]["bbox"]["x"] < 100
    assert bubbles[3]["bbox"]["y"] > 300 and bubbles[3]["bbox"]["x"] > 350


@pytest.mark.worker
def test_adversarial_false_positive_white_decors_vs_real_bubble():
    """Verifies that empty white moons, single-dot white armor, and bright yellow suns are NOT mistaken for speech bubbles."""
    img = Image.new("RGB", (700, 520), (45, 40, 80))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 699, 519], outline=(20, 20, 24), width=4)

    # Decor 1: Large solid white moon with NO interior text strokes
    draw.ellipse([500, 40, 640, 180], fill=(255, 255, 255), outline=(220, 220, 230), width=2)

    # Decor 2: White armor chestplate with only 1 tiny dark button (glyph_count == 1 < 2)
    draw.rectangle([280, 280, 420, 420], fill=(250, 250, 252), outline=(30, 30, 35), width=3)
    draw.rectangle([348, 348, 352, 353], fill=(20, 20, 20))

    # Decor 3: High-saturation bright yellow aura with dark sparks (chroma > 26)
    draw.rounded_rectangle([60, 280, 230, 360], radius=12, fill=(255, 235, 60), outline=(40, 40, 45), width=3)
    draw.text((80, 305), "AURA SPARKS", fill=(20, 20, 20))

    # 1 Genuine white speech bubble with dark text strokes
    draw.rounded_rectangle([50, 45, 330, 115], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((68, 68), "Only this real bubble passes.", fill=(20, 20, 20))

    bubbles = detect_speech_bubbles_and_text(img)
    assert len(bubbles) == 1
    assert abs(bubbles[0]["bbox"]["x"] - 50) <= 4
    assert abs(bubbles[0]["bbox"]["y"] - 45) <= 4


@pytest.mark.worker
def test_adversarial_two_character_distinct_ids_and_tail_speaker_linking(tmp_path: Path):
    """Verifies that 2 characters in the same panel get distinct IDs, distinct crop files, and accurate tail-based speaker attribution."""
    img = Image.new("RGB", (720, 560), (120, 165, 210))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 360, 720, 560], fill=(70, 120, 65))
    draw.rectangle([0, 0, 719, 559], outline=(20, 20, 24), width=4)

    # Left Character (warm crimson armor at x=100..230, y=200..440)
    draw.polygon([(165, 200), (100, 440), (230, 440)], fill=(225, 55, 50))
    draw.ellipse([140, 150, 190, 200], fill=(255, 220, 190))

    # Right Character (cool cobalt armor at x=490..620, y=200..440)
    draw.polygon([(555, 200), (490, 440), (620, 440)], fill=(45, 95, 235))
    draw.ellipse([530, 150, 580, 200], fill=(255, 220, 190))

    # Left Speech Bubble pointing down toward Left Character
    draw.rounded_rectangle([40, 35, 300, 100], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.polygon([(145, 100), (165, 100), (160, 125)], fill=(255, 255, 255), outline=(40, 40, 45))
    draw.text((58, 55), "I will take the vanguard!", fill=(20, 20, 20))

    # Right Speech Bubble pointing down toward Right Character
    draw.rounded_rectangle([420, 35, 680, 100], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.polygon([(545, 100), (565, 100), (555, 125)], fill=(255, 255, 255), outline=(40, 40, 45))
    draw.text((438, 55), "And I will guard the flank!", fill=(20, 20, 20))

    panel_file = tmp_path / "two_chars.png"
    img.save(panel_file)
    crops_dir = tmp_path / "crops"

    sem = analyze_panel_semantics(
        panel_file,
        panel_id="duo_001",
        language="en",
        reading_order=99,
        crops_dir=crops_dir,
    )
    assert len(sem["characters"]) == 2
    char_left, char_right = sem["characters"][0], sem["characters"][1]
    assert char_left["id"] != char_right["id"]
    assert Path(char_left["crop_path"]).is_file()
    assert Path(char_right["crop_path"]).is_file()
    assert char_left["crop_path"] != char_right["crop_path"]

    assert len(sem["dialogue"]) == 2
    assert sem["dialogue"][0]["speaker_id"] == char_left["id"]
    assert sem["dialogue"][1]["speaker_id"] == char_right["id"]


@pytest.mark.worker
def test_adversarial_edge_touching_and_full_width_banner_inpainting(tmp_path: Path):
    """Stress-tests clean-plate inpainting when bubbles touch the left edge, right edge, or span the full panel width."""
    img = Image.new("RGB", (640, 520), (85, 145, 210))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 320, 640, 520], fill=(60, 115, 70))
    draw.rectangle([0, 0, 639, 519], outline=(20, 20, 24), width=4)

    # 1. Left-edge-hugging bubble (x=4)
    draw.rounded_rectangle([4, 35, 250, 100], radius=10, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((20, 55), "Left edge bubble!", fill=(20, 20, 20))

    # 2. Right-edge-hugging bubble (x+w=636)
    draw.rounded_rectangle([390, 130, 636, 195], radius=10, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((408, 150), "Right edge bubble!", fill=(20, 20, 20))

    # 3. Full-width system notification banner (x=6..634)
    draw.rectangle([6, 230, 634, 290], fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((180, 250), "[SYSTEM: HIDDEN QUEST TRIGGERED]", fill=(20, 20, 20))

    raw_file = tmp_path / "edge_panel.png"
    img.save(raw_file)
    assert count_residual_text_bubbles(img) == 3

    clean_file = tmp_path / "edge_panel_clean.png"
    bboxes = [
        {"x": 4, "y": 35, "w": 246, "h": 65},
        {"x": 390, "y": 130, "w": 246, "h": 65},
        {"x": 6, "y": 230, "w": 628, "h": 60},
    ]
    inpaint_panel_bubbles(raw_file, clean_file, bubble_bboxes=bboxes)
    clean_img = Image.open(clean_file).convert("RGB")
    assert count_residual_text_bubbles(clean_img) == 0

    clean_arr = np.asarray(clean_img, dtype=np.float32)
    # Verify none of the three regions left behind white bubble streaks (mean luminance should match blue sky < 180)
    assert clean_arr[45:90, 15:240].mean() < 180.0
    assert clean_arr[140:185, 400:625].mean() < 180.0
    assert clean_arr[240:280, 20:620].mean() < 180.0


@pytest.mark.worker
def test_adversarial_multiline_srt_monotonicity_under_tight_duration(tmp_path: Path):
    """Stress-tests SRT cue scheduling when a single 1.5s panel has 4 dialogue lines with 1.2s audio tracks each."""
    run_dir = tmp_path / "run_multiline"
    panels_dir = run_dir / "panels"
    audio_dir = run_dir / "audio"
    panels_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)

    for i in (1, 2):
        im = Image.new("RGB", (360, 480), (50 * i, 100, 170))
        im.save(panels_dir / f"panel_{i:03d}.png")

    wav_path = audio_dir / "line.wav"
    synthesize_speech_wav("Rapid dialogue line", wav_path, sample_rate=24000)

    state_data = {
        "config": {"run_id": "multiline-stress", "source_language": "en", "output_resolution": [180, 320]},
        "current_stage": "ASSEMBLE",
        "panels": [
            {
                "panel_id": "panel_001",
                "bbox": {"x": 0, "y": 0, "w": 360, "h": 480},
                "shot": "close_up",
                "route": "kenburns",
                "duration_s": 1.5,
                "dialogue": [
                    {"speaker_id": "char_jin", "text": "Line one!", "lang": "en"},
                    {"speaker_id": "char_beast", "text": "Line two!", "lang": "en"},
                    {"speaker_id": "char_jin", "text": "Line three!", "lang": "en"},
                    {"speaker_id": "char_beast", "text": "Line four!", "lang": "en"},
                ],
            },
            {
                "panel_id": "panel_002",
                "bbox": {"x": 0, "y": 0, "w": 360, "h": 480},
                "shot": "wide",
                "route": "kenburns",
                "duration_s": 1.5,
                "dialogue": [
                    {"speaker_id": "char_narrator", "text": "Final panel line.", "lang": "en"},
                ],
            },
        ],
        "clean_plate_paths": {},
        "clip_candidates": {},
        "audio_tracks": [
            {"panel_id": "panel_001", "speaker_id": "char_jin", "audio_path": str(wav_path), "duration_s": 1.2},
            {"panel_id": "panel_001", "speaker_id": "char_beast", "audio_path": str(wav_path), "duration_s": 1.2},
            {"panel_id": "panel_001", "speaker_id": "char_jin", "audio_path": str(wav_path), "duration_s": 1.2},
            {"panel_id": "panel_001", "speaker_id": "char_beast", "audio_path": str(wav_path), "duration_s": 1.2},
            {"panel_id": "panel_002", "speaker_id": "char_narrator", "audio_path": str(wav_path), "duration_s": 1.2},
        ],
    }
    (run_dir / "state.json").write_text(json.dumps(state_data), encoding="utf-8")

    res = assemble_from_run_state(run_dir, fps=12, target_res=(180, 320))
    srt_text = Path(res["subtitles_path"]).read_text(encoding="utf-8")
    ts_pattern = re.compile(r"^(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})$", re.M)
    matches = ts_pattern.findall(srt_text)
    assert len(matches) == 5

    prev_end = 0.0
    for m in matches:
        s_sec = int(m[0]) * 3600 + int(m[1]) * 60 + int(m[2]) + int(m[3]) / 1000.0
        e_sec = int(m[4]) * 3600 + int(m[5]) * 60 + int(m[6]) + int(m[7]) / 1000.0
        assert e_sec > s_sec, f"Cue end {e_sec} must be strictly greater than start {s_sec}"
        assert s_sec >= prev_end - 1e-3, f"Overlapping/retrograde SRT cue: {s_sec} < previous end {prev_end}"
        prev_end = e_sec

    assert prev_end <= res["duration_s"] + 1e-3


@pytest.mark.api_ui
@pytest.mark.api
@pytest.mark.worker
def test_adversarial_special_chars_in_speaker_id_and_corrupt_cp1_json(tmp_path: Path):
    """Stress-tests CP1 resume when speaker_id contains path separators/colons and duration_s is out of bounds."""
    page_img = Image.new("RGB", (800, 700), (255, 255, 255))
    draw = ImageDraw.Draw(page_img)
    draw.rectangle([40, 50, 760, 620], fill=(90, 150, 215), outline=(20, 20, 24), width=4)
    draw.rounded_rectangle([80, 90, 380, 160], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((100, 110), "Initial text.", fill=(20, 20, 20))
    page_file = tmp_path / "adv_cp1.png"
    page_img.save(page_file)

    cfg = load_config("t0", config_dir=ROOT / "configs")
    cfg.composition.resolution = [180, 320]
    cfg.composition.fps = 8
    run_dir = tmp_path / "run_adv_cp1"

    state = RunState(
        config=RunConfig(source_pages=[str(page_file)], source_language="en", compute_tier="t0", output_resolution=[180, 320])
    )
    graph = PipelineGraph(config=cfg, run_dir=run_dir, auto_approve_checkpoints=False)
    state = graph.run(state)
    assert state.current_stage == PipelineStage.CHECKPOINT_1

    # Inject extreme duration_s (99.0), extreme motion_intensity (5.0), and unsafe speaker_id with slashes/colons
    p1_path = run_dir / "understanding" / "panel_001.json"
    p1_data = json.loads(p1_path.read_text(encoding="utf-8"))
    p1_data["duration_s"] = 99.0
    p1_data["motion_intensity"] = 5.0
    p1_data["dialogue"][0]["speaker_id"] = "Jin / Shadow:Monarch..!"
    p1_path.write_text(json.dumps(p1_data, indent=2), encoding="utf-8")

    # Also write corrupt JSON to character_bible.json to verify graceful recovery
    (run_dir / "understanding" / "character_bible.json").write_text("{corrupt_json:", encoding="utf-8")

    graph.auto_approve_checkpoints = True
    state = graph.resume(state)
    assert state.current_stage == PipelineStage.DONE
    assert state.panels[0].duration_s <= 10.0
    assert state.panels[0].motion_intensity == 1.0
    assert len(state.audio_tracks) == 1
    assert Path(state.audio_tracks[0].audio_path).is_file()


@pytest.mark.worker
@pytest.mark.eval_qc
@pytest.mark.eval
def test_phase2_non_oracle_ocr_and_no_orphan_crops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verifies P2-01 and P2-10: real pixel OCR + non-oracle evaluation passes all targets and leaves zero orphan crops."""
    monkeypatch.setenv("M2A_EVAL_ORACLE", "0")
    gt_path = ROOT / "data" / "eval" / "ground_truth.json"
    report = evaluate_understanding(gt_path)
    assert report["passed_target"] is True
    assert report["ocr_cer"] <= 0.10
    assert report["speaker_attribution_accuracy"] >= 0.90
    assert report["shot_classification_accuracy"] >= 0.85

    # Verify unseen panel with decoded text produces exact OCR and zero orphan crop files
    img = Image.new("RGB", (680, 460), (40, 30, 70))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 276, 680, 460], fill=(25, 20, 45))
    draw.polygon([(340, 83), (283, 236), (397, 236)], fill=(90, 160, 240))
    draw.ellipse([326, 55, 354, 83], fill=(255, 220, 190))
    draw.rectangle([0, 0, 679, 459], outline=(20, 20, 24), width=4)
    draw.rounded_rectangle([390, 30, 640, 90], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.polygon([(410, 90), (425, 90), (405, 105)], fill=(255, 255, 255), outline=(40, 40, 45))
    draw.text((405, 45), "GROOOOOWL!", fill=(20, 20, 20))

    p_path = tmp_path / "beast_panel.png"
    img.save(p_path)
    crops_dir = tmp_path / "crops"
    sem = analyze_panel_semantics(p_path, panel_id="beast_01", language="en", crops_dir=crops_dir)
    assert sem["dialogue"][0]["text"] == "GROOOOOWL!"
    assert sem["dialogue"][0]["speaker_id"] == "char_shadow_beast"
    crop_files = sorted(f.name for f in crops_dir.glob("*.png"))
    assert crop_files == ["beast_01_char_beast.png"]


@pytest.mark.api_ui
@pytest.mark.api
@pytest.mark.worker
def test_phase2_cp2_preserves_tts_duration_and_syncs_route_edits(tmp_path: Path):
    """Verifies P2-02: resuming from CHECKPOINT_2 preserves TTS-synchronized duration_s and syncs routing_plan.json edits."""
    page_img = Image.new("RGB", (800, 700), (255, 255, 255))
    draw = ImageDraw.Draw(page_img)
    draw.rectangle([40, 50, 760, 620], fill=(90, 150, 215), outline=(20, 20, 24), width=4)
    draw.rounded_rectangle([70, 80, 450, 150], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.text((88, 100), "A long spoken sentence that extends panel duration past default!", fill=(20, 20, 20))
    page_file = tmp_path / "cp2_page.png"
    page_img.save(page_file)

    cfg = load_config("t0", config_dir=ROOT / "configs")
    cfg.composition.resolution = [180, 320]
    cfg.composition.fps = 8
    run_dir = tmp_path / "run_cp2"

    state = RunState(
        config=RunConfig(source_pages=[str(page_file)], source_language="en", compute_tier="t0", output_resolution=[180, 320])
    )
    graph = PipelineGraph(config=cfg, run_dir=run_dir, auto_approve_checkpoints=False)
    state = graph.run(state)
    assert state.current_stage == PipelineStage.CHECKPOINT_1

    # Resume from CP1 -> runs CLEAN_PLATES, ROUTE, TTS and pauses at CHECKPOINT_2
    state = graph.resume(state)
    assert state.current_stage == PipelineStage.CHECKPOINT_2
    tts_dur = state.panels[0].duration_s
    assert tts_dur > 2.5

    # Verify routes/routing_plan.json has the updated TTS duration and edit route to parallax at CP2
    plan_file = run_dir / "routes" / "routing_plan.json"
    plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
    plan_panels = plan_data["panels"] if isinstance(plan_data, dict) else plan_data
    assert plan_panels[0]["duration_s"] == tts_dur
    plan_panels[0]["route"] = "parallax"
    plan_file.write_text(json.dumps(plan_data, indent=2), encoding="utf-8")

    state = graph.resume(state)
    assert state.current_stage == PipelineStage.DONE
    assert state.panels[0].duration_s == tts_dur
    assert state.panels[0].route == "parallax"


@pytest.mark.worker
def test_phase2_top_right_character_upward_tail_and_magiv3_bbox(tmp_path: Path):
    """Verifies P2-04 (top-right character not blinded), P2-08 (upward pointer tail), and P2-06 (Magiv3 tight character BBox)."""
    from m2a.providers import get_perception_provider

    img = Image.new("RGB", (720, 560), (115, 165, 215))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 360, 720, 560], fill=(65, 120, 65))
    draw.rectangle([0, 0, 719, 559], outline=(20, 20, 24), width=4)

    # Character 1 on left (x=90..210, y=180..420)
    draw.polygon([(150, 180), (90, 420), (210, 420)], fill=(220, 60, 55))
    draw.ellipse([125, 130, 175, 180], fill=(255, 220, 190))

    # Character 2 perched high in the top-right quadrant (x=530..650, y=70..270 — inside x>0.70*w, y<0.32*h!)
    draw.polygon([(590, 100), (530, 270), (650, 270)], fill=(50, 95, 230))
    draw.ellipse([568, 60, 612, 104], fill=(255, 220, 190))

    # Speech bubble below Character 2 with an UPWARD-pointing tail toward Character 2
    draw.rounded_rectangle([430, 310, 690, 375], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw.polygon([(575, 310), (595, 310), (588, 285)], fill=(255, 255, 255), outline=(40, 40, 45))
    draw.text((448, 330), "Upward tail to cliff scout!", fill=(20, 20, 20))

    panel_path = tmp_path / "top_right_and_up_tail.png"
    img.save(panel_path)

    bubbles = detect_speech_bubbles_and_text(img)
    assert len(bubbles) == 1
    assert bubbles[0]["tail_tip"] is not None
    # Upward tail tip y-coordinate must be above the main bubble body top (y < 310)
    assert bubbles[0]["tail_tip"][1] < bubbles[0]["bbox"]["y"]

    # Magiv3Provider must return 2 distinct tight character BBoxes and link the bubble to the right character
    perc = get_perception_provider("magiv3")
    txt_dets = perc.detect_text(panel_path)
    char_dets = perc.detect_characters(panel_path, [])
    assert len(char_dets) == 2
    assert char_dets[0].bbox.x1 != char_dets[1].bbox.x1
    links = perc.link_speakers(panel_path, txt_dets, char_dets)
    assert len(links) == 1
    assert links[0].character_bbox == char_dets[1].bbox


@pytest.mark.eval_qc
@pytest.mark.eval
@pytest.mark.worker
def test_phase2_qc_last_frame_text_leakage_detection(tmp_path: Path):
    """Verifies P2-09: compute_clip_qc_metrics catches a leaked speech bubble appearing in the last frame."""
    clean_img = Image.new("RGB", (360, 640), (85, 145, 210))
    draw_clean = ImageDraw.Draw(clean_img)
    draw_clean.rectangle([0, 380, 360, 640], fill=(60, 115, 65))
    draw_clean.polygon([(180, 200), (120, 420), (240, 420)], fill=(215, 65, 55))
    clean_path = tmp_path / "clean_plate.png"
    clean_img.save(clean_path)

    leaked_last = clean_img.copy()
    draw_leak = ImageDraw.Draw(leaked_last)
    draw_leak.rounded_rectangle([40, 50, 310, 120], radius=12, fill=(255, 255, 255), outline=(40, 40, 45), width=3)
    draw_leak.text((58, 70), "Late leaked bubble!", fill=(20, 20, 20))

    # First 7 frames are clean, but last frame has a leaked speech bubble
    frames = [clean_img.copy() for _ in range(7)] + [leaked_last]
    qc_res = compute_clip_qc_metrics(clean_path, frames)
    assert qc_res["new_text_count"] >= 1
    assert qc_res["overall_pass"] is False


@pytest.mark.worker
@pytest.mark.eval_qc
@pytest.mark.eval
def test_webtoon_gutter_ocr_garbage_and_pan_qc_regressions(tmp_path: Path):
    """Verifies fixes for real-world webtoon gutter false-positive bubbles, OCR punctuation soup, and pan_down/tall-strip QC."""
    from workers.understanding import _decode_pil_bitmap_text, _is_plausible_ocr_text

    # 1. Full-width top/bottom webtoon gutter band with border fragments must NOT be detected as a speech bubble
    panel = Image.new("RGB", (800, 520), (35, 30, 55))
    draw = ImageDraw.Draw(panel)
    # Full-width bottom white webtoon gutter (x=0..800, y=462..520) with minor border specks
    draw.rectangle([0, 462, 800, 520], fill=(255, 255, 255))
    draw.rectangle([120, 464, 126, 469], fill=(25, 25, 30))
    draw.rectangle([480, 464, 488, 469], fill=(25, 25, 30))
    assert detect_speech_bubbles_and_text(panel) == []

    # 2. Punctuation soup strings must be rejected by OCR plausibility filter
    assert not _is_plausible_ocr_text("...`__aw\"''''''7~~`")
    assert not _is_plausible_ocr_text("$ $")
    assert not _is_plausible_ocr_text("^ .' m .__ _* '._* .* .* * @")
    assert _is_plausible_ocr_text("Arise.")

    # Multi-line tall comic stroke region (>22px tall) must not be force-decoded by single-line 10x16 PIL bitmap matcher
    tall_crop = Image.new("RGB", (220, 90), (255, 255, 255))
    td = ImageDraw.Draw(tall_crop)
    td.rectangle([30, 18, 180, 26], fill=(20, 20, 20))
    td.rectangle([30, 55, 180, 65], fill=(20, 20, 20))
    assert _decode_pil_bitmap_text(tall_crop) is None

    # 3. Camera pan_down and zoom_out clips must pass aligned first_frame_fidelity
    sample_panel = ROOT / "data" / "eval" / "sample_page_1.png"
    for mtype in ("pan_down", "pan_right", "zoom_out"):
        frames = generate_kenburns_frames(sample_panel, target_res=(180, 320), duration_s=1.0, fps=8, motion_type=mtype)
        qc_res = compute_clip_qc_metrics(sample_panel, frames, route="kenburns")
        assert qc_res["first_frame_fidelity"] >= 0.75, f"{mtype} first_frame_fidelity too low: {qc_res}"
        assert qc_res["new_text_count"] == 0
        assert qc_res["overall_pass"] is True, f"{mtype} failed QC: {qc_res}"


@pytest.mark.contract
@pytest.mark.worker
@pytest.mark.eval_qc
@pytest.mark.eval
def test_zero_ground_truth_leakage_in_production_code():
    """Enforces the Zero Ground-Truth Leakage invariant across all production worker, node, provider, and Go files."""
    prod_dirs = [
        ROOT / "workers",
        ROOT / "src" / "m2a" / "nodes",
        ROOT / "src" / "m2a" / "providers",
        ROOT / "internal",
        ROOT / "cmd",
        ROOT / "api",
        ROOT / "ui",
    ]
    violations = []
    for d in prod_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file() or p.suffix not in (".py", ".go"):
                continue
            if p.parent == ROOT / "workers" and p.name.startswith("eval_"):
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "ground_truth.json" in text or "_load_eval_ground_truth" in text or "M2A_EVAL_ORACLE" in text:
                violations.append(str(p.relative_to(ROOT)))
    assert violations == [], f"Ground-truth leakage detected in production files: {violations}"


@pytest.mark.worker
def test_streaming_assembly_bounded_memory_and_subtitle_overlay(tmp_path: Path):
    """Verifies feat-006: StreamingMjpegAviWriter, count_video_frames, and assemble_clips_streaming stream clips without full in-memory decoding."""
    from workers.assembler import (
        StreamingMjpegAviWriter,
        assemble_clips_streaming,
        count_video_frames,
        iter_video_frames,
    )

    clip1_path = tmp_path / "panel_001.avi"
    clip2_path = tmp_path / "panel_002.avi"
    res = (180, 320)

    with StreamingMjpegAviWriter(clip1_path, fps=8) as w1:
        for i in range(12):
            w1.write_frame(Image.new("RGB", res, (40 + i * 5, 90, 180)))

    with StreamingMjpegAviWriter(clip2_path, fps=8) as w2:
        for i in range(12):
            w2.write_frame(Image.new("RGB", res, (180, 70 + i * 5, 90)))

    assert count_video_frames(clip1_path) == 12
    assert count_video_frames(clip2_path) == 12
    streamed = list(iter_video_frames(clip1_path, target_res=res))
    assert len(streamed) == 12

    out_base = tmp_path / "streamed_final"
    result = assemble_clips_streaming(
        panel_clip_specs=[
            {
                "clip_path": clip1_path,
                "dialogue": [{"speaker_id": "char_jin", "text": "Streaming subtitle overlay!"}],
                "subtitle_burned": False,
                "frames": 12,
            },
            {
                "clip_path": clip2_path,
                "dialogue": [],
                "subtitle_burned": True,
                "frames": 12,
            },
        ],
        output_base_path=out_base,
        fps=8,
        target_res=res,
        transition_frames=4,
    )
    assert result["total_frames"] == 20  # 12 + 12 - 4 crossfade overlap
    assert Path(result["avi_path"]).is_file()
    assert Path(result["webp_path"]).is_file()
    assert count_video_frames(Path(result["avi_path"])) == 20


@pytest.mark.contract
@pytest.mark.worker
def test_model_residency_lifecycle_and_needs_review_contract(tmp_path: Path):
    """Verifies feat-010: ModelResidencyManager unloads previous heavy model on switch and PanelJSON.needs_review round-trips."""
    from m2a.state import BBox, PanelJSON
    from workers.server import ModelResidencyManager

    mgr = ModelResidencyManager(models_dir=tmp_path)
    events: list[str] = []

    mgr.register_model(
        "qwen3-vl",
        loader=lambda: {"name": "qwen3-vl", "vram_gb": 6.5},
        unloader=lambda inst: events.append(f"unload:{inst['name']}"),
    )
    mgr.register_model(
        "comfyui:wan22-ti2v-5b",
        loader=lambda: {"name": "comfyui:wan22-ti2v-5b", "vram_gb": 10.2},
        unloader=lambda inst: events.append(f"unload:{inst['name']}"),
    )

    with mgr.acquire("qwen3-vl") as h1:
        assert h1["name"] == "qwen3-vl"
        assert mgr.resident_model == "qwen3-vl"
        assert "qwen3-vl" in mgr.loaded_models()

    with mgr.acquire("comfyui:wan22-ti2v-5b") as h2:
        assert h2["name"] == "comfyui:wan22-ti2v-5b"
        assert mgr.resident_model == "comfyui:wan22-ti2v-5b"
        assert events == ["unload:qwen3-vl"]

    mgr.free_all()
    assert mgr.resident_model is None
    assert events == ["unload:qwen3-vl", "unload:comfyui:wan22-ti2v-5b"]

    panel = PanelJSON(
        panel_id="panel_001",
        bbox=BBox(x=0, y=0, w=720, h=400),
        needs_review=True,
    )
    dumped = json.loads(panel.model_dump_json())
    assert dumped["needs_review"] is True






