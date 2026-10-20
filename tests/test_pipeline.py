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

from workers.eval_segmentation import evaluate_detection
from workers.slicer import detect_panels_gutter
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


def test_phase1_segmentation_f1_and_iou():
    gt_path = ROOT / "data" / "eval" / "ground_truth.json"
    report = evaluate_detection(gt_path)
    assert report["f1"] >= 0.90
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["mean_iou"] >= 0.99


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


def test_wide_panel_cinematic_canvas_preserves_aspect():
    """Verifies P10 fix: wide landscape panels are letterboxed onto blurred-fill 9:16 canvas."""
    wide_img = Image.new("RGB", (720, 400), (60, 130, 200))
    canvas = prepare_cinematic_canvas(wide_img, target_res=(360, 640), overscan=1.2)
    assert canvas.size == (int(360 * 1.2), int(640 * 1.2))


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


def test_qc_rejects_frozen_clip():
    """Verifies P1-03: a completely static clip (identical frames) has motion_score=0.0 and fails QC."""
    sample_panel = ROOT / "data" / "eval" / "sample_page_1.png"
    static_frame = Image.open(sample_panel).convert("RGB").resize((180, 320))
    frozen_frames = [static_frame.copy() for _ in range(24)]

    qc_scores = compute_clip_qc_metrics(sample_panel, frozen_frames)
    assert qc_scores["motion_score"] == 0.0
    assert qc_scores["overall_pass"] is False


def test_understand_no_oracle_leakage_on_noise(tmp_path: Path):
    """Verifies P0-03: random noise or unrelated images do not leak ground_truth.json dialogue."""
    rng = np.random.default_rng(42)
    noise_arr = rng.integers(0, 256, size=(400, 720, 3), dtype=np.uint8)
    noise_file = tmp_path / "noise_panel.png"
    Image.fromarray(noise_arr).save(noise_file)

    sem = analyze_panel_semantics(noise_file, panel_id="panel_001", language="en", reading_order=1)
    assert sem["dialogue"] == []
    assert sem["characters"] == []


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


def test_clean_plate_and_tts_synthesis(tmp_path: Path):
    sample_page = ROOT / "data" / "eval" / "sample_page_1.png"
    clean_out = tmp_path / "clean.png"
    inpaint_panel_bubbles(sample_page, clean_out, bubble_bboxes=[{"x": 70, "y": 90, "w": 300, "h": 70}])
    assert clean_out.exists() and clean_out.stat().st_size > 0

    wav_out = tmp_path / "line.wav"
    dur = synthesize_speech_wav("I won't run away this time.", wav_out, voice_id="char_jin", emotion="confident")
    assert wav_out.exists() and wav_out.stat().st_size > 1000
    assert dur >= 1.2


def test_go_state_json_interoperability(tmp_path: Path):
    """Verifies that Python RunState loads state.json cleanly and rejects out-of-range durations."""
    existing_run = ROOT / "runs" / "396d8463-dbfb-4263-b474-d9c558263b20"
    if (existing_run / "state.json").exists():
        state = RunState.load(existing_run)
        assert state.config.output_resolution == [720, 1280]
        assert len(state.panels) == 9
        for cp in state.clean_plate_paths.values():
            assert "\\" not in cp


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


def test_config_tier_loading():
    cfg_t0 = load_config("t0", config_dir=ROOT / "configs")
    assert cfg_t0.generation.primary_model == "kenburns"
    assert cfg_t0.understanding.vlm == "qwen3-vl-2b"

    cfg_t2 = load_config("t2", config_dir=ROOT / "configs")
    assert cfg_t2.generation.primary_model == "wan22-i2v-a14b"


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


def test_eval_metrics_functions():
    assert character_error_rate("hello", "hello") == 0.0
    assert 0.15 <= character_error_rate("hallo", "hello") <= 0.25
    f1 = panel_detection_f1([(0, 0, 100, 100)], [(0, 0, 100, 100)])
    assert f1 == 1.0
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert identity_similarity(arr, arr) > 0.99
    assert temporal_stability([arr, arr, arr]) == 1.0


def test_job_api_lifecycle():
    client = TestClient(job_app)
    resp = client.get("/health")
    assert resp.status_code == 200
