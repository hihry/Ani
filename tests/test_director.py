"""Comprehensive tests for Phases A, B, C, and D:
  - Phase A: Story Director, 18-Genre Theme Engine, 10 Layout Topologies, 8 Arc Modes, Incomplete-Story Fallback
  - Phase B: Expressive Speech Synthesis, Genre BGM, Transition SFX, and EBU R128 LUFS Ducking
  - Phase C: 2D->3D Multi-Plane Stage, Character Cutout, Conditional 2D Background, and 10 Transition Styles
  - Phase D: Custom Dataset SHA-256 Verification & LoRA/QLoRA Fine-Tuning Config Generation
"""
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from workers.story_director import (
    ALL_18_GENRES,
    ARC_MODES,
    LAYOUT_TOPOLOGIES,
    build_story_director_plan,
    classify_panel_layout_topology,
    load_dataset_index,
    load_themes_config,
    resolve_genre_id,
    score_narrative_completeness,
)
from workers.audio_director import (
    generate_genre_bgm_wave,
    generate_sfx_wave,
    measure_lufs,
    mix_cinematic_soundtrack,
    synthesize_expressive_speech_wav,
)
from workers.stage_3d import (
    estimate_depth_map,
    extract_character_cutout,
    generate_3d_stage_parallax_frames,
    generate_contextual_2d_background,
)
from workers.assembler import (
    compute_clip_qc_metrics,
    render_transition_frames,
)
from workers.pipeline_cli import assemble_from_run_state
from scripts.training.finetune_all import generate_training_configs, verify_datasets


def test_phase_d_dataset_manifest_and_training_configs(tmp_path: Path) -> None:
    """Verifies all 5 purpose-specific datasets in data/datasets/ pass SHA-256 and 18-genre coverage checks."""
    ver = verify_datasets(ROOT / "data" / "datasets")
    assert ver["all_passed"] is True
    assert ver["genres_count"] == 18
    assert ver["topologies_count"] == 10
    assert ver["arc_modes_count"] == 8
    assert ver["total_records"] == 612
    assert ver["splits"]["vlm_story_director"]["records"] == 432
    assert ver["splits"]["perception_ocr_speaker"]["records"] == 54
    assert ver["splits"]["bg_3d_stage"]["records"] == 54
    assert ver["splits"]["wan22_motion_lora"]["records"] == 54
    assert ver["splits"]["audio_tts_lufs"]["records"] == 18

    cfg_res = generate_training_configs(tmp_path / "train_specs", compute_target="local_rtx4060_8gb")
    for key in ("vlm_story_director", "bg_3d_stage", "wan22_motion_lora"):
        cfg_file = Path(cfg_res["configs"][key])
        assert cfg_file.is_file()
        parsed = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert "base_model" in parsed


def test_phase_a_all_18_genres_and_aliases() -> None:
    """Verifies themes.yaml and dataset index cover all 18 canonical genres plus user aliases."""
    themes = load_themes_config()
    ds_idx = load_dataset_index()
    assert len(ALL_18_GENRES) == 18
    for g in ALL_18_GENRES:
        assert g in themes.get("genres", {}), f"Missing genre {g} in configs/themes.yaml"
        assert g in ds_idx["by_genre"], f"Missing genre {g} in data/datasets/vlm_story_director"
        assert g in ds_idx["audio_by_genre"], f"Missing genre {g} in data/datasets/audio_tts_lufs"

    assert resolve_genre_id("violence") == "seinen"
    assert resolve_genre_id("romance") == "romance_drama"
    assert resolve_genre_id("horror") == "horror_thriller"
    assert resolve_genre_id("epic_awakening") == "isekai"
    assert resolve_genre_id("comedy") == "comedy_slice_of_life"


def test_phase_a_story_director_complete_and_incomplete_fallback(tmp_path: Path) -> None:
    """Verifies Story Director generates complete 4-beat arc and activates 2-tier fallback on incomplete input."""
    # Case 1: Incomplete 1-panel input -> triggers story_expansion_fallback (narration bridge + synthesized bridge panel)
    incomplete_panels = [
        {
            "panel_id": "panel_001",
            "bbox": {"x": 0, "y": 0, "w": 640, "h": 1100},
            "shot": "close_up",
            "emotion": "intense",
            "motion_score": 0.72,
            "dialogue": [],
            "sfx": [],
        }
    ]
    plan_inc = build_story_director_plan(
        panels=incomplete_panels,
        genre_hint="violence",
        run_dir=tmp_path / "run_inc",
        output_path=tmp_path / "run_inc" / "story" / "story_plan.json",
    )
    assert plan_inc["genre_id"] == "seinen"
    assert plan_inc["is_incomplete_story"] is True
    assert plan_inc["fallback_plan"]["fallback_active"] is True
    assert plan_inc["fallback_plan"]["tier1_narration_bridge"] is True
    assert plan_inc["fallback_plan"]["tier2_bridge_panels_count"] >= 1
    bridge_path = Path(plan_inc["fallback_plan"]["synthesized_bridge_panels"][0]["image_path"])
    assert bridge_path.is_file()
    assert len(plan_inc["beats"]) >= 2

    # Case 2: Complete 4-panel romance sequence -> no bridge panel needed
    complete_panels = [
        {
            "panel_id": f"panel_{i:03d}",
            "bbox": {"x": 0, "y": (i - 1) * 500, "w": 720, "h": 480},
            "shot": "wide" if i == 1 else ("close_up" if i == 3 else "medium"),
            "emotion": "romantic",
            "motion_score": 0.35,
            "dialogue": [{"speaker_id": "char_a", "text": f"Under the cherry blossoms, line {i}."}],
            "sfx": ["Doki"],
        }
        for i in range(1, 5)
    ]
    plan_comp = build_story_director_plan(
        panels=complete_panels,
        genre_hint="romance",
        run_dir=tmp_path / "run_comp",
    )
    assert plan_comp["genre_id"] == "romance_drama"
    assert plan_comp["is_incomplete_story"] is False
    assert plan_comp["completeness_score"] >= 0.62
    assert plan_comp["theme_profile"]["transition_type"] == "dreamy_dissolve"


def test_phase_b_expressive_tts_and_lufs_ducking_verification(tmp_path: Path) -> None:
    """Verifies multi-formant TTS, genre BGM, transition SFX drift <= 1.0ms, and EBU R128 LUFS sidechain ducking."""
    wav1 = tmp_path / "line_01.wav"
    wav2 = tmp_path / "line_02.wav"
    d1 = synthesize_expressive_speech_wav(
        "The gates have opened. Every hunter must stand their ground now!",
        wav1,
        voice_id="am_adam",
        emotion="fierce",
        genre_id="shonen",
    )
    d2 = synthesize_expressive_speech_wav(
        "I will not retreat a single step.",
        wav2,
        voice_id="af_bella",
        emotion="determined",
        genre_id="shonen",
    )
    assert wav1.is_file() and d1 >= 1.2
    assert wav2.is_file() and d2 >= 1.2

    master_wav = tmp_path / "master_cinematic.wav"
    metrics_json = tmp_path / "audio_metrics.json"
    cuts = [2.5, 5.0, 7.5]

    metrics = mix_cinematic_soundtrack(
        dialogue_events=[(0.4, wav1), (4.2, wav2)],
        total_duration_s=9.0,
        output_wav_path=master_wav,
        genre_id="shonen",
        cut_timestamps_s=cuts,
        metrics_output_path=metrics_json,
    )
    assert master_wav.is_file()
    assert metrics_json.is_file()
    assert metrics["snr_pass"] is True
    assert metrics["measured_snr_margin_db"] >= 6.0
    assert metrics["ducking_applied_db"] >= 8.0
    assert metrics["max_sfx_alignment_drift_ms"] <= 1.0
    assert metrics["clipping_samples"] == 0
    assert abs(metrics["dialogue_measured_lufs"] - metrics["dialogue_target_lufs"]) <= 1.5


def test_phase_c_3d_stage_parallax_and_10_transitions(tmp_path: Path) -> None:
    """Verifies 2D->3D multi-plane stage, flat-background replacement, and all 10 transition styles."""
    # Create a panel with a flat uniform background + character in center
    flat_panel = tmp_path / "flat_panel.png"
    img = Image.new("RGB", (540, 960), (185, 198, 215))
    draw = ImageDraw.Draw(img)
    draw.polygon([(270, 260), (160, 780), (380, 780)], fill=(45, 55, 95), outline=(15, 18, 28))
    draw.ellipse([220, 160, 320, 265], fill=(245, 215, 195), outline=(20, 20, 25))
    img.save(flat_panel)

    rgba_char, mask_l, bg_flatness_ratio = extract_character_cutout(img)
    assert rgba_char.mode == "RGBA"
    assert mask_l.mode == "L"
    assert bg_flatness_ratio >= 0.45

    bg_2d = generate_contextual_2d_background((540, 960), genre_id="isekai", is_flat_source=True, source_img=img)
    assert bg_2d.size == (540, 960)
    depth = estimate_depth_map(img, mask_l)
    assert depth.shape == (960, 540)

    frames = generate_3d_stage_parallax_frames(
        flat_panel,
        target_res=(360, 640),
        duration_s=1.0,
        fps=12,
        genre_id="isekai",
        motion_primitive="dolly_zoom_parallax",
    )
    assert len(frames) == 12
    assert frames[0].size == (360, 640)
    qc = compute_clip_qc_metrics(flat_panel, frames, route="parallax")
    assert qc["overall_pass"] is True, f"QC failed: {qc}"

    # Test all 10 transition styles
    clip_b = generate_3d_stage_parallax_frames(
        flat_panel,
        target_res=(360, 640),
        duration_s=1.0,
        fps=12,
        genre_id="seinen",
        motion_primitive="impact_push_in",
    )
    for ttype in (
        "crossfade",
        "impact_flash",
        "whip_pan",
        "dreamy_dissolve",
        "fade_through_black",
        "light_burst_flash",
        "glitch_cut",
        "wipe_cut",
        "hud_scan_wipe",
        "petal_dissolve",
    ):
        trans = render_transition_frames(frames[-6:], clip_b[:6], transition_type=ttype)
        assert len(trans) == 6
        assert trans[0].size == (360, 640)


def test_end_to_end_dramatic_assemble_from_run_state(tmp_path: Path) -> None:
    """Verifies assemble_from_run_state produces story_plan.json, audio_metrics.json, and voiced/scored video."""
    run_dir = tmp_path / "dramatic_run"
    (run_dir / "panels").mkdir(parents=True)
    (run_dir / "audio").mkdir(parents=True)

    sample_page = ROOT / "data" / "eval" / "sample_page_1.png"
    src_im = Image.open(sample_page).convert("RGB")
    for i in range(1, 4):
        p_path = run_dir / "panels" / f"panel_{i:03d}.png"
        src_im.save(p_path)

    wav_p = run_dir / "audio" / "panel_001_line_00.wav"
    dur = synthesize_expressive_speech_wav("This dungeon hides a second awakening!", wav_p, genre_id="isekai")

    state_doc = {
        "config": {
            "run_id": "dramatic_01",
            "source_language": "en",
            "compute_tier": "t1",
            "genre_theme": "epic_awakening",
            "output_resolution": [180, 320],
        },
        "current_stage": "ASSEMBLE",
        "panels": [
            {
                "panel_id": "panel_001",
                "bbox": {"x": 0, "y": 0, "w": 360, "h": 480},
                "shot": "wide",
                "route": "parallax",
                "duration_s": 1.5,
                "dialogue": [{"speaker_id": "char_jin", "text": "This dungeon hides a second awakening!", "emotion": "fierce"}],
                "sfx": ["Rumble"],
            },
            {
                "panel_id": "panel_002",
                "bbox": {"x": 0, "y": 480, "w": 360, "h": 480},
                "shot": "close_up",
                "route": "parallax",
                "duration_s": 1.5,
                "dialogue": [],
                "sfx": ["Slash"],
            },
            {
                "panel_id": "panel_003",
                "bbox": {"x": 0, "y": 960, "w": 360, "h": 480},
                "shot": "medium",
                "route": "kenburns",
                "duration_s": 1.5,
                "dialogue": [],
                "sfx": [],
            },
        ],
        "clean_plate_paths": {},
        "clip_candidates": {},
        "audio_tracks": [
            {
                "panel_id": "panel_001",
                "line_idx": 0,
                "speaker_id": "char_jin",
                "audio_path": str(wav_p.as_posix()),
                "duration_s": dur,
            }
        ],
    }
    (run_dir / "state.json").write_text(json.dumps(state_doc), encoding="utf-8")

    res = assemble_from_run_state(run_dir, fps=12, target_res=(180, 320))
    assert Path(res["story_plan_path"]).is_file()
    assert Path(res["audio_metrics_path"]).is_file()
    assert res["genre_id"] == "isekai"
    assert res["transition_type"] == "light_burst_flash"
    assert res["audio_metrics"]["snr_pass"] is True
