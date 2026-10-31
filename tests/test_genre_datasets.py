"""Verification tests for the Unified 18-Genre, Panel-Topology & Storytelling Datasets (`tests/test_genre_datasets.py`)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.training.build_unified_genre_datasets import build_all_datasets

ROOT = Path(__file__).resolve().parent.parent
THEMES_FILE = ROOT / "configs" / "themes.yaml"
DATASETS_DIR = ROOT / "data" / "datasets"

EXPECTED_DEMOGRAPHICS = {"shonen", "shojo", "seinen", "josei", "kodomomuke"}
EXPECTED_CORE_GENRES = {
    "action_adventure",
    "comedy_slice_of_life",
    "romance_drama",
    "scifi_fantasy",
    "horror_thriller",
    "sports",
}
EXPECTED_SUBGENRES = {
    "isekai",
    "mecha",
    "mahou_shoujo",
    "idol_music",
    "harem_reverse_harem",
    "ecchi",
    "yaoi_yuri",
}
ALL_18_GENRES = EXPECTED_DEMOGRAPHICS | EXPECTED_CORE_GENRES | EXPECTED_SUBGENRES


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_themes_yaml_covers_all_18_genres_and_subtypes():
    """Verifies configs/themes.yaml defines all 5 demographics, 6 core genres, and 7 subgenres with required specs."""
    assert THEMES_FILE.is_file()
    cfg = yaml.safe_load(THEMES_FILE.read_text(encoding="utf-8"))
    genres = cfg["genres"]
    assert set(genres.keys()) == ALL_18_GENRES

    demographics = {k for k, v in genres.items() if v["category"] == "demographic"}
    core_genres = {k for k, v in genres.items() if v["category"] == "core_genre"}
    subgenres = {k for k, v in genres.items() if v["category"] == "subgenre"}
    assert demographics == EXPECTED_DEMOGRAPHICS
    assert core_genres == EXPECTED_CORE_GENRES
    assert subgenres == EXPECTED_SUBGENRES

    # Verify Isekai 0.7 prior, Mecha (real_robot & super_robot), and Yaoi/Yuri subtypes
    assert genres["isekai"]["default_prior_weight"] == 0.70
    assert set(genres["mecha"]["subtypes"].keys()) == {"real_robot", "super_robot"}
    assert set(genres["yaoi_yuri"]["subtypes"].keys()) == {"yaoi_bl", "yuri_gl"}

    # Verify panel layout topologies and storytelling arc modes
    assert len(cfg["panel_layout_topologies"]) >= 10
    assert len(cfg["storytelling_arc_modes"]) >= 8
    fallback_modes = [k for k, v in cfg["storytelling_arc_modes"].items() if v.get("requires_fallback")]
    assert len(fallback_modes) >= 4


def test_five_purpose_specific_datasets_and_manifest_integrity():
    """Verifies all 5 purpose-specific JSONL datasets cover all 18 genres and satisfy implementation_plan.md invariants."""
    manifest = build_all_datasets(themes_path=THEMES_FILE, out_dir=DATASETS_DIR, fetch_hf=False)
    assert manifest["total_genres"] == 18

    ds_map = manifest["purpose_specific_datasets"]
    expected_keys = {
        "vlm_story_director",
        "perception_ocr_speaker",
        "bg_3d_stage",
        "wan22_motion_lora",
        "audio_tts_lufs",
    }
    assert set(ds_map.keys()) == expected_keys

    for key, meta in ds_map.items():
        fpath = ROOT / meta["path"]
        assert fpath.is_file(), f"Missing dataset file for {key}: {fpath}"
        assert _sha256(fpath) == meta["sha256"]

        rows = [json.loads(line) for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == meta["records_count"]
        genres_in_ds = {r["genre_id"] for r in rows}
        assert genres_in_ds == ALL_18_GENRES, f"Dataset {key} missing genres: {ALL_18_GENRES - genres_in_ds}"

    # 1. Verify VLM Story Director incomplete-story fallback records
    vlm_rows = [
        json.loads(line)
        for line in (DATASETS_DIR / "vlm_story_director" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(vlm_rows) == 432
    incomplete_rows = [r for r in vlm_rows if r["incomplete_story_fallback"]]
    assert len(incomplete_rows) == 216
    for r in incomplete_rows:
        fb = r["structured_label"]["story_expansion_fallback"]
        assert fb["incomplete_story_input"] is True
        assert fb["prologue_hook"] != ""
        assert fb["climax_outro"] != ""
        assert fb["bridge_panel_generator"]["base_model"] == "OnomaAIResearch/Illustrious-xl-early-release-v0"

    # 2. Verify 3D Stage depth ordering (near < character < midground < far)
    bg_rows = [
        json.loads(line)
        for line in (DATASETS_DIR / "bg_3d_stage" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for r in bg_rows:
        planes = r["depth_mesh_3d_stage"]["depth_planes"]
        assert (
            planes["near_foreground_vfx_z"]
            < planes["character_billboard_z"]
            < planes["midground_architecture_z"]
            < planes["far_skybox_horizon_z"]
        )
        assert r["character_cutout_extraction"]["primary_model"] == "ZhengPeng7/BiRefNet_anime"

    # 3. Verify Audio LUFS & Ducking math (bgm_ducked == bgm_active + ducking_db, SNR >= 10 dB, drift <= 1 ms)
    aud_rows = [
        json.loads(line)
        for line in (DATASETS_DIR / "audio_tts_lufs" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(aud_rows) == 18
    for r in aud_rows:
        lufs = r["ebu_r128_loudness_and_ducking"]
        assert abs((lufs["bgm_active_lufs"] + lufs["ducking_attenuation_db"]) - lufs["bgm_ducked_lufs"]) < 1e-4
        assert lufs["min_dialogue_to_bgm_snr_db"] >= 10.0
        assert (lufs["dialogue_target_lufs"] - lufs["bgm_ducked_lufs"]) >= 10.0
        assert r["transition_sync_spec"]["max_sfx_alignment_drift_ms"] <= 1.0


def test_training_harnesses_export_valid_configs(tmp_path: Path):
    """Verifies the 3 fine-tuning scripts validate their respective datasets and export complete configs."""
    from scripts.training.finetune_bg_illustrious_lora import validate_and_export_bg_lora_config
    from scripts.training.finetune_vlm_story_director import validate_and_export_vlm_config
    from scripts.training.finetune_wan22_motion_lora import validate_and_export_wan22_lora_config

    vlm_cfg = validate_and_export_vlm_config(out_dir=tmp_path / "vlm")
    assert vlm_cfg["dataset_stats"]["genres_count"] == 18
    assert vlm_cfg["dataset_stats"]["total_records"] == 432
    assert (tmp_path / "vlm" / "qwen25_vl_qlora_config.json").is_file()

    bg_cfg = validate_and_export_bg_lora_config(out_dir=tmp_path / "bg")
    assert bg_cfg["dataset_stats"]["genres_count"] == 18
    assert bg_cfg["dataset_stats"]["total_records"] == 54
    assert (tmp_path / "bg" / "illustrious_xl_bg_lora_config.json").is_file()

    wan_cfg = validate_and_export_wan22_lora_config(out_dir=tmp_path / "wan")
    assert wan_cfg["dataset_stats"]["genres_count"] == 18
    assert wan_cfg["dataset_stats"]["total_records"] == 54
    assert wan_cfg["trigger_token"] == "m2a_anime_motion"
    assert (tmp_path / "wan" / "musubi_wan22_motion_lora_config.json").is_file()

