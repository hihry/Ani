"""Unified YAML configuration loader for Python matching Go's internal/config/config.go."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class GeneralConfig(BaseModel):
    run_dir: str = "./runs"
    log_level: str = "info"


class PerceptionConfig(BaseModel):
    detector: str = "opencv"
    fallback: str = "opencv"


class UnderstandingConfig(BaseModel):
    vlm: str = "qwen3-vl-8b"
    schema_version: int = 1


class CleanupConfig(BaseModel):
    inpainter: str = "lama"


class CostConfig(BaseModel):
    i2v_budget_s: float = 0.0


class GenerationConfig(BaseModel):
    primary_model: str = "wan22-ti2v-5b"
    video_primary: str | None = None
    video_fallback: str | None = None
    fallback_models: list[str] = Field(default_factory=lambda: ["hunyuan15", "ltx23"])
    max_retries: int = 2
    max_clip_duration_s: float = 5.0
    max_i2v_clip_s: float = 5.0
    max_i2v_panels: int = 4
    max_i2v_takes: int = 2
    calibration_file: str = ""
    comfy_url: str = "http://127.0.0.1:8188"


class AudioConfig(BaseModel):
    tts_primary: str = "melotts"
    tts_fallback: str = "kokoro"


class CompositionConfig(BaseModel):
    output_format: str = "mp4"
    resolution: list[int] = Field(default_factory=lambda: [720, 1280])
    fps: int = 24
    transition_type: str = "crossfade"


class QCRouteThresholds(BaseModel):
    min_identity_similarity: float = 0.85
    max_flicker_score: float = 0.15
    min_motion_score: float = 0.01
    max_motion_score: float = 0.98
    max_warp_error: float = 0.12
    min_first_frame_fidelity: float = 0.75
    max_style_drift: float = 0.28
    max_retries: int = 2


class QCConfig(BaseModel):
    identity_threshold: float = 0.85
    flicker_threshold: float = 0.15
    max_new_text_lines: int = 0
    min_motion: float = 0.01
    max_motion: float = 0.98
    max_warp_error: float = 0.12
    min_first_frame_fidelity: float = 0.75
    max_style_drift: float = 0.28
    require_text_check: bool = True
    per_route: dict[str, QCRouteThresholds] = Field(default_factory=dict)


class WorkerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8100


class PipelineConfig(BaseModel):
    """Main pipeline configuration loaded from configs/default.yaml + configs/tiers/<tier>.yaml."""
    tier: str = "t1"
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    understanding: UnderstandingConfig = Field(default_factory=UnderstandingConfig)
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    composition: CompositionConfig = Field(default_factory=CompositionConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)

    @property
    def resolution(self) -> list[int]:
        return self.composition.resolution

    @property
    def target_fps(self) -> int:
        return self.composition.fps


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(
    tier: str | None = "t1",
    overrides: dict[str, Any] | None = None,
    config_dir: Path | str = "./configs",
) -> PipelineConfig:
    """Loads configs/default.yaml and merges configs/tiers/<tier>.yaml on top."""
    cfg_root = Path(config_dir)
    default_file = cfg_root / "default.yaml"
    raw_data: dict[str, Any] = {}

    if default_file.exists():
        loaded = yaml.safe_load(default_file.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw_data = loaded

    effective_tier = tier if tier and tier != "default" else "t1"
    tier_file = cfg_root / "tiers" / f"{effective_tier}.yaml"
    if tier_file.exists():
        tier_loaded = yaml.safe_load(tier_file.read_text(encoding="utf-8"))
        if isinstance(tier_loaded, dict):
            raw_data = _deep_merge(raw_data, tier_loaded)

    raw_data["tier"] = effective_tier
    if overrides:
        raw_data = _deep_merge(raw_data, overrides)

    return PipelineConfig.model_validate(raw_data)
