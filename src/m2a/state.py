from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _posix_str(val: str | Path) -> str:
    return str(val).replace("\\", "/")


def compute_deterministic_seed(config_hash: str, panel_id: str, take: int) -> int:
    """Derives a reproducible 31-bit positive seed matching Go's ComputeDeterministicSeed."""
    digest = hashlib.sha256(f"{config_hash}:{panel_id}:{take}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class BBox(BaseModel):
    """Bounding box coordinates in pixels."""
    model_config = ConfigDict(frozen=False)
    x: int
    y: int
    w: int
    h: int


class DialogueLine(BaseModel):
    """A line of dialogue spoken by a character."""
    model_config = ConfigDict(frozen=False)
    speaker_id: str
    text: str
    lang: Literal["ko", "ja", "en"] = "ko"
    emotion: str = "neutral"
    bbox: BBox | None = None
    confidence: float = 0.95


class SFX(BaseModel):
    """Sound effect present in a panel."""
    model_config = ConfigDict(frozen=False)
    text: str
    kind: Literal["impact", "ambient", "motion"] = "impact"
    bbox: BBox | None = None


class CharacterRef(BaseModel):
    """Reference to a character in a specific panel."""
    model_config = ConfigDict(frozen=False)
    id: str
    name: str
    emotion: str = "neutral"
    pose: str = "standing"
    crop_path: str | None = None


class PanelJSON(BaseModel):
    """THE core data contract — structured representation of a single manhwa panel."""
    model_config = ConfigDict(frozen=False)
    panel_id: str
    bbox: BBox
    shot: Literal["close_up", "medium", "wide", "splash"] = "medium"
    characters: list[CharacterRef] = Field(default_factory=list)
    action: str = ""
    action_prompt: str = ""
    motion_type: str = ""
    dialogue: list[DialogueLine] = Field(default_factory=list)
    sfx: list[SFX] = Field(default_factory=list)
    motion_prompt: str = ""
    style_anchor: str = "2D anime, flat colors, clean line art"
    motion_intensity: float = Field(ge=0.0, le=1.0, default=0.3)
    route: Literal["i2v", "parallax", "kenburns"] = "kenburns"
    duration_s: float = Field(ge=0.5, le=10.0, default=4.0)


class VoiceProfile(BaseModel):
    """Voice configuration for a character."""
    model_config = ConfigDict(frozen=False)
    voice_id: str
    reference_audio_path: str | None = None
    style: str = "neutral"


class CharacterEntry(BaseModel):
    """Detailed character information for the bible."""
    model_config = ConfigDict(frozen=False)
    name: str
    description: str = ""
    reference_crops: list[str] = Field(default_factory=list)
    voice: VoiceProfile
    appearances: list[str] = Field(default_factory=list)


class CharacterBible(BaseModel):
    """Global character bible across the project."""
    model_config = ConfigDict(frozen=False)
    characters: dict[str, CharacterEntry] = Field(default_factory=dict)


class QCScores(BaseModel):
    """Quality control scores for a generated clip."""
    model_config = ConfigDict(frozen=False)
    identity_similarity: float = 0.0
    new_text_count: int = 0
    flicker_score: float = 0.0
    motion_score: float = 0.0
    warp_error: float = 0.0
    first_frame_fidelity: float = 0.0
    style_drift: float = 0.0
    metrics_version: int = 2
    overall_pass: bool = False
    reasons: list[str] = Field(default_factory=list)


class ClipCandidate(BaseModel):
    """A generated video clip candidate for a panel."""
    model_config = ConfigDict(frozen=False)
    panel_id: str
    path: str = ""
    take_number: int = 1
    seed: int = 0
    route: str = ""
    raw_clip_path: str = ""
    model_id: str = ""
    positive_prompt: str = ""
    negative_prompt: str = ""
    native_width: int = 0
    native_height: int = 0
    native_frames: int = 0
    native_fps: int = 0
    gen_seconds: float = 0.0
    peak_vram_gb: float = 0.0
    cached: bool = False
    fallback_reason: str = ""
    render_error: str = ""
    qc_scores: QCScores = Field(default_factory=QCScores)
    accepted: bool = False

    @model_validator(mode="before")
    @classmethod
    def _accept_phase3_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = dict(data)
            if "clip_path" in d and not d.get("path"):
                d["path"] = d.pop("clip_path")
            if "take" in d and "take_number" not in d:
                d["take_number"] = d.pop("take")
            return d
        return data

    @field_validator("path", "raw_clip_path", mode="before")
    @classmethod
    def _normalize_path(cls, v: Any) -> str:
        return _posix_str(v) if v else ""

    @property
    def clip_path(self) -> str:
        return self.path

    @property
    def take(self) -> int:
        return self.take_number


class AudioTrack(BaseModel):
    """Generated audio track for dialogue/sfx."""
    model_config = ConfigDict(frozen=False)
    panel_id: str
    speaker_id: str
    audio_path: str
    duration_s: float
    emotion: str = "neutral"

    @field_validator("audio_path", mode="before")
    @classmethod
    def _normalize_path(cls, v: Any) -> str:
        return _posix_str(v)


class CostLedger(BaseModel):
    """Tracks compute costs across models and stages (matches Go JSON tags)."""
    model_config = ConfigDict(frozen=False)
    total_gpu_seconds: float = 0.0
    total_wall_seconds: float = 0.0
    per_model_seconds: dict[str, float] = Field(default_factory=dict)
    per_stage_seconds: dict[str, float] = Field(default_factory=dict)

    def add_stage_cost(self, stage: str, seconds: float) -> None:
        self.per_stage_seconds[stage] = self.per_stage_seconds.get(stage, 0.0) + seconds
        self.total_wall_seconds += seconds

    def add_model_cost(self, model: str, seconds: float) -> None:
        self.per_model_seconds[model] = self.per_model_seconds.get(model, 0.0) + seconds
        self.total_gpu_seconds += seconds

    def add_cost(self, key: str, seconds: float) -> None:
        self.add_model_cost(key, seconds)


class RunConfig(BaseModel):
    """Configuration for a specific pipeline run (100% compatible with Go RunConfig)."""
    model_config = ConfigDict(frozen=False)
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config_hash: str = ""
    source_pages: list[str] = Field(default_factory=list)
    source_language: str = "ko"
    output_resolution: list[int] = Field(default_factory=lambda: [720, 1280])
    compute_tier: str = "t1"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("source_pages", mode="before")
    @classmethod
    def _normalize_pages(cls, v: Any) -> list[str]:
        if isinstance(v, list):
            return [_posix_str(p) for p in v]
        return []

    @field_validator("output_resolution", mode="before")
    @classmethod
    def _normalize_resolution(cls, v: Any) -> list[int]:
        if isinstance(v, str) and "x" in v.lower():
            parts = v.lower().split("x")
            return [int(parts[0]), int(parts[1])]
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return [int(v[0]), int(v[1])]
        return [720, 1280]


class PipelineStage(str, Enum):
    """Stages of the Manhwa-to-Anime pipeline."""
    INGEST = "INGEST"
    SEGMENT = "SEGMENT"
    UNDERSTAND = "UNDERSTAND"
    CHECKPOINT_1 = "CHECKPOINT_1"
    CLEAN_PLATES = "CLEAN_PLATES"
    ROUTE = "ROUTE"
    CHECKPOINT_2 = "CHECKPOINT_2"
    GENERATE = "GENERATE"
    QC = "QC"
    TTS = "TTS"
    ASSEMBLE = "ASSEMBLE"
    DONE = "DONE"
    FAILED = "FAILED"


class StageTransition(BaseModel):
    """Records when a stage started and ended (matches Go StageTransition)."""
    model_config = ConfigDict(frozen=False)
    stage: PipelineStage
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: str | None = None
    error: str | None = None


class RunState(BaseModel):
    """The complete state of a pipeline run (interoperable between Go and Python)."""
    model_config = ConfigDict(frozen=False)

    config: RunConfig
    current_stage: PipelineStage = PipelineStage.INGEST
    panels: list[PanelJSON] = Field(default_factory=list)
    character_bible: CharacterBible | None = Field(default_factory=CharacterBible)
    clean_plate_paths: dict[str, str] = Field(default_factory=dict)
    clip_candidates: dict[str, list[ClipCandidate]] = Field(default_factory=dict)
    audio_tracks: list[AudioTrack] = Field(default_factory=list)
    cost_ledger: CostLedger = Field(default_factory=CostLedger)
    errors: list[str] = Field(default_factory=list)
    stage_history: list[StageTransition] = Field(default_factory=list)

    @field_validator("clean_plate_paths", mode="before")
    @classmethod
    def _normalize_clean_plates(cls, v: Any) -> dict[str, str]:
        if isinstance(v, dict):
            return {str(k): _posix_str(val) for k, val in v.items()}
        return {}

    def save(self, run_dir: Path) -> None:
        """Atomically saves the current state to state.json in run_dir with fsync."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        state_file = run_dir / "state.json"
        tmp_file = run_dir / f"state-{uuid.uuid4().hex[:8]}.json.tmp"
        with open(tmp_file, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_file, state_file)

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        """Loads the state from state.json in run_dir."""
        state_file = Path(run_dir) / "state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found at {state_file}")
        return cls.model_validate_json(state_file.read_text(encoding="utf-8"))

    def content_hash(self) -> str:
        """Returns a SHA256 hex digest of config and panels (matching Go ContentHash)."""
        payload = {
            "config": self.config.model_dump(),
            "panels": [p.model_dump() for p in self.panels],
        }
        data = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def advance_to(self, stage: PipelineStage) -> None:
        """Advances the state to a new pipeline stage and records transition timestamps."""
        now = datetime.now(timezone.utc).isoformat()
        if self.stage_history and self.stage_history[-1].ended_at is None:
            self.stage_history[-1].ended_at = now
        self.current_stage = stage
        ended = now if stage == PipelineStage.DONE else None
        self.stage_history.append(StageTransition(stage=stage, started_at=now, ended_at=ended))

    def record_error(self, msg: str) -> None:
        self.errors.append(msg)
        if self.stage_history and not self.stage_history[-1].error:
            self.stage_history[-1].error = msg
