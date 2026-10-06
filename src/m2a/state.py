import json
import uuid
import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Callable, Any

from pydantic import BaseModel, ConfigDict, Field


class BBox(BaseModel):
    """Bounding box coordinates."""
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
    lang: Literal["ko", "ja", "en"]
    emotion: str
    bbox: BBox | None = None
    confidence: float


class SFX(BaseModel):
    """Sound effect present in a panel."""
    model_config = ConfigDict(frozen=False)
    text: str
    kind: Literal["impact", "ambient", "motion"]
    bbox: BBox | None = None


class CharacterRef(BaseModel):
    """Reference to a character in a specific panel."""
    model_config = ConfigDict(frozen=False)
    id: str
    name: str
    emotion: str
    pose: str
    crop_path: Path | None = None


class PanelJSON(BaseModel):
    """Data representation of a single manga/manhwa panel."""
    model_config = ConfigDict(frozen=False)
    panel_id: str
    bbox: BBox
    shot: Literal["close_up", "medium", "wide", "splash"]
    characters: list[CharacterRef]
    action: str
    dialogue: list[DialogueLine]
    sfx: list[SFX]
    motion_prompt: str
    style_anchor: str = "2D anime, flat colors, clean line art"
    motion_intensity: float = Field(ge=0.0, le=1.0, default=0.3)
    route: Literal["i2v", "parallax", "kenburns"] = "kenburns"
    duration_s: float = Field(ge=0.5, le=10.0, default=4.0)


class VoiceProfile(BaseModel):
    """Voice configuration for a character."""
    model_config = ConfigDict(frozen=False)
    voice_id: str
    reference_audio_path: Path | None = None
    style: str


class CharacterEntry(BaseModel):
    """Detailed character information for the bible."""
    model_config = ConfigDict(frozen=False)
    name: str
    description: str
    reference_crops: list[Path] = Field(default_factory=list)
    voice: VoiceProfile
    appearances: list[str] = Field(default_factory=list)


class CharacterBible(BaseModel):
    """Global character bible across the project."""
    model_config = ConfigDict(frozen=False)
    characters: dict[str, CharacterEntry] = Field(default_factory=dict)


class QCScores(BaseModel):
    """Quality control scores for a generated clip."""
    model_config = ConfigDict(frozen=False)
    identity_similarity: float
    new_text_count: int
    flicker_score: float
    motion_score: float
    overall_pass: bool


class ClipCandidate(BaseModel):
    """A generated video clip candidate for a panel."""
    model_config = ConfigDict(frozen=False)
    panel_id: str
    path: Path
    take_number: int
    qc_scores: QCScores
    accepted: bool


class AudioTrack(BaseModel):
    """Generated audio track for dialogue/sfx."""
    model_config = ConfigDict(frozen=False)
    panel_id: str
    speaker_id: str
    audio_path: Path
    duration_s: float
    emotion: str


class CostLedger(BaseModel):
    """Tracks compute costs."""
    model_config = ConfigDict(frozen=False)
    total_gpu_seconds: float = 0.0
    total_wall_seconds: float = 0.0
    per_model_seconds: dict[str, float] = Field(default_factory=dict)
    per_stage_seconds: dict[str, float] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Configuration for a specific pipeline run."""
    model_config = ConfigDict(frozen=False)
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config_hash: str
    source_pages: list[Path]
    source_language: str
    output_resolution: str
    compute_tier: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


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


class RunState(BaseModel):
    """The complete state of a pipeline run."""
    model_config = ConfigDict(frozen=False)
    
    config: RunConfig
    current_stage: PipelineStage
    panels: list[PanelJSON] = Field(default_factory=list)
    character_bible: CharacterBible | None = None
    clean_plate_paths: dict[str, Path] = Field(default_factory=dict)
    clip_candidates: dict[str, list[ClipCandidate]] = Field(default_factory=dict)
    audio_tracks: list[AudioTrack] = Field(default_factory=list)
    cost_ledger: CostLedger = Field(default_factory=CostLedger)
    errors: list[str] = Field(default_factory=list)

    def save(self, run_dir: Path) -> None:
        """Saves the current state to the specified directory."""
        state_file = run_dir / "state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        # Using model_dump_json for Pydantic v2
        state_file.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        """Loads the state from the specified directory."""
        state_file = run_dir / "state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"State file not found at {state_file}")
        return cls.model_validate_json(state_file.read_text())

    def content_hash(self) -> str:
        """Returns a hash of the current state content."""
        data = self.model_dump_json(exclude={"cost_ledger", "current_stage", "errors"})
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def advance_to(self, stage: PipelineStage) -> None:
        """Advances the state to a new pipeline stage."""
        self.current_stage = stage
