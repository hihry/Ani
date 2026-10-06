"""Base abstract classes for all providers."""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TextRegion(BaseModel):
    """Represents a detected text region."""
    text: str
    bbox: List[int]
    confidence: float
    language: str

class BBox(BaseModel):
    """Bounding box coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

class CharacterDetection(BaseModel):
    """Character detected in a panel."""
    bbox: BBox
    character_id: str
    confidence: float

class TextDetection(BaseModel):
    """Text detection details."""
    bbox: BBox
    confidence: float

class SpeakerLink(BaseModel):
    """Links a text region to a character."""
    text_bbox: BBox
    character_bbox: BBox
    confidence: float

class VideoProvider(ABC):
    """Abstract base class for video generation providers."""
    @abstractmethod
    def generate_clip(self, image: Path, prompt: str, duration_s: float, seed: int, **kwargs) -> Path:
        """Generates a video clip from an image."""
        pass

    @abstractmethod
    def estimate_vram_gb(self) -> float:
        """Estimates VRAM usage in GB."""
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Returns the model name."""
        pass

class VLMProvider(ABC):
    """Abstract base class for Vision-Language Models."""
    @abstractmethod
    def analyze_panel(self, image: Path, schema: dict) -> dict:
        """Analyzes a panel and returns structured data matching the schema."""
        pass

    @abstractmethod  
    def ocr(self, image: Path, language: str) -> list[dict]:
        """Performs OCR on an image."""
        pass

class OCRProvider(ABC):
    """Abstract base class for OCR providers."""
    @abstractmethod
    def recognize(self, image: Path, language: str) -> list[TextRegion]:
        """Recognizes text in an image."""
        pass

class TTSProvider(ABC):
    """Abstract base class for Text-to-Speech providers."""
    @abstractmethod
    def synthesize(self, text: str, voice_id: str, emotion: str, output_path: Path) -> float:
        """Synthesizes speech and returns duration."""
        pass

    @abstractmethod
    def clone_voice(self, reference_audio: Path) -> str:
        """Clones a voice and returns a voice_id."""
        pass

class InpaintProvider(ABC):
    """Abstract base class for inpainting providers."""
    @abstractmethod
    def inpaint(self, image: Path, mask: Path, output: Path) -> Path:
        """Inpaints an image using a mask."""
        pass

class PerceptionProvider(ABC):
    """Abstract base class for perception providers."""
    @abstractmethod
    def detect_panels(self, image: Path) -> list[BBox]:
        """Detects panels in an image."""
        pass

    @abstractmethod
    def detect_characters(self, image: Path, panels: list[BBox]) -> list[CharacterDetection]:
        """Detects characters within panels."""
        pass

    @abstractmethod
    def detect_text(self, image: Path) -> list[TextDetection]:
        """Detects text regions."""
        pass

    @abstractmethod
    def link_speakers(self, image: Path, text_regions: list[TextDetection], characters: list[CharacterDetection]) -> list[SpeakerLink]:
        """Links detected text to characters."""
        pass
