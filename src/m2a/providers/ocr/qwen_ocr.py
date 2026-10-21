"""Qwen OCR Provider delegating to Qwen3VLProvider."""
import logging
from pathlib import Path
from typing import List, Optional

from ..base import OCRProvider, TextRegion
from ..vlm.qwen3vl import Qwen3VLProvider

logger = logging.getLogger(__name__)


class QwenOCRProvider(OCRProvider):
    """Delegates to Qwen3VLProvider for OCR."""

    def __init__(self, vlm_provider: Optional[Qwen3VLProvider] = None) -> None:
        self.vlm = vlm_provider or Qwen3VLProvider()
        logger.info("Initialized QwenOCRProvider")

    def recognize(self, image: Path, language: str = "ko") -> List[TextRegion]:
        raw_results = self.vlm.ocr(Path(image), language)
        return [
            TextRegion(
                text=r.get("text", ""),
                bbox=list(r.get("bbox", [0, 0, 0, 0])),
                confidence=float(r.get("confidence", 0.95)),
                language=r.get("language", language),
            )
            for r in raw_results
        ]
