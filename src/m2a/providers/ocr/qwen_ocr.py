"""Qwen OCR Provider delegating to VLM."""
import logging
from pathlib import Path
from typing import List

from ..base import OCRProvider, TextRegion
from ..vlm.qwen3vl import Qwen3VLProvider

logger = logging.getLogger(__name__)

class QwenOCRProvider(OCRProvider):
    """Delegates to Qwen3VLProvider for OCR."""
    def __init__(self, vlm_provider: Qwen3VLProvider):
        self.vlm = vlm_provider
        logger.info("Initialized QwenOCRProvider")

    def recognize(self, image: Path, language: str) -> List[TextRegion]:
        logger.info(f"Delegating OCR for {image.name} to Qwen3VL")
        raw_results = self.vlm.ocr(image, language)
        # TODO: map raw_results to TextRegion
        return []
