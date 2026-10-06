"""PaddleOCR Provider."""
import logging
import time
from pathlib import Path
from typing import List

from ..base import OCRProvider, TextRegion

logger = logging.getLogger(__name__)

class PaddleOCRProvider(OCRProvider):
    """PaddleOCR supporting KR/JA/EN."""
    def __init__(self, use_gpu: bool = True):
        logger.info(f"Initializing PaddleOCR (GPU={use_gpu})")
        # TODO: Initialize PaddleOCR instance
        self.ocr = None

    def recognize(self, image: Path, language: str) -> List[TextRegion]:
        start_time = time.time()
        # TODO: Map language to paddleocr lang code, run recognition
        results = []
        gpu_seconds = time.time() - start_time
        logger.info(f"PaddleOCR processing took {gpu_seconds:.2f}s")
        return results
