"""Manga OCR Provider."""
import logging
import time
from pathlib import Path
from typing import List

from ..base import OCRProvider, TextRegion

logger = logging.getLogger(__name__)

class MangaOCRProvider(OCRProvider):
    """Manga OCR for Japanese text."""
    def __init__(self):
        logger.info("Loading MangaOCR model...")
        # TODO: Import manga_ocr and load model
        self.model = None

    def recognize(self, image: Path, language: str) -> List[TextRegion]:
        if language not in ("ja", "japanese"):
            logger.warning(f"MangaOCR is Japanese-only, requested {language}")
            
        start_time = time.time()
        # TODO: Implement MangaOCR recognition
        results = []
        gpu_seconds = time.time() - start_time
        logger.info(f"MangaOCR processing took {gpu_seconds:.2f}s")
        return results
