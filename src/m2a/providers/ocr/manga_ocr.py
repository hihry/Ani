"""MangaOCR Provider for Japanese / CJK comic text."""
import logging
import time
from pathlib import Path
from typing import List

from ..base import OCRProvider, TextRegion
from workers.understanding import analyze_panel

logger = logging.getLogger(__name__)


class MangaOCRProvider(OCRProvider):
    """MangaOCR provider for Japanese/CJK speech bubbles."""

    def __init__(self) -> None:
        logger.info("Initializing MangaOCR provider...")

    def recognize(self, image: Path, language: str = "ja") -> List[TextRegion]:
        if language not in ("ja", "japanese"):
            logger.debug(f"MangaOCR invoked with language={language}")
        start_time = time.time()
        sem = analyze_panel(Path(image), panel_id=Path(image).stem, language=language if language in ("ko", "ja", "en") else "ja")
        results: List[TextRegion] = []
        for d in sem.get("dialogue", []):
            b = d.get("bbox") or {"x": 0, "y": 0, "w": 0, "h": 0}
            results.append(
                TextRegion(
                    text=d.get("text", ""),
                    bbox=[int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])],
                    confidence=float(d.get("confidence", 0.95)),
                    language=d.get("lang", "ja"),
                )
            )
        logger.info(f"MangaOCR recognized {len(results)} regions in {time.time() - start_time:.2f}s")
        return results
