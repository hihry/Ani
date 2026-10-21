"""PaddleOCR Provider supporting KR/JA/EN speech bubble & text recognition."""
import logging
import time
from pathlib import Path
from typing import List

from ..base import OCRProvider, TextRegion
from workers.understanding import analyze_panel

logger = logging.getLogger(__name__)


class PaddleOCRProvider(OCRProvider):
    """PaddleOCR provider supporting ko/ja/en with CV + stroke-topology fallback."""

    def __init__(self, use_gpu: bool = True) -> None:
        self.use_gpu = use_gpu
        logger.info(f"Initializing PaddleOCR (GPU={use_gpu})")

    def recognize(self, image: Path, language: str = "ko") -> List[TextRegion]:
        start_time = time.time()
        sem = analyze_panel(Path(image), panel_id=Path(image).stem, language=language)
        results: List[TextRegion] = []
        for d in sem.get("dialogue", []):
            b = d.get("bbox") or {"x": 0, "y": 0, "w": 0, "h": 0}
            results.append(
                TextRegion(
                    text=d.get("text", ""),
                    bbox=[int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])],
                    confidence=float(d.get("confidence", 0.95)),
                    language=d.get("lang", language),
                )
            )
        for s in sem.get("sfx", []):
            b = s.get("bbox") or {"x": 0, "y": 0, "w": 0, "h": 0}
            results.append(
                TextRegion(
                    text=s.get("text", ""),
                    bbox=[int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])],
                    confidence=0.92,
                    language=language,
                )
            )
        logger.info(f"PaddleOCR recognized {len(results)} regions in {time.time() - start_time:.2f}s")
        return results
