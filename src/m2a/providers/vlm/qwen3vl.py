"""Qwen3-VL VLM Provider with live HTTP endpoint support and deterministic CV+OCR fallback."""
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from ..base import VLMProvider
from workers.understanding import analyze_panel, call_vlm_http_endpoint

logger = logging.getLogger(__name__)


class Qwen3VLProvider(VLMProvider):
    """Qwen3-VL Vision-Language Model provider."""

    def __init__(self, endpoint: str = "http://localhost:8000/v1") -> None:
        self.endpoint = endpoint
        logger.info(f"Initialized Qwen3VLProvider connected to {endpoint}")

    def analyze_panel(self, image: Path, schema: dict | None = None, language: str = "ko") -> Dict[str, Any]:
        """Analyzes scene understanding and constrains output to PanelJSON schema."""
        start_time = time.time()
        img_path = Path(image)
        res = call_vlm_http_endpoint(img_path, panel_id=img_path.stem, language=language, endpoint=self.endpoint)
        if res is None:
            res = analyze_panel(img_path, panel_id=img_path.stem, language=language)
        logger.debug(f"Panel analysis took {time.time() - start_time:.2f}s")
        return res

    def ocr(self, image: Path, language: str = "ko") -> List[Dict[str, Any]]:
        """Runs OCR using Qwen3-VL / Phase 2 analyzer."""
        start_time = time.time()
        sem = self.analyze_panel(Path(image), schema={}, language=language)
        regions: List[Dict[str, Any]] = []
        for d in sem.get("dialogue", []):
            b = d.get("bbox") or {"x": 0, "y": 0, "w": 0, "h": 0}
            regions.append(
                {
                    "text": d.get("text", ""),
                    "bbox": [int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])],
                    "confidence": float(d.get("confidence", 0.95)),
                    "language": d.get("lang", language),
                }
            )
        for s in sem.get("sfx", []):
            b = s.get("bbox") or {"x": 0, "y": 0, "w": 0, "h": 0}
            regions.append(
                {
                    "text": s.get("text", ""),
                    "bbox": [int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])],
                    "confidence": 0.92,
                    "language": language,
                }
            )
        logger.debug(f"Qwen3-VL OCR took {time.time() - start_time:.2f}s")
        return regions
