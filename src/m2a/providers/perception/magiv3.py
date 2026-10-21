"""Magiv3 Perception Provider — Panel, text, character detection and speaker linking."""
import logging
import time
from pathlib import Path
from typing import List
from PIL import Image

from ..base import BBox, CharacterDetection, PerceptionProvider, SpeakerLink, TextDetection
from workers.slicer import detect_panels_gutter
from workers.understanding import detect_characters_and_link_speakers, detect_speech_bubbles_and_text

logger = logging.getLogger(__name__)


class Magiv3Provider(PerceptionProvider):
    """Panel, text, character detection and speaker linking."""

    def __init__(self) -> None:
        logger.info("Initializing Magiv3 perception provider...")

    def detect_panels(self, image: Path) -> List[BBox]:
        start_time = time.time()
        detected = detect_panels_gutter(Path(image))
        results = [
            BBox(
                x1=int(p["bbox"]["x"]),
                y1=int(p["bbox"]["y"]),
                x2=int(p["bbox"]["x"] + p["bbox"]["w"]),
                y2=int(p["bbox"]["y"] + p["bbox"]["h"]),
            )
            for p in detected
        ]
        logger.debug(f"Panel detection ({len(results)} panels): {time.time() - start_time:.2f}s")
        return results

    def detect_characters(self, image: Path, panels: List[BBox]) -> List[CharacterDetection]:
        img = Image.open(image).convert("RGB")
        results: List[CharacterDetection] = []
        target_boxes = panels or [BBox(x1=0, y1=0, x2=img.width, y2=img.height)]
        for idx, pb in enumerate(target_boxes, start=1):
            crop = img.crop((pb.x1, pb.y1, pb.x2, pb.y2))
            bubbles = detect_speech_bubbles_and_text(crop)
            chars, _ = detect_characters_and_link_speakers(crop, f"panel_{idx:03d}", bubbles, crops_dir=None)
            for c in chars:
                cb = c.get("bbox")
                if isinstance(cb, dict):
                    char_bbox = BBox(
                        x1=pb.x1 + int(cb["x"]),
                        y1=pb.y1 + int(cb["y"]),
                        x2=pb.x1 + int(cb["x"] + cb["w"]),
                        y2=pb.y1 + int(cb["y"] + cb["h"]),
                    )
                else:
                    char_bbox = pb
                results.append(
                    CharacterDetection(
                        bbox=char_bbox,
                        character_id=c["id"],
                        confidence=0.92,
                    )
                )
        return results

    def detect_text(self, image: Path) -> List[TextDetection]:
        img = Image.open(image).convert("RGB")
        bubbles = detect_speech_bubbles_and_text(img)
        return [
            TextDetection(
                bbox=BBox(
                    x1=int(b["bbox"]["x"]),
                    y1=int(b["bbox"]["y"]),
                    x2=int(b["bbox"]["x"] + b["bbox"]["w"]),
                    y2=int(b["bbox"]["y"] + b["bbox"]["h"]),
                ),
                confidence=float(b.get("confidence", 0.92)),
            )
            for b in bubbles
        ]

    def link_speakers(
        self,
        image: Path,
        text_regions: List[TextDetection],
        characters: List[CharacterDetection],
    ) -> List[SpeakerLink]:
        links: List[SpeakerLink] = []
        if not text_regions or not characters:
            return links
        for tr in text_regions:
            tx = 0.5 * (tr.bbox.x1 + tr.bbox.x2)
            ty = float(tr.bbox.y2)
            best_char = min(
                characters,
                key=lambda c: ((0.5 * (c.bbox.x1 + c.bbox.x2) - tx) ** 2 + (0.5 * (c.bbox.y1 + c.bbox.y2) - ty) ** 2),
            )
            links.append(
                SpeakerLink(
                    text_bbox=tr.bbox,
                    character_bbox=best_char.bbox,
                    confidence=round(min(tr.confidence, best_char.confidence), 3),
                )
            )
        return links
