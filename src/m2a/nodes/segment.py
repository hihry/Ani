"""Segmentation node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from PIL import Image

from m2a.config import PipelineConfig
from m2a.state import BBox, PanelJSON, RunState
from workers.slicer import crop_and_save_panels, detect_panels_gutter

logger = logging.getLogger(__name__)


def segment(state: RunState, config: PipelineConfig) -> RunState:
    """Detect panels from the stitched strip using 2D gutter slicing."""
    start_time = time.time()
    logger.info("Starting segment node...")

    run_dir = Path(config.general.run_dir)
    panels_dir = run_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    stitched_path = run_dir / "stitched.png"
    if not stitched_path.exists():
        raise FileNotFoundError(f"segment: {stitched_path} missing (INGEST did not complete)")

    stitched_img = Image.open(stitched_path).convert("RGB")
    detected = detect_panels_gutter(stitched_path, img=stitched_img)
    if not detected:
        raise RuntimeError("segment: detector returned 0 panels")
    crop_and_save_panels(stitched_path, detected, panels_dir, img=stitched_img)

    panels: list[PanelJSON] = []
    for idx, p in enumerate(detected):
        shot_raw = p.get("shot_type", "medium")
        if shot_raw not in ("close_up", "medium", "wide", "splash"):
            shot_raw = "medium"
        route_val = "kenburns" if idx % 2 == 0 else "parallax"
        panels.append(
            PanelJSON(
                panel_id=p["panel_id"],
                bbox=BBox(**p["bbox"]),
                shot=shot_raw,
                route=route_val,
                duration_s=2.5,
            )
        )
    state.panels = panels
    logger.info(f"Detected {len(state.panels)} panels from {stitched_path} in {time.time() - start_time:.2f}s")
    return state
