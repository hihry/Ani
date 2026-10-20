"""Clean plates node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from m2a.config import PipelineConfig
from m2a.state import RunState
from workers.assembler import inpaint_panel_bubbles

logger = logging.getLogger(__name__)


def clean_plates(state: RunState, config: PipelineConfig) -> RunState:
    """Create text masks and run inpainting to remove speech bubbles and text from panels."""
    start_time = time.time()
    logger.info("Starting clean_plates node...")

    try:
        run_dir = Path(config.general.run_dir)
        plates_dir = run_dir / "clean_plates"
        plates_dir.mkdir(parents=True, exist_ok=True)

        for panel in state.panels:
            crop_path = run_dir / "panels" / f"{panel.panel_id}.png"
            clean_path = plates_dir / f"{panel.panel_id}_clean.png"
            if crop_path.exists():
                bboxes = [d.bbox.model_dump() for d in panel.dialogue if d.bbox is not None]
                bboxes.extend([s.bbox.model_dump() for s in panel.sfx if s.bbox is not None])
                inpaint_panel_bubbles(crop_path, clean_path, bubble_bboxes=bboxes)
            state.clean_plate_paths[panel.panel_id] = str(clean_path.as_posix())

        state.cost_ledger.add_model_cost(config.cleanup.inpainter, time.time() - start_time)
    except Exception as e:
        logger.exception("Error in clean_plates node")
        state.record_error(str(e))
        raise

    logger.info(f"Clean plates node completed in {time.time() - start_time:.2f}s")
    return state
