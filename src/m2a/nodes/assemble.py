"""Assembly node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from m2a.config import PipelineConfig
from m2a.state import RunState
from workers.pipeline_cli import assemble_from_run_state

logger = logging.getLogger(__name__)


def assemble(state: RunState, config: PipelineConfig) -> RunState:
    """Build final video timeline, mix audio, burn subtitles, and encode deliverables."""
    start_time = time.time()
    logger.info("Starting assemble node...")

    try:
        run_dir = Path(config.general.run_dir)
        out_dir = run_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        state.save(run_dir)

        has_panels = any((run_dir / "panels" / f"{p.panel_id}.png").exists() for p in state.panels)
        if has_panels:
            assemble_from_run_state(
                run_dir=run_dir,
                fps=config.composition.fps,
                target_res=(config.composition.resolution[0], config.composition.resolution[1]),
            )
    except Exception as e:
        logger.exception("Error in assemble node")
        state.record_error(str(e))
        raise

    logger.info(f"Assemble node completed in {time.time() - start_time:.2f}s")
    return state
