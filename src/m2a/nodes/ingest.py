"""Ingestion node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from m2a.config import PipelineConfig
from m2a.state import RunState
from workers.ingest import stitch_vertical_pages

logger = logging.getLogger(__name__)


def ingest(state: RunState, config: PipelineConfig) -> RunState:
    """Ingest source manhwa pages and stitch them into a single normalized strip."""
    start_time = time.time()
    logger.info("Starting ingest node...")

    source_paths = [Path(p) for p in state.config.source_pages if p]
    if not source_paths:
        raise ValueError("ingest: no source pages specified")
    for p in source_paths:
        if not p.exists():
            raise FileNotFoundError(f"ingest: source page not found: {p}")

    run_dir = Path(config.general.run_dir)
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(source_paths, start=1):
        dst = pages_dir / f"page_{idx:03d}{src.suffix.lower() or '.png'}"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    stitched_path = run_dir / "stitched.png"
    stitch_vertical_pages(source_paths, stitched_path)
    logger.info(f"Saved stitched image to {stitched_path} in {time.time() - start_time:.2f}s")
    return state
