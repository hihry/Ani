"""Routing node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from m2a.config import PipelineConfig
from m2a.state import RunState
from workers.pipeline_cli import CAMERA_MOTION_TAXONOMY

logger = logging.getLogger(__name__)


def route(state: RunState, config: PipelineConfig) -> RunState:
    """Decide generation routing (i2v, parallax, kenburns) and camera motion prompt per panel."""
    start_time = time.time()
    logger.info("Starting route node...")

    try:
        run_dir = Path(config.general.run_dir)
        routes_dir = run_dir / "routes"
        routes_dir.mkdir(parents=True, exist_ok=True)

        allow_i2v = config.generation.primary_model not in ("kenburns", "parallax") and state.config.compute_tier != "t0"
        gpu_budget = 30 if allow_i2v else 0
        routing_plan = []

        for idx, panel in enumerate(state.panels):
            _, default_prompt, _ = CAMERA_MOTION_TAXONOMY.get(panel.shot, CAMERA_MOTION_TAXONOMY["medium"])
            if not panel.motion_prompt:
                panel.motion_prompt = default_prompt

            is_hero = panel.shot in ("close_up", "splash") and panel.motion_intensity >= 0.55
            if is_hero and allow_i2v and gpu_budget >= 10:
                panel.route = "i2v"
                gpu_budget -= 10
            elif panel.shot in ("medium", "close_up", "splash") and (panel.motion_intensity >= 0.35 or idx % 2 == 1):
                panel.route = "parallax"
            else:
                panel.route = "kenburns"

            routing_plan.append(panel.model_dump())

        plan_file = routes_dir / "routing_plan.json"
        plan_file.write_text(json.dumps(routing_plan, indent=2), encoding="utf-8")
    except Exception as e:
        logger.exception("Error in route node")
        state.record_error(str(e))
        raise

    logger.info(f"Route node completed in {time.time() - start_time:.2f}s")
    return state
