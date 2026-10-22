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

        primary_model = config.generation.video_primary or config.generation.primary_model
        allow_i2v = primary_model not in ("kenburns", "parallax") and state.config.compute_tier != "t0"
        max_i2v_panels = config.generation.max_i2v_panels if config.generation.max_i2v_panels > 0 else 3
        max_i2v_clip_s = config.generation.max_i2v_clip_s if config.generation.max_i2v_clip_s > 0 else 5.0
        gpu_budget_s = config.cost.i2v_budget_s if config.cost.i2v_budget_s > 0 else 30.0

        hero_indices = [
            idx
            for idx, panel in enumerate(state.panels)
            if panel.shot in ("close_up", "splash") and panel.motion_intensity >= 0.55
        ]
        hero_indices.sort(key=lambda i: (-state.panels[i].motion_intensity, i))

        selected_i2v: set[int] = set()
        if allow_i2v:
            for idx in hero_indices:
                if len(selected_i2v) >= max_i2v_panels:
                    break
                est_cost_s = min(state.panels[idx].duration_s, max_i2v_clip_s) * 2.0
                if gpu_budget_s >= est_cost_s or gpu_budget_s >= 10.0:
                    selected_i2v.add(idx)
                    gpu_budget_s -= max(10.0, est_cost_s)

        routing_plan = []
        for idx, panel in enumerate(state.panels):
            _, default_prompt, _ = CAMERA_MOTION_TAXONOMY.get(panel.shot, CAMERA_MOTION_TAXONOMY["medium"])
            if not panel.motion_prompt:
                panel.motion_prompt = default_prompt

            if idx in selected_i2v:
                panel.route = "i2v"
                if max_i2v_clip_s > 0 and panel.duration_s > max_i2v_clip_s:
                    panel.duration_s = max_i2v_clip_s
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
