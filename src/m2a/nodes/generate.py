"""Generation node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from m2a.config import PipelineConfig
from m2a.state import ClipCandidate, QCScores, RunState
from workers.pipeline_cli import render_single_panel_clip

logger = logging.getLogger(__name__)


def generate(state: RunState, config: PipelineConfig) -> RunState:
    """Generate animation clips for panels based on their assigned routes."""
    start_time = time.time()
    logger.info("Starting generate node...")

    run_dir = Path(config.general.run_dir)
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    res_w, res_h = config.composition.resolution[0], config.composition.resolution[1]
    fps = config.composition.fps
    min_motion = config.qc.min_motion if config.qc.min_motion > 0 else 0.01
    max_motion = config.qc.max_motion if config.qc.max_motion > 0 else 0.98

    for panel in state.panels:
        clean_plate = Path(state.clean_plate_paths.get(panel.panel_id, ""))
        src_img = clean_plate if clean_plate.is_file() else (run_dir / "panels" / f"{panel.panel_id}.png")
        if not src_img.is_file():
            raise FileNotFoundError(f"generate {panel.panel_id}: source image not found: {src_img}")

        clip_path = clips_dir / f"{panel.panel_id}_clip_01.avi"
        rendered = render_single_panel_clip(
            image_path=src_img,
            output_path=clip_path,
            route=panel.route,
            shot_type=panel.shot,
            duration_s=panel.duration_s,
            fps=fps,
            target_res=(res_w, res_h),
        )
        route_used = rendered.get("route_used", panel.route)
        if route_used in ("i2v", "parallax", "kenburns"):
            panel.route = route_used
        qc_raw = rendered["qc_scores"]
        qc_scores = QCScores(**qc_raw)
        passed = (
            qc_scores.identity_similarity >= config.qc.identity_threshold
            and qc_scores.flicker_score <= config.qc.flicker_threshold
            and 0 <= qc_scores.new_text_count <= config.qc.max_new_text_lines
            and min_motion <= qc_scores.motion_score <= max_motion
        )
        qc_scores.overall_pass = passed

        state.clip_candidates[panel.panel_id] = [
            ClipCandidate(
                panel_id=panel.panel_id,
                path=str(clip_path.as_posix()),
                take_number=1,
                qc_scores=qc_scores,
                accepted=passed,
            )
        ]

    state.cost_ledger.add_model_cost(config.generation.primary_model, time.time() - start_time)
    logger.info(f"Generate node completed in {time.time() - start_time:.2f}s")
    return state
