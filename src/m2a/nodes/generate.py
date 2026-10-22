"""Generation node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from m2a.config import PipelineConfig
from m2a.state import ClipCandidate, QCScores, RunState, compute_deterministic_seed
from m2a.nodes.qc import _evaluate_qc
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
    primary_model = config.generation.video_primary or config.generation.primary_model

    for panel in state.panels:
        clean_plate = Path(state.clean_plate_paths.get(panel.panel_id, ""))
        src_img = clean_plate if clean_plate.is_file() else (run_dir / "panels" / f"{panel.panel_id}.png")
        if not src_img.is_file():
            raise FileNotFoundError(f"generate {panel.panel_id}: source image not found: {src_img}")

        clip_path = clips_dir / f"{panel.panel_id}_clip_01.avi"
        seed = compute_deterministic_seed(state.config.run_id, panel.panel_id, 1)
        model_id = primary_model if panel.route == "i2v" else panel.route
        prompt = panel.action_prompt or panel.motion_prompt or panel.action
        t0 = time.time()
        try:
            rendered = render_single_panel_clip(
                image_path=src_img,
                output_path=clip_path,
                route=panel.route,
                shot_type=panel.shot,
                duration_s=panel.duration_s,
                fps=fps,
                target_res=(res_w, res_h),
                panel_id=panel.panel_id,
                model_id=model_id,
                seed=seed,
                action_prompt=prompt,
                comfy_url=config.generation.comfy_url if panel.route == "i2v" else "",
            )
        except Exception as err:
            elapsed = time.time() - t0
            if panel.route == "i2v":
                state.cost_ledger.add_model_cost(model_id, elapsed)
                logger.warning(f"I2V render failed for {panel.panel_id} ({err}); falling back to parallax")
                panel.route = "parallax"
                model_id = "parallax"
                t0 = time.time()
                rendered = render_single_panel_clip(
                    image_path=src_img,
                    output_path=clip_path,
                    route="parallax",
                    shot_type=panel.shot,
                    duration_s=panel.duration_s,
                    fps=fps,
                    target_res=(res_w, res_h),
                    panel_id=panel.panel_id,
                    model_id="parallax",
                    seed=seed,
                    action_prompt=prompt,
                )
            else:
                raise

        route_used = rendered.get("route_used", panel.route)
        if route_used in ("i2v", "parallax", "kenburns"):
            panel.route = route_used
        eff_model_id = rendered.get("model_id") or (primary_model if route_used == "i2v" else route_used)
        gen_s = float(rendered.get("gen_seconds") or (time.time() - t0))
        state.cost_ledger.add_model_cost(eff_model_id, gen_s)

        qc_raw = rendered["qc_scores"]
        qc_scores = QCScores(**qc_raw)
        passed = _evaluate_qc(qc_scores, config, route=panel.route)
        qc_scores.overall_pass = passed

        state.clip_candidates[panel.panel_id] = [
            ClipCandidate(
                panel_id=panel.panel_id,
                path=str(clip_path.as_posix()),
                raw_clip_path=str(rendered.get("raw_clip_path", "")),
                take_number=1,
                route=panel.route,
                model_id=eff_model_id,
                seed=int(rendered.get("seed", seed)),
                positive_prompt=str(rendered.get("positive_prompt", "")),
                negative_prompt=str(rendered.get("negative_prompt", "")),
                native_width=int(rendered.get("native_width", 0)),
                native_height=int(rendered.get("native_height", 0)),
                native_frames=int(rendered.get("native_frames", 0)),
                native_fps=int(rendered.get("native_fps", 0)),
                gen_seconds=round(gen_s, 3),
                peak_vram_gb=float(rendered.get("peak_vram_gb", 0.0)),
                cached=bool(rendered.get("cached", False)),
                qc_scores=qc_scores,
                accepted=passed,
            )
        ]

    logger.info(f"Generate node completed in {time.time() - start_time:.2f}s")
    return state
