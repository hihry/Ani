"""Quality control node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from m2a.config import PipelineConfig
from m2a.state import ClipCandidate, QCScores, RunState, compute_deterministic_seed
from workers.pipeline_cli import render_single_panel_clip

logger = logging.getLogger(__name__)


def _next_fallback_route(route: str) -> str:
    if route == "i2v":
        return "parallax"
    return "kenburns"


def _evaluate_qc(s: QCScores, config: PipelineConfig, route: str = "parallax") -> bool:
    rt = config.qc.per_route.get(route)
    min_id = rt.min_identity_similarity if rt and rt.min_identity_similarity > 0 else config.qc.identity_threshold
    max_flk = rt.max_flicker_score if rt and rt.max_flicker_score > 0 else config.qc.flicker_threshold
    min_mot = rt.min_motion_score if rt and rt.min_motion_score > 0 else (config.qc.min_motion if config.qc.min_motion > 0 else 0.01)
    max_mot = rt.max_motion_score if rt and rt.max_motion_score > 0 else (config.qc.max_motion if config.qc.max_motion > 0 else 0.98)

    return (
        s.identity_similarity >= min_id
        and s.flicker_score <= max_flk
        and 0 <= s.new_text_count <= config.qc.max_new_text_lines
        and min_mot <= s.motion_score <= max_mot
    )


def qc(state: RunState, config: PipelineConfig) -> RunState:
    """Run quality checks on generated clip candidates and trigger fallback rendering if needed."""
    start_time = time.time()
    logger.info("Starting qc node...")

    run_dir = Path(config.general.run_dir)
    qc_dir = run_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    max_retries = max(1, config.generation.max_retries)
    report_items = []

    for panel in state.panels:
        candidates = state.clip_candidates.get(panel.panel_id, [])
        accepted: ClipCandidate | None = None
        for cand in candidates:
            cand_route = cand.route or panel.route
            passed = _evaluate_qc(cand.qc_scores, config, route=cand_route)
            cand.qc_scores.overall_pass = passed
            cand.accepted = passed
            if passed and accepted is None:
                accepted = cand

        take = len(candidates) + 1
        while accepted is None:
            next_route = _next_fallback_route(panel.route)
            if next_route == panel.route or take > max_retries + 1:
                if candidates:
                    best = max(candidates, key=lambda c: c.qc_scores.identity_similarity)
                    best.accepted = True
                    accepted = best
                    logger.warning(
                        f"Panel {panel.panel_id} QC below threshold on terminal lane {panel.route}; "
                        f"accepting best take {best.take_number} (identity={best.qc_scores.identity_similarity:.3f})"
                    )
                break

            logger.warning(f"Panel {panel.panel_id} failed QC; degrading {panel.route} -> {next_route}")
            panel.route = next_route
            clean_plate = Path(state.clean_plate_paths.get(panel.panel_id, ""))
            src_img = clean_plate if clean_plate.is_file() else (run_dir / "panels" / f"{panel.panel_id}.png")
            if not src_img.is_file():
                raise FileNotFoundError(f"qc fallback {panel.panel_id}: source image not found: {src_img}")

            fb_path = clips_dir / f"{panel.panel_id}_take{take:02d}_{next_route}.avi"
            seed = compute_deterministic_seed(state.config.run_id, panel.panel_id, take)
            t0 = time.time()
            rendered = render_single_panel_clip(
                image_path=src_img,
                output_path=fb_path,
                route=next_route,
                shot_type=panel.shot,
                duration_s=panel.duration_s,
                fps=config.composition.fps,
                target_res=(config.composition.resolution[0], config.composition.resolution[1]),
                panel_id=panel.panel_id,
                model_id=next_route,
                seed=seed,
                action_prompt=panel.action_prompt or panel.motion_prompt or panel.action,
            )
            gen_s = float(rendered.get("gen_seconds") or (time.time() - t0))
            state.cost_ledger.add_model_cost(next_route, gen_s)
            fb_qc = QCScores(**rendered["qc_scores"])
            passed = _evaluate_qc(fb_qc, config, route=next_route)
            fb_qc.overall_pass = passed
            fb_cand = ClipCandidate(
                panel_id=panel.panel_id,
                path=str(fb_path.as_posix()),
                take_number=take,
                route=next_route,
                model_id=next_route,
                seed=seed,
                gen_seconds=round(gen_s, 3),
                qc_scores=fb_qc,
                accepted=passed,
            )
            candidates = state.clip_candidates.setdefault(panel.panel_id, [])
            candidates.append(fb_cand)
            if passed:
                accepted = fb_cand
            take += 1

        if accepted is not None:
            report_items.append(
                {
                    "panel_id": panel.panel_id,
                    "accepted_path": accepted.path,
                    "take_number": accepted.take_number,
                    "qc_scores": accepted.qc_scores.model_dump(),
                }
            )

    (qc_dir / "qc_report.json").write_text(json.dumps(report_items, indent=2), encoding="utf-8")
    logger.info(f"QC node completed in {time.time() - start_time:.2f}s")
    return state
