"""Quality control node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from m2a.config import PipelineConfig
from m2a.state import ClipCandidate, QCScores, RunState
from workers.pipeline_cli import render_single_panel_clip

logger = logging.getLogger(__name__)


def _next_fallback_route(route: str) -> str:
    if route == "i2v":
        return "parallax"
    return "kenburns"


def _evaluate_qc(s: QCScores, config: PipelineConfig) -> bool:
    min_motion = config.qc.min_motion if config.qc.min_motion > 0 else 0.01
    max_motion = config.qc.max_motion if config.qc.max_motion > 0 else 0.98
    return (
        s.identity_similarity >= config.qc.identity_threshold
        and s.flicker_score <= config.qc.flicker_threshold
        and 0 <= s.new_text_count <= config.qc.max_new_text_lines
        and min_motion <= s.motion_score <= max_motion
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
            passed = _evaluate_qc(cand.qc_scores, config)
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
            rendered = render_single_panel_clip(
                image_path=src_img,
                output_path=fb_path,
                route=next_route,
                shot_type=panel.shot,
                duration_s=panel.duration_s,
                fps=config.composition.fps,
                target_res=(config.composition.resolution[0], config.composition.resolution[1]),
            )
            fb_qc = QCScores(**rendered["qc_scores"])
            passed = _evaluate_qc(fb_qc, config)
            fb_qc.overall_pass = passed
            fb_cand = ClipCandidate(
                panel_id=panel.panel_id,
                path=str(fb_path.as_posix()),
                take_number=take,
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
