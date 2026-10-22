"""State machine for the Manhwa-to-Anime pipeline (interoperable with Go Engine)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

from m2a.config import PipelineConfig
from m2a.nodes import (
    assemble,
    clean_plates,
    generate,
    ingest,
    qc,
    route,
    segment,
    tts_node,
    understand,
)
from m2a.state import CharacterBible, PanelJSON, PipelineStage, RunState

logger = logging.getLogger(__name__)

NodeHandler = Callable[[RunState, PipelineConfig], RunState]

RETRYABLE_STAGES = {PipelineStage.GENERATE, PipelineStage.TTS}


class PipelineGraph:
    """A clean state machine for the Manhwa-to-Anime pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        run_dir: Path,
        auto_approve_checkpoints: bool = False,
        register_defaults: bool = True,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.config.general.run_dir = str(self.run_dir.as_posix())
        self.auto_approve_checkpoints = auto_approve_checkpoints
        self.nodes: dict[PipelineStage, NodeHandler] = {}

        self.transitions = {
            PipelineStage.INGEST: PipelineStage.SEGMENT,
            PipelineStage.SEGMENT: PipelineStage.UNDERSTAND,
            PipelineStage.UNDERSTAND: PipelineStage.CHECKPOINT_1,
            PipelineStage.CHECKPOINT_1: PipelineStage.CLEAN_PLATES,
            PipelineStage.CLEAN_PLATES: PipelineStage.ROUTE,
            PipelineStage.ROUTE: PipelineStage.TTS,
            PipelineStage.TTS: PipelineStage.CHECKPOINT_2,
            PipelineStage.CHECKPOINT_2: PipelineStage.GENERATE,
            PipelineStage.GENERATE: PipelineStage.QC,
            PipelineStage.QC: PipelineStage.ASSEMBLE,
            PipelineStage.ASSEMBLE: PipelineStage.DONE,
        }

        if register_defaults:
            self.register_default_nodes()

    def register_default_nodes(self) -> None:
        self.register_node(PipelineStage.INGEST, ingest)
        self.register_node(PipelineStage.SEGMENT, segment)
        self.register_node(PipelineStage.UNDERSTAND, understand)
        self.register_node(PipelineStage.CLEAN_PLATES, clean_plates)
        self.register_node(PipelineStage.ROUTE, route)
        self.register_node(PipelineStage.TTS, tts_node)
        self.register_node(PipelineStage.GENERATE, generate)
        self.register_node(PipelineStage.QC, qc)
        self.register_node(PipelineStage.ASSEMBLE, assemble)

    def register_node(self, stage: PipelineStage, handler: NodeHandler) -> None:
        self.nodes[stage] = handler

    def _save_checkpoint(self, state: RunState) -> None:
        try:
            state.save(self.run_dir)
            logger.info(f"Checkpoint saved at stage: {state.current_stage}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _execute_stage(self, stage: PipelineStage, state: RunState) -> RunState:
        if stage not in self.nodes:
            logger.info(f"Structural stage {stage.value}, advancing...")
            return state

        handler = self.nodes[stage]
        attempts = max(1, self.config.generation.max_retries + 1) if stage in RETRYABLE_STAGES else 1

        for attempt in range(attempts):
            snapshot = state.model_copy(deep=True)
            try:
                start_time = time.time()
                logger.info(f"Starting stage: {stage.value} (Attempt {attempt + 1}/{attempts})")
                new_state = handler(snapshot, self.config)
                elapsed = time.time() - start_time
                new_state.cost_ledger.add_stage_cost(stage.value, elapsed)
                return new_state
            except Exception as e:
                logger.error(f"Error in stage {stage.value}: {e}")
                if attempt == attempts - 1:
                    state.record_error(f"Stage {stage.value} failed after {attempts} attempts: {e}")
                    state.advance_to(PipelineStage.FAILED)
                    return state
                time.sleep(0.05 * (2 ** attempt))
        return state

    def run(self, state: RunState) -> RunState:
        if state.current_stage != PipelineStage.INGEST:
            state.advance_to(PipelineStage.INGEST)
        self._save_checkpoint(state)
        return self._loop(state)

    def resume(self, state: RunState) -> RunState:
        logger.info(f"Resuming pipeline from stage: {state.current_stage}")
        if state.current_stage == PipelineStage.DONE:
            return state
        if state.current_stage == PipelineStage.FAILED:
            retry_stage = next(
                (
                    t.stage
                    for t in reversed(state.stage_history)
                    if t.stage not in (PipelineStage.FAILED, PipelineStage.DONE)
                ),
                PipelineStage.INGEST,
            )
            state.advance_to(retry_stage)
            self._save_checkpoint(state)
        elif state.current_stage in (PipelineStage.CHECKPOINT_1, PipelineStage.CHECKPOINT_2):
            if state.current_stage == PipelineStage.CHECKPOINT_1:
                self._sync_checkpoint_1_edits(state)
            elif state.current_stage == PipelineStage.CHECKPOINT_2:
                self._sync_checkpoint_2_edits(state)
            next_st = self.transitions[state.current_stage]
            state.advance_to(next_st)
            self._save_checkpoint(state)
        return self._loop(state)

    def _sync_checkpoint_1_edits(self, state: RunState) -> None:
        """Synchronizes any human edits made to understanding/<panel_id>.json or character_bible.json at CP1."""
        vlm_dir = self.run_dir / "understanding"
        if not vlm_dir.is_dir():
            return
        for i, p in enumerate(state.panels):
            panel_file = vlm_dir / f"{p.panel_id}.json"
            if panel_file.is_file():
                try:
                    raw_obj = json.loads(panel_file.read_text(encoding="utf-8"))
                    if isinstance(raw_obj, dict):
                        if "duration_s" in raw_obj:
                            raw_obj["duration_s"] = max(0.5, min(10.0, float(raw_obj["duration_s"])))
                        if "motion_intensity" in raw_obj:
                            raw_obj["motion_intensity"] = max(0.0, min(1.0, float(raw_obj["motion_intensity"])))
                    state.panels[i] = PanelJSON.model_validate(raw_obj)
                except Exception as e:
                    logger.warning(f"Skipping invalid CP1 edit in {panel_file}: {e}")
        bible_file = vlm_dir / "character_bible.json"
        if bible_file.is_file():
            try:
                state.character_bible = CharacterBible.model_validate_json(bible_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Skipping invalid CP1 edit in {bible_file}: {e}")

    def _sync_checkpoint_2_edits(self, state: RunState) -> None:
        """Synchronizes any human edits made to routes/routing_plan.json at CP2."""
        plan_file = self.run_dir / "routes" / "routing_plan.json"
        if not plan_file.is_file():
            return
        try:
            raw_data = json.loads(plan_file.read_text(encoding="utf-8"))
            raw_list = raw_data.get("panels") if isinstance(raw_data, dict) else raw_data
            if isinstance(raw_list, list) and len(raw_list) == len(state.panels):
                for i, raw_obj in enumerate(raw_list):
                    if isinstance(raw_obj, dict):
                        if "duration_s" in raw_obj:
                            raw_obj["duration_s"] = max(0.5, min(10.0, float(raw_obj["duration_s"])))
                        if "motion_intensity" in raw_obj:
                            raw_obj["motion_intensity"] = max(0.0, min(1.0, float(raw_obj["motion_intensity"])))
                        merged = state.panels[i].model_dump()
                        for k in ("route", "duration_s", "motion_intensity", "shot", "motion_prompt", "action"):
                            if k in raw_obj:
                                merged[k] = raw_obj[k]
                        state.panels[i] = PanelJSON.model_validate(merged)
        except Exception as e:
            logger.warning(f"Skipping invalid CP2 edit in {plan_file}: {e}")

    def _loop(self, state: RunState) -> RunState:
        while state.current_stage not in (PipelineStage.DONE, PipelineStage.FAILED):
            current_stage = state.current_stage

            if current_stage in (PipelineStage.CHECKPOINT_1, PipelineStage.CHECKPOINT_2):
                if not self.auto_approve_checkpoints:
                    logger.info(f"Reached {current_stage.value}; pausing for human review")
                    self._save_checkpoint(state)
                    return state
                next_stage = self.transitions[current_stage]
                state.advance_to(next_stage)
                self._save_checkpoint(state)
                continue

            state = self._execute_stage(current_stage, state)
            if state.current_stage == PipelineStage.FAILED:
                self._save_checkpoint(state)
                break

            next_stage = self.transitions.get(current_stage)
            if next_stage is None:
                state.record_error(f"Invalid transition from {current_stage}")
                state.advance_to(PipelineStage.FAILED)
                self._save_checkpoint(state)
                break

            state.advance_to(next_stage)
            self._save_checkpoint(state)

        return state
