import logging
import time
from typing import Callable
from pathlib import Path

from m2a.state import RunState, PipelineStage
from m2a.config import PipelineConfig

logger = logging.getLogger(__name__)

NodeHandler = Callable[[RunState, PipelineConfig], RunState]

class PipelineGraph:
    """A clean, custom state machine for the Manhwa-to-Anime pipeline."""
    
    def __init__(self, config: PipelineConfig, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.nodes: dict[PipelineStage, NodeHandler] = {}
        
        # Define valid transitions
        self.transitions = {
            PipelineStage.INGEST: PipelineStage.SEGMENT,
            PipelineStage.SEGMENT: PipelineStage.UNDERSTAND,
            PipelineStage.UNDERSTAND: PipelineStage.CHECKPOINT_1,
            PipelineStage.CHECKPOINT_1: PipelineStage.CLEAN_PLATES,
            PipelineStage.CLEAN_PLATES: PipelineStage.ROUTE,
            PipelineStage.ROUTE: PipelineStage.CHECKPOINT_2,
            PipelineStage.CHECKPOINT_2: PipelineStage.GENERATE,
            PipelineStage.GENERATE: PipelineStage.QC,
            PipelineStage.QC: PipelineStage.TTS,
            PipelineStage.TTS: PipelineStage.ASSEMBLE,
            PipelineStage.ASSEMBLE: PipelineStage.DONE,
        }

    def register_node(self, stage: PipelineStage, handler: NodeHandler) -> None:
        """Registers a handler function for a specific pipeline stage."""
        self.nodes[stage] = handler

    def _save_checkpoint(self, state: RunState) -> None:
        """Saves the current state."""
        try:
            state.save(self.run_dir)
            logger.info(f"Checkpoint saved at stage: {state.current_stage}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _execute_stage(self, stage: PipelineStage, state: RunState) -> RunState:
        """Executes a single stage with retry logic."""
        if stage not in self.nodes:
            # If no handler is registered, just advance state if it's a structural node
            logger.warning(f"No handler registered for {stage}, skipping...")
            return state
            
        handler = self.nodes[stage]
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                logger.info(f"Starting stage: {stage} (Attempt {attempt + 1})")
                new_state = handler(state, self.config)
                elapsed = time.time() - start_time
                new_state.cost_ledger.per_stage_seconds[stage.value] = elapsed
                return new_state
            except Exception as e:
                logger.error(f"Error in stage {stage}: {e}")
                if attempt == max_retries - 1:
                    state.errors.append(f"Stage {stage} failed after {max_retries} attempts: {str(e)}")
                    state.advance_to(PipelineStage.FAILED)
                    return state
                time.sleep(2 ** attempt)  # Exponential backoff
        return state

    def run(self, state: RunState) -> RunState:
        """Runs the pipeline from the beginning."""
        return self.resume(state)

    def resume(self, state: RunState) -> RunState:
        """Resumes the pipeline from its current stage."""
        logger.info(f"Resuming pipeline from stage: {state.current_stage}")
        
        while state.current_stage != PipelineStage.DONE and state.current_stage != PipelineStage.FAILED:
            current_stage = state.current_stage
            
            # Execute current stage
            state = self._execute_stage(current_stage, state)
            
            if state.current_stage == PipelineStage.FAILED:
                self._save_checkpoint(state)
                break
                
            # Determine next stage
            next_stage = self.transitions.get(current_stage)
            if next_stage is None:
                logger.error(f"No transition defined from {current_stage}")
                state.errors.append(f"Invalid transition from {current_stage}")
                state.advance_to(PipelineStage.FAILED)
                self._save_checkpoint(state)
                break
                
            state.advance_to(next_stage)
            self._save_checkpoint(state)
            
        return state
