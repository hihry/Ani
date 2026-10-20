"""Worker process for processing pipeline jobs from run directories with atomic lock claiming."""
from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path

from m2a.config import load_config
from m2a.graph import PipelineGraph
from m2a.state import PipelineStage, RunState

logger = logging.getLogger(__name__)


def _try_claim_run(run_dir: Path) -> bool:
    """Claims exclusive processing ownership of run_dir via an atomic .lock file."""
    lock_file = run_dir / ".lock"
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def _release_run(run_dir: Path) -> None:
    try:
        (run_dir / ".lock").unlink(missing_ok=True)
    except Exception:
        pass


class Worker:
    def __init__(self, runs_dir: Path = Path("./runs")):
        self.running = True
        self.runs_dir = runs_dir

    def _handle_shutdown(self, sig, frame) -> None:
        logger.info("Graceful shutdown initiated...")
        self.running = False

    def process_run(self, run_dir: Path, auto_approve: bool = False) -> RunState:
        """Processes a single run directory using PipelineGraph."""
        state = RunState.load(run_dir)
        cfg = load_config(tier=state.config.compute_tier)
        graph = PipelineGraph(config=cfg, run_dir=run_dir, auto_approve_checkpoints=auto_approve)
        if state.current_stage == PipelineStage.INGEST and not state.panels:
            return graph.run(state)
        return graph.resume(state)

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        logger.info("Worker started. Polling for queued runs...")

        while self.running:
            if self.runs_dir.exists():
                for run_folder in self.runs_dir.iterdir():
                    state_file = run_folder / "state.json"
                    if state_file.exists() and not (run_folder / ".lock").exists():
                        try:
                            st = RunState.load(run_folder)
                            if st.current_stage == PipelineStage.INGEST and not st.stage_history:
                                if _try_claim_run(run_folder):
                                    try:
                                        logger.info(f"Picked up queued run: {run_folder.name}")
                                        self.process_run(run_folder)
                                    finally:
                                        _release_run(run_folder)
                        except Exception as e:
                            logger.error(f"Failed processing {run_folder}: {e}")
            time.sleep(1.0)

        logger.info("Worker stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = Worker()
    worker.run()
