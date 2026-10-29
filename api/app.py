"""FastAPI application for pipeline job management and human checkpoint editing.

Delegates pipeline orchestration (`run`, `resume`, `inspect`) to the canonical Go `m2a`
engine binary and reads/updates `runs/<job_id>/state.json` atomically (`optim.md` A1 & P0-07).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from m2a.config import load_config
from m2a.graph import (
    inspect_go_engine,
    resume_go_engine,
    run_go_engine,
    save_checkpoint_edits,
    write_config_snapshot,
)
from m2a.state import PanelJSON, PipelineStage, RunConfig, RunState

logger = logging.getLogger(__name__)
app = FastAPI(title="Manhwa-to-Anime Job API", version="0.3.0")

RUNS_DIR = Path("./runs")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

STAGE_PROGRESS = {
    PipelineStage.INGEST: 0.08,
    PipelineStage.SEGMENT: 0.18,
    PipelineStage.UNDERSTAND: 0.30,
    PipelineStage.CHECKPOINT_1: 0.35,
    PipelineStage.CLEAN_PLATES: 0.46,
    PipelineStage.ROUTE: 0.54,
    PipelineStage.TTS: 0.62,
    PipelineStage.CHECKPOINT_2: 0.66,
    PipelineStage.GENERATE: 0.80,
    PipelineStage.QC: 0.90,
    PipelineStage.ASSEMBLE: 0.97,
    PipelineStage.DONE: 1.00,
    PipelineStage.FAILED: 0.00,
}


def _safe_run_dir(job_id: str) -> Path:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id format")
    return RUNS_DIR.resolve() / job_id


class JobSubmitRequest(BaseModel):
    source_pages: List[str] = Field(default_factory=list)
    source_language: str = "ko"
    compute_tier: str = "t1"
    auto_approve: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    current_stage: str
    progress: float
    panels_count: int = 0
    errors: List[str] = Field(default_factory=list)


class CheckpointUpdateRequest(BaseModel):
    panels: List[PanelJSON]
    resume_after_save: bool = True


@app.post("/jobs", response_model=JobStatusResponse)
def submit_job(request: JobSubmitRequest) -> JobStatusResponse:
    """Submits and starts a pipeline job via the canonical Go `m2a` engine."""
    tier = request.compute_tier if request.compute_tier in ("t0", "t1s", "t1", "t2") else "t1"
    lang = request.source_language if request.source_language in ("ko", "ja", "en") else "ko"
    cfg = load_config(tier=tier, overrides=request.config or None)

    run_cfg = RunConfig(
        config_hash="job_api_run",
        source_pages=request.source_pages,
        source_language=lang,
        output_resolution=cfg.resolution,
        compute_tier=tier,
    )
    job_id = run_cfg.run_id
    run_dir = _safe_run_dir(job_id)
    write_config_snapshot(cfg, run_dir)
    state = RunState(config=run_cfg, current_stage=PipelineStage.INGEST)
    state.save(run_dir)

    if request.source_pages:
        state = run_go_engine(
            state=state,
            run_dir=run_dir,
            config=cfg,
            auto_approve=request.auto_approve,
        )

    prog = STAGE_PROGRESS.get(state.current_stage, 0.0)
    return JobStatusResponse(
        job_id=job_id,
        status=state.current_stage.value.lower(),
        current_stage=state.current_stage.value,
        progress=prog,
        panels_count=len(state.panels),
        errors=state.errors,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Gets job status and progress directly from runs/<job_id>/state.json."""
    run_dir = _safe_run_dir(job_id)
    if not (run_dir / "state.json").exists():
        raise HTTPException(status_code=404, detail="Job not found")

    state = RunState.load(run_dir)
    prog = STAGE_PROGRESS.get(state.current_stage, 0.0)
    return JobStatusResponse(
        job_id=job_id,
        status=state.current_stage.value.lower(),
        current_stage=state.current_stage.value,
        progress=prog,
        panels_count=len(state.panels),
        errors=state.errors,
    )


@app.get("/jobs/{job_id}/state")
def get_job_state(job_id: str) -> Dict[str, Any]:
    """Returns full RunState JSON for UI inspection and CP1/CP2 editing."""
    run_dir = _safe_run_dir(job_id)
    if not (run_dir / "state.json").exists():
        raise HTTPException(status_code=404, detail="Job not found")
    state = RunState.load(run_dir)
    return state.model_dump()


@app.get("/jobs/{job_id}/inspect")
def inspect_job(job_id: str) -> Dict[str, Any]:
    """Runs `m2a inspect --run-dir` on the job directory and returns formatted inspection output."""
    run_dir = _safe_run_dir(job_id)
    if not (run_dir / "state.json").exists():
        raise HTTPException(status_code=404, detail="Job not found")
    state = RunState.load(run_dir)
    report = inspect_go_engine(run_dir)
    return {
        "job_id": job_id,
        "current_stage": state.current_stage.value,
        "report": report,
    }


@app.put("/jobs/{job_id}/checkpoint", response_model=JobStatusResponse)
def update_checkpoint(job_id: str, req: CheckpointUpdateRequest) -> JobStatusResponse:
    """Saves human edits at CP1 or CP2 and optionally resumes the pipeline via `m2a resume`."""
    run_dir = _safe_run_dir(job_id)
    if not (run_dir / "state.json").exists():
        raise HTTPException(status_code=404, detail="Job not found")

    state = RunState.load(run_dir)
    if state.current_stage not in (PipelineStage.CHECKPOINT_1, PipelineStage.CHECKPOINT_2):
        raise HTTPException(
            status_code=409,
            detail=f"Job is at stage {state.current_stage.value}, not at a human review checkpoint",
        )

    state.panels = req.panels
    save_checkpoint_edits(state, run_dir)

    if req.resume_after_save:
        cfg = load_config(tier=state.config.compute_tier)
        state = resume_go_engine(
            run_dir=run_dir,
            config=cfg,
            auto_approve=False,
        )

    prog = STAGE_PROGRESS.get(state.current_stage, 0.0)
    return JobStatusResponse(
        job_id=job_id,
        status=state.current_stage.value.lower(),
        current_stage=state.current_stage.value,
        progress=prog,
        panels_count=len(state.panels),
        errors=state.errors,
    )


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    """Downloads result MP4/WebP/AVI."""
    run_dir = _safe_run_dir(job_id)
    if not (run_dir / "state.json").exists():
        raise HTTPException(status_code=404, detail="Job not found")

    for candidate in (
        run_dir / "output" / "final_anime.mp4",
        run_dir / "output" / "final_anime.avi",
        run_dir / "output" / "final_anime.webp",
        run_dir / "baseline.mp4",
        run_dir / "baseline.avi",
        run_dir / "baseline.webp",
    ):
        if candidate.exists():
            return FileResponse(path=str(candidate), filename=candidate.name)

    raise HTTPException(status_code=404, detail=f"No rendered video artifact found yet for job {job_id}")


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> Dict[str, str]:
    """Marks a job as cancelled/failed."""
    run_dir = _safe_run_dir(job_id)
    if not (run_dir / "state.json").exists():
        raise HTTPException(status_code=404, detail="Job not found")

    state = RunState.load(run_dir)
    state.record_error("Job cancelled via API")
    state.advance_to(PipelineStage.FAILED)
    state.save(run_dir)
    return {"message": "Job cancelled"}


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
