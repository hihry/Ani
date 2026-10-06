"""FastAPI application for pipeline job management."""
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from typing import Dict

logger = logging.getLogger(__name__)
app = FastAPI(title="Manhwa-to-Anime API")

# TODO: Implement actual Redis queue integration
# redis_client = Redis()
JOB_QUEUE: Dict[str, dict] = {}

class JobSubmitRequest(BaseModel):
    config: dict
    compute_tier: str = "standard"

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float

@app.post("/jobs", response_model=JobStatusResponse)
async def submit_job(request: JobSubmitRequest):
    """Submits a new pipeline job."""
    job_id = str(uuid.uuid4())
    logger.info(f"Submitting job {job_id}")
    
    # TODO: Push to Redis queue
    job_data = {
        "id": job_id,
        "status": "queued",
        "progress": 0.0,
        "config": request.config,
    }
    JOB_QUEUE[job_id] = job_data
    
    return JobStatusResponse(job_id=job_id, status="queued", progress=0.0)

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Gets job status and progress."""
    if job_id not in JOB_QUEUE:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # TODO: Fetch from Redis
    job_data = JOB_QUEUE[job_id]
    return JobStatusResponse(
        job_id=job_id, 
        status=job_data["status"], 
        progress=job_data["progress"]
    )

@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Downloads result MP4."""
    if job_id not in JOB_QUEUE:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # TODO: Return FileResponse for the generated video
    return {"message": f"Result for {job_id} not implemented yet"}

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancels a job."""
    if job_id not in JOB_QUEUE:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # TODO: Remove from Redis queue and kill worker process if running
    JOB_QUEUE[job_id]["status"] = "cancelled"
    return {"message": "Job cancelled"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
