"""AI Worker Server — HTTP API wrapping Python CV & ML models with Model Residency Management.

The Go backend calls these endpoints over HTTP (:8100) to delegate AI inference and
rendering without paying model cold-start overhead on every stage.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from workers.ingest import stitch_vertical_pages
    from workers.slicer import crop_and_save_panels, detect_panels_gutter
    from workers.assembler import inpaint_panel_bubbles, synthesize_speech_wav
    from workers.pipeline_cli import (
        analyze_panel_semantics,
        assemble_from_run_state,
        render_single_panel_clip,
    )
    from workers.video.comfy_client import ComfyClient
    from workers.video.provider import load_model_specs
except ImportError:
    from ingest import stitch_vertical_pages
    from slicer import crop_and_save_panels, detect_panels_gutter
    from assembler import inpaint_panel_bubbles, synthesize_speech_wav
    from pipeline_cli import (
        analyze_panel_semantics,
        assemble_from_run_state,
        render_single_panel_clip,
    )
    from video.comfy_client import ComfyClient
    from video.provider import load_model_specs

app = FastAPI(title="M2A AI Worker", version="0.3.0")
logger = logging.getLogger("m2a.worker")


class ModelResidencyManager:
    """Enforces single-heavy-model residency on GPU to prevent VRAM OOM."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.resident_model: Optional[str] = None
        self.light_models: set[str] = {"opencv-slicer", "lama-telea"}
        self.last_comfy_url: str = os.environ.get("M2A_COMFY_URL", "")

    def activate(self, model_name: str, comfy_url: str = "") -> None:
        with self._lock:
            if comfy_url:
                self.last_comfy_url = comfy_url
            if self.resident_model and self.resident_model != model_name:
                logger.info(f"Unloading heavy model {self.resident_model} before loading {model_name}")
                if self.resident_model.startswith("comfyui") and self.last_comfy_url:
                    try:
                        ComfyClient(base_url=self.last_comfy_url).free_memory()
                    except Exception:
                        pass
                self._empty_torch_cache()
            self.resident_model = model_name

    @contextmanager
    def acquire(self, owner: str, comfy_url: str = "") -> Iterator[None]:
        self.activate(owner, comfy_url=comfy_url)
        try:
            yield
        finally:
            pass

    def free_all(self, comfy_url: str = "") -> None:
        with self._lock:
            eff_url = comfy_url or self.last_comfy_url or os.environ.get("M2A_COMFY_URL", "")
            if eff_url:
                try:
                    ComfyClient(base_url=eff_url).free_memory()
                except Exception:
                    pass
            self._empty_torch_cache()
            self.resident_model = None

    @staticmethod
    def _empty_torch_cache() -> None:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def loaded_models(self) -> List[str]:
        models = sorted(self.light_models)
        if self.resident_model:
            models.append(self.resident_model)
        return models


RESIDENCY = ModelResidencyManager()
HEAVY_SEMAPHORE = threading.BoundedSemaphore(max(1, int(os.environ.get("M2A_HEAVY_SLOTS", "2"))))
JOBS_LOCK = threading.Lock()
RENDER_JOBS: Dict[str, Dict[str, Any]] = {}


# --- Request/Response Models ---

class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class IngestRequest(BaseModel):
    page_paths: List[str]
    output_path: str
    target_width: int = 800


class IngestResponse(BaseModel):
    stitched_path: str
    width: int
    height: int


class SegmentRequest(BaseModel):
    image_path: str
    output_dir: Optional[str] = None
    method: str = "opencv"


class PanelDetectionItem(BaseModel):
    panel_id: str
    bbox: BBox
    reading_order: int
    shot_type: str = "medium"
    crop_path: Optional[str] = None


class SegmentResponse(BaseModel):
    panels: List[PanelDetectionItem]
    method_used: str


class UnderstandRequest(BaseModel):
    panel_image_path: str
    panel_id: str
    language: str = "ko"
    reading_order: Optional[int] = None
    crops_dir: Optional[str] = None
    annotated_path: Optional[str] = None


class DialogueLineResponse(BaseModel):
    speaker_id: str = ""
    text: str = ""
    lang: str = "ko"
    emotion: str = "neutral"
    bbox: Optional[BBox] = None
    confidence: float = 0.95


class UnderstandResponse(BaseModel):
    panel_id: str
    shot: str = "medium"
    action: str = ""
    motion_prompt: str = ""
    motion_intensity: float = 0.3
    dialogue: List[DialogueLineResponse] = Field(default_factory=list)
    sfx: List[Dict[str, Any]] = Field(default_factory=list)
    characters: List[Dict[str, Any]] = Field(default_factory=list)


class InpaintRequest(BaseModel):
    image_path: str
    output_path: str
    mask_path: Optional[str] = None
    bubble_bboxes: Optional[List[Dict[str, int]]] = None


class InpaintResponse(BaseModel):
    output_path: str
    success: bool


class RenderRequest(BaseModel):
    image_path: str
    output_path: str
    route: str = "kenburns"
    shot_type: str = "medium"
    duration_s: float = 2.5
    fps: int = 24
    width: int = 720
    height: int = 1280
    panel_id: str = ""
    model_id: str = ""
    seed: int = 42
    action_prompt: str = ""
    motion_type: str = ""
    emotion: str = "neutral"
    comfy_url: str = ""


class RenderResponse(BaseModel):
    clip_path: str
    raw_clip_path: str = ""
    frames_count: int
    duration_s: float
    route_used: str
    model_id: str = ""
    seed: int = 42
    positive_prompt: str = ""
    negative_prompt: str = ""
    native_width: int = 0
    native_height: int = 0
    native_frames: int = 0
    native_fps: int = 0
    gen_seconds: float = 0.0
    peak_vram_gb: float = 0.0
    cached: bool = False
    error: str = ""
    qc_scores: Dict[str, Any]


class TTSRequest(BaseModel):
    text: str
    voice_id: str = "default"
    emotion: str = "neutral"
    language: str = "ko"
    output_path: str


class TTSResponse(BaseModel):
    audio_path: str
    duration_s: float


class AssembleRequest(BaseModel):
    run_dir: str
    fps: int = 24
    width: int = 720
    height: int = 1280


class AssembleResponse(BaseModel):
    total_frames: int
    duration_s: float
    avi_path: str
    webp_path: str
    mp4_path: str
    timeline_path: Optional[str] = None
    subtitles_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    device: str = "cpu"
    loaded_models: List[str] = Field(default_factory=list)
    workers: Dict[str, str] = Field(default_factory=dict)
    video_providers: Dict[str, str] = Field(default_factory=dict)
    gpu_available: bool = False


def _check_workflow_weights(workflow_path: Path, object_info: Dict[str, Any]) -> Optional[str]:
    if not workflow_path.is_file():
        return f"missing workflow: {workflow_path.name}"
    try:
        wf = json.loads(workflow_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"invalid workflow: {exc}"
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ctype = str(node.get("class_type") or "")
        if ctype not in ("UNETLoader", "UnetLoaderGGUF", "CheckpointLoaderSimple"):
            continue
        node_meta = object_info.get(ctype)
        if not isinstance(node_meta, dict) or "input" not in node_meta:
            continue
        req_inputs = node_meta.get("input", {}).get("required", {})
        if not isinstance(req_inputs, dict):
            continue
        inputs = node.get("inputs", {})
        for param in ("unet_name", "ckpt_name"):
            if param in inputs and param in req_inputs:
                spec_entry = req_inputs[param]
                if isinstance(spec_entry, list) and spec_entry and isinstance(spec_entry[0], list):
                    choices = spec_entry[0]
                    target_weight = str(inputs[param])
                    if target_weight not in choices:
                        return f"missing weight: {target_weight}"
    return None


def _probe_video_providers() -> Dict[str, str]:
    specs = load_model_specs()
    comfy_url = os.environ.get("M2A_COMFY_URL", "").strip()
    providers: Dict[str, str] = {
        "kenburns": "ready",
        "parallax": "ready",
    }
    if not comfy_url:
        import socket
        try:
            with socket.create_connection(("127.0.0.1", 8188), timeout=0.05):
                comfy_url = "http://127.0.0.1:8188"
        except OSError:
            for mid, spec in specs.items():
                providers[mid] = "disabled" if not spec.enabled else "standby"
            return providers

    client = ComfyClient(base_url=comfy_url)
    try:
        base_probe = client.probe(required_nodes=[])
        avail_nodes = set(base_probe.get("available_nodes") or [])
        obj_resp = client._request("GET", "/object_info", timeout=5.0)
        object_info = obj_resp.json() if obj_resp.status_code == 200 else {}
    except Exception as exc:
        for mid, spec in specs.items():
            providers[mid] = "disabled" if not spec.enabled else f"unavailable: {exc}"
        return providers

    repo_root = Path(__file__).resolve().parents[1]
    for mid, spec in specs.items():
        if not spec.enabled:
            providers[mid] = "disabled"
            continue
        missing_nodes = [n for n in spec.required_nodes if n not in avail_nodes]
        if missing_nodes:
            providers[mid] = f"unavailable: missing nodes {missing_nodes}"
            continue
        wf_path = spec.workflow if spec.workflow.is_absolute() else (repo_root / spec.workflow)
        weight_err = _check_workflow_weights(wf_path, object_info)
        if weight_err:
            providers[mid] = f"unavailable: {weight_err}"
        else:
            providers[mid] = "ready"
    return providers


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — reports loaded models, GPU status, and ComfyUI video provider readiness."""
    gpu_avail = False
    try:
        import torch  # type: ignore
        gpu_avail = bool(torch.cuda.is_available())
    except ImportError:
        pass
    v_providers = _probe_video_providers()
    if not gpu_avail:
        gpu_avail = any(
            v == "ready"
            for k, v in v_providers.items()
            if k not in ("kenburns", "parallax")
        )
    return HealthResponse(
        status="ok",
        device="cuda" if gpu_avail else "cpu",
        loaded_models=RESIDENCY.loaded_models(),
        workers={
            "ingest": "ready",
            "segment": "ready",
            "understand": "ready",
            "inpaint": "ready",
            "render": "ready",
            "tts": "ready",
            "assemble": "ready",
        },
        video_providers=v_providers,
        gpu_available=gpu_avail,
    )


@app.post("/gpu/free")
def free_gpu(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Unloads resident GPU models and tells ComfyUI to free VRAM."""
    comfy_url = str((payload or {}).get("comfy_url") or "")
    RESIDENCY.free_all(comfy_url=comfy_url)
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_pages(req: IngestRequest) -> IngestResponse:
    """Stitch multiple vertical manhwa pages into a single normalized strip."""
    paths = [Path(p) for p in req.page_paths]
    for p in paths:
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Page not found: {p}")
    out_path, (w, h) = stitch_vertical_pages(paths, Path(req.output_path), target_width=req.target_width)
    return IngestResponse(stitched_path=str(out_path.as_posix()), width=w, height=h)


@app.post("/segment", response_model=SegmentResponse)
def segment_panels(req: SegmentRequest) -> SegmentResponse:
    """Detect panels in a manhwa page or stitched strip using real 2D gutter slicing."""
    img_path = Path(req.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    img = Image.open(img_path).convert("RGB")
    detected = detect_panels_gutter(img_path, img=img)
    if req.output_dir:
        crop_paths = crop_and_save_panels(img_path, detected, Path(req.output_dir), img=img)
        for p, cp in zip(detected, crop_paths):
            p["crop_path"] = str(cp.as_posix())

    items = [
        PanelDetectionItem(
            panel_id=p["panel_id"],
            bbox=BBox(**p["bbox"]),
            reading_order=p["reading_order"],
            shot_type=p.get("shot_type", "medium"),
            crop_path=p.get("crop_path"),
        )
        for p in detected
    ]
    return SegmentResponse(panels=items, method_used="opencv")


@app.post("/understand", response_model=UnderstandResponse)
def understand_panel(req: UnderstandRequest) -> UnderstandResponse:
    """Run semantic understanding on a single panel."""
    img_path = Path(req.panel_image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.panel_image_path}")

    with RESIDENCY.acquire("qwen3-vl"):
        sem = analyze_panel_semantics(
            panel_image_path=img_path,
            panel_id=req.panel_id,
            language=req.language,
            reading_order=req.reading_order,
            crops_dir=Path(req.crops_dir) if req.crops_dir else None,
            annotated_path=Path(req.annotated_path) if req.annotated_path else None,
        )
    return UnderstandResponse(**sem)


@app.post("/inpaint", response_model=InpaintResponse)
def inpaint_text(req: InpaintRequest) -> InpaintResponse:
    """Remove speech bubbles and lettering from a panel using inpainting."""
    img_path = Path(req.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    out_path = inpaint_panel_bubbles(
        image_path=img_path,
        output_path=Path(req.output_path),
        bubble_bboxes=req.bubble_bboxes,
    )
    return InpaintResponse(output_path=str(out_path.as_posix()), success=True)


def _execute_render_request(req: RenderRequest, cancel_event: Optional[threading.Event] = None) -> RenderResponse:
    img_path = Path(req.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    with HEAVY_SEMAPHORE:
        owner = f"comfyui:{req.model_id or 'wan22'}" if req.route == "i2v" else f"cpu:{req.route}"
        with RESIDENCY.acquire(owner, comfy_url=req.comfy_url):
            res = render_single_panel_clip(
                image_path=img_path,
                output_path=Path(req.output_path),
                route=req.route,
                shot_type=req.shot_type,
                duration_s=req.duration_s,
                fps=req.fps,
                target_res=(req.width, req.height),
                panel_id=req.panel_id,
                model_id=req.model_id,
                seed=req.seed,
                action_prompt=req.action_prompt,
                motion_type_override=req.motion_type,
                emotion=req.emotion,
                comfy_url=req.comfy_url,
                cancel_event=cancel_event,
            )
    return RenderResponse(**res)


@app.post("/render", response_model=RenderResponse)
def render_panel(req: RenderRequest) -> RenderResponse:
    """Render a panel clip synchronously using Ken Burns, 2.5D Parallax, or I2V and compute QC metrics."""
    return _execute_render_request(req)


@app.post("/render/jobs")
def create_render_job(req: RenderRequest) -> Dict[str, Any]:
    """Submits an asynchronous panel render job (used by Go WorkerClient for long-running I2V clips)."""
    img_path = Path(req.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    job_id = uuid.uuid4().hex
    cancel_ev = threading.Event()
    job_entry: Dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "result": None,
        "error": "",
        "cancel_event": cancel_ev,
        "comfy_url": req.comfy_url or os.environ.get("M2A_COMFY_URL", ""),
    }
    with JOBS_LOCK:
        RENDER_JOBS[job_id] = job_entry

    def _runner() -> None:
        with JOBS_LOCK:
            if job_entry["status"] == "cancelled":
                return
            job_entry["status"] = "running"
        try:
            res = _execute_render_request(req, cancel_event=cancel_ev)
            with JOBS_LOCK:
                if job_entry["status"] != "cancelled":
                    job_entry["result"] = res.model_dump()
                    job_entry["status"] = "done"
        except Exception as exc:
            with JOBS_LOCK:
                if job_entry["status"] != "cancelled":
                    job_entry["error"] = str(exc)
                    job_entry["status"] = "failed"

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/render/jobs/{job_id}")
def get_render_job(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        job = RENDER_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Render job {job_id} not found")
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "result": job["result"],
            "error": job["error"],
        }


@app.delete("/render/jobs/{job_id}")
def cancel_render_job(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        job = RENDER_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Render job {job_id} not found")
        job["cancel_event"].set()
        job["status"] = "cancelled"
        comfy_url = job.get("comfy_url", "")
    if comfy_url:
        try:
            ComfyClient(base_url=comfy_url).interrupt()
        except Exception:
            pass
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/tts", response_model=TTSResponse)
def synthesize_speech(req: TTSRequest) -> TTSResponse:
    """Generate playable WAV speech audio for a dialogue line."""
    with RESIDENCY.acquire("tts-engine"):
        dur = synthesize_speech_wav(
            text=req.text,
            output_path=Path(req.output_path),
            voice_id=req.voice_id,
            emotion=req.emotion,
            language=req.language,
        )
    return TTSResponse(audio_path=str(Path(req.output_path).as_posix()), duration_s=dur)


@app.post("/assemble", response_model=AssembleResponse)
def assemble_video(req: AssembleRequest) -> AssembleResponse:
    """Assemble final video timeline from RunState in run_dir."""
    run_dir = Path(req.run_dir)
    if not (run_dir / "state.json").exists():
        raise HTTPException(status_code=404, detail=f"state.json not found in {req.run_dir}")

    with HEAVY_SEMAPHORE:
        res = assemble_from_run_state(
            run_dir=run_dir,
            fps=req.fps,
            target_res=(req.width, req.height),
        )
    return AssembleResponse(**res)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="info")
