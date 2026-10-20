"""AI Worker Server — HTTP API wrapping Python CV & ML models with Model Residency Management.

The Go backend calls these endpoints over HTTP (:8100) to delegate AI inference and
rendering without paying model cold-start overhead on every stage.
"""
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
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
except ImportError:
    from ingest import stitch_vertical_pages
    from slicer import crop_and_save_panels, detect_panels_gutter
    from assembler import inpaint_panel_bubbles, synthesize_speech_wav
    from pipeline_cli import (
        analyze_panel_semantics,
        assemble_from_run_state,
        render_single_panel_clip,
    )

app = FastAPI(title="M2A AI Worker", version="0.2.0")
logger = logging.getLogger("m2a.worker")


class ModelResidencyManager:
    """Enforces single-heavy-model residency on GPU to prevent VRAM OOM."""

    def __init__(self) -> None:
        self.resident_model: Optional[str] = None
        self.light_models: set[str] = {"opencv-slicer", "lama-telea"}

    def activate(self, model_name: str) -> None:
        if self.resident_model and self.resident_model != model_name:
            logger.info(f"Unloading heavy model {self.resident_model} before loading {model_name}")
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        self.resident_model = model_name

    def loaded_models(self) -> List[str]:
        models = sorted(self.light_models)
        if self.resident_model:
            models.append(self.resident_model)
        return models


RESIDENCY = ModelResidencyManager()
HEAVY_SEMAPHORE = threading.BoundedSemaphore(max(1, int(os.environ.get("M2A_HEAVY_SLOTS", "2"))))


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


class RenderResponse(BaseModel):
    clip_path: str
    frames_count: int
    duration_s: float
    route_used: str
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
    loaded_models: List[str]
    gpu_available: bool


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — reports loaded models and GPU status."""
    gpu_avail = False
    try:
        import torch  # type: ignore
        gpu_avail = bool(torch.cuda.is_available())
    except ImportError:
        pass
    return HealthResponse(
        status="ok",
        loaded_models=RESIDENCY.loaded_models(),
        gpu_available=gpu_avail,
    )


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

    RESIDENCY.activate("qwen3-vl")
    sem = analyze_panel_semantics(
        panel_image_path=img_path,
        panel_id=req.panel_id,
        language=req.language,
        reading_order=req.reading_order,
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


@app.post("/render", response_model=RenderResponse)
def render_panel(req: RenderRequest) -> RenderResponse:
    """Render a panel clip using Ken Burns, 2.5D Parallax, or I2V and compute QC metrics."""
    img_path = Path(req.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")

    with HEAVY_SEMAPHORE:
        if req.route == "i2v":
            RESIDENCY.activate("wan22-i2v")
        res = render_single_panel_clip(
            image_path=img_path,
            output_path=Path(req.output_path),
            route=req.route,
            shot_type=req.shot_type,
            duration_s=req.duration_s,
            fps=req.fps,
            target_res=(req.width, req.height),
        )
    return RenderResponse(**res)


@app.post("/tts", response_model=TTSResponse)
def synthesize_speech(req: TTSRequest) -> TTSResponse:
    """Generate playable WAV speech audio for a dialogue line."""
    RESIDENCY.activate("tts-engine")
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
