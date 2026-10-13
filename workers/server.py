"""AI Worker Server — lightweight HTTP API wrapping Python ML models.

The Go backend calls these endpoints to delegate AI inference.
Each endpoint accepts image paths/data and returns structured JSON.
"""
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="M2A AI Worker", version="0.1.0")
logger = logging.getLogger("m2a.worker")

# --- Request/Response Models ---

class BBox(BaseModel):
    x: int
    y: int  
    w: int
    h: int

class SegmentRequest(BaseModel):
    image_path: str
    method: str = "opencv"  # "opencv" or "magiv3"

class SegmentResponse(BaseModel):
    panels: list[BBox]
    method_used: str

class UnderstandRequest(BaseModel):
    panel_image_path: str
    panel_id: str
    language: str = "ko"

class DialogueLineResponse(BaseModel):
    speaker_id: str = ""
    text: str = ""
    lang: str = "ko"
    emotion: str = "neutral"
    confidence: float = 0.0

class UnderstandResponse(BaseModel):
    panel_id: str
    shot: str = "medium"
    action: str = ""
    motion_prompt: str = ""
    motion_intensity: float = 0.3
    dialogue: list[DialogueLineResponse] = []
    characters: list[dict] = []

class InpaintRequest(BaseModel):
    image_path: str
    mask_path: str
    output_path: str

class InpaintResponse(BaseModel):
    output_path: str
    success: bool

class TTSRequest(BaseModel):
    text: str
    voice_id: str = "default"
    emotion: str = "neutral"
    language: str = "ko"
    output_path: str

class TTSResponse(BaseModel):
    audio_path: str
    duration_s: float

class HealthResponse(BaseModel):
    status: str
    loaded_models: list[str]
    gpu_available: bool

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — reports loaded models and GPU status."""
    return HealthResponse(
        status="ok",
        loaded_models=[],  # Will be populated when models are loaded
        gpu_available=False,  # Stub
    )

@app.post("/segment", response_model=SegmentResponse)
def segment_panels(req: SegmentRequest):
    """Detect panels in a manhwa page image."""
    logger.info(f"Segmenting {req.image_path} with method={req.method}")
    if not Path(req.image_path).exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")
    
    # STUB: Return mock panels
    # In production, this calls OpenCV gutter slicing or Magiv3
    return SegmentResponse(
        panels=[
            BBox(x=0, y=0, w=800, h=400),
            BBox(x=0, y=420, w=800, h=350),
            BBox(x=0, y=790, w=800, h=500),
        ],
        method_used=req.method,
    )

@app.post("/understand", response_model=UnderstandResponse)
def understand_panel(req: UnderstandRequest):
    """Run VLM understanding on a single panel."""
    logger.info(f"Understanding panel {req.panel_id} from {req.panel_image_path}")
    if not Path(req.panel_image_path).exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {req.panel_image_path}")
    
    # STUB: Return mock understanding
    return UnderstandResponse(
        panel_id=req.panel_id,
        shot="medium",
        action="Character stands in dramatic pose",
        motion_prompt="slow push-in, subtle hair sway",
        motion_intensity=0.3,
        dialogue=[],
        characters=[],
    )

@app.post("/inpaint", response_model=InpaintResponse)
def inpaint_text(req: InpaintRequest):
    """Remove text from panel using inpainting."""
    logger.info(f"Inpainting {req.image_path}")
    # STUB
    return InpaintResponse(output_path=req.output_path, success=True)

@app.post("/tts", response_model=TTSResponse)
def synthesize_speech(req: TTSRequest):
    """Generate TTS audio for a dialogue line."""
    logger.info(f"TTS: '{req.text[:50]}...' voice={req.voice_id} lang={req.language}")
    # STUB
    return TTSResponse(audio_path=req.output_path, duration_s=2.5)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="info")
