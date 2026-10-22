"""Deterministic FastAPI Fake ComfyUI server (`tests/fakes/fake_comfyui.py`).

Implements the full ComfyUI HTTP + WebSocket surface (`/system_stats`, `/object_info`,
`/upload/image`, `/prompt`, `/history/{id}`, `/view`, `/free`, `/interrupt`, `/queue`, `/ws`)
and generates real MJPEG AVI video clips from uploaded images with configurable fault
and adversarial modes:
  - `normal`: smooth pan/zoom clip matching workflow `(width, height, length)`
  - `oom`: simulates `CUDA error: out of memory`
  - `node_error`: simulates a ComfyUI custom-node runtime exception
  - `timeout`: leaves the prompt pending until `/interrupt` is called
  - `cached`: emits `execution_cached` and returns immediately
  - `frozen` / `flicker` / `text` / `shuffled` / `hue_shift` / `face_swap`: generates
    adversarial clips via `tests.fixtures.qc_adversarial`
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import Response
from PIL import Image

from tests.fixtures.qc_adversarial import (
    make_face_swap,
    make_flicker,
    make_frozen,
    make_hue_shift,
    make_shuffled,
    make_text_overlay,
)
from workers.assembler import encode_mjpeg_avi
from workers.render_kenburns import prepare_cinematic_canvas

ALL_KNOWN_NODES = [
    "UnetLoaderGGUF",
    "UNETLoader",
    "LoraLoaderModelOnly",
    "CheckpointLoaderSimple",
    "CLIPLoader",
    "DualCLIPLoader",
    "VAELoader",
    "LoadImage",
    "CLIPTextEncode",
    "WanImageToVideo",
    "LTXVImgToVideo",
    "HunyuanImageToVideo",
    "RandomNoise",
    "KSampler",
    "KSamplerAdvanced",
    "VAEDecode",
    "SaveWEBM",
]


class FakeComfyState:
    def __init__(self, tmp_dir: Path, mode: str = "normal", vram_gb: float = 24.0) -> None:
        self.tmp_dir = Path(tmp_dir)
        self.input_dir = self.tmp_dir / "input"
        self.output_dir = self.tmp_dir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.vram_gb = float(vram_gb)
        self.available_nodes: set[str] = set(ALL_KNOWN_NODES)
        self.model_modes: Dict[str, str] = {}
        self.history: Dict[str, Dict[str, Any]] = {}
        self.prompts_submitted: List[Dict[str, Any]] = []
        self.free_calls: List[Dict[str, Any]] = []
        self.interrupted: int = 0

    def resolve_mode_for_workflow(self, wf: Dict[str, Any]) -> str:
        for node in wf.values():
            if not isinstance(node, dict):
                continue
            inp = node.get("inputs") or {}
            for k in ("unet_name", "ckpt_name", "lora_name"):
                val = str(inp.get(k) or "")
                if val:
                    for key_sub, custom_mode in self.model_modes.items():
                        if key_sub in val:
                            return custom_mode
        return self.mode


def _extract_workflow_params(wf: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "image": "input.png",
        "width": 480,
        "height": 832,
        "frames": 25,
        "fps": 16,
        "seed": 42,
        "prefix": "m2a_fake",
    }
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        title = node.get("_meta", {}).get("title")
        inp = node.get("inputs") or {}
        if title == "M2A_IMAGE":
            params["image"] = str(inp.get("image") or "input.png")
        elif title == "M2A_LATENT":
            params["width"] = int(inp.get("width") or 480)
            params["height"] = int(inp.get("height") or 832)
            params["frames"] = int(inp.get("length") or inp.get("num_frames") or 25)
        elif title == "M2A_SEED":
            params["seed"] = int(inp.get("noise_seed") or inp.get("seed") or 42)
        elif title == "M2A_OUTPUT":
            params["fps"] = int(round(float(inp.get("fps") or inp.get("frame_rate") or 16)))
            params["prefix"] = str(inp.get("filename_prefix") or "m2a_fake")
    return params


def _render_synthetic_clip(
    src_img: Image.Image,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int,
    mode: str,
    out_path: Path,
) -> Path:
    size = (int(width), int(height))
    canvas = prepare_cinematic_canvas(src_img.convert("RGB"), target_res=size, overscan=1.10)
    cw, ch = canvas.size
    max_dx = max(1, cw - width)
    max_dy = max(1, ch - height)

    direction = 1 if (seed % 2 == 0) else -1
    base_frames: List[Image.Image] = []
    for i in range(max(2, num_frames)):
        t = i / float(max(1, num_frames - 1))
        u = t if direction > 0 else (1.0 - t)
        x0 = int(round(u * max_dx * 0.7))
        y0 = int(round(u * max_dy * 0.7))
        crop = canvas.crop((x0, y0, x0 + width, y0 + height))
        base_frames.append(crop)

    if mode == "frozen":
        frames = make_frozen(base_frames, num_frames=num_frames, size=size)
    elif mode == "flicker":
        frames = make_flicker(base_frames, num_frames=num_frames, size=size)
    elif mode in ("text", "text_overlay"):
        frames = make_text_overlay(base_frames, num_frames=num_frames, size=size)
    elif mode == "shuffled":
        frames = make_shuffled(base_frames, num_frames=num_frames, size=size, seed=seed)
    elif mode == "hue_shift":
        frames = make_hue_shift(base_frames, num_frames=num_frames, size=size)
    elif mode == "face_swap":
        frames = make_face_swap(base_frames, num_frames=num_frames, size=size)
    else:
        frames = base_frames

    return encode_mjpeg_avi(frames, out_path, fps=fps)


def create_fake_comfy_app(tmp_dir: Path, mode: str = "normal", vram_gb: float = 24.0) -> tuple[FastAPI, FakeComfyState]:
    state = FakeComfyState(tmp_dir=tmp_dir, mode=mode, vram_gb=vram_gb)
    app = FastAPI(title="Fake ComfyUI Server")
    app.state.comfy = state

    @app.get("/system_stats")
    def system_stats() -> Dict[str, Any]:
        vram_bytes = int(state.vram_gb * (1024 ** 3))
        return {
            "system": {"os": "linux", "comfyui_version": "0.3.10"},
            "devices": [
                {
                    "name": "Fake NVIDIA RTX GPU",
                    "type": "cuda",
                    "vram_total": vram_bytes,
                    "vram_free": int(vram_bytes * 0.6),
                    "vram_used": int(vram_bytes * 0.4),
                }
            ],
        }

    @app.get("/object_info")
    def object_info() -> Dict[str, Any]:
        return {node: {"input": {"required": {}}, "output": []} for node in sorted(state.available_nodes)}

    @app.get("/queue")
    def get_queue() -> Dict[str, Any]:
        return {"queue_running": [], "queue_pending": []}

    @app.post("/upload/image")
    async def upload_image(request: Request) -> Dict[str, Any]:
        body = await request.body()
        fname = "input.png"
        m = re.search(rb'filename="([^"]+)"', body)
        if m:
            fname = Path(m.group(1).decode("utf-8", errors="ignore")).name
        # Extract PNG payload from multipart body if present
        png_start = body.find(b"\x89PNG\r\n\x1a\n")
        png_end = body.find(b"IEND")
        if png_start >= 0 and png_end > png_start:
            content = body[png_start : png_end + 8]
        else:
            content = body
        dest = state.input_dir / fname
        dest.write_bytes(content)
        return {"name": fname, "subfolder": "", "type": "input"}

    @app.post("/prompt")
    def submit_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
        wf = payload.get("prompt")
        if not isinstance(wf, dict):
            raise HTTPException(status_code=400, detail="Missing 'prompt' workflow dict")

        prompt_id = uuid.uuid4().hex
        state.prompts_submitted.append({"prompt_id": prompt_id, "workflow": wf})
        eff_mode = state.resolve_mode_for_workflow(wf)

        if eff_mode == "oom":
            state.history[prompt_id] = {
                "prompt": wf,
                "outputs": {},
                "status": {
                    "status_str": "error",
                    "completed": False,
                    "messages": [
                        [
                            "execution_error",
                            {
                                "prompt_id": prompt_id,
                                "node_type": "KSampler",
                                "exception_message": "CUDA error: out of memory",
                            },
                        ]
                    ],
                },
            }
            return {"prompt_id": prompt_id, "number": len(state.prompts_submitted)}

        if eff_mode == "node_error":
            state.history[prompt_id] = {
                "prompt": wf,
                "outputs": {},
                "status": {
                    "status_str": "error",
                    "completed": False,
                    "messages": [
                        [
                            "execution_error",
                            {
                                "prompt_id": prompt_id,
                                "node_type": "WanImageToVideo",
                                "exception_message": "RuntimeError: tensor size mismatch in WanImageToVideo",
                            },
                        ]
                    ],
                },
            }
            return {"prompt_id": prompt_id, "number": len(state.prompts_submitted)}

        if eff_mode == "timeout":
            # Leave history empty so client times out and calls /interrupt
            return {"prompt_id": prompt_id, "number": len(state.prompts_submitted)}

        params = _extract_workflow_params(wf)
        img_path = state.input_dir / params["image"]
        if img_path.is_file():
            src_img = Image.open(img_path).convert("RGB")
        else:
            src_img = Image.new("RGB", (params["width"], params["height"]), (60, 90, 140))

        out_fname = f"{params['prefix']}_{prompt_id[:8]}.avi"
        out_path = state.output_dir / out_fname
        _render_synthetic_clip(
            src_img=src_img,
            width=params["width"],
            height=params["height"],
            num_frames=params["frames"],
            fps=params["fps"],
            seed=params["seed"],
            mode=eff_mode,
            out_path=out_path,
        )

        messages: List[Any] = [["execution_start", {"prompt_id": prompt_id}]]
        if eff_mode == "cached":
            messages.append(["execution_cached", {"prompt_id": prompt_id, "nodes": list(wf.keys())}])
        messages.append(["execution_success", {"prompt_id": prompt_id}])

        state.history[prompt_id] = {
            "prompt": wf,
            "outputs": {
                "save_node": {
                    "videos": [
                        {
                            "filename": out_fname,
                            "subfolder": "",
                            "type": "output",
                        }
                    ]
                }
            },
            "status": {
                "status_str": "success",
                "completed": True,
                "messages": messages,
            },
        }
        return {"prompt_id": prompt_id, "number": len(state.prompts_submitted)}

    @app.get("/history/{prompt_id}")
    def get_history(prompt_id: str) -> Dict[str, Any]:
        if prompt_id in state.history:
            return {prompt_id: state.history[prompt_id]}
        return {}

    @app.get("/view")
    def view_file(
        filename: str = Query(...),
        subfolder: str = Query(""),
        type: str = Query("output"),
    ) -> Response:
        base = state.output_dir if type == "output" else state.input_dir
        fpath = (base / subfolder / Path(filename).name) if subfolder else (base / Path(filename).name)
        if not fpath.is_file():
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        return Response(content=fpath.read_bytes(), media_type="video/x-msvideo")

    @app.post("/free")
    def free_vram(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state.free_calls.append(payload or {})
        return {"status": "ok"}

    @app.post("/interrupt")
    def interrupt_execution() -> Dict[str, Any]:
        state.interrupted += 1
        return {"status": "interrupted"}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.receive_text()
        except Exception:
            pass

    return app, state
