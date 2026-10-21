"""HunyuanVideo 1.5 Video Provider via ComfyUI (`src/m2a/providers/video/hunyuan15.py`)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from workers.video.comfy_client import ComfyClient
from workers.video.provider import ComfyVideoProvider, GenResult

from ..base import VideoProvider

logger = logging.getLogger(__name__)


class Hunyuan15Provider(VideoProvider):
    """Thin wrapper around `ComfyVideoProvider` for HunyuanVideo 1.5."""

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8188",
        model_id: str = "hunyuan15-fp8",
        client: Optional[ComfyClient] = None,
    ) -> None:
        self.api_url = api_url
        self.model_id = model_id
        self.delegate = ComfyVideoProvider.from_model_id(
            self.model_id,
            client=client or ComfyClient(base_url=api_url),
        )
        self.last_result: Optional[GenResult] = None

    def generate_clip(self, image: Path, prompt: str, duration_s: float, seed: int, **kwargs: Any) -> Path:
        out_dir = Path(kwargs.get("out_dir") or image.parent)
        panel = dict(kwargs.get("panel") or {"panel_id": image.stem, "action_prompt": prompt})
        res = self.delegate.generate(
            panel=panel,
            clean_plate=Path(image),
            duration_s=duration_s,
            seed=seed,
            out_dir=out_dir,
            understanding_panel=kwargs.get("understanding_panel"),
            target_res=tuple(kwargs.get("target_res") or (540, 960)),
            fps=int(kwargs.get("fps") or 24),
        )
        self.last_result = res
        return res.clip_path

    def estimate_vram_gb(self) -> float:
        return float(self.delegate.spec.min_vram_gb)

    def model_name(self) -> str:
        return self.model_id
