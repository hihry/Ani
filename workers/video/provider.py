"""Unified ComfyUI I2V provider (`workers/video/provider.py`).

Implements `ModelSpec`, `GenResult`, `load_model_specs`, and `ComfyVideoProvider`
per Phase 3 §5.1.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from workers.video.cache import ClipCache, compute_cache_key
from workers.video.comfy_client import ComfyClient
from workers.video.conform import conform, frames_for, pick_bucket, prepare_input
from workers.video.prompts import build_prompt
from workers.video.workflows import inject, load_template

logger = logging.getLogger("m2a.video.provider")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_MODELS_PATH = REPO_ROOT / "configs" / "video_models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    workflow_path: Path
    family: str
    min_vram_gb: int
    native_fps: int
    frame_step: int
    frame_offset: int
    max_frames: int
    resolution_buckets: List[Tuple[int, int]]
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    required_nodes: List[str] = field(default_factory=list)
    negative_prompt: str = (
        "text, speech bubble, watermark, subtitles, deformed face, extra limbs, "
        "3d render, photorealistic, blurry, low quality, flicker"
    )
    enabled: bool = True


@dataclass
class GenResult:
    clip_path: Path
    raw_clip_path: Path
    model_id: str
    seed: int
    positive_prompt: str
    negative_prompt: str
    native_width: int
    native_height: int
    native_frames: int
    native_fps: int
    gen_seconds: float
    peak_vram_gb: float
    cached: bool


def load_model_specs(config_path: Path = DEFAULT_VIDEO_MODELS_PATH) -> Dict[str, ModelSpec]:
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = REPO_ROOT / config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_models = data.get("models") or {}
    specs: Dict[str, ModelSpec] = {}
    for mid, m in raw_models.items():
        wf_rel = Path(m["workflow"])
        wf_path = wf_rel if wf_rel.is_file() else (REPO_ROOT / wf_rel)
        buckets = [(int(b[0]), int(b[1])) for b in (m.get("resolution_buckets") or [[544, 960]])]
        specs[mid] = ModelSpec(
            model_id=mid,
            workflow_path=wf_path,
            family=str(m.get("family", "wan22")),
            min_vram_gb=int(m.get("min_vram_gb", 12)),
            native_fps=int(m.get("native_fps", 16)),
            frame_step=int(m.get("frame_step", 4)),
            frame_offset=int(m.get("frame_offset", 1)),
            max_frames=int(m.get("max_frames", 81)),
            resolution_buckets=buckets,
            steps=int(m.get("steps", 20)),
            cfg=float(m.get("cfg", 5.0)),
            sampler=str(m.get("sampler", "uni_pc")),
            scheduler=str(m.get("scheduler", "simple")),
            required_nodes=[str(n) for n in (m.get("required_nodes") or [])],
            negative_prompt=str(m.get("negative_prompt", "")),
            enabled=bool(m.get("enabled", True)),
        )
    return specs


class ComfyVideoProvider:
    def __init__(
        self,
        spec: ModelSpec,
        client: ComfyClient,
        cache: Optional[ClipCache] = None,
    ) -> None:
        self.spec = spec
        self.client = client
        self.cache = cache or ClipCache()
        self.template = load_template(spec.workflow_path)

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        client: Optional[ComfyClient] = None,
        cache: Optional[ClipCache] = None,
        config_path: Path = DEFAULT_VIDEO_MODELS_PATH,
    ) -> "ComfyVideoProvider":
        specs = load_model_specs(config_path)
        if model_id not in specs:
            raise KeyError(f"Unknown video model_id {model_id!r}; available: {sorted(specs.keys())}")
        return cls(
            spec=specs[model_id],
            client=client or ComfyClient(),
            cache=cache,
        )

    def health(self) -> Dict[str, Any]:
        """Checks ComfyUI reachability and required custom node availability for this model."""
        if not self.spec.enabled:
            return {"status": "disabled", "ready": False, "reason": "Model disabled in config"}
        try:
            probe_res = self.client.probe(required_nodes=self.spec.required_nodes)
            return {"status": "ready", "ready": True, "vram_gb": probe_res.get("vram_gb", 0.0)}
        except Exception as exc:
            return {"status": f"unavailable: {exc}", "ready": False}

    def generate(
        self,
        panel: Dict[str, Any],
        clean_plate: Path,
        duration_s: float,
        seed: int,
        out_dir: Path,
        understanding_panel: Optional[Dict[str, Any]] = None,
        target_res: Tuple[int, int] = (540, 960),
        fps: int = 24,
        cancel_event: Optional[Any] = None,
    ) -> GenResult:
        """Executes end-to-end I2V generation + conform for a panel."""
        clean_plate = Path(clean_plate)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        panel_id = str(panel.get("panel_id") or clean_plate.stem)

        pos_prompt, neg_prompt = build_prompt(panel, understanding_panel, self.spec)
        from PIL import Image
        from workers.render_kenburns import _trim_outer_white_gutters
        src_trimmed = _trim_outer_white_gutters(Image.open(clean_plate).convert("RGB"))
        native_w, native_h = pick_bucket(src_trimmed.width, src_trimmed.height, self.spec.resolution_buckets)
        native_frames = frames_for(duration_s, self.spec)

        prepared_input = prepare_input(
            clean_plate=clean_plate,
            work_dir=out_dir / "_inputs",
            target_w=native_w,
            target_h=native_h,
        )
        img_bytes = prepared_input.read_bytes()

        cache_key = compute_cache_key(
            model_id=self.spec.model_id,
            workflow_template=self.template,
            image_bytes=img_bytes,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            seed=seed,
            width=native_w,
            height=native_h,
            frames=native_frames,
            steps=self.spec.steps,
            cfg=self.spec.cfg,
        )

        final_clip_path = out_dir / f"{panel_id}_{self.spec.model_id}_s{seed}.avi"
        cached_meta = self.cache.get(cache_key, final_clip_path)
        if cached_meta is not None and final_clip_path.is_file():
            from workers.assembler import decode_video_frames
            expected_frames = max(1, int(round(float(duration_s) * int(fps))))
            cached_frames = decode_video_frames(final_clip_path)
            if not cached_frames or len(cached_frames) != expected_frames or cached_frames[0].size != (int(target_res[0]), int(target_res[1])):
                conform(
                    raw_clip=final_clip_path,
                    clean_plate=clean_plate,
                    duration_s=duration_s,
                    out_path=final_clip_path,
                    fps=fps,
                    width=target_res[0],
                    height=target_res[1],
                    native_fps=self.spec.native_fps,
                )
            return GenResult(
                clip_path=final_clip_path,
                raw_clip_path=final_clip_path,
                model_id=self.spec.model_id,
                seed=int(seed),
                positive_prompt=pos_prompt,
                negative_prompt=neg_prompt,
                native_width=native_w,
                native_height=native_h,
                native_frames=native_frames,
                native_fps=self.spec.native_fps,
                gen_seconds=0.0,
                peak_vram_gb=float(cached_meta.get("peak_vram_gb", 0.0)),
                cached=True,
            )

        t0 = time.monotonic()
        uploaded_name = self.client.upload_image(prepared_input)
        wf = inject(
            self.template,
            image_name=uploaded_name,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            seed=seed,
            width=native_w,
            height=native_h,
            frames=native_frames,
            steps=self.spec.steps,
            cfg=self.spec.cfg,
            fps=self.spec.native_fps,
            filename_prefix=f"m2a_{panel_id}_{self.spec.model_id}",
            sampler=self.spec.sampler,
            scheduler=self.spec.scheduler,
        )

        raw_dir = out_dir / "_raw"
        raw_clip_path, run_meta = self.client.submit_and_wait(
            workflow=wf,
            output_dir=raw_dir,
            cancel_event=cancel_event,
        )
        gen_seconds = round(time.monotonic() - t0, 3)
        self.client.free_memory(unload_models=False, free_memory=True)

        conform(
            raw_clip=raw_clip_path,
            clean_plate=clean_plate,
            duration_s=duration_s,
            out_path=final_clip_path,
            fps=fps,
            width=target_res[0],
            height=target_res[1],
            native_fps=self.spec.native_fps,
        )

        peak_vram = float(run_meta.get("peak_vram_gb", 0.0))
        self.cache.put(
            cache_key,
            final_clip_path,
            {
                "model_id": self.spec.model_id,
                "seed": int(seed),
                "positive_prompt": pos_prompt,
                "negative_prompt": neg_prompt,
                "native_width": native_w,
                "native_height": native_h,
                "native_frames": native_frames,
                "native_fps": self.spec.native_fps,
                "gen_seconds": gen_seconds,
                "peak_vram_gb": peak_vram,
            },
        )

        return GenResult(
            clip_path=final_clip_path,
            raw_clip_path=raw_clip_path,
            model_id=self.spec.model_id,
            seed=int(seed),
            positive_prompt=pos_prompt,
            negative_prompt=neg_prompt,
            native_width=native_w,
            native_height=native_h,
            native_frames=native_frames,
            native_fps=self.spec.native_fps,
            gen_seconds=gen_seconds,
            peak_vram_gb=peak_vram,
            cached=bool(run_meta.get("cached", False)),
        )
