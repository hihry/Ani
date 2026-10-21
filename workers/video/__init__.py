"""Phase 3 ComfyUI I2V video generation, workflow injection, caching, and conform package."""
from workers.video.cache import ClipCache, compute_cache_key
from workers.video.comfy_client import (
    ComfyClient,
    ComfyExecutionError,
    ComfyNodeMissing,
    ComfyOOM,
    ComfyTimeout,
    ComfyUnavailable,
)
from workers.video.conform import conform, frames_for, pick_bucket, prepare_input
from workers.video.prompts import build_prompt
from workers.video.provider import ComfyVideoProvider, GenResult, ModelSpec, load_model_specs
from workers.video.workflows import REQUIRED_TAGS, inject, load_template

__all__ = [
    "ClipCache",
    "compute_cache_key",
    "ComfyClient",
    "ComfyExecutionError",
    "ComfyNodeMissing",
    "ComfyOOM",
    "ComfyTimeout",
    "ComfyUnavailable",
    "conform",
    "frames_for",
    "pick_bucket",
    "prepare_input",
    "build_prompt",
    "ComfyVideoProvider",
    "GenResult",
    "ModelSpec",
    "load_model_specs",
    "REQUIRED_TAGS",
    "inject",
    "load_template",
]
