"""Provider registry and exports."""
import logging
from typing import Type

from .base import (
    VideoProvider, VLMProvider, OCRProvider, TTSProvider, InpaintProvider, PerceptionProvider
)

logger = logging.getLogger(__name__)

# Registries
_VIDEO_PROVIDERS: dict[str, Type[VideoProvider]] = {}
_VLM_PROVIDERS: dict[str, Type[VLMProvider]] = {}
_OCR_PROVIDERS: dict[str, Type[OCRProvider]] = {}
_TTS_PROVIDERS: dict[str, Type[TTSProvider]] = {}
_INPAINT_PROVIDERS: dict[str, Type[InpaintProvider]] = {}
_PERCEPTION_PROVIDERS: dict[str, Type[PerceptionProvider]] = {}

def get_video_provider(name: str, **kwargs) -> VideoProvider:
    if name not in _VIDEO_PROVIDERS:
        raise ValueError(f"VideoProvider '{name}' not found.")
    return _VIDEO_PROVIDERS[name](**kwargs)

def get_vlm_provider(name: str, **kwargs) -> VLMProvider:
    if name not in _VLM_PROVIDERS:
        raise ValueError(f"VLMProvider '{name}' not found.")
    return _VLM_PROVIDERS[name](**kwargs)

def get_ocr_provider(name: str, **kwargs) -> OCRProvider:
    if name not in _OCR_PROVIDERS:
        raise ValueError(f"OCRProvider '{name}' not found.")
    return _OCR_PROVIDERS[name](**kwargs)

def get_tts_provider(name: str, **kwargs) -> TTSProvider:
    if name not in _TTS_PROVIDERS:
        raise ValueError(f"TTSProvider '{name}' not found.")
    return _TTS_PROVIDERS[name](**kwargs)

def get_inpainter(name: str, **kwargs) -> InpaintProvider:
    if name not in _INPAINT_PROVIDERS:
        raise ValueError(f"InpaintProvider '{name}' not found.")
    return _INPAINT_PROVIDERS[name](**kwargs)

def get_perception_provider(name: str, **kwargs) -> PerceptionProvider:
    if name not in _PERCEPTION_PROVIDERS:
        raise ValueError(f"PerceptionProvider '{name}' not found.")
    return _PERCEPTION_PROVIDERS[name](**kwargs)

# TODO: Import and register providers here as they are implemented.
# e.g. from .video.wan22 import Wan22Provider
