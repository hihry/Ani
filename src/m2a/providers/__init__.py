"""Provider registry and exports."""
import logging
from typing import Type

from .base import (
    InpaintProvider,
    OCRProvider,
    PerceptionProvider,
    TTSProvider,
    VideoProvider,
    VLMProvider,
)
from .inpaint.lama import LaMaProvider
from .ocr.manga_ocr import MangaOCRProvider
from .ocr.paddle_ocr import PaddleOCRProvider
from .ocr.qwen_ocr import QwenOCRProvider
from .perception.magiv3 import Magiv3Provider
from .tts.chatterbox import ChatterboxProvider
from .tts.kokoro import KokoroProvider
from .video.hunyuan15 import Hunyuan15Provider
from .video.ltx23 import LTX23Provider
from .video.wan22 import Wan22Provider
from .vlm.qwen3vl import Qwen3VLProvider

logger = logging.getLogger(__name__)

# Registries populated with concrete provider implementations
_VIDEO_PROVIDERS: dict[str, Type[VideoProvider]] = {
    "wan22": Wan22Provider,
    "wan22-ti2v-5b": Wan22Provider,
    "wan22-i2v-a14b": Wan22Provider,
    "hunyuan15": Hunyuan15Provider,
    "ltx23": LTX23Provider,
}
_VLM_PROVIDERS: dict[str, Type[VLMProvider]] = {
    "qwen3-vl": Qwen3VLProvider,
    "qwen3-vl-8b": Qwen3VLProvider,
    "qwen3-vl-8b-4bit": Qwen3VLProvider,
    "qwen3-vl-2b": Qwen3VLProvider,
}
_OCR_PROVIDERS: dict[str, Type[OCRProvider]] = {
    "qwen": QwenOCRProvider,
    "manga_ocr": MangaOCRProvider,
    "paddle": PaddleOCRProvider,
}
_TTS_PROVIDERS: dict[str, Type[TTSProvider]] = {
    "chatterbox": ChatterboxProvider,
    "kokoro": KokoroProvider,
    "melotts": KokoroProvider,
}
_INPAINT_PROVIDERS: dict[str, Type[InpaintProvider]] = {
    "lama": LaMaProvider,
}
_PERCEPTION_PROVIDERS: dict[str, Type[PerceptionProvider]] = {
    "magiv3": Magiv3Provider,
}


def get_video_provider(name: str, **kwargs) -> VideoProvider:
    if name not in _VIDEO_PROVIDERS:
        raise ValueError(f"VideoProvider '{name}' not found. Available: {list(_VIDEO_PROVIDERS)}")
    return _VIDEO_PROVIDERS[name](**kwargs)


def get_vlm_provider(name: str, **kwargs) -> VLMProvider:
    if name not in _VLM_PROVIDERS:
        raise ValueError(f"VLMProvider '{name}' not found. Available: {list(_VLM_PROVIDERS)}")
    return _VLM_PROVIDERS[name](**kwargs)


def get_ocr_provider(name: str, **kwargs) -> OCRProvider:
    if name not in _OCR_PROVIDERS:
        raise ValueError(f"OCRProvider '{name}' not found. Available: {list(_OCR_PROVIDERS)}")
    return _OCR_PROVIDERS[name](**kwargs)


def get_tts_provider(name: str, **kwargs) -> TTSProvider:
    if name not in _TTS_PROVIDERS:
        raise ValueError(f"TTSProvider '{name}' not found. Available: {list(_TTS_PROVIDERS)}")
    return _TTS_PROVIDERS[name](**kwargs)


def get_inpainter(name: str, **kwargs) -> InpaintProvider:
    if name not in _INPAINT_PROVIDERS:
        raise ValueError(f"InpaintProvider '{name}' not found. Available: {list(_INPAINT_PROVIDERS)}")
    return _INPAINT_PROVIDERS[name](**kwargs)


def get_perception_provider(name: str, **kwargs) -> PerceptionProvider:
    if name not in _PERCEPTION_PROVIDERS:
        raise ValueError(f"PerceptionProvider '{name}' not found. Available: {list(_PERCEPTION_PROVIDERS)}")
    return _PERCEPTION_PROVIDERS[name](**kwargs)
