"""Kokoro TTS Provider."""
import logging
import time
from pathlib import Path

from ..base import TTSProvider

logger = logging.getLogger(__name__)

class KokoroProvider(TTSProvider):
    """Lightweight TTS provider with CPU fallback."""
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        logger.info(f"Initializing Kokoro TTS (GPU={use_gpu})")
        # TODO: Load Kokoro voices mapping and model

    def synthesize(self, text: str, voice_id: str, emotion: str, output_path: Path) -> float:
        start_time = time.time()
        # TODO: Implement Kokoro synthesis
        duration = 0.0
        elapsed = time.time() - start_time
        logger.info(f"Kokoro synthesis took {elapsed:.2f}s")
        return duration

    def clone_voice(self, reference_audio: Path) -> str:
        logger.warning("Kokoro TTS does not support zero-shot cloning. Returning default.")
        return "default_kokoro_voice"
