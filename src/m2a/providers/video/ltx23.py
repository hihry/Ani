"""LTX-2.3 Video Provider."""
import logging
import time
from pathlib import Path

from ..base import VideoProvider

logger = logging.getLogger(__name__)

class LTX23Provider(VideoProvider):
    """LTX-2.3 provider."""
    def __init__(self, api_url: str = "http://localhost:8188"):
        self.api_url = api_url
        logger.info("Initialized LTX23Provider")

    def generate_clip(self, image: Path, prompt: str, duration_s: float, seed: int, **kwargs) -> Path:
        start_time = time.time()
        # TODO: Implement LTX-2.3 generation logic
        output_path = Path(f"output_ltx_{seed}.mp4")
        gpu_seconds = time.time() - start_time
        logger.info(f"LTX-2.3 GPU seconds: {gpu_seconds:.2f}s")
        return output_path

    def estimate_vram_gb(self) -> float:
        return 12.0

    def model_name(self) -> str:
        return "LTX-2.3"
