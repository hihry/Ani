"""LaMa Inpainting Provider."""
import logging
import time
from pathlib import Path

from ..base import InpaintProvider

logger = logging.getLogger(__name__)

class LaMaProvider(InpaintProvider):
    """LaMa text mask inpainting provider."""
    def __init__(self, use_gpu: bool = True):
        logger.info(f"Initializing LaMa model (GPU={use_gpu})...")
        # TODO: Load LaMa model

    def inpaint(self, image: Path, mask: Path, output: Path) -> Path:
        start_time = time.time()
        logger.info(f"Inpainting {image.name}")
        # TODO: Execute inpainting
        gpu_seconds = time.time() - start_time
        logger.info(f"Inpainting completed in {gpu_seconds:.2f}s")
        return output
