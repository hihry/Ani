"""LaMa Inpainting Provider."""
import logging
import time
from pathlib import Path
from typing import Optional

from ..base import InpaintProvider
from workers.assembler import inpaint_panel_bubbles

logger = logging.getLogger(__name__)


class LaMaProvider(InpaintProvider):
    """LaMa / Telea text mask inpainting provider."""

    def __init__(self, use_gpu: bool = True) -> None:
        self.use_gpu = use_gpu
        logger.info(f"Initializing LaMa inpainting provider (GPU={use_gpu})...")

    def inpaint(self, image: Path, mask: Optional[Path], output: Path) -> Path:
        start_time = time.time()
        out_path = inpaint_panel_bubbles(
            image_path=Path(image),
            output_path=Path(output),
            mask_path=Path(mask) if mask else None,
        )
        logger.info(f"Inpainting {Path(image).name} completed in {time.time() - start_time:.2f}s")
        return out_path
