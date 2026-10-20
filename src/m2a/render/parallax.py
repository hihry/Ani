"""2.5D Parallax rendering delegating to workers.render_parallax."""
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

from workers.render_parallax import generate_parallax_frames

logger = logging.getLogger(__name__)


def apply_parallax(
    image_path: Path,
    depth_map_path: Optional[Path] = None,
    duration: float = 3.0,
    fps: int = 24,
    target_res: tuple[int, int] = (720, 1280),
) -> List[np.ndarray]:
    """Applies a 2.5D parallax effect using salience/depth masking and returns RGB numpy frames."""
    logger.info(f"Applying parallax to {image_path}")
    pil_frames = generate_parallax_frames(
        image_path=Path(image_path),
        target_res=target_res,
        duration_s=duration,
        fps=fps,
    )
    return [np.asarray(f, dtype=np.uint8) for f in pil_frames]
