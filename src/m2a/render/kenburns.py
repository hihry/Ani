"""Ken Burns effect rendering delegating to workers.render_kenburns."""
import logging
from pathlib import Path
from typing import List
import numpy as np

from workers.render_kenburns import generate_kenburns_frames

logger = logging.getLogger(__name__)


def apply_ken_burns(
    image_path: Path,
    duration: float = 3.0,
    fps: int = 24,
    zoom_range: tuple = (1.0, 1.18),
    pan_direction: str = "zoom_in",
    target_res: tuple[int, int] = (720, 1280),
) -> List[np.ndarray]:
    """Applies a Ken Burns effect to a panel image and returns RGB numpy frames."""
    logger.info(f"Applying Ken Burns to {image_path}")
    motion_map = {
        "in": "zoom_in",
        "out": "zoom_out",
        "down": "pan_down",
        "up": "pan_up",
        "right": "pan_right",
        "random": "zoom_in",
    }
    motion_type = motion_map.get(pan_direction, pan_direction)
    pil_frames = generate_kenburns_frames(
        image_path=Path(image_path),
        target_res=target_res,
        duration_s=duration,
        fps=fps,
        motion_type=motion_type,
    )
    return [np.asarray(f, dtype=np.uint8) for f in pil_frames]
