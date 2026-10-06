"""Video transition effects."""
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

def crossfade(clip_a: List[np.ndarray], clip_b: List[np.ndarray], duration_frames: int) -> List[np.ndarray]:
    """Applies a crossfade transition between two clips."""
    logger.info(f"Applying crossfade transition for {duration_frames} frames")
    if len(clip_a) < duration_frames or len(clip_b) < duration_frames:
        raise ValueError("Clips must be longer than duration_frames")
        
    result = clip_a[:-duration_frames]
    
    # TODO: Implement actual frame blending
    for i in range(duration_frames):
        alpha = (i + 1) / (duration_frames + 1)
        frame_a = clip_a[-duration_frames + i]
        frame_b = clip_b[i]
        
        # blended = cv2.addWeighted(frame_a, 1 - alpha, frame_b, alpha, 0)
        blended = frame_a # Stub
        result.append(blended)
        
    result.extend(clip_b[duration_frames:])
    return result

def hard_cut(clip_a: List[np.ndarray], clip_b: List[np.ndarray]) -> List[np.ndarray]:
    """Concatenates two clips with a hard cut."""
    logger.info("Applying hard cut transition")
    return clip_a + clip_b

def fade_to_black(clip: List[np.ndarray], duration_frames: int) -> List[np.ndarray]:
    """Applies a fade to black at the end of the clip."""
    logger.info(f"Applying fade to black for {duration_frames} frames")
    if len(clip) < duration_frames:
        raise ValueError("Clip must be longer than duration_frames")
        
    result = clip[:-duration_frames]
    
    # TODO: Implement actual darkening of frames
    for i in range(duration_frames):
        alpha = 1.0 - (i + 1) / (duration_frames + 1)
        frame = clip[-duration_frames + i]
        # darkened = (frame * alpha).astype(np.uint8)
        darkened = frame # Stub
        result.append(darkened)
        
    return result

def apply_transition(clip_a: List[np.ndarray], clip_b: List[np.ndarray], transition_type: str = "hard_cut", **kwargs) -> List[np.ndarray]:
    """Applies the specified transition between two clips."""
    if transition_type == "crossfade":
        return crossfade(clip_a, clip_b, kwargs.get("duration_frames", 12))
    elif transition_type == "hard_cut":
        return hard_cut(clip_a, clip_b)
    else:
        logger.warning(f"Unknown transition type {transition_type}, falling back to hard cut")
        return hard_cut(clip_a, clip_b)
