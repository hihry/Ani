"""Ken Burns pan-and-zoom motion generator for 9:16 portrait video.

Given a panel image, calculates smooth cinematic camera trajectory (zoom + pan)
using cubic easing and renders individual frames.
"""
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger("m2a.kenburns")

def cubic_ease_in_out(t: float) -> float:
    """Smooth cubic acceleration and deceleration curve."""
    if t < 0.5:
        return 4.0 * t * t * t
    else:
        p = 2.0 * t - 2.0
        return 0.5 * p * p * p + 1.0

def generate_kenburns_frames(
    image_path: Path,
    target_res: Tuple[int, int] = (720, 1280), # [width, height] 9:16
    duration_s: float = 3.0,
    fps: int = 24,
    motion_type: str = "zoom_in", # "zoom_in", "zoom_out", "pan_down", "pan_up"
) -> List[Image.Image]:
    """Generates a sequence of PIL images with Ken Burns camera motion.
    
    Fits the panel image into target_res using a cinematic letterbox/crop
    and applies smooth pan/zoom interpolation.
    """
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    dst_w, dst_h = target_res
    
    total_frames = max(1, int(duration_s * fps))
    frames = []
    
    # Base scale to cover or letterbox
    scale = max(dst_w / src_w, dst_h / src_h) * 1.15
    scaled_w = int(src_w * scale)
    scaled_h = int(src_h * scale)
    scaled_img = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    
    # Center coordinates on scaled canvas
    cx = scaled_w / 2.0
    cy = scaled_h / 2.0
    
    # Motion parameter endpoints
    if motion_type == "zoom_in":
        zoom_start, zoom_end = 1.0, 1.18
        dx_start, dy_start = 0.0, 0.0
        dx_end, dy_end = 0.0, scaled_h * 0.03
    elif motion_type == "zoom_out":
        zoom_start, zoom_end = 1.18, 1.0
        dx_start, dy_start = 0.0, scaled_h * 0.03
        dx_end, dy_end = 0.0, 0.0
    elif motion_type == "pan_down":
        zoom_start, zoom_end = 1.05, 1.08
        dx_start, dy_start = 0.0, -scaled_h * 0.05
        dx_end, dy_end = 0.0, scaled_h * 0.05
    else: # pan_up
        zoom_start, zoom_end = 1.08, 1.05
        dx_start, dy_start = 0.0, scaled_h * 0.05
        dx_end, dy_end = 0.0, -scaled_h * 0.05
        
    for frame_idx in range(total_frames):
        norm_t = frame_idx / max(1, total_frames - 1)
        eased_t = cubic_ease_in_out(norm_t)
        
        current_zoom = zoom_start + (zoom_end - zoom_start) * eased_t
        current_dx = dx_start + (dx_end - dx_start) * eased_t
        current_dy = dy_start + (dy_end - dy_start) * eased_t
        
        # Viewport dimensions on scaled image
        view_w = dst_w / current_zoom
        view_h = dst_h / current_zoom
        
        view_cx = cx + current_dx
        view_cy = cy + current_dy
        
        left = max(0, min(scaled_w - view_w, view_cx - view_w / 2.0))
        top = max(0, min(scaled_h - view_h, view_cy - view_h / 2.0))
        right = left + view_w
        bottom = top + view_h
        
        crop_box = (int(left), int(top), int(right), int(bottom))
        frame = scaled_img.crop(crop_box).resize((dst_w, dst_h), Image.Resampling.BILINEAR)
        frames.append(frame)
        
    logger.info(f"Rendered {len(frames)} Ken Burns frames for {image_path} ({duration_s}s @ {fps}fps)")
    return frames
