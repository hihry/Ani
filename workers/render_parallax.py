"""2.5D parallax layer animation renderer.

Simulates depth by separating visual content into layers and animating foreground
and background with differential velocity (foreground moves faster).
"""
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageFilter
import numpy as np
import logging
from render_kenburns import cubic_ease_in_out

logger = logging.getLogger("m2a.parallax")

def generate_depth_mask(img: Image.Image) -> Image.Image:
    """Generates a soft foreground depth mask based on visual salience and central focus."""
    w, h = img.size
    # Center-weighted Gaussian oval mask
    y_coords, x_coords = np.ogrid[:h, :w]
    cy, cx = h * 0.55, w * 0.5
    dist = ((x_coords - cx) / (w * 0.38)) ** 2 + ((y_coords - cy) / (h * 0.38)) ** 2
    mask_arr = np.clip(1.0 - dist * 0.6, 0.0, 1.0)
    
    # Combine with image edge salience
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    grad_y, grad_x = np.gradient(gray)
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_norm = np.clip(edge_mag * 4.0, 0.0, 1.0)
    
    combined = np.clip(mask_arr * 0.7 + edge_norm * 0.3, 0.0, 1.0)
    mask_uint8 = (combined * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode="L").filter(ImageFilter.GaussianBlur(radius=8))

def generate_parallax_frames(
    image_path: Path,
    target_res: Tuple[int, int] = (720, 1280),
    duration_s: float = 3.0,
    fps: int = 24,
) -> List[Image.Image]:
    """Renders 2.5D parallax frames with differential layer motion."""
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    dst_w, dst_h = target_res
    
    total_frames = max(1, int(duration_s * fps))
    frames = []
    
    # Scale base image to cover target with margin for shifting
    scale = max(dst_w / src_w, dst_h / src_h) * 1.2
    scaled_w = int(src_w * scale)
    scaled_h = int(src_h * scale)
    scaled_img = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    
    # Generate soft depth mask
    mask = generate_depth_mask(scaled_img)
    
    # Create slightly blurred background and crisp foreground
    bg_img = scaled_img.filter(ImageFilter.GaussianBlur(radius=2))
    fg_img = scaled_img
    
    cx = (scaled_w - dst_w) / 2.0
    cy = (scaled_h - dst_h) / 2.0
    
    max_shift_x = dst_w * 0.04
    max_shift_y = dst_h * 0.03
    
    for frame_idx in range(total_frames):
        norm_t = frame_idx / max(1, total_frames - 1)
        eased_t = cubic_ease_in_out(norm_t) - 0.5 # centered around 0
        
        # Background shifts slowly in one direction
        bg_dx = cx - max_shift_x * 0.4 * eased_t
        bg_dy = cy - max_shift_y * 0.4 * eased_t
        bg_crop = bg_img.crop((int(bg_dx), int(bg_dy), int(bg_dx + dst_w), int(bg_dy + dst_h)))
        
        # Foreground shifts faster in opposite direction (parallax)
        fg_dx = cx + max_shift_x * 0.8 * eased_t
        fg_dy = cy + max_shift_y * 0.8 * eased_t
        fg_crop = fg_img.crop((int(fg_dx), int(fg_dy), int(fg_dx + dst_w), int(fg_dy + dst_h)))
        mask_crop = mask.crop((int(fg_dx), int(fg_dy), int(fg_dx + dst_w), int(fg_dy + dst_h)))
        
        # Composite foreground over background using mask
        composite = Image.composite(fg_crop, bg_crop, mask_crop)
        frames.append(composite)
        
    logger.info(f"Rendered {len(frames)} Parallax frames for {image_path} ({duration_s}s @ {fps}fps)")
    return frames
