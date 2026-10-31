"""2.5D parallax layer animation renderer.

Simulates depth by separating visual content into foreground and background layers,
filling the occluded background region under the dilated foreground mask to prevent
double-image ghosting, and animating layers with differential velocity.
"""
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageFilter
import numpy as np
import logging

try:
    from workers.render_kenburns import cubic_ease_in_out, prepare_cinematic_canvas
except ImportError:
    from render_kenburns import cubic_ease_in_out, prepare_cinematic_canvas

try:
    from workers.stage_3d import generate_3d_stage_parallax_frames
except ImportError:
    from stage_3d import generate_3d_stage_parallax_frames

logger = logging.getLogger("m2a.parallax")


def generate_depth_mask(img: Image.Image) -> Image.Image:
    """Generates a soft foreground depth mask based on visual salience and central focus."""
    w, h = img.size
    y_coords, x_coords = np.ogrid[:h, :w]
    cy, cx = h * 0.52, w * 0.5
    dist = ((x_coords - cx) / (w * 0.36)) ** 2 + ((y_coords - cy) / (h * 0.36)) ** 2
    mask_arr = np.clip(1.0 - dist * 0.65, 0.0, 1.0)

    # Combine with image edge salience
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    grad_y, grad_x = np.gradient(gray)
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_norm = np.clip(edge_mag * 4.0, 0.0, 1.0)

    combined = np.clip(mask_arr * 0.72 + edge_norm * 0.28, 0.0, 1.0)
    mask_uint8 = (combined * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode="L").filter(ImageFilter.GaussianBlur(radius=10))


def generate_parallax_frames(
    image_path: Path,
    target_res: Tuple[int, int] = (720, 1280),
    duration_s: float = 3.0,
    fps: int = 24,
    genre_id: str = "shonen",
    motion_primitive: str = "dolly_zoom_parallax",
    enable_3d_stage: bool = True,
) -> List[Image.Image]:
    """Renders 2.5D/3D multi-plane parallax frames with character cutout, conditional 2D background,
    depth relighting, and differential 4-layer camera motion.
    """
    if enable_3d_stage:
        return generate_3d_stage_parallax_frames(
            image_path=Path(image_path),
            target_res=target_res,
            duration_s=duration_s,
            fps=fps,
            genre_id=genre_id,
            motion_primitive=motion_primitive,
        )

    img = Image.open(image_path).convert("RGB")
    dst_w, dst_h = target_res

    total_frames = max(1, int(round(duration_s * fps)))
    frames: List[Image.Image] = []

    # Prepare aspect-ratio-safe overscanned base canvas (fixes wide-panel crop loss)
    scaled_img = prepare_cinematic_canvas(img, target_res=target_res, overscan=1.20)
    scaled_w, scaled_h = scaled_img.size

    # Generate soft foreground depth mask
    fg_mask = generate_depth_mask(scaled_img)

    # Fill occluded foreground region on background layer to prevent double-silhouette ghosting
    dilated_mask = fg_mask.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(radius=12))
    heavily_blurred_bg = scaled_img.filter(ImageFilter.GaussianBlur(radius=14))
    soft_bg = scaled_img.filter(ImageFilter.GaussianBlur(radius=2))
    bg_img = Image.composite(heavily_blurred_bg, soft_bg, dilated_mask)
    fg_img = scaled_img

    cx = (scaled_w - dst_w) / 2.0
    cy = (scaled_h - dst_h) / 2.0

    max_shift_x = dst_w * 0.035
    max_shift_y = dst_h * 0.025

    for frame_idx in range(total_frames):
        norm_t = frame_idx / max(1, total_frames - 1)
        eased_t = cubic_ease_in_out(norm_t) - 0.5  # centered in [-0.5, 0.5]

        # Background shifts slowly in one direction (sub-pixel bilinear sampling)
        bg_dx = max(0.0, min(float(scaled_w - dst_w), cx - max_shift_x * 0.45 * eased_t))
        bg_dy = max(0.0, min(float(scaled_h - dst_h), cy - max_shift_y * 0.45 * eased_t))
        bg_box = (bg_dx, bg_dy, bg_dx + dst_w, bg_dy + dst_h)
        bg_crop = bg_img.resize((dst_w, dst_h), Image.Resampling.BILINEAR, box=bg_box)

        # Foreground shifts faster in opposite direction (parallax)
        fg_dx = max(0.0, min(float(scaled_w - dst_w), cx + max_shift_x * 0.85 * eased_t))
        fg_dy = max(0.0, min(float(scaled_h - dst_h), cy + max_shift_y * 0.85 * eased_t))
        fg_box = (fg_dx, fg_dy, fg_dx + dst_w, fg_dy + dst_h)
        fg_crop = fg_img.resize((dst_w, dst_h), Image.Resampling.BILINEAR, box=fg_box)
        mask_crop = fg_mask.resize((dst_w, dst_h), Image.Resampling.BILINEAR, box=fg_box)

        composite = Image.composite(fg_crop, bg_crop, mask_crop)
        frames.append(composite)

    logger.info(
        f"Rendered {len(frames)} Parallax frames for {image_path} ({duration_s}s @ {fps}fps)"
    )
    return frames
