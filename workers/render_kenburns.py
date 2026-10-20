"""Ken Burns pan-and-zoom motion generator for 9:16 portrait video.

Given a panel image, prepares an aspect-ratio-safe 9:16 canvas (using a cinematic
blurred-fill letterbox for wide/landscape panels so artwork is never cropped out,
addressing Proposal Problem P10) and renders smooth cubic-eased camera trajectories
with sub-pixel bilinear sampling (eliminating 1px stair-step jitter).
"""
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger("m2a.kenburns")


def cubic_ease_in_out(t: float) -> float:
    """Smooth cubic acceleration and deceleration curve in [0.0, 1.0]."""
    t = max(0.0, min(1.0, float(t)))
    if t < 0.5:
        return 4.0 * t * t * t
    p = 2.0 * t - 2.0
    return 0.5 * p * p * p + 1.0


def prepare_cinematic_canvas(
    img: Image.Image,
    target_res: Tuple[int, int] = (720, 1280),
    overscan: float = 1.18,
    preserve_tall: bool = False,
) -> Image.Image:
    """Composes an arbitrary-aspect-ratio manhwa panel onto a 9:16 overscanned canvas.

    Addresses Architectural Problem P10 (Resolution & Aspect Mismatch):
      - Very tall vertical-scroll panels (when preserve_tall=True and h/w > 1.6 * 16/9)
        retain full height so vertical camera scroll reveals the entire strip.
      - Tall/vertical panels (h/w >= 1.25) cover-fit the 9:16 frame directly.
      - Wide/medium panels (h/w < 1.25) are placed intact over a darkened,
        Gaussian-blurred background fill so 100% of horizontal artwork, characters,
        and action remain visible instead of losing 70%+ of their width to center-cropping.
    """
    src_w, src_h = img.size
    dst_w, dst_h = target_res
    canvas_w = max(1, int(round(dst_w * overscan)))
    canvas_h = max(1, int(round(dst_h * overscan)))

    src_aspect = src_h / float(max(1, src_w))
    dst_aspect = canvas_h / float(max(1, canvas_w))

    if preserve_tall and src_aspect > dst_aspect * 1.6:
        scale = canvas_w / float(max(1, src_w))
        tall_h = max(canvas_h, int(round(src_h * scale)))
        return img.resize((canvas_w, tall_h), Image.Resampling.LANCZOS)

    if src_aspect >= dst_aspect * 0.78:
        cover_scale = max(canvas_w / float(src_w), canvas_h / float(src_h))
        sw = max(canvas_w, int(round(src_w * cover_scale)))
        sh = max(canvas_h, int(round(src_h * cover_scale)))
        resized = img.resize((sw, sh), Image.Resampling.LANCZOS)
        left = (sw - canvas_w) // 2
        top = (sh - canvas_h) // 2
        return resized.crop((left, top, left + canvas_w, top + canvas_h))

    bg_scale = max(canvas_w / float(src_w), canvas_h / float(src_h))
    bg_w = max(canvas_w, int(round(src_w * bg_scale)))
    bg_h = max(canvas_h, int(round(src_h * bg_scale)))
    bg_img = img.resize((bg_w, bg_h), Image.Resampling.BILINEAR)
    bg_left = (bg_w - canvas_w) // 2
    bg_top = (bg_h - canvas_h) // 2
    bg_crop = bg_img.crop((bg_left, bg_top, bg_left + canvas_w, bg_top + canvas_h))
    bg_crop = bg_crop.filter(ImageFilter.GaussianBlur(radius=18))
    bg_crop = ImageEnhance.Brightness(bg_crop).enhance(0.45)

    fg_w = int(round(canvas_w * 0.94))
    fg_scale = fg_w / float(max(1, src_w))
    fg_h = max(1, int(round(src_h * fg_scale)))
    if fg_h > int(canvas_h * 0.92):
        fg_h = int(round(canvas_h * 0.92))
        fg_scale = fg_h / float(max(1, src_h))
        fg_w = max(1, int(round(src_w * fg_scale)))

    fg_img = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)
    paste_x = (canvas_w - fg_w) // 2
    paste_y = (canvas_h - fg_h) // 2

    bg_crop.paste(fg_img, (paste_x, paste_y))
    draw = ImageDraw.Draw(bg_crop)
    draw.rectangle(
        [paste_x - 2, paste_y - 2, paste_x + fg_w + 1, paste_y + fg_h + 1],
        outline=(20, 20, 28),
        width=2,
    )
    return bg_crop


def generate_kenburns_frames(
    image_path: Path,
    target_res: Tuple[int, int] = (720, 1280),
    duration_s: float = 3.0,
    fps: int = 24,
    motion_type: str = "zoom_in",
) -> List[Image.Image]:
    """Generates a sequence of PIL images with sub-pixel Ken Burns camera motion."""
    img = Image.open(image_path).convert("RGB")
    dst_w, dst_h = target_res

    total_frames = max(1, int(round(duration_s * fps)))
    frames: List[Image.Image] = []

    overscan = 1.20
    scaled_img = prepare_cinematic_canvas(
        img, target_res=target_res, overscan=overscan, preserve_tall=True
    )
    scaled_w, scaled_h = scaled_img.size
    base_canvas_h = max(1, int(round(dst_h * overscan)))

    cx = scaled_w / 2.0
    cy = scaled_h / 2.0

    if scaled_h > int(base_canvas_h * 1.15):
        # Tall vertical-scroll strip: smoothly pan down across full panel height
        zoom_start, zoom_end = 1.04, 1.06
        view_h_est = base_canvas_h / 1.05
        max_dy = max(0.0, (scaled_h - view_h_est) * 0.45)
        dx_start, dy_start = 0.0, -max_dy
        dx_end, dy_end = 0.0, max_dy
    elif motion_type == "zoom_in":
        zoom_start, zoom_end = 1.02, 1.18
        dx_start, dy_start = 0.0, 0.0
        dx_end, dy_end = 0.0, -scaled_h * 0.02
    elif motion_type == "zoom_out":
        zoom_start, zoom_end = 1.18, 1.02
        dx_start, dy_start = 0.0, -scaled_h * 0.02
        dx_end, dy_end = 0.0, 0.0
    elif motion_type == "pan_down":
        zoom_start, zoom_end = 1.12, 1.14
        dx_start, dy_start = 0.0, -scaled_h * 0.045
        dx_end, dy_end = 0.0, scaled_h * 0.045
    elif motion_type == "pan_right":
        zoom_start, zoom_end = 1.12, 1.14
        dx_start, dy_start = -scaled_w * 0.045, 0.0
        dx_end, dy_end = scaled_w * 0.045, 0.0
    else:
        zoom_start, zoom_end = 1.14, 1.12
        dx_start, dy_start = 0.0, scaled_h * 0.045
        dx_end, dy_end = 0.0, -scaled_h * 0.045

    for frame_idx in range(total_frames):
        norm_t = frame_idx / max(1, total_frames - 1)
        eased_t = cubic_ease_in_out(norm_t)

        current_zoom = zoom_start + (zoom_end - zoom_start) * eased_t
        current_dx = dx_start + (dx_end - dx_start) * eased_t
        current_dy = dy_start + (dy_end - dy_start) * eased_t

        view_w = scaled_w / current_zoom
        view_h = min(float(scaled_h), base_canvas_h / current_zoom)

        view_cx = cx + current_dx
        view_cy = cy + current_dy

        left = max(0.0, min(scaled_w - view_w, view_cx - view_w / 2.0))
        top = max(0.0, min(scaled_h - view_h, view_cy - view_h / 2.0))
        right = left + view_w
        bottom = top + view_h

        frame = scaled_img.resize(
            (dst_w, dst_h),
            Image.Resampling.BILINEAR,
            box=(left, top, right, bottom),
        )
        frames.append(frame)

    logger.info(
        f"Rendered {len(frames)} Ken Burns ({motion_type}) frames for {image_path} ({duration_s}s @ {fps}fps)"
    )
    return frames
