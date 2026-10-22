"""Ken Burns pan-and-zoom motion generator for 9:16 portrait video.

Given a panel image, prepares an aspect-ratio-safe 9:16 canvas (using a cinematic
blurred-fill letterbox for wide/landscape panels so artwork is never cropped out,
addressing Proposal Problem P10) and renders smooth cubic-eased camera trajectories
with sub-pixel bilinear sampling (eliminating 1px stair-step jitter).
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger("m2a.kenburns")


def _trim_outer_white_gutters(img: Image.Image) -> Image.Image:
    """Trims top/bottom webtoon white gutter bands on sliced panels (where artwork reaches the left edge
    and a protruding speech bubble left a white top/bottom margin above the horizontal panel border),
    while leaving full-page images (which have white side borders) completely untouched.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w, _ = arr.shape
    if h < 64 or w < 64:
        return img

    gray = arr.mean(axis=2)
    chroma = arr.max(axis=2).astype(np.int16) - arr.min(axis=2).astype(np.int16)
    is_white = (gray >= 240.0) & (chroma <= 18)

    # Full-page benchmark images (e.g. sample_page_1.png) have white side margins at x=0..30; do not crop them
    if float(is_white[:, :8].mean()) >= 0.65 and float(is_white[:, -8:].mean()) >= 0.65:
        return img

    if float((~is_white).mean()) < 0.45:
        return img

    row_white_ratio = is_white.mean(axis=1)
    top = 0
    max_top = int(h * 0.35)
    while top < max_top and row_white_ratio[top] >= 0.85:
        top += 1
    if top >= 20:
        border_scan = min(h - 16, top + 6)
        for r in range(top, border_scan):
            if gray[r].mean() < 95.0:
                top = r + 1
    else:
        top = 0

    bot = h
    min_bot = int(h * 0.65)
    while bot > min_bot and row_white_ratio[bot - 1] >= 0.85:
        bot -= 1
    if (h - bot) >= 20:
        border_scan = max(top + 16, bot - 6)
        for r in range(bot - 1, border_scan - 1, -1):
            if gray[r].mean() < 95.0:
                bot = r
    else:
        bot = h

    if (bot - top) >= int(h * 0.55) and (top > 0 or bot < h):
        return img.crop((0, top, w, bot))
    return img


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
    img = _trim_outer_white_gutters(img)
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

    # Build a natural top/bottom atmospheric extension by mirror-extending the top and bottom 28% of the panel
    fg_w = canvas_w
    fg_scale = canvas_w / float(max(1, src_w))
    fg_h = min(canvas_h, max(1, int(round(src_h * fg_scale))))
    fg_img = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)

    paste_x = 0
    paste_y = (canvas_h - fg_h) // 2

    bg_canvas = Image.new("RGB", (canvas_w, canvas_h))
    strip_h = max(16, fg_h // 3)
    if paste_y > 0:
        top_strip = fg_img.crop((0, 0, fg_w, strip_h)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        top_fill = top_strip.resize((canvas_w, paste_y + 24), Image.Resampling.BILINEAR)
        bg_canvas.paste(top_fill, (0, 0))
    if paste_y + fg_h < canvas_h:
        bot_need = canvas_h - (paste_y + fg_h)
        bot_strip = fg_img.crop((0, max(0, fg_h - strip_h), fg_w, fg_h)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        bot_fill = bot_strip.resize((canvas_w, bot_need + 24), Image.Resampling.BILINEAR)
        bg_canvas.paste(bot_fill, (0, max(0, paste_y + fg_h - 24)))
    bg_canvas.paste(fg_img, (0, paste_y))
    bg_crop = bg_canvas.filter(ImageFilter.GaussianBlur(radius=22))
    bg_crop = ImageEnhance.Brightness(bg_crop).enhance(0.58)

    # Soft vertical edge feather so the panel blends into the atmospheric extension without harsh borders
    feather_px = max(6, min(28, fg_h // 14))
    alpha_1d = np.ones(fg_h, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, feather_px, dtype=np.float32)
    alpha_1d[:feather_px] = ramp
    alpha_1d[-feather_px:] = ramp[::-1]
    alpha_mask = Image.fromarray((alpha_1d[:, None] * 255.0).astype(np.uint8).repeat(fg_w, axis=1), mode="L")

    bg_crop.paste(fg_img, (paste_x, paste_y), mask=alpha_mask)
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
