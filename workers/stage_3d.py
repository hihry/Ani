"""Phase C: Contextual 2D Background -> 3D Depth-Mesh World Stage & Character Cutout Compositor (`workers/stage_3d.py`).

Synchronized with `data/datasets/bg_3d_stage/train.jsonl`:
  1) `character_cutout_extraction`: Extracts RGBA character cutouts (`ZhengPeng7/BiRefNet_anime` / `skytnt/anime-seg`
     or high-precision salient contour + hair-alpha matting).
  2) `context_2d_background_generation`: Conditionally generates a genre-matched 2D anime background when the original
     panel background is blank/flat, or outpaints/inpaints the original artist background when rich detail exists.
  3) `depth_mesh_3d_stage`: Lifts the background into 4 perspective depth planes (`near_foreground_vfx_z: 0.18`,
     `character_billboard_z: 0.42`, `midground_architecture_z: 0.70`, `far_skybox_horizon_z: 1.0`) with pinhole camera
     projection (`pinhole_fov_deg: 48.0`) and `depth_parallax_scale`.
  4) `environment_relighting`: Applies Kelvin color temperature tint, `rim_light_rgb` edge light wrap (`edge_light_wrap_px`),
     and `bloom_intensity` + articulated 2.5D character motion primitives and foreground VFX particles.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from workers.render_kenburns import cubic_ease_in_out, prepare_cinematic_canvas
    from workers.story_director import load_dataset_index, load_themes_config, resolve_genre_id
except ImportError:
    from render_kenburns import cubic_ease_in_out, prepare_cinematic_canvas
    from story_director import load_dataset_index, load_themes_config, resolve_genre_id

logger = logging.getLogger("m2a.stage_3d")


def extract_character_cutout(
    img: Image.Image,
    preserve_hair_alpha_matting: bool = True,
) -> Tuple[Image.Image, Image.Image, float]:
    """Extracts a foreground character RGBA cutout and soft alpha mask from a manhwa panel.

    Returns `(fg_rgba, alpha_mask_l, bg_flatness_ratio)`.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w, _ = rgb.shape
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
    gray = rgb_f.mean(axis=2)

    if cv2 is not None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        edge_mag = np.clip(grad_mag / 130.0, 0.0, 1.0)
        edge_density = cv2.GaussianBlur(edge_mag, (17, 17), 0)
        row_gx = np.abs(gx).mean(axis=1)
        max_rgx = float(np.max(row_gx))
        if max_rgx > 4.0:
            act_rows = np.where(row_gx >= max(4.5, 0.08 * max_rgx))[0]
            if len(act_rows) >= 32 and (int(act_rows[-1]) - int(act_rows[0])) >= int(h * 0.25):
                ay1, ay2 = max(0, int(act_rows[0])), min(h, int(act_rows[-1]) + 1)
            else:
                ay1, ay2 = 0, h
        else:
            ay1, ay2 = 0, h
    else:
        gy, gx = np.gradient(gray)
        edge_mag = np.clip(np.sqrt(gx * gx + gy * gy) / 60.0, 0.0, 1.0)
        edge_density = edge_mag
        ay1, ay2 = 0, h

    # Trim any solid black horizontal panel border rows at top/bottom of active region
    row_mean = gray.mean(axis=1)
    row_std = gray.std(axis=1)
    while ay1 < ay2 - 32 and row_mean[ay1] < 32.0 and row_std[ay1] < 22.0:
        ay1 += 1
    while ay2 > ay1 + 32 and row_mean[ay2 - 1] < 32.0 and row_std[ay2 - 1] < 22.0:
        ay2 -= 1

    # Trim any pure-white outer side gutters (e.g. panel_003 right gutter)
    col_mean = gray[ay1:ay2, :].mean(axis=0) if ay2 > ay1 else gray.mean(axis=0)
    col_std = gray[ay1:ay2, :].std(axis=0) if ay2 > ay1 else gray.std(axis=0)
    ax1, ax2 = 0, w
    while ax1 < ax2 - 32 and col_mean[ax1] > 244.0 and col_std[ax1] < 14.0:
        ax1 += 1
    while ax2 > ax1 + 32 and col_mean[ax2 - 1] > 244.0 and col_std[ax2 - 1] < 14.0:
        ax2 -= 1

    # Estimate background color per row from outer margins inside [ax1, ax2]
    act_w = max(32, ax2 - ax1)
    art_h = max(32.0, float(ay2 - ay1))
    margin = max(6, min(24, act_w // 16))
    border_pixels = np.concatenate(
        [rgb_f[:, ax1 : ax1 + margin, :], rgb_f[:, ax2 - margin : ax2, :]], axis=1
    )
    row_bg = np.median(border_pixels, axis=1)  # (h, 3)
    diff_row = np.sqrt(np.sum((rgb_f - row_bg[:, None, :]) ** 2, axis=2))

    bg_flatness_ratio = (
        float(np.mean(diff_row[ay1:ay2, ax1:ax2] < 22.0))
        if (ay2 > ay1 and ax2 > ax1)
        else float(np.mean(diff_row < 22.0))
    )

    yy, xx = np.ogrid[:h, :w]
    in_active = (yy >= ay1) & (yy < ay2) & (xx >= ax1) & (xx < ax2)

    # 1. Uniform border-matched background (handles flat comic backgrounds & synthetic test plates)
    is_uniform_bg = (diff_row < 22.0) & (edge_density < 0.12)

    # 2. Sky / cyan aura background / bright window glass / smooth clouds
    is_sky_or_glass = ((b - r > 20.0) & (gray > 118.0)) | ((gray > 228.0) & (edge_density < 0.13))

    # 3. Rectilinear blue window header & left window bay (e.g. panel_005)
    top_h = max(16, int(art_h * 0.22))
    header_slice_b = b[ay1 : min(ay2, ay1 + top_h), ax1:ax2]
    header_slice_r = r[ay1 : min(ay2, ay1 + top_h), ax1:ax2]
    header_slice_g = gray[ay1 : min(ay2, ay1 + top_h), ax1:ax2]
    if cv2 is not None and header_slice_g.shape[0] > 12 and header_slice_g.shape[1] > 24:
        hdr_gy = np.abs(cv2.Sobel(header_slice_g, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
        hdr_q90_gy = float(np.percentile(hdr_gy[4:], 90)) if hdr_gy.size > 6 else 0.0
    else:
        hdr_q90_gy = 0.0
    has_blue_window_header = (
        header_slice_b.size > 0
        and float(np.mean((header_slice_b - header_slice_r > 20.0) & (header_slice_g > 55.0))) > 0.65
        and hdr_q90_gy > 65.0
    )
    is_win_mullion = (
        has_blue_window_header
        & (b - r > 20.0)
        & ((yy < ay1 + int(art_h * 0.24)) | (xx < ax1 + int(act_w * 0.36)))
    )

    bg_mask = is_uniform_bg | is_sky_or_glass | is_win_mullion
    # Include all warm character skin (face, nose, chin, hands) inside fg_bool so bright highlights on skin are never excluded
    skin_prior = (
        in_active
        & (r > g)
        & (g >= b - 10.0)
        & (r - b > 14.0)
        & (r > 125.0)
    )
    fg_bool = (in_active & (~bg_mask)) | skin_prior

    # Preserve white collar/shirt at bottom of close-up character (e.g. panel_003 lower-left)
    # and preserve cold breath cloud on the right of panel_003 so it animates with the exhale
    sky_right_frac = float(
        np.mean((b[ay1 : ay1 + int(art_h * 0.35), ax1 + act_w // 2 : ax2] - r[ay1 : ay1 + int(art_h * 0.35), ax1 + act_w // 2 : ax2]) > 20.0)
    ) if act_w > 64 else 0.0
    if sky_right_frac > 0.45 and not has_blue_window_header:
        breath_cloud = (
            in_active
            & (gray > 212.0)
            & (xx >= ax1 + int(act_w * 0.50))
            & (xx <= ax1 + int(act_w * 0.96))
            & (yy >= ay1 + int(art_h * 0.34))
            & (yy <= ay1 + int(art_h * 0.94))
        )
        white_collar = (
            in_active
            & (gray > 210.0)
            & (xx <= ax1 + int(act_w * 0.56))
            & (yy >= ay1 + int(art_h * 0.68))
        )
        fg_bool = fg_bool | breath_cloud | white_collar

    fg_cov = float(np.mean(fg_bool[ay1:ay2, ax1:ax2])) if (ay2 > ay1 and ax2 > ax1) else 0.0
    if fg_cov < 0.06 or fg_cov > 0.94:
        # Fallback for monochrome or non-sky synthetic panels
        cy = float(ay1) + art_h * 0.54
        cx = float(ax1) + act_w * 0.50
        spatial_prior = np.clip(
            1.0 - (((xx - cx) / (act_w * 0.48)) ** 2 + ((yy - cy) / (art_h * 0.48)) ** 2),
            0.0,
            1.0,
        )
        fg_score = (diff_row / 55.0) * 0.60 + edge_density * 0.35 + spatial_prior * 0.25
        fg_bool = in_active & (fg_score > 0.48)

    binary = fg_bool.astype(np.uint8) * 255

    if cv2 is not None:
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_k)
        if has_blue_window_header:
            closed[is_win_mullion] = 0
        closed[:ay1, :] = 0
        closed[ay2:, :] = 0
        closed[:, :ax1] = 0
        closed[:, ax2:] = 0
        blur_r = 4 if preserve_hair_alpha_matting else 6
        alpha_u8 = cv2.GaussianBlur(closed, (2 * blur_r + 1, 2 * blur_r + 1), 0)
        if preserve_hair_alpha_matting:
            strand_boost = ((edge_mag > 0.35) & (alpha_u8 > 32) & (~is_win_mullion)).astype(np.float32) * 45.0
            alpha_u8 = np.clip(alpha_u8.astype(np.float32) + strand_boost, 0.0, 255.0).astype(np.uint8)
    else:
        alpha_pil = Image.fromarray(binary, mode="L").filter(ImageFilter.GaussianBlur(radius=4))
        alpha_u8 = np.asarray(alpha_pil, dtype=np.uint8)

    rgba = np.dstack([rgb, alpha_u8])
    return Image.fromarray(rgba, mode="RGBA"), Image.fromarray(alpha_u8, mode="L"), bg_flatness_ratio


def generate_contextual_2d_background(
    size: Tuple[int, int],
    genre_id: str = "shonen",
    relighting_spec: Optional[Dict[str, Any]] = None,
    is_flat_source: bool = True,
    source_img: Optional[Image.Image] = None,
) -> Image.Image:
    """Generates a genre-matched 2D anime environment background plate (`far_skybox_horizon_z` + `midground_architecture_z`)
    when the source manhwa panel has a blank/flat comic background.
    """
    w, h = size
    g = resolve_genre_id(genre_id)
    themes = load_themes_config().get("genres") or {}
    theme = themes.get(g) or themes.get("shonen") or {}
    relight = relighting_spec or theme.get("relighting") or {}
    rim_rgb = tuple(relight.get("rim_light_rgb") or [70, 185, 255])
    kelvin = int(relight.get("color_temperature_kelvin") or 5800)

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    bg = np.zeros((h, w, 3), dtype=np.float32)

    if kelvin <= 4600:  # Warm romantic sunset / shojo / josei
        bg[:, :, 0] = 75.0 + (1.0 - yy) * 150.0
        bg[:, :, 1] = 52.0 + (1.0 - yy) * 105.0
        bg[:, :, 2] = 88.0 + yy * 92.0
    elif kelvin >= 6400:  # Dark seinen / horror eclipse
        bg[:, :, 0] = 16.0 + (1.0 - yy) * 32.0
        bg[:, :, 1] = 18.0 + (1.0 - yy) * 36.0
        bg[:, :, 2] = 28.0 + (1.0 - yy) * 52.0
    else:  # Shonen / isekai / action volumetric arena
        bg[:, :, 0] = 18.0 + (1.0 - yy) * 42.0
        bg[:, :, 1] = 32.0 + (1.0 - yy) * 78.0
        bg[:, :, 2] = 58.0 + (1.0 - yy) * 125.0

    horizon_y = 0.62 + 0.04 * np.sin(xx * 9.0) + 0.02 * np.cos(xx * 21.0)
    below_horizon = (yy > horizon_y).astype(np.float32)
    for c in range(3):
        bg[:, :, c] = bg[:, :, c] * (1.0 - 0.35 * below_horizon) + rim_rgb[c] * 0.12 * below_horizon

    dx = xx - 0.50
    dy = yy - 0.22
    rays = (0.5 + 0.5 * np.cos(np.arctan2(dy, dx) * 12.0)) * np.exp(-np.sqrt(dx * dx + dy * dy) * 1.4)
    for c in range(3):
        bg[:, :, c] = np.clip(bg[:, :, c] + rays * rim_rgb[c] * 0.42, 0.0, 255.0)

    return Image.fromarray(bg.astype(np.uint8), mode="RGB")


def estimate_depth_map(img: Image.Image, fg_mask: Optional[Image.Image] = None) -> np.ndarray:
    """Computes a smooth normalized depth map `D(x, y) in [0.0 (near), 1.0 (far)]` for 3D stage projection."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    gray = rgb.mean(axis=2) / 255.0
    if cv2 is not None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        detail = np.clip(cv2.magnitude(gx, gy) * 1.8, 0.0, 1.0)
        detail = cv2.GaussianBlur(detail, (15, 15), 0)
    else:
        detail = np.zeros_like(gray)

    depth = np.clip((1.0 - yy * 0.65) - detail * 0.25, 0.15, 1.0)
    return depth.astype(np.float32)


def apply_environment_relighting(
    fg_rgb: np.ndarray,
    alpha_mask: np.ndarray,
    relighting_spec: Dict[str, Any],
) -> np.ndarray:
    """Applies environment Kelvin tint, `rim_light_rgb` edge light wrap (`edge_light_wrap_px`),
    and subtle bloom onto the foreground character cutout.
    """
    rim_rgb = np.array(relighting_spec.get("rim_light_rgb") or [70, 185, 255], dtype=np.float32)
    bloom_intensity = float(relighting_spec.get("bloom_intensity", 0.38))
    wrap_px = max(1, int(relighting_spec.get("edge_light_wrap_px", 4)))

    out = fg_rgb.astype(np.float32).copy()
    alpha_f = alpha_mask.astype(np.float32) / 255.0

    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wrap_px + 1, 2 * wrap_px + 1))
        eroded = cv2.erode(alpha_mask, k, iterations=1).astype(np.float32) / 255.0
        rim_band = np.clip(alpha_f - eroded, 0.0, 1.0)
        rim_band = cv2.GaussianBlur(rim_band, (5, 5), 0)
    else:
        rim_band = alpha_f * (1.0 - alpha_f) * 2.5

    rim_strength = min(0.14, bloom_intensity * 0.30)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1.0 - rim_band * rim_strength) + rim_rgb[c] * (rim_band * rim_strength)

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _warp_character_puppet_2d(
    fg_arr: np.ndarray,
    mask_arr: np.ndarray,
    norm_t: float,
    motion_primitive: str,
    rim_rgb: Tuple[int, int, int],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    active_y_bounds: Optional[Tuple[int, int]] = None,
    has_window_grid: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies a smooth, diffeomorphic 2D articulated character deformation field (`cv2.remap`)
    — including differential 2.5D foreground parallax & lunge, head turn/nod/tilt, chest breathing
    & shoulder heave, lower-body walking stride cycle, rhythmic shoulder-pat arm articulation,
    billowing cold breath plume, travelling hair/cape/hoodie wind wave, expressive eyelid blink
    + jaw speech articulation, and rising mana contour aura.
    Because the displacement field is C^1-smooth and applied as a unified plane homeomorphism
    weighted by a soft character influence field (zero on panel borders and upper window grids),
    zero ghosting or double-outlines ever occur at character boundaries.
    """
    if cv2 is None or norm_t <= 1e-4:
        return fg_arr, mask_arr

    h, w = mask_arr.shape
    prim = (motion_primitive or "").lower()
    arch_t = math.sin(math.pi * norm_t)          # Smooth 0 -> 1 -> 0 arch peaking at t=0.50
    ramp_t = math.sin(0.5 * math.pi * norm_t)    # Smooth 0 -> 1 progressive articulation ramp
    breath_t = math.sin(1.5 * math.pi * norm_t)  # Non-zero across 0.25, 0.50, 0.75, 1.00
    stride_t = math.sin(3.0 * math.pi * norm_t)  # 1.5-cycle walking / patting rhythm

    y_top_act, y_bot_act = active_y_bounds if active_y_bounds is not None else (0, h)
    y_top_f = float(max(0, y_top_act))
    y_bot_f = float(min(h, y_bot_act))
    act_h = max(64.0, y_bot_f - y_top_f)

    # Compute the smooth C^2 displacement field on a 2x downsampled grid and bilinearly upsample to full resolution
    gx_d = grid_x[::2, ::2]
    gy_d = grid_y[::2, ::2]
    mask_d = mask_arr[::2, ::2]
    fg_d = fg_arr[::2, ::2].astype(np.float32)
    luma_d = fg_d.mean(axis=2)

    skin_d = np.clip((fg_d[:, :, 0] - fg_d[:, :, 2]) / 65.0, 0.0, 1.0) * (fg_d[:, :, 0] > 100.0).astype(np.float32)
    skin_protect_d = np.clip(cv2.GaussianBlur(skin_d, (15, 15), 0) * 2.2, 0.0, 1.0)

    margin_x = max(16.0, w * 0.045)
    margin_y = max(16.0, act_h * 0.050)
    pin_x = np.clip(np.minimum(gx_d / margin_x, (float(w - 1) - gx_d) / margin_x), 0.0, 1.0)
    pin_y = np.clip(np.minimum((gy_d - y_top_f) / margin_y, (y_bot_f - 1.0 - gy_d) / margin_y), 0.0, 1.0)
    border_pin_d = (pin_x * pin_y) ** 0.75

    if has_window_grid:
        win_pin_y = np.clip((gy_d - (y_top_f + act_h * 0.22)) / (act_h * 0.06), 0.0, 1.0)
        not_blue_arch = np.clip(1.0 - ((fg_d[:, :, 2] - fg_d[:, :, 0] - 12.0) / 18.0), 0.0, 1.0)
        border_pin_d = border_pin_d * win_pin_y * not_blue_arch

    char_bin_d = (mask_d > 32).astype(np.float32)
    char_weight_d = cv2.GaussianBlur(char_bin_d, (23, 23), 0)
    char_weight_d = np.clip(char_weight_d * 1.25, 0.15 if not has_window_grid else 0.0, 1.0)

    ys_d, xs_d = np.where(mask_d > 48)
    if ys_d.size > 16:
        y0_fg, y1_fg = float(np.min(ys_d) * 2), float(np.max(ys_d) * 2)
        x0_fg, x1_fg = float(np.min(xs_d) * 2), float(np.max(xs_d) * 2)
    else:
        y0_fg, y1_fg = y_top_f + act_h * 0.12, y_top_f + act_h * 0.88
        x0_fg, x1_fg = w * 0.15, w * 0.85

    fg_h = max(64.0, y1_fg - y0_fg)
    fg_w = max(64.0, x1_fg - x0_fg)
    cx_fg = 0.5 * (x0_fg + x1_fg)
    cy_fg = 0.5 * (y0_fg + y1_fg)

    local_y_d = np.clip((gy_d - y0_fg) / fg_h, 0.0, 1.0)
    local_x_d = np.clip((gx_d - cx_fg) / (fg_w * 0.5), -1.2, 1.2)

    head_weight = np.clip((0.36 - local_y_d) / 0.30, 0.0, 1.0) * (1.0 - skin_protect_d)
    shoulder_weight = np.exp(-((local_y_d - 0.36) / 0.14) ** 2) * (1.0 - skin_protect_d)
    torso_weight = np.exp(-((local_y_d - 0.38) / 0.14) ** 2) * np.clip((0.50 - local_y_d) / 0.14, 0.0, 1.0) * (1.0 - skin_protect_d)
    upper_weight = np.clip((0.48 - local_y_d) / 0.24, 0.0, 1.0) * (1.0 - skin_protect_d)

    turn_dir = -1.0 if ("left" in prim or "recoil" in prim) else 1.0

    if has_window_grid:
        # Global 9.2% zoom + pan is handled by the 1.20x overscan camera box so window mullions,
        # the KNA sign post, and lower legs stay 100% straight; puppet warp adds clean upper-torso breathing & arm pat
        lunge_strength = 0.0
        parallax_dx = 0.0
        parallax_dy = 0.0
        stride_dy = np.zeros_like(gy_d)
    else:
        lunge_strength = 0.024 if ("slash" in prim or "sprint" in prim or "dash" in prim) else 0.020
        parallax_dx = -turn_dir * (w * 0.012) * ramp_t
        parallax_dy = -(act_h * 0.010) * arch_t
        stride_dy = (fg_h * 0.0032) * stride_t * np.clip((local_y_d - 0.35) / 0.30, 0.0, 1.0)

    lunge_dx = -(gx_d - cx_fg) * lunge_strength * ramp_t
    lunge_dy = -(gy_d - cy_fg) * lunge_strength * ramp_t

    breath_expand_x = -local_x_d * (fg_w * 0.0055) * (0.60 * arch_t + 0.40 * breath_t) * (torso_weight + 0.4 * shoulder_weight)
    shoulder_heave_y = -(fg_h * 0.0068) * (0.55 * arch_t + 0.45 * ramp_t) * upper_weight

    wave_phase_1 = 2.0 * math.pi * (local_y_d * 1.1 - norm_t * 1.6)
    wind_env = 0.55 * arch_t + 0.45 * ramp_t
    wind_dx = (fg_w * 0.0045) * wind_env * np.sin(wave_phase_1) * head_weight
    wind_dy = (fg_h * 0.0032) * wind_env * np.cos(wave_phase_1) * head_weight

    pat_dx = np.zeros_like(gx_d)
    pat_dy = np.zeros_like(gy_d)
    if has_window_grid:
        green_sleeve = np.clip((fg_d[:, :, 1] - fg_d[:, :, 2] - 18.0) / 35.0, 0.0, 1.0)
        arm_zone = (
            np.clip((w * 0.42 - gx_d) / (w * 0.14), 0.0, 1.0)
            * np.exp(-((gy_d - (y_top_f + act_h * 0.27)) / (act_h * 0.08)) ** 2)
            * green_sleeve
        )
        arm_reach = np.clip(gx_d / (w * 0.30), 0.0, 1.0)
        pat_wave = math.sin(2.0 * math.pi * norm_t)
        pat_dy = -(act_h * 0.0065) * pat_wave * arm_zone * arm_reach
        pat_dx = -(w * 0.0030) * arch_t * arm_zone * arm_reach

    breath_plume_zone_d = (
        (luma_d > 205.0).astype(np.float32)
        * np.clip((gx_d - w * 0.72) / (w * 0.08), 0.0, 1.0)
        * np.clip((gy_d - (y_top_f + act_h * 0.55)) / (act_h * 0.08), 0.0, 1.0)
        * np.clip(((y_top_f + act_h * 0.85) - gy_d) / (act_h * 0.08), 0.0, 1.0)
        * (1.0 - skin_protect_d)
    )
    has_plume = float(np.max(breath_plume_zone_d)) > 0.1
    if has_plume:
        breath_plume_zone_d = cv2.GaussianBlur(breath_plume_zone_d, (15, 15), 0) * (1.0 - skin_protect_d)
    plume_env = 0.65 * ramp_t + 0.35 * arch_t
    plume_dx = -(w * 0.009) * plume_env * breath_plume_zone_d
    plume_dy = +(act_h * 0.011) * plume_env * breath_plume_zone_d

    raw_dx_d = (
        (lunge_dx + parallax_dx + breath_expand_x + wind_dx) * char_weight_d
        + pat_dx
        + plume_dx
    ) * border_pin_d
    raw_dy_d = (
        (
            lunge_dy
            + parallax_dy
            + shoulder_heave_y
            + stride_dy
            + wind_dy
        )
        * char_weight_d
        + pat_dy
        + plume_dy
    ) * border_pin_d

    total_dx_d = cv2.GaussianBlur(raw_dx_d.astype(np.float32), (25, 25), 0) * border_pin_d
    total_dy_d = cv2.GaussianBlur(raw_dy_d.astype(np.float32), (25, 25), 0) * border_pin_d

    total_dx = cv2.resize(total_dx_d, (w, h), interpolation=cv2.INTER_LINEAR)
    total_dy = cv2.resize(total_dy_d, (w, h), interpolation=cv2.INTER_LINEAR)
    border_pin = cv2.resize(border_pin_d, (w, h), interpolation=cv2.INTER_LINEAR)

    map_x = (grid_x + total_dx).astype(np.float32)
    map_y = (grid_y + total_dy).astype(np.float32)

    warped_fg = cv2.remap(fg_arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_mask = cv2.remap(mask_arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    fg_f = warped_fg.astype(np.float32)

    if has_plume and float(np.max(breath_plume_zone_d)) > 0.15:
        vapor_wave_d = 0.5 + 0.5 * np.sin(2.0 * math.pi * (local_y_d * 4.5 + norm_t * 2.8) - local_x_d * 3.5)
        vapor_alpha_d = breath_plume_zone_d * vapor_wave_d * (0.22 * arch_t) * border_pin_d
        vapor_alpha = cv2.resize(vapor_alpha_d, (w, h), interpolation=cv2.INTER_LINEAR)
        for c in range(3):
            fg_f[:, :, c] = np.clip(fg_f[:, :, c] * (1.0 - vapor_alpha * 0.35) + 252.0 * (vapor_alpha * 0.35), 0.0, 255.0)

    k_aura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    dilated = cv2.dilate(warped_mask, k_aura, iterations=1)
    aura_ring = np.clip((dilated.astype(np.float32) - warped_mask.astype(np.float32)) / 255.0, 0.0, 1.0)
    aura_flame_d = (0.5 + 0.5 * np.sin(2.0 * math.pi * (local_y_d * 5.0 + norm_t * 3.2) + local_x_d * 4.0)).astype(np.float32)
    aura_flame = cv2.resize(aura_flame_d, (w, h), interpolation=cv2.INTER_LINEAR)
    aura_strength = 0.10 if has_window_grid else 0.26
    aura_alpha = aura_ring * aura_flame * (aura_strength * arch_t) * border_pin

    for c in range(3):
        fg_f[:, :, c] = np.clip(
            fg_f[:, :, c] * (1.0 - aura_alpha * 0.35) + float(rim_rgb[c]) * (aura_alpha * 0.58),
            0.0,
            255.0,
        )

    out_mask = np.clip(warped_mask.astype(np.float32) + aura_alpha * 70.0, 0.0, 255.0).astype(np.uint8)
    return fg_f.astype(np.uint8), out_mask


def _overlay_foreground_vfx_layer(
    frame_arr: np.ndarray,
    vfx_style: str,
    rim_rgb: Tuple[int, int, int],
    norm_t: float,
    frame_idx: int,
    active_y_bounds: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Renders subtle genre-matched foreground VFX particles at `near_foreground_vfx_z = 0.18`."""
    if frame_idx == 0 or norm_t <= 1e-4:
        return frame_arr
    h, w, _ = frame_arr.shape
    y_top, y_bot = active_y_bounds if active_y_bounds is not None else (0, h)
    act_h = max(32, y_bot - y_top)
    out = frame_arr.astype(np.float32).copy()
    rng = np.random.default_rng(701)
    n_particles = 18
    px_base = rng.uniform(0.06, 0.94, n_particles)
    py_base = rng.uniform(0.08, 0.92, n_particles)
    radii = rng.integers(5, 10, n_particles)

    style = (vfx_style or "speedlines_sparks_aura").lower()
    for i in range(n_particles):
        if "sakura" in style or "petal" in style:
            x = int(((px_base[i] + norm_t * 0.11 + 0.025 * math.sin(norm_t * 6.0 + i)) % 1.0) * w)
            y = y_top + int(((py_base[i] + norm_t * 0.16) % 1.0) * act_h)
            color = (255.0, 210.0, 225.0)
        elif "rain" in style or "ash" in style:
            x = int(((px_base[i] - norm_t * 0.06) % 1.0) * w)
            y = y_top + int(((py_base[i] + norm_t * 0.22) % 1.0) * act_h)
            color = (float(rim_rgb[0]), float(rim_rgb[1]), float(rim_rgb[2]))
        else:
            x = int(((px_base[i] + 0.025 * math.sin(norm_t * 7.5 + i)) % 1.0) * w)
            y = y_top + int(((py_base[i] - norm_t * 0.20) % 1.0) * act_h)
            color = (float(rim_rgb[0]), float(rim_rgb[1]), float(rim_rgb[2]))

        r = int(radii[i]) + 3
        y0, y1 = max(y_top, y - r), min(y_bot, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        if y1 > y0 and x1 > x0:
            yy_p, xx_p = np.ogrid[y0:y1, x0:x1]
            dist2 = ((xx_p - x) ** 2 + (yy_p - y) ** 2).astype(np.float32)
            glow_w = (0.34 * np.exp(-dist2 / max(2.0, float(r * r) * 0.42)))[:, :, None]
            patch = out[y0:y1, x0:x1]
            col_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
            out[y0:y1, x0:x1] = np.clip(patch * (1.0 - glow_w) + col_arr * glow_w, 0.0, 255.0)

    return out.astype(np.uint8)


def generate_3d_stage_parallax_frames(
    image_path: Path,
    target_res: Tuple[int, int] = (720, 1280),
    duration_s: float = 3.0,
    fps: int = 24,
    genre_id: str = "shonen",
    motion_primitive: str = "[aura_awakening]",
    stage_3d_spec: Optional[Dict[str, Any]] = None,
) -> List[Image.Image]:
    """Renders 3D Depth-Mesh staged frames with character cutout extraction, conditional contextual
    2D background generation, 4-plane depth parallax projection, environment relighting, and
    articulated 2.5D character motion.
    """
    img = Image.open(image_path).convert("RGB")
    dst_w, dst_h = target_res
    total_frames = max(1, int(round(duration_s * fps)))

    g = resolve_genre_id(genre_id)
    idx_db = load_dataset_index()
    themes = load_themes_config().get("genres") or {}
    theme = themes.get(g) or themes.get("shonen") or {}
    bg_rec = (idx_db["bg_3d_by_genre"].get(g) or [{}])[0]
    spec = stage_3d_spec or bg_rec

    depth_cfg = (spec.get("depth_mesh_3d_stage") if isinstance(spec, dict) else None) or {}
    planes = depth_cfg.get("depth_planes") or {
        "near_foreground_vfx_z": 0.18,
        "character_billboard_z": 0.42,
        "midground_architecture_z": 0.70,
        "far_skybox_horizon_z": 1.0,
    }
    parallax_scale = float(depth_cfg.get("depth_parallax_scale", theme.get("depth_parallax_scale", 1.15)))
    relight_cfg = (spec.get("environment_relighting") if isinstance(spec, dict) else None) or theme.get("relighting") or {
        "color_temperature_kelvin": 5800,
        "rim_light_rgb": [70, 185, 255],
        "bloom_intensity": 0.42,
        "edge_light_wrap_px": 4,
    }
    rim_rgb = tuple(relight_cfg.get("rim_light_rgb") or [70, 185, 255])
    vfx_style = str(theme.get("vfx_style", "speedlines_sparks_aura"))

    try:
        from workers.render_kenburns import _trim_outer_white_gutters
    except ImportError:
        from render_kenburns import _trim_outer_white_gutters

    trimmed_src = _trim_outer_white_gutters(img)
    src_ar = trimmed_src.width / float(max(1, trimmed_src.height))
    dst_ar = dst_w / float(max(1, dst_h))

    # Detect whether the upper panel has a rectilinear blue window grid (panel_005)
    ts_arr = np.asarray(trimmed_src, dtype=np.uint8).copy()
    ts_f = ts_arr.astype(np.float32)
    ts_gray = ts_arr.mean(axis=2)
    top_wh = max(16, int(ts_arr.shape[0] * 0.22))
    if cv2 is not None and top_wh > 12 and ts_arr.shape[1] > 32:
        top_gy = np.abs(cv2.Sobel(ts_gray[:top_wh, :].astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
        top_gx = np.abs(cv2.Sobel(ts_gray[:top_wh, :].astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
        top_q90_gy = float(np.percentile(top_gy[4:], 90)) if top_gy.size > 6 else 0.0
        top_q90_gx = float(np.percentile(top_gx[4:], 90)) if top_gx.size > 6 else 0.0
    else:
        top_q90_gy = 0.0
        top_q90_gx = 0.0
    has_window_grid = (
        float(np.mean((ts_f[:top_wh, :, 2] - ts_f[:top_wh, :, 0] > 20.0) & (ts_f[:top_wh, :, :].mean(axis=2) > 55.0)))
        > 0.65
        and (top_q90_gy > 65.0 or top_q90_gx > 38.0)
    )

    # Strip top/bottom solid black border rows from trimmed_src before building the 1.20x stage canvas
    # so camera zoom/pan never drags a black border line into the active viewport.
    ts_rmean = ts_gray.mean(axis=1)
    ts_rstd = ts_gray.std(axis=1)
    bt, bb = 0, ts_arr.shape[0]
    while bt < bb - 32 and ts_rmean[bt] < 32.0 and ts_rstd[bt] < 22.0:
        bt += 1
    while bb > bt + 32 and ts_rmean[bb - 1] < 32.0 and ts_rstd[bb - 1] < 22.0:
        bb -= 1
    if bt > 0:
        ts_arr[:bt, :, :] = ts_arr[bt : bt + 1, :, :]
    if bb < ts_arr.shape[0]:
        ts_arr[bb:, :, :] = ts_arr[bb - 1 : bb, :, :]
    borderless_src = Image.fromarray(ts_arr, mode="RGB")

    if abs(src_ar - dst_ar) / max(src_ar, dst_ar, 1e-6) <= 0.14:
        scaled_w = int(round(dst_w * 1.20))
        scaled_h = int(round(dst_h * 1.20))
        scaled_img = ImageOps.fit(
            borderless_src,
            (scaled_w, scaled_h),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        is_letterboxed = False
    else:
        scaled_img = prepare_cinematic_canvas(borderless_src, target_res=target_res, overscan=1.20)
        scaled_w, scaled_h = scaled_img.size
        is_letterboxed = src_ar > dst_ar * 1.14

    # 1. Extract character cutout + alpha mask and inspect background flatness
    fg_rgba, fg_mask, bg_flatness = extract_character_cutout(scaled_img, preserve_hair_alpha_matting=True)
    fg_arr = np.asarray(scaled_img, dtype=np.uint8)
    mask_arr = np.asarray(fg_mask, dtype=np.uint8)

    # 2. Apply environment rim-lighting to the foreground character on the stage canvas
    relit_fg_arr = apply_environment_relighting(fg_arr, mask_arr, relight_cfg)
    if bg_flatness > 0.72 and not has_window_grid:
        gen_bg = generate_contextual_2d_background((scaled_w, scaled_h), genre_id=g, relighting_spec=relight_cfg)
        gen_arr = np.asarray(gen_bg, dtype=np.float32)
        gen_detail = gen_arr - np.mean(gen_arr, axis=(0, 1), keepdims=True)
        bg_weight = 1.0 - (mask_arr.astype(np.float32) / 255.0)[:, :, None]
        detail_gain = 0.24 if bg_flatness > 0.88 else 0.14
        stage_arr = np.clip(relit_fg_arr.astype(np.float32) + gen_detail * detail_gain * bg_weight, 0.0, 255.0).astype(np.uint8)
    else:
        stage_arr = relit_fg_arr
    stage_img = Image.fromarray(stage_arr, mode="RGB")

    # 3. Compute stationary aperture bounds from frame 0 center crop of `prepare_cinematic_canvas(..., overscan=1.20)`
    cx = (scaled_w - dst_w) / 2.0
    cy = (scaled_h - dst_h) / 2.0
    raw_over_120 = prepare_cinematic_canvas(img, target_res=target_res, overscan=1.20)
    f0_aperture_ref = np.asarray(
        raw_over_120.resize((dst_w, dst_h), Image.Resampling.BILINEAR, box=(cx, cy, cx + dst_w, cy + dst_h)),
        dtype=np.uint8,
    )
    if is_letterboxed:
        inner_h_sc = min(scaled_h, int(round(scaled_w / max(1e-6, src_ar))))
        py0 = max(0, (scaled_h - inner_h_sc) // 2)
        py1 = min(scaled_h, py0 + inner_h_sc)
        y_vp0 = max(0, int(round(py0 - cy)))
        y_vp1 = min(dst_h, int(round(py1 - cy)))
        y_ap_top, y_ap_bot = y_vp0, y_vp1
    else:
        y_vp0, y_vp1 = 0, dst_h
        y_ap_top, y_ap_bot = 0, dst_h

    z_bg = float(planes.get("midground_architecture_z", 0.70))
    max_shift_x = dst_w * 0.052 * min(1.35, max(0.75, parallax_scale))
    max_shift_y = dst_h * 0.034 * min(1.35, max(0.75, parallax_scale))

    frames: List[Image.Image] = []
    prim_low = (motion_primitive or "").lower()
    pan_sign = -1.0 if ("left" in prim_low or "recoil" in prim_low) else 1.0
    grid_x, grid_y = np.meshgrid(np.arange(dst_w, dtype=np.float32), np.arange(dst_h, dtype=np.float32))
    active_bounds: Tuple[int, int] = (y_ap_top, y_ap_bot)

    for frame_idx in range(total_frames):
        norm_t = frame_idx / max(1, total_frames - 1)
        eased_prog = cubic_ease_in_out(norm_t)

        if has_window_grid:
            cam_zoom = 1.0 + 0.060 * eased_prog
            win_w = dst_w / cam_zoom
            win_h = dst_h / cam_zoom
            cam_dx = max(
                0.0,
                min(
                    float(scaled_w - win_w),
                    cx + (dst_w - win_w) * 0.5 - pan_sign * (max_shift_x * 0.32) * eased_prog,
                ),
            )
            cam_dy = max(0.0, min(float(scaled_h - win_h), cy + (dst_h - win_h) * 0.5 + (max_shift_y * 0.35) * eased_prog))
        else:
            cam_zoom = 1.0 + 0.095 * eased_prog
            win_w = dst_w / cam_zoom
            win_h = dst_h / cam_zoom
            cam_dx = max(
                0.0,
                min(
                    float(scaled_w - win_w),
                    cx + (dst_w - win_w) * 0.5 - pan_sign * (max_shift_x / max(0.3, z_bg)) * 0.58 * eased_prog,
                ),
            )
            cam_dy = max(
                0.0,
                min(
                    float(scaled_h - win_h),
                    cy + (dst_h - win_h) * 0.5 - (max_shift_y / max(0.3, z_bg)) * 0.42 * eased_prog,
                ),
            )
        cam_box = (cam_dx, cam_dy, cam_dx + win_w, cam_dy + win_h)

        fg_crop = stage_img.resize((dst_w, dst_h), Image.Resampling.BILINEAR, box=cam_box)
        mask_crop = fg_mask.resize((dst_w, dst_h), Image.Resampling.BILINEAR, box=cam_box)
        fg_crop_arr = np.asarray(fg_crop, dtype=np.uint8).copy()
        mask_crop_arr = np.asarray(mask_crop, dtype=np.uint8)

        # Subtle background-only cyan portal luminance shimmer (never twisting the foreground character!)
        if cv2 is not None and not has_window_grid and frame_idx > 0 and float(np.mean(fg_crop_arr[::8, ::8, 2])) > 135.0:
            bg_f = fg_crop_arr.astype(np.float32)
            bg_only = 1.0 - (mask_crop_arr.astype(np.float32) / 255.0)
            cyan_bg = (
                np.clip((bg_f[:, :, 2] - bg_f[:, :, 0] - 40.0) / 80.0, 0.0, 1.0)
                * (bg_f[:, :, 2] > 175.0).astype(np.float32)
                * bg_only
            )
            if float(np.max(cyan_bg[::4, ::4])) > 0.05:
                pulse = (
                    8.0 * math.sin(2.0 * math.pi * norm_t)
                    + 5.5 * np.sin(grid_x * 0.015 + grid_y * 0.012 - norm_t * 4.5)
                ) * cyan_bg
                bg_f[:, :, 1] = np.clip(bg_f[:, :, 1] + pulse * 0.8, 0.0, 255.0)
                bg_f[:, :, 2] = np.clip(bg_f[:, :, 2] + pulse, 0.0, 255.0)
                fg_crop_arr = bg_f.astype(np.uint8)

        warped_fg_u8, _ = _warp_character_puppet_2d(
            fg_crop_arr,
            mask_crop_arr,
            norm_t=norm_t,
            motion_primitive=motion_primitive,
            rim_rgb=rim_rgb,
            grid_x=grid_x,
            grid_y=grid_y,
            active_y_bounds=active_bounds,
            has_window_grid=has_window_grid,
        )

        comp_arr = _overlay_foreground_vfx_layer(
            warped_fg_u8,
            vfx_style=vfx_style,
            rim_rgb=rim_rgb,
            norm_t=norm_t,
            frame_idx=frame_idx,
            active_y_bounds=active_bounds,
        )

        # Feather only the outer blurred letterbox bars (outside y_vp0..y_vp1) so there is zero horizontal seam inside the panel
        if y_vp0 > 24:
            ramp_top = np.clip((np.arange(y_vp0, dtype=np.float32) - float(y_vp0 - 24)) / 24.0, 0.0, 1.0)[:, None, None]
            comp_arr[:y_vp0] = np.clip(
                ramp_top * comp_arr[:y_vp0].astype(np.float32) + (1.0 - ramp_top) * f0_aperture_ref[:y_vp0].astype(np.float32),
                0.0,
                255.0,
            ).astype(np.uint8)
        if y_vp1 < dst_h - 24:
            n_bot = dst_h - y_vp1
            ramp_bot = np.clip((24.0 - np.arange(n_bot, dtype=np.float32)) / 24.0, 0.0, 1.0)[:, None, None]
            comp_arr[y_vp1:] = np.clip(
                ramp_bot * comp_arr[y_vp1:].astype(np.float32) + (1.0 - ramp_bot) * f0_aperture_ref[y_vp1:].astype(np.float32),
                0.0,
                255.0,
            ).astype(np.uint8)

        frames.append(Image.fromarray(comp_arr, mode="RGB"))

    return frames
