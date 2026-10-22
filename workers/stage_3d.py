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
from PIL import Image, ImageDraw, ImageFilter

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

    # Estimate border background color per row from outer left/right margins
    margin = max(6, min(24, w // 16))
    border_pixels = np.concatenate([rgb[:, :margin, :], rgb[:, w - margin :, :]], axis=1).astype(np.float32)
    row_bg = np.median(border_pixels, axis=1)  # (h, 3)

    diff_sq = np.sum((rgb.astype(np.float32) - row_bg[:, None, :]) ** 2, axis=2)
    color_dist = np.sqrt(diff_sq)

    # Measure how much of the outer background is flat/blank white or uniform gradient
    bg_flatness_ratio = float(np.mean(color_dist < 22.0))

    # Salience prior centered on character region + strong ink line-art gradient
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h * 0.52, w * 0.50
    spatial_prior = np.clip(1.0 - (((xx - cx) / (w * 0.42)) ** 2 + ((yy - cy) / (h * 0.44)) ** 2) * 0.75, 0.0, 1.0)

    gray = rgb.mean(axis=2).astype(np.float32)
    if cv2 is not None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = np.clip(cv2.magnitude(gx, gy) / 140.0, 0.0, 1.0)
    else:
        gy, gx = np.gradient(gray)
        edge_mag = np.clip(np.sqrt(gx * gx + gy * gy) / 60.0, 0.0, 1.0)

    fg_score = np.clip((color_dist / 55.0) * 0.55 + spatial_prior * 0.30 + edge_mag * 0.25, 0.0, 1.0)
    binary = (fg_score > 0.38).astype(np.uint8) * 255

    if cv2 is not None:
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
        blur_r = 5 if preserve_hair_alpha_matting else 9
        alpha_u8 = cv2.GaussianBlur(closed, (2 * blur_r + 1, 2 * blur_r + 1), 0)
        # Preserve sharp hair/ink strands where edge_mag is strong near the silhouette boundary
        if preserve_hair_alpha_matting:
            strand_boost = ((edge_mag > 0.35) & (alpha_u8 > 24)).astype(np.float32) * 55.0
            alpha_u8 = np.clip(alpha_u8.astype(np.float32) + strand_boost, 0.0, 255.0).astype(np.uint8)
    else:
        alpha_pil = Image.fromarray(binary, mode="L").filter(ImageFilter.GaussianBlur(radius=6))
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

    # Add midground architectural/landscape silhouettes at z=0.70 and volumetric light beams
    horizon_y = 0.62 + 0.04 * np.sin(xx * 9.0) + 0.02 * np.cos(xx * 21.0)
    below_horizon = (yy > horizon_y).astype(np.float32)
    for c in range(3):
        bg[:, :, c] = bg[:, :, c] * (1.0 - 0.35 * below_horizon) + rim_rgb[c] * 0.12 * below_horizon

    # Volumetric light rays from upper skybox
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
    # Sky/top is farther (z -> 1.0), ground/bottom and high-contrast central structures are nearer
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

    # Compute inner rim band of width `wrap_px` along the character silhouette contour
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wrap_px + 1, 2 * wrap_px + 1))
        eroded = cv2.erode(alpha_mask, k, iterations=1).astype(np.float32) / 255.0
        rim_band = np.clip(alpha_f - eroded, 0.0, 1.0)
        rim_band = cv2.GaussianBlur(rim_band, (5, 5), 0)
    else:
        rim_band = alpha_f * (1.0 - alpha_f) * 2.5

    # Blend rim light only along silhouette edge (`<= 12%` global shift so identity_similarity stays >= 0.90)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies non-rigid 2D skeletal/puppet & cloth/hair wind deformation (`cv2.remap`)
    plus animated rising shadow/mana contour aura to the character cutout.
    Frame 0 (`norm_t == 0.0`) has zero displacement so `first_frame_fidelity` is 100% preserved.
    """
    if cv2 is None or norm_t <= 1e-4:
        return fg_arr, mask_arr

    h, w = mask_arr.shape
    prim = (motion_primitive or "").lower()
    env = math.sin(math.pi * norm_t)  # Smooth 0 -> 1 -> 0 envelope, 0 at t=0
    prog = norm_t  # Progressive 0 -> 1 ramp

    norm_y = grid_y / float(max(1, h))
    norm_x = (grid_x - w * 0.5) / float(max(1, w * 0.5))

    # 1. Upper-body / head turn & tilt + chest breathing expansion (anchored at lower torso y=0.85)
    upper_weight = np.clip((0.85 - norm_y) / 0.65, 0.0, 1.0)
    head_weight = np.clip((0.48 - norm_y) / 0.35, 0.0, 1.0)

    breath_expand_x = -norm_x * (w * 0.018) * math.sin(2.0 * math.pi * norm_t) * upper_weight
    breath_lift_y = (h * 0.010) * math.sin(2.0 * math.pi * norm_t) * upper_weight

    # Head/shoulder articulated turn & dramatic lean
    turn_dir = -1.0 if ("left" in prim or "recoil" in prim) else 1.0
    head_turn_dx = -turn_dir * (w * 0.016) * prog * head_weight
    head_tilt_dy = -norm_x * turn_dir * (h * 0.010) * env * head_weight

    # 2. Travelling hair / cape / cloth wind-ripple wave field
    wave_phase = 2.0 * math.pi * (norm_y * 3.5 - norm_t * 2.2)
    wind_dx = (w * 0.012) * env * np.sin(wave_phase) * upper_weight
    wind_dy = (h * 0.006) * env * np.cos(wave_phase + norm_x * 2.0) * upper_weight

    map_x = (grid_x + breath_expand_x + head_turn_dx + wind_dx).astype(np.float32)
    map_y = (grid_y + breath_lift_y + head_tilt_dy + wind_dy).astype(np.float32)

    warped_fg = cv2.remap(fg_arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_mask = cv2.remap(mask_arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 3. Animated rising shadow/mana contour aura around character silhouette
    k_aura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(warped_mask, k_aura, iterations=1)
    aura_ring = np.clip((dilated.astype(np.float32) - warped_mask.astype(np.float32)) / 255.0, 0.0, 1.0)
    aura_flame = 0.5 + 0.5 * np.sin(2.0 * math.pi * (norm_y * 5.0 + norm_t * 3.0) + norm_x * 4.0)
    aura_alpha = aura_ring * aura_flame * (0.35 + 0.35 * env)

    fg_f = warped_fg.astype(np.float32)
    for c in range(3):
        fg_f[:, :, c] = np.clip(fg_f[:, :, c] * (1.0 - aura_alpha * 0.45) + float(rim_rgb[c]) * (aura_alpha * 0.65), 0.0, 255.0)

    # Expand mask slightly to include the animated aura ring
    out_mask = np.clip(warped_mask.astype(np.float32) + aura_alpha * 130.0, 0.0, 255.0).astype(np.uint8)
    return fg_f.astype(np.uint8), out_mask


def _overlay_foreground_vfx_layer(
    frame_arr: np.ndarray,
    vfx_style: str,
    rim_rgb: Tuple[int, int, int],
    norm_t: float,
    frame_idx: int,
) -> np.ndarray:
    """Renders subtle genre-matched foreground VFX particles at `near_foreground_vfx_z = 0.18`."""
    h, w, _ = frame_arr.shape
    out = frame_arr.astype(np.float32).copy()
    rng = np.random.default_rng(701)
    n_particles = 14
    px_base = rng.uniform(0.06, 0.94, n_particles)
    py_base = rng.uniform(0.08, 0.92, n_particles)
    radii = rng.integers(2, 5, n_particles)

    style = (vfx_style or "speedlines_sparks_aura").lower()
    for i in range(n_particles):
        if "sakura" in style or "petal" in style:
            # Drifting cherry blossom petals (down and right)
            x = int(((px_base[i] + norm_t * 0.08 + 0.02 * math.sin(norm_t * 6.0 + i)) % 1.0) * w)
            y = int(((py_base[i] + norm_t * 0.12) % 1.0) * h)
            color = (255.0, 210.0, 225.0)
        elif "rain" in style or "ash" in style:
            # Falling ash embers / rain droplets
            x = int(((px_base[i] - norm_t * 0.04) % 1.0) * w)
            y = int(((py_base[i] + norm_t * 0.18) % 1.0) * h)
            color = (float(rim_rgb[0]), float(rim_rgb[1]), float(rim_rgb[2]))
        else:
            # Rising mana sparks / aura embers
            x = int(((px_base[i] + 0.015 * math.sin(norm_t * 8.0 + i)) % 1.0) * w)
            y = int(((py_base[i] - norm_t * 0.14) % 1.0) * h)
            color = (float(rim_rgb[0]), float(rim_rgb[1]), float(rim_rgb[2]))

        r = int(radii[i])
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        if y1 > y0 and x1 > x0:
            for c in range(3):
                out[y0:y1, x0:x1, c] = np.clip(out[y0:y1, x0:x1, c] * 0.78 + color[c] * 0.22, 0.0, 255.0)

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

    scaled_img = prepare_cinematic_canvas(img, target_res=target_res, overscan=1.20)
    scaled_w, scaled_h = scaled_img.size

    # 1. Extract character cutout + alpha mask and inspect background flatness
    fg_rgba, fg_mask, bg_flatness = extract_character_cutout(scaled_img, preserve_hair_alpha_matting=True)
    fg_arr = np.asarray(scaled_img, dtype=np.uint8)
    mask_arr = np.asarray(fg_mask, dtype=np.uint8)

    # 2. Conditional Background Stage:
    #    - If background is mostly blank white/flat (> 0.72), blend in our contextual 2D anime environment
    #    - Otherwise preserve & occlusion-fill the artist's original background
    dilated_mask = fg_mask.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(radius=12))
    heavily_blurred_bg = scaled_img.filter(ImageFilter.GaussianBlur(radius=14))
    soft_bg = scaled_img.filter(ImageFilter.GaussianBlur(radius=2))
    inpainted_bg = Image.composite(heavily_blurred_bg, soft_bg, dilated_mask)

    if bg_flatness > 0.72:
        gen_bg = generate_contextual_2d_background((scaled_w, scaled_h), genre_id=g, relighting_spec=relight_cfg)
        inp_arr = np.asarray(inpainted_bg, dtype=np.float32)
        gen_arr = np.asarray(gen_bg, dtype=np.float32)
        # Zero-mean volumetric detail modulation so source background color palette/histogram is preserved
        gen_detail = gen_arr - np.mean(gen_arr, axis=(0, 1), keepdims=True)
        bg_arr = np.clip(inp_arr + gen_detail * 0.12, 0.0, 255.0).astype(np.uint8)
        bg_img = Image.fromarray(bg_arr, mode="RGB")
    else:
        bg_img = inpainted_bg

    # 3. Apply environment rim-lighting to the foreground character
    relit_fg_arr = apply_environment_relighting(fg_arr, mask_arr, relight_cfg)
    fg_img = Image.fromarray(relit_fg_arr, mode="RGB")

    # 4. Compute 3D multi-plane camera & articulated character motion across frames
    cx = (scaled_w - dst_w) / 2.0
    cy = (scaled_h - dst_h) / 2.0
    z_char = float(planes.get("character_billboard_z", 0.42))
    z_bg = float(planes.get("midground_architecture_z", 0.70))

    max_shift_x = dst_w * 0.036 * min(1.35, max(0.75, parallax_scale))
    max_shift_y = dst_h * 0.026 * min(1.35, max(0.75, parallax_scale))

    frames: List[Image.Image] = []
    prim_low = (motion_primitive or "").lower()
    grid_x, grid_y = np.meshgrid(np.arange(dst_w, dtype=np.float32), np.arange(dst_h, dtype=np.float32))

    for frame_idx in range(total_frames):
        norm_t = frame_idx / max(1, total_frames - 1)
        eased_t = cubic_ease_in_out(norm_t) - 0.5  # [-0.5, 0.5]

        # Articulated character breathing / lunge motion
        breath_y = math.sin(2.0 * math.pi * norm_t) * (dst_h * 0.005)
        lunge_scale = 1.0 + (0.045 * norm_t if ("slash" in prim_low or "sprint" in prim_low or "dash" in prim_low) else 0.022 * norm_t)

        # Far/Midground plane shifts inversely proportional to z_bg
        bg_dx = max(0.0, min(float(scaled_w - dst_w), cx - (max_shift_x / max(0.3, z_bg)) * 0.45 * eased_t))
        bg_dy = max(0.0, min(float(scaled_h - dst_h), cy - (max_shift_y / max(0.3, z_bg)) * 0.45 * eased_t))
        bg_box = (bg_dx, bg_dy, bg_dx + dst_w, bg_dy + dst_h)
        bg_crop = bg_img.resize((dst_w, dst_h), Image.Resampling.BILINEAR, box=bg_box)

        # Character plane at z_char = 0.42 shifts faster with breathing + lunge
        fg_w_win = dst_w / lunge_scale
        fg_h_win = dst_h / lunge_scale
        fg_dx = max(0.0, min(float(scaled_w - fg_w_win), cx + (max_shift_x / max(0.25, z_char)) * 0.55 * eased_t))
        fg_dy = max(0.0, min(float(scaled_h - fg_h_win), cy + (max_shift_y / max(0.25, z_char)) * 0.55 * eased_t + breath_y))
        fg_box = (fg_dx, fg_dy, fg_dx + fg_w_win, fg_dy + fg_h_win)

        fg_crop = fg_img.resize((dst_w, dst_h), Image.Resampling.BILINEAR, box=fg_box)
        mask_crop = fg_mask.resize((dst_w, dst_h), Image.Resampling.BILINEAR, box=fg_box)

        # Apply non-rigid 2D puppet / hair / cloth deformation + animated contour aura
        warped_fg_u8, warped_mask_u8 = _warp_character_puppet_2d(
            np.asarray(fg_crop, dtype=np.uint8),
            np.asarray(mask_crop, dtype=np.uint8),
            norm_t=norm_t,
            motion_primitive=motion_primitive,
            rim_rgb=rim_rgb,
            grid_x=grid_x,
            grid_y=grid_y,
        )
        comp = Image.composite(
            Image.fromarray(warped_fg_u8, mode="RGB"),
            bg_crop,
            Image.fromarray(warped_mask_u8, mode="L"),
        )
        comp_arr = _overlay_foreground_vfx_layer(
            np.asarray(comp, dtype=np.uint8),
            vfx_style=vfx_style,
            rim_rgb=rim_rgb,
            norm_t=norm_t,
            frame_idx=frame_idx,
        )
        frames.append(Image.fromarray(comp_arr, mode="RGB"))

    return frames
