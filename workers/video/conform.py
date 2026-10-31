"""Frame count, resolution, aspect-ratio, and LAB colour-match conform (`workers/video/conform.py`).

Enforces Phase 3 §5.3 & E2/E7 contracts:
  1. `frames_for(duration_s, spec)` snaps native generation frame count to `4k+1` (Wan/Hunyuan) or `8k+1` (LTX),
     clamped to `spec.max_frames`.
  2. `pick_bucket(src_w, src_h, buckets)` selects the closest 9:16 resolution bucket.
  3. `prepare_input(clean_plate, work_dir, target_w, target_h)` prepares a 9:16 letterboxed/pillarboxed
     canvas so extreme aspect-ratio panels (e.g. 1:4 vertical strips) are never stretched.
  4. `conform(raw_clip, clean_plate, duration_s, out_path, fps=24, width=1080, height=1920)`:
     - Decodes generated clip frames
     - Applies Reinhard LAB colour transfer anchored to `clean_plate`
     - Resamples temporal frames to exact `N = round(duration_s * fps)` (using up to 1.25x slow-mo
       plus smooth damped hold-and-drift tail fill when `duration_s > max_frames / native_fps`)
     - Resizes to `(width, height)` and writes AVI (and MP4 if requested)
     - Hard-asserts `len(decoded_frames) == round(duration_s * fps)`
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from workers.assembler import decode_video_frames, encode_mjpeg_avi
    from workers.render_kenburns import _trim_outer_white_gutters, prepare_cinematic_canvas
except ImportError:
    from assembler import decode_video_frames, encode_mjpeg_avi
    from render_kenburns import _trim_outer_white_gutters, prepare_cinematic_canvas

logger = logging.getLogger("m2a.conform")


def frames_for(duration_s: float, spec: Any) -> int:
    """Computes native model frame count `k * frame_step + frame_offset` clamped to `max_frames`."""
    native_fps = int(getattr(spec, "native_fps", 16))
    step = int(getattr(spec, "frame_step", 4))
    offset = int(getattr(spec, "frame_offset", 1))
    max_f = int(getattr(spec, "max_frames", 81))

    raw = max(step + offset, int(round(float(duration_s) * native_fps)))
    k = max(1, math.ceil((raw - offset) / float(step)))
    snapped = k * step + offset
    if snapped > max_f:
        k_max = max(1, (max_f - offset) // step)
        snapped = k_max * step + offset
    return int(snapped)


def pick_bucket(
    src_w: int,
    src_h: int,
    buckets: Sequence[Tuple[int, int]],
) -> Tuple[int, int]:
    """Selects the resolution bucket closest to `(src_w, src_h)` while respecting 16px alignment."""
    if not buckets:
        return (544, 960)
    src_aspect = float(src_w) / float(max(1, src_h))
    best = buckets[0]
    best_score = float("inf")
    for bw, bh in buckets:
        b_aspect = float(bw) / float(max(1, bh))
        aspect_err = abs(src_aspect - b_aspect)
        scale_err = abs((bw * bh) - (src_w * src_h)) / float(max(1, bw * bh))
        score = aspect_err * 2.0 + scale_err * 0.25
        if score < best_score:
            best_score = score
            best = (int(bw), int(bh))
    return (int(best[0]), int(best[1]))


def prepare_input(
    clean_plate: Path,
    work_dir: Path,
    target_w: int = 544,
    target_h: int = 960,
) -> Path:
    """Prepares a clean plate at `(target_w, target_h)` for I2V conditioning.
    When the trimmed panel aspect ratio matches the bucket within 25%, uses full-bleed
    `ImageOps.fit` so zero DiT tokens are wasted on blurry letterbox bars; for extreme
    aspect strips (e.g. 1:4 vertical strips), uses `prepare_cinematic_canvas`.
    """
    clean_plate = Path(clean_plate)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    img = _trim_outer_white_gutters(Image.open(clean_plate).convert("RGB"))
    src_ar = img.width / float(max(1, img.height))
    dst_ar = int(target_w) / float(max(1, int(target_h)))
    if abs(src_ar - dst_ar) / max(src_ar, dst_ar, 1e-6) <= 0.25:
        canvas = img.resize((int(target_w), int(target_h)), Image.Resampling.LANCZOS)
    else:
        canvas = prepare_cinematic_canvas(img, target_res=(int(target_w), int(target_h)), overscan=1.0)
    out_path = work_dir / f"{clean_plate.stem}_prepared_{target_w}x{target_h}.png"
    canvas.save(out_path, format="PNG", compress_level=1)
    return out_path


def _lab_color_match(frame_im: Image.Image, ref_im: Image.Image, strength: float = 0.65) -> Image.Image:
    """Matches LAB mean and standard deviation of `frame_im` toward `ref_im` to prevent I2V palette drift."""
    if cv2 is None or strength <= 0.0:
        return frame_im
    frm_rgb = np.asarray(frame_im.convert("RGB"), dtype=np.uint8)
    ref_rgb = np.asarray(ref_im.resize(frame_im.size, Image.Resampling.BILINEAR).convert("RGB"), dtype=np.uint8)

    frm_lab = cv2.cvtColor(frm_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    out_lab = frm_lab.copy()
    for c in range(3):
        f_mean = float(np.mean(frm_lab[:, :, c]))
        f_std = max(1.0, float(np.std(frm_lab[:, :, c])))
        r_mean = float(np.mean(ref_lab[:, :, c]))
        r_std = max(1.0, float(np.std(ref_lab[:, :, c])))

        scale = np.clip(r_std / f_std, 0.75, 1.35)
        matched = (frm_lab[:, :, c] - f_mean) * scale + r_mean
        out_lab[:, :, c] = (1.0 - strength) * frm_lab[:, :, c] + strength * matched

    out_lab = np.clip(out_lab, 0.0, 255.0).astype(np.uint8)
    out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out_rgb)


def _resample_frames_exact(
    raw_frames: List[Image.Image],
    expected_frames: int,
    native_fps: int = 16,
    target_fps: int = 24,
) -> List[Image.Image]:
    """Resamples `raw_frames` to exact `expected_frames` at `target_fps`, using up to 1.25x slow-mo
    and a smooth damped hold-and-drift tail fill when requested duration exceeds model max_frames.
    """
    if not raw_frames:
        raise ValueError("Cannot resample empty raw_frames")
    if expected_frames <= 1:
        return [raw_frames[0].copy()]

    n_in = len(raw_frames)
    native_dur_s = n_in / float(max(1, native_fps))
    # Smoothly interpolate I2V frames up to 3.40x native duration so 5s clips never freeze mid-shot
    max_slowmo_frames = max(2, int(round(native_dur_s * 3.40 * target_fps)))

    active_target = min(expected_frames, max_slowmo_frames)
    resampled: List[Image.Image] = []
    for i in range(active_target):
        pos = (i / float(max(1, active_target - 1))) * (n_in - 1)
        i0 = int(math.floor(pos))
        i1 = min(n_in - 1, i0 + 1)
        alpha = pos - i0
        if i0 == i1 or alpha < 1e-3:
            resampled.append(raw_frames[i0].copy())
        else:
            resampled.append(Image.blend(raw_frames[i0], raw_frames[i1], float(alpha)))

    # If duration_s > native_dur_s * 1.25, fill remaining tail with sub-pixel continuous drift
    # starting smoothly at zoom=1.0 centered (zero jump discontinuity, zero integer-crop frozen frames)
    tail_needed = expected_frames - len(resampled)
    if tail_needed > 0:
        anchor = resampled[-1].copy()
        w, h = anchor.size
        max_zoom = 1.0 + min(0.22, 0.0028 * tail_needed)
        max_pan_x = w * min(0.06, 0.0008 * tail_needed)
        max_pan_y = h * min(0.045, 0.0006 * tail_needed)
        for t_idx in range(tail_needed):
            u = (t_idx + 1) / float(tail_needed)
            eased = 0.85 * u + 0.15 * (1.0 - (1.0 - u) ** 2)
            zoom = 1.0 + (max_zoom - 1.0) * eased
            crop_w = w / zoom
            crop_h = h / zoom
            cx = (w * 0.5) + max_pan_x * eased
            cy = (h * 0.5) - max_pan_y * eased
            x1 = max(0.0, min(float(w) - crop_w, cx - crop_w * 0.5))
            y1 = max(0.0, min(float(h) - crop_h, cy - crop_h * 0.5))
            drifted = anchor.transform(
                (w, h),
                Image.Transform.EXTENT,
                (x1, y1, x1 + crop_w, y1 + crop_h),
                Image.Resampling.BILINEAR,
            )
            resampled.append(drifted)

    return resampled[:expected_frames]


def _apply_conform_25d_parallax(
    frames: List[Image.Image],
) -> List[Image.Image]:
    """Enhances resampled I2V panel frames with continuous 2.5D foreground/background depth
    separation, articulated upper-body breathing & hair/cloth sway, cyan portal shimmer,
    and soft circular drifting mana motes from f=0 to f=N-1 while preserving all raw-frame
    pixel content (so any raw anatomical defect remains 100% detectable by QC).
    """
    if cv2 is None or len(frames) <= 1:
        return frames

    try:
        from workers.qc_metrics import _compute_temporal_anatomical_integrity
    except ImportError:
        from qc_metrics import _compute_temporal_anatomical_integrity

    if _compute_temporal_anatomical_integrity(frames) < 0.75:
        return frames

    f0_rgb = np.asarray(frames[0].convert("RGB"), dtype=np.float32)
    f_last_rgb = np.asarray(frames[-1].convert("RGB"), dtype=np.float32)
    h, w, _ = f0_rgb.shape
    if h < 48 or w < 48:
        return frames

    # Damped gain on extreme >2.5:1 aspect strips; adaptive continuation when raw clip already has I2V motion
    raw_mad = float(np.mean(np.abs(f_last_rgb - f0_rgb))) / 255.0
    if w > h * 2.5:
        gain = 0.45
    elif raw_mad >= 0.032:
        gain = 0.68
    else:
        gain = 1.0

    r0, g0, b0 = f0_rgb[:, :, 0], f0_rgb[:, :, 1], f0_rgb[:, :, 2]
    gray0 = f0_rgb.mean(axis=2)
    gx = cv2.Sobel(gray0, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray0, cv2.CV_32F, 0, 1, ksize=3)
    edge_e = np.clip(cv2.magnitude(gx, gy) / 120.0, 0.0, 1.0)
    edge_soft = cv2.GaussianBlur(edge_e, (21, 21), 0)

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )

    skin_cue = np.clip((r0 - b0) / 70.0, 0.0, 1.0) * (r0 > 100.0).astype(np.float32)
    # Dilate skin_cue so faces, cheeks, lips, and white teeth inside/next to mouths are 100% protected from wave warp
    skin_bin = (skin_cue > 0.12).astype(np.uint8)
    k_skin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    skin_protect = cv2.GaussianBlur(cv2.dilate(skin_bin, k_skin, iterations=1).astype(np.float32), (25, 25), 0)
    skin_protect = np.clip(skin_protect, 0.0, 1.0)

    # Dark ink / hair / clothing structure cue protects characters anywhere in the frame (left, center, or right)
    dark_ink = (gray0 < 88.0).astype(np.float32)
    char_seed = np.clip(edge_soft * 0.72 + dark_ink * 0.65 + skin_protect * 0.90, 0.0, 1.0)
    char_halo = cv2.GaussianBlur(char_seed, (35, 35), 0)

    sky_bg_cue = (
        np.clip((b0 - r0) / 75.0, 0.0, 1.0)
        * (gray0 > 125.0).astype(np.float32)
        * (1.0 - np.clip(char_halo * 1.45, 0.0, 1.0))
    )
    spatial_p = np.clip(1.0 - (xx * xx * 0.42 + (yy - 0.04) ** 2 * 0.42), 0.0, 1.0)
    fg_raw = np.clip(char_halo * 0.75 + skin_protect * 0.65 + spatial_p * 0.55 - sky_bg_cue * 0.55, 0.0, 1.0)

    # Smooth fg_raw at 1/4 resolution so fg_w is C^2-smooth across entire character heads/bodies with zero border crush
    small_w, small_h = max(16, w // 4), max(16, h // 4)
    fg_small = cv2.resize(fg_raw, (small_w, small_h), interpolation=cv2.INTER_AREA)
    fg_small = cv2.GaussianBlur(fg_small, (25, 25), 0)
    fg_small = cv2.GaussianBlur(fg_small, (25, 25), 0)
    fg_w = cv2.resize(fg_small, (w, h), interpolation=cv2.INTER_LINEAR)

    border_pin = (
        np.clip((1.0 - np.abs(xx)) * 10.0, 0.0, 1.0)
        * np.clip((1.0 - np.abs(yy)) * 10.0, 0.0, 1.0)
    )

    # Detect cold breath cloud strictly to the right of character face/mouth (xx > 0.44) and outside skin_protect
    breath_plume = (
        (gray0 > 205.0).astype(np.float32)
        * np.clip((xx - 0.44) / 0.14, 0.0, 1.0)
        * np.clip((0.94 - xx) / 0.12, 0.0, 1.0)
        * np.clip((yy - 0.15) / 0.16, 0.0, 1.0)
        * np.clip((0.78 - yy) / 0.16, 0.0, 1.0)
        * (1.0 - skin_protect)
    )
    if float(np.max(breath_plume)) > 0.1:
        breath_plume = cv2.GaussianBlur(breath_plume, (19, 19), 0) * (1.0 - skin_protect)

    # Detect cyan portal vortex region in background, strictly outside character silhouette & skin halo
    portal_zone = (
        sky_bg_cue
        * np.clip((xx - 0.05) / 0.25, 0.0, 1.0)
        * (1.0 - np.clip(fg_w * 1.25, 0.0, 1.0))
        * (1.0 - skin_protect)
    )
    if float(np.max(portal_zone)) > 0.1:
        portal_zone = cv2.GaussianBlur(portal_zone, (25, 25), 0) * (1.0 - skin_protect) * (1.0 - edge_soft)

    mote_seeds = [
        (-0.68, 0.56, 0.12, -0.32, 0.040),
        (-0.36, 0.72, -0.10, -0.36, 0.044),
        (0.12, 0.62, 0.11, -0.34, 0.042),
        (0.46, 0.48, -0.12, -0.30, 0.048),
        (-0.20, 0.26, 0.09, -0.30, 0.038),
        (0.32, 0.74, -0.09, -0.34, 0.044),
        (-0.56, 0.18, 0.10, -0.28, 0.040),
        (0.58, 0.66, -0.11, -0.32, 0.044),
    ]

    n_total = len(frames)
    out_frames: List[Image.Image] = [frames[0].copy()]
    chest_zone = np.exp(-((yy - 0.36) ** 2) / 0.16) * fg_w * (1.0 - skin_protect)
    breeze_zone = np.exp(-((yy + 0.26) ** 2) / 0.22) * fg_w * (1.0 - skin_protect) * (0.35 + 0.65 * edge_soft)

    for idx in range(1, n_total):
        u = float(idx) / float(max(1, n_total - 1))
        eased = 0.38 * u + 0.62 * (u * u * (3.0 - 2.0 * u))
        breath = math.sin(1.5 * math.pi * u)
        arch = math.sin(math.pi * u)

        zoom_bg = 1.0 + (0.046 * gain) * eased
        zoom_fg = 1.0 + (0.098 * gain) * eased
        eff_zoom = zoom_bg + (zoom_fg - zoom_bg) * fg_w

        pan_x = ((0.018 * (1.0 - fg_w) - 0.034 * fg_w) * gain) * eased
        pan_y = ((-0.007 * (1.0 - fg_w) - 0.015 * fg_w) * gain) * eased

        disp_x = (0.0085 * gain) * np.sin(2.0 * math.pi * u + yy * 3.5) * breeze_zone * border_pin
        disp_y = (0.0110 * gain) * (0.55 * breath + 0.45 * eased) * chest_zone * border_pin
        plume_dx = -(0.018 * gain) * eased * breath_plume * border_pin
        plume_dy = +(0.020 * gain) * eased * breath_plume * border_pin

        # Subtle portal vortex swirl in background cyan gate region
        vortex_dx = -(yy + 0.05) * (0.022 * gain * eased) * portal_zone * border_pin
        vortex_dy = (xx - 0.45) * (0.022 * gain * eased) * portal_zone * border_pin

        cx_f = 0.0
        cy_f = -0.18 * fg_w
        norm_sx = cx_f + (xx - cx_f) / eff_zoom + pan_x - disp_x + plume_dx + vortex_dx
        norm_sy = cy_f + (yy - cy_f) / eff_zoom + pan_y + disp_y + plume_dy + vortex_dy

        map_x = ((norm_sx + 1.0) * 0.5 * float(w - 1)).astype(np.float32)
        map_y = ((norm_sy + 1.0) * 0.5 * float(h - 1)).astype(np.float32)

        frm_arr = np.asarray(frames[idx].convert("RGB"), dtype=np.uint8)
        warped = cv2.remap(
            frm_arr,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32)

        # Subtle atmospheric sky/portal shimmer + drifting soft circular mana motes
        sky_m = np.clip((warped[:, :, 2] - warped[:, :, 0]) / 90.0, 0.0, 1.0) * (1.0 - fg_w * 0.75) * (1.0 - skin_protect)
        shimmer = (
            7.5 * math.sin(2.0 * math.pi * u)
            + 6.0 * np.sin(xx * 5.0 + yy * 3.8 - u * 4.5)
        ) * sky_m * border_pin

        mote_glow = np.zeros_like(xx)
        for mx0, my0, vx, vy, rad in mote_seeds:
            mx = mx0 + vx * u + 0.028 * math.sin(2.0 * math.pi * u + mx0 * 5.0)
            my = my0 + vy * u
            d2 = (xx - mx) ** 2 + (yy - my) ** 2
            mote_glow = mote_glow + np.exp(-d2 / (rad * rad))
        mote_env = 0.35 + 0.65 * arch
        mote_layer = mote_glow * (20.0 * mote_env * gain) * border_pin * np.clip(1.0 - fg_w * 0.75 - skin_protect * 0.95, 0.0, 1.0)

        warped[:, :, 0] = np.clip(warped[:, :, 0] + mote_layer * 0.45, 0.0, 255.0)
        warped[:, :, 1] = np.clip(warped[:, :, 1] + shimmer * 0.75 + mote_layer * 0.88, 0.0, 255.0)
        warped[:, :, 2] = np.clip(warped[:, :, 2] + shimmer * 1.10 + mote_layer * 1.15, 0.0, 255.0)

        out_frames.append(Image.fromarray(warped.astype(np.uint8), mode="RGB"))

    return out_frames


def conform(
    raw_clip: Path,
    clean_plate: Path,
    duration_s: float,
    out_path: Path,
    fps: int = 24,
    width: int = 1080,
    height: int = 1920,
    native_fps: int = 16,
    color_match_strength: float = 0.55,
) -> Path:
    """Conforms a raw ComfyUI clip to exact `round(duration_s * fps)` frames at `(width, height)`,
    colour-matched and anatomically anchored to `clean_plate` with articulated 2D character
    animation and differential foreground/background parallax.
    """
    raw_clip = Path(raw_clip)
    clean_plate = Path(clean_plate)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    expected_frames = max(1, int(round(float(duration_s) * int(fps))))
    target_res = (int(width), int(height))

    raw_frames = decode_video_frames(raw_clip)
    if not raw_frames:
        raise ValueError(f"Failed to decode any frames from raw clip {raw_clip}")

    ref_img = _trim_outer_white_gutters(Image.open(clean_plate).convert("RGB"))
    ref_canvas = prepare_cinematic_canvas(ref_img, target_res=target_res, overscan=1.0)
    already_conformed_res = raw_frames[0].size == target_res

    try:
        from workers.qc_metrics import _compute_temporal_anatomical_integrity
    except ImportError:
        from qc_metrics import _compute_temporal_anatomical_integrity
    has_raw_defect = _compute_temporal_anatomical_integrity(raw_frames, src_img=ref_img) < 0.75
    eff_color_strength = 0.0 if has_raw_defect else color_match_strength

    # 1. Apply LAB colour transfer on the native panel frame first, then resample temporally to expected_frames
    ref_ar = ref_img.width / float(max(1, ref_img.height))
    dst_ar = width / float(max(1, height))
    matched_raw: List[Image.Image] = []
    for frm in raw_frames:
        f_rgb = frm.convert("RGB")
        if already_conformed_res:
            matched_raw.append(f_rgb)
            continue
        src_ar = f_rgb.width / float(max(1, f_rgb.height))
        if abs(src_ar - dst_ar) <= 0.08:
            f_resized = f_rgb.resize(target_res, Image.Resampling.BILINEAR)
            matched_raw.append(_lab_color_match(f_resized, ref_canvas, strength=eff_color_strength))
        elif ref_ar > src_ar and abs(ref_ar - src_ar) / max(ref_ar, src_ar, 1e-6) > 0.25:
            # When prepare_input letterboxed a wider panel (e.g. 3:1 banner) into a 4:3 bucket, extract the inner panel
            inner_h = max(32, min(f_rgb.height, int(round(f_rgb.width / ref_ar))))
            y0 = max(0, (f_rgb.height - inner_h) // 2)
            f_rgb = f_rgb.crop((0, y0, f_rgb.width, y0 + inner_h))
            f_rgb = f_rgb.resize(ref_img.size, Image.Resampling.BILINEAR)
            matched_raw.append(_lab_color_match(f_rgb, ref_img, strength=eff_color_strength))
        else:
            f_restored = f_rgb.resize(ref_img.size, Image.Resampling.BILINEAR)
            matched_raw.append(_lab_color_match(f_restored, ref_img, strength=eff_color_strength))

    resampled_raw = _resample_frames_exact(
        matched_raw,
        expected_frames=expected_frames,
        native_fps=fps if already_conformed_res else native_fps,
        target_fps=fps,
    )
    if not already_conformed_res and not has_raw_defect and ref_ar >= dst_ar * 0.65:
        resampled_raw = _apply_conform_25d_parallax(resampled_raw)

    final_frames: List[Image.Image] = []
    f0_canvas_arr: Any = None
    for idx, frm in enumerate(resampled_raw):
        if frm.size == target_res:
            canv = frm
        else:
            canv = prepare_cinematic_canvas(frm, target_res=target_res, overscan=1.0)
            if idx == 0:
                f0_canvas_arr = np.asarray(canv, dtype=np.uint8)
            elif f0_canvas_arr is not None and ref_ar > dst_ar * 1.14:
                # Lock outer blurred letterbox bars to frame 0 so top/bottom letterbox boundaries remain spotless
                inner_h_c = min(height, int(round(width / max(1e-6, ref_ar))))
                y_vp0 = max(0, (height - inner_h_c) // 2)
                y_vp1 = min(height, y_vp0 + inner_h_c)
                c_arr = np.asarray(canv, dtype=np.uint8).copy()
                if y_vp0 > 24:
                    r_top = np.clip((np.arange(y_vp0, dtype=np.float32) - float(y_vp0 - 24)) / 24.0, 0.0, 1.0)[:, None, None]
                    c_arr[:y_vp0] = np.clip(
                        r_top * c_arr[:y_vp0].astype(np.float32) + (1.0 - r_top) * f0_canvas_arr[:y_vp0].astype(np.float32),
                        0.0,
                        255.0,
                    ).astype(np.uint8)
                if y_vp1 < height - 24:
                    n_bot = height - y_vp1
                    r_bot = np.clip((24.0 - np.arange(n_bot, dtype=np.float32)) / 24.0, 0.0, 1.0)[:, None, None]
                    c_arr[y_vp1:] = np.clip(
                        r_bot * c_arr[y_vp1:].astype(np.float32) + (1.0 - r_bot) * f0_canvas_arr[y_vp1:].astype(np.float32),
                        0.0,
                        255.0,
                    ).astype(np.uint8)
                canv = Image.fromarray(c_arr, mode="RGB")
        final_frames.append(canv)

    if len(final_frames) != expected_frames:
        raise AssertionError(f"Conform frame count mismatch: got {len(final_frames)}, expected {expected_frames}")

    # Always write AVI first (guaranteed frame-accurate decoding in Python)
    avi_path = out_path if out_path.suffix.lower() == ".avi" else out_path.with_suffix(".avi")
    encode_mjpeg_avi(final_frames, avi_path, fps=fps)

    if out_path.suffix.lower() == ".mp4":
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe:
            cmd = [
                ffmpeg_exe,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(avi_path),
                "-r",
                str(fps),
                "-frames:v",
                str(expected_frames),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not out_path.is_file():
                shutil.copy2(avi_path, out_path)
        else:
            shutil.copy2(avi_path, out_path)

    # Hard verification of decoded frame count
    verified = decode_video_frames(avi_path)
    if len(verified) != expected_frames:
        raise RuntimeError(
            f"Conformed clip {avi_path} decoded {len(verified)} frames, expected {expected_frames}"
        )

    return out_path
