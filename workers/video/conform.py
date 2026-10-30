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
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from workers.assembler import decode_video_frames, encode_mjpeg_avi
    from workers.render_kenburns import prepare_cinematic_canvas
except ImportError:
    from assembler import decode_video_frames, encode_mjpeg_avi
    from render_kenburns import prepare_cinematic_canvas

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
    """Prepares a 9:16 letterboxed/pillarboxed clean plate at `(target_w, target_h)` for I2V conditioning."""
    clean_plate = Path(clean_plate)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(clean_plate).convert("RGB")
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
    # Max frames covered by 1.25x slow-mo of the generated clip
    max_slowmo_frames = max(2, int(round(native_dur_s * 1.25 * target_fps)))

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
    colour-matched to `clean_plate`.
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

    ref_img = Image.open(clean_plate).convert("RGB")
    ref_canvas = prepare_cinematic_canvas(ref_img, target_res=target_res, overscan=1.0)

    # Resize raw frames to target_res and apply LAB colour transfer
    sized_frames: List[Image.Image] = []
    for frm in raw_frames:
        f_rgb = frm.convert("RGB")
        if f_rgb.size != target_res:
            # Check aspect ratio: if close to target_res, resize; otherwise canvas-conform
            src_ar = f_rgb.width / float(max(1, f_rgb.height))
            dst_ar = width / float(max(1, height))
            if abs(src_ar - dst_ar) <= 0.08:
                f_rgb = f_rgb.resize(target_res, Image.Resampling.BILINEAR)
            else:
                f_rgb = prepare_cinematic_canvas(f_rgb, target_res=target_res, overscan=1.0)
        f_matched = _lab_color_match(f_rgb, ref_canvas, strength=color_match_strength)
        sized_frames.append(f_matched)

    final_frames = _resample_frames_exact(
        sized_frames,
        expected_frames=expected_frames,
        native_fps=native_fps,
        target_fps=fps,
    )
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
