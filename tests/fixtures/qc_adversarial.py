"""Adversarial video clip fixture generators for Phase 3 QC Metrics v2 (`E6`).

Generates six synthetic defect classes from a reference panel image or baseline clip:
  1. `frozen`       — identical frames repeated N times (`motion_score < 0.005`)
  2. `shuffled`     — frames randomly permuted (`warp_error > 0.12`)
  3. `flicker`      — every other frame brightness +/- 40% (`flicker_score > 0.15`)
  4. `text_overlay` — synthetic speech bubble / banner text burned onto middle frames (`new_text_count >= 1`)
  5. `hue_shift`    — RGB channel permutation across second half (`identity_similarity < 0.65`, `style_drift > 0.25`)
  6. `face_swap`    — central subject crop replaced with inverted/checkerboard alien patch (`identity_similarity < 0.70`)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

try:
    from workers.render_kenburns import generate_kenburns_frames
except ImportError:
    from render_kenburns import generate_kenburns_frames


def _ensure_base_frames(
    src: Union[Path, str, Image.Image, List[Image.Image]],
    num_frames: int = 24,
    size: Tuple[int, int] = (540, 960),
) -> List[Image.Image]:
    if isinstance(src, list) and src:
        return [f.convert("RGB").copy() for f in src]
    if isinstance(src, Image.Image):
        tmp_frames: List[Image.Image] = []
        base = src.convert("RGB").resize(size, Image.Resampling.BICUBIC)
        w, h = size
        for i in range(num_frames):
            t = i / float(max(1, num_frames - 1))
            dx = int(round(t * 16))
            dy = int(round(t * 12))
            crop = base.crop((dx, dy, w, h)).resize(size, Image.Resampling.BILINEAR)
            tmp_frames.append(crop)
        return tmp_frames
    return generate_kenburns_frames(
        image_path=Path(src),
        duration_s=max(1.0, num_frames / 24.0),
        fps=24,
        motion_type="zoom_in",
        target_res=size,
    )


def make_frozen(
    src: Union[Path, str, Image.Image, List[Image.Image]],
    num_frames: int = 24,
    size: Tuple[int, int] = (540, 960),
) -> List[Image.Image]:
    """Repeats frame 0 identically across all frames."""
    base = _ensure_base_frames(src, num_frames=num_frames, size=size)
    first = base[0].copy()
    return [first.copy() for _ in range(len(base))]


def make_shuffled(
    src: Union[Path, str, Image.Image, List[Image.Image]],
    num_frames: int = 24,
    size: Tuple[int, int] = (540, 960),
    seed: int = 42,
) -> List[Image.Image]:
    """Permutes frames in a deterministic non-monotonic order with large spatial jumps."""
    base = _ensure_base_frames(src, num_frames=num_frames, size=size)
    n = len(base)
    # Interleave far-apart frames and horizontal flips so optical flow residual is high
    shuffled: List[Image.Image] = []
    for i in range(n):
        idx = (i * (n // 2 + 1)) % n
        frm = base[idx].copy()
        if i % 2 == 1:
            frm = frm.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        shuffled.append(frm)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def make_flicker(
    src: Union[Path, str, Image.Image, List[Image.Image]],
    num_frames: int = 24,
    size: Tuple[int, int] = (540, 960),
) -> List[Image.Image]:
    """Alternates frame luminance by +/- 45% on every other frame."""
    base = _ensure_base_frames(src, num_frames=num_frames, size=size)
    out: List[Image.Image] = []
    for i, f in enumerate(base):
        arr = np.asarray(f, dtype=np.float32)
        factor = 1.45 if (i % 2 == 0) else 0.55
        mod = np.clip(arr * factor, 0.0, 255.0).astype(np.uint8)
        out.append(Image.fromarray(mod))
    return out


def make_text_overlay(
    src: Union[Path, str, Image.Image, List[Image.Image]],
    num_frames: int = 24,
    size: Tuple[int, int] = (540, 960),
) -> List[Image.Image]:
    """Burns a high-contrast comic speech bubble with dark glyph strokes onto the middle frames."""
    base = _ensure_base_frames(src, num_frames=num_frames, size=size)
    out: List[Image.Image] = []
    n = len(base)
    for i, f in enumerate(base):
        frm = f.copy()
        if n // 5 <= i <= (4 * n) // 5:
            w, h = frm.size
            draw = ImageDraw.Draw(frm)
            bx1, by1 = int(w * 0.18), int(h * 0.16)
            bx2, by2 = int(w * 0.82), int(h * 0.34)
            draw.rounded_rectangle(
                [bx1, by1, bx2, by2],
                radius=22,
                fill=(252, 252, 252),
                outline=(12, 12, 12),
                width=4,
            )
            # Draw dense dark multi-line glyph strokes inside the bubble
            for row_idx in range(3):
                ry = by1 + 24 + row_idx * 34
                for col_x in range(bx1 + 26, bx2 - 30, 18):
                    draw.rectangle([col_x, ry, col_x + 11, ry + 18], fill=(15, 15, 15))
                    draw.line([col_x, ry + 9, col_x + 12, ry + 9], fill=(252, 252, 252), width=2)
        out.append(frm)
    return out


def make_hue_shift(
    src: Union[Path, str, Image.Image, List[Image.Image]],
    num_frames: int = 24,
    size: Tuple[int, int] = (540, 960),
) -> List[Image.Image]:
    """Permutes RGB channels and shifts chrominance across the second half of the clip."""
    base = _ensure_base_frames(src, num_frames=num_frames, size=size)
    out: List[Image.Image] = []
    n = len(base)
    for i, f in enumerate(base):
        if i >= n // 3:
            arr = np.asarray(f, dtype=np.uint8)
            # Roll channels (R->G, G->B, B->R) and invert green channel
            shifted = np.stack([255 - arr[:, :, 1], arr[:, :, 2], arr[:, :, 0]], axis=-1)
            out.append(Image.fromarray(shifted))
        else:
            out.append(f.copy())
    return out


def make_face_swap(
    src: Union[Path, str, Image.Image, List[Image.Image]],
    num_frames: int = 24,
    size: Tuple[int, int] = (540, 960),
) -> List[Image.Image]:
    """Replaces the central character/face region with an unrelated high-contrast alien pattern."""
    base = _ensure_base_frames(src, num_frames=num_frames, size=size)
    out: List[Image.Image] = []
    n = len(base)
    for i, f in enumerate(base):
        arr = np.asarray(f, dtype=np.uint8).copy()
        if i >= n // 4:
            h, w, _ = arr.shape
            y1, y2 = int(h * 0.20), int(h * 0.80)
            x1, x2 = int(w * 0.20), int(w * 0.80)
            yy, xx = np.meshgrid(np.arange(y2 - y1), np.arange(x2 - x1), indexing="ij")
            alien = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            checker = (((xx // 16) + (yy // 16)) % 2).astype(np.uint8)
            alien[:, :, 0] = checker * 20
            alien[:, :, 1] = (1 - checker) * 240
            alien[:, :, 2] = checker * 230
            arr[y1:y2, x1:x2, :] = alien
        out.append(Image.fromarray(arr))
    return out


def generate_all_adversarial_clips(
    src: Union[Path, str, Image.Image, List[Image.Image]],
    num_frames: int = 24,
    size: Tuple[int, int] = (540, 960),
) -> Dict[str, List[Image.Image]]:
    return {
        "frozen": make_frozen(src, num_frames=num_frames, size=size),
        "shuffled": make_shuffled(src, num_frames=num_frames, size=size),
        "flicker": make_flicker(src, num_frames=num_frames, size=size),
        "text_overlay": make_text_overlay(src, num_frames=num_frames, size=size),
        "hue_shift": make_hue_shift(src, num_frames=num_frames, size=size),
        "face_swap": make_face_swap(src, num_frames=num_frames, size=size),
    }
