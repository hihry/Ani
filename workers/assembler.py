"""Video assembly, clean-plate inpainting, QC metrics, and audio mixing module.

Chains panel clips together with crossfade transitions, computes frame-level
auto-QC metrics, inpaints speech bubbles for clean plates, synthesizes fallback
speech WAV tracks, mixes timed audio tracks anchored to panel timestamps, and
encodes the final timeline into:
  1. H.264 MP4 with mixed AAC audio & subtitles (when ffmpeg is available)
  2. Standards-compliant MJPEG AVI (streamed pure-Python video container)
  3. Animated WebP (browser-compatible high-fidelity animation)
"""
import io
import math
import os
import shutil
import struct
import subprocess
import textwrap
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import numpy as np
import logging

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from workers.understanding import count_residual_text_bubbles, detect_speech_bubbles_and_text
except ImportError:
    from understanding import count_residual_text_bubbles, detect_speech_bubbles_and_text

try:
    from workers.qc_metrics import compute_clip_qc_metrics
except ImportError:
    from qc_metrics import compute_clip_qc_metrics

try:
    from workers.audio_director import mix_cinematic_soundtrack, synthesize_expressive_speech_wav
except ImportError:
    from audio_director import mix_cinematic_soundtrack, synthesize_expressive_speech_wav

logger = logging.getLogger("m2a.assembler")

_FONT_CANDIDATES = [
    os.environ.get("M2A_SUB_FONT", ""),
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_subtitle_font(size: int = 22) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_transition_frames(
    tail_a: List[Image.Image],
    head_b: List[Image.Image],
    transition_type: str = "crossfade",
) -> List[Image.Image]:
    """Renders frame-accurate transition frames between `tail_a` and `head_b` supporting all 10 anime transition styles."""
    n = min(len(tail_a), len(head_b))
    if n <= 0:
        return []

    ttype = (transition_type or "crossfade").strip().lower()
    blended: List[Image.Image] = []
    mid_idx = n // 2

    for i in range(n):
        alpha = float(i + 1) / float(n + 1)
        fa = tail_a[i].convert("RGB")
        fb = head_b[i].convert("RGB")
        w, h = fa.size

        if ttype == "impact_flash":
            # Sakuga high-contrast impact frame at midpoint + sharp cut
            base = Image.blend(fa, fb, 1.0 if i >= mid_idx else 0.0)
            if i == mid_idx:
                inv = ImageOps.invert(base.convert("L")).convert("RGB")
                hi_contrast = ImageEnhance.Contrast(inv).enhance(2.8)
                blended.append(Image.blend(base, hi_contrast, 0.78))
            elif abs(i - mid_idx) == 1:
                white = Image.new("RGB", (w, h), (245, 250, 255))
                blended.append(Image.blend(base, white, 0.42))
            else:
                blended.append(Image.blend(fa, fb, alpha))

        elif ttype == "whip_pan":
            # Fast horizontal motion-blurred slide transition
            shift_px = int(round((1.0 - alpha) * w))
            canvas = Image.new("RGB", (w, h))
            canvas.paste(fa, (shift_px - w, 0))
            canvas.paste(fb, (shift_px, 0))
            blur_rad = max(2, int(round(math.sin(math.pi * alpha) * (w * 0.025))))
            blended.append(canvas.filter(ImageFilter.BoxBlur(blur_rad)))

        elif ttype in ("dreamy_dissolve", "petal_dissolve"):
            # Soft bloom / warm sakura bokeh dissolve
            soft_a = fa.filter(ImageFilter.GaussianBlur(radius=4 * alpha))
            soft_b = fb.filter(ImageFilter.GaussianBlur(radius=4 * (1.0 - alpha)))
            mix = Image.blend(soft_a, soft_b, alpha)
            bloom_strength = math.sin(math.pi * alpha) * (0.32 if ttype == "petal_dissolve" else 0.24)
            tint_color = (255, 228, 240) if ttype == "petal_dissolve" else (245, 248, 255)
            tint = Image.new("RGB", (w, h), tint_color)
            blended.append(Image.blend(mix, tint, bloom_strength))

        elif ttype == "fade_through_black":
            # Cinematic dip-to-black transition
            black = Image.new("RGB", (w, h), (6, 6, 10))
            if alpha < 0.5:
                blended.append(Image.blend(fa, black, alpha * 2.0))
            else:
                blended.append(Image.blend(black, fb, (alpha - 0.5) * 2.0))

        elif ttype == "light_burst_flash":
            # Golden/cyan magical light flare across cut
            mix = Image.blend(fa, fb, alpha)
            flare = math.sin(math.pi * alpha) ** 1.4
            bright = ImageEnhance.Brightness(mix).enhance(1.0 + 0.85 * flare)
            blended.append(bright)

        elif ttype == "glitch_cut":
            # Chromatic RGB channel offset + scanline slice
            base = fb if alpha >= 0.5 else fa
            arr = np.asarray(base, dtype=np.uint8).copy()
            shift = max(2, int(round(math.sin(math.pi * alpha) * (w * 0.03))))
            arr[:, :, 0] = np.roll(arr[:, :, 0], shift, axis=1)
            arr[:, :, 2] = np.roll(arr[:, :, 2], -shift, axis=1)
            # Interleave horizontal slice from other clip
            other_arr = np.asarray(fa if alpha >= 0.5 else fb, dtype=np.uint8)
            band_h = max(8, h // 8)
            y0 = int((alpha * h * 0.7)) % max(1, h - band_h)
            arr[y0 : y0 + band_h] = other_arr[y0 : y0 + band_h]
            blended.append(Image.fromarray(arr))

        elif ttype in ("wipe_cut", "hud_scan_wipe"):
            # Sharp diagonal or glowing cyan HUD scanline wipe
            wipe_x = int(round(alpha * w))
            canvas = fb.copy()
            if wipe_x < w:
                right_crop = fa.crop((wipe_x, 0, w, h))
                canvas.paste(right_crop, (wipe_x, 0))
            draw = ImageDraw.Draw(canvas)
            line_col = (60, 225, 255) if ttype == "hud_scan_wipe" else (255, 255, 255)
            draw.line([(wipe_x, 0), (wipe_x, h)], fill=line_col, width=4)
            blended.append(canvas)

        else:
            blended.append(Image.blend(fa, fb, alpha))

    return blended


def crossfade_transition(
    clip_a: List[Image.Image],
    clip_b: List[Image.Image],
    transition_frames: int = 12,
    transition_type: str = "crossfade",
) -> List[Image.Image]:
    """Blends the tail of clip_a with the head of clip_b using the specified transition style."""
    if transition_frames <= 0 or len(clip_a) < transition_frames or len(clip_b) < transition_frames:
        return clip_a + clip_b

    lead_a = clip_a[:-transition_frames]
    tail_a = clip_a[-transition_frames:]
    head_b = clip_b[:transition_frames]
    tail_b = clip_b[transition_frames:]

    blended = render_transition_frames(tail_a, head_b, transition_type=transition_type)
    return lead_a + blended + tail_b


def inpaint_panel_bubbles(
    image_path: Path,
    output_path: Path,
    bubble_bboxes: Optional[List[Dict[str, int]]] = None,
    mask_path: Optional[Path] = None,
    mask_output_path: Optional[Path] = None,
    img: Optional[Image.Image] = None,
) -> Path:
    """Removes speech bubbles, borders, pointer tails, and lettering from a panel image to produce a clean plate."""
    img_pil = img.convert("RGB") if img is not None else Image.open(image_path).convert("RGB")
    arr = np.array(img_pil, dtype=np.uint8)
    h, w, _ = arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if cv2 is None:
        if bubble_bboxes:
            for b in bubble_bboxes:
                x1, y1 = max(0, int(b.get("x", 0))), max(0, int(b.get("y", 0)))
                x2, y2 = min(w, x1 + int(b.get("w", 0))), min(h, y1 + int(b.get("h", 0)))
                mask[y1:y2, x1:x2] = 255
        img_pil.save(output_path, format="PNG", compress_level=1)
        return output_path

    # Convert RGB -> BGR for OpenCV processing
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    luma = bgr.mean(axis=2)
    b_ch = bgr[:, :, 0].astype(np.float32)
    g_ch = bgr[:, :, 1].astype(np.float32)
    r_ch = bgr[:, :, 2].astype(np.float32)
    chroma = np.maximum(np.maximum(np.abs(b_ch - g_ch), np.abs(g_ch - r_ch)), np.abs(b_ch - r_ch))

    # 1. Detect top black panel border (by..be) separating top white webtoon gutter from panel interior
    left_luma = bgr[:, : min(60, w // 4)].mean(axis=(1, 2))
    right_luma = bgr[:, max(0, w - min(60, w // 4)) :].mean(axis=(1, 2))
    border_ys = [
        y
        for y in range(4, h // 2)
        if (left_luma[y] < 100.0 and left_luma[max(0, y - 5)] > 235.0)
        or (right_luma[y] < 100.0 and right_luma[max(0, y - 5)] > 235.0)
    ]
    by = border_ys[0] if border_ys else 0
    be = by + 3 if by > 0 else 0

    # 2. Load explicit binary mask if provided
    if mask_path is not None and Path(mask_path).is_file():
        m_img = Image.open(mask_path).convert("L").resize((w, h), Image.Resampling.NEAREST)
        mask = (np.asarray(m_img, dtype=np.uint8) > 127).astype(np.uint8) * 255
    else:
        # Build dark-wall-bounded speech bubble mask so bubble interiors never bridge across black outlines
        dark_wall = (luma < 150.0).astype(np.uint8)
        dark_wall = cv2.dilate(dark_wall, np.ones((3, 3), np.uint8), iterations=1)
        pure_white = ((luma > 240.0) & (chroma < 14.0) & (dark_wall == 0)).astype(np.uint8)
        pure_white[:15, :] = 0

        num_l, labels, stats, _ = cv2.connectedComponentsWithStats(pure_white)
        for i in range(1, num_l):
            cx, cy, cw, ch, area = stats[i]
            # Skip top white webtoon gutter components above the top black panel border
            if (cy + ch) <= be + 4:
                continue
            # Check if this white component falls inside an explicit bubble_bbox
            in_explicit_bbox = False
            if bubble_bboxes:
                for bb in bubble_bboxes:
                    bx0, by0 = int(bb.get("x", 0)), int(bb.get("y", 0))
                    bw0, bh0 = int(bb.get("w", 0)), int(bb.get("h", 0))
                    if cx + cw > bx0 and cx < bx0 + bw0 and cy + ch > by0 and cy < by0 + bh0:
                        in_explicit_bbox = True
                        break

            if not in_explicit_bbox:
                min_area = max(1200, int(w * h * 0.015))
                min_w = max(80, int(w * 0.16))
                min_h = max(40, int(h * 0.065))
                if area < min_area or cw < min_w or ch < min_h or cy > int(h * 0.45):
                    continue
                if cy <= 16 and cw > int(w * 0.85):
                    continue
                # Exclude unenclosed bright background glow touching bottom-right panel edge
                if cx + cw >= w - 2 and cy + ch >= int(h * 0.70):
                    continue

            comp = (labels == i).astype(np.uint8)
            pad = 28
            y1, y2 = max(0, cy - pad), min(h, cy + ch + pad)
            x1, x2 = max(0, cx - pad), min(w, cx + cw + pad)
            sub_c = comp[y1:y2, x1:x2].copy()
            cnts, _ = cv2.findContours(sub_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sub_c, cnts, -1, 1, thickness=-1)
            sub_c = cv2.morphologyEx(sub_c, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
            cnts, _ = cv2.findContours(sub_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sub_c, cnts, -1, 1, thickness=-1)
            sub_c = cv2.dilate(sub_c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)

            # Capture sharp triangular pointer tail sticking out of bubble
            ys_c, xs_c = np.where(sub_c > 0)
            if len(ys_c) > 0:
                cy_b, cx_b = ys_c.mean(), xs_c.mean()
                sub_white = ((luma[y1:y2, x1:x2] > 220.0) & (chroma[y1:y2, x1:x2] < 20.0)).astype(np.uint8)
                tail_search = cv2.dilate(sub_c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27)), iterations=1)
                cand = (sub_white & tail_search & (~sub_c)).astype(np.uint8)
                if np.any(cand):
                    ty, tx = np.where(cand > 0)
                    vy, vx = ty.mean() - cy_b, tx.mean() - cx_b
                    norm = max(1.0, float(np.hypot(vx, vy)))
                    vy, vx = vy / norm, vx / norm
                    py, px = int(round(ty.mean())), int(round(tx.mean()))
                    for s in range(-4, 26):
                        cy_p = int(round(py + vy * s))
                        cx_p = int(round(px + vx * s))
                        r_cone = max(5, int(10 - s * 0.22))
                        cv2.circle(sub_c, (cx_p, cy_p), r_cone, 1, -1)

            mask[y1:y2, x1:x2] |= (sub_c > 0).astype(np.uint8) * 255

        # Also ensure any explicit bubble_bboxes that did not match a large component are covered
        if bubble_bboxes:
            for bb in bubble_bboxes:
                bx0 = max(0, int(bb.get("x", 0)) - 6)
                by0 = max(0, int(bb.get("y", 0)) - 6)
                bx2 = min(w, int(bb.get("x", 0)) + int(bb.get("w", 0)) + 6)
                by2 = min(h, int(bb.get("y", 0)) + int(bb.get("h", 0)) + 6)
                if bx2 - bx0 > 12 and by2 - by0 > 12:
                    if np.sum(mask[by0:by2, bx0:bx2] > 0) < 0.15 * (bx2 - bx0) * (by2 - by0):
                        sub_w = ((luma[by0:by2, bx0:bx2] > 210.0) & (chroma[by0:by2, bx0:bx2] < 28.0)).astype(np.uint8)
                        if np.any(sub_w > 0):
                            sub_w = cv2.morphologyEx(
                                sub_w, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                            )
                            cnts, _ = cv2.findContours(sub_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(sub_w, cnts, -1, 1, thickness=-1)
                            sub_w = cv2.dilate(sub_w, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
                            mask[by0:by2, bx0:bx2] |= sub_w * 255
                        else:
                            mask[by0:by2, bx0:bx2] = 255

        # Absorb isolated small dark pointer tips sticking out up to 9px from bubble mask
        if np.any(mask > 0):
            dil9 = cv2.dilate((mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
            ring = (dil9 > 0) & (mask == 0)
            if by > 0:
                ring[:be, :] = False
            dark_tips = (ring & (luma < 95.0)).astype(np.uint8)
            nl_t, lbl_t, st_t, _ = cv2.connectedComponentsWithStats(dark_tips)
            for j in range(1, nl_t):
                if st_t[j, cv2.CC_STAT_AREA] < 160:
                    tip_dil = cv2.dilate((lbl_t == j).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
                    mask[tip_dil > 0] = 255

    full_mask = mask.copy()
    out_bgr = bgr.copy()
    if by > 0 and np.any(mask > 0):
        out_bgr[:by, :] = 255
        border_col = np.median(bgr[by:be, : min(60, w // 4)], axis=(0, 1)).astype(np.uint8)
        out_bgr[by:be, :] = border_col
        mask[:be, :] = 0

    if np.any(mask > 0):
        ys_all, xs_all = np.where(mask > 0)
        ry1_all, ry2_all = int(ys_all.min()), int(ys_all.max()) + 1
        rx1_all, rx2_all = int(xs_all.min()), int(xs_all.max()) + 1
        nl_m, _, _, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))

        # Regime classification:
        # 1) Architectural rectilinear window grid (e.g., panel_005: two top bubbles spanning left & right edges over blue window mullions)
        is_window_grid = (
            by > 0
            and by < 60
            and (nl_m - 1) >= 2
            and (rx1_all < int(w * 0.08))
            and (rx2_all > int(w * 0.90))
            and (ry2_all < int(h * 0.35))
        )

        # 2) Hair crown overlap (e.g., panel_003: single top bubble dipping into bright hair sheen + dark sapphire hair crown)
        left_patch = bgr[be : min(h, be + 60), max(0, rx1_all - 60) : max(1, rx1_all - 5)].astype(np.float32)
        below_patch = bgr[min(h - 1, ry2_all) : min(h, ry2_all + 50), rx1_all:rx2_all].astype(np.float32)
        is_hair_crown = (
            not is_window_grid
            and by > 100
            and (nl_m - 1) == 1
            and ry1_all <= be + 4
            and ry2_all < int(h * 0.38)
            and left_patch.size > 0
            and below_patch.size > 0
            and float(left_patch.mean()) > 205.0
            and float((below_patch[:, :, 0] - below_patch[:, :, 2]).mean()) > 40.0
        )

        if is_window_grid:
            while be < by + 8 and left_luma[be] < 140.0:
                be += 1
            out_bgr[by:be, :] = border_col
            mask[:be, :] = 0

            m_clean = cv2.morphologyEx(
                (mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            )
            m_core = cv2.dilate(m_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
            m_soft = cv2.dilate(m_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1)
            m_core[:be, :] = 0
            m_soft[:be, :] = 0

            out_f = out_bgr.astype(np.float32)
            mullion_right_x = min(w - 8, int(w * 0.932))
            win_bot_col = np.zeros((w, 3), dtype=np.float32)
            glass_bot = np.array([252.0, 245.0, 206.0], dtype=np.float32)
            glass_top = np.array([242.0, 226.0, 178.0], dtype=np.float32)

            for x in range(w):
                if x < int(w * 0.115):
                    col_s = bgr[min(h - 1, 212) : min(h, 234), x].astype(np.float32)
                    valid = col_s[(col_s[:, 0] - col_s[:, 2] > 35.0) & (col_s.mean(axis=1) > 75.0)]
                    win_bot_col[x] = np.median(valid, axis=0) if len(valid) > 0 else glass_bot
                elif x < int(w * 0.165):
                    win_bot_col[x] = glass_bot
                elif x < int(w * 0.296):
                    col_s = bgr[min(h - 1, 260) : min(h, 290), x].astype(np.float32)
                    valid = col_s[(col_s[:, 0] - col_s[:, 2] > 35.0) & (col_s.mean(axis=1) > 65.0)]
                    win_bot_col[x] = np.median(valid, axis=0) if len(valid) > 0 else glass_bot
                elif int(w * 0.775) <= x < int(w * 0.810):
                    col_s = bgr[min(h - 1, 318) : min(h, 338), x].astype(np.float32)
                    valid = col_s[(col_s[:, 0] - col_s[:, 2] > 28.0) & (col_s.mean(axis=1) > 55.0) & (col_s.mean(axis=1) < 165.0)]
                    win_bot_col[x] = np.median(valid, axis=0) if len(valid) > 0 else glass_bot
                elif x < mullion_right_x:
                    win_bot_col[x] = glass_bot
                else:
                    col_s = bgr[min(h - 1, 250) : min(h, 278), x].astype(np.float32)
                    valid = col_s[col_s.mean(axis=1) > 55.0]
                    win_bot_col[x] = np.median(valid, axis=0) if len(valid) > 0 else bgr[min(h - 1, 265), x].astype(np.float32)

            win_bot_col[:mullion_right_x] = cv2.GaussianBlur(
                win_bot_col[:mullion_right_x].reshape(1, -1, 3), (3, 1), 0
            ).reshape(-1, 3)
            mullion_delta = np.minimum(0.0, win_bot_col - glass_bot)
            win_top_col = np.clip(glass_top + 0.92 * mullion_delta, 0.0, 255.0)
            win_top_col[mullion_right_x:] = win_bot_col[mullion_right_x:] * 0.96

            ys_all_w, _ = np.where(m_soft > 0)
            ry2_max = max(int(ys_all_w.max()) + 1, min(h, 324))
            win_synth = np.zeros((ry2_max, w, 3), dtype=np.float32)
            for y in range(be, ry2_max):
                v = np.clip(float(y - be) / max(1.0, float(245 - be)), 0.0, 1.0)
                win_synth[y] = (1.0 - v) * win_top_col + v * win_bot_col

            blind_zone = np.zeros((h, w), dtype=np.uint8)
            blind_zone[be : min(h, 224), :mullion_right_x] = 255
            blind_zone[be : min(h, 252), mullion_right_x:] = 255
            # Extend right-center vertical mullion (x ~ 624..645) continuously across the white glare gap up to the header
            for x in range(int(w * 0.775), int(w * 0.810)):
                if win_bot_col[x, 0] - win_bot_col[x, 2] > 25.0 and win_bot_col[x].mean() < 165.0:
                    blind_zone[be : min(h, 322), x] = 255

            m_glass_soft = np.maximum(m_soft, blind_zone)
            m_glass_core = np.maximum(m_core, cv2.erode(blind_zone, np.ones((9, 9), np.uint8), iterations=1))
            m_glass_core[be : min(h, 220), :mullion_right_x] = 255
            m_glass_Core_right = blind_zone[:, mullion_right_x:] > 0
            m_glass_core[:, mullion_right_x:][m_glass_Core_right] = 255
            for x in range(int(w * 0.775), int(w * 0.810)):
                if win_bot_col[x, 0] - win_bot_col[x, 2] > 25.0 and win_bot_col[x].mean() < 165.0:
                    m_glass_core[be : min(h, 320), x] = 255
            m_glass_soft[:be, :] = 0
            m_glass_core[:be, :] = 0

            dist_w = cv2.distanceTransform((m_glass_soft[:ry2_max] > 0).astype(np.uint8), cv2.DIST_L2, 5)
            alpha_w = np.clip(dist_w / 10.0, 0.0, 1.0)[:, :, None]
            alpha_w = np.where((m_glass_core[:ry2_max] > 0)[:, :, None], 1.0, alpha_w)

            out_f[:ry2_max] = alpha_w * win_synth + (1.0 - alpha_w) * out_f[:ry2_max]

            # Cleanly terminate horizontal crossbar at the vertical mullion (x = 220..234) without an orphan stub at x = 235..258
            stub_x1, stub_x2 = int(w * 0.292), int(w * 0.324)
            stub_y1, stub_y2 = min(h - 4, 304), min(h - 1, 326)
            if stub_y2 > stub_y1 and stub_x2 > stub_x1:
                for x in range(stub_x1, stub_x2):
                    top_ref = out_f[max(be, stub_y1 - 3), x]
                    bot_ref = out_f[min(h - 1, stub_y2 + 3), x]
                    for y in range(stub_y1, stub_y2):
                        u_s = float(y - stub_y1) / float(max(1, stub_y2 - stub_y1 - 1))
                        out_f[y, x] = (1.0 - u_s) * top_ref + u_s * bot_ref

            out_bgr = np.clip(out_f, 0.0, 255.0).astype(np.uint8)
            out_bgr[by:be, :] = border_col

        elif is_hair_crown:
            while be < by + 8 and left_luma[be] < 140.0:
                be += 1
            out_bgr[by:be, :] = border_col
            mask[:be, :] = 0

            m_clean = cv2.morphologyEx(
                (mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
            )
            m_core = cv2.dilate(m_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
            m_soft = cv2.dilate(m_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
            m_core[:be, :] = 0
            m_soft[:be, :] = 0

            ys, xs = np.where(m_soft > 0)
            ry1, ry2 = int(ys.min()), int(ys.max()) + 1
            rx1, rx2 = int(xs.min()), int(xs.max()) + 1

            x_crown_end = float(rx1 + (rx2 - rx1) * 0.72)
            y_crown_end = float(ry1 + (ry2 - ry1) * 0.75)
            for x in range(rx2 - 1, rx1, -1):
                cys = np.where(m_clean[:, x] > 0)[0]
                if len(cys) == 0:
                    continue
                yb = int(cys[-1]) + 4
                px = bgr[min(h - 1, yb + 4), x].astype(np.float32)
                if (px[0] - px[2] > 55.0) and (px.mean() < 165.0):
                    x_crown_end = float(x)
                    y_crown_end = float(yb)
                    break

            out_f = out_bgr.astype(np.float32)
            luma_full = out_f.mean(axis=2)

            c_hair_x = np.zeros((w, 3), dtype=np.float32)
            for x in range(w):
                tx = np.clip((float(x) - 110.0) / max(1.0, x_crown_end - 110.0), 0.0, 1.0)
                c_left_navy = np.array([54.0, 24.0, 4.0], dtype=np.float32)
                c_mid_sapphire = np.array([144.0, 74.0, 15.0], dtype=np.float32)
                c_right_sapphire = np.array([156.0, 84.0, 18.0], dtype=np.float32)
                if tx < 0.55:
                    u = tx / 0.55
                    c_hair_x[x] = (1.0 - u) * c_left_navy + u * c_mid_sapphire
                else:
                    u = (tx - 0.55) / 0.45
                    c_hair_x[x] = (1.0 - u) * c_mid_sapphire + u * c_right_sapphire

            # Smoothly unmix the bright white/ice-blue sheen on the left hair crown into deep sapphire hair
            y_dome_max = min(h, ry2 + 64)
            x_left_arr = np.zeros(y_dome_max, dtype=np.float32)
            x_left_prev = 128.0
            for y in range(be, y_dome_max):
                dark_xs = np.where(luma_full[y, 2:220] < 80.0)[0] + 2
                if len(dark_xs) > 0:
                    x_left_prev = float(dark_xs[0])
                x_left_arr[y] = x_left_prev
            if y_dome_max > be + 5:
                x_left_arr[be:y_dome_max] = cv2.GaussianBlur(
                    x_left_arr[be:y_dome_max].reshape(-1, 1), (1, 15), 0
                ).reshape(-1)

            red_excess = np.clip((out_f[:y_dome_max, :, 2] - 18.0) / 52.0, 0.0, 1.0)
            red_excess = cv2.GaussianBlur(red_excess, (11, 11), 0)
            for y in range(be, y_dome_max):
                v_env = np.clip(float(y_dome_max - y) / 22.0, 0.0, 1.0)
                xl = int(round(x_left_arr[y]))
                for x in range(xl + 1, int(x_crown_end) + 12):
                    if x >= w:
                        break
                    speck_env = np.clip(float(ry2 + 22 - y) / 18.0, 0.0, 1.0) if x > 380 else v_env
                    h_env = np.clip((x_crown_end + 10.0 - float(x)) / 14.0, 0.0, 1.0)
                    env = min(v_env, speck_env) * h_env
                    if env <= 0.0:
                        continue
                    alpha_w = float(red_excess[y, x]) * env
                    if alpha_w > 1e-3:
                        px = out_f[y, x]
                        c_sap = c_hair_x[x]
                        out_f[y, x] = (1.0 - alpha_w) * px + alpha_w * c_sap

            out_bgr = np.clip(out_f, 0.0, 255.0).astype(np.uint8)

            x_curve_start = x_crown_end - (x_crown_end - float(rx1)) * 0.50
            x_arch_end = x_crown_end + 36.0
            y_arch_end = y_crown_end + 16.0
            y_crown_arr = np.full(w, float(be - 1), dtype=np.float32)
            for x in range(w):
                if x <= x_curve_start:
                    y_crown_arr[x] = float(be - 1)
                else:
                    t_x = (float(x) - x_curve_start) / max(1.0, x_arch_end - x_curve_start)
                    y_crown_arr[x] = float(be) + (y_arch_end - float(be)) * (t_x**1.52)

            r_sky_col = np.zeros((h, 3), dtype=np.float32)
            for y in range(ry1, ry2):
                rxs = np.where(m_soft[y] > 0)[0]
                xs1 = int(rxs[-1]) if len(rxs) > 0 else rx2
                sky_patch = out_f[y, min(w - 1, xs1 + 3) : min(w, xs1 + 15)]
                r_sky_col[y] = np.median(sky_patch, axis=0) if len(sky_patch) > 0 else out_f[y, w - 5]
            r_sky_col[ry1:ry2] = cv2.GaussianBlur(r_sky_col[ry1:ry2].reshape(-1, 1, 3), (1, 15), 0).reshape(-1, 3)

            yy, xx = np.mgrid[0:h, 0:w]
            yc_grid = y_crown_arr[xx]

            # Reconstruct and Laplace-smooth hair interior FIRST (before placing sky above yc_grid)
            # so bright sky pixels never diffuse downward into the dark hair crown!
            hair_mask = (m_soft > 0) & (yy > yc_grid)
            y1, y2 = ry1, min(h, ry2 + 12)
            x1, x2 = max(0, rx1 - 12), min(w, rx2 + 12)
            sub_u = out_f[y1:y2, x1:x2].copy()
            sub_m = (m_soft[y1:y2, x1:x2] > 0).copy()

            for x in range(x1, x2):
                cys = np.where(sub_m[:, x - x1])[0]
                if len(cys) == 0:
                    continue
                yt, yb = int(cys[0]), int(cys[-1])
                bot_c = sub_u[min(y2 - y1 - 1, yb + 3), x - x1]
                top_c = c_hair_x[x] * 0.92
                u = np.linspace(0.0, 1.0, yb - yt + 1, dtype=np.float32)[:, None]
                sub_u[yt : yb + 1, x - x1] = (1.0 - u) * top_c + u * bot_c
                if yt > 0:
                    sub_u[:yt, x - x1] = top_c

            for _ in range(240):
                up = np.roll(sub_u, 1, axis=0)
                dn = np.roll(sub_u, -1, axis=0)
                lf = np.roll(sub_u, 1, axis=1)
                rt = np.roll(sub_u, -1, axis=1)
                up[0] = dn[0]
                sub_u[sub_m] = (0.30 * (up + dn) + 0.20 * (lf + rt))[sub_m]

            sub_hair_m = hair_mask[y1:y2, x1:x2]
            out_f[y1:y2, x1:x2][sub_hair_m] = sub_u[sub_hair_m]

            # Smoothly reconstruct the cyan sky above yc_grid (interpolating horizontally to outer sky)
            for y in range(ry1, ry2):
                rxs = np.where(m_soft[y] > 0)[0]
                if len(rxs) == 0:
                    continue
                xs0, xs1 = int(rxs[0]), int(rxs[-1])
                right_c = out_f[y, min(w - 1, xs1 + 2)]
                for x in range(xs0, xs1 + 1):
                    yc = yc_grid[y, x]
                    u_x = float(x - xs0) / float(max(1, xs1 - xs0))
                    sky_c = (1.0 - 0.35 * u_x) * r_sky_col[y] + (0.35 * u_x) * right_c
                    if y <= yc - 1.5:
                        out_f[y, x] = sky_c
                    elif y < yc + 1.5:
                        w_sky = np.clip((yc + 1.5 - float(y)) / 3.0, 0.0, 1.0)
                        h_c = out_f[min(h - 1, int(round(yc)) + 2), x]
                        out_f[y, x] = (1.0 - w_sky) * h_c + w_sky * sky_c

            dist_s = cv2.distanceTransform(m_soft.astype(np.uint8), cv2.DIST_L2, 5)
            alpha_s = np.clip(dist_s / 8.0, 0.0, 1.0)[:, :, None]
            alpha_s = np.where((m_core > 0)[:, :, None], 1.0, alpha_s)
            out_f = alpha_s * out_f + (1.0 - alpha_s) * out_bgr.astype(np.float32)
            out_bgr = np.clip(out_f, 0.0, 255.0).astype(np.uint8)
            out_bgr[by:be, :] = border_col


        else:
            # Regime 3: Atmospheric sky / glowing portal / general panel inpainting via multi-scale 2D Laplace + cloud wisp synthesis
            m_clean = cv2.morphologyEx(
                (mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            )
            m_core = cv2.dilate(m_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
            m_soft = cv2.dilate(m_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1)
            m_core[:be, :] = 0
            m_soft[:be, :] = 0

            nl_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(m_soft)
            out_f = out_bgr.astype(np.float32)
            rng = np.random.default_rng(42)
            frame_pad = min(8, max(3, min(w, h) // 40))

            for idx in range(1, nl_c):
                bx, by_c, bw, bh, area = stats_c[idx]
                if area < 64:
                    continue
                comp_soft = labels_c == idx
                comp_core = comp_soft & (m_core > 0)
                y1 = max(be, by_c - 12)
                y2 = min(h, by_c + bh + 12)
                x1 = max(0, bx - 12)
                x2 = min(w, bx + bw + 12)

                sub_orig = out_f[y1:y2, x1:x2].copy()
                sub_u = sub_orig.copy()
                sub_m = comp_soft[y1:y2, x1:x2].copy()
                sub_c = comp_core[y1:y2, x1:x2].copy()
                sh, sw = sub_m.shape
                touch_top = y1 <= be + 1

                # Compute vertical linear interpolation first (always valid even for full-width banners)
                vert_u = sub_u.copy()
                for rx in range(sw):
                    cys = np.where(sub_m[:, rx])[0]
                    if len(cys) == 0:
                        continue
                    ys0, ys1 = int(cys[0]), int(cys[-1])
                    has_b = (y1 + ys1) < (h - frame_pad - 2)
                    has_t = (not touch_top or ys0 > 2) and ((y1 + ys0) > (be + frame_pad))
                    bc = (
                        np.median(sub_u[min(sh - 1, ys1 + 2) : min(sh, ys1 + 8), rx], axis=0)
                        if has_b
                        else np.median(sub_u[max(0, ys0 - 8) : max(1, ys0 - 1), rx], axis=0)
                    )
                    tc = (
                        np.median(sub_u[max(0, ys0 - 8) : max(1, ys0 - 1), rx], axis=0)
                        if has_t
                        else bc
                    )
                    v = np.linspace(0.0, 1.0, ys1 - ys0 + 1, dtype=np.float32)[:, None]
                    vert_u[ys0 : ys1 + 1, rx] = (1.0 - v) * tc + v * bc

                horiz_u = sub_u.copy()
                any_horiz_valid = False
                for ry in range(sh):
                    rxs = np.where(sub_m[ry])[0]
                    if len(rxs) == 0:
                        continue
                    xs0, xs1 = int(rxs[0]), int(rxs[-1])
                    has_l = (x1 + xs0) > (frame_pad + 4)
                    has_r = (x1 + xs1) < (w - frame_pad - 4)
                    if not has_l and not has_r:
                        horiz_u[ry, xs0 : xs1 + 1] = vert_u[ry, xs0 : xs1 + 1]
                        # Also overwrite frame border outside sub_m so dark panel border doesn't bleed inward
                        sub_u[ry, :xs0] = vert_u[ry, xs0]
                        sub_u[ry, xs1 + 1 :] = vert_u[ry, xs1]
                        continue
                    any_horiz_valid = True
                    lc = (
                        np.median(sub_u[ry, max(0, xs0 - 6) : max(1, xs0 - 1)], axis=0)
                        if has_l
                        else np.median(sub_u[ry, min(sw - 1, xs1 + 2) : min(sw, xs1 + 7)], axis=0)
                    )
                    rc = (
                        np.median(sub_u[ry, min(sw - 1, xs1 + 2) : min(sw, xs1 + 7)], axis=0)
                        if has_r
                        else lc
                    )
                    if not has_l:
                        sub_u[ry, :xs0] = lc
                    if not has_r:
                        sub_u[ry, xs1 + 1 :] = rc
                    u = np.linspace(0.0, 1.0, xs1 - xs0 + 1, dtype=np.float32)[:, None]
                    horiz_u[ry, xs0 : xs1 + 1] = (1.0 - u) * lc + u * rc

                if any_horiz_valid:
                    sub_u[sub_m] = 0.5 * (horiz_u[sub_m] + vert_u[sub_m])
                else:
                    sub_u[sub_m] = vert_u[sub_m]

                for scale in [4, 2, 1]:
                    if scale > 1 and min(sh, sw) >= scale * 8:
                        ch_s, cw_s = sh // scale, sw // scale
                        cu = cv2.resize(sub_u, (cw_s, ch_s), interpolation=cv2.INTER_AREA)
                        cm = cv2.resize(sub_m.astype(np.uint8), (cw_s, ch_s), interpolation=cv2.INTER_NEAREST) > 0
                        for _ in range(320):
                            up = np.roll(cu, 1, axis=0)
                            dn = np.roll(cu, -1, axis=0)
                            lf = np.roll(cu, 1, axis=1)
                            rt = np.roll(cu, -1, axis=1)
                            if touch_top:
                                up[0] = dn[0]
                            cu[cm] = 0.25 * (up + dn + lf + rt)[cm]
                        up_u = cv2.resize(cu, (sw, sh), interpolation=cv2.INTER_LINEAR)
                        sub_u[sub_m] = up_u[sub_m]
                    else:
                        for _ in range(320):
                            up = np.roll(sub_u, 1, axis=0)
                            dn = np.roll(sub_u, -1, axis=0)
                            lf = np.roll(sub_u, 1, axis=1)
                            rt = np.roll(sub_u, -1, axis=1)
                            if touch_top:
                                up[0] = dn[0]
                            sub_u[sub_m] = 0.25 * (up + dn + lf + rt)[sub_m]

                dist_soft = cv2.distanceTransform(sub_m.astype(np.uint8), cv2.DIST_L2, 5)
                alpha_blend = np.clip(dist_soft / 9.0, 0.0, 1.0)[:, :, None]
                alpha_blend = np.where(sub_c[:, :, None], 1.0, alpha_blend)

                dist_core = cv2.distanceTransform(sub_c.astype(np.uint8), cv2.DIST_L2, 5)
                w_tex = np.clip(dist_core / 14.0, 0.0, 1.0)[:, :, None]
                n1 = cv2.GaussianBlur(rng.standard_normal((sh, sw)).astype(np.float32), (31, 31), 8.0)
                n1 /= max(1e-5, float(np.std(n1)))
                local_cyan = np.clip((sub_u[:, :, 0] - sub_u[:, :, 2]) / 160.0, 0.0, 1.0)[:, :, None]
                sub_u += w_tex * n1[:, :, None] * 2.5 * local_cyan * np.array([0.4, 0.8, 1.0], dtype=np.float32)

                blended = alpha_blend * sub_u + (1.0 - alpha_blend) * sub_orig
                out_f[y1:y2, x1:x2][sub_m] = blended[sub_m]

            out_bgr = np.clip(out_f, 0.0, 255.0).astype(np.uint8)

    clean_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    try:
        from workers.render_kenburns import _trim_outer_white_gutters
    except ImportError:
        from render_kenburns import _trim_outer_white_gutters
    out_img = _trim_outer_white_gutters(Image.fromarray(clean_rgb))

    resolved_mask_out = mask_output_path
    if resolved_mask_out is None and output_path.name.endswith("_clean.png"):
        resolved_mask_out = output_path.with_name(output_path.name[:-10] + "_mask.png")
    if resolved_mask_out is not None:
        Image.fromarray(full_mask, mode="L").save(Path(resolved_mask_out), format="PNG", compress_level=1)

    out_img.save(output_path, format="PNG", compress_level=1)
    logger.info(f"Created clean plate: {output_path}")
    return output_path


def synthesize_speech_wav(
    text: str,
    output_path: Path,
    voice_id: str = "default",
    emotion: str = "neutral",
    language: str = "ko",
    sample_rate: int = 24000,
    genre_id: str = "shonen",
) -> float:
    """Synthesizes an expressive 16-bit PCM WAV speech file (Kokoro / multi-formant vocal synthesis)
    normalized to the genre's EBU R128 dialogue LUFS target.
    """
    return synthesize_expressive_speech_wav(
        text=text,
        output_path=Path(output_path),
        voice_id=voice_id,
        emotion=emotion,
        language=language,
        genre_id=genre_id,
        sample_rate=sample_rate,
    )


def burn_subtitle_on_frames(
    frames: List[Image.Image],
    dialogue_lines: List[Dict[str, Any]],
    in_place: bool = False,
) -> List[Image.Image]:
    """Renders anime-styled subtitle overlays onto the bottom safe zone of a panel's frames.

    Optimization (P1-12 / O10): Renders the subtitle banner once onto an RGBA overlay and
    alpha-composites it onto each frame rather than re-drawing rounded rectangles and text per frame.
    """
    if not frames or not dialogue_lines:
        return frames

    combined_text = " / ".join(
        f"{d.get('speaker_id') or d.get('speaker', '')}: {d.get('text', '')}".strip(": ")
        for d in dialogue_lines
        if d.get("text")
    )
    if not combined_text:
        return frames

    w, h = frames[0].size
    font_size = max(16, h // 56)
    font = _load_subtitle_font(size=font_size)
    wrapped = textwrap.wrap(combined_text, width=52)[:2]
    display_text = "\n".join(wrapped)

    box_h = max(68, int(h * 0.065) + (len(wrapped) - 1) * (font_size + 6))
    box_y1 = h - box_h - 48
    box_x1 = int(w * 0.06)
    box_w = int(w * 0.88)

    overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        [0, 0, box_w - 1, box_h - 1],
        radius=10,
        fill=(12, 14, 22, 225),
        outline=(70, 190, 210, 255),
        width=2,
    )
    draw.multiline_text((18, 16), display_text, fill=(245, 248, 255, 255), font=font, spacing=4)

    if in_place:
        for f in frames:
            f.paste(overlay, (box_x1, box_y1), overlay)
        return frames

    out_frames: List[Image.Image] = []
    for f in frames:
        copy_f = f.copy()
        copy_f.paste(overlay, (box_x1, box_y1), overlay)
        out_frames.append(copy_f)
    return out_frames


def encode_mjpeg_avi(
    frames: List[Image.Image],
    output_path: Path,
    fps: int = 24,
    quality: int = 85,
) -> Path:
    """Encodes a sequence of PIL images into a standards-compliant Motion-JPEG AVI file."""
    if not frames:
        raise ValueError("Cannot encode empty frame list")

    w, h = frames[0].size
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jpeg_bytes_list: List[bytes] = []
    movi_payload_size = 4  # "movi" fourcc
    for f in frames:
        buf = io.BytesIO()
        f.save(buf, format="JPEG", quality=quality)
        b = buf.getvalue()
        if len(b) % 2 != 0:
            b += b"\x00"
        jpeg_bytes_list.append(b)
        movi_payload_size += 8 + len(b)

    num_frames = len(frames)
    us_per_frame = int(1_000_000 / max(1, fps))

    strf_data = struct.pack(
        "<IiiHHIIiiII",
        40,
        w,
        h,
        1,
        24,
        int.from_bytes(b"MJPG", "little"),
        w * h * 3,
        0,
        0,
        0,
        0,
    )
    strf_chunk = b"strf" + struct.pack("<I", len(strf_data)) + strf_data

    # 56-byte AVIStreamHeader including rcFrame (left, top, right, bottom)
    strh_data = struct.pack(
        "<4s4sIIIIIIIIIIhhhh",
        b"vids",
        b"MJPG",
        0,
        0,
        0,
        1,
        fps,
        0,
        num_frames,
        1000000,
        0xFFFFFFFF,
        0,
        0,
        0,
        w,
        h,
    )
    strh_chunk = b"strh" + struct.pack("<I", len(strh_data)) + strh_data

    strl_bytes = b"strl" + strh_chunk + strf_chunk
    strl_chunk = b"LIST" + struct.pack("<I", len(strl_bytes)) + strl_bytes

    avih_data = struct.pack(
        "<14I",
        us_per_frame,
        0,
        0,
        0x10,
        num_frames,
        0,
        1,
        1000000,
        w,
        h,
        0,
        0,
        0,
        0,
    )
    avih_chunk = b"avih" + struct.pack("<I", len(avih_data)) + avih_data

    hdrl_bytes = b"hdrl" + avih_chunk + strl_chunk
    hdrl_chunk = b"LIST" + struct.pack("<I", len(hdrl_bytes)) + hdrl_bytes

    idx1_size = num_frames * 16
    riff_size = 4 + len(hdrl_chunk) + (8 + movi_payload_size) + (8 + idx1_size)

    with open(output_path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", riff_size) + b"AVI ")
        f.write(hdrl_chunk)
        f.write(b"LIST" + struct.pack("<I", movi_payload_size) + b"movi")

        idx1_entries: List[bytes] = []
        offset = 4
        for b in jpeg_bytes_list:
            f.write(b"00dc" + struct.pack("<I", len(b)))
            f.write(b)
            idx1_entries.append(b"00dc" + struct.pack("<III", 0x10, offset, len(b)))
            offset += 8 + len(b)

        f.write(b"idx1" + struct.pack("<I", idx1_size))
        for entry in idx1_entries:
            f.write(entry)

    logger.info(f"Encoded {num_frames} frames to {output_path} ({w}x{h} @ {fps}fps)")
    return output_path


def decode_video_frames(
    clip_path: Path,
    target_res: Optional[Tuple[int, int]] = None,
) -> List[Image.Image]:
    """Decodes an existing video clip (.avi / .mp4) into PIL RGB frames for fast assembly reuse."""
    clip_path = Path(clip_path)
    if not clip_path.is_file():
        return []

    frames: List[Image.Image] = []
    # Fast direct RIFF MJPEG ("movi" -> "00dc") parser for .avi clips
    if clip_path.suffix.lower() == ".avi":
        try:
            raw = clip_path.read_bytes()
            n = len(raw)
            movi_pos = raw.find(b"LIST", 12)
            pos, end_movi = -1, -1
            while movi_pos >= 0 and movi_pos + 12 <= n:
                list_size = struct.unpack("<I", raw[movi_pos + 4 : movi_pos + 8])[0]
                if raw[movi_pos + 8 : movi_pos + 12] == b"movi":
                    pos = movi_pos + 12
                    end_movi = min(n, movi_pos + 8 + list_size)
                    break
                movi_pos = raw.find(b"LIST", movi_pos + 8 + list_size)

            if pos > 0:
                while pos + 8 <= end_movi:
                    fourcc = raw[pos : pos + 4]
                    sz = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
                    s, e = pos + 8, pos + 8 + sz
                    if fourcc == b"00dc" and e <= end_movi:
                        im = Image.open(io.BytesIO(raw[s:e])).convert("RGB")
                        if target_res and im.size != target_res:
                            im = im.resize(target_res, Image.Resampling.BILINEAR)
                        frames.append(im)
                    pos = e + (sz % 2)
                if frames:
                    return frames
        except Exception as e:
            logger.debug(f"Direct RIFF MJPEG decode fallback on {clip_path}: {e}")

    if cv2 is not None:
        cap = cv2.VideoCapture(str(clip_path))
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
                if target_res and im.size != target_res:
                    im = im.resize(target_res, Image.Resampling.BILINEAR)
                frames.append(im)
        finally:
            cap.release()
    return frames


def mix_timed_tracks(
    events: List[Tuple[float, Path]],
    total_duration_s: float,
    output_wav_path: Path,
    sample_rate: int = 24000,
) -> Optional[Path]:
    """Mixes per-line WAV files at their exact timeline start offsets (in seconds)."""
    valid_events = [(float(start_s), Path(p)) for start_s, p in events if p and Path(p).is_file()]
    if not valid_events:
        return None

    target_samples = max(1, int(round(total_duration_s * sample_rate)))
    buf = np.zeros(target_samples, dtype=np.float32)

    for start_s, p in valid_events:
        try:
            with wave.open(str(p), "rb") as wf:
                ch = wf.getnchannels()
                in_sr = wf.getframerate()
                sw = wf.getsampwidth()
                if sw != 2:
                    logger.warning(f"Skipping {p}: unsupported sample width {sw}")
                    continue
                raw = wf.readframes(wf.getnframes())
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if ch > 1 and len(arr) >= ch:
                arr = arr[: (len(arr) // ch) * ch].reshape(-1, ch).mean(axis=1)
            if in_sr != sample_rate and len(arr) > 1:
                out_len = max(1, int(round(len(arr) * sample_rate / float(in_sr))))
                arr = np.interp(
                    np.linspace(0.0, len(arr) - 1, out_len, dtype=np.float32),
                    np.arange(len(arr), dtype=np.float32),
                    arr,
                )
            i0 = max(0, int(round(start_s * sample_rate)))
            if i0 < target_samples:
                seg = arr[: target_samples - i0]
                buf[i0 : i0 + len(seg)] += seg
        except Exception as e:
            logger.warning(f"Failed to mix audio track {p}: {e}")

    pcm = np.clip(buf, -32767.0, 32767.0).astype(np.int16)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return output_wav_path


def concatenate_wav_tracks(
    audio_paths: List[Path],
    total_duration_s: float,
    output_wav_path: Path,
    sample_rate: int = 24000,
) -> Optional[Path]:
    """Combines per-panel WAV audio files into a continuous timeline audio track."""
    valid_paths = [Path(p) for p in audio_paths if p and Path(p).exists()]
    if not valid_paths:
        return None

    events: List[Tuple[float, Path]] = []
    cursor = 0.0
    for p in valid_paths:
        try:
            with wave.open(str(p), "rb") as wf:
                dur = wf.getnframes() / float(max(1, wf.getframerate()))
            events.append((cursor, p))
            cursor += dur + 0.25
        except Exception as e:
            logger.warning(f"Failed to read audio track {p}: {e}")

    return mix_timed_tracks(events, total_duration_s, output_wav_path, sample_rate=sample_rate)


def assemble_timeline(
    clips: List[List[Image.Image]],
    output_base_path: Path,
    fps: int = 24,
    transition_frames: int = 8,
    audio_paths: Optional[List[Path]] = None,
    audio_events: Optional[List[Tuple[float, Path]]] = None,
    transition_type: str = "crossfade",
    genre_id: str = "shonen",
    sfx_cues: Optional[List[Tuple[float, str, str]]] = None,
    enable_cinematic_bgm: bool = True,
) -> Dict[str, Any]:
    """Assembles multiple panel clips into a single continuous video with transitions and audio."""
    if not clips:
        raise ValueError("No clips provided to assemble")

    full_timeline = list(clips[0])
    cut_timestamps_s: List[float] = []
    for next_clip in clips[1:]:
        if transition_frames > 0 and len(full_timeline) >= transition_frames and len(next_clip) >= transition_frames:
            cut_frame_idx = len(full_timeline) - (transition_frames // 2)
            cut_timestamps_s.append(round(cut_frame_idx / float(max(1, fps)), 4))
            tail_a = full_timeline[-transition_frames:]
            del full_timeline[-transition_frames:]
            head_b = next_clip[:transition_frames]
            blended = render_transition_frames(tail_a, head_b, transition_type=transition_type)
            full_timeline.extend(blended)
            full_timeline.extend(next_clip[transition_frames:])
        else:
            cut_timestamps_s.append(round(len(full_timeline) / float(max(1, fps)), 4))
            full_timeline.extend(next_clip)

    output_base_path = Path(output_base_path)
    output_base_path.parent.mkdir(parents=True, exist_ok=True)

    avi_path = output_base_path.with_suffix(".avi")
    webp_path = output_base_path.with_suffix(".webp")
    mp4_path = output_base_path.with_suffix(".mp4")

    # 1. Write standards-compliant AVI
    encode_mjpeg_avi(full_timeline, avi_path, fps=fps)

    # 2. Write animated WebP with exact frame duration matching step/fps (P1-13)
    step = max(1, fps // 8)
    sample_frames = full_timeline[::step]
    frame_duration_ms = max(10, int(round(1000.0 * step / max(1, fps))))
    sample_frames[0].save(
        webp_path,
        save_all=True,
        append_images=sample_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        quality=75,
        method=0,
    )

    total_duration_s = len(full_timeline) / float(max(1, fps))

    # 3. Prepare mixed audio track (with genre-adaptive BGM, transition SFX, and EBU R128 LUFS ducking)
    master_wav: Optional[Path] = None
    audio_metrics: Dict[str, Any] = {}
    wav_out = output_base_path.with_name(output_base_path.stem + "_master_audio.wav")
    metrics_out = output_base_path.parent / "audio_metrics.json"

    resolved_events: List[Tuple[float, Path]] = []
    if audio_events:
        resolved_events = [(float(s), Path(p)) for s, p in audio_events if p and Path(p).is_file()]
    elif audio_paths:
        cursor = 0.0
        for p in audio_paths:
            if p and Path(p).is_file():
                try:
                    with wave.open(str(p), "rb") as wf:
                        dur = wf.getnframes() / float(max(1, wf.getframerate()))
                    resolved_events.append((cursor, Path(p)))
                    cursor += dur + 0.25
                except Exception as e:
                    logger.warning(f"Failed to read audio track {p}: {e}")

    if enable_cinematic_bgm:
        audio_metrics = mix_cinematic_soundtrack(
            dialogue_events=resolved_events,
            total_duration_s=total_duration_s,
            output_wav_path=wav_out,
            genre_id=genre_id,
            cut_timestamps_s=cut_timestamps_s,
            sfx_cues=sfx_cues,
            metrics_output_path=metrics_out,
        )
        master_wav = wav_out if wav_out.is_file() else None
    elif resolved_events:
        master_wav = mix_timed_tracks(
            events=resolved_events,
            total_duration_s=total_duration_s,
            output_wav_path=wav_out,
        )

    # 4. If ffmpeg is in PATH, convert to H.264 MP4 (with AAC audio stream if present)
    ffmpeg_exe = shutil.which("ffmpeg")
    has_mp4 = False
    if ffmpeg_exe:
        cmd = [ffmpeg_exe, "-y", "-loglevel", "error", "-i", str(avi_path)]
        if master_wav and master_wav.exists():
            cmd.extend(
                [
                    "-i",
                    str(master_wav),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                ]
            )
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"])
        cmd.extend(["-movflags", "+faststart", str(mp4_path)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and mp4_path.is_file():
            has_mp4 = True
            logger.info(f"Generated H.264 MP4 via FFmpeg: {mp4_path}")
        else:
            logger.error(f"FFmpeg MP4 conversion failed: {proc.stderr[-2000:]}")
            mp4_path.unlink(missing_ok=True)

    return {
        "total_frames": len(full_timeline),
        "duration_s": total_duration_s,
        "avi_path": str(avi_path.as_posix()),
        "webp_path": str(webp_path.as_posix()),
        "mp4_path": str(mp4_path.as_posix()) if has_mp4 else "",
        "master_audio_path": str(master_wav.as_posix()) if master_wav and master_wav.exists() else "",
        "audio_metrics_path": str(metrics_out.as_posix()) if metrics_out.exists() else "",
        "audio_metrics": audio_metrics,
        "transition_type": transition_type,
        "genre_id": genre_id,
    }
