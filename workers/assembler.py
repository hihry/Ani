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

    # 1. Load explicit binary mask if provided
    if mask_path is not None and Path(mask_path).is_file():
        m_img = Image.open(mask_path).convert("L").resize((w, h), Image.Resampling.NEAREST)
        mask = (np.asarray(m_img, dtype=np.uint8) > 127).astype(np.uint8) * 255

    def _add_tight_bubble_mask(bx0: int, by0: int, bw0: int, bh0: int, pad_px: int = 6) -> None:
        x1 = max(0, bx0)
        y1 = max(0, by0)
        x2 = min(w, bx0 + bw0)
        y2 = min(h, by0 + bh0)
        if x2 - x1 <= 8 or y2 - y1 <= 8:
            return
        if cv2 is not None:
            sub = arr[y1:y2, x1:x2]
            r_c, g_c, b_c = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
            max_c = cv2.max(cv2.max(r_c, g_c), b_c)
            min_c = cv2.min(cv2.min(r_c, g_c), b_c)
            chroma_u8 = cv2.subtract(max_c, min_c)
            gray_u16 = r_c.astype(np.uint16) + g_c.astype(np.uint16) + b_c.astype(np.uint16)
            sub_white = ((gray_u16 >= 645) & (chroma_u8 <= 36)).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(sub_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sub_area = float((y2 - y1) * (x2 - x1))
            valid_cnts = [c for c in cnts if cv2.contourArea(c) >= max(64.0, sub_area * 0.08)]
            if valid_cnts:
                local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                cv2.drawContours(local_mask, valid_cnts, -1, 255, thickness=cv2.FILLED)
                k_sz = max(5, 2 * pad_px + 1)
                dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
                local_mask = cv2.dilate(local_mask, dil_k, iterations=1)
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], local_mask)
                return
        mask[y1:y2, x1:x2] = 255

    # 2. Mask explicit dialogue / SFX bounding boxes using tight bubble contour masking
    if bubble_bboxes:
        for b in bubble_bboxes:
            bx = max(0, int(b.get("x", 0)) - 8)
            by = max(0, int(b.get("y", 0)) - 8)
            bw = int(b.get("w", 0)) + 16
            bh = int(b.get("h", 0)) + 22
            if bw > 16 and bh > 16:
                _add_tight_bubble_mask(bx, by, bw, bh, pad_px=9)

    # 3. Always union with Phase 2 morphological bubble detector envelope when no precomputed mask_path was given
    if mask_path is None:
        detected = detect_speech_bubbles_and_text(img_pil)
        for reg in detected:
            eb = reg.get("envelope_bbox") or reg["bbox"]
            ex = max(0, int(eb["x"]) - 6)
            ey = max(0, int(eb["y"]) - 6)
            ew = int(eb["w"]) + 12
            eh = int(eb["h"]) + 12
            _add_tight_bubble_mask(ex, ey, ew, eh, pad_px=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if cv2 is not None and np.any(mask > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Pre-fill masked spans using 2D inverse-distance harmonic boundary interpolation
        clean_arr = arr.copy()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        frame_pad = min(6, max(2, min(w, h) // 40))
        for lbl in range(1, num_labels):
            rx = int(stats[lbl, cv2.CC_STAT_LEFT])
            ry = int(stats[lbl, cv2.CC_STAT_TOP])
            rw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            rh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            ry2 = min(h, ry + rh)
            rx2 = min(w, rx + rw)
            ny = ry2 - ry
            nx = rx2 - rx
            if ny <= 0 or nx <= 0:
                continue

            has_left = (rx - 4) >= frame_pad
            has_right = (rx + rw + 3) < (w - frame_pad)
            lx = max(frame_pad, rx - 4)
            tx = min(w - frame_pad - 1, rx + rw + 3)
            top_y = max(frame_pad, ry - 4)
            bot_y = min(h - frame_pad - 1, ry + rh + 3)

            comp_mask_2d = labels[ry:ry2, rx:rx2] == lbl
            if has_left or has_right:
                left_col = np.zeros((ny, 3), dtype=np.float32)
                right_col = np.zeros((ny, 3), dtype=np.float32)
                x_starts = np.zeros(ny, dtype=np.float32)
                x_ends = np.full(ny, float(nx - 1), dtype=np.float32)
                is_gutter_row = np.zeros(ny, dtype=bool)

                panel_gray = clean_arr.mean(axis=2)
                panel_chroma = clean_arr.max(axis=2).astype(np.int16) - clean_arr.min(axis=2).astype(np.int16)
                panel_white_rows = ((panel_gray >= 240.0) & (panel_chroma <= 18)).mean(axis=1)
                top_gutter_y = 0
                while top_gutter_y < int(h * 0.35) and panel_white_rows[top_gutter_y] >= 0.52:
                    top_gutter_y += 1

                for r_idx in range(ny):
                    abs_y = ry + r_idx
                    row_m = np.where(comp_mask_2d[r_idx])[0]
                    if len(row_m) > 0:
                        x_s, x_e = int(row_m[0]), int(row_m[-1])
                        x_starts[r_idx] = float(x_s)
                        x_ends[r_idx] = float(x_e)
                        c_left = max(frame_pad, rx + x_s - 7) if has_left else min(w - frame_pad - 1, rx + x_e + 7)
                        c_right = min(w - frame_pad - 1, rx + x_e + 7) if has_right else max(frame_pad, rx + x_s - 7)
                    else:
                        c_left = lx if has_left else tx
                        c_right = tx if has_right else lx
                    lc = clean_arr[abs_y, c_left].astype(np.float32)
                    rc = clean_arr[abs_y, c_right].astype(np.float32)
                    l_white = (lc.mean() >= 240.0) and ((lc.max() - lc.min()) <= 18.0)
                    r_white = (rc.mean() >= 240.0) and ((rc.max() - rc.min()) <= 18.0)

                    if abs_y < top_gutter_y and (l_white or r_white):
                        is_gutter_row[r_idx] = True
                        left_col[r_idx] = (255.0, 255.0, 255.0)
                        right_col[r_idx] = (255.0, 255.0, 255.0)
                        continue

                    l_lum = float(lc.mean())
                    r_lum = float(rc.mean())
                    if abs(l_lum - r_lum) > 25.0 and max(l_lum, r_lum) >= 135.0:
                        if l_lum > r_lum:
                            rc = lc.copy()
                        else:
                            lc = rc.copy()

                    left_col[r_idx] = lc
                    right_col[r_idx] = rc

                art_rows = np.where(~is_gutter_row)[0]
                if len(art_rows) >= 5:
                    ay1, ay2 = int(art_rows[0]), int(art_rows[-1]) + 1
                    ky_col = max(7, min(61, ((ay2 - ay1) // 2) | 1))
                    if ky_col % 2 == 0:
                        ky_col += 1
                    left_col[ay1:ay2] = cv2.GaussianBlur(left_col[ay1:ay2, None, :], (1, ky_col), 0)[:, 0, :]
                    right_col[ay1:ay2] = cv2.GaussianBlur(right_col[ay1:ay2, None, :], (1, ky_col), 0)[:, 0, :]

                grid_x = np.arange(nx, dtype=np.float32)[None, :]
                span = np.maximum(1.0, (x_ends - x_starts)[:, None])
                u_x = np.clip((grid_x - x_starts[:, None]) / span, 0.0, 1.0)[:, :, None]
                block = np.clip((1.0 - u_x) * left_col[:, None, :] + u_x * right_col[:, None, :], 0.0, 255.0).astype(np.uint8)
                block[is_gutter_row] = 255

                if len(art_rows) >= 5:
                    ay1, ay2 = int(art_rows[0]), int(art_rows[-1]) + 1
                    kx = max(5, min(41, (nx // 3) | 1))
                    ky = max(5, min(41, ((ay2 - ay1) // 3) | 1))
                    if kx % 2 == 0:
                        kx += 1
                    if ky % 2 == 0:
                        ky += 1
                    block[ay1:ay2] = cv2.GaussianBlur(block[ay1:ay2], (kx, ky), 0)
            else:
                top_row = clean_arr[top_y, rx:rx2, :].astype(np.float32)[None, :, :]
                bot_row = clean_arr[bot_y, rx:rx2, :].astype(np.float32)[None, :, :]
                alphas_y = np.linspace(0.0, 1.0, ny, dtype=np.float32)[:, None, None]
                block = np.clip((1.0 - alphas_y) * top_row + alphas_y * bot_row, 0.0, 255.0).astype(np.uint8)

            sub_view = clean_arr[ry:ry2, rx:rx2]
            sub_view[comp_mask_2d] = block[comp_mask_2d]

        # Smooth boundary transition ring via Telea inpainting across a dilated/eroded collar
        outer_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        inner_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dil_m = cv2.dilate(mask, outer_k, iterations=1)
        ero_m = cv2.erode(mask, inner_k, iterations=1)
        boundary_ring = cv2.subtract(dil_m, ero_m)
        if np.any(boundary_ring > 0):
            bgr = cv2.cvtColor(clean_arr, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(bgr, boundary_ring, 9, cv2.INPAINT_TELEA)
            clean_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        else:
            clean_rgb = clean_arr
        out_img = Image.fromarray(clean_rgb)
    else:
        out_img = img_pil

    # Persist binary inpaint mask artifact alongside clean plate when requested or following *_clean.png convention
    resolved_mask_out = mask_output_path
    if resolved_mask_out is None and output_path.name.endswith("_clean.png"):
        resolved_mask_out = output_path.with_name(output_path.name[:-10] + "_mask.png")
    if resolved_mask_out is not None:
        Image.fromarray(mask, mode="L").save(Path(resolved_mask_out), format="PNG", compress_level=1)

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
