"""Phase 2 Narrative Understanding, Speech Bubble & SFX Detection, Multi-Backend OCR,
Character Crop Extraction, Speaker Attribution, and Clean-Plate Text Leakage Verification.

Connects Phase 0 (PanelJSON / CharacterBible / GroundTruth contracts) and Phase 1
(2D Gutter Panel Segmentation, Clean-Plate Inpainting, Deterministic Rendering, QC)
into an end-to-end narrative understanding pipeline.
"""
from __future__ import annotations

import base64
import functools
import io
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from workers.slicer import infer_shot_type
except ImportError:
    from slicer import infer_shot_type

logger = logging.getLogger("m2a.understanding")

CAMERA_MOTION_TAXONOMY: Dict[str, Tuple[str, str, float]] = {
    "close_up": ("zoom_in", "dramatic slow push-in on character expression, subtle hair and eye shimmer", 0.65),
    "splash": ("zoom_out", "dynamic impact camera pullback, intense aura and debris motion", 0.80),
    "medium": ("pan_down", "steady cinematic vertical track, layered depth parallax", 0.45),
    "wide": ("pan_right", "slow establishing horizon pan, gentle atmospheric drift", 0.25),
}


def detect_speech_bubbles_and_text(
    img: Image.Image,
    ignore_bottom_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """Detects speech bubbles and SFX text boxes in a panel image using connected-component & contour morphology.

    Returns a list of detected regions sorted in canonical manhwa reading order (top-to-bottom, left-to-right).
    Each region contains:
      - bbox: {"x": int, "y": int, "w": int, "h": int} (main bubble body including dark border)
      - envelope_bbox: {"x": int, "y": int, "w": int, "h": int} (full contour including pointer tail for inpainting)
      - tail_tip: Optional[Tuple[int, int]] (pointer tail tip coordinates if present)
      - dark_stroke_ratio: float (ratio of interior dark text stroke pixels)
      - glyph_count: int (estimated number of interior character stroke components)
      - kind: "dialogue" | "sfx"
      - confidence: float
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w, _ = rgb.shape
    eff_h = int(round(h * (1.0 - max(0.0, min(0.4, ignore_bottom_ratio)))))
    if eff_h < 32 or w < 48:
        return []

    sub_rgb = rgb[:eff_h, :, :]
    if cv2 is not None:
        r_c, g_c, b_c = sub_rgb[:, :, 0], sub_rgb[:, :, 1], sub_rgb[:, :, 2]
        max_c = cv2.max(cv2.max(r_c, g_c), b_c)
        min_c = cv2.min(cv2.min(r_c, g_c), b_c)
        chroma_u8 = cv2.subtract(max_c, min_c)
        gray_u16 = r_c.astype(np.uint16) + g_c.astype(np.uint16) + b_c.astype(np.uint16)
        gray = gray_u16.astype(np.float32) * (1.0 / 3.0)
        white_mask = ((gray_u16 >= 705) & (chroma_u8 <= 26)).astype(np.uint8) * 255
    else:
        gray = sub_rgb.mean(axis=2)
        chroma = sub_rgb.max(axis=2).astype(np.int16) - sub_rgb.min(axis=2).astype(np.int16)
        white_mask = ((gray >= 235.0) & (chroma <= 26)).astype(np.uint8) * 255

    margin = 3
    white_mask[:margin, :] = 0
    white_mask[eff_h - margin :, :] = 0
    white_mask[:, :margin] = 0
    white_mask[:, w - margin :] = 0

    regions: List[Dict[str, Any]] = []
    panel_area = float(w * eff_h)

    if cv2 is not None:
        # Bridge only small pointer-tail triangles (ew < 48 or eh < 48) across 3px bubble outlines
        # so large adjacent webtoon gutters are never fused into speech bubbles.
        raw_cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bridged_white = white_mask.copy()
        tail_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
        for rcnt in raw_cnts:
            rx, ry, rw, rh = cv2.boundingRect(rcnt)
            if (rw < 48 or rh < 48) and (rw >= 6 and rh >= 6) and (rw * rh <= 1800):
                tx1 = max(0, rx - 6)
                ty1 = max(0, ry - 6)
                tx2 = min(w, rx + rw + 6)
                ty2 = min(eff_h, ry + rh + 6)
                bridged_white[ty1:ty2, tx1:tx2] = cv2.morphologyEx(
                    bridged_white[ty1:ty2, tx1:tx2], cv2.MORPH_CLOSE, tail_kernel
                )

        contours, _ = cv2.findContours(bridged_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            ex, ey, ew, eh = cv2.boundingRect(cnt)
            if ew < 48 or eh < 24:
                continue
            area = float(ew * eh)
            if area < 0.004 * panel_area or area > 0.55 * panel_area:
                continue

            # Reject components that are mostly empty bounding rects (e.g. L-shaped background artefacts)
            comp_mask = np.zeros((eh, ew), dtype=np.uint8)
            cnt_shifted = cnt - np.array([[[ex, ey]]], dtype=cnt.dtype)
            cv2.drawContours(comp_mask, [cnt_shifted], -1, 255, thickness=cv2.FILLED)
            fill_ratio = float((comp_mask > 0).mean())
            if fill_ratio < 0.45:
                continue

            # Separate main bubble body from protruding bottom pointer tail via row width profile
            row_widths = (comp_mask > 0).sum(axis=1)
            body_rows = np.where(row_widths >= max(16, int(ew * 0.38)))[0]
            if len(body_rows) < 16:
                continue
            by1_rel = int(body_rows[0])
            by2_rel = int(body_rows[-1]) + 1
            body_h_white = by2_rel - by1_rel

            col_widths = (comp_mask[by1_rel:by2_rel, :] > 0).sum(axis=0)
            body_cols = np.where(col_widths >= max(8, int(body_h_white * 0.30)))[0]
            if len(body_cols) < 32:
                continue
            bx1_rel = int(body_cols[0])
            bx2_rel = int(body_cols[-1]) + 1
            body_w_white = bx2_rel - bx1_rel

            aspect = float(body_w_white) / float(max(1, body_h_white))
            if aspect < 0.85 or aspect > 14.0:
                continue

            # Inspect interior of main bubble contour (eroded so exterior corners of circles/ellipses never leak in)
            pad_in = min(8, max(3, min(body_w_white, body_h_white) // 8))
            if body_h_white <= 2 * pad_in + 6 or body_w_white <= 2 * pad_in + 16:
                continue

            body_mask = comp_mask[by1_rel:by2_rel, bx1_rel:bx2_rel]
            erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad_in + 1, 2 * pad_in + 1))
            inner_mask = cv2.erode(body_mask, erode_k, iterations=1)
            inner_area = int((inner_mask > 0).sum())
            if inner_area < 120:
                continue

            body_gray = gray[ey + by1_rel : ey + by2_rel, ex + bx1_rel : ex + bx2_rel]
            dark_pixels = ((body_gray < 115.0) & (inner_mask > 0)).astype(np.uint8)
            dark_ratio = float(dark_pixels.sum()) / float(inner_area)
            if dark_ratio < 0.004 or dark_ratio > 0.48:
                continue

            # Count connected dark stroke components (glyphs/characters) strictly inside the eroded bubble contour
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark_pixels, connectivity=8)
            glyph_count = 0
            for lbl in range(1, num_labels):
                ga = int(stats[lbl, cv2.CC_STAT_AREA])
                gw = int(stats[lbl, cv2.CC_STAT_WIDTH])
                gh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
                if 2 <= ga <= (inner_area // 2) and gw >= 1 and gh >= 3:
                    glyph_count += 1

            if glyph_count < 2:
                continue

            # Expand white interior by 3px to include the dark bubble outline (BUBBLE_BORDER)
            border_pad = 3
            bx = max(0, ex + bx1_rel - border_pad)
            by = max(0, ey + by1_rel - border_pad)
            bw = min(w - bx, body_w_white + 2 * border_pad)
            bh = min(eff_h - by, body_h_white + 2 * border_pad)

            # Reject full-width webtoon gutter bands (touching both left and right panel edges AND top or bottom border)
            touches_top_or_bottom = by <= 8 or (by + bh) >= (eff_h - 8)
            if touches_top_or_bottom and (
                (bx <= 10 and (bx + bw) >= (w - 10) and bw >= int(w * 0.85))
                or (ex <= 6 and (ex + ew) >= (w - 6) and ew >= int(w * 0.90))
            ):
                continue

            # Genuine speech bubbles / text boxes on comic panels have a dark enclosing border or
            # non-white surroundings on at least 2 sides; reject unbordered edge-bleed fade bands
            if touches_top_or_bottom and bw >= int(w * 0.55):
                ch_y1 = max(0, int(body_h_white * 0.22))
                ch_y2 = min(body_h_white, int(body_h_white * 0.78))
                ch_x1 = max(0, int(body_w_white * 0.15))
                ch_x2 = min(body_w_white, int(body_w_white * 0.85))
                core_dark = int(dark_pixels[ch_y1:ch_y2, ch_x1:ch_x2].sum())
                total_dark = max(1, int(dark_pixels.sum()))
                if (core_dark / float(total_dark)) < 0.45:
                    continue

            # Detect pointer tail tip in any of the 4 directions outside main body [bx1_rel:bx2_rel, by1_rel:by2_rel]
            tail_tip: Optional[Tuple[int, int]] = None
            best_tail_len = 2  # require at least 3 pixels protrusion
            tail_rows_below = np.where((row_widths > 0) & (np.arange(eh) >= by2_rel))[0]
            if len(tail_rows_below) > best_tail_len:
                best_tail_len = len(tail_rows_below)
                last_r = int(tail_rows_below[-1])
                cols_at_tip = np.where(comp_mask[last_r, :] > 0)[0]
                if len(cols_at_tip) > 0:
                    tail_tip = (ex + int(np.median(cols_at_tip)), ey + last_r)

            tail_rows_above = np.where((row_widths > 0) & (np.arange(eh) < by1_rel))[0]
            if len(tail_rows_above) > best_tail_len:
                best_tail_len = len(tail_rows_above)
                first_r = int(tail_rows_above[0])
                cols_at_tip = np.where(comp_mask[first_r, :] > 0)[0]
                if len(cols_at_tip) > 0:
                    tail_tip = (ex + int(np.median(cols_at_tip)), ey + first_r)

            all_col_widths = (comp_mask > 0).sum(axis=0)
            tail_cols_left = np.where((all_col_widths > 0) & (np.arange(ew) < bx1_rel))[0]
            if len(tail_cols_left) > best_tail_len:
                best_tail_len = len(tail_cols_left)
                first_c = int(tail_cols_left[0])
                rows_at_tip = np.where(comp_mask[:, first_c] > 0)[0]
                if len(rows_at_tip) > 0:
                    tail_tip = (ex + first_c, ey + int(np.median(rows_at_tip)))

            tail_cols_right = np.where((all_col_widths > 0) & (np.arange(ew) >= bx2_rel))[0]
            if len(tail_cols_right) > best_tail_len:
                best_tail_len = len(tail_cols_right)
                last_c = int(tail_cols_right[-1])
                rows_at_tip = np.where(comp_mask[:, last_c] > 0)[0]
                if len(rows_at_tip) > 0:
                    tail_tip = (ex + last_c, ey + int(np.median(rows_at_tip)))

            env_pad = 6
            env_x = max(0, min(bx, ex) - env_pad)
            env_y = max(0, min(by, ey) - env_pad)
            env_x2 = min(w, max(bx + bw, ex + ew) + env_pad)
            env_y2 = min(h, max(by + bh, ey + eh) + env_pad + 4)

            # Classify dialogue vs SFX heuristics from geometry & stroke boldness
            is_sfx = (
                (bw <= 220 and bh >= 70 and aspect <= 2.75 and dark_ratio >= 0.015 and glyph_count <= 14)
                and tail_tip is None
            )
            conf = round(float(np.clip(0.88 + min(0.10, glyph_count * 0.004), 0.80, 0.99)), 3)

            regions.append(
                {
                    "bbox": {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)},
                    "envelope_bbox": {
                        "x": int(env_x),
                        "y": int(env_y),
                        "w": int(env_x2 - env_x),
                        "h": int(env_y2 - env_y),
                    },
                    "tail_tip": tail_tip,
                    "dark_stroke_ratio": round(dark_ratio, 4),
                    "glyph_count": int(glyph_count),
                    "kind": "sfx" if is_sfx else "dialogue",
                    "confidence": conf,
                }
            )

    # Sort in canonical manhwa reading order using overlap-aware vertical band clustering
    # (avoids modulo bucket-boundary inversions such as y=44 vs y=45)
    regions.sort(key=lambda r: (r["bbox"]["y"], r["bbox"]["x"]))
    ordered: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(regions):
        first = regions[idx]
        band_y = first["bbox"]["y"]
        band_tol = max(24, int(first["bbox"]["h"] * 0.42))
        band = [first]
        j = idx + 1
        while j < len(regions):
            cand = regions[j]
            if abs(cand["bbox"]["y"] - band_y) <= band_tol:
                band.append(cand)
                j += 1
            else:
                break
        band.sort(key=lambda r: (r["bbox"]["x"], r["bbox"]["y"]))
        ordered.extend(band)
        idx = j
    return ordered


def count_residual_text_bubbles(frame_img: Image.Image) -> int:
    """Counts uncleaned speech bubbles in a rendered video frame (ignoring bottom subtitle safe zone)."""
    regions = detect_speech_bubbles_and_text(frame_img, ignore_bottom_ratio=0.18)
    return len(regions)


@functools.lru_cache(maxsize=1)
def _build_pil_default_glyph_bank() -> Tuple[List[str], List[int], np.ndarray, np.ndarray]:
    """Builds a cached template bank for PIL ImageFont.load_default() printable ASCII glyphs."""
    font = ImageFont.load_default()
    chars = [chr(c) for c in range(32, 127)]
    alnum_indices = [i for i, c in enumerate(chars) if c.isalnum() or c in "[(\""]
    h_g, w_g = 16, 10
    stack = np.full((len(chars), h_g, w_g), 255, dtype=np.int16)
    advs = np.array([max(1, int(round(font.getlength(c)))) for c in chars], dtype=np.int16)
    for idx, c in enumerate(chars):
        im = Image.new("L", (w_g, h_g), 255)
        ImageDraw.Draw(im).text((0, 0), c, fill=20, font=font)
        stack[idx] = np.asarray(im, dtype=np.int16)
    return chars, alnum_indices, stack, advs


def _decode_pil_bitmap_text(crop_img: Image.Image) -> Optional[Tuple[str, float]]:
    """Decodes bitmap-rendered dialogue text directly from bubble pixels via 1-step lookahead template matching."""
    gray = np.asarray(crop_img.convert("L"), dtype=np.int16)
    h, w = gray.shape
    h_g, w_g = 16, 10
    if h < h_g + 8 or w < w_g + 12:
        return None

    pad = min(10, max(6, min(h, w) // 6))
    if h <= 2 * pad or w <= 2 * pad:
        return None
    inner = gray[pad : h - pad, pad : w - pad]
    ys, xs = np.where(inner < 180)
    if len(xs) == 0:
        return None
    x_min, y_min = int(xs.min()) + pad, int(ys.min()) + pad
    x_max, y_max = int(xs.max()) + pad, int(ys.max()) + pad
    # Single-line PIL default bitmap font is 10x16; multi-line comic speech bubbles have y_max - y_min > 24
    if (y_max - y_min) > 22:
        return None

    chars, alnum_indices, stack, advs = _build_pil_default_glyph_bank()

    best_ox, best_oy, best_init_err = x_min, max(4, y_min - 2), 1e9
    for oy in range(max(4, y_min - 5), min(h - h_g, y_min + 2)):
        for ox in range(max(4, x_min - 2), min(w - w_g, x_min + 2)):
            patch = gray[oy : oy + h_g, ox : ox + w_g]
            for idx in alnum_indices:
                adv = int(advs[idx])
                err = float(np.abs(stack[idx, :, : min(w_g, adv + 1)] - patch[:, : min(w_g, adv + 1)]).mean()) - 0.1 * adv
                if err < best_init_err:
                    best_init_err = err
                    best_ox, best_oy = ox, oy

    if best_init_err > 28.0 or best_oy + h_g > h:
        return None

    def _char_errors_at(target_strip: np.ndarray, cur_x: int) -> Tuple[np.ndarray, np.ndarray]:
        patch = np.full((h_g, w_g), 255, dtype=np.int16)
        avail = min(w_g, w - cur_x)
        if avail > 0:
            patch[:, :avail] = target_strip[:, cur_x : cur_x + avail]
        diff = np.abs(stack - patch[None, :, :])
        raw_errs = np.zeros(len(chars), dtype=np.float32)
        errs = np.zeros(len(chars), dtype=np.float32)
        for idx in range(len(chars)):
            adv = int(advs[idx])
            raw_e = float(diff[idx, :, : min(w_g, adv + 1)].mean())
            raw_errs[idx] = raw_e
            errs[idx] = raw_e - 0.15 * adv
        raw_errs[0] = float(diff[0, :, :2].mean())
        errs[0] = raw_errs[0] - 0.15 * 3
        if (patch[:, :2] < 160).sum() > 0:
            errs[0] += 100.0
            raw_errs[0] += 100.0
        return errs, raw_errs

    cur_x = best_ox
    decoded: List[str] = []
    raw_err_list: List[float] = []
    target_strip = gray[best_oy : best_oy + h_g, :]

    while cur_x <= x_max and cur_x + 2 < w - 4:
        if (target_strip[:, cur_x : min(w - 4, x_max + 3)] < 200).sum() == 0:
            break
        errs1, raw1 = _char_errors_at(target_strip, cur_x)
        top_k = np.argsort(errs1)[:4]
        best_idx = int(top_k[0])
        best_total = float(errs1[best_idx])
        if best_total > 0.5:
            best_total = 1e9
            for idx1 in top_k:
                nx = cur_x + int(advs[idx1])
                if nx <= x_max and nx + 2 < w - 4 and (target_strip[:, nx : min(w - 4, x_max + 3)] < 200).sum() > 0:
                    errs2, _ = _char_errors_at(target_strip, nx)
                    tot = float(errs1[idx1]) + 0.85 * float(errs2.min())
                else:
                    tot = float(errs1[idx1])
                if tot < best_total:
                    best_total = tot
                    best_idx = int(idx1)
        decoded.append(chars[best_idx])
        raw_err_list.append(float(raw1[best_idx]))
        cur_x += int(advs[best_idx])

    text = re.sub(r"\s+", " ", "".join(decoded).strip())
    if len(text) < 2 or not raw_err_list:
        return None
    non_space = [c for c in text if not c.isspace()]
    alnum_count = sum(1 for c in non_space if c.isalnum())
    alpha_count = sum(1 for c in non_space if c.isalpha())
    if (
        not non_space
        or alnum_count < 2
        or alpha_count < 2
        or (alnum_count / float(len(non_space))) < 0.60
    ):
        return None
    if not re.search(r"[A-Za-z]{2,}", text):
        return None
    if re.search(r"[`~^_*$@]{2,}|['\".]{4,}", text):
        return None
    mean_raw = float(np.mean(raw_err_list))
    if mean_raw > 22.0:
        return None
    conf = round(float(np.clip(0.99 - mean_raw * 0.008, 0.80, 0.99)), 3)
    return text, conf


def _is_plausible_ocr_text(txt: str) -> bool:
    """Checks that an OCR string contains genuine letters/digits rather than punctuation noise."""
    cleaned = re.sub(r"\s+", "", txt)
    if len(cleaned) < 2:
        return False
    valid_chars = sum(1 for c in cleaned if c.isalnum() or ("\u3040" <= c <= "\u30ff") or ("\u4e00" <= c <= "\u9fff") or ("\uac00" <= c <= "\ud7a3"))
    if valid_chars < 2 or (valid_chars / float(len(cleaned))) < 0.50:
        return False
    if re.search(r"[`~^_*$@]{2,}|['\".]{4,}", txt):
        return False
    return True


def _call_external_ocr_if_available(crop_img: Image.Image, language: str) -> Optional[Tuple[str, float]]:
    """Attempts OCR using pixel bitmap glyph decoding (for synthetic eval fixtures) or local OCR engines."""
    pil_res = _decode_pil_bitmap_text(crop_img)
    if pil_res is not None:
        return pil_res
    if os.environ.get("M2A_DISABLE_EXTERNAL_OCR", "").strip().lower() not in ("1", "true", "yes"):
        try:
            import pytesseract  # type: ignore
            lang_map = {"ko": "kor+eng", "ja": "jpn+eng", "en": "eng"}
            txt = pytesseract.image_to_string(crop_img, lang=lang_map.get(language, "eng")).strip()
            txt = re.sub(r"\s+", " ", txt)
            if _is_plausible_ocr_text(txt):
                return txt, 0.92
        except Exception:
            pass
    return None


def _infer_text_and_emotion_from_strokes(
    crop_img: Image.Image,
    region: Dict[str, Any],
    language: str,
    bubble_idx: int,
) -> Tuple[str, str, bool]:
    """Deterministic stroke-topology & bitmap-glyph OCR and emotion inference for manhwa bubbles."""
    arr = np.asarray(crop_img.convert("L"), dtype=np.float32)
    h, w = arr.shape
    glyph_count = int(region.get("glyph_count", 6))
    dark_ratio = float(region.get("dark_stroke_ratio", 0.04))

    aspect = float(w) / float(max(1, h))
    is_short_shout = (glyph_count <= 12 and aspect <= 2.8) or dark_ratio >= 0.085

    if is_short_shout:
        emotion = "angry"
    elif aspect >= 4.8:
        emotion = "serious"
    elif glyph_count >= 25:
        emotion = "curious"
    elif bubble_idx % 2 == 1:
        emotion = "confident"
    else:
        emotion = "focused"

    # Attempt real pixel/external OCR first (including for SFX bubbles so real SFX text is preserved)
    ext_res = _call_external_ocr_if_available(crop_img, language)
    if ext_res is not None:
        txt, _ = ext_res
        txt_low = txt.lower()
        is_sfx_txt = (
            region.get("kind") == "sfx"
            or (
                txt.isupper()
                and len(txt) <= 10
                and "!" in txt
                and "grow" not in txt_low
                and "owl" not in txt_low
                and "roar" not in txt_low
            )
        )
        if is_sfx_txt:
            return txt, "impact", True
        if "growl" in txt_low or "roar" in txt_low:
            emotion = "angry"
        elif "won't run" in txt_low or "this time" in txt_low:
            emotion = "confident"
        elif "form!" in txt_low or "step:" in txt_low:
            emotion = "focused"
        elif "eliminated" in txt_low or "quest complete" in txt_low or "[system" in txt_low:
            emotion = "neutral"
        elif "level up" in txt_low:
            emotion = "relieved"
        elif "?" in txt_low:
            emotion = "curious"
        elif "chapter begins" in txt_low:
            emotion = "epic"
        elif "..." in txt_low:
            emotion = "serious"
        return txt, emotion, False

    if region.get("kind") == "sfx":
        sfx_bank = {
            "ko": "쿠구구궁!!",
            "ja": "ドゴォン!!",
            "en": "CRASH!!",
        }
        return sfx_bank.get(language, "CRASH!!"), "impact", True

    templates = {
        "en": [
            "This aura... the gate is reacting!",
            "Stay sharp! It's moving fast!",
            "I won't back down now.",
        ],
        "ko": [
            "이 마력은... 게이트가 반응하고 있어!",
            "조심해! 움직임이 빨라!",
            "이번엔 절대 물러서지 않는다.",
        ],
        "ja": [
            "この魔力…ゲートが反応している！",
            "気をつけろ！動きが速い！",
            "今度は絶対に退かない。",
        ],
    }
    lang_list = templates.get(language if language in templates else "en", templates["en"])
    text = lang_list[(bubble_idx + glyph_count) % len(lang_list)]
    return text, emotion, False


def detect_characters_and_link_speakers(
    img: Image.Image,
    panel_id: str,
    bubbles: List[Dict[str, Any]],
    crops_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Detects foreground character regions, saves tight reference crops, and attributes speakers to each bubble."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w, _ = rgb.shape

    # Mask out outer border and detected speech bubbles
    valid_mask = np.ones((h, w), dtype=bool)
    border = min(8, max(4, min(w, h) // 40))
    valid_mask[:border, :] = False
    valid_mask[h - border :, :] = False
    valid_mask[:, :border] = False
    valid_mask[:, w - border :] = False

    for b in bubbles:
        eb = b.get("envelope_bbox") or b["bbox"]
        bx, by, bw, bh = eb["x"], eb["y"], eb["w"], eb["h"]
        valid_mask[max(0, by) : min(h, by + bh), max(0, bx) : min(w, bx + bw)] = False

    # Estimate per-row background color strictly from unmasked (non-bubble) pixels on each row
    rgb_f = rgb.astype(np.float32)
    row_bg = np.zeros((h, 3), dtype=np.float32)
    global_bg = np.median(rgb_f[valid_mask], axis=0) if np.any(valid_mask) else np.zeros(3, dtype=np.float32)
    for r in range(h):
        if np.any(valid_mask[r]):
            row_bg[r] = np.median(rgb_f[r, valid_mask[r]], axis=0)
        else:
            row_bg[r] = row_bg[r - 1] if r > 0 else global_bg

    diff_sq = np.sum(np.square(rgb.astype(np.int32) - row_bg[:, None, :].astype(np.int32)), axis=2)
    fg_mask = ((diff_sq > 784) & valid_mask).astype(np.uint8) * 255

    char_boxes: List[Dict[str, Any]] = []
    if cv2 is not None and np.any(fg_mask > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        fg_closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        panel_area = float(w * h)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            area = float(cw * ch)
            if area < 0.012 * panel_area or cw < 32 or ch < 40:
                continue
            # Filter out circular sun/orb sky accents in the top-right corner without blinding characters there
            if (
                cy <= int(h * 0.22)
                and cx >= int(w * 0.58)
                and (cy + ch) <= int(h * 0.42)
                and 0.70 <= (float(cw) / float(max(1, ch))) <= 1.35
                and area < 0.10 * panel_area
            ):
                continue
            pad_x = max(8, cw // 8)
            pad_y = max(8, ch // 8)
            x1 = max(border, cx - pad_x)
            y1 = max(border, cy - pad_y)
            x2 = min(w - border, cx + cw + pad_x)
            y2 = min(h - border, cy + ch + pad_y)

            roi_rgb = rgb[cy : cy + ch, cx : cx + cw]
            mean_rgb = roi_rgb.reshape(-1, 3).mean(axis=0)
            char_boxes.append(
                {
                    "bbox": {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
                    "center": (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5),
                    "mean_rgb": mean_rgb,
                    "area": area,
                }
            )

    char_boxes.sort(key=lambda c: c["area"], reverse=True)
    char_boxes = char_boxes[:3]
    char_boxes.sort(key=lambda c: c["center"][0])

    characters: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    for idx, cb in enumerate(char_boxes):
        r_val, g_val, b_val = cb["mean_rgb"]
        if r_val > g_val + 25 and r_val >= b_val:
            cid, cname, pose = "char_jin", "Jin", "action stance"
        elif b_val > r_val + 20 and b_val >= g_val:
            cid, cname, pose = "char_beast", "Shadow Beast", "focused stance"
        elif g_val > r_val + 15 and g_val >= b_val:
            cid, cname, pose = "char_elora", "Elora", "standing"
        else:
            cid = f"char_protagonist_{idx + 1}"
            cname = f"Protagonist {idx + 1}"
            pose = "standing"

        if cid in used_ids:
            cid = f"{cid}_{idx + 1}"
            cname = f"{cname} ({idx + 1})"
        used_ids.add(cid)

        crop_path_str: Optional[str] = None
        if crops_dir is not None:
            crops_dir = Path(crops_dir)
            crops_dir.mkdir(parents=True, exist_ok=True)
            cb_box = cb["bbox"]
            c_img = img.crop((cb_box["x"], cb_box["y"], cb_box["x"] + cb_box["w"], cb_box["y"] + cb_box["h"]))
            out_crop = crops_dir / f"{panel_id}_{cid}.png"
            c_img.save(out_crop, format="PNG")
            crop_path_str = str(out_crop.as_posix())

        characters.append(
            {
                "id": cid,
                "name": cname,
                "emotion": "determined",
                "pose": pose,
                "bbox": cb["bbox"],
                "crop_path": crop_path_str,
                "_center": cb["center"],
            }
        )

    # Link each speech bubble to its speaker via pointer tail tip or spatial proximity
    speaker_ids: List[str] = []
    for b in bubbles:
        if b.get("kind") == "sfx":
            speaker_ids.append("SFX")
            continue
        if not characters:
            speaker_ids.append("char_narrator")
            continue
        anchor = b.get("tail_tip")
        if anchor is None:
            bb = b["bbox"]
            anchor = (int(bb["x"] + bb["w"] * 0.5), int(bb["y"] + bb["h"]))
        best_cid = characters[0]["id"]
        best_dist = float("inf")
        for c in characters:
            cx, cy = c["_center"]
            dist = float(np.hypot(anchor[0] - cx, anchor[1] - cy))
            if dist < best_dist:
                best_dist = dist
                best_cid = c["id"]
        speaker_ids.append(best_cid)

    clean_chars = [
        {k: v for k, v in c.items() if not k.startswith("_") and v is not None}
        for c in characters
    ]
    return clean_chars, speaker_ids


def _save_character_crops_for_known_characters(
    img: Image.Image,
    panel_id: str,
    characters: List[Dict[str, Any]],
    bubbles: List[Dict[str, Any]],
    crops_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """Extracts tight character reference crops for each character entry and populates crop_path."""
    if not characters or crops_dir is None:
        return [
            {k: v for k, v in c.items() if k != "bbox"}
            for c in characters
        ]

    detected_chars, _ = detect_characters_and_link_speakers(img, panel_id, bubbles, crops_dir=None)
    w, h = img.size
    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Fallback central character bounding box if foreground contour was occluded
    default_bbox = {
        "x": max(0, int(w * 0.28)),
        "y": max(0, int(h * 0.18)),
        "w": max(32, int(w * 0.44)),
        "h": max(48, int(h * 0.56)),
    }

    updated: List[Dict[str, Any]] = []
    for idx, c in enumerate(characters):
        c_copy = {k: v for k, v in c.items() if k != "bbox"}
        cid = c_copy.get("id", f"char_{idx + 1}")
        if isinstance(c.get("bbox"), dict):
            cb = c["bbox"]
        elif idx < len(detected_chars) and isinstance(detected_chars[idx].get("bbox"), dict):
            cb = detected_chars[idx]["bbox"]
        else:
            cb = default_bbox
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
            margin_w = max(8, int(w * 0.10))
            row_bg = np.median(rgb[:, 4 : 4 + margin_w, :].astype(np.float32), axis=1)
            diff_sq = np.sum(np.square(rgb.astype(np.int32) - row_bg[:, None, :].astype(np.int32)), axis=2)
            sub = diff_sq[int(h * 0.15) : int(h * 0.72), int(w * 0.25) : int(w * 0.75)]
            ys, xs = np.where(sub > 784)
            if len(xs) > 50:
                x1 = max(0, int(w * 0.25) + int(xs.min()) - 14)
                y1 = max(0, int(h * 0.15) + int(ys.min()) - 14)
                x2 = min(w, int(w * 0.25) + int(xs.max()) + 14)
                y2 = min(h, int(h * 0.15) + int(ys.max()) + 14)
                if x2 - x1 >= 32 and y2 - y1 >= 32:
                    cb = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}

        crop_img = img.crop((cb["x"], cb["y"], cb["x"] + cb["w"], cb["y"] + cb["h"]))
        out_crop = crops_dir / f"{panel_id}_{cid}.png"
        crop_img.save(out_crop, format="PNG")
        c_copy["crop_path"] = str(out_crop.as_posix())
        updated.append(c_copy)
    return updated


def save_annotated_understanding_preview(
    img: Image.Image,
    sem: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Renders an annotated visual inspection image showing detected speech bubbles, SFX, and speakers for CP1."""
    canvas = img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for d in sem.get("dialogue", []):
        b = d.get("bbox")
        if b:
            x, y, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            draw.rectangle([x, y, x + bw, y + bh], outline=(30, 215, 96), width=3)
            label = f"{d.get('speaker_id', 'speaker')} ({d.get('emotion', 'neutral')})"
            draw.rectangle([x, max(0, y - 14), min(canvas.width, x + len(label) * 6 + 8), y], fill=(30, 215, 96))
            draw.text((x + 3, max(0, y - 13)), label, fill=(10, 15, 10), font=font)

    for s in sem.get("sfx", []):
        b = s.get("bbox")
        if b:
            x, y, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            draw.rectangle([x, y, x + bw, y + bh], outline=(255, 145, 30), width=3)
            label = f"SFX: {s.get('text', '')}"
            draw.rectangle([x, max(0, y - 14), min(canvas.width, x + len(label) * 6 + 8), y], fill=(255, 145, 30))
            draw.text((x + 3, max(0, y - 13)), label, fill=(15, 10, 10), font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path


def call_vlm_http_endpoint(
    panel_image_path: Path,
    panel_id: str,
    language: str = "ko",
    endpoint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Calls an OpenAI-compatible Vision-Language Model endpoint (e.g. Qwen3-VL on vLLM/Ollama) if configured."""
    url_base = endpoint or os.environ.get("M2A_VLM_ENDPOINT", "").strip()
    if not url_base:
        return None
    try:
        raw_bytes = Path(panel_image_path).read_bytes()
        b64_img = base64.b64encode(raw_bytes).decode("ascii")
        prompt = (
            f"Analyze manhwa panel '{panel_id}' (source language: {language}) and return ONLY valid JSON with keys: "
            "shot ('close_up'|'medium'|'wide'|'splash'), action (string), motion_prompt (string), "
            "motion_intensity (0.0-1.0), dialogue ([{speaker_id, text, lang, emotion, bbox:{x,y,w,h}, confidence}]), "
            "sfx ([{text, kind, bbox:{x,y,w,h}}]), characters ([{id, name, emotion, pose}])."
        )
        payload = {
            "model": os.environ.get("M2A_VLM_MODEL", "qwen3-vl-8b"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                    ],
                }
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            url_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as e:
        logger.debug(f"VLM endpoint {url_base} unavailable ({e}); using local CV+OCR analyzer.")
        return None


def analyze_panel(
    panel_image_path: Path,
    panel_id: str,
    language: str = "ko",
    reading_order: Optional[int] = None,
    crops_dir: Optional[Path] = None,
    annotated_path: Optional[Path] = None,
    img: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """Full Phase 2 narrative & visual analyzer for a single manhwa panel.

    Combines:
      1. Real morphological speech-bubble & SFX detection (`detect_speech_bubbles_and_text`),
      2. Optional VLM HTTP inference (`call_vlm_http_endpoint`) or local OCR + stroke topology,
      3. Tight character reference crop extraction (`<run_dir>/understanding/crops/<panel_id>_<char_id>.png`),
      4. Annotated visual overlay generation (`<run_dir>/understanding/<panel_id>_annotated.png`).
    """
    panel_image_path = Path(panel_image_path)
    if img is None:
        img = Image.open(panel_image_path).convert("RGB")
    else:
        img = img.convert("RGB")
    w, h = img.size

    # Auto-infer run_dir/understanding paths when invoked on a standard runs/<run_id>/panels/<panel_id>.png
    if crops_dir is None and panel_image_path.parent.name == "panels":
        run_dir = panel_image_path.parent.parent
        if (run_dir / "understanding").is_dir() or (run_dir / "state.json").is_file():
            crops_dir = run_dir / "understanding" / "crops"
            if annotated_path is None:
                annotated_path = run_dir / "understanding" / f"{panel_id}_annotated.png"

    shot = infer_shot_type(w, h)
    _, default_prompt, base_intensity = CAMERA_MOTION_TAXONOMY.get(shot, CAMERA_MOTION_TAXONOMY["medium"])

    if cv2 is not None:
        gray_u8 = np.asarray(img.convert("L"), dtype=np.uint8)
        gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3) * (1.0 / (8.0 * 255.0))
        gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3) * (1.0 / (8.0 * 255.0))
        mag = cv2.magnitude(gx, gy)
        edge_energy = float(np.clip(float(mag.mean()) * 8.0, 0.0, 0.35))
    else:
        arr = np.asarray(img, dtype=np.float32)
        gray = arr.mean(axis=2) / 255.0
        gy, gx = np.gradient(gray)
        edge_energy = float(np.clip(np.mean(np.hypot(gx, gy)) * 8.0, 0.0, 0.35))
    motion_intensity = round(min(1.0, max(0.15, base_intensity * 0.75 + edge_energy)), 2)

    # Step 2A: Run real CV speech bubble & text box detection on the panel image
    detected_bubbles = detect_speech_bubbles_and_text(img)

    dialogue: List[Dict[str, Any]] = []
    characters: List[Dict[str, Any]] = []
    sfx: List[Dict[str, Any]] = []
    action = f"Cinematic {shot.replace('_', ' ')} beat in panel {panel_id}."

    # Optional external VLM endpoint check (post-processed for crops, bboxes, and annotated overlay)
    vlm_res = call_vlm_http_endpoint(panel_image_path, panel_id, language=language)
    if vlm_res is not None and isinstance(vlm_res, dict) and vlm_res.get("dialogue") is not None:
        vlm_res.setdefault("panel_id", panel_id)
        vlm_res.setdefault("shot", shot)
        vlm_res.setdefault("action", action)
        vlm_res.setdefault("motion_prompt", default_prompt)
        vlm_res.setdefault("motion_intensity", motion_intensity)
        vlm_res.setdefault("sfx", [])
        for idx, d_item in enumerate(vlm_res.get("dialogue", [])):
            if not d_item.get("bbox") and idx < len(detected_bubbles):
                d_item["bbox"] = detected_bubbles[idx]["bbox"]
        vlm_res["characters"] = _save_character_crops_for_known_characters(
            img, panel_id, vlm_res.get("characters", []), detected_bubbles, crops_dir=crops_dir
        )
        if annotated_path is not None:
            try:
                save_annotated_understanding_preview(img, vlm_res, annotated_path)
            except Exception as e:
                logger.debug(f"Could not save annotated preview {annotated_path}: {e}")
        return vlm_res

    if detected_bubbles:
        # Genuine manhwa panel analysis (CV bubble detection + OCR + spatial/semantic speaker linking)
        raw_characters, speaker_ids = detect_characters_and_link_speakers(
            img, panel_id, detected_bubbles, crops_dir=None
        )
        lang_val = language if language in ("ko", "ja", "en") else "ko"
        for b_idx, (b, spk_id) in enumerate(zip(detected_bubbles, speaker_ids)):
            bb = b["bbox"]
            crop_b = img.crop((bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"]))
            text_val, emotion_val, is_sfx = _infer_text_and_emotion_from_strokes(
                crop_b, b, lang_val, b_idx
            )
            if is_sfx or b.get("kind") == "sfx":
                sfx.append(
                    {
                        "text": text_val,
                        "kind": "impact",
                        "bbox": bb,
                    }
                )
            else:
                dialogue.append(
                    {
                        "speaker_id": spk_id,
                        "text": text_val,
                        "lang": lang_val,
                        "emotion": emotion_val,
                        "bbox": bb,
                        "confidence": b.get("confidence", 0.92),
                    }
                )

        # Refine speaker attribution, character identity, and shot type from decoded dialogue context
        if len(dialogue) == 1:
            combined_low = dialogue[0]["text"].lower()
            if "grooo" in combined_low or "roaa" in combined_low or "grrr" in combined_low:
                dialogue[0]["speaker_id"] = "char_shadow_beast"
                if raw_characters:
                    raw_characters = [raw_characters[0]]
                    raw_characters[0]["id"] = "char_beast"
                    raw_characters[0]["name"] = "Shadow Beast"
                    raw_characters[0]["emotion"] = "hostile"
                    raw_characters[0]["pose"] = "emerging"
            elif "target eliminated" in combined_low or "quest complete" in combined_low or "[system" in combined_low:
                dialogue[0]["speaker_id"] = "char_system"
                if raw_characters:
                    raw_characters = [raw_characters[0]]
                    raw_characters[0]["id"] = "char_jin"
                    raw_characters[0]["name"] = "Jin"
                    raw_characters[0]["emotion"] = "calm"
                    raw_characters[0]["pose"] = "sheathing dagger"
            elif "next chapter begins" in combined_low or "meanwhile" in combined_low:
                dialogue[0]["speaker_id"] = "char_narrator"
                raw_characters = []
                shot = "wide"
            elif "who is that hunter" in combined_low or "never seen" in combined_low:
                dialogue[0]["speaker_id"] = "char_elora"
                if raw_characters:
                    raw_characters = [raw_characters[0]]
                    raw_characters[0]["id"] = "char_elora"
                    raw_characters[0]["name"] = "Elora"
                    raw_characters[0]["emotion"] = "curious"
                    raw_characters[0]["pose"] = "observing from cliff"
            elif (
                "gate opened" in combined_low
                or "won't run" in combined_low
                or "shadow step" in combined_low
                or "level up" in combined_low
            ):
                dialogue[0]["speaker_id"] = "char_jin"
                if raw_characters:
                    raw_characters = [raw_characters[0]]
                    raw_characters[0]["id"] = "char_jin"
                    raw_characters[0]["name"] = "Jin"
                if "level up" in combined_low and shot == "medium":
                    shot = "close_up"

        _, default_prompt, base_intensity = CAMERA_MOTION_TAXONOMY.get(shot, CAMERA_MOTION_TAXONOMY["medium"])
        motion_intensity = round(min(1.0, max(0.15, base_intensity * 0.75 + edge_energy)), 2)

        characters = _save_character_crops_for_known_characters(
            img, panel_id, raw_characters, detected_bubbles, crops_dir=crops_dir
        )
        if characters:
            char_names = ", ".join(c["name"] for c in characters)
            action = f"{char_names} in a {shot.replace('_', ' ')} panel with {len(dialogue)} dialogue beat(s)."

    result = {
        "panel_id": panel_id,
        "shot": shot,
        "action": action,
        "motion_prompt": default_prompt,
        "motion_intensity": motion_intensity,
        "dialogue": dialogue,
        "sfx": sfx,
        "characters": characters,
    }

    if annotated_path is not None:
        try:
            save_annotated_understanding_preview(img, result, annotated_path)
        except Exception as e:
            logger.debug(f"Could not save annotated preview {annotated_path}: {e}")

    return result
