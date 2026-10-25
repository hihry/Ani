"""Automatic Video Quality Control (QC) Metrics v2 (`workers/qc_metrics.py`).

Implements Phase 3 calibrated, optical-flow and structural QC metrics:
  - `motion_score`: Farneback dense optical flow magnitude normalized to [0, 1].
  - `warp_error`: Flow-warped temporal consistency error (catches shuffled frames & morphing).
  - `flicker_score`: Frame-to-frame luminance/delta oscillation (catches brightness/flicker spikes).
  - `identity_similarity`: Multi-region palette + structural crop similarity (catches hue shifts & face swaps).
  - `first_frame_fidelity`: SSIM + color fidelity between frame 0 and conformed source plate.
  - `style_drift`: Canny line-art edge density + LAB chrominance drift relative to source plate.
  - `new_text_count`: Multi-frame residual speech-bubble & text-overlay detector relative to clean plate.
  - `metrics_version`: 2
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from workers.understanding import count_residual_text_bubbles
except ImportError:
    from understanding import count_residual_text_bubbles

logger = logging.getLogger("m2a.qc_metrics")

METRICS_VERSION = 2


def _hist_vec(im: Image.Image, bins_per_channel: int = 16) -> np.ndarray:
    arr_rgb = np.asarray(im.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    bins = [np.histogram(arr_rgb[:, :, c], bins=bins_per_channel, range=(0.0, 256.0))[0] for c in range(3)]
    h = np.concatenate(bins).astype(np.float32)
    norm = float(np.linalg.norm(h))
    return h / max(1e-6, norm)


def _spatial_subject_similarity(ref_im: Image.Image, frame_im: Image.Image) -> float:
    """Computes central subject similarity invariant to small camera pans/zooms (up to ~15%)
    while strongly penalizing central region replacement (face swap) or channel permutation.
    """
    ref_arr = np.asarray(ref_im.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    frm_arr = np.asarray(frame_im.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    # Central subject crop (inner 50% width x 50% height: [32:96, 32:96])
    tpl = frm_arr[32:96, 32:96, :]
    if cv2 is not None:
        best_ncc = -1.0
        for s_tpl in (1.0, 0.94, 0.89):
            ts = max(32, int(round(64 * s_tpl)))
            tpl_s = cv2.resize(tpl, (ts, ts), interpolation=cv2.INTER_LINEAR) if ts != 64 else tpl
            res = cv2.matchTemplate(ref_arr, tpl_s, cv2.TM_CCOEFF_NORMED)
            best_ncc = max(best_ncc, float(np.max(res)))
        ncc_sim = max(0.0, min(1.0, 0.5 * (best_ncc + 1.0)))
    else:
        diff = float(np.mean(np.abs(ref_arr[32:96, 32:96, :] - tpl)))
        ncc_sim = max(0.0, 1.0 - 2.0 * diff)

    ref_center_hist = _hist_vec(Image.fromarray((ref_arr[24:104, 24:104, :] * 255.0).astype(np.uint8)))
    frm_center_hist = _hist_vec(Image.fromarray((tpl * 255.0).astype(np.uint8)))
    center_hist_sim = float(np.dot(ref_center_hist, frm_center_hist))
    return 0.55 * ncc_sim + 0.45 * center_hist_sim


def _compute_temporal_anatomical_integrity(
    frames: List[Image.Image],
    src_overscan: Optional[Image.Image] = None,
    src_img: Optional[Image.Image] = None,
) -> float:
    """Checks temporal anatomical/structural coherence across sampled frames relative to frame 0
    and the overscan source plate (supporting both zoom_in and zoom_out camera trajectories).
    Computes exact geometric active-panel bounds (excluding blurred letterbox bars on 720x1280
    conformed clips), estimates the global 2D camera pan/zoom of the active artwork region,
    and verifies an overlapping 5x4 sliding patch grid within +-15px (~14%) local deformation tolerance.
    Catches localized face melting (panel_001) and hallucinated extra limbs/objects (panel_002)
    on both raw and conformed letterboxed clips.
    """
    if cv2 is None or len(frames) < 4:
        return 1.0
    w_s, h_s = 256, 448
    f0 = np.asarray(frames[0].convert("RGB").resize((w_s, h_s), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    search_planes = [f0]
    if src_overscan is not None:
        w_ov = int(round(w_s * 1.20))
        ov_ar = float(src_overscan.height) / float(max(1, src_overscan.width))
        frm_ar_hw = float(h_s) / float(w_s)
        h_ov = max(int(round(h_s * 1.20)), int(round(w_ov * (ov_ar / max(0.5, frm_ar_hw)) * frm_ar_hw)))
        h_ov = min(h_s * 4, h_ov)
        f_ov = np.asarray(src_overscan.convert("RGB").resize((w_ov, h_ov), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        search_planes.append(f_ov)

    ay1, ay2 = 16, h_s - 16
    g0 = f0.mean(axis=2)
    gx_f0 = np.abs(cv2.Sobel(g0, cv2.CV_32F, 1, 0, ksize=3))
    if src_img is not None:
        try:
            from workers.render_kenburns import _trim_outer_white_gutters
        except ImportError:
            from render_kenburns import _trim_outer_white_gutters
        trimmed = _trim_outer_white_gutters(src_img)
        ar_src = float(trimmed.size[0]) / float(max(1, trimmed.size[1]))
        fw, fh = frames[0].size
        ar_frm = float(fw) / float(max(1, fh))
        if ar_src > ar_frm * 1.08:
            act_h = int(round(h_s * (ar_frm / ar_src)))
            cand_ay1 = max(12, (h_s - act_h) // 2 + 8)
            cand_ay2 = min(h_s - 12, (h_s + act_h) // 2 - 8)
            if cand_ay2 - cand_ay1 >= 48 and cand_ay1 > 20 and cand_ay2 < h_s - 20:
                inner_gx = float(np.mean(gx_f0[cand_ay1:cand_ay2, :]))
                outer_gx = 0.5 * (
                    float(np.mean(gx_f0[: cand_ay1 - 8, :]))
                    + float(np.mean(gx_f0[cand_ay2 + 8 :, :]))
                )
                # Only narrow to letterbox viewport if outer bars are genuinely blurred letterbox bars
                if outer_gx < inner_gx * 0.62:
                    ay1, ay2 = cand_ay1, cand_ay2
    else:
        act = np.where(gx_f0.mean(axis=1) > 0.018)[0]
        if len(act) > 50 and (int(act[-1]) - int(act[0])) >= 80:
            ay1 = int(act[0]) + 14
            ay2 = int(act[-1]) - 14

    if ay2 - ay1 < 48:
        return 1.0

    # Restrict f0 search plane to active viewport + 24px margin so inner crops cannot jump into letterbox bars
    f0_vy1 = max(0, ay1 - 24)
    f0_vy2 = min(h_s, ay2 + 24)
    search_planes[0] = f0[f0_vy1:f0_vy2, :]

    indices = sorted(
        {
            max(1, len(frames) // 16),
            max(1, len(frames) // 8),
            max(1, (3 * len(frames)) // 16),
            max(1, len(frames) // 4),
            max(1, len(frames) // 2),
            max(1, (3 * len(frames)) // 4),
            len(frames) - 1,
        }
    )
    worst_sim = 1.0
    search_planes_half = [
        (ref_b, cv2.resize(ref_b, (max(16, ref_b.shape[1] // 2), max(16, ref_b.shape[0] // 2)), interpolation=cv2.INTER_AREA))
        for ref_b in search_planes
    ]
    for idx in indices:
        fk = np.asarray(frames[idx].convert("RGB").resize((w_s, h_s), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        fk_act = fk[ay1:ay2, 16 : w_s - 16]
        ah_k, aw_k, _ = fk_act.shape
        fk_act_half = cv2.resize(fk_act, (max(16, aw_k // 2), max(16, ah_k // 2)), interpolation=cv2.INTER_AREA)
        ah_h, aw_h, _ = fk_act_half.shape
        best_g_ncc = -1.0
        best_aligned_ref = f0[ay1:ay2, 16 : w_s - 16]
        for ref_base, ref_half in search_planes_half:
            rh, rw, _ = ref_base.shape
            rh_h, rw_h, _ = ref_half.shape
            for s in (1.18, 1.12, 1.06, 1.0, 0.97, 0.94, 0.91, 0.88, 0.85):
                tw_h = max(16, min(rw_h, int(round(aw_h * s))))
                th_h = max(16, min(rh_h, int(round(ah_h * s))))
                tpl_g = cv2.resize(fk_act_half, (tw_h, th_h), interpolation=cv2.INTER_LINEAR)
                res_g = cv2.matchTemplate(ref_half, tpl_g, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res_g)
                if float(mv) > best_g_ncc:
                    best_g_ncc = float(mv)
                    tw = max(32, min(rw, int(round(aw_k * s))))
                    th = max(32, min(rh, int(round(ah_k * s))))
                    mx = max(0, min(rw - tw, int(round(ml[0] * (rw / float(rw_h))))))
                    my = max(0, min(rh - th, int(round(ml[1] * (rh / float(rh_h))))))
                    best_aligned_ref = cv2.resize(
                        ref_base[my : my + th, mx : mx + tw], (aw_k, ah_k), interpolation=cv2.INTER_LINEAR
                    )

        pad = 10
        padded_ref = cv2.copyMakeBorder(best_aligned_ref, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        pw, ph = max(16, aw_k // 5), max(16, ah_k // 4)
        sy_step = max(8, (ah_k - ph) // 5)
        sx_step = max(8, (aw_k - pw) // 6)
        for y1 in range(0, ah_k - ph + 1, sy_step):
            for x1 in range(0, aw_k - pw + 1, sx_step):
                p_fk = fk_act[y1 : y1 + ph, x1 : x1 + pw]
                if float(np.std(p_fk)) < 0.052:
                    continue
                win_ref = padded_ref[y1 : y1 + ph + 2 * pad, x1 : x1 + pw + 2 * pad]
                res_p = cv2.matchTemplate(win_ref, p_fk, cv2.TM_CCOEFF_NORMED)
                _, p_ncc, _, p_loc = cv2.minMaxLoc(res_p)
                px, py = int(p_loc[0]), int(p_loc[1])
                diff_p = np.abs(win_ref[py : py + ph, px : px + pw] - p_fk)
                p_mae = float(np.mean(diff_p))
                p_p90 = float(np.percentile(diff_p, 90))
                if float(p_ncc) < 0.45 and (p_mae >= 0.058 or p_p90 >= 0.15):
                    p_sim = max(0.0, min(0.55, 0.5 * (float(p_ncc) + 1.0) - 2.5 * max(0.0, p_mae - 0.04)))
                    worst_sim = min(worst_sim, p_sim)

    return worst_sim


def _compute_ssim_rgb(im_a: Image.Image, im_b: Image.Image) -> float:
    """Computes fast global SSIM + color similarity on 160x284 downsampled RGB images."""
    a = np.asarray(im_a.resize((160, 284), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    b = np.asarray(im_b.resize((160, 284), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    var_a = float(np.var(a))
    var_b = float(np.var(b))
    cov = float(np.mean((a - mu_a) * (b - mu_b)))
    ssim_val = ((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    )
    hist_sim = float(np.dot(_hist_vec(im_a), _hist_vec(im_b)))
    return float(np.clip(0.5 * max(0.0, ssim_val) + 0.5 * hist_sim, 0.0, 1.0))


def _compute_aligned_first_frame_fidelity(
    src_img: Image.Image,
    src_ref: Image.Image,
    src_overscan: Image.Image,
    first_frame: Image.Image,
) -> float:
    """Computes first-frame fidelity allowing legitimate camera start pan/zoom/scroll offsets
    within the source panel canvas (e.g. top-of-strip scroll start or Ken Burns/Parallax overscan).
    """
    best = max(
        _compute_ssim_rgb(src_ref, first_frame),
        _compute_ssim_rgb(src_overscan, first_frame),
        _compute_ssim_rgb(src_img, first_frame),
    )
    if cv2 is None:
        return round(best, 4)

    try:
        from workers.render_kenburns import prepare_cinematic_canvas
    except ImportError:
        from render_kenburns import prepare_cinematic_canvas

    for overscan in (1.07, 1.16, 1.20):
        canvas = prepare_cinematic_canvas(
            src_img, target_res=first_frame.size, overscan=overscan, preserve_tall=True
        )
        cw, ch = canvas.size
        fw, fh = first_frame.size
        if cw < fw or ch < fh:
            continue
        # Downsample both proportionally so template matching is < 2ms
        scale = 160.0 / float(max(1, fw))
        tw = 160
        th = max(32, int(round(fh * scale)))
        sw = max(tw, int(round(cw * scale)))
        sh = max(th, int(round(ch * scale)))

        search_arr = np.asarray(canvas.resize((sw, sh), Image.Resampling.BILINEAR), dtype=np.uint8)
        tpl_arr = np.asarray(first_frame.resize((tw, th), Image.Resampling.BILINEAR), dtype=np.uint8)
        if sh > th or sw > tw:
            res = cv2.matchTemplate(search_arr, tpl_arr, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            mx, my = int(max_loc[0]), int(max_loc[1])
            matched_patch = Image.fromarray(search_arr[my : my + th, mx : mx + tw, :])
            best = max(best, _compute_ssim_rgb(matched_patch, first_frame))
        else:
            best = max(best, _compute_ssim_rgb(canvas, first_frame))

    return round(float(np.clip(best, 0.0, 1.0)), 4)


def _compute_style_drift(ref_im: Image.Image, sampled_frames: List[Image.Image]) -> float:
    """Measures line-art edge density drift + LAB chrominance drift relative to source plate."""
    if not sampled_frames:
        return 1.0

    ref_small = np.asarray(ref_im.resize((160, 284), Image.Resampling.BILINEAR), dtype=np.uint8)
    if cv2 is not None:
        ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_RGB2GRAY)
        ref_edges = float(np.mean(cv2.Canny(ref_gray, 80, 180) > 0))
        ref_lab = cv2.cvtColor(ref_small, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_ab_mean = np.mean(ref_lab[:, :, 1:], axis=(0, 1))
    else:
        ref_edges = 0.05
        ref_ab_mean = np.mean(ref_small.astype(np.float32), axis=(0, 1))

    drifts: List[float] = []
    for frm in sampled_frames:
        f_small = np.asarray(frm.resize((160, 284), Image.Resampling.BILINEAR), dtype=np.uint8)
        if cv2 is not None:
            f_gray = cv2.cvtColor(f_small, cv2.COLOR_RGB2GRAY)
            f_edges = float(np.mean(cv2.Canny(f_gray, 80, 180) > 0))
            edge_drift = abs(f_edges - ref_edges) / max(ref_edges, 0.025)

            f_lab = cv2.cvtColor(f_small, cv2.COLOR_RGB2LAB).astype(np.float32)
            f_ab_mean = np.mean(f_lab[:, :, 1:], axis=(0, 1))
            chroma_drift = float(np.linalg.norm(f_ab_mean - ref_ab_mean)) / 45.0
            drifts.append(0.45 * min(1.0, edge_drift) + 0.55 * min(1.0, chroma_drift))
        else:
            f_mean = np.mean(f_small.astype(np.float32), axis=(0, 1))
            drifts.append(min(1.0, float(np.linalg.norm(f_mean - ref_ab_mean)) / 60.0))

    return round(float(np.clip(np.mean(drifts), 0.0, 1.0)), 4)


def _detect_overlay_text_blocks(ref_im: Image.Image, frame_im: Image.Image) -> int:
    """Detects newly introduced text overlays or banner boxes on a frame compared to the clean plate."""
    if cv2 is None:
        return 0
    w, h = 240, 426
    ref_g = np.asarray(ref_im.resize((w, h), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32)
    frm_g = np.asarray(frame_im.resize((w, h), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32)

    # Compute high-frequency stroke energy difference (Sobel gradient magnitude delta)
    ref_gx = cv2.Sobel(ref_g, cv2.CV_32F, 1, 0, ksize=3)
    ref_gy = cv2.Sobel(ref_g, cv2.CV_32F, 0, 1, ksize=3)
    frm_gx = cv2.Sobel(frm_g, cv2.CV_32F, 1, 0, ksize=3)
    frm_gy = cv2.Sobel(frm_g, cv2.CV_32F, 0, 1, ksize=3)

    ref_mag = cv2.magnitude(ref_gx, ref_gy)
    frm_mag = cv2.magnitude(frm_gx, frm_gy)

    # Check horizontal bands in the middle 80% of the frame for sharp new glyph stroke clusters
    # bundled inside a bright or dark overlay patch
    abs_diff = np.abs(frm_g - ref_g)
    stroke_excess = np.maximum(0.0, frm_mag - ref_mag - 35.0)
    # Require both strong intensity change AND dense sharp glyph strokes in a compact horizontal strip
    mask = ((abs_diff > 45.0) & (stroke_excess > 40.0)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    count = 0
    for lbl in range(1, num_labels):
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        # Typical text overlay / watermark / bubble lettering cluster
        if bw >= 28 and 8 <= bh <= 120 and area >= 140 and (bw / float(max(1, bh))) >= 1.3:
            count += 1
    return count


def compute_clip_qc_metrics(
    source_image_path: Path,
    frames: List[Image.Image],
    route: str = "parallax",
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Computes automatic QC Metrics v2 on rendered frames against the source clean plate."""
    if not frames:
        return {
            "identity_similarity": 0.0,
            "new_text_count": 0,
            "flicker_score": 1.0,
            "motion_score": 0.0,
            "warp_error": 1.0,
            "first_frame_fidelity": 0.0,
            "style_drift": 1.0,
            "metrics_version": METRICS_VERSION,
            "overall_pass": False,
            "reasons": ["empty_frames"],
        }

    src_img = Image.open(source_image_path).convert("RGB")
    first_frame = frames[0].convert("RGB")
    q1_frame = frames[len(frames) // 5].convert("RGB")
    mid_frame = frames[len(frames) // 2].convert("RGB")
    q3_frame = frames[(4 * len(frames)) // 5].convert("RGB")
    last_frame = frames[-1].convert("RGB")

    src_ref = src_img
    src_overscan = src_img
    src_overscan_120 = src_img
    src_tall = src_img
    if src_img.size != first_frame.size:
        try:
            from workers.render_kenburns import prepare_cinematic_canvas
        except ImportError:
            from render_kenburns import prepare_cinematic_canvas
        src_ref = prepare_cinematic_canvas(src_img, target_res=first_frame.size, overscan=1.0)
        src_overscan = prepare_cinematic_canvas(src_img, target_res=first_frame.size, overscan=1.16)
        src_overscan_120 = prepare_cinematic_canvas(src_img, target_res=first_frame.size, overscan=1.20)
        src_tall = prepare_cinematic_canvas(
            src_img, target_res=first_frame.size, overscan=1.20, preserve_tall=True
        )

    # 1. Identity Similarity (global palette + central subject structural match + temporal self-consistency)
    v_src = _hist_vec(src_img)
    v_ref = _hist_vec(src_ref)
    v_over = _hist_vec(src_overscan_120)
    v_first = _hist_vec(first_frame)
    v_mid = _hist_vec(mid_frame)
    v_last = _hist_vec(last_frame)

    sim_palette_src = max(
        float(np.mean([np.dot(v_src, v_first), np.dot(v_src, v_mid), np.dot(v_src, v_last)])),
        float(np.mean([np.dot(v_ref, v_first), np.dot(v_ref, v_mid), np.dot(v_ref, v_last)])),
        float(np.mean([np.dot(v_over, v_first), np.dot(v_over, v_mid), np.dot(v_over, v_last)])),
    )
    sim_temporal = float(0.5 * np.dot(v_first, v_last) + 0.5 * np.dot(v_first, v_mid))

    def _frame_subj_sim(frm: Image.Image) -> float:
        return max(
            _spatial_subject_similarity(src_ref, frm),
            _spatial_subject_similarity(src_overscan_120, frm),
            _spatial_subject_similarity(first_frame, frm),
        )

    subj_sim_q1 = _frame_subj_sim(q1_frame)
    subj_sim_mid = _frame_subj_sim(mid_frame)
    subj_sim_q3 = _frame_subj_sim(q3_frame)
    subj_sim_last = _frame_subj_sim(last_frame)
    subj_sim_src = max(
        _spatial_subject_similarity(src_ref, first_frame),
        _spatial_subject_similarity(src_overscan, first_frame),
        _spatial_subject_similarity(src_overscan_120, first_frame),
        _spatial_subject_similarity(src_img, first_frame),
    )
    anatomical_sim = _compute_temporal_anatomical_integrity(
        frames, src_overscan=src_tall, src_img=src_img
    )
    subj_sim = min(subj_sim_src, subj_sim_q1, subj_sim_mid, subj_sim_q3, subj_sim_last, anatomical_sim)

    raw_identity = 0.40 * sim_palette_src + 0.25 * sim_temporal + 0.35 * subj_sim
    # If either global palette or subject/anatomical integrity was severely corrupted, cap identity_similarity
    if sim_palette_src < 0.55 or subj_sim < 0.62:
        raw_identity = min(raw_identity, min(sim_palette_src, subj_sim))
    identity_similarity = round(float(np.clip(raw_identity, 0.0, 1.0)), 4)

    # 2. First-frame fidelity (comparing against overscan & tall-strip camera start windows)
    first_frame_fidelity = _compute_aligned_first_frame_fidelity(
        src_img=src_img,
        src_ref=src_ref,
        src_overscan=src_overscan,
        first_frame=first_frame,
    )

    # 3. Optical flow motion score, flow-warped temporal error, and flicker score
    if len(frames) > 1:
        step = max(1, len(frames) // 16)
        sampled_pil = frames[::step]
        if len(sampled_pil) < 2:
            sampled_pil = [frames[0], frames[-1]]

        sampled_gray = [
            np.asarray(f.resize((160, 284), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255.0
            for f in sampled_pil
        ]
        # When a wide panel is letterboxed into a 9:16 frame, measure motion on the active panel viewport
        try:
            from workers.render_kenburns import _trim_outer_white_gutters
        except ImportError:
            from render_kenburns import _trim_outer_white_gutters
        tr_m = _trim_outer_white_gutters(src_img)
        ar_src_m = float(tr_m.width) / float(max(1, tr_m.height))
        ar_frm_m = float(first_frame.width) / float(max(1, first_frame.height))
        if ar_src_m > ar_frm_m * 1.18:
            act_h_m = max(48, min(284, int(round(284 * (ar_frm_m / ar_src_m)))))
            ym0 = max(0, (284 - act_h_m) // 2)
            ym1 = min(284, ym0 + act_h_m)
        else:
            ym0, ym1 = 0, 284

        deltas = [
            float(np.mean(np.abs(sampled_gray[i + 1][ym0:ym1, :] - sampled_gray[i][ym0:ym1, :])))
            for i in range(len(sampled_gray) - 1)
        ]
        flicker_std = float(np.std(deltas))

        # Check sign-alternating brightness oscillation (strobe/flicker)
        lum_means = [float(np.mean(g[ym0:ym1, :])) for g in sampled_gray]
        lum_diffs = np.diff(lum_means)
        sign_flips = 0
        for i in range(len(lum_diffs) - 1):
            if abs(lum_diffs[i]) > 0.025 and abs(lum_diffs[i + 1]) > 0.025:
                if (lum_diffs[i] * lum_diffs[i + 1]) < 0:
                    sign_flips += 1
        strobe_penalty = (sign_flips / float(max(1, len(lum_diffs) - 1))) * 0.25

        flow_mags: List[float] = []
        warp_errs: List[float] = []
        if cv2 is not None:
            h_g, w_g = sampled_gray[0].shape
            grid_x, grid_y = np.meshgrid(np.arange(w_g, dtype=np.float32), np.arange(h_g, dtype=np.float32))
            for i in range(len(sampled_gray) - 1):
                g0 = (sampled_gray[i] * 255.0).astype(np.uint8)
                g1 = (sampled_gray[i + 1] * 255.0).astype(np.uint8)
                flow = cv2.calcOpticalFlowFarneback(
                    g0, g1, None,
                    pyr_scale=0.5, levels=2, winsize=13,
                    iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
                )
                mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
                flow_mags.append(float(np.mean(mag[ym0:ym1, :])))

                # Warp g0 toward g1 using inverse flow mapping: g1(x, y) ~ g0(x - dx, y - dy)
                map_x = grid_x - flow[:, :, 0]
                map_y = grid_y - flow[:, :, 1]
                warped_g0 = cv2.remap(
                    sampled_gray[i],
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                # Measure interior residual error (ignoring 6px border)
                residual = np.abs(sampled_gray[i + 1][6:-6, 6:-6] - warped_g0[6:-6, 6:-6])
                warp_errs.append(float(np.mean(residual)))

            mean_flow_px = float(np.mean(flow_mags))
            # Also check flow direction coherence for shuffled frames detection
            warp_err_mean = float(np.mean(warp_errs))
            warp_err_std = float(np.std(warp_errs))
            # If consecutive steps jump back and forth wildly (shuffled frames), penalize warp_error
            if len(sampled_gray) >= 4:
                step1_diff = np.mean([
                    float(np.mean(np.abs(sampled_gray[i + 1] - sampled_gray[i])))
                    for i in range(len(sampled_gray) - 1)
                ])
                step2_diff = np.mean([
                    float(np.mean(np.abs(sampled_gray[i + 2] - sampled_gray[i])))
                    for i in range(len(sampled_gray) - 2)
                ])
                # In smooth motion, step2_diff > step1_diff * 1.25. In shuffled frames, step1_diff >= step2_diff!
                if step1_diff > 0.008 and step2_diff < step1_diff * 1.12:
                    warp_err_mean = max(warp_err_mean, 0.16)

            warp_error = round(float(np.clip(warp_err_mean + 0.5 * warp_err_std, 0.0, 1.0)), 4)
            mean_delta = float(np.mean(deltas))
            # Combine optical flow magnitude and pixel delta so flat gradients + textured illustrations both work
            motion_raw = max(mean_flow_px * 0.18, mean_delta * 18.0)
            if mean_delta < 1e-4 and mean_flow_px < 0.01:
                motion_raw = 0.0
            elif warp_error < 0.09 and flicker_std < 0.06:
                motion_raw = min(motion_raw, 0.92)
            motion_score = round(float(np.clip(motion_raw, 0.0, 1.0)), 4)
        else:
            mean_delta = float(np.mean(deltas))
            motion_score = round(float(np.clip(mean_delta * 18.0, 0.0, 1.0)), 4)
            warp_error = round(float(np.clip(flicker_std * 1.5, 0.0, 1.0)), 4)

        flicker_score = round(float(np.clip(flicker_std + strobe_penalty, 0.0, 1.0)), 4)
    else:
        sampled_pil = [first_frame]
        flicker_score = 0.0
        motion_score = 0.0
        warp_error = 0.0

    # 4. Style Drift (Canny edge density + LAB chrominance drift)
    style_Sample = [first_frame, mid_frame, last_frame]
    style_drift = _compute_style_drift(src_ref, style_Sample)

    # 5. Text Leakage QC across sampled frames relative to source plate
    src_candidates = [src_img, src_ref, src_overscan, src_overscan_120, src_tall]
    if src_tall.size[1] > first_frame.size[1]:
        tw, th = src_tall.size
        fw, fh = first_frame.size
        x0 = max(0, (tw - fw) // 2)
        src_candidates.append(src_tall.crop((x0, 0, x0 + fw, fh)))
        src_candidates.append(src_tall.crop((x0, th - fh, x0 + fw, th)))

    src_bubbles = max(count_residual_text_bubbles(im) for im in src_candidates)
    if first_frame_fidelity >= 0.88 and _detect_overlay_text_blocks(src_ref, first_frame) == 0:
        src_bubbles = max(src_bubbles, count_residual_text_bubbles(first_frame))
    sample_indices = sorted({0, len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4, len(frames) - 1})
    max_frame_bubbles = 0
    for idx in sample_indices:
        frm_rgb = frames[idx].convert("RGB")
        max_frame_bubbles = max(max_frame_bubbles, count_residual_text_bubbles(frm_rgb))

    new_text_count = max(0, int(max_frame_bubbles - src_bubbles))

    # 6. Route-aware pass/fail evaluation
    cfg = thresholds or {}
    min_id = float(cfg.get("min_identity_similarity", 0.72 if route == "i2v" else 0.85))
    max_flicker = float(cfg.get("max_flicker_score", 0.15))
    min_motion = float(cfg.get("min_motion_score", 0.01))
    max_motion = float(cfg.get("max_motion_score", 0.98))
    max_warp = float(cfg.get("max_warp_error", 0.14))
    min_first = float(cfg.get("min_first_frame_fidelity", 0.70))
    max_style = float(cfg.get("max_style_drift", 0.35))

    reasons: List[str] = []
    if identity_similarity < min_id:
        reasons.append("identity_drift")
    if flicker_score > max_flicker:
        reasons.append("flicker")
    if motion_score < min_motion:
        reasons.append("frozen_motion")
    if motion_score > max_motion:
        reasons.append("chaotic_motion")
    if warp_error > max_warp:
        reasons.append("temporal_warp")
    if first_frame_fidelity < min_first:
        reasons.append("first_frame_mismatch")
    if style_drift > max_style:
        reasons.append("style_drift")
    if new_text_count > 0:
        reasons.append("text_leakage")

    overall_pass = len(reasons) == 0
    return {
        "identity_similarity": identity_similarity,
        "new_text_count": new_text_count,
        "flicker_score": flicker_score,
        "motion_score": motion_score,
        "warp_error": warp_error,
        "first_frame_fidelity": first_frame_fidelity,
        "style_drift": style_drift,
        "metrics_version": METRICS_VERSION,
        "overall_pass": bool(overall_pass),
        "reasons": reasons,
    }
