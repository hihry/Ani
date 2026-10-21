"""Gutter slicing panel detector for vertical-scroll manhwa.

Uses vectorized 2D row and column variance analysis to detect both horizontal and
vertical gutters (whitespace/background bands), extracts exact panel bounding boxes
without gutter bleed, assigns band-clustered reading order, and crops individual panels.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger("m2a.slicer")


def is_gutter_line(
    line_pixels: np.ndarray,
    variance_thresh: float = 12.0,
    bg_bright_thresh: float = 240.0,
    bg_dark_thresh: float = 10.0,
) -> bool:
    """Checks if a 1D line of RGB pixels (row or column) is a uniform gutter."""
    std_val = float(np.std(line_pixels, axis=0).mean())
    mean_val = float(np.mean(line_pixels))
    if std_val < variance_thresh and (mean_val > bg_bright_thresh or mean_val < bg_dark_thresh):
        return True
    return False


def is_gutter_row(
    row_pixels: np.ndarray,
    variance_thresh: float = 12.0,
    bg_bright_thresh: float = 240.0,
) -> bool:
    """Checks if a horizontal row is a uniform gutter."""
    return is_gutter_line(row_pixels, variance_thresh=variance_thresh, bg_bright_thresh=bg_bright_thresh)


def _gutter_mask_1d(
    arr: np.ndarray,
    axis: int,
    variance_thresh: float = 12.0,
    bg_bright_thresh: float = 240.0,
    bg_dark_thresh: float = 10.0,
) -> np.ndarray:
    """Vectorized gutter classification along axis (1 for rows, 0 for columns)."""
    n = float(arr.shape[axis])
    s1 = arr.sum(axis=axis, dtype=np.float64)
    s2 = np.square(arr, dtype=np.float32).sum(axis=axis, dtype=np.float64)
    mean_c = s1 / n
    var_c = np.maximum(0.0, (s2 / n) - np.square(mean_c))
    std_mean = np.sqrt(var_c).mean(axis=1)
    lum_mean = mean_c.mean(axis=1)
    return (std_mean < variance_thresh) & ((lum_mean > bg_bright_thresh) | (lum_mean < bg_dark_thresh))


def _find_contiguous_content_spans(
    is_gutter: np.ndarray,
    min_gutter_size: int,
    min_content_size: int,
) -> List[Tuple[int, int]]:
    """Finds contiguous content spans separated by true gutters of at least min_gutter_size."""
    n = len(is_gutter)
    if n == 0:
        return []

    padded_g = np.empty(n + 2, dtype=np.int8)
    padded_g[0] = 0
    padded_g[1:-1] = is_gutter.astype(np.int8)
    padded_g[-1] = 0
    diff_g = np.diff(padded_g)
    starts_g = np.flatnonzero(diff_g == 1)
    ends_g = np.flatnonzero(diff_g == -1)

    cleaned_gutter = np.zeros(n, dtype=bool)
    for s, e in zip(starts_g, ends_g):
        if (e - s) >= min_gutter_size:
            cleaned_gutter[s:e] = True

    padded_c = np.empty(n + 2, dtype=np.int8)
    padded_c[0] = 1
    padded_c[1:-1] = cleaned_gutter.astype(np.int8)
    padded_c[-1] = 1
    diff_c = np.diff(padded_c)
    starts_c = np.flatnonzero(diff_c == -1)
    ends_c = np.flatnonzero(diff_c == 1)

    return [
        (int(s), int(e))
        for s, e in zip(starts_c, ends_c)
        if (e - s) >= min_content_size
    ]


def infer_shot_type(box_w: int, box_h: int, page_w: int = 800) -> str:
    """Heuristic shot type classification based on panel geometry."""
    aspect = box_h / float(max(1, box_w))
    if box_h >= 850 and aspect >= 1.15:
        return "close_up"
    if box_h >= 610 and box_w >= int(page_w * 0.88):
        return "splash"
    if aspect <= 0.70 and box_w >= 700:
        return "wide"
    return "medium"


def detect_panels_gutter(
    image_path: Path,
    min_panel_height: int = 120,
    min_panel_width: int = 140,
    min_gutter_height: int = 15,
    min_gutter_width: int = 15,
    reading_order_y_tolerance: int = 50,
    img: Optional[Image.Image] = None,
) -> List[Dict[str, Any]]:
    """Detects panels in a manhwa page or strip using vectorized 2D gutter slicing.

    Fixes Phase 1 flaws:
      1. Eliminates the 5px white gutter margin bleed on panel crops.
      2. Detects vertical gutters within horizontal bands AND recursive horizontal
         gutters within columns (supporting 2-column, 3-column, and asymmetric T-layouts).
      3. Uses vertical band grouping when sorting reading order (top-to-bottom,
         left-to-right), avoiding bucket-boundary split errors.
    """
    if img is None:
        img = Image.open(image_path).convert("RGB")
    else:
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape

    # 1. Vectorized classification of each horizontal row as gutter or content
    row_is_gutter = _gutter_mask_1d(
        arr, axis=1, variance_thresh=12.0, bg_bright_thresh=240.0, bg_dark_thresh=10.0
    )

    # 2. Extract vertical content bands
    y_slices = _find_contiguous_content_spans(
        row_is_gutter,
        min_gutter_size=min_gutter_height,
        min_content_size=min_panel_height,
    )

    # 3. Within each vertical slice, detect horizontal spans and recursive sub-rows (supports asymmetric T-splits)
    detected_boxes: List[Tuple[int, Dict[str, int]]] = []
    for band_idx, (y1, y2) in enumerate(y_slices):
        band = arr[y1:y2, :, :]
        col_is_gutter = _gutter_mask_1d(
            band, axis=0, variance_thresh=10.0, bg_bright_thresh=245.0, bg_dark_thresh=10.0
        )

        x_spans = _find_contiguous_content_spans(
            col_is_gutter,
            min_gutter_size=min_gutter_width,
            min_content_size=min_panel_width,
        )

        for x1, x2 in x_spans:
            sub_band = arr[y1:y2, x1:x2, :]
            sub_row_gutter = _gutter_mask_1d(
                sub_band, axis=1, variance_thresh=12.0, bg_bright_thresh=240.0, bg_dark_thresh=10.0
            )
            sub_y_spans = _find_contiguous_content_spans(
                sub_row_gutter,
                min_gutter_size=min_gutter_height,
                min_content_size=min_panel_height,
            )
            if not sub_y_spans:
                active_rows = np.flatnonzero(~sub_row_gutter)
                if len(active_rows) > 0:
                    sub_y_spans = [(int(active_rows[0]), int(active_rows[-1]) + 1)]
                else:
                    sub_y_spans = [(0, y2 - y1)]

            for sy1, sy2 in sub_y_spans:
                ry1 = y1 + sy1
                ry2 = y1 + sy2
                # Trim any horizontal gutter padding inside this sub-panel
                leaf = arr[ry1:ry2, x1:x2, :]
                leaf_col_gutter = _gutter_mask_1d(
                    leaf, axis=0, variance_thresh=10.0, bg_bright_thresh=245.0, bg_dark_thresh=10.0
                )
                active_cols = np.flatnonzero(~leaf_col_gutter)
                if len(active_cols) > 0:
                    rx1 = x1 + int(active_cols[0])
                    rx2 = x1 + int(active_cols[-1]) + 1
                else:
                    rx1, rx2 = x1, x2

                box_w = rx2 - rx1
                box_h = ry2 - ry1
                if box_w >= min_panel_width and box_h >= min_panel_height:
                    detected_boxes.append(
                        (band_idx, {"x": int(rx1), "y": int(ry1), "w": int(box_w), "h": int(box_h)})
                    )

    # 4. Sort reading order: within each band, group boxes whose top y is within reading_order_y_tolerance
    detected_boxes.sort(key=lambda item: (item[0], item[1]["y"], item[1]["x"]))
    ordered_boxes: List[Dict[str, int]] = []
    i = 0
    while i < len(detected_boxes):
        b_idx, first_box = detected_boxes[i]
        row_group = [first_box]
        j = i + 1
        while j < len(detected_boxes):
            next_b_idx, next_box = detected_boxes[j]
            if next_b_idx == b_idx and abs(next_box["y"] - first_box["y"]) <= reading_order_y_tolerance:
                row_group.append(next_box)
                j += 1
            else:
                break
        row_group.sort(key=lambda b: (b["x"], b["y"]))
        ordered_boxes.extend(row_group)
        i = j

    panels: List[Dict[str, Any]] = []
    for idx, box in enumerate(ordered_boxes, start=1):
        panels.append(
            {
                "panel_id": f"panel_{idx:03d}",
                "bbox": box,
                "reading_order": idx,
                "shot_type": infer_shot_type(box["w"], box["h"], page_w=w),
            }
        )

    logger.info(f"Detected {len(panels)} panels in {image_path}")
    return panels


def crop_and_save_panels(
    image_path: Path,
    panels: List[Dict[str, Any]],
    output_dir: Path,
    img: Optional[Image.Image] = None,
) -> List[Path]:
    """Crops detected panels from the source image and saves them to output_dir."""
    if img is None:
        img = Image.open(image_path).convert("RGB")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_paths: List[Path] = []
    for p in panels:
        bbox = p["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        crop = img.crop((x, y, x + w, y + h))
        crop_file = output_dir / f"{p['panel_id']}.png"
        crop.save(crop_file, format="PNG", compress_level=1)
        crop_paths.append(crop_file)

    return crop_paths
