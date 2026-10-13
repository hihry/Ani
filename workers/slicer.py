"""Gutter slicing panel detector for vertical-scroll manhwa.

Uses row and column variance analysis to detect gutters (whitespace/background bands),
extracts panel bounding boxes, assigns reading order, and crops individual panels.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Any
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger("m2a.slicer")

def is_gutter_row(row_pixels: np.ndarray, variance_thresh: float = 12.0, bg_bright_thresh: float = 240.0) -> bool:
    """Checks if a horizontal row is a uniform gutter."""
    # Std dev across width for each channel
    std_val = float(np.std(row_pixels, axis=0).mean())
    mean_val = float(np.mean(row_pixels))
    # Row is gutter if low variance AND (high brightness or dark border)
    if std_val < variance_thresh and (mean_val > bg_bright_thresh or mean_val < 25.0):
        return True
    return False

def detect_panels_gutter(
    image_path: Path,
    min_panel_height: int = 150,
    min_panel_width: int = 200,
    min_gutter_height: int = 15,
) -> List[Dict[str, Any]]:
    """Detects panels in a manhwa page or strip using gutter slicing.
    
    Returns a list of dicts with:
      - panel_id: e.g. "panel_001"
      - bbox: {"x": int, "y": int, "w": int, "h": int}
      - reading_order: int (1-indexed)
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    
    # 1. Classify each row as gutter or content
    is_gutter = np.zeros(h, dtype=bool)
    for y in range(h):
        row = arr[y, :, :]
        is_gutter[y] = is_gutter_row(row)
        
    # 2. Smooth small gutter interruptions (noise < min_gutter_height)
    # A true gutter must be at least min_gutter_height contiguous rows
    cleaned_gutter = np.zeros(h, dtype=bool)
    start_g = None
    for y in range(h):
        if is_gutter[y]:
            if start_g is None:
                start_g = y
        else:
            if start_g is not None:
                if (y - start_g) >= min_gutter_height:
                    cleaned_gutter[start_g:y] = True
                start_g = None
    if start_g is not None and (h - start_g) >= min_gutter_height:
        cleaned_gutter[start_g:h] = True
        
    # 3. Extract content slices (non-gutter spans)
    slices = []
    in_panel = False
    start_y = 0
    for y in range(h):
        if not cleaned_gutter[y] and not in_panel:
            in_panel = True
            start_y = y
        elif cleaned_gutter[y] and in_panel:
            in_panel = False
            panel_h = y - start_y
            if panel_h >= min_panel_height:
                slices.append((start_y, y))
    if in_panel:
        panel_h = h - start_y
        if panel_h >= min_panel_height:
            slices.append((start_y, h))
            
    # 4. For each vertical slice, refine horizontal boundaries [x_min, x_max]
    detected_boxes = []
    for (y1, y2) in slices:
        panel_slice = arr[y1:y2, :, :]
        
        # Check column variance within this slice
        col_std = np.std(panel_slice, axis=0).mean(axis=1) # shape (w,)
        col_mean = np.mean(panel_slice, axis=(0, 2))       # shape (w,)
        
        is_col_content = (col_std > 10.0) | ((col_mean < 245.0) & (col_mean > 20.0))
        content_indices = np.where(is_col_content)[0]
        
        if len(content_indices) > 0:
            x1 = max(0, int(content_indices[0]) - 5)
            x2 = min(w, int(content_indices[-1]) + 5)
        else:
            x1 = 0
            x2 = w
            
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w >= min_panel_width and box_h >= min_panel_height:
            detected_boxes.append({"x": x1, "y": y1, "w": box_w, "h": box_h})
            
    # 5. Sort reading order (top to bottom, left to right with tolerance)
    # Group by vertical proximity (within 50px)
    detected_boxes.sort(key=lambda b: (b["y"], b["x"]))
    
    panels = []
    for idx, box in enumerate(detected_boxes, start=1):
        panels.append({
            "panel_id": f"panel_{idx:03d}",
            "bbox": box,
            "reading_order": idx,
        })
        
    logger.info(f"Detected {len(panels)} panels in {image_path}")
    return panels

def crop_and_save_panels(image_path: Path, panels: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Crops detected panels from the source image and saves them to output_dir."""
    img = Image.open(image_path).convert("RGB")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    crop_paths = []
    for p in panels:
        bbox = p["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        crop = img.crop((x, y, x + w, y + h))
        crop_file = output_dir / f"{p['panel_id']}.png"
        crop.save(crop_file, quality=95)
        crop_paths.append(crop_file)
        
    return crop_paths
