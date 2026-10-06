"""Segmentation node for the Manhwa-to-Anime pipeline."""

import logging
import time
from pathlib import Path
import cv2
import numpy as np

# TODO: Import actual state types
# from m2a.state import RunState, PipelineConfig

logger = logging.getLogger(__name__)

def segment(state: 'RunState', config: 'PipelineConfig') -> 'RunState':
    """Detect panels from the stitched strip.
    
    Args:
        state: Current run state.
        config: Pipeline configuration.
        
    Returns:
        Updated run state.
    """
    start_time = time.time()
    logger.info("Starting segment node...")
    
    try:
        stitched_path = state.stitched_image_path
        if not stitched_path or not stitched_path.exists():
            raise FileNotFoundError("Stitched image not found in state.")
            
        img = cv2.imread(str(stitched_path))
        if img is None:
            raise ValueError(f"Could not read image from {stitched_path}")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Dilate to connect components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        panels_dir = state.workspace_dir / "run" / "panels"
        panels_dir.mkdir(parents=True, exist_ok=True)
        
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter out tiny noise contours
            if w > 50 and h > 50:
                bboxes.append((x, y, w, h))
                
        # Sort by y first, then x (with some tolerance for y)
        def sort_key(bbox):
            x, y, w, h = bbox
            return (y // 50, x)
            
        bboxes.sort(key=sort_key)
        
        for i, (x, y, w, h) in enumerate(bboxes):
            panel_crop = img[y:y+h, x:x+w]
            panel_path = panels_dir / f"panel_{i:03d}.png"
            cv2.imwrite(str(panel_path), panel_crop)
            
            # Create PanelJSON stub
            panel_stub = {
                "id": f"panel_{i:03d}",
                "path": str(panel_path),
                "bbox": [x, y, w, h]
            }
            state.panels.append(panel_stub)
            
        logger.info(f"Detected {len(bboxes)} panels.")
        
    except Exception as e:
        logger.exception("Error in segment node")
        state.errors.append(str(e))
        
    logger.info(f"Segment node completed in {time.time() - start_time:.2f}s")
    return state
