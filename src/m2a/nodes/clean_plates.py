"""Clean plates node for the Manhwa-to-Anime pipeline."""

import logging
import time
from pathlib import Path
import cv2
import numpy as np

# TODO: Import actual state types
# from m2a.state import RunState, PipelineConfig

logger = logging.getLogger(__name__)

def clean_plates(state: 'RunState', config: 'PipelineConfig') -> 'RunState':
    """Create text masks and run inpainting to remove text from panels.
    
    Args:
        state: Current run state.
        config: Pipeline configuration.
        
    Returns:
        Updated run state.
    """
    start_time = time.time()
    logger.info("Starting clean_plates node...")
    
    try:
        plates_dir = state.workspace_dir / "run" / "clean_plates"
        plates_dir.mkdir(parents=True, exist_ok=True)
        
        for panel in state.panels:
            panel_path = Path(panel["path"])
            img = cv2.imread(str(panel_path))
            
            if img is None:
                continue
                
            # Create a blank mask
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            # Draw filled rectangles on text bboxes
            dialogue = panel.get("understanding", {}).get("dialogue", [])
            for text_item in dialogue:
                if "bbox" in text_item:
                    x, y, w, h = text_item["bbox"]
                    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                    
            # TODO: Call LaMa provider for actual inpainting instead of this simple cv2 inpaint
            # For now, using cv2.inpaint as a stub
            clean_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            
            # TODO: Handle aspect ratio normalization for video generation (e.g., pad to 16:9)
            
            clean_path = plates_dir / f"clean_{panel_path.name}"
            cv2.imwrite(str(clean_path), clean_img)
            
            panel["clean_plate_path"] = str(clean_path)
            
            # Update cost ledger
            if hasattr(state, "cost_ledger"):
                state.cost_ledger.add_cost("inpainting_api", 0.005)
            
    except Exception as e:
        logger.exception("Error in clean_plates node")
        state.errors.append(str(e))
        
    logger.info(f"Clean plates node completed in {time.time() - start_time:.2f}s")
    return state
