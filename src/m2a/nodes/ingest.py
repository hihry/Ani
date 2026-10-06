"""Ingestion node for the Manhwa-to-Anime pipeline."""

import logging
import time
from pathlib import Path
from PIL import Image

# TODO: Import actual state types
# from m2a.state import RunState, PipelineConfig

logger = logging.getLogger(__name__)

def ingest(state: 'RunState', config: 'PipelineConfig') -> 'RunState':
    """Ingest source manhwa pages and stitch them into a single strip.
    
    Args:
        state: Current run state.
        config: Pipeline configuration.
        
    Returns:
        Updated run state.
    """
    start_time = time.time()
    logger.info("Starting ingest node...")
    
    try:
        # TODO: Get input directory from state or config
        input_dir = state.workspace_dir / "inputs"
        run_folder = state.workspace_dir / "run"
        run_folder.mkdir(parents=True, exist_ok=True)
        
        image_paths = sorted(
            [p for p in input_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        )
        
        if not image_paths:
            raise FileNotFoundError(f"No images found in {input_dir}")
            
        images = []
        for p in image_paths:
            try:
                img = Image.open(p)
                img.load() # Validate it's a real image
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to load image {p}: {e}")
                
        if not images:
            raise ValueError("No valid images could be loaded.")
            
        # Stitch images vertically
        widths, heights = zip(*(i.size for i in images))
        
        max_width = max(widths)
        total_height = sum(heights)
        
        stitched_image = Image.new('RGB', (max_width, total_height), color='white')
        
        y_offset = 0
        for img in images:
            # Center horizontally if needed
            x_offset = (max_width - img.width) // 2
            stitched_image.paste(img, (x_offset, y_offset))
            y_offset += img.height
            
        output_path = run_folder / "stitched_strip.png"
        stitched_image.save(output_path)
        logger.info(f"Saved stitched image to {output_path}")
        
        # Update state
        state.stitched_image_path = output_path
        
    except Exception as e:
        logger.exception("Error in ingest node")
        state.errors.append(str(e))
        
    logger.info(f"Ingest node completed in {time.time() - start_time:.2f}s")
    return state
