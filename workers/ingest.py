"""Ingest module for vertical-scroll manhwa.

Normalizes image sizes and stitches multiple consecutive pages into
a single seamless vertical strip.
"""
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import logging

logger = logging.getLogger("m2a.ingest")

def stitch_vertical_pages(page_paths: List[Path], output_path: Path, target_width: int = 800) -> Tuple[Path, Tuple[int, int]]:
    """Stitches a list of vertical-scroll manhwa pages into a single vertical strip.
    
    Args:
        page_paths: List of file paths to the input pages in reading order.
        output_path: Path where the stitched image will be saved.
        target_width: Normalized width to resize pages to (preserves aspect ratio).
        
    Returns:
        (output_path, (total_width, total_height))
    """
    if not page_paths:
        raise ValueError("No pages provided to ingest")
        
    loaded_images = []
    total_height = 0
    
    for p in page_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Page file not found: {p}")
        img = Image.open(p).convert("RGB")
        
        # Scale to target width while keeping aspect ratio
        if img.width != target_width:
            scale = target_width / img.width
            new_h = int(img.height * scale)
            img = img.resize((target_width, new_h), Image.Resampling.LANCZOS)
            
        loaded_images.append(img)
        total_height += img.height
        
    # Create canvas
    canvas = Image.new("RGB", (target_width, total_height), (255, 255, 255))
    current_y = 0
    for img in loaded_images:
        canvas.paste(img, (0, current_y))
        current_y += img.height
        
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    logger.info(f"Stitched {len(page_paths)} pages -> {output_path} ({target_width}x{total_height})")
    return output_path, (target_width, total_height)
