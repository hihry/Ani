"""Ingest module for vertical-scroll manhwa.

Normalizes image sizes, applies EXIF orientation, composites alpha channels onto a
white background, and stitches multiple consecutive pages into a single seamless
vertical strip while tracking and persisting per-page vertical offsets.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageOps
import logging

logger = logging.getLogger("m2a.ingest")


def _load_rgb_page(path: Path) -> Image.Image:
    """Opens an image, applies EXIF transpose, and converts to RGB with white background for alpha."""
    raw = ImageOps.exif_transpose(Image.open(path))
    if raw.mode in ("RGBA", "LA") or (raw.mode == "P" and "transparency" in raw.info):
        rgba = raw.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return raw.convert("RGB")


def stitch_vertical_pages_with_metadata(
    page_paths: List[Path],
    output_path: Path,
    target_width: int = 800,
) -> Dict[str, Any]:
    """Stitches vertical-scroll manhwa pages and returns stitch metadata including page offsets."""
    if not page_paths:
        raise ValueError("No pages provided to ingest")

    loaded_images: List[Image.Image] = []
    page_offsets: List[Dict[str, Any]] = []
    total_height = 0

    for p in page_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Page file not found: {p}")
        img = _load_rgb_page(p)
        orig_w, orig_h = img.size

        # Scale to target width while keeping aspect ratio
        scale = 1.0
        if img.width != target_width:
            scale = target_width / float(img.width)
            new_h = max(1, int(round(img.height * scale)))
            img = img.resize((target_width, new_h), Image.Resampling.LANCZOS)

        page_offsets.append(
            {
                "page": p.name,
                "source_path": str(p.as_posix()),
                "y_offset": total_height,
                "height": img.height,
                "width": target_width,
                "orig_width": orig_w,
                "orig_height": orig_h,
                "scale": scale,
            }
        )
        loaded_images.append(img)
        total_height += img.height

    canvas = Image.new("RGB", (target_width, total_height), (255, 255, 255))
    current_y = 0
    for img in loaded_images:
        canvas.paste(img, (0, current_y))
        current_y += img.height

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=1)

    meta_path = output_path.parent / "stitch_meta.json"
    try:
        meta_path.write_text(
            json.dumps(
                {
                    "stitched_path": str(output_path.as_posix()),
                    "width": target_width,
                    "height": total_height,
                    "page_offsets": page_offsets,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug(f"Could not write stitch_meta.json: {e}")

    logger.info(
        f"Stitched {len(page_paths)} pages -> {output_path} ({target_width}x{total_height})"
    )
    return {
        "stitched_path": output_path,
        "width": target_width,
        "height": total_height,
        "page_offsets": page_offsets,
    }


def stitch_vertical_pages(
    page_paths: List[Path],
    output_path: Path,
    target_width: int = 800,
) -> Tuple[Path, Tuple[int, int]]:
    """Stitches a list of vertical-scroll manhwa pages into a single vertical strip.

    Args:
        page_paths: List of file paths to the input pages in reading order.
        output_path: Path where the stitched image will be saved.
        target_width: Normalized width to resize pages to (preserves aspect ratio).

    Returns:
        (output_path, (total_width, total_height))
    """
    meta = stitch_vertical_pages_with_metadata(
        page_paths=page_paths,
        output_path=output_path,
        target_width=target_width,
    )
    return meta["stitched_path"], (meta["width"], meta["height"])
