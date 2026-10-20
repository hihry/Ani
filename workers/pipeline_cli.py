"""Command-line bridge for Go backend to invoke Python AI/CV workers.

Commands:
  - ingest: stitch vertical pages into single strip
  - segment: detect panels via gutter slicing & save crops
  - baseline: full Phase 1 end-to-end motion-comic generator
"""
import sys
import json
import argparse
from pathlib import Path

from ingest import stitch_vertical_pages
from slicer import detect_panels_gutter, crop_and_save_panels
from render_kenburns import generate_kenburns_frames
from render_parallax import generate_parallax_frames
from assembler import assemble_timeline

def cmd_ingest(args):
    pages = [Path(p) for p in args.pages.split(",")]
    out = Path(args.output)
    out_path, (w, h) = stitch_vertical_pages(pages, out)
    print(json.dumps({"stitched_path": str(out_path), "width": w, "height": h}))

def cmd_segment(args):
    image_path = Path(args.image)
    panels = detect_panels_gutter(image_path)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        crop_paths = crop_and_save_panels(image_path, panels, out_dir)
        for p, cp in zip(panels, crop_paths):
            p["crop_path"] = str(cp)
    print(json.dumps({"panels": panels}))

def cmd_baseline(args):
    pages = [Path(p) for p in args.pages.split(",")]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ingest / stitch
    stitched_path = run_dir / "stitched.png"
    stitch_vertical_pages(pages, stitched_path)
    
    # 2. Segment
    panels_dir = run_dir / "panels"
    panels = detect_panels_gutter(stitched_path)
    crop_paths = crop_and_save_panels(stitched_path, panels, panels_dir)
    for p, cp in zip(panels, crop_paths):
        p["crop_path"] = str(cp)
        
    # 3. Render 2D Motion (alternating Ken Burns and Parallax)
    clips = []
    fps = args.fps
    duration_per_panel = args.duration
    
    for idx, (p, cp) in enumerate(zip(panels, crop_paths)):
        # Alternate motion type for visual variety
        if idx % 2 == 0:
            frames = generate_kenburns_frames(
                cp,
                target_res=(720, 1280),
                duration_s=duration_per_panel,
                fps=fps,
                motion_type="zoom_in" if idx % 4 == 0 else "pan_down",
            )
        else:
            frames = generate_parallax_frames(
                cp,
                target_res=(720, 1280),
                duration_s=duration_per_panel,
                fps=fps,
            )
        clips.append(frames)
        
    # 4. Assemble with transitions
    output_base = run_dir / "baseline"
    result = assemble_timeline(clips, output_base, fps=fps, transition_frames=8)
    
    result["panels"] = panels
    result["num_panels"] = len(panels)
    print(json.dumps(result))

def main():
    parser = argparse.ArgumentParser(description="M2A Python Worker CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_ingest = subparsers.add_parser("ingest")
    p_ingest.add_argument("--pages", required=True, help="Comma-separated page paths")
    p_ingest.add_argument("--output", required=True, help="Output stitched image path")
    
    p_segment = subparsers.add_parser("segment")
    p_segment.add_argument("--image", required=True, help="Stitched image path")
    p_segment.add_argument("--output-dir", help="Directory to save panel crops")
    
    p_base = subparsers.add_parser("baseline")
    p_base.add_argument("--pages", required=True, help="Comma-separated page paths")
    p_base.add_argument("--run-dir", required=True, help="Run output directory")
    p_base.add_argument("--duration", type=float, default=2.5, help="Duration per panel in seconds")
    p_base.add_argument("--fps", type=int, default=24, help="Frames per second")
    
    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "segment":
        cmd_segment(args)
    elif args.command == "baseline":
        cmd_baseline(args)

if __name__ == "__main__":
    main()
