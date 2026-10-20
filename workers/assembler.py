"""Video assembly module for Phase 1.

Chains panel clips together with crossfade transitions and encodes the final
timeline into:
  1. H.264 MP4 (if ffmpeg is available)
  2. MJPEG AVI (zero-dependency pure-Python video container)
  3. Animated WebP (browser-compatible high fidelity animation)
"""
import io
import struct
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger("m2a.assembler")

def crossfade_transition(clip_a: List[Image.Image], clip_b: List[Image.Image], transition_frames: int = 12) -> List[Image.Image]:
    """Blends the tail of clip_a with the head of clip_b using linear alpha crossfade."""
    if len(clip_a) < transition_frames or len(clip_b) < transition_frames:
        return clip_a + clip_b
        
    lead_a = clip_a[:-transition_frames]
    tail_a = clip_a[-transition_frames:]
    head_b = clip_b[:transition_frames]
    tail_b = clip_b[transition_frames:]
    
    blended = []
    for i in range(transition_frames):
        alpha = float(i + 1) / float(transition_frames + 1)
        # PIL blend
        blended_frame = Image.blend(tail_a[i], head_b[i], alpha)
        blended.append(blended_frame)
        
    return lead_a + blended + tail_b

def encode_mjpeg_avi(frames: List[Image.Image], output_path: Path, fps: int = 24, quality: int = 85) -> Path:
    """Encodes a sequence of PIL images into a playable Motion-JPEG AVI file."""
    if not frames:
        raise ValueError("Cannot encode empty frame list")
        
    w, h = frames[0].size
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    jpeg_bytes_list = []
    for f in frames:
        buf = io.BytesIO()
        f.save(buf, format="JPEG", quality=quality)
        b = buf.getvalue()
        if len(b) % 2 != 0:
            b += b"\x00"
        jpeg_bytes_list.append(b)
        
    num_frames = len(frames)
    us_per_frame = int(1_000_000 / fps)
    
    # 1. movi chunk
    movi_data = io.BytesIO()
    idx1_entries = []
    offset = 4 # offset relative to movi chunk body
    
    for b in jpeg_bytes_list:
        chunk_header = b"00dc" + struct.pack("<I", len(b))
        movi_data.write(chunk_header + b)
        idx1_entries.append((b"00dc", 0x10, offset, len(b)))
        offset += len(chunk_header) + len(b)
        
    movi_bytes = b"movi" + movi_data.getvalue()
    movi_chunk = b"LIST" + struct.pack("<I", len(movi_bytes)) + movi_bytes
    
    # 2. idx1 chunk
    idx1_data = io.BytesIO()
    for fourcc, flags, off, length in idx1_entries:
        idx1_data.write(fourcc + struct.pack("<III", flags, off, length))
    idx1_chunk = b"idx1" + struct.pack("<I", len(idx1_data.getvalue())) + idx1_data.getvalue()
    
    # 3. strf chunk (BITMAPINFOHEADER)
    strf_data = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, int.from_bytes(b"MJPG", "little"), w * h * 3, 0, 0, 0, 0)
    strf_chunk = b"strf" + struct.pack("<I", len(strf_data)) + strf_data
    
    # 4. strh chunk
    strh_data = struct.pack("<4s4sIIIIIIIIII", b"vids", b"MJPG", 0, 0, 0, 1, fps, 0, num_frames, 1000000, 0xFFFFFFFF, 0)
    strh_chunk = b"strh" + struct.pack("<I", len(strh_data)) + strh_data
    
    strl_bytes = b"strl" + strh_chunk + strf_chunk
    strl_chunk = b"LIST" + struct.pack("<I", len(strl_bytes)) + strl_bytes
    
    # 5. avih chunk (14 DWORDs)
    avih_data = struct.pack("<14I", us_per_frame, 0, 0, 0x10, num_frames, 0, 1, 1000000, w, h, 0, 0, 0, 0)
    avih_chunk = b"avih" + struct.pack("<I", len(avih_data)) + avih_data
    
    hdrl_bytes = b"hdrl" + avih_chunk + strl_chunk
    hdrl_chunk = b"LIST" + struct.pack("<I", len(hdrl_bytes)) + hdrl_bytes
    
    # Final RIFF header
    riff_data = hdrl_chunk + movi_chunk + idx1_chunk
    riff_bytes = b"AVI " + riff_data
    riff_chunk = b"RIFF" + struct.pack("<I", len(riff_bytes)) + riff_bytes
    
    with open(output_path, "wb") as f:
        f.write(riff_chunk)
        
    logger.info(f"Encoded {num_frames} frames to {output_path} ({w}x{h} @ {fps}fps)")
    return output_path

def assemble_timeline(
    clips: List[List[Image.Image]],
    output_base_path: Path,
    fps: int = 24,
    transition_frames: int = 8,
) -> dict:
    """Assembles multiple panel clips into a single continuous video with transitions.
    
    Produces:
      - .avi: Standard video
      - .webp: Web animation
      - .mp4: H.264 MP4 if FFmpeg is installed
    """
    if not clips:
        raise ValueError("No clips provided to assemble")
        
    # Stitch clips with transitions
    full_timeline = clips[0]
    for next_clip in clips[1:]:
        full_timeline = crossfade_transition(full_timeline, next_clip, transition_frames=transition_frames)
        
    output_base_path = Path(output_base_path)
    output_base_path.parent.mkdir(parents=True, exist_ok=True)
    
    avi_path = output_base_path.with_suffix(".avi")
    webp_path = output_base_path.with_suffix(".webp")
    mp4_path = output_base_path.with_suffix(".mp4")
    
    # 1. Always write AVI
    encode_mjpeg_avi(full_timeline, avi_path, fps=fps)
    
    # 2. Write animated WebP (sampling at 12fps for compact file size)
    step = max(1, fps // 12)
    sample_frames = full_timeline[::step]
    sample_frames[0].save(
        webp_path,
        save_all=True,
        append_images=sample_frames[1:],
        duration=int(1000 / 12),
        loop=0,
    )
    
    # 3. If ffmpeg is in PATH, convert to H.264 MP4
    ffmpeg_exe = shutil.which("ffmpeg")
    has_mp4 = False
    if ffmpeg_exe:
        try:
            cmd = [
                ffmpeg_exe, "-y",
                "-i", str(avi_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(mp4_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            has_mp4 = True
            logger.info(f"Generated H.264 MP4 via FFmpeg: {mp4_path}")
        except Exception as e:
            logger.warning(f"FFmpeg MP4 conversion failed: {e}")
            
    return {
        "total_frames": len(full_timeline),
        "duration_s": len(full_timeline) / float(fps),
        "avi_path": str(avi_path),
        "webp_path": str(webp_path),
        "mp4_path": str(mp4_path) if has_mp4 else str(avi_path),
    }
