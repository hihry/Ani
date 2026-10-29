"""Video assembly, clean-plate inpainting, QC metrics, and audio mixing module.

Chains panel clips together with crossfade transitions, computes frame-level
auto-QC metrics, inpaints speech bubbles for clean plates, synthesizes fallback
speech WAV tracks, mixes timed audio tracks anchored to panel timestamps, and
encodes the final timeline into:
  1. H.264 MP4 with mixed AAC audio & subtitles (when ffmpeg is available)
  2. Standards-compliant MJPEG AVI (streamed pure-Python video container)
  3. Animated WebP (browser-compatible high-fidelity animation)
"""
from collections import deque
import io
import math
import os
import shutil
import struct
import subprocess
import textwrap
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from workers.understanding import count_residual_text_bubbles, detect_speech_bubbles_and_text
except ImportError:
    from understanding import count_residual_text_bubbles, detect_speech_bubbles_and_text

try:
    from workers.qc_metrics import compute_clip_qc_metrics
except ImportError:
    from qc_metrics import compute_clip_qc_metrics

logger = logging.getLogger("m2a.assembler")

_FONT_CANDIDATES = [
    os.environ.get("M2A_SUB_FONT", ""),
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_subtitle_font(size: int = 22) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def crossfade_transition(
    clip_a: List[Image.Image],
    clip_b: List[Image.Image],
    transition_frames: int = 12,
) -> List[Image.Image]:
    """Blends the tail of clip_a with the head of clip_b using linear alpha crossfade."""
    if transition_frames <= 0 or len(clip_a) < transition_frames or len(clip_b) < transition_frames:
        return clip_a + clip_b

    lead_a = clip_a[:-transition_frames]
    tail_a = clip_a[-transition_frames:]
    head_b = clip_b[:transition_frames]
    tail_b = clip_b[transition_frames:]

    blended: List[Image.Image] = []
    for i in range(transition_frames):
        alpha = float(i + 1) / float(transition_frames + 1)
        blended_frame = Image.blend(tail_a[i], head_b[i], alpha)
        blended.append(blended_frame)

    return lead_a + blended + tail_b


def inpaint_panel_bubbles(
    image_path: Path,
    output_path: Path,
    bubble_bboxes: Optional[List[Dict[str, int]]] = None,
    mask_path: Optional[Path] = None,
    mask_output_path: Optional[Path] = None,
    img: Optional[Image.Image] = None,
) -> Path:
    """Removes speech bubbles, borders, pointer tails, and lettering from a panel image to produce a clean plate."""
    img_pil = img.convert("RGB") if img is not None else Image.open(image_path).convert("RGB")
    arr = np.array(img_pil, dtype=np.uint8)
    h, w, _ = arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Load explicit binary mask if provided
    if mask_path is not None and Path(mask_path).is_file():
        m_img = Image.open(mask_path).convert("L").resize((w, h), Image.Resampling.NEAREST)
        mask = (np.asarray(m_img, dtype=np.uint8) > 127).astype(np.uint8) * 255

    # 2. Mask explicit dialogue / SFX bounding boxes (expanding downward to cover pointer tails & dark outlines)
    if bubble_bboxes:
        for b in bubble_bboxes:
            bx = max(0, int(b.get("x", 0)) - 8)
            by = max(0, int(b.get("y", 0)) - 8)
            bw = int(b.get("w", 0)) + 16
            bh = int(b.get("h", 0)) + 28  # +20px below bbox bottom to capture speech-bubble pointer tail
            if bw > 16 and bh > 16:
                mask[by : min(h, by + bh), bx : min(w, bx + bw)] = 255

    # 3. Always union with Phase 2 morphological bubble detector envelope when no precomputed mask_path was given
    if mask_path is None:
        detected = detect_speech_bubbles_and_text(img_pil)
        for reg in detected:
            eb = reg.get("envelope_bbox") or reg["bbox"]
            ex = max(0, int(eb["x"]) - 4)
            ey = max(0, int(eb["y"]) - 4)
            ew = int(eb["w"]) + 8
            eh = int(eb["h"]) + 8
            mask[ey : min(h, ey + eh), ex : min(w, ex + ew)] = 255

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if cv2 is not None and np.any(mask > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Pre-fill masked spans using vectorized 2D boundary interpolation so large or edge-touching bubbles leave zero white haze
        clean_arr = arr.copy()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        frame_pad = min(6, max(2, min(w, h) // 40))
        for lbl in range(1, num_labels):
            rx = int(stats[lbl, cv2.CC_STAT_LEFT])
            ry = int(stats[lbl, cv2.CC_STAT_TOP])
            rw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            rh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            ry2 = min(h, ry + rh)
            rx2 = min(w, rx + rw)
            ny = ry2 - ry
            nx = rx2 - rx
            if ny <= 0 or nx <= 0:
                continue

            has_left = (rx - 4) >= frame_pad
            has_right = (rx + rw + 3) < (w - frame_pad)
            lx = max(frame_pad, rx - 4)
            tx = min(w - frame_pad - 1, rx + rw + 3)
            top_y = max(frame_pad, ry - 4)
            bot_y = min(h - frame_pad - 1, ry + rh + 3)

            comp_mask_2d = labels[ry:ry2, rx:rx2] == lbl
            if has_left or has_right:
                left_col = clean_arr[ry:ry2, lx if has_left else tx, :].astype(np.float32)[:, None, :]
                right_col = clean_arr[ry:ry2, tx if has_right else lx, :].astype(np.float32)[:, None, :]
                alphas_x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, :, None]
                block = np.clip((1.0 - alphas_x) * left_col + alphas_x * right_col, 0.0, 255.0).astype(np.uint8)
            else:
                top_row = clean_arr[top_y, rx:rx2, :].astype(np.float32)[None, :, :]
                bot_row = clean_arr[bot_y, rx:rx2, :].astype(np.float32)[None, :, :]
                alphas_y = np.linspace(0.0, 1.0, ny, dtype=np.float32)[:, None, None]
                block = np.clip((1.0 - alphas_y) * top_row + alphas_y * bot_row, 0.0, 255.0).astype(np.uint8)

            sub_view = clean_arr[ry:ry2, rx:rx2]
            sub_view[comp_mask_2d] = block[comp_mask_2d]

        # Smooth boundary transition ring via Telea inpainting
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary_ring = cv2.subtract(mask, eroded)
        if np.any(boundary_ring > 0):
            bgr = cv2.cvtColor(clean_arr, cv2.COLOR_RGB2BGR)
            inpainted_bgr = cv2.inpaint(bgr, boundary_ring, 5, cv2.INPAINT_TELEA)
            clean_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        else:
            clean_rgb = clean_arr
        out_img = Image.fromarray(clean_rgb)
    else:
        out_img = img_pil

    # Persist binary inpaint mask artifact alongside clean plate when requested or following *_clean.png convention
    resolved_mask_out = mask_output_path
    if resolved_mask_out is None and output_path.name.endswith("_clean.png"):
        resolved_mask_out = output_path.with_name(output_path.name[:-10] + "_mask.png")
    if resolved_mask_out is not None:
        Image.fromarray(mask, mode="L").save(Path(resolved_mask_out), format="PNG", compress_level=1)

    out_img.save(output_path, format="PNG", compress_level=1)
    logger.info(f"Created clean plate: {output_path}")
    return output_path


def synthesize_speech_wav(
    text: str,
    output_path: Path,
    voice_id: str = "default",
    emotion: str = "neutral",
    language: str = "ko",
    sample_rate: int = 24000,
) -> float:
    """Synthesizes a playable 16-bit PCM WAV file with cadence and pitch matching the speaker/emotion."""
    clean_text = (text or "").strip()
    duration_s = round(max(1.2, min(8.0, len(clean_text) * 0.065 + 0.6)), 2)
    num_samples = int(duration_s * sample_rate)

    base_freq = 165.0
    if any(k in voice_id.lower() for k in ("elora", "female", "girl")):
        base_freq = 235.0
    elif any(k in voice_id.lower() for k in ("beast", "monster", "system")):
        base_freq = 115.0

    if emotion in ("angry", "urgent", "intense", "fierce"):
        base_freq *= 1.12
    elif emotion in ("sad", "tired", "relieved", "calm"):
        base_freq *= 0.92

    t = np.linspace(0.0, duration_s, num_samples, endpoint=False, dtype=np.float32)
    syllable_env = 0.55 + 0.45 * np.sin(2.0 * math.pi * 4.5 * t)
    fade_len = min(int(0.08 * sample_rate), num_samples // 4)
    envelope = np.ones_like(t) * syllable_env
    if fade_len > 0:
        envelope[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        envelope[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

    wave_sig = (
        0.60 * np.sin(2.0 * math.pi * base_freq * t)
        + 0.25 * np.sin(2.0 * math.pi * (base_freq * 2.0) * t)
        + 0.15 * np.sin(2.0 * math.pi * (base_freq * 3.0) * t)
    )
    pcm = np.clip(wave_sig * envelope * 0.18 * 32767.0, -32767, 32767).astype(np.int16)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    logger.info(f"Synthesized audio track {output_path} ({duration_s:.2f}s)")
    return duration_s


def create_subtitle_overlay(
    frame_size: Tuple[int, int],
    dialogue_lines: List[Dict[str, Any]],
) -> Optional[Tuple[Image.Image, int, int]]:
    """Builds a reusable RGBA subtitle banner overlay and its (x, y) placement coordinates."""
    if not dialogue_lines:
        return None

    combined_text = " / ".join(
        f"{d.get('speaker_id') or d.get('speaker', '')}: {d.get('text', '')}".strip(": ")
        for d in dialogue_lines
        if d.get("text")
    )
    if not combined_text:
        return None

    w, h = frame_size
    font_size = max(16, h // 56)
    font = _load_subtitle_font(size=font_size)
    wrapped = textwrap.wrap(combined_text, width=52)[:2]
    display_text = "\n".join(wrapped)

    box_h = max(68, int(h * 0.065) + (len(wrapped) - 1) * (font_size + 6))
    box_y1 = h - box_h - 48
    box_x1 = int(w * 0.06)
    box_w = int(w * 0.88)

    overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        [0, 0, box_w - 1, box_h - 1],
        radius=10,
        fill=(12, 14, 22, 225),
        outline=(70, 190, 210, 255),
        width=2,
    )
    draw.multiline_text((18, 16), display_text, fill=(245, 248, 255, 255), font=font, spacing=4)
    return overlay, box_x1, box_y1


def burn_subtitle_on_frames(
    frames: List[Image.Image],
    dialogue_lines: List[Dict[str, Any]],
    in_place: bool = False,
) -> List[Image.Image]:
    """Renders anime-styled subtitle overlays onto the bottom safe zone of a panel's frames.

    Optimization (P1-12 / O10): Renders the subtitle banner once onto an RGBA overlay and
    alpha-composites it onto each frame rather than re-drawing rounded rectangles and text per frame.
    """
    if not frames or not dialogue_lines:
        return frames

    ov = create_subtitle_overlay(frames[0].size, dialogue_lines)
    if ov is None:
        return frames
    overlay, box_x1, box_y1 = ov

    if in_place:
        for f in frames:
            f.paste(overlay, (box_x1, box_y1), overlay)
        return frames

    out_frames: List[Image.Image] = []
    for f in frames:
        copy_f = f.copy()
        copy_f.paste(overlay, (box_x1, box_y1), overlay)
        out_frames.append(copy_f)
    return out_frames


class StreamingMjpegAviWriter:
    """Streams PIL frames directly to a Motion-JPEG AVI file on disk with O(1) image memory.

    Rather than accumulating all compressed JPEG frames in RAM, this writer reserves the fixed-size
    RIFF/AVI header on disk, writes each `00dc` chunk immediately as frames arrive, and patches the
    final frame counts and chunk sizes upon `close()`.
    """

    def __init__(self, output_path: Path, fps: int = 24, quality: int = 85) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = max(1, int(fps))
        self.quality = int(quality)
        self.num_frames = 0
        self.width = 0
        self.height = 0
        self._fh: Optional[io.BufferedRandom] = None
        self._idx1_entries: List[bytes] = []
        self._movi_payload_size = 4  # "movi" fourcc
        self._movi_offset = 4
        self._hdrl_size = 0

    def _build_hdrl_chunk(self, w: int, h: int, num_frames: int) -> bytes:
        us_per_frame = int(1_000_000 / self.fps)
        strf_data = struct.pack(
            "<IiiHHIIiiII",
            40,
            w,
            h,
            1,
            24,
            int.from_bytes(b"MJPG", "little"),
            w * h * 3,
            0,
            0,
            0,
            0,
        )
        strf_chunk = b"strf" + struct.pack("<I", len(strf_data)) + strf_data
        strh_data = struct.pack(
            "<4s4sIIIIIIIIIIhhhh",
            b"vids",
            b"MJPG",
            0,
            0,
            0,
            1,
            self.fps,
            0,
            num_frames,
            1000000,
            0xFFFFFFFF,
            0,
            0,
            0,
            w,
            h,
        )
        strh_chunk = b"strh" + struct.pack("<I", len(strh_data)) + strh_data
        strl_bytes = b"strl" + strh_chunk + strf_chunk
        strl_chunk = b"LIST" + struct.pack("<I", len(strl_bytes)) + strl_bytes

        avih_data = struct.pack(
            "<14I",
            us_per_frame,
            0,
            0,
            0x10,
            num_frames,
            0,
            1,
            1000000,
            w,
            h,
            0,
            0,
            0,
            0,
        )
        avih_chunk = b"avih" + struct.pack("<I", len(avih_data)) + avih_data
        hdrl_bytes = b"hdrl" + avih_chunk + strl_chunk
        return b"LIST" + struct.pack("<I", len(hdrl_bytes)) + hdrl_bytes

    def write_frame(self, frame: Image.Image) -> None:
        if self._fh is None:
            self.width, self.height = frame.size
            self._fh = open(self.output_path, "w+b")
            hdrl_chunk = self._build_hdrl_chunk(self.width, self.height, 0)
            self._hdrl_size = len(hdrl_chunk)
            self._fh.write(b"RIFF" + struct.pack("<I", 0) + b"AVI ")
            self._fh.write(hdrl_chunk)
            self._fh.write(b"LIST" + struct.pack("<I", 0) + b"movi")

        if frame.size != (self.width, self.height):
            frame = frame.resize((self.width, self.height), Image.Resampling.BILINEAR)
        if frame.mode != "RGB":
            frame = frame.convert("RGB")

        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=self.quality)
        b = buf.getvalue()
        if len(b) % 2 != 0:
            b += b"\x00"

        self._fh.write(b"00dc" + struct.pack("<I", len(b)))
        self._fh.write(b)
        self._idx1_entries.append(b"00dc" + struct.pack("<III", 0x10, self._movi_offset, len(b)))
        self._movi_offset += 8 + len(b)
        self._movi_payload_size += 8 + len(b)
        self.num_frames += 1

    def close(self) -> Path:
        if self._fh is None or self.num_frames == 0:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            raise ValueError("Cannot encode empty frame list")

        idx1_size = self.num_frames * 16
        self._fh.write(b"idx1" + struct.pack("<I", idx1_size))
        for entry in self._idx1_entries:
            self._fh.write(entry)

        riff_size = 4 + self._hdrl_size + (8 + self._movi_payload_size) + (8 + idx1_size)
        hdrl_chunk = self._build_hdrl_chunk(self.width, self.height, self.num_frames)
        self._fh.seek(0)
        self._fh.write(b"RIFF" + struct.pack("<I", riff_size) + b"AVI ")
        self._fh.write(hdrl_chunk)
        self._fh.write(b"LIST" + struct.pack("<I", self._movi_payload_size) + b"movi")
        self._fh.close()
        self._fh = None
        logger.info(
            f"Encoded {self.num_frames} frames to {self.output_path} ({self.width}x{self.height} @ {self.fps}fps)"
        )
        return self.output_path

    def __enter__(self) -> "StreamingMjpegAviWriter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._fh is not None:
            if exc_type is None and self.num_frames > 0:
                self.close()
            else:
                self._fh.close()
                self._fh = None


def encode_mjpeg_avi(
    frames: Iterable[Image.Image],
    output_path: Path,
    fps: int = 24,
    quality: int = 85,
) -> Path:
    """Encodes a sequence or iterator of PIL images into a standards-compliant Motion-JPEG AVI file."""
    writer = StreamingMjpegAviWriter(output_path=output_path, fps=fps, quality=quality)
    for f in frames:
        writer.write_frame(f)
    return writer.close()


def count_video_frames(clip_path: Path) -> int:
    """Counts the number of frames in an on-disk video clip without decoding full RGB images into RAM."""
    clip_path = Path(clip_path)
    if not clip_path.is_file():
        return 0

    if clip_path.suffix.lower() == ".avi":
        try:
            with open(clip_path, "rb") as fh:
                header = fh.read(64)
                if len(header) >= 52 and header[:4] == b"RIFF" and header[8:12] == b"AVI ":
                    # Scan RIFF chunks for 'movi' list and count '00dc' chunks by seeking over payloads
                    fh.seek(12)
                    file_size = clip_path.stat().st_size
                    while fh.tell() + 8 <= file_size:
                        chunk_hdr = fh.read(8)
                        if len(chunk_hdr) < 8:
                            break
                        fourcc = chunk_hdr[:4]
                        sz = struct.unpack("<I", chunk_hdr[4:8])[0]
                        if fourcc == b"LIST":
                            list_type = fh.read(4)
                            if list_type == b"movi":
                                end_movi = min(file_size, fh.tell() - 4 + sz)
                                count = 0
                                while fh.tell() + 8 <= end_movi:
                                    sub_hdr = fh.read(8)
                                    if len(sub_hdr) < 8:
                                        break
                                    sub_fcc = sub_hdr[:4]
                                    sub_sz = struct.unpack("<I", sub_hdr[4:8])[0]
                                    if sub_fcc == b"00dc":
                                        count += 1
                                    fh.seek(sub_sz + (sub_sz % 2), os.SEEK_CUR)
                                if count > 0:
                                    return count
                                break
                            else:
                                fh.seek(max(0, sz - 4) + (sz % 2), os.SEEK_CUR)
                        else:
                            fh.seek(sz + (sz % 2), os.SEEK_CUR)
        except Exception as e:
            logger.debug(f"Fast RIFF frame count fallback on {clip_path}: {e}")

    if cv2 is not None:
        cap = cv2.VideoCapture(str(clip_path))
        try:
            prop_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if prop_count > 0:
                return prop_count
            count = 0
            while cap.grab():
                count += 1
            return count
        finally:
            cap.release()
    return 0


def iter_video_frames(
    clip_path: Path,
    target_res: Optional[Tuple[int, int]] = None,
) -> Iterator[Image.Image]:
    """Streams PIL RGB frames one at a time from an existing video clip (.avi / .mp4)."""
    clip_path = Path(clip_path)
    if not clip_path.is_file():
        return

    yielded_any = False
    if clip_path.suffix.lower() == ".avi":
        try:
            with open(clip_path, "rb") as fh:
                file_size = clip_path.stat().st_size
                fh.seek(12)
                while fh.tell() + 8 <= file_size:
                    chunk_hdr = fh.read(8)
                    if len(chunk_hdr) < 8:
                        break
                    fourcc = chunk_hdr[:4]
                    sz = struct.unpack("<I", chunk_hdr[4:8])[0]
                    if fourcc == b"LIST":
                        list_type = fh.read(4)
                        if list_type == b"movi":
                            end_movi = min(file_size, fh.tell() - 4 + sz)
                            while fh.tell() + 8 <= end_movi:
                                sub_hdr = fh.read(8)
                                if len(sub_hdr) < 8:
                                    break
                                sub_fcc = sub_hdr[:4]
                                sub_sz = struct.unpack("<I", sub_hdr[4:8])[0]
                                payload = fh.read(sub_sz)
                                if sub_sz % 2 != 0:
                                    fh.seek(1, os.SEEK_CUR)
                                if sub_fcc == b"00dc" and len(payload) == sub_sz:
                                    im = Image.open(io.BytesIO(payload)).convert("RGB")
                                    if target_res and im.size != target_res:
                                        im = im.resize(target_res, Image.Resampling.BILINEAR)
                                    yielded_any = True
                                    yield im
                            break
                        else:
                            fh.seek(max(0, sz - 4) + (sz % 2), os.SEEK_CUR)
                    else:
                        fh.seek(sz + (sz % 2), os.SEEK_CUR)
            if yielded_any:
                return
        except Exception as e:
            logger.debug(f"Streaming RIFF MJPEG decode fallback on {clip_path}: {e}")

    if cv2 is not None:
        cap = cv2.VideoCapture(str(clip_path))
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
                if target_res and im.size != target_res:
                    im = im.resize(target_res, Image.Resampling.BILINEAR)
                yield im
        finally:
            cap.release()


def decode_video_frames(
    clip_path: Path,
    target_res: Optional[Tuple[int, int]] = None,
) -> List[Image.Image]:
    """Decodes an existing video clip (.avi / .mp4) into PIL RGB frames."""
    return list(iter_video_frames(clip_path, target_res=target_res))


def mix_timed_tracks(
    events: List[Tuple[float, Path]],
    total_duration_s: float,
    output_wav_path: Path,
    sample_rate: int = 24000,
) -> Optional[Path]:
    """Mixes per-line WAV files at their exact timeline start offsets (in seconds)."""
    valid_events = [(float(start_s), Path(p)) for start_s, p in events if p and Path(p).is_file()]
    if not valid_events:
        return None

    target_samples = max(1, int(round(total_duration_s * sample_rate)))
    buf = np.zeros(target_samples, dtype=np.float32)

    for start_s, p in valid_events:
        try:
            with wave.open(str(p), "rb") as wf:
                ch = wf.getnchannels()
                in_sr = wf.getframerate()
                sw = wf.getsampwidth()
                if sw != 2:
                    logger.warning(f"Skipping {p}: unsupported sample width {sw}")
                    continue
                raw = wf.readframes(wf.getnframes())
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if ch > 1 and len(arr) >= ch:
                arr = arr[: (len(arr) // ch) * ch].reshape(-1, ch).mean(axis=1)
            if in_sr != sample_rate and len(arr) > 1:
                out_len = max(1, int(round(len(arr) * sample_rate / float(in_sr))))
                arr = np.interp(
                    np.linspace(0.0, len(arr) - 1, out_len, dtype=np.float32),
                    np.arange(len(arr), dtype=np.float32),
                    arr,
                )
            i0 = max(0, int(round(start_s * sample_rate)))
            if i0 < target_samples:
                seg = arr[: target_samples - i0]
                buf[i0 : i0 + len(seg)] += seg
        except Exception as e:
            logger.warning(f"Failed to mix audio track {p}: {e}")

    pcm = np.clip(buf, -32767.0, 32767.0).astype(np.int16)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return output_wav_path


def concatenate_wav_tracks(
    audio_paths: List[Path],
    total_duration_s: float,
    output_wav_path: Path,
    sample_rate: int = 24000,
) -> Optional[Path]:
    """Combines per-panel WAV audio files into a continuous timeline audio track."""
    valid_paths = [Path(p) for p in audio_paths if p and Path(p).exists()]
    if not valid_paths:
        return None

    events: List[Tuple[float, Path]] = []
    cursor = 0.0
    for p in valid_paths:
        try:
            with wave.open(str(p), "rb") as wf:
                dur = wf.getnframes() / float(max(1, wf.getframerate()))
            events.append((cursor, p))
            cursor += dur + 0.25
        except Exception as e:
            logger.warning(f"Failed to read audio track {p}: {e}")

    return mix_timed_tracks(events, total_duration_s, output_wav_path, sample_rate=sample_rate)


def assemble_timeline(
    clips: List[List[Image.Image]],
    output_base_path: Path,
    fps: int = 24,
    transition_frames: int = 8,
    audio_paths: Optional[List[Path]] = None,
    audio_events: Optional[List[Tuple[float, Path]]] = None,
) -> Dict[str, Any]:
    """Assembles multiple panel clips into a single continuous video with transitions and audio."""
    if not clips:
        raise ValueError("No clips provided to assemble")

    full_timeline = list(clips[0])
    for next_clip in clips[1:]:
        if transition_frames > 0 and len(full_timeline) >= transition_frames and len(next_clip) >= transition_frames:
            tail_a = full_timeline[-transition_frames:]
            del full_timeline[-transition_frames:]
            head_b = next_clip[:transition_frames]
            for i in range(transition_frames):
                alpha = float(i + 1) / float(transition_frames + 1)
                full_timeline.append(Image.blend(tail_a[i], head_b[i], alpha))
            full_timeline.extend(next_clip[transition_frames:])
        else:
            full_timeline.extend(next_clip)

    output_base_path = Path(output_base_path)
    output_base_path.parent.mkdir(parents=True, exist_ok=True)

    avi_path = output_base_path.with_suffix(".avi")
    webp_path = output_base_path.with_suffix(".webp")
    mp4_path = output_base_path.with_suffix(".mp4")

    # 1. Write standards-compliant AVI
    encode_mjpeg_avi(full_timeline, avi_path, fps=fps)

    # 2. Write animated WebP with exact frame duration matching step/fps (P1-13)
    step = max(1, fps // 8)
    sample_frames = full_timeline[::step]
    frame_duration_ms = max(10, int(round(1000.0 * step / max(1, fps))))
    sample_frames[0].save(
        webp_path,
        save_all=True,
        append_images=sample_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        quality=75,
        method=0,
    )

    total_duration_s = len(full_timeline) / float(max(1, fps))

    # 3. Prepare mixed audio track (anchored to panel timestamps when audio_events is provided)
    master_wav: Optional[Path] = None
    wav_out = output_base_path.with_name(output_base_path.stem + "_master_audio.wav")
    if audio_events:
        master_wav = mix_timed_tracks(
            events=audio_events,
            total_duration_s=total_duration_s,
            output_wav_path=wav_out,
        )
    elif audio_paths:
        master_wav = concatenate_wav_tracks(
            audio_paths=audio_paths,
            total_duration_s=total_duration_s,
            output_wav_path=wav_out,
        )

    # 4. If ffmpeg is in PATH, convert to H.264 MP4 (with AAC audio stream if present)
    ffmpeg_exe = shutil.which("ffmpeg")
    has_mp4 = False
    if ffmpeg_exe:
        cmd = [ffmpeg_exe, "-y", "-loglevel", "error", "-i", str(avi_path)]
        if master_wav and master_wav.exists():
            cmd.extend(
                [
                    "-i",
                    str(master_wav),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                ]
            )
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"])
        cmd.extend(["-movflags", "+faststart", str(mp4_path)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and mp4_path.is_file():
            has_mp4 = True
            logger.info(f"Generated H.264 MP4 via FFmpeg: {mp4_path}")
        else:
            logger.error(f"FFmpeg MP4 conversion failed: {proc.stderr[-2000:]}")
            mp4_path.unlink(missing_ok=True)

    return {
        "total_frames": len(full_timeline),
        "duration_s": total_duration_s,
        "avi_path": str(avi_path.as_posix()),
        "webp_path": str(webp_path.as_posix()),
        "mp4_path": str(mp4_path.as_posix()) if has_mp4 else "",
    }


def _encode_mp4_via_ffmpeg_filtergraph(
    ffmpeg_exe: str,
    panel_clip_specs: List[Dict[str, Any]],
    mp4_path: Path,
    fps: int,
    target_res: Tuple[int, int],
    transition_frames: int,
    master_wav: Optional[Path],
) -> bool:
    """Streams panel clip files through an FFmpeg filter_complex (`overlay` + `xfade`) into H.264 MP4."""
    if not panel_clip_specs or len(panel_clip_specs) > 24:
        return False

    w, h = target_res
    xfade_dur = transition_frames / float(max(1, fps)) if transition_frames > 0 else 0.0
    temp_overlays: List[Path] = []

    try:
        cmd: List[str] = [ffmpeg_exe, "-y", "-loglevel", "error"]
        for spec in panel_clip_specs:
            cmd.extend(["-i", str(spec["clip_path"])])

        overlay_inputs: Dict[int, Tuple[int, int, int]] = {}
        next_input_idx = len(panel_clip_specs)
        for idx, spec in enumerate(panel_clip_specs):
            dialogue = spec.get("dialogue") or []
            if dialogue and not spec.get("subtitle_burned", False):
                ov = create_subtitle_overlay(target_res, dialogue)
                if ov is not None:
                    ov_img, bx, by = ov
                    ov_path = mp4_path.parent / f".tmp_sub_overlay_{idx:03d}.png"
                    ov_img.save(ov_path, format="PNG")
                    temp_overlays.append(ov_path)
                    cmd.extend(["-i", str(ov_path)])
                    overlay_inputs[idx] = (next_input_idx, bx, by)
                    next_input_idx += 1

        audio_input_idx: Optional[int] = None
        if master_wav and master_wav.is_file():
            cmd.extend(["-i", str(master_wav)])
            audio_input_idx = next_input_idx

        filter_parts: List[str] = []
        stage_labels: List[str] = []
        for idx, spec in enumerate(panel_clip_specs):
            base_lbl = f"v{idx}"
            filter_parts.append(
                f"[{idx}:v]scale={w}:{h}:flags=bilinear,fps={fps},format=yuv420p,settb=AVTB[{base_lbl}]"
            )
            if idx in overlay_inputs:
                ov_in, bx, by = overlay_inputs[idx]
                sub_lbl = f"vsub{idx}"
                filter_parts.append(
                    f"[{base_lbl}][{ov_in}:v]overlay={bx}:{by}:format=auto,format=yuv420p[{sub_lbl}]"
                )
                stage_labels.append(sub_lbl)
            else:
                stage_labels.append(base_lbl)

        if len(stage_labels) == 1:
            final_v_label = stage_labels[0]
        else:
            curr_label = stage_labels[0]
            first_frames = int(panel_clip_specs[0].get("frames") or count_video_frames(panel_clip_specs[0]["clip_path"]))
            cum_dur = first_frames / float(max(1, fps))
            for idx in range(1, len(stage_labels)):
                next_lbl = stage_labels[idx]
                out_lbl = f"vx{idx}"
                next_frames = int(
                    panel_clip_specs[idx].get("frames") or count_video_frames(panel_clip_specs[idx]["clip_path"])
                )
                next_dur = next_frames / float(max(1, fps))
                if xfade_dur > 0.0 and cum_dur > xfade_dur and next_dur > xfade_dur:
                    offset = max(0.0, cum_dur - xfade_dur)
                    filter_parts.append(
                        f"[{curr_label}][{next_lbl}]xfade=transition=fade:duration={xfade_dur:.4f}:offset={offset:.4f}[{out_lbl}]"
                    )
                    cum_dur = offset + next_dur
                else:
                    filter_parts.append(f"[{curr_label}][{next_lbl}]concat=n=2:v=1:a=0[{out_lbl}]")
                    cum_dur += next_dur
                curr_label = out_lbl
            final_v_label = curr_label

        cmd.extend(["-filter_complex", ";".join(filter_parts), "-map", f"[{final_v_label}]"])
        if audio_input_idx is not None:
            cmd.extend(["-map", f"{audio_input_idx}:a", "-c:a", "aac", "-shortest"])
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ]
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and mp4_path.is_file():
            logger.info(f"Generated H.264 MP4 via FFmpeg filter_complex: {mp4_path}")
            return True
        logger.debug(f"FFmpeg filter_complex MP4 fallback triggered: {proc.stderr[-1000:]}")
        mp4_path.unlink(missing_ok=True)
        return False
    finally:
        for tmp_p in temp_overlays:
            tmp_p.unlink(missing_ok=True)


def assemble_clips_streaming(
    panel_clip_specs: List[Dict[str, Any]],
    output_base_path: Path,
    fps: int = 24,
    target_res: Tuple[int, int] = (720, 1280),
    transition_frames: int = 8,
    audio_paths: Optional[List[Path]] = None,
    audio_events: Optional[List[Tuple[float, Path]]] = None,
) -> Dict[str, Any]:
    """Streams on-disk panel clips into final AVI, WebP, and FFmpeg MP4 deliverables with bounded RAM.

    Memory invariant (feat-006 / optim.md P1-01 & P1-02):
    Holds at most `2 * transition_frames` full-resolution frames in memory at any point during
    crossfade streaming, plus subsampled preview frames for animated WebP output.
    """
    if not panel_clip_specs:
        raise ValueError("No panel clip specs provided to assemble")

    output_base_path = Path(output_base_path)
    output_base_path.parent.mkdir(parents=True, exist_ok=True)

    avi_path = output_base_path.with_suffix(".avi")
    webp_path = output_base_path.with_suffix(".webp")
    mp4_path = output_base_path.with_suffix(".mp4")

    step = max(1, fps // 8)
    sample_frames: List[Image.Image] = []
    prev_tail: List[Image.Image] = []

    with StreamingMjpegAviWriter(avi_path, fps=fps) as avi_writer:

        def _emit_frame(frame: Image.Image) -> None:
            idx = avi_writer.num_frames
            avi_writer.write_frame(frame)
            if idx % step == 0:
                sample_frames.append(frame.copy())

        for spec in panel_clip_specs:
            clip_path = Path(spec["clip_path"])
            dialogue = spec.get("dialogue") or []
            subtitle_burned = bool(spec.get("subtitle_burned", False))
            clip_frames_total = int(spec.get("frames") or count_video_frames(clip_path))

            ov = (
                create_subtitle_overlay(target_res, dialogue)
                if (dialogue and not subtitle_burned)
                else None
            )

            can_blend_in = (
                transition_frames > 0
                and len(prev_tail) == transition_frames
                and clip_frames_total >= transition_frames
            )
            if not can_blend_in and prev_tail:
                for tail_f in prev_tail:
                    _emit_frame(tail_f)
                prev_tail.clear()

            hold_tail = transition_frames > 0 and clip_frames_total >= transition_frames
            curr_tail: deque[Image.Image] = deque()

            for frame_i, frame in enumerate(iter_video_frames(clip_path, target_res=target_res)):
                if ov is not None:
                    ov_img, bx, by = ov
                    frame.paste(ov_img, (bx, by), ov_img)
                if can_blend_in and frame_i < transition_frames:
                    alpha = float(frame_i + 1) / float(transition_frames + 1)
                    frame = Image.blend(prev_tail[frame_i], frame, alpha)

                if hold_tail:
                    if len(curr_tail) == transition_frames:
                        _emit_frame(curr_tail.popleft())
                    curr_tail.append(frame)
                else:
                    _emit_frame(frame)

            prev_tail = list(curr_tail)

        for tail_f in prev_tail:
            _emit_frame(tail_f)
        prev_tail.clear()
        total_frames = avi_writer.num_frames

    if not sample_frames:
        raise ValueError("No frames decoded from panel clip specs")

    frame_duration_ms = max(10, int(round(1000.0 * step / max(1, fps))))
    sample_frames[0].save(
        webp_path,
        save_all=True,
        append_images=sample_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        quality=75,
        method=0,
    )
    sample_frames.clear()

    total_duration_s = total_frames / float(max(1, fps))

    master_wav: Optional[Path] = None
    wav_out = output_base_path.with_name(output_base_path.stem + "_master_audio.wav")
    if audio_events:
        master_wav = mix_timed_tracks(
            events=audio_events,
            total_duration_s=total_duration_s,
            output_wav_path=wav_out,
        )
    elif audio_paths:
        master_wav = concatenate_wav_tracks(
            audio_paths=audio_paths,
            total_duration_s=total_duration_s,
            output_wav_path=wav_out,
        )

    ffmpeg_exe = shutil.which("ffmpeg")
    has_mp4 = False
    if ffmpeg_exe:
        has_mp4 = _encode_mp4_via_ffmpeg_filtergraph(
            ffmpeg_exe=ffmpeg_exe,
            panel_clip_specs=panel_clip_specs,
            mp4_path=mp4_path,
            fps=fps,
            target_res=target_res,
            transition_frames=transition_frames,
            master_wav=master_wav,
        )
        if not has_mp4:
            cmd = [ffmpeg_exe, "-y", "-loglevel", "error", "-i", str(avi_path)]
            if master_wav and master_wav.exists():
                cmd.extend(
                    [
                        "-i",
                        str(master_wav),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-shortest",
                    ]
                )
            else:
                cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"])
            cmd.extend(["-movflags", "+faststart", str(mp4_path)])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and mp4_path.is_file():
                has_mp4 = True
                logger.info(f"Generated H.264 MP4 via FFmpeg: {mp4_path}")
            else:
                logger.error(f"FFmpeg MP4 conversion failed: {proc.stderr[-2000:]}")
                mp4_path.unlink(missing_ok=True)

    return {
        "total_frames": total_frames,
        "duration_s": total_duration_s,
        "avi_path": str(avi_path.as_posix()),
        "webp_path": str(webp_path.as_posix()),
        "mp4_path": str(mp4_path.as_posix()) if has_mp4 else "",
    }
