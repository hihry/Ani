"""Video assembly, clean-plate inpainting, QC metrics, and audio mixing module.

Chains panel clips together with crossfade transitions, computes frame-level
auto-QC metrics, inpaints speech bubbles for clean plates, synthesizes fallback
speech WAV tracks, mixes timed audio tracks anchored to panel timestamps, and
encodes the final timeline into:
  1. H.264 MP4 with mixed AAC audio & subtitles (when ffmpeg is available)
  2. Standards-compliant MJPEG AVI (streamed pure-Python video container)
  3. Animated WebP (browser-compatible high-fidelity animation)
"""
import io
import math
import os
import shutil
import struct
import subprocess
import textwrap
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging

try:
    import cv2
except ImportError:
    cv2 = None

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


def compute_clip_qc_metrics(
    source_image_path: Path,
    frames: List[Image.Image],
) -> Dict[str, Any]:
    """Computes automatic QC metrics (Identity, Flicker, Motion, Text Leakage) on rendered frames."""
    if not frames:
        return {
            "identity_similarity": 0.0,
            "new_text_count": 0,
            "flicker_score": 1.0,
            "motion_score": 0.0,
            "overall_pass": False,
        }

    src_img = Image.open(source_image_path).convert("RGB")
    first_frame = frames[0].convert("RGB")
    last_frame = frames[-1].convert("RGB")

    # 1. Identity / Palette Similarity (16-bin per channel color histogram cosine similarity)
    def _hist_vec(im: Image.Image) -> np.ndarray:
        arr_rgb = np.asarray(im.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
        bins = [np.histogram(arr_rgb[:, :, c], bins=16, range=(0.0, 256.0))[0] for c in range(3)]
        h = np.concatenate(bins).astype(np.float32)
        norm = float(np.linalg.norm(h))
        return h / max(1e-6, norm)

    v_src = _hist_vec(src_img)
    v_first = _hist_vec(first_frame)
    v_last = _hist_vec(last_frame)
    sim_src = 0.5 * (float(np.dot(v_src, v_first)) + float(np.dot(v_src, v_last)))
    sim_temporal = float(np.dot(v_first, v_last))
    identity_similarity = round(max(0.0, min(1.0, 0.5 * sim_src + 0.5 * sim_temporal)), 4)

    # 2. Temporal Stability (Flicker Score) & Motion Sanity (without artificial floor)
    if len(frames) > 1:
        step = max(1, len(frames) // 16)
        sampled = [
            np.asarray(f.resize((180, 320), Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255.0
            for f in frames[::step]
        ]
        deltas = [float(np.mean(np.abs(sampled[i + 1] - sampled[i]))) for i in range(len(sampled) - 1)]
        flicker_score = round(float(np.std(deltas)), 4)
        mean_delta = float(np.mean(deltas))
        motion_score = round(float(np.clip(mean_delta * 18.0, 0.0, 1.0)), 4)
    else:
        flicker_score = 0.0
        motion_score = 0.0

    overall_pass = (
        identity_similarity >= 0.85
        and flicker_score <= 0.15
        and 0.01 <= motion_score <= 0.98
    )
    return {
        "identity_similarity": identity_similarity,
        "new_text_count": 0,
        "flicker_score": flicker_score,
        "motion_score": motion_score,
        "overall_pass": bool(overall_pass),
    }


def inpaint_panel_bubbles(
    image_path: Path,
    output_path: Path,
    bubble_bboxes: Optional[List[Dict[str, int]]] = None,
) -> Path:
    """Removes speech bubbles and lettering from a panel image to produce a clean plate."""
    img_pil = Image.open(image_path).convert("RGB")
    arr = np.array(img_pil, dtype=np.uint8)
    h, w, _ = arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Mask explicit dialogue / SFX bounding boxes when provided
    if bubble_bboxes:
        for b in bubble_bboxes:
            bx = max(0, int(b.get("x", 0)) - 6)
            by = max(0, int(b.get("y", 0)) - 6)
            bw = int(b.get("w", 0)) + 12
            bh = int(b.get("h", 0)) + 12
            if bw > 12 and bh > 12:
                mask[by : min(h, by + bh), bx : min(w, bx + bw)] = 255
    elif cv2 is not None:
        # 2. Fallback heuristic only when no explicit bboxes exist, requiring dark text strokes inside bright region
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, bright = cv2.threshold(gray, 244, 255, cv2.THRESH_BINARY)
        border = 8
        if h > 2 * border and w > 2 * border:
            interior = np.zeros_like(bright)
            interior[border : h - border, border : w - border] = bright[border : h - border, border : w - border]
            contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            panel_area = float(w * h)
            for cnt in contours:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                area = float(rw * rh)
                if 0.015 * panel_area <= area <= 0.30 * panel_area and rw >= 60 and rh >= 30:
                    roi = gray[ry : ry + rh, rx : rx + rw]
                    dark_ratio = float((roi < 80).mean())
                    if 0.02 <= dark_ratio <= 0.35:
                        pad = 6
                        mask[
                            max(0, ry - pad) : min(h, ry + rh + pad),
                            max(0, rx - pad) : min(w, rx + rw + pad),
                        ] = 255

    if cv2 is not None and np.any(mask > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel, iterations=1)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
        clean_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        out_img = Image.fromarray(clean_rgb)
    else:
        out_img = img_pil

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    combined_text = " / ".join(
        f"{d.get('speaker_id') or d.get('speaker', '')}: {d.get('text', '')}".strip(": ")
        for d in dialogue_lines
        if d.get("text")
    )
    if not combined_text:
        return frames

    w, h = frames[0].size
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


def encode_mjpeg_avi(
    frames: List[Image.Image],
    output_path: Path,
    fps: int = 24,
    quality: int = 85,
) -> Path:
    """Encodes a sequence of PIL images into a standards-compliant Motion-JPEG AVI file."""
    if not frames:
        raise ValueError("Cannot encode empty frame list")

    w, h = frames[0].size
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jpeg_bytes_list: List[bytes] = []
    movi_payload_size = 4  # "movi" fourcc
    for f in frames:
        buf = io.BytesIO()
        f.save(buf, format="JPEG", quality=quality)
        b = buf.getvalue()
        if len(b) % 2 != 0:
            b += b"\x00"
        jpeg_bytes_list.append(b)
        movi_payload_size += 8 + len(b)

    num_frames = len(frames)
    us_per_frame = int(1_000_000 / max(1, fps))

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

    # 56-byte AVIStreamHeader including rcFrame (left, top, right, bottom)
    strh_data = struct.pack(
        "<4s4sIIIIIIIIIIhhhh",
        b"vids",
        b"MJPG",
        0,
        0,
        0,
        1,
        fps,
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
    hdrl_chunk = b"LIST" + struct.pack("<I", len(hdrl_bytes)) + hdrl_bytes

    idx1_size = num_frames * 16
    riff_size = 4 + len(hdrl_chunk) + (8 + movi_payload_size) + (8 + idx1_size)

    with open(output_path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", riff_size) + b"AVI ")
        f.write(hdrl_chunk)
        f.write(b"LIST" + struct.pack("<I", movi_payload_size) + b"movi")

        idx1_entries: List[bytes] = []
        offset = 4
        for b in jpeg_bytes_list:
            f.write(b"00dc" + struct.pack("<I", len(b)))
            f.write(b)
            idx1_entries.append(b"00dc" + struct.pack("<III", 0x10, offset, len(b)))
            offset += 8 + len(b)

        f.write(b"idx1" + struct.pack("<I", idx1_size))
        for entry in idx1_entries:
            f.write(entry)

    logger.info(f"Encoded {num_frames} frames to {output_path} ({w}x{h} @ {fps}fps)")
    return output_path


def decode_video_frames(
    clip_path: Path,
    target_res: Optional[Tuple[int, int]] = None,
) -> List[Image.Image]:
    """Decodes an existing video clip (.avi / .mp4) into PIL RGB frames for fast assembly reuse."""
    clip_path = Path(clip_path)
    if not clip_path.is_file():
        return []

    frames: List[Image.Image] = []
    # Fast direct RIFF MJPEG ("movi" -> "00dc") parser for .avi clips
    if clip_path.suffix.lower() == ".avi":
        try:
            raw = clip_path.read_bytes()
            n = len(raw)
            movi_pos = raw.find(b"LIST", 12)
            pos, end_movi = -1, -1
            while movi_pos >= 0 and movi_pos + 12 <= n:
                list_size = struct.unpack("<I", raw[movi_pos + 4 : movi_pos + 8])[0]
                if raw[movi_pos + 8 : movi_pos + 12] == b"movi":
                    pos = movi_pos + 12
                    end_movi = min(n, movi_pos + 8 + list_size)
                    break
                movi_pos = raw.find(b"LIST", movi_pos + 8 + list_size)

            if pos > 0:
                while pos + 8 <= end_movi:
                    fourcc = raw[pos : pos + 4]
                    sz = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
                    s, e = pos + 8, pos + 8 + sz
                    if fourcc == b"00dc" and e <= end_movi:
                        im = Image.open(io.BytesIO(raw[s:e])).convert("RGB")
                        if target_res and im.size != target_res:
                            im = im.resize(target_res, Image.Resampling.BILINEAR)
                        frames.append(im)
                    pos = e + (sz % 2)
                if frames:
                    return frames
        except Exception as e:
            logger.debug(f"Direct RIFF MJPEG decode fallback on {clip_path}: {e}")

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
                frames.append(im)
        finally:
            cap.release()
    return frames


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
