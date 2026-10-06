"""Assembly node for the Manhwa-to-Anime pipeline."""

import logging
import time
from pathlib import Path
import subprocess

# TODO: Import actual state types
# from m2a.state import RunState, PipelineConfig

logger = logging.getLogger(__name__)

def assemble(state: 'RunState', config: 'PipelineConfig') -> 'RunState':
    """Build final video timeline, mix audio, add subtitles, and assemble with FFmpeg.
    
    Args:
        state: Current run state.
        config: Pipeline configuration.
        
    Returns:
        Updated run state.
    """
    start_time = time.time()
    logger.info("Starting assemble node...")
    
    try:
        run_folder = state.workspace_dir / "run"
        final_output = run_folder / "final_output.mp4"
        list_file_path = run_folder / "concat_list.txt"
        srt_file_path = run_folder / "subtitles.srt"
        
        # 1. Generate SRT and build concat list
        srt_content = ""
        srt_index = 1
        current_time = 0.0
        
        with open(list_file_path, "w") as f_list:
            for panel in state.panels:
                clip_path = panel.get("final_clip")
                if not clip_path:
                    continue
                    
                # Mock duration handling
                # TODO: Retrieve actual duration from video file
                duration = 3.0
                
                f_list.write(f"file '{clip_path}'\n")
                
                # Subtitles
                for track in panel.get("audio_tracks", []):
                    # Mock timing based on current sequence
                    start_sec = current_time
                    end_sec = current_time + duration
                    
                    def format_time(seconds):
                        h = int(seconds // 3600)
                        m = int((seconds % 3600) // 60)
                        s = int(seconds % 60)
                        ms = int((seconds - int(seconds)) * 1000)
                        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                        
                    srt_content += f"{srt_index}\n"
                    srt_content += f"{format_time(start_sec)} --> {format_time(end_sec)}\n"
                    srt_content += f"{track['text']}\n\n"
                    srt_index += 1
                    
                current_time += duration
                
        with open(srt_file_path, "w", encoding="utf-8") as f_srt:
            f_srt.write(srt_content)
            
        # 2. Assemble with FFmpeg
        # TODO: Add complex filter for crossfade transitions and audio mixing
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file_path),
            "-vf", f"subtitles={srt_file_path}",
            "-c:v", "libx264",
            "-c:a", "aac",
            str(final_output)
        ]
        
        logger.info(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
        # Uncomment in production when dependencies and inputs exist
        # subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        state.final_video_path = str(final_output)
        
    except Exception as e:
        logger.exception("Error in assemble node")
        state.errors.append(str(e))
        
    logger.info(f"Assemble node completed in {time.time() - start_time:.2f}s")
    return state
