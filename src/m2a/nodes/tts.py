"""Text-to-Speech node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from m2a.config import PipelineConfig
from m2a.state import AudioTrack, RunState
from workers.assembler import synthesize_speech_wav

logger = logging.getLogger(__name__)


def tts_node(state: RunState, config: PipelineConfig) -> RunState:
    """Generate speech WAV files for dialogue lines and synchronize panel durations."""
    start_time = time.time()
    logger.info("Starting tts node...")

    try:
        run_dir = Path(config.general.run_dir)
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        tracks: list[AudioTrack] = []
        for panel in state.panels:
            panel_audio_dur = 0.0
            for idx, line in enumerate(panel.dialogue, start=1):
                voice_id = line.speaker_id
                if state.character_bible and line.speaker_id in state.character_bible.characters:
                    voice_id = state.character_bible.characters[line.speaker_id].voice.voice_id

                audio_path = audio_dir / f"{panel.panel_id}_{idx:02d}_{line.speaker_id}.wav"
                dur = synthesize_speech_wav(
                    text=line.text,
                    output_path=audio_path,
                    voice_id=voice_id,
                    emotion=line.emotion,
                    language=line.lang,
                )
                panel_audio_dur += dur + 0.25
                tracks.append(
                    AudioTrack(
                        panel_id=panel.panel_id,
                        speaker_id=line.speaker_id,
                        audio_path=str(audio_path.as_posix()),
                        duration_s=dur,
                        emotion=line.emotion,
                    )
                )

            if panel_audio_dur > 0:
                panel.duration_s = round(min(10.0, max(panel.duration_s, panel_audio_dur + 0.3)), 2)

        state.audio_tracks = tracks
        state.cost_ledger.add_model_cost(config.audio.tts_primary, time.time() - start_time)
    except Exception as e:
        logger.exception("Error in tts node")
        state.record_error(str(e))
        raise

    logger.info(f"TTS node completed in {time.time() - start_time:.2f}s")
    return state
