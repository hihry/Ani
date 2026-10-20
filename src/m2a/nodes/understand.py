"""Understanding node for the Manhwa-to-Anime pipeline."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from m2a.config import PipelineConfig
from m2a.state import (
    BBox,
    CharacterBible,
    CharacterEntry,
    CharacterRef,
    DialogueLine,
    RunState,
    SFX,
    VoiceProfile,
)
from workers.pipeline_cli import analyze_panel_semantics

logger = logging.getLogger(__name__)


def understand(state: RunState, config: PipelineConfig) -> RunState:
    """Analyze panels to extract dialogue, characters, shot framing, and motion prompts."""
    start_time = time.time()
    logger.info("Starting understand node...")

    try:
        run_dir = Path(config.general.run_dir)
        vlm_dir = run_dir / "understanding"
        vlm_dir.mkdir(parents=True, exist_ok=True)

        if state.character_bible is None:
            state.character_bible = CharacterBible()

        for idx, panel in enumerate(state.panels, start=1):
            crop_path = run_dir / "panels" / f"{panel.panel_id}.png"
            if crop_path.exists():
                sem = analyze_panel_semantics(
                    panel_image_path=crop_path,
                    panel_id=panel.panel_id,
                    language=state.config.source_language,
                    reading_order=idx,
                )
                shot_val = sem.get("shot", panel.shot)
                if shot_val in ("close_up", "medium", "wide", "splash"):
                    panel.shot = shot_val
                panel.action = sem.get("action", panel.action)
                panel.motion_prompt = sem.get("motion_prompt", panel.motion_prompt)
                panel.motion_intensity = float(sem.get("motion_intensity", panel.motion_intensity))

                panel.dialogue = [
                    DialogueLine(
                        speaker_id=d["speaker_id"],
                        text=d["text"],
                        lang=d.get("lang", "ko"),
                        emotion=d.get("emotion", "neutral"),
                        bbox=BBox(**d["bbox"]) if d.get("bbox") else None,
                        confidence=float(d.get("confidence", 0.95)),
                    )
                    for d in sem.get("dialogue", [])
                ]
                panel.sfx = [
                    SFX(
                        text=s["text"],
                        kind=s.get("kind", "impact"),
                        bbox=BBox(**s["bbox"]) if s.get("bbox") else None,
                    )
                    for s in sem.get("sfx", [])
                ]
                panel.characters = [
                    CharacterRef(
                        id=c["id"],
                        name=c["name"],
                        emotion=c.get("emotion", "neutral"),
                        pose=c.get("pose", "standing"),
                    )
                    for c in sem.get("characters", [])
                ]

                for c in panel.characters:
                    if c.id not in state.character_bible.characters:
                        state.character_bible.characters[c.id] = CharacterEntry(
                            name=c.name,
                            description=f"Character {c.name}",
                            reference_crops=[str(crop_path.as_posix())],
                            voice=VoiceProfile(voice_id=f"voice_{c.id}", style=c.emotion),
                            appearances=[panel.panel_id],
                        )
                    elif panel.panel_id not in state.character_bible.characters[c.id].appearances:
                        state.character_bible.characters[c.id].appearances.append(panel.panel_id)

            panel_json_file = vlm_dir / f"{panel.panel_id}.json"
            panel_json_file.write_text(panel.model_dump_json(indent=2), encoding="utf-8")

        bible_file = vlm_dir / "character_bible.json"
        bible_file.write_text(state.character_bible.model_dump_json(indent=2), encoding="utf-8")
        state.cost_ledger.add_model_cost(config.understanding.vlm, time.time() - start_time)
    except Exception as e:
        logger.exception("Error in understand node")
        state.record_error(str(e))
        raise

    logger.info(f"Understand node completed in {time.time() - start_time:.2f}s")
    return state
