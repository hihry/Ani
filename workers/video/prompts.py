"""Prompt synthesizer for Phase 3 I2V video generation (`workers/video/prompts.py`).

Builds concise, motion-focused positive prompts (< 60 tokens) from Phase 2 panel
metadata + understanding artifacts without re-describing static character clothing,
and pairs them with the model family's negative prompt.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

_CAMERA_HINTS = {
    "zoom_in": "slow cinematic push-in camera movement",
    "zoom_out": "smooth pull-back camera movement",
    "pan_up": "gentle vertical tilt-up camera motion",
    "pan_down": "gentle vertical tilt-down camera motion",
    "pan_left": "smooth horizontal pan left",
    "pan_right": "smooth horizontal pan right",
    "shake": "dynamic impact camera vibration and energy ripples",
    "static": "locked camera with subtle breathing and hair movement",
}

_STYLE_LOCK = "2D anime style, subtle motion, consistent character line art, stable background"


def build_prompt(
    panel: Dict[str, Any],
    understanding_panel: Optional[Dict[str, Any]] = None,
    spec: Optional[Any] = None,
) -> Tuple[str, str]:
    u = understanding_panel or {}
    action = (
        panel.get("action_prompt")
        or u.get("action_description")
        or panel.get("scene_description")
        or u.get("scene_description")
        or "character subtle breathing, hair and cloth swaying gently in the wind"
    ).strip()

    motion_type = str(
        panel.get("motion_type")
        or u.get("camera_motion")
        or "zoom_in"
    ).lower()
    camera_clause = _CAMERA_HINTS.get(motion_type, "smooth cinematic camera movement")

    emotion = str(panel.get("emotion") or u.get("emotion") or "").strip().lower()
    mood_clause = f"{emotion} expression and atmosphere" if emotion and emotion != "neutral" else "dramatic lighting"

    clauses = [action.rstrip("."), camera_clause, mood_clause, _STYLE_LOCK]
    positive = ", ".join(c for c in clauses if c)
    # Keep under ~60 words
    words = positive.split()
    if len(words) > 55:
        positive = " ".join(words[:55]).rstrip(",") + ", " + _STYLE_LOCK

    default_neg = (
        "text, speech bubble, watermark, subtitles, deformed face, extra limbs, "
        "3d render, photorealistic, blurry, low quality, flicker"
    )
    negative = getattr(spec, "negative_prompt", None) or default_neg
    return positive, negative
