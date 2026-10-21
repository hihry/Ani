"""Streamlit Blind Human Review UI for Phase 3 Video Bake-off (`ui/pages/2_Bakeoff_Review.py`).

Displays anonymized clips alongside source clean plates without revealing model names,
collects 5-axis rubric scores (1..5) + binary pass/fail, and appends ratings to
`human_ratings.jsonl`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from m2a.eval.rubric import RubricRating


def load_blind_manifest(bakeoff_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = Path(bakeoff_dir) / "blind_manifest.json"
    if not manifest_path.is_file():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_human_rating(bakeoff_dir: Path, rating: RubricRating) -> Path:
    out_path = Path(bakeoff_dir) / "human_ratings.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anon_clip_id": rating.anon_clip_id,
        "rater_id": rating.rater_id,
        "motion_plausibility": rating.motion_plausibility,
        "character_fidelity": rating.character_fidelity,
        "temporal_stability": rating.temporal_stability,
        "prompt_adherence": rating.prompt_adherence,
        "style_preservation": rating.style_preservation,
        "overall_pass": rating.overall_pass,
        "quality_score": rating.quality_score,
    }
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
    return out_path


def render_bakeoff_review_page() -> None:
    import streamlit as st

    st.set_page_config(page_title="M2A Phase 3 Blind Bake-off Review", layout="wide")
    st.title("Phase 3 Blind Video Bake-off Review")

    bakeoff_dir = Path(st.text_input("Bake-off Directory", value="runs/bakeoff_phase3"))
    rater_id = st.text_input("Reviewer ID", value="rater_1")
    items = load_blind_manifest(bakeoff_dir)
    if not items:
        st.info(f"No `blind_manifest.json` found in `{bakeoff_dir}`. Run `python -m m2a.eval.bakeoff` first.")
        return

    idx = st.number_input("Clip Index", min_value=0, max_value=len(items) - 1, value=0, step=1)
    item = items[int(idx)]
    st.subheader(f"Clip `{item['anon_clip_id']}` — Panel `{item['panel_id']}` ({item.get('category', '')})")

    with st.form(key=f"form_{item['anon_clip_id']}"):
        c1, c2, c3, c4, c5 = st.columns(5)
        motion = c1.slider("Motion Plausibility", 1, 5, 4)
        fidelity = c2.slider("Character Fidelity", 1, 5, 4)
        stability = c3.slider("Temporal Stability", 1, 5, 4)
        prompt_adh = c4.slider("Prompt Adherence", 1, 5, 4)
        style = c5.slider("Style Preservation", 1, 5, 4)
        overall = st.checkbox("Overall Pass (Shippable without fallback)", value=True)
        submitted = st.form_submit_button("Save Rating")
        if submitted:
            rating = RubricRating(
                anon_clip_id=item["anon_clip_id"],
                rater_id=rater_id,
                motion_plausibility=motion,
                character_fidelity=fidelity,
                temporal_stability=stability,
                prompt_adherence=prompt_adh,
                style_preservation=style,
                overall_pass=overall,
            )
            p = save_human_rating(bakeoff_dir, rating)
            st.success(f"Saved rating (Q={rating.quality_score:.3f}) to {p}")


if __name__ == "__main__":
    render_bakeoff_review_page()
