"""Streamlit app for Manhwa-to-Anime pipeline editing and monitoring (CP1 & CP2)."""
import json
import logging
from pathlib import Path
import streamlit as st

from m2a.config import load_config
from m2a.graph import PipelineGraph
from m2a.state import PanelJSON, PipelineStage, RunState

logger = logging.getLogger(__name__)
RUNS_DIR = Path("./runs")

st.set_page_config(page_title="Manhwa-to-Anime Editor", layout="wide")
st.title("Manhwa-to-Anime (M2A) Studio & Checkpoint Editor")

# Discover existing runs
existing_runs = sorted(
    [d.name for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "state.json").exists()],
    reverse=True,
) if RUNS_DIR.exists() else []

selected_run_id = st.sidebar.selectbox("Select Run ID", options=[""] + existing_runs)
compute_tier = st.sidebar.selectbox("Compute Tier", ["t0", "t1", "t2"], index=1)

if selected_run_id:
    run_dir = RUNS_DIR / selected_run_id
    state = RunState.load(run_dir)
    st.sidebar.markdown(f"**Current Stage:** `{state.current_stage.value}`")
    st.sidebar.markdown(f"**Panels:** `{len(state.panels)}`")

    st.divider()
    st.header("Checkpoint 1: Panel & Dialogue Review")
    with st.expander("View and Edit Panel JSON Contracts", expanded=(state.current_stage == PipelineStage.CHECKPOINT_1)):
        col_img, col_json = st.columns([1, 1])
        with col_img:
            stitched_file = run_dir / "stitched.png"
            if stitched_file.exists():
                st.image(str(stitched_file), caption="Stitched Strip", use_container_width=True)
        with col_json:
            panels_str = json.dumps([p.model_dump() for p in state.panels], indent=2)
            edited_json = st.text_area("Panels JSON", value=panels_str, height=420)
            if st.button("Save CP1 Edits & Resume"):
                raw_list = json.loads(edited_json)
                state.panels = [PanelJSON.model_validate(item) for item in raw_list]
                state.save(run_dir)
                cfg = load_config(tier=state.config.compute_tier)
                graph = PipelineGraph(config=cfg, run_dir=run_dir)
                state = graph.resume(state)
                st.success(f"CP1 saved! Advanced to {state.current_stage.value}")

    st.divider()
    st.header("Checkpoint 2: Shot Routing Review")
    with st.expander("Approve or Override Panel Animation Routes", expanded=(state.current_stage == PipelineStage.CHECKPOINT_2)):
        for idx, p in enumerate(state.panels):
            new_route = st.selectbox(
                f"Panel {p.panel_id} ({p.shot}, intensity={p.motion_intensity})",
                options=["kenburns", "parallax", "i2v"],
                index=["kenburns", "parallax", "i2v"].index(p.route),
                key=f"route_{p.panel_id}",
            )
            state.panels[idx].route = new_route  # type: ignore
        if st.button("Approve Routes & Render Video"):
            state.save(run_dir)
            cfg = load_config(tier=state.config.compute_tier)
            graph = PipelineGraph(config=cfg, run_dir=run_dir)
            state = graph.resume(state)
            st.success(f"Routes approved! Pipeline stage: {state.current_stage.value}")
