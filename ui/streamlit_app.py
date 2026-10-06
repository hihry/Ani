"""Streamlit app for Manhwa-to-Anime pipeline editing and monitoring."""
import streamlit as st
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Manhwa-to-Anime Editor", layout="wide")

st.title("Manhwa-to-Anime Editor")

# Upload and Config
with st.expander("Job Configuration", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader("Upload Manhwa Pages", accept_multiple_files=True, type=['png', 'jpg'])
    with col2:
        compute_tier = st.selectbox("Compute Tier", ["standard", "high-perf", "dev"])
        
    if st.button("Start Pipeline Run"):
        # TODO: Call API to submit job
        st.success("Pipeline started!")

st.divider()

# CP1: Panel Editor
st.header("Checkpoint 1: Panel Detection")
with st.expander("View and Edit Panels"):
    col_img, col_json = st.columns([1, 1])
    with col_img:
        st.image("https://via.placeholder.com/400x600?text=Panel+Image", caption="Detected Panels")
    with col_json:
        # TODO: Load actual panel JSON
        panel_json = st.text_area("Panel JSON", value='{"panels": []}', height=400)
        if st.button("Save Panels"):
            st.success("Panels updated!")

st.divider()

# CP2: Routing Decisions
st.header("Checkpoint 2: Route Editor")
with st.expander("View and Edit Routing Decisions"):
    # TODO: Load actual routing decisions
    st.write("Route for Panel 1: **Image-to-Video (Wan 2.2)**")
    new_route = st.selectbox("Change Route", ["Wan 2.2", "Ken Burns", "Parallax", "Hunyuan 1.5"])
    if st.button("Approve Routes"):
        st.success("Routes approved!")

st.divider()

# Progress and Results
st.header("Job Status")
progress_bar = st.progress(0)
# TODO: Poll API for status
st.write("Current Status: Idle")
st.text_area("Logs", value="Waiting for job...", height=200, disabled=True)

if st.button("Download Final MP4"):
    # TODO: Download from API
    st.info("Downloading file...")
