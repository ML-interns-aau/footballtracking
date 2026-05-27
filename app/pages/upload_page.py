import os
import streamlit as st
from app.config import RAW_DIR, VIDEO_EXTENSIONS
from app.utils import page_header, render_pipeline, nav_button, metric_card
def render():
    page_header("Upload Video", "Provide a match video to start the automated pipeline.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(f"""
        <div style="font-size:0.82rem; font-weight:600; color:#f5f5f5; margin-bottom:0.5rem;">
            Upload a video file
        </div>
        <div style="font-size:0.72rem; color:#6b6b78; margin-bottom:0.8rem;">
            Accepted: {', '.join(f'.{e}' for e in VIDEO_EXTENSIONS)}
        </div>
        <div style="font-size:0.82rem; font-weight:600; color:#f5f5f5; margin-bottom:0.5rem;">
            Or select an existing video
        </div>
        <div style="font-size:0.82rem; font-weight:600; color:#f5f5f5; margin-bottom:0.8rem;">
            Preview
        </div>
            <div style="background:#0d0d12; border:1px dashed rgba(255,255,255,0.06);
                        border-radius:12px; padding:3rem 1rem; text-align:center;
                        color:
                Upload or select a video to preview
            </div>