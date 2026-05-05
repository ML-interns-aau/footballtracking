# app/pages/upload_page.py
"""Upload Page."""

import os
import streamlit as st
from app.config import RAW_DIR, VIDEO_EXTENSIONS
from app.utils import page_header, render_pipeline, nav_button, metric_card


def render():
    page_header("Upload Video", "Provide a match video to start the automated pipeline.")
    # render_pipeline(active=0)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(f"""
        <div style="font-size:0.82rem; font-weight:600; color:#f5f5f5; margin-bottom:0.5rem;">
            Upload a video file
        </div>
        <div style="font-size:0.72rem; color:#6b6b78; margin-bottom:0.8rem;">
            Accepted: {', '.join(f'.{e}' for e in VIDEO_EXTENSIONS)}
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Choose video", type=VIDEO_EXTENSIONS, label_visibility="collapsed",
        )

        if uploaded is not None:
            save_path = os.path.join(RAW_DIR, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())

            st.session_state.uploaded_video = save_path
            st.session_state.uploaded_video_name = uploaded.name
            st.session_state.pop("processed_video", None)
            st.session_state.pop("analysis_done", None)
            st.session_state.pop("tracked_video", None)

            st.success(f"Saved: data/raw/{uploaded.name}")

            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(metric_card("Size", f"{size_mb:.1f} MB"), unsafe_allow_html=True)
            with c2:
                ext = uploaded.name.rsplit(".", 1)[-1].upper()
                st.markdown(metric_card("Format", ext), unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.82rem; font-weight:600; color:#f5f5f5; margin-bottom:0.5rem;">
            Or select an existing video
        </div>
        """, unsafe_allow_html=True)

        existing = sorted(
            f for f in os.listdir(RAW_DIR)
            if f.rsplit(".", 1)[-1].lower() in VIDEO_EXTENSIONS
        )
        if existing:
            sel = st.selectbox("Videos in data/raw/", ["— Select —"] + existing,
                               label_visibility="collapsed")
            if sel != "— Select —":
                path = os.path.join(RAW_DIR, sel)
                st.session_state.uploaded_video = path
                st.session_state.uploaded_video_name = sel
        else:
            st.caption("No videos in data/raw/ yet.")

    with right:
        st.markdown(f"""
        <div style="font-size:0.82rem; font-weight:600; color:#f5f5f5; margin-bottom:0.8rem;">
            Preview
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.get("uploaded_video"):
            st.video(st.session_state.uploaded_video)
        else:
            st.markdown(f"""
            <div style="background:#0d0d12; border:1px dashed rgba(255,255,255,0.06);
                        border-radius:12px; padding:3rem 1rem; text-align:center;
                        color:#6b6b78; font-size:0.82rem;">
                Upload or select a video to preview
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    _, right_col = st.columns([3, 1])
    with right_col:
        if st.session_state.get("uploaded_video"):
            nav_button("Next: Preprocess →", "Preprocess")
        else:
            st.button("Next: Preprocess →", disabled=True, width='stretch')
