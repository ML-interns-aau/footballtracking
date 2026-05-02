# dashboard/pages/analysis_page.py
"""
Analysis Page — fully automated pipeline.

One button runs: Preprocess → Detection → Tracking → Post-processing.
No user input required after clicking "Run".
"""

import os
import json
import shutil
import cv2

import numpy as np
from types import SimpleNamespace

import streamlit as st

from dashboard.config import (
    MODEL_PATH, PROCESSED_DIR, ANNOTATIONS_DIR, INSIGHTS_DIR,
    PLAYER_CLASS_IDS, BALL_CLASS_ID, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMGSZ,
    DEFAULT_TARGET_FPS, DEFAULT_RESIZE_W,
)
from dashboard.utils import (
    page_header, render_pipeline, nav_button, metric_card,
    ACCENT, TEXT_PRIMARY, TEXT_MUTED, BG_CARD,
)


def _full_pipeline(raw_video: str, max_frames: int, progress, status):
    """Wrapper to call the repository's main.py pipeline."""
    try:
        import main as pipeline_main
    except Exception as e:
        raise RuntimeError(f"Unable to import main.py: {e}")

    from dashboard.config import PROJECT_ROOT, INSIGHTS_DIR
    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    args = SimpleNamespace(input=raw_video, output_dir=out_dir, max_frames=max_frames)

    status.text("Running analysis pipeline (this may take a while)...")
    pipeline_main.main(args)

    status.text("Finalizing insights and summaries...")
    try:
        import post_process_results
        import importlib
        importlib.reload(post_process_results)
        
        vname = st.session_state.get("uploaded_video_name", "video.mp4")
        post_process_results.post_process(out_dir, INSIGHTS_DIR, video_name=vname)
    except Exception as e:
        st.error(f"Post-processing failed: {e}")
        import traceback
        st.code(traceback.format_exc())

    out_video = os.path.join(out_dir, "annotated_football_analysis.mp4")
    result = {"output_video": out_video if os.path.exists(out_video) else None}
    return result


def render():
    page_header("Analysis",
                "Run the full pipeline automatically on your uploaded video.")

    analysis_done = st.session_state.get("analysis_done", False)
    has_video = st.session_state.get("uploaded_video") is not None

    done_up_to = -1
    if has_video:
        done_up_to = 0
    if st.session_state.get("processed_video"):
        done_up_to = 1
    if analysis_done:
        done_up_to = 3

    render_pipeline(done_up_to=done_up_to)
    st.markdown("---")

    raw_video = st.session_state.get("uploaded_video")

    if not raw_video or not os.path.exists(raw_video):
        st.warning("No video uploaded. Go to the Upload page first.")
        _, r = st.columns([3, 1])
        with r:
            nav_button("Go to Upload", "Upload")
        return

    # ── Show what will be processed ──────────────────────────────────────────
    st.markdown("##### Input")
    c1, c2 = st.columns([2, 1])
    with c1:
        vname = st.session_state.get("uploaded_video_name", "video.mp4")
        cap = cv2.VideoCapture(raw_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = total / fps if fps > 0 else 0
        cap.release()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card("Video", vname), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("Resolution", f"{w}x{h}"),
                        unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("Frames", f"{total:,}"),
                        unsafe_allow_html=True)
        with m4:
            mm, ss = divmod(int(dur), 60)
            st.markdown(metric_card("Duration", f"{mm}m {ss}s"),
                        unsafe_allow_html=True)

    with c2:
        with st.expander("Preview"):
            st.video(raw_video)

    # ── GPU check ────────────────────────────────────────────────────────────
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_gpu else None
    except ImportError:
        has_gpu = False
        gpu_name = None

    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(metric_card("Device", gpu_name or "CPU"),
                    unsafe_allow_html=True)
    with g2:
        if has_gpu:
            est = "~2-5 min for a 5-min video"
        else:
            est = "Slow on CPU — may take a while"
        st.markdown(metric_card("Estimate", est), unsafe_allow_html=True)

    if not has_gpu:
        st.caption(
            "No GPU detected. The pipeline will run on CPU. "
            "This works but is slower than GPU. For faster processing, "
            "use Google Colab with a T4 GPU."
        )

    # ── Model check ──────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model weights not found. "
            "Place the trained YOLO model in the models/ directory or project root."
        )
        return

    # ── Run button ───────────────────────────────────────────────────────────
    st.markdown("---")

    if analysis_done:
        result = st.session_state.get("analysis_results", {})

        st.success("Pipeline completed successfully.")

        if result:
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(metric_card("Frames Processed",
                                         f"{result.get('total_frames', 0):,}"),
                            unsafe_allow_html=True)
            with r2:
                st.markdown(metric_card("Output FPS",
                                         f"{result.get('fps', 0):.1f}"),
                            unsafe_allow_html=True)
            with r3:
                st.markdown(metric_card("Replays Detected",
                                         str(result.get("replays_detected", 0))),
                            unsafe_allow_html=True)
            with r4:
                st.markdown(metric_card("Replay Frames",
                                         f"{result.get('replay_frames_skipped', 0):,}"),
                            unsafe_allow_html=True)

        col_rerun, col_results = st.columns(2)
        with col_rerun:
            if st.button("Re-run Pipeline", use_container_width=True):
                st.session_state.analysis_done = False
                st.session_state.pop("analysis_results", None)
                st.session_state.pop("processed_video", None)
                st.session_state.pop("tracked_video", None)
                st.rerun()
        with col_results:
            nav_button("View Results", "Results", key="an_to_results")

    else:
        st.markdown(f"""
        <div style="background: {BG_CARD}; border: 1px solid rgba(255,255,255,0.05);
                    border-radius: 10px; padding: 1.2rem; margin-bottom: 1rem;
                    font-size: 0.88rem; color: {TEXT_MUTED}; line-height: 1.6;">
            Clicking <strong style="color: {TEXT_PRIMARY};">Run Full Pipeline</strong>
            will automatically:
            <br>1. Preprocess the video (resize + FPS normalization)
            <br>2. Run YOLO object detection on every frame
            <br>3. Track players with ByteTrack + camera compensation
            <br>4. Segment teams, track possession, detect replays
            <br>5. Compute velocity, distance, and export all stats
            <br><br>
            The output will be an annotated video and CSV data files.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Sample Processing")
        max_frames_to_process = st.slider(
            "Process only a sample (0 = full video)",
            min_value=0,
            max_value=min(total, 500) if total > 0 else 500,
            value=50,
            step=10,
            key="max_frames_to_process",
            help="Process a small sample to see results quickly."
        )

        if st.button("Run Full Pipeline", type="primary",
                      use_container_width=True):
            progress = st.progress(0, text="Starting pipeline...")
            status = st.empty()

            try:
                max_f = st.session_state.get("max_frames_to_process", 0)
                result = _full_pipeline(raw_video, max_f, progress, status)
                st.session_state.analysis_done = True
                st.session_state.analysis_results = result
                st.session_state.tracked_video = result.get("output_video")

                status.empty()
                st.success("Pipeline complete. Redirecting to results...")
                st.session_state.page = "Results"
                st.rerun()

            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Navigation
    st.markdown("---")
    left, _, right = st.columns([1, 2, 1])
    with left:
        nav_button("Back to Upload", "Upload", key="an_back")
    with right:
        if analysis_done:
            nav_button("View Results", "Results", key="an_next")
