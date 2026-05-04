# app/pages/preprocess_page.py
"""Preprocess Page — resize and normalise the uploaded video before analysis."""

import os
from pathlib import Path

import cv2
import streamlit as st

from app.config import PROCESSED_DIR, DEFAULT_TARGET_FPS, DEFAULT_RESIZE_W
from app.utils import page_header, nav_button, metric_card
from src.preprocessing import preprocess_video


def _video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def render():
    page_header("Preprocessing", "Resize and normalise your video before running analysis.")

    raw = st.session_state.get("uploaded_video")
    proc = st.session_state.get("processed_video")

    if not raw or not os.path.exists(raw):
        st.warning("No video uploaded yet.")
        _, r = st.columns([3, 1])
        with r:
            nav_button("Go to Upload", "Upload")
        return

    info = _video_info(raw)

    st.markdown("##### Input Video")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Resolution", f"{info['width']}x{info['height']}"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("FPS", f"{info['fps']:.1f}"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Frames", f"{info['frames']:,}"), unsafe_allow_html=True)
    with c4:
        m, s = divmod(int(info["duration"]), 60)
        st.markdown(metric_card("Duration", f"{m}m {s}s"), unsafe_allow_html=True)

    with st.expander("Preview raw video", expanded=False):
        st.video(raw)

    st.markdown("---")
    st.markdown("##### Settings")

    s1, s2 = st.columns(2)
    with s1:
        fps = st.slider(
            "Target FPS",
            min_value=5,
            max_value=60,
            value=int(st.session_state.get("target_fps", DEFAULT_TARGET_FPS)),
            step=1,
            key="preprocess_fps",
            help="Lower FPS reduces the number of frames sent to analysis.",
        )
    with s2:
        width = st.select_slider(
            "Output Width (px)",
            options=[640, 960, 1280, 1920],
            value=int(st.session_state.get("resize_width", DEFAULT_RESIZE_W)),
            key="preprocess_width",
            help="Frames are resized before analysis.",
        )

    st.session_state.target_fps = fps
    st.session_state.resize_width = width

    scale = width / float(info["width"])
    new_h = max(1, int(round(info["height"] * scale)))
    interval = max(1, int(round(info["fps"] / fps)))
    est_frames = max(1, info["frames"] // interval)

    st.caption(
        f"Output: {width}x{new_h} at {fps} FPS — about {est_frames:,} frames (every {interval}th frame)."
    )

    if proc and os.path.exists(proc):
        st.success(f"Preprocessing complete: {os.path.relpath(proc)}")

        pinfo = _video_info(proc)
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(metric_card("Output Resolution", f"{pinfo['width']}x{pinfo['height']}"), unsafe_allow_html=True)
        with p2:
            st.markdown(metric_card("Output FPS", f"{pinfo['fps']:.1f}"), unsafe_allow_html=True)
        with p3:
            st.markdown(metric_card("Output Frames", f"{pinfo['frames']:,}"), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### Homography Calibration (Optional)")
        st.write("To get accurate player speeds and pitch maps, click the 4 corners of the pitch area visible in the camera. Order: **Bottom-Left, Bottom-Right, Top-Right, Top-Left**.")
        
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            from src.homography.pitch_mapping import PitchMapping
            
            cap = cv2.VideoCapture(proc)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                value = streamlit_image_coordinates(frame_rgb, key="pil")
                
                if value is not None:
                    point = [value["x"], value["y"]]
                    if "src_points" not in st.session_state:
                        st.session_state.src_points = []
                    
                    if point not in st.session_state.src_points and len(st.session_state.src_points) < 4:
                        st.session_state.src_points.append(point)
                        st.rerun()
                
                points = st.session_state.get("src_points", [])
                if points:
                    st.write(f"**Points selected ({len(points)}/4):** {points}")
                    if st.button("Clear Points"):
                        st.session_state.src_points = []
                        st.rerun()
                        
                if len(points) == 4:
                    st.success("4 points collected!")
                    if st.button("Save Calibration & Lock", type="primary"):
                        dst_pts = [[0, 68], [105, 68], [105, 0], [0, 0]]
                        mapper = PitchMapping(points, dst_pts)
                        mapper.save_config("configs/homography.json")
                        st.session_state.calibration_saved = True
                        st.rerun()
                        
                if st.session_state.get("calibration_saved", False):
                    st.success("✅ Calibration saved to `configs/homography.json`. The pipeline will now use this for accurate mapping!")
        except ImportError:
            st.warning("`streamlit-image-coordinates` is not installed. Please pip install it to use the visual calibration tool.")

        st.markdown("---")
        back, _, next_col = st.columns([1, 2, 1])
        with back:
            nav_button("← Back to Upload", "Upload", key="pre_back_done")
        with next_col:
            nav_button("Run Analysis →", "Analysis", key="pre_next_done")
        return

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        nav_button("← Back to Upload", "Upload", key="pre_back")
    with right:
        if st.button("Process Video", type="primary", width='stretch'):
            output_name = f"{Path(raw).stem}_preprocessed.mp4"
            output_path = Path(PROCESSED_DIR) / output_name

            progress = st.progress(0.0, text="Starting preprocessing...")
            status = st.empty()

            try:
                def _on_progress(current: int, total: int):
                    if total > 0:
                        progress.progress(min(current / total, 1.0), text=f"Processing frame {current}/{total}...")

                status.info("Preprocessing video...")
                preprocess_video(raw, output_path, float(fps), int(width), progress_callback=_on_progress)
                st.session_state.processed_video = str(output_path)
                status.success("Preprocessing complete.")
                progress.progress(1.0, text="Preprocessing complete.")
                st.rerun()
            except Exception as exc:
                status.error(f"Preprocessing failed: {exc}")
                st.error(f"Preprocessing failed: {exc}")
