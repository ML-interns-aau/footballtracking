# app/pages/analysis_page.py
"""
Analysis Page — fully automated pipeline.

One button runs: Preprocess → Detection → Tracking → Post-processing.
No user input required after clicking "Run".
"""

import os
import json
import shutil
import subprocess
import sys
import re
import html
import cv2

import numpy as np
from types import SimpleNamespace
from pathlib import Path

import streamlit as st

from app.config import (
    MODEL_PATH, PROCESSED_DIR, ANNOTATIONS_DIR, INSIGHTS_DIR,
    PLAYER_CLASS_IDS, BALL_CLASS_ID, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMGSZ,
    DEFAULT_TARGET_FPS, DEFAULT_RESIZE_W,
)
from app.utils import (
    page_header, render_pipeline, nav_button, metric_card,
    ACCENT, TEXT_PRIMARY, TEXT_MUTED, BG_CARD,
)


def _pipeline_command():
    project_root = Path(__file__).resolve().parents[2]
    main_py = project_root / "main.py"
    out_dir = Path(INSIGHTS_DIR)
    input_video = st.session_state.get("processed_video") or st.session_state.get("uploaded_video", "")

    args = [
        sys.executable,
        "-u",
        str(main_py),
        "--input",
        input_video,
        "--output_dir",
        str(out_dir),
        "--max_frames",
        str(st.session_state.get("analysis_max_frames", 0)),
        "--target_fps",
        str(st.session_state.get("analysis_target_fps", DEFAULT_TARGET_FPS)),
        "--resize_width",
        str(st.session_state.get("analysis_resize_width", DEFAULT_RESIZE_W)),
        "--conf",
        str(st.session_state.get("analysis_conf", DEFAULT_CONF)),
        "--iou",
        str(st.session_state.get("analysis_iou", DEFAULT_IOU)),
        "--imgsz",
        str(st.session_state.get("analysis_imgsz", DEFAULT_IMGSZ)),
        "--model_path",
        MODEL_PATH,
    ]

    device = st.session_state.get("analysis_device", None)
    if device is not None:
        args.extend(["--device", str(device)])

    return args, out_dir


def _estimate_processed_frames(total_frames: int, source_fps: float) -> int:
    target_fps = float(st.session_state.get("analysis_target_fps", DEFAULT_TARGET_FPS) or 0)
    max_frames = int(st.session_state.get("analysis_max_frames", 0) or 0)

    if total_frames <= 0:
        return max_frames if max_frames > 0 else 0

    frame_step = 1
    if target_fps > 0 and source_fps > 0:
        frame_step = max(1, round(source_fps / target_fps))

    estimated = (total_frames + frame_step - 1) // frame_step
    if max_frames > 0:
        estimated = min(estimated, max_frames)
    return max(estimated, 1)


def _run_pipeline_with_logs(status, progress, log_placeholder, total_estimated: int):
    command, out_dir = _pipeline_command()
    status.info("Starting pipeline...")
    progress.progress(0, text="Preparing pipeline...")

    log_lines = ["$ " + " ".join(command)]

    def render_logs():
        safe_text = html.escape("\n".join(log_lines))
        log_placeholder.markdown(
            f"""
            <div style="max-height: 320px; overflow-y: auto; padding: 0.75rem;
                        border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
                        background: rgba(0,0,0,0.22); font-family: monospace;
                        font-size: 0.8rem; line-height: 1.45; white-space: pre-wrap;">
                <pre style="margin: 0; white-space: pre-wrap;">{safe_text}</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_logs()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    progress_pattern = re.compile(r"\[PROGRESS\]\s+(\d+)\/(\d+)")
    total = total_estimated
    seen_progress = 0

    assert process.stdout is not None
    for raw_line in iter(process.stdout.readline, ""):
        line = raw_line.rstrip("\n")
        if not line:
            continue

        match = progress_pattern.search(line)
        if match:
            seen_progress = int(match.group(1))
            if total <= 0:
                total = int(match.group(2))
            if total > 0:
                progress.progress(min(seen_progress / total, 1.0), text=f"Processing frame {seen_progress}/{total}...")

        log_lines.append(line)
        log_lines = log_lines[-200:]
        render_logs()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Pipeline exited with code {return_code}")

    progress.progress(1.0, text="Pipeline complete.")
    status.success("Pipeline complete.")

    out_video = out_dir / "annotated_football_analysis.mp4"
    return {"output_video": str(out_video) if out_video.exists() else None, "return_code": return_code}


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

    # render_pipeline(done_up_to=done_up_to)
    st.markdown("---")

    raw_video = st.session_state.get("uploaded_video")
    processed_video = st.session_state.get("processed_video")

    if not raw_video or not os.path.exists(raw_video):
        st.warning("No video uploaded. Go to the Upload page first.")
        _, r = st.columns([3, 1])
        with r:
            nav_button("Go to Upload", "Upload")
        return

    if not processed_video or not os.path.exists(processed_video):
        st.warning("Preprocess the video first so Analysis can use the normalized output.")
        _, r = st.columns([3, 1])
        with r:
            nav_button("Go to Preprocess", "Preprocess")
        return

    # ── Show what will be processed ──────────────────────────────────────────
    st.markdown("##### Input")
    c1, c2 = st.columns([2, 1])
    with c1:
        vname = st.session_state.get("uploaded_video_name", "video.mp4")
        cap = cv2.VideoCapture(raw_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur = total / source_fps if source_fps > 0 else 0
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

    st.markdown("##### Pipeline Settings")
    settings_left, settings_right = st.columns(2)

    with settings_left:
        conf = st.slider(
            "Detection Confidence",
            min_value=0.05,
            max_value=0.95,
            value=float(DEFAULT_CONF),
            step=0.01,
            key="analysis_conf_ui",
            help="Higher values reduce false positives but may miss players or the ball.",
        )
        iou = st.slider(
            "NMS IOU Threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(DEFAULT_IOU),
            step=0.01,
            key="analysis_iou_ui",
            help="Higher values keep more overlapping detections.",
        )

    with settings_right:
        imgsz = st.select_slider(
            "Inference Image Size",
            options=[640, 960, 1280, 1536],
            value=DEFAULT_IMGSZ,
            key="analysis_imgsz_ui",
            help="Larger sizes can improve small-object detection at the cost of speed.",
        )

        device_options = ["Auto", "CPU"]
        if has_gpu:
            device_options.append("GPU (cuda:0)")
        device_choice = st.radio(
            "Compute Device",
            options=device_options,
            index=0,
            horizontal=True,
            key="analysis_device_ui",
            help="Auto uses GPU when available, otherwise CPU.",
        )

        max_frames_to_process = st.slider(
            "Process only a sample (0 = full video)",
            min_value=0,
            max_value=max(total, 500) if total > 0 else 500,
            value=st.session_state.get("max_frames_to_process", 50),
            step=10,
            key="max_frames_to_process",
            help="Process a small sample to see results quickly.",
        )

    device_value = None
    if device_choice == "CPU":
        device_value = "cpu"
    elif device_choice.startswith("GPU"):
        device_value = 0

    # Read preprocessing settings (set by Preprocess page)
    st.session_state.analysis_target_fps = st.session_state.get("target_fps", DEFAULT_TARGET_FPS)
    st.session_state.analysis_resize_width = st.session_state.get("resize_width", DEFAULT_RESIZE_W)
    st.session_state.analysis_conf = conf
    st.session_state.analysis_iou = iou
    st.session_state.analysis_imgsz = imgsz
    st.session_state.analysis_device = device_value
    st.session_state.analysis_max_frames = max_frames_to_process

    estimated_frames = _estimate_processed_frames(total, source_fps)

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

        with st.expander("Live Logs", expanded=True):
            log_placeholder = st.empty()
            log_placeholder.markdown(
                """
                <div style="max-height: 320px; overflow-y: auto; padding: 0.75rem;
                            border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
                            background: rgba(0,0,0,0.22); font-family: monospace;
                            font-size: 0.8rem; line-height: 1.45;">
                    Ones you run the pipeline logs will appear here...
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Run Full Pipeline", type="primary",
                      use_container_width=True):
            progress = st.progress(0, text="Starting pipeline...")
            status = st.empty()

            try:
                result = _run_pipeline_with_logs(status, progress, log_placeholder, estimated_frames)
                st.session_state.analysis_done = True
                st.session_state.analysis_results = result
                st.session_state.tracked_video = result.get("output_video")
                st.session_state.processed_video = raw_video  # Mark preprocessing as done

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
