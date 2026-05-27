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
    DEFAULT_TARGET_FPS, DEFAULT_RESIZE_W, create_game_folder, update_game_status,
)
from app.utils import (
    page_header, render_pipeline, nav_button, metric_card,
    ACCENT, TEXT_PRIMARY, TEXT_MUTED, BG_CARD,
)
def _pipeline_command():
    project_root = Path(__file__).resolve().parents[2]
    main_py = project_root / "main.py"
    input_video = st.session_state.get("processed_video") or st.session_state.get("uploaded_video", "")
    video_name = os.path.basename(input_video)
    game_id = create_game_folder(video_name)
    out_dir = Path(INSIGHTS_DIR) / game_id
    st.session_state["current_game_id"] = game_id
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
        "--game_id",
        game_id,
    ]
    device = st.session_state.get("analysis_device", None)
    if device is not None:
        args.extend(["--device", str(device)])
    return args, out_dir, game_id
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
    command, out_dir, game_id = _pipeline_command()
    status.info("Starting pipeline...")
    progress.progress(0, text="Preparing pipeline...")
    update_game_status(game_id, "Processing", started=True)
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
                <div style="max-height: 320px; overflow-y: auto; padding: 0.75rem;
                            border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
                            background: rgba(0,0,0,0.22); font-family: monospace;
                            font-size: 0.8rem; line-height: 1.45;">
                    Ones you run the pipeline logs will appear here...
                </div>