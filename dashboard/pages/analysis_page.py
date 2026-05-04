# dashboard/pages/analysis_page.py
"""
Analysis Page — fully automated pipeline.

Runs the pipeline in a background thread so the Streamlit UI stays
responsive and can display a live progress bar.
"""

from __future__ import annotations

import os
import json
import time
import threading
import traceback
from types import SimpleNamespace

import cv2
import streamlit as st

from dashboard.config import (
    MODEL_PATH, PROCESSED_DIR, ANNOTATIONS_DIR, INSIGHTS_DIR,
    PLAYER_CLASS_IDS, BALL_CLASS_ID, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMGSZ,
    DEFAULT_TARGET_FPS, DEFAULT_RESIZE_W, PROJECT_ROOT,
)
from dashboard.utils import (
    page_header, render_pipeline, nav_button, metric_card,
    ACCENT, TEXT_PRIMARY, TEXT_MUTED, BG_CARD,
)


# ── GPU detection ─────────────────────────────────────────────────────────────

def _detect_gpu():
    """
    Return (has_gpu: bool, gpu_name: str | None, device: str).

    Checks CUDA via PyTorch.  If the installed PyTorch is CPU-only (common
    when using Python 3.13+ where CUDA wheels are not yet published), we
    fall back to CPU and surface a helpful message.
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, name, "cuda"
        # Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "Apple MPS", "mps"
        # Detect CPU-only torch build so we can give a better message
        build = torch.__version__
        if "+cpu" in build or "cpu" in build.lower():
            return False, f"CPU-only PyTorch ({build})", "cpu"
    except ImportError:
        return False, "PyTorch not installed", "cpu"
    except Exception:
        pass
    return False, None, "cpu"


# ── Pipeline runner (background thread) ──────────────────────────────────────

class _PipelineState:
    """Shared state between the background thread and the Streamlit main thread."""

    def __init__(self):
        self.lock        = threading.Lock()
        self.running     = False
        self.done        = False
        self.error       = None          # str | None
        self.current     = 0             # frames processed
        self.total       = 1             # total frames to process
        self.result      = {}            # populated on success


def _run_pipeline_thread(raw_video: str, max_frames: int, state: _PipelineState, device: str,
                         target_fps: float, resize_width: int, conf: float, iou: float, imgsz: int):
    """Target function for the background pipeline thread."""
    try:
        import main as pipeline_main
        import importlib
        importlib.reload(pipeline_main)
        
        # Create game-specific folder
        from dashboard.config import create_game_folder, update_game_status
        video_name = os.path.basename(raw_video)
        game_id = create_game_folder(video_name)
        
        # Use insights/game_id as output directory
        out_dir = os.path.join(INSIGHTS_DIR, game_id)
        os.makedirs(out_dir, exist_ok=True)
        
        # Update game status to processing
        update_game_status(game_id, "Processing", started=True)

        args = SimpleNamespace(
            input=raw_video,
            output_dir=out_dir,
            max_frames=max_frames,
            target_fps=target_fps,
            resize_width=resize_width,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            model_path=MODEL_PATH,
            game_id=game_id,  # Pass game_id to pipeline
        )

        def _progress(current, total):
            with state.lock:
                state.current = current
                state.total   = max(total, 1)

        # Patch device into pipeline_main so FootballDetector picks it up
        result = pipeline_main.main(args, progress_callback=_progress)
        
        # Update game status to completed
        update_game_status(game_id, "Completed", 
                          total_frames=result.get("total_frames", 0),
                          fps=result.get("fps", 0),
                          resolution=result.get("resolution", ""),
                          output_video=result.get("output_video", ""))

        # Store game_id in result for later use
        result["game_id"] = game_id
        result["game_folder"] = out_dir

        with state.lock:
            state.result  = result or {}
            state.done    = True
            state.running = False

    except Exception as exc:
        with state.lock:
            state.error   = f"{exc}\n\n{traceback.format_exc()}"
            state.done    = True
            state.running = False


# ── Page render ───────────────────────────────────────────────────────────────

def render():
    page_header("Analysis",
                "Run the full pipeline automatically on your uploaded video.")

    analysis_done = st.session_state.get("analysis_done", False)
    has_video     = bool(st.session_state.get("uploaded_video"))

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

    if not raw_video or not os.path.exists(raw_video):
        st.warning("No video uploaded. Go to the Upload page first.")
        _, r = st.columns([3, 1])
        with r:
            nav_button("Go to Upload", "Upload")
        return

    # Use preprocessed video if available, otherwise raw
    proc_video = st.session_state.get("processed_video")
    input_video = proc_video if (proc_video and os.path.exists(proc_video)) else raw_video
    using_preprocessed = input_video != raw_video

    # ── Video info ────────────────────────────────────────────────────
    st.markdown("##### Input")
    c1, c2 = st.columns([2, 1])
    with c1:
        vname = st.session_state.get("uploaded_video_name", os.path.basename(raw_video))
        cap   = cv2.VideoCapture(input_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dur   = total / fps if fps > 0 else 0
        cap.release()

        if using_preprocessed:
            st.info("Using preprocessed video (resized + FPS normalised).")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card("Video", vname), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("Resolution", f"{w}×{h}"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("Frames", f"{total:,}"), unsafe_allow_html=True)
        with m4:
            mm, ss = divmod(int(dur), 60)
            st.markdown(metric_card("Duration", f"{mm}m {ss}s"), unsafe_allow_html=True)

    with c2:
        with st.expander("Preview"):
            with open(input_video, "rb") as vf:
                st.video(vf.read())

    # ── GPU / device info ─────────────────────────────────────────────
    has_gpu, gpu_name, device = _detect_gpu()

    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        display_device = gpu_name if has_gpu else "CPU (no GPU detected)"
        st.markdown(metric_card("Device", display_device), unsafe_allow_html=True)
    with g2:
        est = "~2–5 min for a 5-min video" if has_gpu else "CPU mode — may be slow"
        st.markdown(metric_card("Estimate", est), unsafe_allow_html=True)

    if not has_gpu:
        import sys
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        if gpu_name and "CPU-only" in gpu_name:
            st.warning(
                f"**CPU-only PyTorch detected** (`{gpu_name}`).  \n"
                f"Your system has an NVIDIA GPU but PyTorch CUDA wheels are not yet "
                f"available for Python {py_ver}.  \n"
                "To enable GPU acceleration, create a new virtual environment with "
                "**Python 3.11 or 3.12** and run:  \n"
                "```\npip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n```"
            )
        else:
            st.info(
                "No GPU detected. The pipeline will run on CPU — this works but is slower. "
                "For faster processing install CUDA-enabled PyTorch or use Google Colab (T4 GPU)."
            )

    # ── Model check ───────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model weights not found at `{MODEL_PATH}`. "
            "Place the trained YOLO model in the `models/` directory or project root."
        )
        return

    st.markdown("---")

    # ── Already done ──────────────────────────────────────────────────
    if analysis_done:
        result = st.session_state.get("analysis_results", {})
        st.success("Pipeline completed successfully.")

        if result:
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(
                    metric_card("Frames Processed", f"{result.get('total_frames', 0):,}"),
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    metric_card("Output FPS", f"{result.get('fps', fps):.1f}"),
                    unsafe_allow_html=True,
                )
            with r3:
                st.markdown(
                    metric_card("Resolution", result.get("resolution", f"{w}×{h}")),
                    unsafe_allow_html=True,
                )
            with r4:
                out_vid = result.get("output_video", "")
                size_mb = (
                    os.path.getsize(out_vid) / (1024 * 1024)
                    if out_vid and os.path.exists(out_vid)
                    else 0
                )
                st.markdown(
                    metric_card("Output Size", f"{size_mb:.1f} MB"),
                    unsafe_allow_html=True,
                )

        col_rerun, col_results = st.columns(2)
        with col_rerun:
            if st.button("Re-run Pipeline", width='stretch'):
                for k in ["analysis_done", "analysis_results", "tracked_video",
                          "_pipeline_state", "_pipeline_thread"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col_results:
            nav_button("View Results →", "Results", key="an_to_results")

        st.markdown("---")
        left, _, right = st.columns([1, 2, 1])
        with left:
            nav_button("← Back to Upload", "Upload", key="an_back_done")
        with right:
            nav_button("View Results →", "Results", key="an_next_done")
        return

    # ── Pipeline currently running ────────────────────────────────────
    state: _PipelineState | None = st.session_state.get("_pipeline_state")

    if state is not None and state.running:
        with state.lock:
            current = state.current
            total_f = state.total
            done    = state.done
            error   = state.error

        if error:
            st.error(f"Pipeline failed:\n\n{error}")
            st.session_state.pop("_pipeline_state", None)
            st.session_state.pop("_pipeline_thread", None)
            return

        if done:
            # Thread finished — collect results
            with state.lock:
                result = dict(state.result)
            st.session_state.analysis_done     = True
            st.session_state.analysis_results  = result
            st.session_state.tracked_video     = result.get("output_video")
            st.session_state.pop("_pipeline_state", None)
            st.session_state.pop("_pipeline_thread", None)
            st.session_state.page = "Results"
            st.rerun()
            return

        # Still running — show live progress
        pct = min(current / max(total_f, 1), 1.0)
        st.markdown("#### Pipeline running…")
        progress_bar = st.progress(pct, text=f"Processing frame {current:,} / {total_f:,}")
        st.caption(
            "The pipeline is running in the background. "
            "This page refreshes automatically every 2 seconds."
        )
        # Auto-refresh every 2 s while the thread is alive
        time.sleep(2)
        st.rerun()
        return

    # ── Not yet started — show run button ─────────────────────────────
    st.markdown(f"""
    <div style="background:{BG_CARD}; border:1px solid rgba(255,255,255,0.05);
                border-radius:10px; padding:1.2rem; margin-bottom:1rem;
                font-size:0.88rem; color:{TEXT_MUTED}; line-height:1.6;">
        Clicking <strong style="color:{TEXT_PRIMARY};">Run Full Pipeline</strong> will:
        <br>1. Preprocess the video (resize + FPS normalisation)
        <br>2. Run YOLO object detection on every frame
        <br>3. Track players with ByteTrack + camera compensation
        <br>4. Segment teams, track possession
        <br>5. Compute velocity, distance, and export all stats
        <br><br>
        The output is an annotated video and CSV data files saved to <code>results/</code>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Sample Processing")
    max_val = min(total, 500) if total > 0 else 500
    max_frames_to_process = st.slider(
        "Process only a sample (0 = full video)",
        min_value=0,
        max_value=max_val,
        value=min(50, max_val),
        step=10,
        key="max_frames_slider",
        help="Set to 0 to process the entire video.",
    )

    if st.button("▶  Run Full Pipeline", type="primary", width='stretch'):
        max_f = max_frames_to_process  # read directly from the widget return value
        
        # Use preprocessed values if available, otherwise use user selections
        effective_fps = preproc_fps if using_preprocessed else target_fps
        effective_width = preproc_width if using_preprocessed else resize_width

        state = _PipelineState()
        state.running = True
        state.total   = max_f if max_f > 0 else max(total, 1)

        thread = threading.Thread(
            target=_run_pipeline_thread,
            args=(input_video, max_f, state, device, effective_fps, effective_width, conf, iou, imgsz),
            daemon=True,
        )
        thread.start()

        st.session_state["_pipeline_state"]  = state
        st.session_state["_pipeline_thread"] = thread
        st.rerun()

    # Navigation
    st.markdown("---")
    left, _, right = st.columns([1, 2, 1])
    with left:
        nav_button("← Back to Upload", "Upload", key="an_back")
