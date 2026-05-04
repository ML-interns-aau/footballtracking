# dashboard/pages/preprocess_page.py
"""
Preprocess Page — resize video and normalise FPS before analysis.

Runs cv2-based preprocessing in a background thread so the UI stays
responsive. Saves the output to data/processed/ and stores the path
in session state so the Analysis page picks it up automatically.
"""

import os
import time
import threading
import traceback

import cv2
import streamlit as st

from dashboard.config import PROCESSED_DIR, DEFAULT_TARGET_FPS, DEFAULT_RESIZE_W
from dashboard.utils import page_header, render_pipeline, nav_button, metric_card, BG_CARD, TEXT_MUTED, TEXT_PRIMARY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    info = {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


# ── Background preprocessing thread ──────────────────────────────────────────

class _PreprocState:
    def __init__(self):
        self.lock      = threading.Lock()
        self.running   = False
        self.done      = False
        self.error     = None   # str | None
        self.current   = 0      # frames written
        self.total     = 1
        self.out_path  = ""


def _run_preprocess(raw_path: str, out_path: str,
                    target_fps: int, target_width: int,
                    state: _PreprocState):
    """Resize + FPS-normalise the video using OpenCV."""
    try:
        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {raw_path}")

        src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_src  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Compute output dimensions (keep aspect ratio)
        scale   = target_width / src_w
        out_w   = target_width
        out_h   = int(src_h * scale)
        # Ensure even dimensions (required by some codecs)
        out_w  += out_w % 2
        out_h  += out_h % 2

        # Frame sampling interval
        frame_step = max(1, round(src_fps / target_fps))
        est_total  = max(1, total_src // frame_step)

        with state.lock:
            state.total = est_total

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Use mp4v — browser-compatible H.264 re-encode happens via _reencode_h264
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(target_fps), (out_w, out_h))

        frame_idx   = 0
        written     = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                if out_w != src_w or out_h != src_h:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1
                with state.lock:
                    state.current = written

            frame_idx += 1

        cap.release()
        writer.release()

        with state.lock:
            state.out_path = out_path
            state.done     = True
            state.running  = False

    except Exception as exc:
        with state.lock:
            state.error   = f"{exc}\n\n{traceback.format_exc()}"
            state.done    = True
            state.running = False


# ── Page render ───────────────────────────────────────────────────────────────

def render():
    page_header("Preprocessing",
                "Resize and normalise your video before running the analysis pipeline.")

    done_preproc = st.session_state.get("processed_video") is not None
    render_pipeline(active=1, done_up_to=1 if done_preproc else 0)

    raw = st.session_state.get("uploaded_video")

    if not raw or not os.path.exists(raw):
        st.warning("No video uploaded yet.")
        _, r = st.columns([3, 1])
        with r:
            nav_button("Go to Upload", "Upload")
        return

    info = _video_info(raw)

    # ── Input info ────────────────────────────────────────────────────
    st.markdown("##### Input Video")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Resolution", f"{info['width']}×{info['height']}"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("FPS", f"{info['fps']:.1f}"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Frames", f"{info['frames']:,}"), unsafe_allow_html=True)
    with c4:
        m, s = divmod(int(info["duration"]), 60)
        st.markdown(metric_card("Duration", f"{m}m {s}s"), unsafe_allow_html=True)

    with st.expander("▶ Preview raw video"):
        with open(raw, "rb") as vf:
            st.video(vf.read())

    st.markdown("---")

    # ── Already preprocessed ─────────────────────────────────────────
    proc = st.session_state.get("processed_video")
    if proc and os.path.exists(proc):
        st.success(f"Preprocessing complete → `{os.path.relpath(proc)}`")

        pinfo = _video_info(proc)
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown(metric_card("Output Resolution", f"{pinfo['width']}×{pinfo['height']}"), unsafe_allow_html=True)
        with p2:
            st.markdown(metric_card("Output FPS", f"{pinfo['fps']:.1f}"), unsafe_allow_html=True)
        with p3:
            st.markdown(metric_card("Output Frames", f"{pinfo['frames']:,}"), unsafe_allow_html=True)
        with p4:
            size_mb = os.path.getsize(proc) / (1024 * 1024)
            st.markdown(metric_card("File Size", f"{size_mb:.1f} MB"), unsafe_allow_html=True)

        with st.expander("▶ Preview preprocessed video"):
            with open(proc, "rb") as vf:
                st.video(vf.read())

        col_redo, col_next = st.columns(2)
        with col_redo:
            if st.button("Re-run Preprocessing", width='stretch'):
                st.session_state.pop("processed_video", None)
                st.session_state.pop("_preproc_state", None)
                st.session_state.pop("_preproc_thread", None)
                st.rerun()
        with col_next:
            nav_button("Run Analysis →", "Analysis", key="pre_to_analysis")

        st.markdown("---")
        left, _, right = st.columns([1, 2, 1])
        with left:
            nav_button("← Back to Upload", "Upload", key="pre_back_done")
        with right:
            nav_button("Run Analysis →", "Analysis", key="pre_next_done")
        return

    # ── Preprocessing running ─────────────────────────────────────────
    state: _PreprocState | None = st.session_state.get("_preproc_state")

    if state is not None and state.running:
        with state.lock:
            current = state.current
            total_f = state.total
            done    = state.done
            error   = state.error
            out_p   = state.out_path

        if error:
            st.error(f"Preprocessing failed:\n\n{error}")
            st.session_state.pop("_preproc_state", None)
            st.session_state.pop("_preproc_thread", None)
            return

        if done:
            st.session_state.processed_video = out_p
            st.session_state.pop("_preproc_state", None)
            st.session_state.pop("_preproc_thread", None)
            st.rerun()
            return

        pct = min(current / max(total_f, 1), 1.0)
        st.markdown("#### Preprocessing…")
        st.progress(pct, text=f"Frame {current:,} / {total_f:,}")
        st.caption("Resizing and normalising FPS. This page refreshes every 2 seconds.")
        time.sleep(2)
        st.rerun()
        return

    # ── Settings + run button ─────────────────────────────────────────
    st.markdown("##### Settings")
    s1, s2 = st.columns(2)
    with s1:
        fps = st.slider("Target FPS", 5, 60, DEFAULT_TARGET_FPS,
                        key="preprocess_fps",
                        help="Lower FPS = faster analysis. 15 FPS is a good balance.")
    with s2:
        width = st.select_slider("Output Width (px)", [640, 960, 1280, 1920],
                                 value=DEFAULT_RESIZE_W, key="preprocess_width",
                                 help="Smaller = faster. 1280 is recommended.")

    scale     = width / info["width"]
    new_h     = int(info["height"] * scale)
    interval  = max(1, int(info["fps"] / fps))
    est_frames = info["frames"] // interval
    size_reduction = (1 - (width * new_h) / (info["width"] * info["height"])) * 100

    st.markdown(f"""
    <div style="background:{BG_CARD}; border:1px solid rgba(255,255,255,0.05);
                border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0 1.2rem;
                font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.7;">
        Output: <strong style="color:{TEXT_PRIMARY}">{width}×{new_h}</strong>
        at <strong style="color:{TEXT_PRIMARY}">{fps} FPS</strong>
        — ~<strong style="color:{TEXT_PRIMARY}">{est_frames:,} frames</strong>
        (every {interval}th frame sampled)
        &nbsp;·&nbsp; ~{size_reduction:.0f}% smaller than original
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶  Run Preprocessing", type="primary", width='stretch'):
        vname    = st.session_state.get("uploaded_video_name", os.path.basename(raw))
        stem     = os.path.splitext(vname)[0]
        out_path = os.path.join(PROCESSED_DIR, f"{stem}_preprocessed.mp4")

        state = _PreprocState()
        state.running = True
        state.total   = est_frames

        thread = threading.Thread(
            target=_run_preprocess,
            args=(raw, out_path, fps, width, state),
            daemon=True,
        )
        thread.start()

        st.session_state["_preproc_state"]  = state
        st.session_state["_preproc_thread"] = thread
        st.rerun()

    st.markdown("---")
    left, _, right = st.columns([1, 2, 1])
    with left:
        nav_button("← Back to Upload", "Upload", key="pre_back")
    with right:
        st.caption("You can also skip preprocessing →")
        nav_button("Skip → Run Analysis", "Analysis", key="pre_skip")
