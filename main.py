import cv2
import argparse
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

from app.config import (
    MODEL_PATH,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_IMGSZ,
    DEFAULT_TARGET_FPS,
    DEFAULT_RESIZE_W,
)
from src.pipeline.detector import FootballDetector
from src.pipeline.tracker import FootballTracker
from src.pipeline.team_classifier import TeamClassifier
from src.pipeline.camera_motion import CameraMotionEstimator
from src.pipeline.pitch_mapper import PitchMapper
from src.pipeline.speed_estimator import SpeedEstimator
from src.pipeline.ball_tracker import BallTracker
from src.pipeline.data_exporter import DataExporter
from src.pipeline.heatmap_analyzer import HeatmapAnalyzer
from src.pipeline.visualizer import PipelineVisualizer
from src.pipeline.tracking_csv_builder import TrackingCSVBuilder
from src.preprocessing.resolution_normalization import resize_frame


def _get_device():
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(args, progress_callback=None):
    """
    Run the full football analytics pipeline.

    Parameters
    ----------
    args : namespace with .input, .output_dir, .max_frames
    progress_callback : optional callable(current_frame, total_frames)
        Called after every frame so callers (e.g. Streamlit) can update a
        progress bar without blocking the UI.
    """
    device = _get_device()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_fps = float(getattr(args, "target_fps", DEFAULT_TARGET_FPS) or 0)
    resize_width = int(getattr(args, "resize_width", DEFAULT_RESIZE_W) or 0)
    conf = float(getattr(args, "conf", DEFAULT_CONF))
    iou = float(getattr(args, "iou", DEFAULT_IOU))
    imgsz = int(getattr(args, "imgsz", DEFAULT_IMGSZ))
    device = getattr(args, "device", None)
    model_path = getattr(args, "model_path", MODEL_PATH)

    if isinstance(device, str) and device.isdigit():
        device = int(device)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if resize_width > 0:
        scale = resize_width / float(width)
        width = resize_width
        height = max(1, int(round(height * scale)))

    frame_step = 1
    effective_fps = fps
    if target_fps > 0 and fps > 0:
        frame_step = max(1, round(fps / target_fps))
        effective_fps = fps / frame_step

    planned_frames = 0
    if total_frames > 0:
        planned_frames = (total_frames + frame_step - 1) // frame_step
    if args.max_frames and args.max_frames > 0:
        planned_frames = min(planned_frames, args.max_frames) if planned_frames > 0 else args.max_frames

    print(
        f"[START] input={input_path} output_dir={output_dir} fps={fps:.2f} target_fps={target_fps:.2f} "
        f"frame_step={frame_step} planned_frames={planned_frames} conf={conf:.2f} iou={iou:.2f} "
        f"imgsz={imgsz} device={device if device is not None else 'auto'}",
        flush=True,
    )

    out_video_path = output_dir / "annotated_football_analysis.mp4"
    # Use avc1 (H.264) so the output is browser-playable in st.video()
    # Falls back to mp4v if avc1 is not available on this system
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    ret, initial_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read first frame.")

    # ── Initialise pipeline components ───────────────────────────────
    model_path = "yolov8m_fixed.pt"
    if not Path(model_path).exists():
        from dashboard.config import MODEL_PATH as DASHBOARD_MODEL_PATH
        model_path = DASHBOARD_MODEL_PATH

    detector        = FootballDetector(model_path=model_path, conf=0.30, iou=0.40, device=device)
    tracker         = FootballTracker(track_thresh=0.20, track_buffer=60, match_thresh=0.80)
    team_classifier = TeamClassifier(n_teams=2, history_len=15, refit_interval=150)
    ball_tracker    = BallTracker(max_trail=25, max_missed=30)
    camera_motion   = CameraMotionEstimator(initial_frame)
    data_exporter   = DataExporter(output_dir=str(output_dir))
    heatmap_analyzer = HeatmapAnalyzer(pitch_width=105, pitch_height=68)
    visualizer      = PipelineVisualizer()

    src_pts = [
        [0,           height],
        [width,       height],
        [width * 0.75, height * 0.3],
        [width * 0.25, height * 0.3],
    ]
    dst_pts = [[0, 68], [105, 68], [105, 0], [0, 0]]
    pitch_mapper    = PitchMapper(src_points=src_pts, dst_points=dst_pts)
    speed_estimator = SpeedEstimator(fps=effective_fps, pitch_mapper=pitch_mapper, window_size=8)
    csv_builder     = TrackingCSVBuilder(pitch_mapper=pitch_mapper, fps=effective_fps)
    print("[PHASE] tracking and export components initialized", flush=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    source_frame_idx = 0
    processed_frame_idx = 0
    pbar       = tqdm(total=max_frames, unit="frame")

    # ── Main processing loop ──────────────────────────────────────────
    while cap.isOpened():
        if max_frames > 0 and processed_frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        if source_frame_idx % frame_step != 0:
            source_frame_idx += 1
            continue

        if processed_frame_idx % 25 == 0:
            total_label = planned_frames if planned_frames > 0 else "unknown"
            print(f"[PROGRESS] {processed_frame_idx}/{total_label}", flush=True)

        frame = resize_frame(frame, resize_width)

        detections = detector.detect(frame)
        tracked    = tracker.update(detections)
        team_ids   = team_classifier.assign_teams(frame, tracked)
        cam_dx, cam_dy = camera_motion.update(frame)

        ball_mask       = tracked.class_id == 32
        ball_detections = tracked[ball_mask] if np.any(ball_mask) else None
        ball_cx, ball_cy, ball_predicted = ball_tracker.update(frame, ball_detections)

        player_pixels      = []
        player_tracker_ids = []
        for i, (bbox, class_id, t_id) in enumerate(
            zip(tracked.xyxy, tracked.class_id, tracked.tracker_id)
        ):
            if class_id != 0 or t_id is None:
                continue
            x1, y1, x2, y2 = bbox
            player_pixels.append([(x1 + x2) / 2, y2])
            player_tracker_ids.append(t_id)

        if player_pixels:
            speed_estimator.estimate_speed(
                processed_frame_idx, player_tracker_ids,
                np.array(player_pixels), cam_dx, cam_dy,
            )

        ball_pos_m = pitch_mapper.transform_point((ball_cx, ball_cy))
        speed_estimator.estimate_speed(
            processed_frame_idx,
            [BallTracker.BALL_ID],
            np.array([[ball_cx, ball_cy]]),
            cam_dx, cam_dy,
        )
        ball_speed_kmh, _, _ = speed_estimator.get_stats(BallTracker.BALL_ID)

        frame_objs       = []
        player_positions = {}

        for i in range(len(tracked)):
            bbox     = tracked.xyxy[i]
            class_id = int(tracked.class_id[i])  if tracked.class_id  is not None else -1
            t_id     = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
            tid      = int(team_ids[i])

            if class_id == 0 and t_id is not None:
                speed, dist, (x_m, y_m) = speed_estimator.get_stats(t_id)
                heatmap_analyzer.add_point(t_id, tid, x_m, y_m)
                player_positions[t_id] = (x_m, y_m)

                frame_objs.append({
                    "id":       t_id,
                    "class":    "referee" if tid == -2 else "player",
                    "team":     "Referee" if tid == -2 else (f"Team {tid}" if tid >= 0 else "Unknown"),
                    "x_m":     x_m,  "y_m":     y_m,
                    "speed":   speed, "distance": dist,
                })

            elif class_id == 32:
                frame_objs.append({
                    "id":    "",  "team":  "",  "class": "ball",
                    "x_m":  round(ball_pos_m[0], 2),
                    "y_m":  round(ball_pos_m[1], 2),
                    "speed": round(ball_speed_kmh, 2),
                })

        data_exporter.log_frame(processed_frame_idx, frame_objs)
        data_exporter.update_passes(processed_frame_idx, ball_pos_m, player_positions)
        csv_builder.add_frame(processed_frame_idx, tracked, team_ids)

        annotated = visualizer.annotate_frame(
            frame             = frame,
            detections        = tracked,
            team_ids          = team_ids,
            speed_estimator   = speed_estimator,
            ball_trail        = ball_tracker.get_trail(),
            ball_speed_kmh    = ball_speed_kmh,
            ball_is_predicted = ball_predicted,
            frame_idx         = processed_frame_idx,
        )
        out.write(annotated)

        processed_frame_idx += 1
        source_frame_idx += 1
        pbar.update(1)

        # Notify caller of progress (used by Streamlit progress bar)
        if progress_callback is not None:
            progress_callback(processed_frame_idx, max_frames)

    pbar.close()
    cap.release()
    out.release()

    data_exporter.finalize()
    heatmap_analyzer.save_team_heatmap(0, str(output_dir / "team_0_heatmap.png"))
    heatmap_analyzer.save_team_heatmap(1, str(output_dir / "team_1_heatmap.png"))

    tracking_csv_path = output_dir / "tracking_output.csv"
    csv_builder.finalize_and_write(tracking_csv_path)

    return {
        "total_frames": processed_frame_idx,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "output_video": str(out_video_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Analytics Pipeline")
    parser.add_argument("--input",      type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--target_fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--resize_width", type=int, default=DEFAULT_RESIZE_W)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    main(args)
