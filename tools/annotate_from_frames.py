"""
tools/annotate_from_frames.py
==============================
Processes the first N frames from data/frames/ through the full football
analytics pipeline and writes an annotated video to results/.
Does NOT overwrite the existing annotated_football_analysis.mp4.

Usage (from project root):
    python tools/annotate_from_frames.py --max_frames 500
"""

import cv2
import argparse
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.detector import FootballDetector
from src.pipeline.tracker import FootballTracker
from src.pipeline.team_classifier import TeamClassifier
from src.pipeline.camera_motion import CameraMotionEstimator
from src.homography.pitch_mapping import PitchMapping
from src.pipeline.speed_estimator import SpeedEstimator
from src.pipeline.ball_tracker import BallTracker
from src.pipeline.heatmap_analyzer import HeatmapAnalyzer
from src.pipeline.visualizer import PipelineVisualizer
from src.pipeline.tracking_csv_builder import TrackingCSVBuilder


def main(args):
    frames_dir  = Path("data/frames")
    output_dir  = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort frame paths
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        print("[ERROR] No frames found in data/frames/"); sys.exit(1)

    max_frames = min(args.max_frames, len(frame_paths))
    frame_paths = frame_paths[:max_frames]
    print(f"[INFO] Processing {max_frames} frames from {frames_dir}")

    # Read first frame to get dimensions
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        print("[ERROR] Cannot read first frame."); sys.exit(1)

    height, width = first.shape[:2]
    fps = args.fps

    # Output video — new filename so existing video is preserved
    out_video_path = output_dir / f"annotated_from_frames_{max_frames}f.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # ---- Initialise components ----
    print("[INFO] Initializing components...")
    detector        = FootballDetector(model_path="yolov8m_fixed.pt", conf=0.30, iou=0.40)
    tracker         = FootballTracker(track_thresh=0.20, track_buffer=60, match_thresh=0.80)
    team_classifier = TeamClassifier(n_teams=2, history_len=15, refit_interval=150)
    ball_tracker    = BallTracker(max_trail=25, max_missed=30)
    camera_motion   = CameraMotionEstimator(first)
    heatmap_analyzer= HeatmapAnalyzer(pitch_width=105, pitch_height=68)
    visualizer      = PipelineVisualizer()

    src_pts = [[0, height], [width, height], [width * 0.75, height * 0.3], [width * 0.25, height * 0.3]]
    dst_pts = [[0, 68],     [105, 68],       [105, 0],                       [0, 0]]
    pitch_mapper    = PitchMapping(src_points=src_pts, dst_points=dst_pts)
    speed_estimator = SpeedEstimator(fps=fps, pitch_mapper=pitch_mapper, window_size=8)
    csv_builder     = TrackingCSVBuilder(pitch_mapper=pitch_mapper, fps=fps)

    # ---- Main loop ----
    print(f"[INFO] Annotating {max_frames} frames at {fps} fps ...")
    for frame_idx, frame_path in enumerate(tqdm(frame_paths, unit="frame")):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        detections          = detector.detect(frame)
        tracked             = tracker.update(detections)
        team_ids            = team_classifier.assign_teams(frame, tracked)

        if pitch_mapper is not None:
            for i in range(len(tracked)):
                if team_ids[i] == -2 and tracked.class_id[i] == 0:
                    x1, y1, x2, y2 = tracked.xyxy[i]
                    foot_x, foot_y = (x1 + x2) / 2, y2
                    px, py = pitch_mapper.transform_point((foot_x, foot_y))
                    if px < 25:
                        team_ids[i] = -3 # GK0
                    elif px > 80:
                        team_ids[i] = -4 # GK1
        cam_dx, cam_dy      = camera_motion.update(frame)

        ball_mask           = tracked.class_id == 32
        ball_detections     = tracked[ball_mask] if np.any(ball_mask) else None
        ball_cx, ball_cy, ball_predicted = ball_tracker.update(frame, ball_detections)

        # Player speeds
        player_pixels, player_tids = [], []
        for i, (bbox, class_id, t_id) in enumerate(
            zip(tracked.xyxy, tracked.class_id, tracked.tracker_id)
        ):
            if class_id != 0 or t_id is None:
                continue
            x1, y1, x2, y2 = bbox
            player_pixels.append([(x1 + x2) / 2, y2])
            player_tids.append(t_id)
        if player_pixels:
            speed_estimator.estimate_speed(frame_idx, player_tids, np.array(player_pixels), cam_dx, cam_dy)

        ball_pos_m = pitch_mapper.transform_point((ball_cx, ball_cy))
        speed_estimator.estimate_speed(frame_idx, [BallTracker.BALL_ID], np.array([[ball_cx, ball_cy]]), cam_dx, cam_dy)
        ball_speed_kmh, _, _ = speed_estimator.get_stats(BallTracker.BALL_ID)

        # Heatmap & CSV
        for i in range(len(tracked)):
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            t_id     = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
            tid      = int(team_ids[i])
            if class_id == 0 and t_id is not None:
                _, _, (x_m, y_m) = speed_estimator.get_stats(t_id)
                heatmap_analyzer.add_point(t_id, tid, x_m, y_m)

        csv_builder.add_frame(frame_idx + 1, tracked, team_ids)

        annotated = visualizer.annotate_frame(
            frame=frame, detections=tracked, team_ids=team_ids,
            speed_estimator=speed_estimator, ball_trail=ball_tracker.get_trail(),
            ball_speed_kmh=ball_speed_kmh, ball_is_predicted=ball_predicted,
            frame_idx=frame_idx,
        )
        out.write(annotated)

    # ---- Cleanup ----
    out.release()
    heatmap_analyzer.save_team_heatmap(0, str(output_dir / "team_0_heatmap_500f.png"))
    heatmap_analyzer.save_team_heatmap(1, str(output_dir / "team_1_heatmap_500f.png"))

    tracking_csv_path = output_dir / f"tracking_output_{max_frames}f.csv"
    csv_builder.finalize_and_write(str(tracking_csv_path))

    print(f"\n[DONE] Annotated video  : {out_video_path}")
    print(f"       Tracking CSV      : {tracking_csv_path}")
    print(f"       Heatmaps          : {output_dir}/team_*_heatmap_500f.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_frames", type=int, default=500, help="Number of frames to process")
    p.add_argument("--fps",        type=float, default=25.0, help="Output video FPS")
    main(p.parse_args())
