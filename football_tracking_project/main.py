import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

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


def main(args):
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = output_dir / "annotated_football_analysis.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    ret, initial_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read first frame.")

    detector       = FootballDetector(model_path="yolov8m_fixed.pt", conf=0.30, iou=0.40)
    tracker        = FootballTracker(track_thresh=0.20, track_buffer=60, match_thresh=0.80)
    team_classifier = TeamClassifier(n_teams=2, history_len=15, refit_interval=150)
    ball_tracker   = BallTracker(max_trail=25, max_missed=30)
    camera_motion  = CameraMotionEstimator(initial_frame)
    data_exporter  = DataExporter(output_dir=str(output_dir))
    heatmap_analyzer = HeatmapAnalyzer(pitch_width=105, pitch_height=68)
    visualizer     = PipelineVisualizer()

    src_pts = [[0, height], [width, height], [width * 0.75, height * 0.3], [width * 0.25, height * 0.3]]
    dst_pts = [[0, 68],     [105, 68],       [105, 0],                       [0, 0]]
    pitch_mapper    = PitchMapper(src_points=src_pts, dst_points=dst_pts)
    speed_estimator = SpeedEstimator(fps=fps, pitch_mapper=pitch_mapper, window_size=8)
    csv_builder     = TrackingCSVBuilder(pitch_mapper=pitch_mapper, fps=fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    frame_idx = 0
    pbar = tqdm(total=max_frames, unit="frame")

    while cap.isOpened():
        if frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        team_ids = team_classifier.assign_teams(frame, tracked)
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
                frame_idx, player_tracker_ids,
                np.array(player_pixels), cam_dx, cam_dy
            )

        ball_pos_m = pitch_mapper.transform_point((ball_cx, ball_cy))
        speed_estimator.estimate_speed(
            frame_idx,
            [BallTracker.BALL_ID],
            np.array([[ball_cx, ball_cy]]),
            cam_dx, cam_dy
        )
        ball_speed_kmh, _, _ = speed_estimator.get_stats(BallTracker.BALL_ID)

        frame_objs       = []
        player_positions = {}

        for i in range(len(tracked)):
            bbox      = tracked.xyxy[i]
            class_id  = int(tracked.class_id[i])  if tracked.class_id  is not None else -1
            t_id      = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
            tid       = int(team_ids[i])

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
                    "id":    "",   "team":  "",   "class": "ball",
                    "x_m":  round(ball_pos_m[0], 2),
                    "y_m":  round(ball_pos_m[1], 2),
                    "speed": round(ball_speed_kmh, 2),
                })

        data_exporter.log_frame(frame_idx, frame_objs)
        data_exporter.update_passes(frame_idx, ball_pos_m, player_positions)
        csv_builder.add_frame(frame_idx, tracked, team_ids)

        annotated = visualizer.annotate_frame(
            frame          = frame,
            detections     = tracked,
            team_ids       = team_ids,
            speed_estimator= speed_estimator,
            ball_trail     = ball_tracker.get_trail(),
            ball_speed_kmh = ball_speed_kmh,
            ball_is_predicted = ball_predicted,
            frame_idx      = frame_idx,
        )
        out.write(annotated)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    data_exporter.finalize()
    heatmap_analyzer.save_team_heatmap(0, str(output_dir / "team_0_heatmap.png"))
    heatmap_analyzer.save_team_heatmap(1, str(output_dir / "team_1_heatmap.png"))
    
    tracking_csv_path = output_dir / "tracking_output.csv"
    csv_builder.finalize_and_write(tracking_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Analytics Pipeline")
    parser.add_argument("--input",      type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()
    main(args)
