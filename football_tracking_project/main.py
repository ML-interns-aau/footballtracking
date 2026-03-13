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
from src.pipeline.data_exporter import DataExporter
from src.pipeline.heatmap_analyzer import HeatmapAnalyzer
from src.pipeline.visualizer import PipelineVisualizer

def main(args):
    # Setup Paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_video_path = output_dir / "annotated_football_analysis.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    
    # Read first frame to initialize camera motion
    ret, initial_frame = cap.read()
    if not ret:
        raise ValueError("Failed to read first frame.")
    
    # Initialize Pipeline Modules
    print("Initializing components...")
    detector = FootballDetector(model_path="yolov8m_fixed.pt")
    tracker = FootballTracker()
    team_classifier = TeamClassifier(n_colors=2)
    camera_motion = CameraMotionEstimator(initial_frame)
    data_exporter = DataExporter(output_dir=str(output_dir))
    heatmap_analyzer = HeatmapAnalyzer(pitch_width=105, pitch_height=68)
    visualizer = PipelineVisualizer()
    
    # Define generic pitch mapping source points (trapezoid in image) and destination (meters)
    # These are rough estimates for a standard broadcast angle
    src_pts = [[0, height], [width, height], [width*0.75, height*0.3], [width*0.25, height*0.3]]
    dst_pts = [[0, 68], [105, 68], [105, 0], [0, 0]]
    pitch_mapper = PitchMapper(src_points=src_pts, dst_points=dst_pts)
    
    speed_estimator = SpeedEstimator(fps=fps, pitch_mapper=pitch_mapper)

    # Reset video pointer to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process Frame Loop
    print(f"Processing {total_frames} frames...")
    frame_idx = 0
    pbar = tqdm(total=total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Detect & Track
        detections = detector.detect(frame)
        tracked_detections = tracker.update(detections)
        
        # 2. Assign Teams
        team_ids = team_classifier.assign_teams(frame, tracked_detections)
        
        # 3. Motion & Spatial Calculation
        cam_dx, cam_dy = camera_motion.update(frame)
        
        # Get bounding box centers (bottom-center for persons, exact center for ball)
        pixels = []
        tracker_id_list = []
        ball_pos = (0.0, 0.0)
        
        for bbox, class_id, t_id in zip(tracked_detections.xyxy, tracked_detections.class_id, tracked_detections.tracker_id):
            x1, y1, x2, y2 = bbox
            if class_id == 0: # Person: Bottom center
                cx, cy = (x1 + x2) / 2, y2
                pixels.append([cx, cy])
                tracker_id_list.append(t_id)
            elif class_id == 32: # Ball: Center
                ball_px = ((x1 + x2) / 2, (y1 + y2) / 2)
                ball_pos = pitch_mapper.transform_point(ball_px)
                # Apply camera motion backwards to find static position
                # Actually, the tracker typically handles moving coords, but we don't track the ball ID here
                
        # 4. Update Speed & Distance
        if pixels:
            speed_estimator.estimate_speed(frame_idx, tracker_id_list, np.array(pixels), cam_dx, cam_dy)
            
        # 5. Export Data & Heatmaps per frame
        frame_objs = []
        player_positions_m = {}
        
        for i in range(len(tracked_detections)):
            bbox = tracked_detections.xyxy[i]
            conf = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 1.0
            class_id = int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else -1
            t_id = int(tracked_detections.tracker_id[i]) if tracked_detections.tracker_id is not None else None

            if class_id == 0 and t_id is not None:
                tid = team_ids[i]
                speed, dist, (x_m, y_m) = speed_estimator.get_stats(t_id)
                heatmap_analyzer.add_point(t_id, tid, x_m, y_m)
                player_positions_m[t_id] = (x_m, y_m)
                
                frame_objs.append({
                    "id": t_id,
                    "class": "player",
                    "team": f"Team {tid}" if tid != -1 else "Unknown",
                    "x_m": x_m, "y_m": y_m,
                    "speed": speed, "distance": dist
                })
            elif class_id == 32:
                frame_objs.append({
                    "id": "",
                    "team": "",
                    "class": "ball",
                    "x_m": round(ball_pos[0], 2), "y_m": round(ball_pos[1], 2)
                })
                
        data_exporter.log_frame(frame_idx, frame_objs)
        data_exporter.update_passes(frame_idx, ball_pos, player_positions_m)
        
        # 6. Visualize
        annotated_frame = visualizer.annotate_frame(frame, tracked_detections, team_ids, speed_estimator)
        out.write(annotated_frame)
        
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    
    # Finalize outputs
    data_exporter.finalize()
    heatmap_analyzer.save_team_heatmap(0, str(output_dir / "team_0_heatmap.png"))
    heatmap_analyzer.save_team_heatmap(1, str(output_dir / "team_1_heatmap.png"))
    
    print("\nPipeline execution complete!")
    print(f"Annotated Video: {out_video_path}")
    print(f"Data Exports: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Analytics Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output files")
    args = parser.parse_args()
    main(args)
