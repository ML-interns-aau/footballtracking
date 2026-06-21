from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from app.config import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_IMGSZ,
    DEFAULT_RESIZE_W,
    DEFAULT_TARGET_FPS,
    MODEL_PATH,
)
from src.config import CONFIG

from src.homography.pitch_mapping import PitchMapping
from src.engine.ball_tracker import BallTracker
from src.analytics.camera_motion import CameraMotionEstimator
from src.exporters.data_exporter import DataExporter
from src.engine.detector import FootballDetector
from src.analytics.events import EventsDetector
from src.analytics.heatmap_analyzer import HeatmapAnalyzer
from src.exporters.output_schema import OutputFiles, OutputPathResolver
from src.exporters.player_summary_csv_builder import PlayerSummaryCSVBuilder
from src.exporters.possession_summary_csv_builder import PossessionSummaryCSVBuilder
from src.analytics.speed_estimator import SpeedEstimator
from src.engine.team_classifier import TeamClassifier
from src.engine.tracker import FootballTracker
from src.exporters.tracking_csv_builder import TrackingCSVBuilder
from src.visualization.visualizer import PipelineVisualizer
from src.preprocessing.resolution_normalization import resize_frame


def _get_device() -> str | int:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class FootballPipelineRunner:
    """
    Encapsulates the end-to-end execution of the football analytics pipeline.
    """
    def __init__(self, args, progress_callback=None):
        self.args = args
        self.progress_callback = progress_callback

    def run(self) -> dict:
        args = self.args
        progress_callback = self.progress_callback

        try:
            CONFIG.validate(raise_on_error=True)
            if CONFIG.get_environment() != "default":
                print(f"[CONFIG] Environment: {CONFIG.get_environment()}", flush=True)
                print(f"[CONFIG] Loaded from: {CONFIG.config_path}", flush=True)
        except ValueError as e:
            print(f"[CONFIG ERROR] {e}", flush=True)
            raise

        device     = _get_device()
        input_path = Path(args.input)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        game_id         = getattr(args, "game_id", None)
        resolver        = OutputPathResolver(output_dir, None)
        game_output_dir = output_dir

        target_fps   = float(getattr(args, "target_fps",   DEFAULT_TARGET_FPS) or 0)
        resize_width = int(  getattr(args, "resize_width", DEFAULT_RESIZE_W)   or 0)
        conf         = float(getattr(args, "conf",         DEFAULT_CONF))
        iou          = float(getattr(args, "iou",          DEFAULT_IOU))
        imgsz        = int(  getattr(args, "imgsz",        DEFAULT_IMGSZ))
        device       =       getattr(args, "device",       device)
        model_path   =       getattr(args, "model_path",   MODEL_PATH)

        if isinstance(device, str) and device.isdigit():
            device = int(device)

        # ── Open video ────────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if resize_width > 0:
            scale  = resize_width / float(width)
            width  = resize_width
            height = max(1, int(round(height * scale)))

        frame_step    = 1
        effective_fps = fps
        if target_fps > 0 and fps > 0:
            frame_step    = max(1, round(fps / target_fps))
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

        # ── Video writer ──────────────────────────────────────────────────────────
        out_video_path = resolver.annotated_video()
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        ret, initial_frame = cap.read()
        if not ret:
            raise ValueError("Failed to read first frame.")

        if not Path(model_path).exists():
            from dashboard.config import MODEL_PATH as DASHBOARD_MODEL_PATH
            model_path = DASHBOARD_MODEL_PATH

        # ── Component initialisation ──────────────────────────────────────────────
        detector = FootballDetector(
            model_path=model_path,
            conf=CONFIG.get("detection.confidence_threshold", 0.30),
            iou=CONFIG.get("detection.iou_threshold", 0.40),
            device=device,
        )
        tracker = FootballTracker(
            track_thresh=CONFIG.get("tracking.track_threshold", 0.20),
            track_buffer=CONFIG.get("tracking.track_buffer", 60),
            match_thresh=CONFIG.get("tracking.match_threshold", 0.80),
        )
        team_classifier = TeamClassifier(
            n_teams=CONFIG.get("team_classification.n_teams", 2),
            history_len=CONFIG.get("team_classification.history_length", 15),
            refit_interval=CONFIG.get("team_classification.refit_interval", 150),
        )
        ball_tracker = BallTracker(
            max_trail=CONFIG.get("ball.max_trail_length", 25),
            max_missed=CONFIG.get("ball.max_missed_frames", 30),
        )
        camera_motion  = CameraMotionEstimator(initial_frame)
        data_exporter  = DataExporter(output_dir=str(game_output_dir))

        match_info = {
            "match_id":       getattr(args, "match_id",   "match_001") or "match_001",
            "home_team":      getattr(args, "home_team",  "Home Team") or "Home Team",
            "away_team":      getattr(args, "away_team",  "Away Team") or "Away Team",
            "match_start_ms": 0,
            "period":         1,
            "team_id_map":    {0: "home", 1: "away"},
        }
        data_exporter.set_match_info(match_info)

        heatmap_analyzer = HeatmapAnalyzer(
            pitch_width=CONFIG.get("pitch.width_m", 105),
            pitch_height=CONFIG.get("pitch.height_m", 68),
        )
        visualizer      = PipelineVisualizer()
        events_detector = EventsDetector(
            fps=effective_fps,
            pitch_width_m=CONFIG.get("pitch.width_m", 105),
            pitch_height_m=CONFIG.get("pitch.height_m", 68),
        )

        src_pts = [
            [0,            height],
            [width,        height],
            [width * 0.75, height * 0.3],
            [width * 0.25, height * 0.3],
        ]
        dst_pts = [[0, 68], [105, 68], [105, 0], [0, 0]]

        try:
            config_path = Path("configs/homography.json")
            pitch_mapper = (
                PitchMapping.from_config(str(config_path))
                if config_path.exists()
                else PitchMapping(src_pts, dst_pts)
            )
        except Exception:
            pitch_mapper = PitchMapping(src_pts, dst_pts)

        speed_estimator = SpeedEstimator(
            fps=effective_fps,
            pitch_mapper=pitch_mapper,
            window_size=CONFIG.get("speed.window_size", 8),
        )

        try:
            data_exporter.set_fps(effective_fps)
        except Exception:
            pass

        csv_builder        = TrackingCSVBuilder(pitch_mapper=pitch_mapper, fps=effective_fps)
        player_builder     = PlayerSummaryCSVBuilder()
        possession_builder = PossessionSummaryCSVBuilder()
        print("[PHASE] tracking and export components initialized", flush=True)

        # ── Frame loop ────────────────────────────────────────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        source_frame_idx    = 0
        processed_frame_idx = 0
        pbar = tqdm(total=planned_frames, unit="frame")

        while cap.isOpened():
            if planned_frames > 0 and processed_frame_idx >= planned_frames:
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

            frame      = resize_frame(frame, resize_width)
            detections = detector.detect(frame)
            tracked    = tracker.update(detections)
            team_ids   = team_classifier.assign_teams(frame, tracked)

            cam_dx, cam_dy = camera_motion.update(frame)

            ball_mask       = tracked.class_id == 32
            ball_detections = tracked[ball_mask] if np.any(ball_mask) else None
            ball_cx, ball_cy, ball_predicted = ball_tracker.update(frame, ball_detections)

            player_pixels:      list = []
            player_tracker_ids: list = []
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

            frame_objs:       list[dict] = []
            player_positions: dict       = {}

            for i in range(len(tracked)):
                bbox     = tracked.xyxy[i]
                class_id = int(tracked.class_id[i])   if tracked.class_id   is not None else -1
                t_id     = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
                tid      = int(team_ids[i])

                if class_id == 0 and t_id is not None:
                    speed, dist, (x_m, y_m) = speed_estimator.get_stats(t_id)
                    heatmap_analyzer.add_point(t_id, tid, x_m, y_m)
                    player_positions[t_id] = (x_m, y_m)
                    dist_to_ball = ((x_m - ball_pos_m[0])**2 + (y_m - ball_pos_m[1])**2)**0.5
                    possession = dist_to_ball < CONFIG.get("possession.max_distance_m", 2.0)

                    frame_objs.append({
                        "id":       t_id,
                        "class":    "referee" if tid == -2 else "player",
                        "team":     "Referee" if tid == -2 else (f"Team {tid}" if tid >= 0 else "Unknown"),
                        "x_m":      x_m,   "y_m":      y_m,
                        "speed":    speed,  "distance": dist,
                        "possession": possession,
                    })
                elif class_id == 32:
                    frame_objs.append({
                        "id":    "", "team": "", "class": "ball",
                        "x_m":   round(ball_pos_m[0], 2),
                        "y_m":   round(ball_pos_m[1], 2),
                        "speed": round(ball_speed_kmh, 2),
                    })

            data_exporter.log_frame(processed_frame_idx, frame_objs)

            player_teams_map: dict[int, str] = {
                obj["id"]: obj["team"]
                for obj in frame_objs
                if obj.get("class") in ("player", "referee") and obj.get("id") is not None
            }
            events_detector.process_frame(
                frame_idx=processed_frame_idx,
                ball_pos_m=ball_pos_m,
                player_positions=player_positions,
                player_teams=player_teams_map,
                ball_speed_kmh=ball_speed_kmh,
                data_exporter=data_exporter,
                ball_trail=ball_tracker.get_trail(),
            )

            csv_builder.add_frame(processed_frame_idx, tracked, team_ids)

            tracked_objects: list[dict] = []
            for i in range(len(tracked)):
                bbox     = tracked.xyxy[i]
                class_id = int(tracked.class_id[i])   if tracked.class_id   is not None else -1
                t_id     = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                team_id  = int(team_ids[i])
                speed_km_h, distance_m, _ = (
                    speed_estimator.get_stats(t_id) if t_id > 0 else (0.0, 0.0, (0.0, 0.0))
                )
                possession = False
                if class_id == 0 and t_id > 0:
                    player_pos   = pitch_mapper.transform_point(((bbox[0] + bbox[2]) / 2, bbox[3]))
                    dist_to_ball = (
                        (player_pos[0] - ball_pos_m[0]) ** 2
                        + (player_pos[1] - ball_pos_m[1]) ** 2
                    ) ** 0.5
                    possession = dist_to_ball < CONFIG.get("possession.max_distance_m", 2.0)

                tracked_objects.append({
                    "tracker_id": t_id,
                    "team_id":    team_id,
                    "class_id":   class_id,
                    "speed_km_h": speed_km_h,
                    "distance_m": distance_m,
                    "possession": possession,
                    "team":       f"Team {team_id}" if team_id >= 0 else "Unknown",
                    "role":       "player" if class_id == 0 else ("referee" if class_id == 3 else "unknown"),
                })

            player_builder.add_frame(processed_frame_idx, tracked_objects)
            possession_builder.add_frame(processed_frame_idx, tracked_objects)

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
            source_frame_idx    += 1
            pbar.update(1)

            if progress_callback is not None:
                progress_callback(processed_frame_idx, planned_frames)

        pbar.close()
        cap.release()
        out.release()

        # ── Finalise outputs ──────────────────────────────────────────────────────
        data_exporter.finalize()
        heatmap_analyzer.save_team_heatmap(0, output_path=str(resolver.heatmap_path(0)))
        heatmap_analyzer.save_team_heatmap(1, output_path=str(resolver.heatmap_path(1)))

        tracking_csv_path       = resolver.tracking_csv()
        player_summary_path     = resolver.player_summary_csv()
        possession_summary_path = resolver.possession_summary_csv()

        csv_builder.finalize_and_write(tracking_csv_path)
        player_builder.finalize_and_write(player_summary_path)
        possession_builder.finalize_and_write(possession_summary_path)

        print(f"[UNIFIED OUTPUTS] Generated all files in {game_output_dir}:")
        print(f"  - {OutputFiles.TRACKING} ({len(tracking_csv_path.read_text().splitlines()) if tracking_csv_path.exists() else 0} lines)")
        print(f"  - {OutputFiles.PLAYER_SUMMARY} ({len(player_summary_path.read_text().splitlines()) if player_summary_path.exists() else 0} lines)")
        print(f"  - {OutputFiles.POSSESSION_SUMMARY} ({len(possession_summary_path.read_text().splitlines()) if possession_summary_path.exists() else 0} lines)")

        analytics_json_path = resolver.analytics_json()
        print(f"  - {OutputFiles.ANALYTICS_JSON} ({len(analytics_json_path.read_text().splitlines()) if analytics_json_path.exists() else 0} lines)")
        print(f"  - {OutputFiles.HEATMAP_TEAM_0} ({'exists' if resolver.heatmap_path(0).exists() else 'missing'})")
        print(f"  - {OutputFiles.HEATMAP_TEAM_1} ({'exists' if resolver.heatmap_path(1).exists() else 'missing'})")

        return {
            "total_frames": processed_frame_idx,
            "fps":          fps,
            "resolution":   f"{width}x{height}",
            "output_video": str(out_video_path),
            "game_id":      game_id,
            "game_folder":  str(game_output_dir),
        }
