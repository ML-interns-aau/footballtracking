import numpy as np
from collections import defaultdict
from src.pipeline.pitch_mapper import PitchMapper

class SpeedEstimator:
    def __init__(self, fps: float, pitch_mapper: PitchMapper, window_size: int = 5):
        self.fps = fps
        self.pitch_mapper = pitch_mapper
        self.window_size = window_size
        
        self.history = defaultdict(list)
        
        self.speeds = {}
        self.distances = defaultdict(float)

    def estimate_speed(self, frame_idx: int, tracker_ids: list[int], points: np.ndarray, cam_dx: float, cam_dy: float):
        compensated_pts = np.copy(points)
        compensated_pts[:, 0] += cam_dx
        compensated_pts[:, 1] += cam_dy
        
        meter_pts = self.pitch_mapper.transform_points(compensated_pts)
        
        for idx, tracker_id in enumerate(tracker_ids):
            if tracker_id is None:
                continue
                
            x_m, y_m = meter_pts[idx]
            player_history = self.history[tracker_id]
            
            if len(player_history) > 0:
                last_frame, last_x, last_y = player_history[-1]
                
                step_dist = np.sqrt((x_m - last_x)**2 + (y_m - last_y)**2)
                
                frame_gap = frame_idx - last_frame
                if frame_gap > 0 and (step_dist / frame_gap) < 2.0:
                     self.distances[tracker_id] += step_dist
            
            player_history.append((frame_idx, x_m, y_m))
            
            if len(player_history) > self.window_size:
                player_history.pop(0)
                
            if len(player_history) >= 2:
                old_frame, old_x, old_y = player_history[0]
                frames_elapsed = frame_idx - old_frame
                
                if frames_elapsed > 0:
                    dist_moved = np.sqrt((x_m - old_x)**2 + (y_m - old_y)**2)
                    time_elapsed = frames_elapsed / self.fps
                    speed_ms = dist_moved / time_elapsed
                    speed_kmh = speed_ms * 3.6
                    self.speeds[tracker_id] = round(speed_kmh, 2)
            else:
                self.speeds[tracker_id] = 0.0
                
    def get_stats(self, tracker_id: int) -> tuple[float, float, tuple[float, float]]:
        speed = self.speeds.get(tracker_id, 0.0)
        dist = round(self.distances[tracker_id], 2)
        history = self.history.get(tracker_id, [])
        coord = (round(history[-1][1], 2), round(history[-1][2], 2)) if history else (0.0, 0.0)
        
        return speed, dist, coord
