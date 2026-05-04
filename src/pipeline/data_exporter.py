import csv
import json
from pathlib import Path

class DataExporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.output_dir / "analytics.csv"
        self.json_path = self.output_dir / "analytics.json"
        
        self.frame_data = []
        
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "object_id", "class", "team", "x_m", "y_m", "speed_kmh", "distance_m"])
            
        self.current_possessor_id = None
        self.contact_frames = 0
        self.loose_ball_frames = 0
        self.last_ball_possessor = None # (player_id, frame_idx)
        self.passes = []

        # Thresholds
        self.POSSESSION_RADIUS = 1.5   # meters
        self.MIN_POSSESSION_FRAMES = 5 # smoothing
        self.MIN_LOOSE_FRAMES = 15     # before declaring loose
        self.PASS_SPEED_THRESHOLD = 20 # km/h

    def log_frame(self, frame_idx: int, frame_objects: list[dict]):
        self.frame_data.append({
            "frame": frame_idx,
            "objects": frame_objects
        })
        
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            for obj in frame_objects:
                writer.writerow([
                    frame_idx,
                    obj.get("id", ""),
                    obj.get("class", "unknown"),
                    obj.get("team", ""),
                    round(obj.get("x_m", 0.0), 2),
                    round(obj.get("y_m", 0.0), 2),
                    round(obj.get("speed", 0.0), 2) if "speed" in obj else "",
                    round(obj.get("distance", 0.0), 2) if "distance" in obj else ""
                ])

    def update_passes(self, frame_idx: int, ball_pos: tuple[float, float], player_positions: dict[int, tuple[float, float]], ball_speed_kmh: float = 0.0):
        if ball_pos == (0.0, 0.0):
            return
            
        closest_dist = float('inf')
        closest_player_id = None
        
        for p_id, p_pos in player_positions.items():
            dist = ((ball_pos[0] - p_pos[0])**2 + (ball_pos[1] - p_pos[1])**2)**0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_player_id = p_id
        
        # 1. Logic for "Potential Touch"
        in_range = closest_player_id is not None and closest_dist <= self.POSSESSION_RADIUS

        # 2. Update Possession State
        if in_range:
            if closest_player_id == self.current_possessor_id:
                # Still has it
                self.contact_frames += 1
                self.loose_ball_frames = 0
            else:
                # Might be a change or new gain
                self.contact_frames += 1
                if self.contact_frames >= self.MIN_POSSESSION_FRAMES:
                    # VALID CHANGE OF POSSESSION
                    if self.current_possessor_id is not None:
                        # Detect if this was a pass
                        if ball_speed_kmh > self.PASS_SPEED_THRESHOLD:
                            pass_record = {
                                "passer_id": self.current_possessor_id,
                                "receiver_id": closest_player_id,
                                "start_frame": self.last_ball_possessor[1] if self.last_ball_possessor else 0,
                                "end_frame": frame_idx,
                                "ball_speed": round(ball_speed_kmh, 2)
                            }
                            self.passes.append(pass_record)
                            print(f"[EVENT] Pass detected: Player {self.current_possessor_id} -> Player {closest_player_id} (Speed: {ball_speed_kmh:.1f} km/h)")
                    
                    self.current_possessor_id = closest_player_id
                    self.last_ball_possessor = (closest_player_id, frame_idx)
                    self.contact_frames = 0
                    self.loose_ball_frames = 0
        else:
            # Ball is away from everyone
            self.contact_frames = 0
            if self.current_possessor_id is not None:
                self.loose_ball_frames += 1
                # Lose possession if away for too long or kicked hard
                if self.loose_ball_frames > self.MIN_LOOSE_FRAMES or ball_speed_kmh > self.PASS_SPEED_THRESHOLD:
                    self.current_possessor_id = None
                    self.loose_ball_frames = 0

    def finalize(self):
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        final_data = {
            "metadata": {"total_passes": len(self.passes)},
            "passes": self.passes,
            "frames": self.frame_data
        }
        
        final_data = convert_numpy(final_data)

        with open(self.json_path, 'w') as f:
            json.dump(final_data, f, indent=2)
            
        print(f"Data exporter finalized. Saved {self.csv_path} and {self.json_path}")
