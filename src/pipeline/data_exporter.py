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
            
        self.last_ball_possessor = None
        self.passes = []

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

    def update_passes(self, frame_idx: int, ball_pos: tuple[float, float], player_positions: dict[int, tuple[float, float]]):
        if ball_pos == (0.0, 0.0):
            return
            
        closest_dist = float('inf')
        closest_player_id = None
        
        for p_id, p_pos in player_positions.items():
            dist = ((ball_pos[0] - p_pos[0])**2 + (ball_pos[1] - p_pos[1])**2)**0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_player_id = p_id
                
        possession_radius = 1.0
        
        if closest_dist <= possession_radius:
            current_possessor = closest_player_id
            
            if self.last_ball_possessor is not None:
                last_pid, last_frame = self.last_ball_possessor
                
                if current_possessor != last_pid and (frame_idx - last_frame) > 3:
                    pass_record = {
                        "passer_id": last_pid,
                        "receiver_id": current_possessor,
                        "start_frame": last_frame,
                        "end_frame": frame_idx
                    }
                    self.passes.append(pass_record)
                    print(f"[EVENT] Pass detected: Player {last_pid} -> Player {current_possessor} (completed at frame {frame_idx})")
                    
            self.last_ball_possessor = (current_possessor, frame_idx)

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
