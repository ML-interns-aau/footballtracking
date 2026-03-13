import csv
import json
from pathlib import Path

class DataExporter:
    def __init__(self, output_dir: str):
        """Initialize the data exporter for frame-by-frame stats and pass detection."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.output_dir / "analytics.csv"
        self.json_path = self.output_dir / "analytics.json"
        
        self.frame_data = [] # List of dicts for JSON
        
        # Initialize CSV
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "object_id", "class", "team", "x_m", "y_m", "speed_kmh", "distance_m"])
            
        # Pass detection state
        self.last_ball_possessor = None # (player_id, frame_id)
        self.passes = [] # List of completed passes

    def log_frame(self, frame_idx: int, frame_objects: list[dict]):
        """Log all objects detected and tracked in a single frame.
        
        Args:
            frame_idx: The current frame number
            frame_objects: List of dicts containing keys: 
                id, class, team, x_m, y_m, speed, distance
        """
        # Append to JSON structure
        self.frame_data.append({
            "frame": frame_idx,
            "objects": frame_objects
        })
        
        # Append to CSV
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
        """Detect possession and passes.
        
        Args:
            frame_idx: Current frame
            ball_pos: (x_m, y_m) coordinates of the ball
            player_positions: Dict mapping player_id -> (x_m, y_m)
        """
        if ball_pos == (0.0, 0.0):
            return # Ball not visible or invalid coords
            
        # Find closest player to the ball
        closest_dist = float('inf')
        closest_player_id = None
        
        for p_id, p_pos in player_positions.items():
            dist = ((ball_pos[0] - p_pos[0])**2 + (ball_pos[1] - p_pos[1])**2)**0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_player_id = p_id
                
        # Assume possession if distance is < 1.0 meter
        possession_radius = 1.0 # meters
        
        if closest_dist <= possession_radius:
            current_possessor = closest_player_id
            
            if self.last_ball_possessor is not None:
                last_pid, last_frame = self.last_ball_possessor
                
                # Check if possession changed and at least 3 frames passed (to avoid instant flickers)
                if current_possessor != last_pid and (frame_idx - last_frame) > 3:
                    # A pass occurred!
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
        """Save the accumulated JSON file including final pass list."""
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
        
        # Deep convert any numpy types to python primitives
        final_data = convert_numpy(final_data)

        with open(self.json_path, 'w') as f:
            json.dump(final_data, f, indent=2)
            
        print(f"Data exporter finalized. Saved {self.csv_path} and {self.json_path}")
