import cv2
import numpy as np
import json
from pathlib import Path

class PitchMapper:
    def __init__(self, src_points: list[list[float]], dst_points: list[list[float]]):
        self.src_pts = np.array(src_points, dtype=np.float32)
        self.dst_pts = np.array(dst_points, dtype=np.float32)
        
        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)

    @classmethod
    def from_config(cls, video_path: str, config_path: str = "data/calibrations.json", default_src=None, default_dst=None):
        v_name = Path(video_path).name
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = json.load(f)
                
            if v_name in data:
                print(f"[PitchMapper] Loaded calibration for {v_name}")
                return cls(data[v_name]["src_pts"], data[v_name]["dst_pts"])
            elif "default" in data:
                print(f"[PitchMapper] Warning: No specific calibration for {v_name}, using default from config.")
                return cls(data["default"]["src_pts"], data["default"]["dst_pts"])
                
        print(f"[PitchMapper] Warning: Config {config_path} not found or no default. Using fallback.")
        return cls(default_src, default_dst)

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        if self.H is None:
            return 0.0, 0.0
            
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, self.H)
        return float(dst[0][0][0]), float(dst[0][0][1])
        
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if self.H is None or len(points) == 0:
            return np.zeros_like(points)
            
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        dst = cv2.perspectiveTransform(pts, self.H)
        return dst.reshape(-1, 2)
