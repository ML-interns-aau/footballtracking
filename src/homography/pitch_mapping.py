from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .homography_transform import HomographyTransform

class PitchMapping:
    """High-level wrapper for pitch mapping, handles configuration and validation."""
    
    def __init__(self, src_points, dst_points, pitch_width_m=105.0, pitch_height_m=68.0):
        self.pitch_width_m = float(pitch_width_m)
        self.pitch_height_m = float(pitch_height_m)
        self.transform = HomographyTransform(src_points, dst_points)

    @classmethod
    def from_config(cls, config_path: str = "configs/homography.json") -> PitchMapping:
        """Loads homography point pairs from a JSON config."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Homography config not found: {config_path}")
            
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        return cls(
            src_points=cfg["src_points"],
            dst_points=cfg["dst_points"],
            pitch_width_m=cfg.get("pitch_width_m", 105.0),
            pitch_height_m=cfg.get("pitch_height_m", 68.0)
        )

    def save_config(self, config_path: str) -> None:
        """Saves current point pairs to a JSON config."""
        cfg = {
            "src_points": self.transform.src_pts.tolist(),
            "dst_points": self.transform.dst_pts.tolist(),
            "pitch_width_m": self.pitch_width_m,
            "pitch_height_m": self.pitch_height_m
        }
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def scale_src_points(self, scale_x: float, scale_y: float) -> None:
        """Recompute homography if the video resolution has been scaled."""
        new_src = self.transform.src_pts * np.array([scale_x, scale_y], dtype=np.float32)
        self.transform = HomographyTransform(new_src, self.transform.dst_pts)
    
    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """Transform a single point from pixels to pitch coordinates (meters)."""
        return self.transform.transform_point(point)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Wrapper for batch point transformation."""
        return self.transform.pixel_to_pitch(points)

    def is_valid_point(self, x_m: float, y_m: float) -> bool:
        """Check if a transformed point falls within the physical pitch bounds."""
        return 0 <= x_m <= self.pitch_width_m and 0 <= y_m <= self.pitch_height_m
