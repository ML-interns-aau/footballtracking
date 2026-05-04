from __future__ import annotations

import cv2
import numpy as np

class HomographyTransform:
    """Core mathematics for homography transformations."""
    
    def __init__(self, src_points, dst_points, method=0):
        self.src_pts = np.array(src_points, dtype=np.float32)
        self.dst_pts = np.array(dst_points, dtype=np.float32)
        
        if len(self.src_pts) < 4 or len(self.dst_pts) < 4:
            raise ValueError("At least 4 points are required to compute homography.")
            
        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts, method=method)
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
        else:
            self.H_inv = None

    def pixel_to_pitch(self, points: np.ndarray) -> np.ndarray:
        """Transforms an array of pixel coordinates (N, 2) to pitch coordinates (metres)."""
        if self.H is None or len(points) == 0:
            return np.zeros_like(points)
            
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        dst = cv2.perspectiveTransform(pts, self.H)
        return dst.reshape(-1, 2)

    def pitch_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Transforms an array of pitch coordinates (metres) to pixel coordinates (N, 2)."""
        if self.H_inv is None or len(points) == 0:
            return np.zeros_like(points)
            
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        dst = cv2.perspectiveTransform(pts, self.H_inv)
        return dst.reshape(-1, 2)

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """Transforms a single point (x, y) from pixels to metres."""
        if self.H is None:
            return 0.0, 0.0
            
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, self.H)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def is_valid(self) -> bool:
        """Checks if the homography matrix was successfully computed."""
        return self.H is not None
