import cv2
import numpy as np

class PitchMapper:
    def __init__(self, src_points: list[list[float]], dst_points: list[list[float]]):
        self.src_pts = np.array(src_points, dtype=np.float32)
        self.dst_pts = np.array(dst_points, dtype=np.float32)
        
        
        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)

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
