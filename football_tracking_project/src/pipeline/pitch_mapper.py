import cv2
import numpy as np

class PitchMapper:
    def __init__(self, src_points: list[list[float]], dst_points: list[list[float]]):
        """Initialize the pitch mapper with source pixels and destination coordinates (meters).
        
        Args:
            src_points: List of 4 [x,y] points in the video frame (pixels)
            dst_points: List of corresponding 4 [x,y] points on the pitch map (meters)
        """
        self.src_pts = np.array(src_points, dtype=np.float32)
        self.dst_pts = np.array(dst_points, dtype=np.float32)
        
        
        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """Convert a single (x, y) pixel coordinate to (x, y) meter coordinate.
        
        Args:
            point: (x, y) in pixels
            
        Returns:
            (x, y) real world coordinates in meters. Returns (0, 0) if invalid.
        """
        if self.H is None:
            return 0.0, 0.0
            
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, self.H)
        return float(dst[0][0][0]), float(dst[0][0][1])
        
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Convert multiple pixel points to meter coordinates.
        
        Args:
            points: Array of shape (N, 2) with pixel coords
            
        Returns:
            Array of shape (N, 2) with meter coords
        """
        if self.H is None or len(points) == 0:
            return np.zeros_like(points)
            
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        dst = cv2.perspectiveTransform(pts, self.H)
        return dst.reshape(-1, 2)
