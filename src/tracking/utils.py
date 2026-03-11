import cv2
import numpy as np
from pathlib import Path


class VideoUtils:
    @staticmethod
    def get_writer(cap, output_path):
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        return cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    @staticmethod
    def draw_custom_ui(frame, centers):
        """Draws a simple circle at the 'feet' of tracked objects."""
        for item in centers:
            pos = tuple(map(int, item["pos"]))
            cv2.circle(frame, pos, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID: {item['id']}", (pos[0], pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


class Annotators:
    @staticmethod
    def draw_ellipse(frame, center, axes, color, thickness=2):
        cv2.ellipse(frame, center, axes, 0, 0, 360, color, thickness)

    @staticmethod
    def draw_triangle(frame, center, size, color, thickness=2):
        x, y = center
        pts = np.array([
            (x, y - size),
            (x - size, y + size),
            (x + size, y + size),
        ], dtype=np.int32)
        cv2.polylines(frame, [pts], True, color, thickness)

    @staticmethod
    def draw_edge_box(frame, box, color, edge=12, thickness=2):
        x1, y1, x2, y2 = box
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + edge, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + edge), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - edge, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + edge), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + edge, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - edge), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - edge, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - edge), color, thickness)

    @staticmethod
    def draw_vertex(frame, point, color=(255, 255, 255), radius=3):
        cv2.circle(frame, point, radius, color, -1)
