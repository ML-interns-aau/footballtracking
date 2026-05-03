import supervision as sv
from ultralytics import YOLO
import numpy as np


class FootballDetector:
    """
    Improved YOLO-based detector.
    - Runs at imgsz=1280 for better small-object (ball) recall
    - Lower conf threshold (0.30) reduces missed player detections
    - Tighter IOU (0.40) removes duplicate boxes in crowded scenes
    - Returns player and ball detections separately for clean downstream use
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.30, iou: float = 0.40):
        self.model = YOLO(model_path)
        self.CLASS_NAMES_DICT = self.model.model.names
        self.conf = conf
        self.iou = iou

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run inference and return all detections (persons + ball)."""
        results = self.model(
            frame,
            classes=[0, 32],        # 0=person, 32=sports ball
            conf=self.conf,
            iou=self.iou,
            imgsz=960,              # Higher resolution → better small-object recall (960 balances speed & accuracy)
            agnostic_nms=True,      # Merge overlapping boxes across classes
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        """Return only person detections."""
        detections = self.detect(frame)
        return detections[detections.class_id == 0]

    def detect_ball(self, frame: np.ndarray) -> sv.Detections:
        """Return only ball detections."""
        detections = self.detect(frame)
        return detections[detections.class_id == 32]
