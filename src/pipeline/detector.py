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

    def __init__(
        self,
        model_path: str = "yolov8m_fixed.pt",
        conf: float = 0.30,
        iou: float = 0.40,
        imgsz: int = 960,
        device=None,
    ):
        self.model = YOLO(model_path)
        self.CLASS_NAMES_DICT = self.model.model.names
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run inference and return all detections (persons + ball)."""
        kwargs = dict(
            classes=[0, 32],  # 0=person, 32=sports ball
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            agnostic_nms=True,
            verbose=False,
        )
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model(frame, **kwargs)[0]
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
