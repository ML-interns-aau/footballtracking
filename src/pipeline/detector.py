import supervision as sv
from ultralytics import YOLO
import numpy as np
import pathlib
import platform

# Fix for loading models saved on Linux (PosixPath) on Windows
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath


class FootballDetector:
    """
    YOLO-based detector with explicit device selection.
    - Runs at imgsz=960 for a good speed/accuracy balance
    - Lower conf threshold (0.30) reduces missed player detections
    - Tighter IOU (0.40) removes duplicate boxes in crowded scenes
    - Accepts a `device` argument so GPU is used when available
    """

    def __init__(
        self,
        model_path: str = "yolov8m_fixed.pt",
        conf: float = 0.30,
        iou: float = 0.40,
        device: str = "cpu",
    ):
        self.model = YOLO(model_path)
        # Move model to the requested device (cuda / cpu)
        self.model.to(device)
        self.CLASS_NAMES_DICT = self.model.model.names
        self.conf   = conf
        self.iou    = iou
        self.device = device

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run inference and return all detections (persons + ball)."""
        results = self.model(
            frame,
            classes=[0, 32],    # 0=person, 32=sports ball
            conf=self.conf,
            iou=self.iou,
            imgsz=960,
            agnostic_nms=True,
            verbose=False,
            device=self.device,
        )[0]
        return sv.Detections.from_ultralytics(results)

    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        """Return only person detections."""
        detections = self.detect(frame)
        return detections[detections.class_id == 0]

    def detect_ball(self, frame: np.ndarray) -> sv.Detections:
        """Return only ball detections."""
        detections = self.detect(frame)
        return detections[detections.class_id == 32]
