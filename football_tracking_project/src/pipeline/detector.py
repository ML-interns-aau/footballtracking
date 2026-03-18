import supervision as sv
from ultralytics import YOLO
import numpy as np
import sys
class FootballDetector:
    def __init__(self, model_path: str = "yolov8m.pt"):
        self.model = YOLO(model_path)
        self.CLASS_NAMES_DICT = self.model.model.names
        
    def detect(self, frame: np.ndarray) -> sv.Detections:
        results = self.model(frame, classes=[0, 32], verbose=False)[0]
        
        detections = sv.Detections.from_ultralytics(results)
        
        return detections
