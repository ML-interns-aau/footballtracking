import supervision as sv
from ultralytics import YOLO
import numpy as np

class FootballDetector:
    def __init__(self, model_path: str = "yolov8m.pt"):
        """Initialize the YOLOv8 model for football detection.
        
        Args:
            model_path: Path to the YOLOv8 weights (e.g. 'yolov8m.pt')
        """
        self.model = YOLO(model_path)
        # Assuming YOLOv8 COCO model: 
        # 0: person, 32: sports ball. 
        # We will map 'person' to generic player for now until team classification refines it.
        self.CLASS_NAMES_DICT = self.model.model.names
        
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run inference on a single frame and return supervision Detections.
        
        Args:
            frame: BGR numpy array image (from cv2.VideoCapture)
            
        Returns:
            sv.Detections object containing person and ball detections.
        """
        results = self.model(frame, classes=[0, 32], verbose=False)[0]
        
        # Convert YOLO results to supervision Detections natively
        detections = sv.Detections.from_ultralytics(results)
        
        # Optional: Further filter confidences or specific bounding box sizes here if needed.
        return detections
