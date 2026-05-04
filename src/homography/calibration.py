import cv2
import numpy as np
from pathlib import Path
from .pitch_mapping import PitchMapping

class HomographyCalibrator:
    """Interactive tool to click 4 points on a pitch to calibrate homography."""
    
    def __init__(self, image_path_or_frame, config_path="configs/homography.json"):
        if isinstance(image_path_or_frame, (str, Path)):
            self.frame = cv2.imread(str(image_path_or_frame))
        else:
            self.frame = image_path_or_frame.copy()
            
        self.config_path = config_path
        self.src_points = []
        self.dst_points = [[0, 68], [105, 68], [105, 0], [0, 0]] # Default FIFA corners
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.src_points) < 4:
                self.src_points.append([x, y])
                cv2.circle(self.frame_copy, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Calibration", self.frame_copy)

    def calibrate(self):
        print("Click 4 points corresponding to the pitch corners.")
        print("Order: Bottom-Left, Bottom-Right, Top-Right, Top-Left")
        
        self.frame_copy = self.frame.copy()
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self._mouse_callback)
        
        while True:
            cv2.imshow("Calibration", self.frame_copy)
            key = cv2.waitKey(1) & 0xFF
            
            if len(self.src_points) == 4:
                # Basic notification
                cv2.putText(self.frame_copy, "4 points selected. Press 's' to save or 'r' to reset.", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if key == ord('r'):
                self.src_points = []
                self.frame_copy = self.frame.copy()
                print("Reset points.")
            elif key == ord('s') and len(self.src_points) == 4:
                mapping = PitchMapping(self.src_points, self.dst_points)
                mapping.save_config(self.config_path)
                print(f"Saved config to {self.config_path}")
                break
            elif key == ord('q') or key == 27: # Esc
                break
                
        cv2.destroyAllWindows()
