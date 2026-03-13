import cv2
import supervision as sv
import numpy as np

class PipelineVisualizer:
    def __init__(self):
        """Initialize the annotators for bounding boxes and text overlays."""
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        
        # Color palette for teams. We map predictions (0, 1) and raw classes.
        # Team 0: Red, Team 1: Blue, Referee: Yellow, Ball: White
        self.team_colors = {
            0: sv.Color(r=230, g=40, b=40),  # Red
            1: sv.Color(r=40, g=100, b=230), # Blue
            -1: sv.Color(r=150, g=150, b=150), # Unassigned
            "referee": sv.Color(r=240, g=240, b=40), # Yellow
            "goalkeeper": sv.Color(r=40, g=230, b=40), # Green
            "ball": sv.Color(r=255, g=255, b=255) # White
        }

    def annotate_frame(
        self, 
        frame: np.ndarray, 
        detections: sv.Detections, 
        team_ids: np.ndarray,
        speed_estimator
    ) -> np.ndarray:
        """Draw bounding boxes and stat overlays on the frame.
        
        Args:
            frame: Raw BGR frame
            detections: Tracked detections
            team_ids: Array of team mapping IDs corresponding to detections
            speed_estimator: Initialized SpeedEstimator to get player stats
        """
        annotated_frame = frame.copy()
        
        # We need to construct custom colors for BoxAnnotator
        # supervision requires a list of colors when using color_lookup=sv.ColorLookup.CUSTOM
        colors = []
        labels = []
        
        for i in range(len(detections)):
            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else None
            
            # Determine Color and Base Label
            if class_id == 0 and tracker_id is not None:
                tid = team_ids[i]
                c = self.team_colors.get(tid, self.team_colors[-1])
                base_label = f"T{tid} | #{tracker_id}"
            elif class_id == 32:
                c = self.team_colors["ball"]
                base_label = "Ball"
            else:
                c = self.team_colors[-1]
                base_label = f"ID: {tracker_id if tracker_id else '?'}"
                
            colors.append(c)
            
            # Fetch Stats
            if tracker_id is not None and class_id == 0:
                speed, dist, (x_m, y_m) = speed_estimator.get_stats(tracker_id)
                # Multi-line strings aren't natively supported by basic putText, 
                # but we will just write the primary label with the box, and draw custom text later.
                labels.append(base_label)
            else:
                labels.append(base_label)
                
        # 1. Draw Bounding Boxes Manually to avoid supervision version conflicts
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            c = colors[i].as_bgr()  # Get back tuple (B, G, R)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), c, 2)
        
        # 2. Draw Custom Multi-Line Overlays
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else None

            if class_id != 0 or tracker_id is None:
                continue
                
            speed, dist, (x_m, y_m) = speed_estimator.get_stats(tracker_id)
            tid = team_ids[i]
            
            x1, y1, x2, y2 = bbox
            txt_x = int(x1)
            txt_y = int(y2) + 20 # Below bounding box
            
            # Overlay lines
            lines = [
                f"T{tid} | #{tracker_id}",
                f"{speed:.1f} km/h",
                f"{dist:.1f} m",
                f"({x_m:.1f}, {y_m:.1f})"
            ]
            
            for line_idx, text in enumerate(lines):
                y_offset = txt_y + (line_idx * 15)
                # Black stroke
                cv2.putText(annotated_frame, text, (txt_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                # White text
                cv2.putText(annotated_frame, text, (txt_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return annotated_frame
