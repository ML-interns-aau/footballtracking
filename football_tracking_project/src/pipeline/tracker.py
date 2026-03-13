import supervision as sv

class FootballTracker:
    def __init__(self, track_thresh: float = 0.25, track_buffer: int = 30, match_thresh: float = 0.8):
        """Initialize the ByteTrack multi-object tracker for football players.
        
        Args:
            track_thresh: Tracking confidence threshold
            track_buffer: How many frames to keep a lost track alive
            match_thresh: IOU matching threshold
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh
        )
        
    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with new detections and assign track IDs.
        
        Args:
            detections: supervision Detections from the detector
            
        Returns:
            sv.Detections: The same detections but now with tracker_id fields populated.
        """
        # We only want to track players and referees, not the ball. 
        # For simplicity, we track everything passed in, but the detector 
        # should ideally only pass 'person' (class_id=0) to the tracker.
        
        # Filter out ball (class_id=32) from tracking, as its movement is too erratic for ByteTrack
        player_detections = detections[detections.class_id == 0]
        ball_detections = detections[detections.class_id == 32]
        
        tracked_players = self.tracker.update_with_detections(player_detections)
        
        # Combine back the tracked players and the untracked ball
        if len(ball_detections) > 0:
            # We must ensure both sets have the same fields for sv.Detections.merge
            # Assign a dummy tracker_id (-1) to the ball
            import numpy as np
            ball_detections.tracker_id = np.array([-1] * len(ball_detections))
            
            # Reset .data to avoid "All data dictionaries must have the same keys to merge" error
            tracked_players.data = {}
            ball_detections.data = {}
            
            final_detections = sv.Detections.merge([tracked_players, ball_detections])
        else:
            final_detections = tracked_players
            
        return final_detections
