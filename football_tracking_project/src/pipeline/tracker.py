import supervision as sv

class FootballTracker:
    def __init__(self, track_thresh: float = 0.25, track_buffer: int = 30, match_thresh: float = 0.8):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh
        )
        
    def update(self, detections: sv.Detections) -> sv.Detections:
        player_detections = detections[detections.class_id == 0]
        ball_detections = detections[detections.class_id == 32]
        
        tracked_players = self.tracker.update_with_detections(player_detections)
        
        if len(ball_detections) > 0:
            import numpy as np
            ball_detections.tracker_id = np.array([-1] * len(ball_detections))
            
            tracked_players.data = {}
            ball_detections.data = {}
            
            final_detections = sv.Detections.merge([tracked_players, ball_detections])
        else:
            final_detections = tracked_players
            
        return final_detections
