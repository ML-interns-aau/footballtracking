import supervision as sv
import numpy as np


class FootballTracker:
    """
    ByteTrack-based player tracker with improved ID persistence.
    - lost_track_buffer=60  → holds IDs for 2 s at 30 fps during occlusion / missed frames
    - track_activation_threshold=0.20 → activates tracks from weaker YOLO detections
    - minimum_matching_threshold=0.80 → consistent association
    - Ball gets pseudo-tracker_id = -99 so speed estimator can track it the same way
    """

    BALL_TRACKER_ID = -99

    def __init__(
        self,
        track_thresh: float = 0.20,
        track_buffer: int = 60,
        match_thresh: float = 0.80,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            minimum_consecutive_frames=1,   # Accept tracks from first frame
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Track person detections; attach ball detections with fixed tracker_id.
        Returns a merged Detections object where:
          - class_id == 0  → player  (real ByteTrack ID ≥ 1)
          - class_id == 32 → ball    (tracker_id == BALL_TRACKER_ID)
        """
        player_mask = detections.class_id == 0
        ball_mask = detections.class_id == 32

        player_detections = detections[player_mask]
        ball_detections = detections[ball_mask]

        # Track players
        tracked_players = self.tracker.update_with_detections(player_detections)

        # Attach fixed tracker_id to ball; keep highest-confidence ball only
        if len(ball_detections) > 0:
            if len(ball_detections) > 1:
                # Keep the detection with the highest confidence
                best_idx = int(np.argmax(ball_detections.confidence))
                ball_detections = ball_detections[[best_idx]]

            ball_detections.tracker_id = np.array([self.BALL_TRACKER_ID])

            # Clear data dicts to allow merging
            tracked_players.data = {}
            ball_detections.data = {}

            return sv.Detections.merge([tracked_players, ball_detections])

        return tracked_players
