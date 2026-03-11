try:
    from .orchestrator import TrackingOrchestrator
except ImportError:
    from orchestrator import TrackingOrchestrator


class BaseTracker:
    """
    Compatibility wrapper exposing previous tracker API on top of orchestrator.
    """

    def __init__(self, config):
        self.config = config
        self.orchestrator = TrackingOrchestrator(config)

    def get_tracks(self, frame):
        return self.orchestrator.process_frame(frame)

    @staticmethod
    def extract_coords(frame_output):
        return [
            {"id": t.track_id, "pos": t.bottom_center, "role": t.role, "jersey": t.jersey_label}
            for t in frame_output.detections
        ]
