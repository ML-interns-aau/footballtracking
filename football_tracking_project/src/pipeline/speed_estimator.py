import numpy as np
from collections import defaultdict
from src.pipeline.pitch_mapper import PitchMapper


class SpeedEstimator:
    """
    Improved speed estimator with:
    - Exponential moving average (EMA) smoothing per tracker → no wild speed jumps
    - Unrealistic jump clamping (> 12 m in one frame → skip)
    - Ball speed support (tracker_id = BallTracker.BALL_ID = -99)
    - window_size increased to 8 frames for a smoother rolling average
    """

    EMA_ALPHA = 0.30          # Smoothing factor: closer to 1 = noisier, closer to 0 = smoother
    MAX_STEP_M = 12.0         # Metres — discard steps larger than this (outliers / homography errors)
    MAX_SPEED_KMH = 55.0      # Physical upper bound for any human or ball measurement

    def __init__(self, fps: float, pitch_mapper: PitchMapper, window_size: int = 8):
        self.fps = fps
        self.pitch_mapper = pitch_mapper
        self.window_size = window_size

        self.history: dict[int, list] = defaultdict(list)    # tracker_id → [(frame_idx, x_m, y_m)]
        self.speeds: dict[int, float] = {}                    # tracker_id → smoothed speed km/h
        self.distances: dict[int, float] = defaultdict(float) # tracker_id → total distance m

    # ------------------------------------------------------------------
    def estimate_speed(
        self,
        frame_idx: int,
        tracker_ids: list[int],
        points: np.ndarray,
        cam_dx: float,
        cam_dy: float,
    ):
        """
        Update speed estimates for all tracked objects.

        Parameters
        ----------
        tracker_ids : list of ints  (use -99 for ball)
        points      : (N, 2) pixel coordinates (feet/bottom-centre for players, centre for ball)
        cam_dx, cam_dy : camera motion compensation in pixels
        """
        if len(points) == 0:
            return

        # Camera-motion compensation
        compensated = points.copy().astype(np.float32)
        compensated[:, 0] += cam_dx
        compensated[:, 1] += cam_dy

        meter_pts = self.pitch_mapper.transform_points(compensated)

        for idx, tracker_id in enumerate(tracker_ids):
            if tracker_id is None:
                continue

            x_m, y_m = float(meter_pts[idx][0]), float(meter_pts[idx][1])
            history = self.history[tracker_id]

            if history:
                last_frame, last_x, last_y = history[-1]
                frame_gap = frame_idx - last_frame

                if frame_gap > 0:
                    step_dist = np.hypot(x_m - last_x, y_m - last_y)
                    step_per_frame = step_dist / frame_gap

                    # Discard physically impossible jumps
                    if step_per_frame <= self.MAX_STEP_M:
                        self.distances[tracker_id] += step_dist

            history.append((frame_idx, x_m, y_m))
            if len(history) > self.window_size:
                history.pop(0)

            if len(history) >= 2:
                old_frame, old_x, old_y = history[0]
                frames_elapsed = frame_idx - old_frame

                if frames_elapsed > 0:
                    dist_moved = np.hypot(x_m - old_x, y_m - old_y)
                    time_elapsed = frames_elapsed / self.fps
                    raw_speed_kmh = min((dist_moved / time_elapsed) * 3.6, self.MAX_SPEED_KMH)

                    # EMA smoothing
                    prev = self.speeds.get(tracker_id, raw_speed_kmh)
                    self.speeds[tracker_id] = round(
                        self.EMA_ALPHA * raw_speed_kmh + (1 - self.EMA_ALPHA) * prev, 2
                    )
            else:
                if tracker_id not in self.speeds:
                    self.speeds[tracker_id] = 0.0

    # ------------------------------------------------------------------
    def get_stats(self, tracker_id: int) -> tuple[float, float, tuple[float, float]]:
        """Return (speed_kmh, total_distance_m, (x_m, y_m)) for a tracker."""
        speed = self.speeds.get(tracker_id, 0.0)
        dist = round(self.distances[tracker_id], 2)
        history = self.history.get(tracker_id, [])
        coord = (round(history[-1][1], 2), round(history[-1][2], 2)) if history else (0.0, 0.0)
        return speed, dist, coord
