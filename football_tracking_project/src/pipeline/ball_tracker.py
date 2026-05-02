"""
BallTracker — Kalman filter-based ball tracking with optical-flow fallback.

State vector: [x, y, vx, vy]
- On a YOLO detection frame : update Kalman with measurement
- On a missed detection frame: predict-only (returns estimated position)
- Optical-flow fallback      : used when Kalman diverges or has not been initialised
- Trail buffer               : stores last N positions for visualisation
"""

import cv2
import numpy as np
from collections import deque


class BallTracker:
    BALL_ID = -99          # matches FootballTracker.BALL_TRACKER_ID

    def __init__(
        self,
        max_trail: int = 25,
        max_missed: int = 30,
        diverge_px: float = 150.0,
    ):
        self.max_trail = max_trail
        self.max_missed = max_missed     # After this many consecutive missed frames, reset
        self.diverge_px = diverge_px    # Kalman is ignored if prediction jumps > this many px

        self._kf = self._build_kalman()
        self._initialised = False
        self._missed_count = 0

        self.trail: deque = deque(maxlen=max_trail)   # list of (cx, cy) pixel positions
        self._prev_gray: np.ndarray | None = None     # for optical-flow fallback

    # ------------------------------------------------------------------
    # Kalman filter construction
    # ------------------------------------------------------------------
    @staticmethod
    def _build_kalman() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)                    # 4 state vars, 2 measurement vars
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)

        dt = 1.0
        kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]], dtype=np.float32)

        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    # ------------------------------------------------------------------
    # Optical-flow fallback
    # ------------------------------------------------------------------
    def _optical_flow_estimate(self, gray: np.ndarray) -> tuple[float, float] | None:
        """Try to locate the ball via sparse optical flow from the previous frame."""
        if self._prev_gray is None or len(self.trail) == 0:
            return None

        last_cx, last_cy = self.trail[-1]
        pt = np.array([[[last_cx, last_cy]]], dtype=np.float32)

        try:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pt, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
        except cv2.error:
            return None

        if p1 is not None and st is not None and st[0][0] == 1:
            nx, ny = p1[0][0]
            return float(nx), float(ny)
        return None

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(
        self,
        frame: np.ndarray,
        ball_detections: "sv.Detections | None",   # noqa: F821
    ) -> tuple[float, float, bool]:
        """
        Call once per frame.

        Parameters
        ----------
        frame       : Current BGR frame (for optical-flow fallback)
        ball_detections : sv.Detections filtered to ball (class_id 32), may be empty

        Returns
        -------
        (cx, cy, is_predicted)
            cx, cy        — pixel-space ball centre
            is_predicted  — True when position is estimated (no YOLO hit this frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- YOLO gave us the ball ----
        has_detection = ball_detections is not None and len(ball_detections) > 0

        if has_detection:
            x1, y1, x2, y2 = ball_detections.xyxy[0]
            cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
            measurement = np.array([[cx], [cy]], dtype=np.float32)

            if not self._initialised:
                # Warm-start the Kalman state
                self._kf.statePre = np.array(
                    [[cx], [cy], [0.0], [0.0]], dtype=np.float32)
                self._kf.statePost = self._kf.statePre.copy()
                self._initialised = True
            else:
                self._kf.predict()
                self._kf.correct(measurement)

            self._missed_count = 0
            self.trail.append((cx, cy))
            self._prev_gray = gray
            return cx, cy, False

        # ---- YOLO missed the ball ----
        self._missed_count += 1

        if self._missed_count > self.max_missed:
            self._initialised = False
            self._prev_gray = gray
            if len(self.trail) > 0:
                return *self.trail[-1], True
            return 0.0, 0.0, True

        if self._initialised:
            pred = self._kf.predict()
            px, py = float(pred[0][0]), float(pred[1][0])

            # Sanity-check: if prediction diverges far from last known → use optical flow
            if len(self.trail) > 0:
                lx, ly = self.trail[-1]
                if np.hypot(px - lx, py - ly) > self.diverge_px:
                    of = self._optical_flow_estimate(gray)
                    if of is not None:
                        px, py = of

            self.trail.append((px, py))
            self._prev_gray = gray
            return px, py, True

        # Kalman not initialised yet — try optical flow
        of = self._optical_flow_estimate(gray)
        self._prev_gray = gray
        if of is not None:
            self.trail.append(of)
            return *of, True

        if len(self.trail) > 0:
            return *self.trail[-1], True
        return 0.0, 0.0, True

    def get_trail(self) -> list[tuple[float, float]]:
        """Return list of (cx, cy) positions for drawing the ball trail."""
        return list(self.trail)
