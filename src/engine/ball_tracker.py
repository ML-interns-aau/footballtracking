from __future__ import annotations
import cv2
import numpy as np
from collections import deque
class BallTracker:
    BALL_ID = -99
    def __init__(
        self,
        max_trail: int = 25,
        max_missed: int = 30,
        diverge_px: float = 150.0,
    ):
        self.max_trail = max_trail
        self.max_missed = max_missed
        self.diverge_px = diverge_px
        self._kf = self._build_kalman()
        self._initialised = False
        self._missed_count = 0
        self.trail: deque = deque(maxlen=max_trail)
        self._prev_gray: np.ndarray | None = None
    @staticmethod
    def _build_kalman() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
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
    def _optical_flow_estimate(self, gray: np.ndarray) -> tuple[float, float] | None:
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
    def update(
        self,
        frame: np.ndarray,
        ball_detections: "sv.Detections | None",
    ) -> tuple[float, float, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        has_detection = ball_detections is not None and len(ball_detections) > 0
        if has_detection:
            x1, y1, x2, y2 = ball_detections.xyxy[0]
            cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            if not self._initialised:
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
            if len(self.trail) > 0:
                lx, ly = self.trail[-1]
                if np.hypot(px - lx, py - ly) > self.diverge_px:
                    of = self._optical_flow_estimate(gray)
                    if of is not None:
                        px, py = of
            self.trail.append((px, py))
            self._prev_gray = gray
            return px, py, True
        of = self._optical_flow_estimate(gray)
        self._prev_gray = gray
        if of is not None:
            self.trail.append(of)
            return *of, True
        if len(self.trail) > 0:
            return *self.trail[-1], True
        return 0.0, 0.0, True
    def get_trail(self) -> list[tuple[float, float]]:
        return list(self.trail)


def _build_constant_velocity_kalman() -> cv2.KalmanFilter:
    return BallTracker._build_kalman()


class BallCandidateClusterer:
    """Groups raw ball-class detections into spatial clusters in
    camera-compensated coordinates, so a persistently static object (a spare
    ball on the sideline, a sponsor logo on the perimeter LED boards) can be
    told apart from the real corner-kick ball.

    The real ball is also stationary for a while before the kick — so "static
    for N frames" alone can't be the rejection rule, or it would blacklist the
    genuine pre-kick resting ball too. What actually distinguishes a permanent
    fixture is that it keeps getting (re-)detected at the same spot both near
    the start of the clip AND much later, well past when any single kick event
    could plausibly still be in its run-up: the real ball moves away after
    being struck and does not return to sit at that exact pixel again.
    """

    def __init__(
        self,
        cluster_radius_px: float = 25.0,
        early_frac: float = 0.10,
        late_frac: float = 0.60,
        min_frames_for_decoy: int = 6,
        static_spread_px: float = 8.0,
        min_span_frac_for_decoy: float = 0.40,
        straggler_gap_frames: int = 25,
    ):
        self.cluster_radius_px = cluster_radius_px
        self.early_frac = early_frac
        self.late_frac = late_frac
        self.min_frames_for_decoy = min_frames_for_decoy
        self.static_spread_px = static_spread_px
        # A decoy doesn't have to start in the first early_frac of the clip to be
        # a fixture -- any object that sits still for a long stretch (regardless
        # of when it starts) is very unlikely to be the ball, since the ball is
        # only ever stationary briefly (pre-kick) or fleetingly (post-contact).
        self.min_span_frac_for_decoy = min_span_frac_for_decoy
        # A single stray misdetection near a real resting-ball spot long after
        # it stopped being there (a shadow flicker, a pitch marking briefly
        # crossing the confidence threshold) can sit tens of frames away from
        # every other detection in the cluster. Using raw min/max frame as
        # first_frame/last_frame lets that one point drag the reported span
        # (and "is this late" check) out arbitrarily far, condemning the whole
        # cluster -- including genuine early frames that are the real ball.
        # Trimming leading/trailing points that are isolated by more than this
        # many frames from their nearest neighbor in the same cluster keeps
        # span/lateness anchored to the cluster's actual dense presence.
        self.straggler_gap_frames = straggler_gap_frames
        self._clusters: dict[int, dict] = {}
        self._next_id = 0
        self.total_frames: int = 0

    def add(self, frame_idx: int, comp_xy: tuple[float, float], confidence: float) -> int:
        """Assigns a compensated ball observation to a cluster (creating one if
        none is close enough), returning that cluster's id."""
        self.total_frames = max(self.total_frames, frame_idx + 1)
        cx, cy = comp_xy

        best_id, best_dist = None, None
        for cid, c in self._clusters.items():
            ccx, ccy = c["sum_x"] / c["count"], c["sum_y"] / c["count"]
            d = float(np.hypot(cx - ccx, cy - ccy))
            if d < self.cluster_radius_px and (best_dist is None or d < best_dist):
                best_id, best_dist = cid, d

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1
            self._clusters[best_id] = {
                "frames": [], "points": [], "confidences": [],
                "sum_x": 0.0, "sum_y": 0.0, "count": 0,
            }

        c = self._clusters[best_id]
        c["frames"].append(frame_idx)
        c["points"].append((cx, cy))
        c["confidences"].append(confidence)
        c["sum_x"] += cx
        c["sum_y"] += cy
        c["count"] += 1
        return best_id

    @staticmethod
    def _trim_stragglers(frames_sorted: list[int], max_gap: int) -> tuple[int, int]:
        """Returns (effective_first_frame, effective_last_frame) after dropping
        leading/trailing points isolated from the rest of the cluster by more
        than max_gap frames -- a single distant straggler shouldn't stretch
        the reported span or lateness of an otherwise tightly-grouped run."""
        lo, hi = 0, len(frames_sorted) - 1
        while lo < hi and frames_sorted[lo + 1] - frames_sorted[lo] > max_gap:
            lo += 1
        while hi > lo and frames_sorted[hi] - frames_sorted[hi - 1] > max_gap:
            hi -= 1
        return frames_sorted[lo], frames_sorted[hi]

    def classify(self) -> dict[int, dict]:
        """Returns cluster_id -> summary dict, including `is_decoy` (permanent
        fixture, excluded from the active-ball selection) and `is_static`
        (spread-based, informational only)."""
        n = max(self.total_frames, 1)
        result = {}
        for cid, c in self._clusters.items():
            centroid = (c["sum_x"] / c["count"], c["sum_y"] / c["count"])
            pts = np.array(c["points"])
            spread = float(np.max(np.hypot(pts[:, 0] - centroid[0], pts[:, 1] - centroid[1])))
            first_frame, last_frame = min(c["frames"]), max(c["frames"])
            is_static = spread < self.static_spread_px

            eff_first, eff_last = self._trim_stragglers(sorted(c["frames"]), self.straggler_gap_frames)
            span = eff_last - eff_first

            spans_early_and_late = eff_first <= self.early_frac * n and eff_last >= self.late_frac * n
            long_static_run = is_static and span >= self.min_span_frac_for_decoy * n
            is_decoy = c["count"] >= self.min_frames_for_decoy and (spans_early_and_late or long_static_run)

            result[cid] = {
                "centroid": centroid,
                "spread_px": spread,
                "frame_count": c["count"],
                "first_frame": first_frame,
                "last_frame": last_frame,
                "mean_confidence": float(np.mean(c["confidences"])),
                "is_static": is_static,
                "is_decoy": is_decoy,
            }
        return result


class CompensatedBallSmoother:
    """Kalman-smooths a camera-compensated ball trajectory (constant-velocity
    model, mirroring BallTracker), with an optical-flow fallback for gaps.

    Unlike BallTracker, everything here happens in camera-compensated
    coordinates so the smoothed track — and any timing computed from it — isn't
    corrupted by camera pan. Optical flow itself only makes sense in raw pixel
    space (it tracks image content), so the fallback estimate is computed in
    raw coordinates and converted back to compensated space before use.
    """

    def __init__(self, max_missed: int = 30, diverge_px: float = 250.0):
        self.max_missed = max_missed
        self.diverge_px = diverge_px
        self._kf = _build_constant_velocity_kalman()
        self._initialised = False
        self._missed_count = 0
        self._last_comp: tuple[float, float] | None = None
        self._last_raw: tuple[float, float] | None = None
        self._prev_gray: np.ndarray | None = None

    def _optical_flow_raw(self, gray: np.ndarray) -> tuple[float, float] | None:
        if self._prev_gray is None or self._last_raw is None:
            return None
        pt = np.array([[[self._last_raw[0], self._last_raw[1]]]], dtype=np.float32)
        try:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pt, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
        except cv2.error:
            return None
        if p1 is not None and st is not None and st[0][0] == 1:
            nx, ny = float(p1[0][0][0]), float(p1[0][0][1])
            self._last_raw = (nx, ny)
            return self._last_raw
        return None

    def update(
        self,
        gray: np.ndarray,
        comp_measurement: tuple[float, float] | None,
        raw_measurement: tuple[float, float] | None,
        cum_cam_offset: tuple[float, float],
    ) -> tuple[float, float, bool]:
        """Returns (comp_x, comp_y, is_predicted) for this frame."""
        has_detection = comp_measurement is not None
        if has_detection:
            cx, cy = comp_measurement
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            if not self._initialised:
                self._kf.statePre = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
                self._kf.statePost = self._kf.statePre.copy()
                self._initialised = True
            else:
                self._kf.predict()
                self._kf.correct(measurement)
            self._missed_count = 0
            self._last_comp = (cx, cy)
            self._last_raw = raw_measurement
            self._prev_gray = gray
            return cx, cy, False

        self._missed_count += 1
        if self._missed_count > self.max_missed:
            self._initialised = False
            self._prev_gray = gray
            if self._last_comp is not None:
                return self._last_comp[0], self._last_comp[1], True
            return 0.0, 0.0, True

        if self._initialised:
            pred = self._kf.predict()
            px, py = float(pred[0][0]), float(pred[1][0])
            of = self._optical_flow_raw(gray)
            if of is not None and self._last_comp is not None:
                of_comp = (of[0] - cum_cam_offset[0], of[1] - cum_cam_offset[1])
                if np.hypot(px - self._last_comp[0], py - self._last_comp[1]) > self.diverge_px:
                    px, py = of_comp
            self._last_comp = (px, py)
            self._prev_gray = gray
            return px, py, True

        of = self._optical_flow_raw(gray)
        self._prev_gray = gray
        if of is not None:
            comp = (of[0] - cum_cam_offset[0], of[1] - cum_cam_offset[1])
            self._last_comp = comp
            return comp[0], comp[1], True
        if self._last_comp is not None:
            return self._last_comp[0], self._last_comp[1], True
        return 0.0, 0.0, True