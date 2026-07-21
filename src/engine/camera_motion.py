"""Global camera-motion (pan) estimation, for compensating ball-motion signals.

In pixel space, apparent ball speed = true ball motion + camera motion. Any
timing heuristic built on raw pixel velocity is corrupted whenever the camera
pans/zooms during the clip. This module estimates the frame-to-frame camera
motion from background feature flow (masking out players, so the fit locks
onto the pitch/stands rather than moving people) so it can be subtracted back
out of the ball's apparent velocity before any event timing is computed.
"""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotionEstimator:
    def __init__(
        self,
        motion_model: str = "affine",
        max_corners: int = 300,
        quality_level: float = 0.01,
        min_distance: int = 8,
        min_good_points: int = 10,
        ransac_reproj_threshold: float = 3.0,
    ):
        if motion_model not in ("affine", "homography"):
            raise ValueError(f"motion_model must be 'affine' or 'homography', got {motion_model!r}")
        self.motion_model = motion_model
        self.feature_params = dict(
            maxCorners=max_corners, qualityLevel=quality_level,
            minDistance=min_distance, blockSize=7,
        )
        self.lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.min_good_points = min_good_points
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.history: list[tuple[float, float]] = []

    def _feature_mask(self, shape: tuple[int, int], player_boxes) -> np.ndarray | None:
        if not player_boxes:
            return None
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        for box in player_boxes:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    def estimate(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        player_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> tuple[float, float, np.ndarray | None]:
        """Returns (dx, dy, M) — the median inlier background flow and the fitted motion matrix.

        (dx, dy) is defined as curr - prev (i.e. add it to a previous-frame point
        to predict where the same background point lands this frame). Degrades
        to (0.0, 0.0, None) when there aren't enough trackable background points.
        """
        mask = self._feature_mask(prev_gray.shape[:2], player_boxes)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **self.feature_params)
        if p0 is None or len(p0) < self.min_good_points:
            self.history.append((0.0, 0.0))
            return 0.0, 0.0, None

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
        if p1 is None or st is None:
            self.history.append((0.0, 0.0))
            return 0.0, 0.0, None

        st = st.reshape(-1).astype(bool)
        good_prev = p0.reshape(-1, 2)[st]
        good_curr = p1.reshape(-1, 2)[st]
        if len(good_prev) < self.min_good_points:
            self.history.append((0.0, 0.0))
            return 0.0, 0.0, None

        if self.motion_model == "homography":
            M, inlier_mask = cv2.findHomography(
                good_prev, good_curr, method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold,
            )
        else:
            M, inlier_mask = cv2.estimateAffinePartial2D(
                good_prev, good_curr, method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold,
            )

        if M is None:
            self.history.append((0.0, 0.0))
            return 0.0, 0.0, None

        flow = good_curr - good_prev
        if inlier_mask is not None:
            inliers = inlier_mask.reshape(-1).astype(bool)
            if inliers.sum() >= max(3, self.min_good_points // 2):
                flow = flow[inliers]

        dx, dy = float(np.median(flow[:, 0])), float(np.median(flow[:, 1]))
        self.history.append((dx, dy))
        return dx, dy, M

    @staticmethod
    def compensate_velocity(
        ball_v: tuple[float, float], cam_v: tuple[float, float],
    ) -> tuple[float, float]:
        """Removes camera motion from an apparent ball-velocity vector."""
        return ball_v[0] - cam_v[0], ball_v[1] - cam_v[1]
