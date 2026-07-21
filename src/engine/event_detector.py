"""Kick-moment and first-contact timing for a corner-kick clip.

Operates ONLY on the camera-compensated, Kalman-smoothed ball trajectory
(see src/engine/ball_tracker.py::CompensatedBallSmoother) — never on raw
pixel velocity, and never on a raw twice-differentiated acceleration signal.
Raw pixel velocity is corrupted by camera pan; naive acceleration amplifies
per-frame noise.

Contact is detected as a sharp direction reversal between consecutive REAL
(non-predicted) ball detections -- a genuine touch/header abruptly redirects
the ball, which shows up as a large angle between the velocity vector into a
frame and the velocity vector out of it. This only ever compares two
adjacent real samples close together in time, which matters because this
footage's ball detections are sparse enough to have long (10-25+ frame)
gaps with zero real detections, coasted over by the Kalman filter. An
earlier approach fit a single quadratic (ballistic) curve across the whole
post-kick flight and flagged contact as residual departure from it -- but
extrapolating that fit across a real detection gap is numerically unstable:
the residual blows up arbitrarily at the first real point after ANY gap,
regardless of whether that's the actual touch, which made it fire on
whichever frame happened to end the gap rather than on genuine contact. The
quadratic fit is kept only as a secondary, gap-bounded corroborating value
attached to the result -- it no longer gates the decision.
"""
from __future__ import annotations

import numpy as np


def compensated_speed_series(comp_positions: list[tuple[float, float]]) -> np.ndarray:
    n = len(comp_positions)
    speed = np.zeros(n)
    for i in range(1, n):
        x0, y0 = comp_positions[i - 1]
        x1, y1 = comp_positions[i]
        speed[i] = float(np.hypot(x1 - x0, y1 - y0))
    return speed


def _nearest_player_distance(
    ball_xy: tuple[float, float],
    players: dict[int, tuple[float, float]],
    exclude_id: int | None = None,
) -> tuple[int | None, float | None]:
    bx, by = ball_xy
    best_id, best_dist = None, None
    for tid, (px, py) in players.items():
        if tid == exclude_id:
            continue
        d = float(np.hypot(px - bx, py - by))
        if best_dist is None or d < best_dist:
            best_dist, best_id = d, tid
    return best_id, best_dist


class EventTimingDetector:
    def __init__(
        self,
        baseline_window: int = 40,
        stationary_thresh: float = 4.0,
        kick_sigma: float = 4.0,
        persistence_frames: int = 3,
        max_gap_frames: int = 5,
        corner_margin_frac: float = 0.22,
        ballistic_fit_frames: int = 8,
        ballistic_fit_window_frames: int = 20,
        residual_thresh_px: float = 15.0,
        contact_proximity_px: float = 60.0,
        contact_search_frames: int = 45,
        contact_persistence_frames: int = 3,
        reversal_angle_thresh_deg: float = 80.0,
        min_flight_speed_px: float = 6.0,
        max_segment_gap_frames: int = 3,
    ):
        self.baseline_window = baseline_window
        self.stationary_thresh = stationary_thresh
        self.kick_sigma = kick_sigma
        self.persistence_frames = max(2, persistence_frames)
        self.max_gap_frames = max_gap_frames
        self.corner_margin_frac = corner_margin_frac
        self.ballistic_fit_frames = ballistic_fit_frames
        self.ballistic_fit_window_frames = ballistic_fit_window_frames
        self.residual_thresh_px = residual_thresh_px
        self.contact_proximity_px = contact_proximity_px
        self.contact_search_frames = contact_search_frames
        self.contact_persistence_frames = max(1, contact_persistence_frames)
        self.reversal_angle_thresh_deg = reversal_angle_thresh_deg
        self.min_flight_speed_px = min_flight_speed_px
        self.max_segment_gap_frames = max_segment_gap_frames

    # ------------------------------------------------------------------ #
    # Kick onset
    # ------------------------------------------------------------------ #

    def detect_kick(
        self,
        comp_positions: list[tuple[float, float]],
        is_predicted: list[bool],
        frame_size: tuple[int, int] | None = None,
    ) -> dict:
        speed = compensated_speed_series(comp_positions)
        n = len(speed)
        real_mask = np.array([not p for p in is_predicted])

        search_n = min(self.baseline_window, n)

        # Longest run of consecutive real, near-stationary frames in the search window.
        candidate_runs: list[tuple[int, int]] = []
        cur_start = None
        for i in range(search_n):
            stationary = real_mask[i] and speed[i] < self.stationary_thresh
            if stationary:
                if cur_start is None:
                    cur_start = i
            elif cur_start is not None:
                candidate_runs.append((cur_start, i - cur_start))
                cur_start = None
        if cur_start is not None:
            candidate_runs.append((cur_start, search_n - cur_start))

        if not candidate_runs:
            start, length = 0, search_n
        elif len(candidate_runs) == 1 or frame_size is None:
            start, length = max(candidate_runs, key=lambda r: r[1])
        else:
            # Weak corner-proximity tie-break among comparably long runs.
            w, h = frame_size
            margin_x, margin_y = w * self.corner_margin_frac, h * self.corner_margin_frac
            best_run, best_score = None, None
            longest = max(r[1] for r in candidate_runs)
            for s, l in candidate_runs:
                if l < 0.6 * longest:
                    continue
                cx, cy = comp_positions[s + l // 2]
                near_corner = (cx < margin_x or cx > w - margin_x) and (cy < margin_y or cy > h - margin_y)
                score = l + (1000 if near_corner else 0)
                if best_score is None or score > best_score:
                    best_score, best_run = score, (s, l)
            start, length = best_run if best_run is not None else max(candidate_runs, key=lambda r: r[1])

        baseline_vals = speed[start:start + length][real_mask[start:start + length]]
        if len(baseline_vals) == 0:
            baseline_vals = speed[start:start + length]
        baseline_mean = float(np.mean(baseline_vals)) if len(baseline_vals) else 0.0
        baseline_std = float(np.std(baseline_vals)) if len(baseline_vals) else 0.0
        kick_threshold = max(self.stationary_thresh, baseline_mean + self.kick_sigma * baseline_std)

        spike_idx = None
        for i in range(start, n - self.persistence_frames + 1):
            if np.all(speed[i:i + self.persistence_frames] > kick_threshold):
                spike_idx = i
                break

        confidence_penalty = 1.0
        if spike_idx is None:
            tail_start = start + max(length, 1)
            tail = speed[tail_start:]
            spike_idx = tail_start + int(np.argmax(tail)) if len(tail) else n - 1
            confidence_penalty *= 0.5

        # Prefer the last REAL stationary observation over a predicted one.
        # A detection gap right before the kick means the predicted/Kalman-
        # held frames trivially read as "stationary" (the filter just coasts
        # at ~0 velocity) all the way to the edge of the gap, which pushes the
        # reported kick frame to the END of a blind window that may span many
        # frames -- conveying false precision about a moment we have no data
        # for. The last confirmed-real resting position is an actual
        # observation and, empirically, lands much closer to the true kick.
        kick_frame = spike_idx
        found_real = False
        for k in range(spike_idx, -1, -1):
            if speed[k] < self.stationary_thresh and real_mask[k]:
                kick_frame = k
                found_real = True
                break
        if not found_real:
            for k in range(spike_idx, -1, -1):
                if speed[k] < self.stationary_thresh:
                    kick_frame = k
                    break

        lo = max(0, kick_frame - self.max_gap_frames)
        hi = min(n, spike_idx + self.max_gap_frames + 1)
        gap = int(np.sum(~real_mask[lo:hi]))
        if gap > self.max_gap_frames:
            confidence_penalty *= 0.5

        onset_speed = float(speed[min(spike_idx + self.persistence_frames - 1, n - 1)])
        sharpness = (onset_speed - baseline_mean) / (self.kick_sigma * baseline_std + 1e-6)
        confidence = float(np.clip(sharpness / 3.0, 0.0, 1.0)) * confidence_penalty

        return {
            "frame": kick_frame,
            "confidence": round(confidence, 3),
            "baseline_mean_px_per_frame": baseline_mean,
            "baseline_std_px_per_frame": baseline_std,
            "threshold_px_per_frame": kick_threshold,
        }

    # ------------------------------------------------------------------ #
    # First contact — ballistic residual
    # ------------------------------------------------------------------ #

    def _fit_ballistic(
        self,
        comp_positions: list[tuple[float, float]],
        is_predicted: list[bool],
        start: int,
        min_points: int,
        search_limit: int,
    ) -> tuple[tuple[np.ndarray, np.ndarray] | None, int]:
        """Fits a per-axis quadratic (constant-acceleration) model, since a real
        corner-kick ball follows a gravity-curved arc, not a straight line.

        Uses only REAL (non-predicted) samples -- a Kalman-held/optical-flow
        "phantom" position isn't an observation of the ball, it's a guess, and
        fitting to a run of phantom (near-static) positions produces a
        degenerate fit that any later real detection would appear to "depart"
        from, falsely flagging contact the moment tracking resumes.

        Returns (fit_or_None, frame_index_of_last_point_used).
        """
        n = len(comp_positions)
        ts: list[float] = []
        xs: list[float] = []
        ys: list[float] = []
        last_t = start
        t = start
        while t < min(n, search_limit) and len(ts) < min_points:
            if not is_predicted[t]:
                ts.append(float(t - start))
                xs.append(comp_positions[t][0])
                ys.append(comp_positions[t][1])
                last_t = t
            t += 1
        if len(ts) < 4:
            return None, last_t
        ts_arr = np.array(ts, dtype=np.float64)
        A = np.vstack([ts_arr**2, ts_arr, np.ones_like(ts_arr)]).T
        cx = np.linalg.lstsq(A, np.array(xs), rcond=None)[0]
        cy = np.linalg.lstsq(A, np.array(ys), rcond=None)[0]
        return (cx, cy), last_t

    @staticmethod
    def _predict_ballistic(coeffs: tuple[np.ndarray, np.ndarray], dt: float) -> np.ndarray:
        cx, cy = coeffs
        return np.array([
            cx[0] * dt * dt + cx[1] * dt + cx[2],
            cy[0] * dt * dt + cy[1] * dt + cy[2],
        ])

    def detect_contact(
        self,
        comp_positions: list[tuple[float, float]],
        raw_positions: list[tuple[float, float]],
        is_predicted: list[bool],
        kick_frame: int,
        player_positions_per_frame: list[dict[int, tuple[float, float]]],
        kicker_id: int | None,
    ) -> dict:
        n = len(comp_positions)
        hi = min(n, kick_frame + 1 + self.contact_search_frames)
        real_ts = [t for t in range(kick_frame + 1, hi) if not is_predicted[t]]

        # Corroborating signal only: a quadratic fit over an early, gap-bounded
        # window gives a residual to report alongside the pick when data is
        # dense enough to extrapolate safely. It does NOT gate contact or feed
        # confidence -- extrapolating it across a real long detection gap (a
        # stretch with zero real ball detections, common in this footage)
        # blows the prediction up arbitrarily, so residual would spike at the
        # very first real point after ANY gap regardless of whether that's
        # the real touch.
        fit_window_limit = min(hi, kick_frame + 1 + self.ballistic_fit_window_frames)
        fit, _ = self._fit_ballistic(
            comp_positions, is_predicted, kick_frame + 1,
            min_points=max(4, self.ballistic_fit_frames), search_limit=fit_window_limit,
        )

        def residual_at(t: int) -> float | None:
            if fit is None:
                return None
            predicted = self._predict_ballistic(fit, float(t - (kick_frame + 1)))
            observed = np.array(comp_positions[t])
            return float(np.hypot(*(observed - predicted)))

        # Primary signal: a genuine touch sharply redirects the ball. Compare
        # velocity between successive REAL detections only (skipping over any
        # gap in between, since a Kalman/optical-flow-held position isn't an
        # observation of the ball) and flag a large angle change. This only
        # ever compares two adjacent real samples, so a long detection gap
        # just means a longer segment -- never the unstable long-range
        # extrapolation a parametric fit is forced into across the same gap.
        candidates: dict[int, float] = {}  # frame -> reversal angle (degrees)
        for i in range(1, len(real_ts) - 1):
            t_prev, t_cur, t_next = real_ts[i - 1], real_ts[i], real_ts[i + 1]
            gap1, gap2 = t_cur - t_prev, t_next - t_cur
            # A velocity across a multi-frame detection gap is an average, not
            # an instantaneous sample -- comparing it against an adjacent
            # single-frame (or short) velocity is apples-to-oranges and
            # produces spurious "reversals" right where tracking resumes after
            # a gap, not at a real touch. Only compare segments that are both
            # genuinely close in time.
            if gap1 > self.max_segment_gap_frames or gap2 > self.max_segment_gap_frames:
                continue
            v1 = (np.array(comp_positions[t_cur]) - np.array(comp_positions[t_prev])) / max(1, gap1)
            v2 = (np.array(comp_positions[t_next]) - np.array(comp_positions[t_cur])) / max(1, gap2)
            s1, s2 = float(np.hypot(*v1)), float(np.hypot(*v2))
            if s1 < self.min_flight_speed_px or s2 < self.min_flight_speed_px:
                continue
            cos_ang = float(np.clip(np.dot(v1, v2) / (s1 * s2), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_ang)))
            if angle >= self.reversal_angle_thresh_deg:
                candidates[t_cur] = angle

        primary, alt = None, None
        for t in candidates:
            tid, dist = _nearest_player_distance(raw_positions[t], player_positions_per_frame[t], exclude_id=kicker_id)
            if tid is not None and dist is not None and dist < self.contact_proximity_px:
                primary = t
                break
            if alt is None:
                alt = t  # sharp redirect without confirmed proximity — corroborating candidate only
        if primary is None and alt is not None:
            primary, alt = alt, None

        if primary is None:
            return {"frame": None, "confidence": 0.0, "candidates": [], "residual_at_pick_px": None}

        _, dist = _nearest_player_distance(raw_positions[primary], player_positions_per_frame[primary], exclude_id=kicker_id)
        proximity_score = float(np.clip(1.0 - (dist / self.contact_proximity_px), 0.0, 1.0)) if dist is not None else 0.3
        # Angle score: how far past the threshold the reversal was, saturating at +60deg over it.
        angle_score = float(np.clip((candidates[primary] - self.reversal_angle_thresh_deg) / 60.0, 0.0, 1.0))
        confidence = round(angle_score * 0.5 + proximity_score * 0.5, 3)
        r = residual_at(primary)

        return {
            "frame": primary,
            "confidence": confidence,
            "candidates": [c for c in (primary, alt) if c is not None],
            "reversal_angle_deg": round(candidates[primary], 1),
            "residual_at_pick_px": round(r, 2) if r is not None else None,
        }

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def run(
        self,
        comp_positions: list[tuple[float, float]],
        raw_positions: list[tuple[float, float]],
        is_predicted: list[bool],
        player_positions_per_frame: list[dict[int, tuple[float, float]]],
        kicker_id_finder,
        frame_size: tuple[int, int] | None = None,
    ) -> dict:
        kick = self.detect_kick(comp_positions, is_predicted, frame_size=frame_size)
        kicker_id = kicker_id_finder(kick["frame"])
        contact = self.detect_contact(
            comp_positions, raw_positions, is_predicted, kick["frame"], player_positions_per_frame, kicker_id,
        )
        return {
            "kick_frame": kick["frame"],
            "kick_confidence": kick["confidence"],
            "kick_debug": kick,
            "contact_frame": contact["frame"],
            "contact_confidence": contact["confidence"],
            "contact_debug": contact,
            "kicker_id": kicker_id,
        }
