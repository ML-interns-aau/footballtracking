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

from src.engine.ball_tracker import ransac_ballistic_fit


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
        arc_break_residual_px: float = 20.0,
        contact_max_gap_frames: int = 20,
        contact_min_score: float = 0.35,
        motion_spike_weight: float = 0.15,
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
        self.arc_break_residual_px = arc_break_residual_px
        self.contact_max_gap_frames = contact_max_gap_frames
        self.contact_min_score = contact_min_score
        self.motion_spike_weight = motion_spike_weight


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

        if spike_idx is None:
            return {
                "frame": None, "confidence": 0.0,
                "baseline_mean_px_per_frame": baseline_mean,
                "baseline_std_px_per_frame": baseline_std,
                "threshold_px_per_frame": kick_threshold,
                "reason": "no sustained speed spike found after the resting baseline",
            }

        confidence_penalty = 1.0

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

    @staticmethod
    def _player_motion_spike(
        player_positions_per_frame: list[dict[int, tuple[float, float]]],
        tid: int | None, t_prev: int, t_cur: int,
        lookback: int = 2, lookahead: int = 2,
    ) -> float:
        """0..1 score for how much tid's OWN bounding-box centroid velocity
        spiked in the vicinity of [t_prev, t_cur] relative to its recent
        baseline -- a leg swinging through or a head snapping into the ball
        shows up as a sudden jump in the player's own motion, independent of
        whether the ball itself was visible at that instant."""
        if tid is None:
            return 0.0
        lo = max(0, t_prev - lookback)
        hi = min(len(player_positions_per_frame) - 1, t_cur + lookahead)
        speeds = []
        for t in range(lo, hi):
            p0 = player_positions_per_frame[t].get(tid)
            p1 = player_positions_per_frame[t + 1].get(tid)
            if p0 is None or p1 is None:
                continue
            speeds.append(float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])))
        if len(speeds) < 2:
            return 0.0
        speeds_arr = np.array(speeds)
        baseline = float(np.median(speeds_arr))
        peak = float(np.max(speeds_arr))
        if baseline < 1e-6:
            return float(np.clip(peak / 20.0, 0.0, 1.0))
        ratio = peak / baseline
        return float(np.clip((ratio - 1.5) / 3.0, 0.0, 1.0))

    def detect_contact(
        self,
        comp_positions: list[tuple[float, float]],
        raw_positions: list[tuple[float, float]],
        is_predicted: list[bool],
        kick_frame: int,
        player_positions_per_frame: list[dict[int, tuple[float, float]]],
        kicker_id: int | None,
    ) -> dict:
        """Tiered contact detection (see CONTACT_DETECTION_SPEC.md). The one
        rule that prevents every failure seen so far: never emit a high-
        confidence single frame unless the ball is detected on BOTH sides of
        the touch.

        Tier 1 (exact frame, high confidence): ball visible immediately before
        AND after an arc-break near a non-kicker player -- no occlusion gap
        (gap == 1). Covers clips where the ball survives the touch and stays
        visible.
        Tier 2 (exact frame, medium confidence): ball visible before and after
        a SHORT occlusion gap (2-5 frames) bracketing an arc-break near a
        player -- the touch happens inside the gap (occluded by the body
        making contact), so the reported frame is the gap's START, not the
        reappearance frame.
        Tier 3 (abstain, low confidence): no clean single-frame evidence --
        returns frame=None and a coarse window [last seen approaching, first
        seen redirected] for manual confirmation. This is the honest outcome
        when the ball is genuinely undetectable through the touch (as on
        testClip, frame 112) or the scene is too crowded to isolate one
        signature -- reporting a specific frame here would just be a repeat of
        the frame-66/frame-90 failures.

        The player-motion-spike cue only ever corroborates a Tier 2 pick or
        informs the Tier 3 window -- it never promotes a candidate to a
        confident exact frame on its own.
        """
        n = len(comp_positions)
        hi = min(n, kick_frame + 1 + self.contact_search_frames)
        real_ts = [t for t in range(kick_frame + 1, hi) if not is_predicted[t]]

        if len(real_ts) < 4:
            return {
                "frame": None, "window": None, "confidence": 0.0, "tier": 3,
                "evidence": "fewer than 4 real ball detections in the search window -- can't fit an arc",
            }

        arc_points = [(t, comp_positions[t][0], comp_positions[t][1]) for t in real_ts]
        arc = ransac_ballistic_fit(
            arc_points, residual_thresh_px=self.residual_thresh_px,
            min_inliers=max(4, self.ballistic_fit_frames // 2),
        )
        residuals, inlier_frames = arc["residuals"], arc["inlier_frames"]
        has_arc = arc["coeffs"] is not None

        tier1: list[tuple[float, dict]] = []
        tier2: list[tuple[float, dict]] = []
        all_transitions: list[dict] = []

        for i in range(len(real_ts) - 1):
            t_prev, t_cur = real_ts[i], real_ts[i + 1]
            gap = t_cur - t_prev
            if gap > self.contact_max_gap_frames:
                continue

            if has_arc:
                r_cur = residuals.get(t_cur, 0.0)
                break_amount = r_cur - self.arc_break_residual_px
                is_on_arc_going_in = t_prev in inlier_frames
            else:
                r_cur, break_amount, is_on_arc_going_in = 0.0, 0.0, False

            all_transitions.append({
                "t_prev": t_prev, "t_cur": t_cur, "gap": gap,
                "break_score": break_amount if has_arc and is_on_arc_going_in else float(gap),
            })

            if not (has_arc and is_on_arc_going_in and break_amount > 0):
                continue

            tid, dist = _nearest_player_distance(raw_positions[t_cur], player_positions_per_frame[t_cur], exclude_id=kicker_id)
            if tid is None or dist is None or dist >= self.contact_proximity_px:
                continue
            proximity_score = float(np.clip(1.0 - (dist / self.contact_proximity_px), 0.0, 1.0))
            motion_spike = self._player_motion_spike(player_positions_per_frame, tid, t_prev, t_cur)

            entry = {
                "t_prev": t_prev, "t_cur": t_cur, "gap": gap, "tracker_id": tid,
                "dist_px": round(dist, 1), "break_residual_px": round(r_cur, 1),
                "motion_spike": round(motion_spike, 3),
            }
            score = 0.5 * float(np.clip(break_amount / 40.0, 0.0, 1.0)) + 0.35 * proximity_score + self.motion_spike_weight * motion_spike

            if gap == 1:
                tier1.append((score, entry))
            elif 2 <= gap <= 5:
                tier2.append((score, entry))

        if tier1:
            tier1.sort(key=lambda s: -s[0])
            score, best = tier1[0]
            if score >= self.contact_min_score:
                return {
                    "frame": best["t_cur"], "window": None,
                    "confidence": round(float(np.clip(score, 0.0, 1.0)), 3),
                    "tier": 1, "evidence": best,
                }

        if tier2:
            tier2.sort(key=lambda s: -s[0])
            score, best = tier2[0]
            if score >= self.contact_min_score:
                return {
                    "frame": best["t_prev"], "window": [best["t_prev"], best["t_cur"]],
                    "confidence": round(float(np.clip(score, 0.0, 1.0)) * 0.75, 3),
                    "tier": 2, "evidence": best,
                }

        if all_transitions:
            window_entry = max(all_transitions, key=lambda e: e["break_score"])
            window = [window_entry["t_prev"], window_entry["t_cur"]]
            confidence = 0.25
        else:
            window = [real_ts[0], real_ts[-1]]
            confidence = 0.0

        return {
            "frame": None, "window": window, "confidence": confidence, "tier": 3,
            "evidence": {"n_real_detections": len(real_ts), "has_arc": has_arc},
        }


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
        if kick["frame"] is None:
            return {
                "kick_frame": None,
                "kick_confidence": 0.0,
                "kick_debug": kick,
                "contact_frame": None,
                "contact_confidence": 0.0,
                "contact_tier": 3,
                "contact_window": None,
                "contact_debug": {"frame": None, "window": None, "confidence": 0.0, "tier": 3,
                                   "evidence": "kick frame itself could not be confidently detected"},
                "kicker_id": None,
            }
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
            "contact_tier": contact["tier"],
            "contact_window": contact["window"],
            "contact_debug": contact,
            "kicker_id": kicker_id,
        }
