"""
src/pipeline/events.py
======================
EventsDetector – frame-level event detection for football tracking.

Detects the following event types:
  • pass        – pass completed to a teammate or intercepted by opponent
  • skill_move  – player performs a confirmed direction-change with the ball
  • cross       – confirmed high-speed ball trajectory from a wide position
                  toward the box

Attempt events (e.g. PASS_ATTEMPT) where the outcome is unknown are **not** emitted.

All events are reported by calling ``data_exporter.add_event(event_dict)``
rather than being written to disk directly.

Event dict schema
-----------------
{
    "type":         str,           # "pass" | "skill_move" | "cross"
    "frame":        int,           # frame index where event was detected
    "timestamp_ms": int,           # millisecond timestamp (frame / fps * 1000)
    "game_clock":   str,           # formatted game clock e.g. "01:23"

    # -- pass only --
    "passer_id":    int | None,
    "receiver_id":  int | None,
    "passer_team":  str | None,    # e.g. "Team 0"
    "receiver_team":str | None,
    "ball_speed_kmh": float,

    # -- skill_move only --
    "player_id":    int | None,
    "team":         str | None,
    "direction_change_deg": float, # angle of the direction change

    # -- cross only --
    "player_id":    int | None,
    "team":         str | None,
    "origin_x_m":  float,          # pitch-space origin
    "origin_y_m":  float,
    "ball_speed_kmh": float,
}
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from src.utils.integration_helpers import format_game_clock

if TYPE_CHECKING:
    from .data_exporter import DataExporter


# ---------------------------------------------------------------------------
# Tunable thresholds (all distances in metres, speeds in km/h, angles in deg)
# ---------------------------------------------------------------------------

# --- Pass ---
_PASS_POSSESSION_RADIUS_M: float = 3.5       # ball must be this close to grant possession
_PASS_MIN_POSSESSION_FRAMES: int = 2         # frames of contact needed to confirm possession
_PASS_MIN_LOOSE_FRAMES: int = 8             # frames the ball must be free before a pass is logged
_PASS_MIN_BALL_SPEED_KMH: float = 5.0       # minimum ball speed to even consider a pass
_PASS_HIGH_SPEED_RELEASE_KMH: float = 15.0  # immediate possession-loss if ball this fast

# --- Skill move ---
_SKILL_MIN_POSSESSION_FRAMES: int = 6        # player must be near ball for at least N frames
_SKILL_MIN_DIRECTION_CHANGE_DEG: float = 60.0  # minimum angle of direction-change
_SKILL_MAX_BALL_SPEED_KMH: float = 35.0      # ball must not be flying (it's a dribble)
_SKILL_COOLDOWN_FRAMES: int = 30             # don't re-detect for this many frames per player

# --- Cross ---
_CROSS_WIDE_X_FRACTION: float = 0.20        # within 20 % of pitch width from touchline (x < 21 m or x > 84 m)
_CROSS_MIN_BALL_SPEED_KMH: float = 30.0     # fast kicks only
_CROSS_MIN_INWARD_Y_M: float = 6.0          # ball must be heading toward the box (|Δy| > threshold)
_CROSS_COOLDOWN_FRAMES: int = 40            # minimum frames between consecutive cross detections


class EventsDetector:
    """
    Stateful per-frame event detector.

    Parameters
    ----------
    fps : float
        Frames per second of the processed video – used for timestamp calculation.
    pitch_width_m : float
        Width of the pitch in metres (default 105 m).
    pitch_height_m : float
        Height of the pitch in metres (default 68 m).
    """

    def __init__(
        self,
        fps: float = 25.0,
        pitch_width_m: float = 105.0,
        pitch_height_m: float = 68.0,
    ) -> None:
        self.fps = max(1.0, float(fps))
        self.pitch_width_m = float(pitch_width_m)
        self.pitch_height_m = float(pitch_height_m)

        # ------------------------------------------------------------------ #
        # Possession / pass state                                              #
        # ------------------------------------------------------------------ #
        self._state: str = "LOOSE_BALL"
        self._possessor_id: int | None = None
        self._possessor_team: str | None = None
        self._contact_frames: int = 0
        self._candidate_id: int | None = None
        self._candidate_team: str | None = None
        self._pass_start_frame: int = 0
        self._pass_start_xy: tuple[float, float] = (0.0, 0.0)
        self._pass_id_counter: int = 1
        self._pass_duration_frames: int = 0
        self._MAX_PASS_DURATION_FRAMES: int = 75
        self._last_possessor_id: int | None = None
        self._last_possessor_team: str | None = None
        self._last_possession_frame: int = 0

        # ------------------------------------------------------------------ #
        # Skill-move state                                                     #
        # ------------------------------------------------------------------ #
        # Per-player: deque of recent (frame_idx, x_m, y_m) while in possession
        self._player_ball_contacts: dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
        self._skill_cooldown: dict[int, int] = {}  # player_id -> last skill frame

        # ------------------------------------------------------------------ #
        # Cross state                                                          #
        # ------------------------------------------------------------------ #
        self._cross_last_frame: int = -_CROSS_COOLDOWN_FRAMES  # last detected cross frame

        # ------------------------------------------------------------------ #
        # Ball trail (last N pitch-space positions)                           #
        # ------------------------------------------------------------------ #
        self._ball_trail: deque = deque(maxlen=30)   # (frame_idx, x_m, y_m)

    # ---------------------------------------------------------------------- #
    # Public API                                                              #
    # ---------------------------------------------------------------------- #

    def process_frame(
        self,
        frame_idx: int,
        ball_pos_m: tuple[float, float],
        player_positions: dict[int, tuple[float, float]],
        player_teams: dict[int, str],
        ball_speed_kmh: float,
        data_exporter: "DataExporter",
        ball_trail: list[tuple[float, float]] | None = None,
    ) -> None:
        """
        Analyse a single frame and emit any detected events via
        ``data_exporter.add_event()``.

        Parameters
        ----------
        frame_idx : int
            Current (processed) frame index.
        ball_pos_m : tuple[float, float]
            Ball position in pitch metres ``(x_m, y_m)``.
        player_positions : dict[int, tuple[float, float]]
            Mapping of tracker-id → pitch-space position ``(x_m, y_m)``.
        player_teams : dict[int, str]
            Mapping of tracker-id → team string, e.g. ``"Team 0"``.
        ball_speed_kmh : float
            Current estimated ball speed in km/h.
        data_exporter : DataExporter
            The shared exporter; events are submitted via ``add_event()``.
        ball_trail : list[tuple[float, float]] | None
            Optional list of recent ball pixel-space positions (from
            ``BallTracker.get_trail()``).  When provided, used to compute
            ball direction for cross and skill-move detection.  If ``None``
            the pitch-space trail is used instead.
        """
        timestamp_ms = int((frame_idx / self.fps) * 1000)

        # Update internal pitch-space ball trail
        bx, by = float(ball_pos_m[0]), float(ball_pos_m[1])
        self._ball_trail.append((frame_idx, bx, by))

        # ---- Run detectors -----------------------------------------------
        self._detect_pass(
            frame_idx, timestamp_ms, bx, by,
            player_positions, player_teams, ball_speed_kmh, data_exporter,
        )
        self._detect_skill_move(
            frame_idx, timestamp_ms, bx, by,
            player_positions, player_teams, ball_speed_kmh, data_exporter,
        )
        self._detect_cross(
            frame_idx, timestamp_ms, bx, by,
            player_positions, player_teams, ball_speed_kmh, data_exporter,
        )

    # ---------------------------------------------------------------------- #
    # Pass detection                                                          #
    # ---------------------------------------------------------------------- #

    def _find_closest_player(
        self,
        bx: float,
        by: float,
        player_positions: dict[int, tuple[float, float]],
    ) -> tuple[int | None, float]:
        """Return (player_id, distance_m) of the closest player to the ball."""
        best_id: int | None = None
        best_dist = float("inf")
        for pid, (px, py) in player_positions.items():
            d = math.hypot(bx - px, by - py)
            if d < best_dist:
                best_dist = d
                best_id = pid
        return best_id, best_dist

    def _detect_pass(
        self,
        frame_idx: int,
        timestamp_ms: int,
        bx: float,
        by: float,
        player_positions: dict[int, tuple[float, float]],
        player_teams: dict[int, str],
        ball_speed_kmh: float,
        data_exporter: "DataExporter",
    ) -> None:
        closest_id, closest_dist = self._find_closest_player(bx, by, player_positions)
        
        confidence = 0.0
        if closest_id is not None and closest_dist <= _PASS_POSSESSION_RADIUS_M:
            confidence = max(0.0, 1.0 - (closest_dist / _PASS_POSSESSION_RADIUS_M))
            if closest_id == self._candidate_id:
                self._contact_frames += 1
            else:
                self._candidate_id = closest_id
                self._candidate_team = player_teams.get(closest_id)
                self._contact_frames = 1
        else:
            self._contact_frames = 0
            self._candidate_id = None
            self._candidate_team = None

        is_controlled = (confidence > 0.75 and self._contact_frames >= 2) or (self._contact_frames >= _PASS_MIN_POSSESSION_FRAMES)

        if self._state == "LOOSE_BALL":
            if is_controlled and self._candidate_id is not None:
                self._state = "CONTROLLED"
                self._possessor_id = self._candidate_id
                self._possessor_team = self._candidate_team
                self._last_possessor_id = self._possessor_id
                self._last_possessor_team = self._possessor_team
                self._last_possession_frame = frame_idx

        elif self._state == "CONTROLLED":
            if is_controlled and self._candidate_id == self._possessor_id:
                self._last_possession_frame = frame_idx
            elif is_controlled and self._candidate_id != self._possessor_id:
                self._state = "PASS_IN_PROGRESS"
                self._pass_start_frame = self._last_possession_frame
                self._pass_start_xy = (bx, by)
                self._pass_duration_frames = frame_idx - self._pass_start_frame
            else:
                if ball_speed_kmh >= 5.0:
                    self._state = "PASS_IN_PROGRESS"
                    self._pass_start_frame = self._last_possession_frame
                    self._pass_start_xy = (bx, by)
                    self._pass_duration_frames = 0
                    # No PASS_ATTEMPT event is emitted; we wait for the
                    # outcome (completion or interception) before reporting.
                else:
                    self._state = "LOOSE_BALL"
                    self._possessor_id = None
                    self._possessor_team = None

        if self._state == "PASS_IN_PROGRESS":
            self._pass_duration_frames += 1
            has_touch = (closest_id is not None and closest_dist <= _PASS_POSSESSION_RADIUS_M)
            
            if has_touch:
                if closest_id == self._possessor_id:
                    self._state = "CONTROLLED"
                else:
                    receiver_id = closest_id
                    receiver_team = player_teams.get(receiver_id)
                    same_team = (receiver_team == self._possessor_team and receiver_team is not None)
                    
                    dist_m = math.hypot(bx - self._pass_start_xy[0], by - self._pass_start_xy[1])
                    if dist_m > 25.0:
                        pass_type = "long_ball"
                    
                    dist_valid = dist_m > 2.0
                    speed_valid = ball_speed_kmh < 150.0

                    elapsed_seconds = frame_idx / self.fps
                    game_clock = format_game_clock(elapsed_seconds)

                    completed_event = {
                        "type": "pass",
                        "pass_id": self._pass_id_counter,
                        "frame": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "game_clock": game_clock,
                        "passer_id": self._possessor_id,
                        "passer_team": self._possessor_team,
                        "receiver_id": receiver_id,
                        "receiver_team": receiver_team,
                        "ball_speed_kmh": round(float(ball_speed_kmh), 2),
                        "start_xy": [round(self._pass_start_xy[0], 2), round(self._pass_start_xy[1], 2)],
                        "end_xy": [round(bx, 2), round(by, 2)],
                        "successful": same_team,
                        "distance_m": round(dist_m, 2),
                        "duration_frames": self._pass_duration_frames,
                        "validation": {
                            "same_team": same_team,
                            "distance_valid": dist_valid,
                            "speed_valid": speed_valid
                        }
                    }
                    data_exporter.add_event(completed_event)
                    self._pass_id_counter += 1
                    
                    self._state = "CONTROLLED"
                    self._possessor_id = receiver_id
                    self._possessor_team = receiver_team
                    self._last_possessor_id = receiver_id
                    self._last_possessor_team = receiver_team
                    self._last_possession_frame = frame_idx
                    
            elif self._pass_duration_frames > self._MAX_PASS_DURATION_FRAMES:
                self._state = "LOOSE_BALL"
                self._possessor_id = None
                self._possessor_team = None

    # ---------------------------------------------------------------------- #
    # Skill-move detection                                                    #
    # ---------------------------------------------------------------------- #

    def _detect_skill_move(
        self,
        frame_idx: int,
        timestamp_ms: int,
        bx: float,
        by: float,
        player_positions: dict[int, tuple[float, float]],
        player_teams: dict[int, str],
        ball_speed_kmh: float,
        data_exporter: "DataExporter",
    ) -> None:
        """
        Detect skill moves: player keeps the ball close while making a sharp
        direction change (≥ 60 °) with a moderate ball speed (not a clearance).

        Strategy:
          1. For each player within possession radius, record their ball-contact
             position history.
          2. When ≥ MIN_FRAMES of contacts exist, check the direction change
             between the first half and the second half of the history window.
          3. Emit skill_move if angle exceeds threshold.
        """
        if ball_speed_kmh > _SKILL_MAX_BALL_SPEED_KMH:
            # Ball moving too fast – it's a kick, not a dribble
            return

        closest_id, closest_dist = self._find_closest_player(bx, by, player_positions)
        if closest_id is None or closest_dist > _PASS_POSSESSION_RADIUS_M:
            return

        # Accumulate contact history for this player
        history = self._player_ball_contacts[closest_id]
        history.append((frame_idx, bx, by))

        if len(history) < _SKILL_MIN_POSSESSION_FRAMES:
            return

        # Cooldown check
        last_skill = self._skill_cooldown.get(closest_id, -_SKILL_COOLDOWN_FRAMES)
        if frame_idx - last_skill < _SKILL_COOLDOWN_FRAMES:
            return

        # Compute direction change: vector across first half vs second half
        mid = len(history) // 2
        early = list(history)[:mid]
        late = list(history)[mid:]

        dx_early = early[-1][1] - early[0][1]
        dy_early = early[-1][2] - early[0][2]
        dx_late = late[-1][1] - late[0][1]
        dy_late = late[-1][2] - late[0][2]

        mag_early = math.hypot(dx_early, dy_early)
        mag_late = math.hypot(dx_late, dy_late)

        if mag_early < 0.3 or mag_late < 0.3:
            # Barely moved – not a dribble
            return

        cos_angle = (
            (dx_early * dx_late + dy_early * dy_late)
            / (mag_early * mag_late)
        )
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))

        if angle_deg >= _SKILL_MIN_DIRECTION_CHANGE_DEG:
            elapsed_seconds = frame_idx / self.fps
            game_clock = format_game_clock(elapsed_seconds)

            event = {
                "type": "skill_move",
                "frame": frame_idx,
                "timestamp_ms": timestamp_ms,
                "game_clock": game_clock,
                "player_id": closest_id,
                "team": player_teams.get(closest_id),
                "direction_change_deg": round(angle_deg, 1),
                "ball_speed_kmh": round(float(ball_speed_kmh), 2),
                # pass-only fields
                "passer_id": None,
                "receiver_id": None,
                "passer_team": None,
                "receiver_team": None,
                # cross-only fields
                "origin_x_m": None,
                "origin_y_m": None,
            }
            data_exporter.add_event(event)
            self._skill_cooldown[closest_id] = frame_idx
            # Clear history so we do not double-fire on the same move
            history.clear()

    # ---------------------------------------------------------------------- #
    # Cross detection                                                         #
    # ---------------------------------------------------------------------- #

    def _detect_cross(
        self,
        frame_idx: int,
        timestamp_ms: int,
        bx: float,
        by: float,
        player_positions: dict[int, tuple[float, float]],
        player_teams: dict[int, str],
        ball_speed_kmh: float,
        data_exporter: "DataExporter",
    ) -> None:
        """
        Detect crosses: high-speed ball kicked from a wide position (within
        ``_CROSS_WIDE_X_FRACTION`` of either touchline) with a significant
        inward trajectory (toward the centre / box).

        Uses the last two pitch-space ball trail entries to determine direction.
        """
        # Cooldown guard
        if frame_idx - self._cross_last_frame < _CROSS_COOLDOWN_FRAMES:
            return

        if ball_speed_kmh < _CROSS_MIN_BALL_SPEED_KMH:
            return

        # Is the ball in a wide area?
        wide_threshold_m = self.pitch_width_m * _CROSS_WIDE_X_FRACTION
        is_wide_left = bx < wide_threshold_m
        is_wide_right = bx > (self.pitch_width_m - wide_threshold_m)
        if not (is_wide_left or is_wide_right):
            return

        # Check direction from ball trail (pitch space)
        if len(self._ball_trail) < 2:
            return

        # Use a few frames back to get a stable direction
        lookback = min(5, len(self._ball_trail) - 1)
        trail_list = list(self._ball_trail)
        _, prev_bx, prev_by = trail_list[-1 - lookback]
        delta_x = bx - prev_bx
        delta_y = by - prev_by

        # Ball must move inward (toward pitch centre in x) and significantly in y
        moving_inward = (is_wide_left and delta_x > 0) or (is_wide_right and delta_x < 0)
        if not moving_inward:
            return

        if abs(delta_y) < _CROSS_MIN_INWARD_Y_M:
            return

        # Attribute to the last known possessor (or closest current player)
        player_id: int | None = self._last_possessor_id or self._possessor_id
        team: str | None = None
        if player_id is not None:
            team = player_teams.get(player_id)
        else:
            # Fall back to closest player
            closest_id, closest_dist = self._find_closest_player(bx, by, player_positions)
            if closest_id is not None and closest_dist <= _PASS_POSSESSION_RADIUS_M * 2:
                player_id = closest_id
                team = player_teams.get(closest_id)

        elapsed_seconds = frame_idx / self.fps
        game_clock = format_game_clock(elapsed_seconds)

        event = {
            "type": "cross",
            "frame": frame_idx,
            "timestamp_ms": timestamp_ms,
            "game_clock": game_clock,
            "player_id": player_id,
            "team": team,
            "origin_x_m": round(bx, 2),
            "origin_y_m": round(by, 2),
            "ball_speed_kmh": round(float(ball_speed_kmh), 2),
            # pass-only fields
            "passer_id": None,
            "receiver_id": None,
            "passer_team": None,
            "receiver_team": None,
            # skill-only fields
            "direction_change_deg": None,
        }
        data_exporter.add_event(event)
        self._cross_last_frame = frame_idx

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    @property
    def current_possessor_id(self) -> int | None:
        """Tracker-id of the player currently in possession, or ``None``."""
        return self._possessor_id

    @property
    def current_possessor_team(self) -> str | None:
        """Team string of current possessor, or ``None``."""
        return self._possessor_team
