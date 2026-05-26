import csv
import json
from pathlib import Path

class DataExporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        from .output_schema import OutputFiles, AnalyticsCSVColumns
        self.csv_path = self.output_dir / OutputFiles.ANALYTICS
        self.json_path = self.output_dir / OutputFiles.ANALYTICS_JSON
        
        self.frame_data = []

        # storage for enriched output
        self.match_info = {}
        self.events = []
        self._last_speeds = {}  # tracker_id -> last speed in m/s
        self.fps = 25.0

        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(AnalyticsCSVColumns.all_columns())
            
        self.current_possessor_id = None
        self.contact_frames = 0
        self.loose_ball_frames = 0
        self.last_ball_possessor = None # (player_id, frame_idx)
        self.passes = []

        # Thresholds
        self.POSSESSION_RADIUS = 1.5   # meters
        self.MIN_POSSESSION_FRAMES = 5 # smoothing
        self.MIN_LOOSE_FRAMES = 15     # before declaring loose
        self.PASS_SPEED_THRESHOLD = 20 # km/h

    def log_frame(self, frame_idx: int, frame_objects: list[dict]):
        self.frame_data.append({
            "frame": frame_idx,
            "objects": frame_objects
        })
        
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            for obj in frame_objects:
                writer.writerow([
                    frame_idx,
                    obj.get("id", ""),
                    obj.get("class", "unknown"),
                    obj.get("team", ""),
                    round(obj.get("x_m", 0.0), 2),
                    round(obj.get("y_m", 0.0), 2),
                    round(obj.get("speed", 0.0), 2) if "speed" in obj else "",
                    round(obj.get("distance", 0.0), 2) if "distance" in obj else ""
                ])

        # Update last speed tracking for acceleration computation later
        for obj in frame_objects:
            tracker_id = obj.get("id")
            if tracker_id is None:
                continue

            # Prefer explicit km/h fields if present
            speed_kmh = None
            if "speed_kmh" in obj:
                speed_kmh = obj.get("speed_kmh")
            elif "speed_km_h" in obj:
                speed_kmh = obj.get("speed_km_h")
            elif "speed" in obj:
                speed_kmh = obj.get("speed")

            if speed_kmh is None:
                continue

            try:
                speed_kmh = float(speed_kmh)
            except Exception:
                continue

            speed_ms = speed_kmh / 3.6

            # compute acceleration using previous speed if present
            try:
                tid = int(tracker_id)
            except Exception:
                tid = None

            if tid is not None and tid in self._last_speeds:
                prev_v = self._last_speeds.get(tid, 0.0)
                # dt in seconds
                dt = 1.0 / float(self.fps) if self.fps > 0 else 1.0
                acceleration = (speed_ms - prev_v) / dt
                # Attach acceleration to the object for later serialization
                try:
                    obj["acceleration"] = round(acceleration, 2)
                except Exception:
                    pass

            try:
                self._last_speeds[tid] = speed_ms
            except Exception:
                # ignore non-int ids
                pass

    def update_passes(self, frame_idx: int, ball_pos: tuple[float, float], player_positions: dict[int, tuple[float, float]], ball_speed_kmh: float = 0.0):
        if ball_pos == (0.0, 0.0):
            return
            
        closest_dist = float('inf')
        closest_player_id = None
        
        for p_id, p_pos in player_positions.items():
            dist = ((ball_pos[0] - p_pos[0])**2 + (ball_pos[1] - p_pos[1])**2)**0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_player_id = p_id
        
        # 1. Logic for "Potential Touch"
        in_range = closest_player_id is not None and closest_dist <= self.POSSESSION_RADIUS

        # 2. Update Possession State
        if in_range:
            if closest_player_id == self.current_possessor_id:
                # Still has it
                self.contact_frames += 1
                self.loose_ball_frames = 0
            else:
                # Might be a change or new gain
                self.contact_frames += 1
                if self.contact_frames >= self.MIN_POSSESSION_FRAMES:
                    # VALID CHANGE OF POSSESSION
                    if self.current_possessor_id is not None:
                        # Detect if this was a pass
                        if ball_speed_kmh > self.PASS_SPEED_THRESHOLD:
                            pass_record = {
                                "passer_id": self.current_possessor_id,
                                "receiver_id": closest_player_id,
                                "start_frame": self.last_ball_possessor[1] if self.last_ball_possessor else 0,
                                "end_frame": frame_idx,
                                "ball_speed": round(ball_speed_kmh, 2)
                            }
                            self.passes.append(pass_record)
                            print(f"[EVENT] Pass detected: Player {self.current_possessor_id} -> Player {closest_player_id} (Speed: {ball_speed_kmh:.1f} km/h)")
                    
                    self.current_possessor_id = closest_player_id
                    self.last_ball_possessor = (closest_player_id, frame_idx)
                    self.contact_frames = 0
                    self.loose_ball_frames = 0
        else:
            # Ball is away from everyone
            self.contact_frames = 0
            if self.current_possessor_id is not None:
                self.loose_ball_frames += 1
                # Lose possession if away for too long or kicked hard
                if self.loose_ball_frames > self.MIN_LOOSE_FRAMES or ball_speed_kmh > self.PASS_SPEED_THRESHOLD:
                    self.current_possessor_id = None
                    self.loose_ball_frames = 0

    def set_match_info(self, match_info: dict):
        """Store match-level metadata used in the final JSON output.

        Example: {"match_id":"...","home_team":"...","away_team":"...","period": "1"}
        """
        self.match_info = dict(match_info) if match_info is not None else {}

    def add_event(self, event: dict):
        """Append a single event record produced by EventsDetector.

        Event must be a dict following agreed schema. Exporter will include events
        in the final `events` array. If no events are added, existing `passes` are used.
        """
        if event is None:
            return
        self.events.append(event)

    def set_fps(self, fps: float):
        """Set the processed frames-per-second used to compute timestamps and acceleration."""
        try:
            self.fps = float(fps)
        except Exception:
            pass

    def finalize(self):
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        # Build enriched frames and events output
        serialized_frames = []
        match_start_ms = int(self.match_info.get("match_start_ms", 0)) if self.match_info else 0
        home_team_name = self.match_info.get("home_team") if self.match_info else None
        away_team_name = self.match_info.get("away_team") if self.match_info else None

        team_id_map = self.match_info.get("team_id_map") if self.match_info else None

        for frame_entry in self.frame_data:
            fidx = frame_entry.get("frame")
            objs = frame_entry.get("objects", [])

            players = []
            ball = None

            for obj in objs:
                cls = obj.get("class", "")
                if cls == "ball":
                    # ball speed conversion
                    speed_kmh = obj.get("speed_kmh") if obj.get("speed_kmh") is not None else obj.get("speed")
                    speed_ms = round(float(speed_kmh) / 3.6, 2) if speed_kmh is not None else 0.0
                    ball = {
                        "x": round(obj.get("x_m", 0.0), 2),
                        "y": round(obj.get("y_m", 0.0), 2),
                        "z": obj.get("z", 0),
                        "speed_ms": speed_ms,
                        "possession_team": self.match_info.get("possession_team") if self.match_info else None,
                        "possession_player_id": self.current_possessor_id,
                    }
                    continue

                if cls in ("player", "referee"):
                    tracker_id = obj.get("id")
                    raw_team = obj.get("team", "Unknown")

                    # Determine team side: prefer explicit mapping in match_info
                    team_side = None
                    if isinstance(team_id_map, dict):
                        # raw_team may be like 'Team 0' — try to parse the numeric id
                        try:
                            if isinstance(raw_team, str) and raw_team.startswith("Team "):
                                tid = int(raw_team.split()[1])
                                team_side = team_id_map.get(tid)
                        except Exception:
                            team_side = None

                    # Fallback: match by team name if provided
                    if team_side is None and home_team_name and isinstance(raw_team, str) and home_team_name in raw_team:
                        team_side = "home"
                    if team_side is None and away_team_name and isinstance(raw_team, str) and away_team_name in raw_team:
                        team_side = "away"

                    # Final fallback: keep raw_team
                    team = team_side if team_side is not None else raw_team

                    # Player id formatting — use home_/away_ prefix when team_side known
                    if tracker_id is not None:
                        if team_side in ("home", "away"):
                            player_id = f"{team_side}_{tracker_id}"
                        else:
                            player_id = f"player_{tracker_id}"
                    else:
                        player_id = "unknown"

                    # speed: convert from km/h to m/s if possible
                    speed_kmh = obj.get("speed_kmh") if obj.get("speed_kmh") is not None else obj.get("speed")
                    speed_ms = round(float(speed_kmh) / 3.6, 2) if speed_kmh is not None else 0.0

                    # acceleration: use provided field or default 0.0
                    acceleration = obj.get("acceleration", 0.0)

                    in_poss = obj.get("possession", False)
                    # If exporter-level possessor known, use it
                    try:
                        if tracker_id is not None and int(tracker_id) == int(self.current_possessor_id):
                            in_poss = True
                    except Exception:
                        pass

                    players.append({
                        "player_id": player_id,
                        "team": team,
                        "jersey_number": obj.get("jersey_number", None),
                        "position_label": obj.get("position_label", None),
                        "x": round(obj.get("x_m", 0.0), 2),
                        "y": round(obj.get("y_m", 0.0), 2),
                        "speed_ms": speed_ms,
                        "acceleration": acceleration,
                        "in_possession": bool(in_poss),
                    })

            # default ball object if missing
            if ball is None:
                ball = {"x": 0.0, "y": 0.0, "z": 0, "speed_ms": 0.0, "possession_team": None, "possession_player_id": None}

            # compute timestamp (game_clock formatting handled by integration utilities)
            elapsed_ms = int((fidx / self.fps) * 1000)
            timestamp_ms = match_start_ms + elapsed_ms
            game_clock = None

            # Build per-frame record
            serialized_frames.append({
                "frame_id": fidx,
                "timestamp_ms": int(timestamp_ms),
                "game_clock": game_clock,
                "period": self.match_info.get("period") if self.match_info else None,
                "players": players,
                "ball": ball,
            })

        # Decide events array (prefer explicit events produced by EventsDetector)
        events_array = self.events if self.events else self.passes

        # Write separate events.json for consumers that expect only events
        events_path = self.output_dir / "events.json"
        from .output_schema import write_json_atomic
        write_json_atomic(events_path, convert_numpy(events_array))

        # Analytics JSON: match_info + frames + metadata (no events included here)
        final_data = {
            "match_info": self.match_info or {},
            "frames": serialized_frames,
            "metadata": {"total_passes": len(self.passes), "frames": len(serialized_frames)},
        }

        final_data = convert_numpy(final_data)
        write_json_atomic(self.json_path, final_data)

        print(f"Data exporter finalized. Saved {self.csv_path}, {self.json_path}, and {events_path}")
