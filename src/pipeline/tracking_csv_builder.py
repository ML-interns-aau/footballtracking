"""
tracking_csv_builder.py
========================
Builds a professional-grade football tracking CSV with:
  - Persistent track IDs (live mode: ByteTrack IDs / offline mode: IoU-based assignment)
  - Motion features: velocity, speed, acceleration, direction (EMA-smoothed)
  - Pitch coordinates via homography (PitchMapper)
  - Football context: possession, role, goalkeeper flag, team side, zone, distance to ball
  - Missing-frame interpolation (linear, up to 10-frame gaps)
  - Clean 27-column output CSV ready for analysis / ML
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
#  IoU helper – used for offline nearest-neighbour tracker
# ---------------------------------------------------------------------------
def _iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    inter_x1 = max(xa1, xb1); inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2); inter_y2 = min(ya2, yb2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = (xa2 - xa1) * (ya2 - ya1)
    areaB = (xb2 - xb1) * (yb2 - yb1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
#  Lightweight IoU nearest-neighbour tracker (offline mode only)
# ---------------------------------------------------------------------------
class _IoUTracker:
    """
    Frame-by-frame IoU tracker used when ByteTrack IDs are unavailable.
    Assigns stable track_ids by matching each new detection to the closest
    existing active track (IoU >= iou_threshold). Unmatched detections get
    a new incrementing ID. Tracks inactive for > max_missed frames are dropped.
    """
    def __init__(self, iou_threshold: float = 0.30, max_missed: int = 10):
        self.iou_threshold = iou_threshold
        self.max_missed    = max_missed
        self._next_id      = 1
        self._tracks: Dict[int, Dict] = {}

    def update(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        if len(boxes_xyxy) == 0:
            for t in self._tracks.values():
                t["missed"] += 1
            self._purge()
            return np.array([], dtype=np.int32)

        assigned = np.full(len(boxes_xyxy), -1, dtype=np.int32)
        track_ids = list(self._tracks.keys())

        if track_ids:
            iou_matrix = np.zeros((len(boxes_xyxy), len(track_ids)))
            for di, dbox in enumerate(boxes_xyxy):
                for ti, tid in enumerate(track_ids):
                    iou_matrix[di, ti] = _iou(dbox, self._tracks[tid]["box"])

            used_tracks = set()
            for _ in range(min(len(boxes_xyxy), len(track_ids))):
                if iou_matrix.max() < self.iou_threshold:
                    break
                di, ti = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[ti]
                if assigned[di] == -1 and tid not in used_tracks:
                    assigned[di] = tid
                    used_tracks.add(tid)
                    self._tracks[tid]["box"]    = boxes_xyxy[di]
                    self._tracks[tid]["missed"] = 0
                iou_matrix[di, :] = -1
                iou_matrix[:, ti] = -1

            for tid in track_ids:
                if tid not in used_tracks:
                    self._tracks[tid]["missed"] += 1

        for di in range(len(boxes_xyxy)):
            if assigned[di] == -1:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = {"box": boxes_xyxy[di], "missed": 0}
                assigned[di] = new_id

        self._purge()
        return assigned

    def _purge(self):
        to_del = [tid for tid, t in self._tracks.items() if t["missed"] > self.max_missed]
        for tid in to_del:
            del self._tracks[tid]


# ---------------------------------------------------------------------------
#  Main class
# ---------------------------------------------------------------------------
class TrackingCSVBuilder:
    """
    Collects per-frame detections (live or from CSV) and produces a structured
    tracking CSV with all spatial, motion, pitch, and football-context features.
    """

    BALL_CLASS_ID    = 32
    PITCH_W          = 105.0
    PITCH_H          = 68.0
    POSSESSION_RADIUS = 2.0

    CSV_COLUMNS = [
        "frame", "track_id", "team_id", "player_id",
        "bb_left", "bb_top", "bb_width", "bb_height",
        "center_x", "center_y",
        "velocity_x", "velocity_y", "speed", "acceleration", "direction",
        "pitch_x", "pitch_y",
        "is_ball", "possession", "role", "is_goalkeeper", "team_side",
        "distance_to_ball", "in_possession_zone",
        "confidence", "extra",
    ]

    def __init__(
        self,
        pitch_mapper=None,
        fps: float = 30.0,
        ema_alpha: float = 0.35,
        interpolate_max_gap: int = 10,
        video_width: int = 1920,
        video_height: int = 1080,
    ):
        self.pitch_mapper        = pitch_mapper
        self.fps                 = fps
        self.ema_alpha           = ema_alpha
        self.interpolate_max_gap = interpolate_max_gap
        self.video_width         = video_width
        self.video_height        = video_height
        self._offline_tracker    = _IoUTracker()
        self._records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    #  Public API — LIVE mode
    # ------------------------------------------------------------------
    def add_frame(self, frame_idx: int, detections, team_ids=None):
        if detections is None or len(detections) == 0:
            return

        team_ids_arr = (
            np.asarray(team_ids, dtype=np.int32)
            if team_ids is not None
            else np.full(len(detections), -1, dtype=np.int32)
        )

        for i in range(len(detections)):
            bbox     = detections.xyxy[i]
            t_id     = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            class_id = int(detections.class_id[i])  if detections.class_id  is not None else -1
            conf     = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            team_id  = int(team_ids_arr[i])

            self._records.append(self._make_record(
                frame_idx=frame_idx, track_id=t_id, team_id=team_id,
                player_id=t_id, bb_left=float(bbox[0]), bb_top=float(bbox[1]),
                bb_width=float(bbox[2] - bbox[0]), bb_height=float(bbox[3] - bbox[1]),
                is_ball=1 if class_id == self.BALL_CLASS_ID else 0,
                confidence=conf, extra=-1,
            ))

    # ------------------------------------------------------------------
    #  Public API — OFFLINE mode
    # ------------------------------------------------------------------
    def load_from_csv(self, input_path: str):
        df = pd.read_csv(input_path)
        for frame_idx, grp in df.groupby("frame"):
            frame_idx = int(frame_idx)
            boxes = grp[["bb_left", "bb_top"]].copy()
            boxes["x2"] = grp["bb_left"] + grp["bb_width"]
            boxes["y2"] = grp["bb_top"]  + grp["bb_height"]
            boxes_xyxy = boxes[["bb_left", "bb_top", "x2", "y2"]].to_numpy()
            track_ids = self._offline_tracker.update(boxes_xyxy)

            for i, (_, row) in enumerate(grp.iterrows()):
                class_id = int(row["class_id"]) if "class_id" in row else -1
                team_id  = int(row["team_id"])
                is_ball  = 1 if (class_id == self.BALL_CLASS_ID or team_id == 32) else 0
                self._records.append(self._make_record(
                    frame_idx=frame_idx, track_id=int(track_ids[i]),
                    team_id=team_id, player_id=int(row["player_id"]),
                    bb_left=float(row["bb_left"]), bb_top=float(row["bb_top"]),
                    bb_width=float(row["bb_width"]), bb_height=float(row["bb_height"]),
                    is_ball=is_ball, confidence=float(row["confidence"]),
                    extra=int(row.get("extra", -1)),
                ))

    # ------------------------------------------------------------------
    #  Public API — export
    # ------------------------------------------------------------------
    def finalize_and_write(self, output_path) -> pd.DataFrame:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._records:
            print("[TrackingCSVBuilder] No detections — nothing to export.")
            return pd.DataFrame(columns=self.CSV_COLUMNS)

        df = pd.DataFrame(self._records)
        df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
        df = self._interpolate(df)
        df = self._apply_pitch_mapping(df)
        df = self._calculate_motion_features(df)
        df = self._infer_football_context(df)
        df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)

        final = df[self.CSV_COLUMNS].copy()
        num_cols = final.select_dtypes(include=[np.number]).columns
        final[num_cols] = final[num_cols].round(4)
        final.to_csv(output_path, index=False)
        print(f"[TrackingCSVBuilder] Success: Tracking CSV saved to {output_path}  ({len(final)} rows)")
        return final

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_record(frame_idx, track_id, team_id, player_id,
                     bb_left, bb_top, bb_width, bb_height,
                     is_ball, confidence, extra) -> Dict[str, Any]:
        return {
            "frame": frame_idx, "track_id": track_id, "team_id": team_id,
            "player_id": player_id, "bb_left": bb_left, "bb_top": bb_top,
            "bb_width": bb_width, "bb_height": bb_height,
            "center_x": bb_left + bb_width / 2.0,
            "center_y": bb_top + bb_height / 2.0,
            "is_ball": is_ball, "confidence": confidence, "extra": extra,
        }

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        spatial_cols = ["bb_left", "bb_top", "bb_width", "bb_height", "center_x", "center_y", "confidence"]
        chunks: List[pd.DataFrame] = []
        for tid, grp in df.groupby("track_id"):
            grp = grp.sort_values("frame")
            min_f, max_f = grp["frame"].min(), grp["frame"].max()
            full_idx = list(range(min_f, max_f + 1))
            grp = grp.set_index("frame").reindex(full_idx)
            for col in ["track_id", "team_id", "player_id", "is_ball", "extra"]:
                grp[col] = grp[col].ffill().bfill()
            for col in spatial_cols:
                grp[col] = grp[col].interpolate(
                    method="linear", limit=self.interpolate_max_gap, limit_direction="both"
                )
            grp = grp.reset_index().rename(columns={"index": "frame"})
            chunks.append(grp)
        return pd.concat(chunks, ignore_index=True) if chunks else df

    def _apply_pitch_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        foot_x = df["center_x"].copy()
        foot_y = (df["bb_top"] + df["bb_height"]).copy()
        ball_mask = df["is_ball"] == 1
        foot_y[ball_mask] = df.loc[ball_mask, "center_y"]
        df = df.copy()
        if self.pitch_mapper is not None:
            coords = np.column_stack([foot_x.to_numpy(), foot_y.to_numpy()])
            mapped = self.pitch_mapper.transform_points(coords)
            df["pitch_x"] = mapped[:, 0]
            df["pitch_y"] = mapped[:, 1]
        else:
            df["pitch_x"] = foot_x / self.video_width  * self.PITCH_W
            df["pitch_y"] = foot_y / self.video_height * self.PITCH_H
        return df

    def _calculate_motion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["track_id", "frame"]).copy()
        dt = 1.0 / self.fps
        results = []
        for tid, grp in df.groupby("track_id"):
            grp = grp.sort_values("frame").copy()
            raw_vx = grp["pitch_x"].diff() / dt
            raw_vy = grp["pitch_y"].diff() / dt
            raw_sp = np.sqrt(raw_vx**2 + raw_vy**2)
            vx = raw_vx.ewm(alpha=self.ema_alpha, adjust=False).mean()
            vy = raw_vy.ewm(alpha=self.ema_alpha, adjust=False).mean()
            sp = raw_sp.ewm(alpha=self.ema_alpha, adjust=False).mean()
            ac = sp.diff().ewm(alpha=self.ema_alpha, adjust=False).mean()
            di = np.degrees(np.arctan2(vy, vx))
            grp["velocity_x"]   = vx
            grp["velocity_y"]   = vy
            grp["speed"]        = sp
            grp["acceleration"] = ac
            grp["direction"]    = di
            results.append(grp)
        return pd.concat(results, ignore_index=True)

    def _infer_football_context(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ball_rows = df[df["is_ball"] == 1][["frame", "pitch_x", "pitch_y"]]
        if not ball_rows.empty:
            ball_pos = ball_rows.groupby("frame").first().rename(
                columns={"pitch_x": "_ball_x", "pitch_y": "_ball_y"}
            )
        else:
            ball_pos = pd.DataFrame(columns=["frame", "_ball_x", "_ball_y"]).set_index("frame")
        df = df.merge(ball_pos, on="frame", how="left")
        df["distance_to_ball"] = np.sqrt(
            (df["pitch_x"] - df["_ball_x"]) ** 2 +
            (df["pitch_y"] - df["_ball_y"]) ** 2
        ).fillna(-1.0)

        df["possession"] = -1
        player_only = df[(df["is_ball"] == 0) & (df["team_id"] != 2)].copy()
        if not player_only.empty:
            valid = player_only[player_only["distance_to_ball"] >= 0].copy()
            if not valid.empty:
                closest_idx = valid.groupby("frame")["distance_to_ball"].idxmin()
                closest = valid.loc[closest_idx]
                closest = closest[closest["distance_to_ball"] < self.POSSESSION_RADIUS]
                poss_map = dict(zip(closest["frame"], closest["track_id"].astype(int)))
                df.loc[df["frame"].isin(poss_map), "possession"] = (
                    df.loc[df["frame"].isin(poss_map), "frame"].map(poss_map)
                )

        avg_x_map = df.groupby("track_id")["pitch_x"].mean()
        df["_avg_x"]      = df["track_id"].map(avg_x_map)
        df["is_goalkeeper"] = 0
        df["role"]          = "unknown"
        df.loc[df["team_id"] == 2, "role"] = "referee"

        gk_mask_0 = (df["team_id"] == 0) & (df["_avg_x"] < 15)
        gk_mask_1 = (df["team_id"] == 1) & (df["_avg_x"] > 90)
        df.loc[gk_mask_0 | gk_mask_1, "is_goalkeeper"] = 1
        df.loc[gk_mask_0 | gk_mask_1, "role"]          = "goalkeeper"

        outfield_0 = df[(df["team_id"] == 0) & (df["is_goalkeeper"] == 0) & (df["role"] == "unknown")]
        df.loc[outfield_0.index[outfield_0["_avg_x"] < 35], "role"] = "defender"
        df.loc[outfield_0.index[(outfield_0["_avg_x"] >= 35) & (outfield_0["_avg_x"] < 65)], "role"] = "midfielder"
        df.loc[outfield_0.index[outfield_0["_avg_x"] >= 65], "role"] = "forward"

        outfield_1 = df[(df["team_id"] == 1) & (df["is_goalkeeper"] == 0) & (df["role"] == "unknown")]
        df.loc[outfield_1.index[outfield_1["_avg_x"] > 70], "role"] = "defender"
        df.loc[outfield_1.index[(outfield_1["_avg_x"] <= 70) & (outfield_1["_avg_x"] > 40)], "role"] = "midfielder"
        df.loc[outfield_1.index[outfield_1["_avg_x"] <= 40], "role"] = "forward"

        df["team_side"] = df["pitch_x"].apply(lambda x: "left" if x < 52.5 else "right")
        df["in_possession_zone"] = df["pitch_x"].apply(
            lambda x: "defensive third" if x < 35 else ("middle third" if x < 70 else "attacking third")
        )
        df = df.drop(columns=["_ball_x", "_ball_y", "_avg_x"], errors="ignore")
        return df
