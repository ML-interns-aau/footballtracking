import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from src.pipeline.output_schema import (
    TrackingCSVColumns,
    write_csv_headers,
)


def _iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    inter_x1 = max(xa1, xb1); inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2); inter_y2 = min(ya2, yb2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = (xa2 - xa1) * (ya2 - ya1)
    areaB = (xb2 - xb1) * (yb2 - yb1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class _IoUTracker:
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


class TrackingCSVBuilder:
    BALL_CLASS_ID    = 32
    PITCH_W          = 105.0
    PITCH_H          = 68.0
    POSSESSION_RADIUS = 2.0

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

    def finalize_and_write(self, output_path) -> pd.DataFrame:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._records:
            print("[TrackingCSVBuilder] No records to export")
            return pd.DataFrame(columns=TrackingCSVColumns.all_columns())

        df = pd.DataFrame(self._records)
        df = df.sort_values([TrackingCSVColumns.TRACK_ID, TrackingCSVColumns.FRAME]).reset_index(drop=True)
        df = self._interpolate(df)
        df = self._apply_pitch_mapping(df)
        df = self._calculate_motion_features(df)
        df = self._infer_football_context(df)
        df = df.sort_values([TrackingCSVColumns.FRAME, TrackingCSVColumns.TRACK_ID]).reset_index(drop=True)

        # Ensure all required columns exist
        required_columns = TrackingCSVColumns.all_columns()
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0 if col in [TrackingCSVColumns.SPEED, TrackingCSVColumns.VELOCITY_X, 
                                         TrackingCSVColumns.VELOCITY_Y, TrackingCSVColumns.ACCELERATION,
                                         TrackingCSVColumns.DISTANCE_TO_BALL] else ""

        final = df[required_columns].copy()
        num_cols = final.select_dtypes(include=[np.number]).columns
        final[num_cols] = final[num_cols].round(4)
        
        # Write CSV with headers
        write_csv_headers(output_path, required_columns)
        final.to_csv(output_path, index=False, mode='a', header=False)
        
        print(f"[TrackingCSVBuilder] Exported {len(final)} records to {output_path}")
        return final

    @staticmethod
    def _make_record(frame_idx, track_id, team_id, player_id,
                     bb_left, bb_top, bb_width, bb_height,
                     is_ball, confidence, extra) -> Dict[str, Any]:
        return {
            TrackingCSVColumns.FRAME: frame_idx,
            TrackingCSVColumns.TRACK_ID: track_id,
            TrackingCSVColumns.TEAM_ID: team_id,
            TrackingCSVColumns.PLAYER_ID: player_id,
            TrackingCSVColumns.BB_LEFT: bb_left,
            TrackingCSVColumns.BB_TOP: bb_top,
            TrackingCSVColumns.BB_WIDTH: bb_width,
            TrackingCSVColumns.BB_HEIGHT: bb_height,
            TrackingCSVColumns.CENTER_X: bb_left + bb_width / 2.0,
            TrackingCSVColumns.CENTER_Y: bb_top + bb_height / 2.0,
            TrackingCSVColumns.IS_BALL: is_ball,
            TrackingCSVColumns.CONFIDENCE: confidence,
            TrackingCSVColumns.EXTRA: extra,
        }

    def _interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        spatial_cols = [
            TrackingCSVColumns.BB_LEFT, TrackingCSVColumns.BB_TOP, 
            TrackingCSVColumns.BB_WIDTH, TrackingCSVColumns.BB_HEIGHT,
            TrackingCSVColumns.CENTER_X, TrackingCSVColumns.CENTER_Y,
            TrackingCSVColumns.CONFIDENCE
        ]
        chunks: List[pd.DataFrame] = []
        for tid, grp in df.groupby(TrackingCSVColumns.TRACK_ID):
            grp = grp.sort_values(TrackingCSVColumns.FRAME)
            min_f, max_f = grp[TrackingCSVColumns.FRAME].min(), grp[TrackingCSVColumns.FRAME].max()
            full_idx = list(range(min_f, max_f + 1))
            grp = grp.set_index(TrackingCSVColumns.FRAME).reindex(full_idx)
            for col in [TrackingCSVColumns.TRACK_ID, TrackingCSVColumns.TEAM_ID, 
                       TrackingCSVColumns.PLAYER_ID, TrackingCSVColumns.IS_BALL, 
                       TrackingCSVColumns.EXTRA]:
                grp[col] = grp[col].ffill().bfill()
            for col in spatial_cols:
                grp[col] = grp[col].interpolate(
                    method="linear", limit=self.interpolate_max_gap, limit_direction="both"
                )
            grp = grp.reset_index().rename(columns={"index": TrackingCSVColumns.FRAME})
            chunks.append(grp)
        return pd.concat(chunks, ignore_index=True) if chunks else df

    def _apply_pitch_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        foot_x = df[TrackingCSVColumns.CENTER_X].copy()
        foot_y = (df[TrackingCSVColumns.BB_TOP] + df[TrackingCSVColumns.BB_HEIGHT]).copy()
        ball_mask = df[TrackingCSVColumns.IS_BALL] == 1
        foot_y[ball_mask] = df.loc[ball_mask, TrackingCSVColumns.CENTER_Y]
        df = df.copy()
        if self.pitch_mapper is not None:
            coords = np.column_stack([foot_x.to_numpy(), foot_y.to_numpy()])
            mapped = self.pitch_mapper.transform_points(coords)
            df[TrackingCSVColumns.PITCH_X] = mapped[:, 0]
            df[TrackingCSVColumns.PITCH_Y] = mapped[:, 1]
        else:
            df[TrackingCSVColumns.PITCH_X] = foot_x / self.video_width  * self.PITCH_W
            df[TrackingCSVColumns.PITCH_Y] = foot_y / self.video_height * self.PITCH_H
        return df

    def _calculate_motion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values([TrackingCSVColumns.TRACK_ID, TrackingCSVColumns.FRAME]).copy()
        dt = 1.0 / self.fps
        results = []
        for tid, grp in df.groupby(TrackingCSVColumns.TRACK_ID):
            grp = grp.sort_values(TrackingCSVColumns.FRAME).copy()
            grp[TrackingCSVColumns.PITCH_X] = pd.to_numeric(grp[TrackingCSVColumns.PITCH_X], errors='coerce')
            grp[TrackingCSVColumns.PITCH_Y] = pd.to_numeric(grp[TrackingCSVColumns.PITCH_Y], errors='coerce')
            
            raw_vx = grp[TrackingCSVColumns.PITCH_X].diff() / dt
            raw_vy = grp[TrackingCSVColumns.PITCH_Y].diff() / dt
            raw_sp = (raw_vx**2 + raw_vy**2).pow(0.5)
            vx = raw_vx.ewm(alpha=self.ema_alpha, adjust=False).mean()
            vy = raw_vy.ewm(alpha=self.ema_alpha, adjust=False).mean()
            sp = raw_sp.ewm(alpha=self.ema_alpha, adjust=False).mean()
            ac = sp.diff().ewm(alpha=self.ema_alpha, adjust=False).mean()
            di = np.degrees(np.arctan2(vy, vx))
            grp[TrackingCSVColumns.VELOCITY_X] = vx
            grp[TrackingCSVColumns.VELOCITY_Y] = vy
            grp[TrackingCSVColumns.SPEED] = sp
            grp[TrackingCSVColumns.ACCELERATION] = ac
            grp[TrackingCSVColumns.DIRECTION] = di
            results.append(grp)
        return pd.concat(results, ignore_index=True)

    def _infer_football_context(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ball_rows = df[df[TrackingCSVColumns.IS_BALL] == 1][[
            TrackingCSVColumns.FRAME, TrackingCSVColumns.PITCH_X, TrackingCSVColumns.PITCH_Y
        ]]
        if not ball_rows.empty:
            ball_pos = ball_rows.groupby(TrackingCSVColumns.FRAME).first().rename(
                columns={TrackingCSVColumns.PITCH_X: "_ball_x", TrackingCSVColumns.PITCH_Y: "_ball_y"}
            )
        else:
            ball_pos = pd.DataFrame(columns=[TrackingCSVColumns.FRAME, "_ball_x", "_ball_y"]).set_index(TrackingCSVColumns.FRAME)
        df = df.merge(ball_pos, on=TrackingCSVColumns.FRAME, how="left")
        for col in [TrackingCSVColumns.PITCH_X, TrackingCSVColumns.PITCH_Y, "_ball_x", "_ball_y"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        dist_sq = (df[TrackingCSVColumns.PITCH_X] - df["_ball_x"]).pow(2) + (df[TrackingCSVColumns.PITCH_Y] - df["_ball_y"]).pow(2)
        df[TrackingCSVColumns.DISTANCE_TO_BALL] = dist_sq.pow(0.5).fillna(-1.0)

        df[TrackingCSVColumns.POSSESSION] = -1
        player_only = df[(df[TrackingCSVColumns.IS_BALL] == 0) & (df[TrackingCSVColumns.TEAM_ID] != 2)].copy()
        if not player_only.empty:
            valid = player_only[player_only[TrackingCSVColumns.DISTANCE_TO_BALL] >= 0].copy()
            if not valid.empty:
                closest_idx = valid.groupby(TrackingCSVColumns.FRAME)[TrackingCSVColumns.DISTANCE_TO_BALL].idxmin()
                closest = valid.loc[closest_idx]
                closest = closest[closest[TrackingCSVColumns.DISTANCE_TO_BALL] < self.POSSESSION_RADIUS]
                poss_map = dict(zip(closest[TrackingCSVColumns.FRAME], closest[TrackingCSVColumns.TRACK_ID].astype(int)))
                df.loc[df[TrackingCSVColumns.FRAME].isin(poss_map), TrackingCSVColumns.POSSESSION] = (
                    df.loc[df[TrackingCSVColumns.FRAME].isin(poss_map), TrackingCSVColumns.FRAME].map(poss_map)
                )

        avg_x_map = df.groupby(TrackingCSVColumns.TRACK_ID)[TrackingCSVColumns.PITCH_X].mean()
        df["_avg_x"]      = df[TrackingCSVColumns.TRACK_ID].map(avg_x_map)
        df[TrackingCSVColumns.IS_GOALKEEPER] = 0
        df[TrackingCSVColumns.ROLE] = "unknown"
        df.loc[df[TrackingCSVColumns.TEAM_ID] == 2, TrackingCSVColumns.ROLE] = "referee"

        gk_mask_0 = (df[TrackingCSVColumns.TEAM_ID] == 0) & (df["_avg_x"] < 15)
        gk_mask_1 = (df[TrackingCSVColumns.TEAM_ID] == 1) & (df["_avg_x"] > 90)
        df.loc[gk_mask_0 | gk_mask_1, TrackingCSVColumns.IS_GOALKEEPER] = 1
        df.loc[gk_mask_0 | gk_mask_1, TrackingCSVColumns.ROLE] = "goalkeeper"

        outfield_0 = df[(df[TrackingCSVColumns.TEAM_ID] == 0) & (df[TrackingCSVColumns.IS_GOALKEEPER] == 0) & (df[TrackingCSVColumns.ROLE] == "unknown")]
        df.loc[outfield_0.index[outfield_0["_avg_x"] < 35], TrackingCSVColumns.ROLE] = "defender"
        df.loc[outfield_0.index[(outfield_0["_avg_x"] >= 35) & (outfield_0["_avg_x"] < 65)], TrackingCSVColumns.ROLE] = "midfielder"
        df.loc[outfield_0.index[outfield_0["_avg_x"] >= 65], TrackingCSVColumns.ROLE] = "forward"

        outfield_1 = df[(df[TrackingCSVColumns.TEAM_ID] == 1) & (df[TrackingCSVColumns.IS_GOALKEEPER] == 0) & (df[TrackingCSVColumns.ROLE] == "unknown")]
        df.loc[outfield_1.index[outfield_1["_avg_x"] > 70], TrackingCSVColumns.ROLE] = "defender"
        df.loc[outfield_1.index[(outfield_1["_avg_x"] <= 70) & (outfield_1["_avg_x"] > 40)], TrackingCSVColumns.ROLE] = "midfielder"
        df.loc[outfield_1.index[outfield_1["_avg_x"] <= 40], TrackingCSVColumns.ROLE] = "forward"

        df[TrackingCSVColumns.TEAM_SIDE] = df[TrackingCSVColumns.PITCH_X].apply(lambda x: "left" if x < 52.5 else "right")
        df[TrackingCSVColumns.IN_POSSESSION_ZONE] = df[TrackingCSVColumns.PITCH_X].apply(
            lambda x: "defensive third" if x < 35 else ("middle third" if x < 70 else "attacking third")
        )
        df = df.drop(columns=["_ball_x", "_ball_y", "_avg_x"], errors="ignore")
        return df
