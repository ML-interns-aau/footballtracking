"""
Post-processing: reads pipeline outputs and generates insight CSVs + summary JSON.
"""

import pandas as pd
import json
import os
from pathlib import Path


def post_process(results_dir, insights_dir, video_name="video.mp4"):
    results_dir  = Path(results_dir)
    insights_dir = Path(insights_dir)
    insights_dir.mkdir(parents=True, exist_ok=True)

    analytics_path = results_dir / "analytics.csv"
    tracking_path  = results_dir / "tracking_output.csv"

    if not analytics_path.exists():
        raise FileNotFoundError(
            f"analytics.csv not found in {results_dir}. "
            "The pipeline may not have completed successfully."
        )

    df = pd.read_csv(analytics_path)

    required_cols = {"frame", "object_id", "team", "speed_kmh"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"analytics.csv is missing columns: {missing}")

    # ── Team mapping ──────────────────────────────────────────────────
    def map_team(team_str):
        t = str(team_str).lower()
        if "team 0" in t:   return 0
        if "team 1" in t:   return 1
        if "referee" in t:  return -2
        return -1

    df["team_id"]  = df["team"].apply(map_team)
    total_frames   = int(df["frame"].nunique()) if not df.empty else 0

    # ── Player summary ────────────────────────────────────────────────
    player_stats = []
    if not df.empty:
        for (obj_id, team_id), group in df.groupby(["object_id", "team_id"], dropna=True):
            if pd.isna(obj_id) or str(obj_id).strip() == "" or team_id == -2:
                continue
            speeds = group["speed_kmh"].dropna()
            player_stats.append({
                "object_id":      int(float(obj_id)),
                "team_id":        int(team_id),
                "top_speed_km_h": float(speeds.max())  if not speeds.empty else 0.0,
                "avg_speed_km_h": float(speeds.mean()) if not speeds.empty else 0.0,
                "total_frames":   int(len(group)),
                "poss_pct":       0.0,
            })

    player_df = pd.DataFrame(player_stats)

    # ── Possession summary ────────────────────────────────────────────
    # Simple heuristic: split frames where a player is closest to the ball
    poss_counts = {0: 0, 1: 0}
    if not df.empty and "class" in df.columns:
        ball_frames  = df[df["class"] == "ball"][["frame", "x_m", "y_m"]].copy()
        player_frames = df[df["class"] == "player"][["frame", "object_id", "team_id", "x_m", "y_m"]].copy()

        if not ball_frames.empty and not player_frames.empty:
            merged = player_frames.merge(ball_frames, on="frame", suffixes=("_p", "_b"))
            merged["dist"] = (
                (merged["x_m_p"] - merged["x_m_b"]) ** 2
                + (merged["y_m_p"] - merged["y_m_b"]) ** 2
            ) ** 0.5
            closest = merged.loc[merged.groupby("frame")["dist"].idxmin()]
            for tid in [0, 1]:
                poss_counts[tid] = int((closest["team_id"] == tid).sum())

    total_poss = sum(poss_counts.values()) or 1
    poss_df = pd.DataFrame([
        {"team_id": 0, "possession_pct": round(poss_counts[0] / total_poss * 100, 1)},
        {"team_id": 1, "possession_pct": round(poss_counts[1] / total_poss * 100, 1)},
    ])

    # Update poss_pct in player_df
    if not player_df.empty and not poss_df.empty:
        for _, row in poss_df.iterrows():
            mask = player_df["team_id"] == row["team_id"]
            n    = mask.sum()
            if n > 0:
                player_df.loc[mask, "poss_pct"] = round(row["possession_pct"] / n, 2)

    # ── Save CSVs ─────────────────────────────────────────────────────
    player_df.to_csv(insights_dir / "player_summary.csv", index=False)
    poss_df.to_csv(insights_dir / "possession_summary.csv", index=False)

    if tracking_path.exists():
        track_df = pd.read_csv(tracking_path)
        rename_map = {}
        if "frame"    in track_df.columns: rename_map["frame"]    = "frame_id"
        if "track_id" in track_df.columns: rename_map["track_id"] = "object_id"
        if "center_x" in track_df.columns: rename_map["center_x"] = "cx"
        if "center_y" in track_df.columns: rename_map["center_y"] = "cy"
        track_df = track_df.rename(columns=rename_map)
        track_df.to_csv(insights_dir / "tracking_enriched.csv", index=False)

    # ── Resolution from annotated video ──────────────────────────────
    resolution = "unknown"
    annotated_video = results_dir / "annotated_football_analysis.mp4"
    if annotated_video.exists():
        try:
            import cv2
            cap = cv2.VideoCapture(str(annotated_video))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                resolution = f"{w}×{h}"
        except Exception:
            pass

    # ── Pipeline summary JSON ─────────────────────────────────────────
    summary = {
        "video":             video_name,
        "total_frames":      total_frames,
        "resolution":        resolution,
        "replays_detected":  0,
        "players_tracked":   len(player_df),
        "team_0_possession": poss_df.loc[poss_df["team_id"] == 0, "possession_pct"].values[0]
                             if not poss_df.empty else 50.0,
        "team_1_possession": poss_df.loc[poss_df["team_id"] == 1, "possession_pct"].values[0]
                             if not poss_df.empty else 50.0,
    }
    with open(insights_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[post_process] Done — {total_frames} frames, "
        f"{len(player_df)} players, saved to {insights_dir}"
    )


if __name__ == "__main__":
    post_process("results", "data/insights")
