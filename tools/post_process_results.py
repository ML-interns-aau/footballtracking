"""Post-process raw pipeline outputs into player/possession insights.

Reads the analytics CSV produced by the main pipeline and writes:
  - player_summary.csv
  - possession_summary.csv
  - tracking_enriched.csv
  - pipeline_summary.json

Usage:
    python tools/post_process_results.py
    python tools/post_process_results.py --results results/ --insights data/insights/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so src.* imports resolve from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.exporters.output_schema import (
    AnalyticsCSVColumns,
    OutputFiles,
    PlayerSummaryCSVColumns,  # noqa: F401  (re-exported for callers)
    PossessionSummaryCSVColumns,  # noqa: F401
    TrackingCSVColumns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_team(team_str: str) -> int:
    t = str(team_str).lower()
    if "team 0"  in t: return 0
    if "team 1"  in t: return 1
    if "referee" in t: return -2
    return -1


def _read_resolution(video_path: Path) -> str:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return f"{w}×{h}"
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def post_process(
    results_dir: str | Path,
    insights_dir: str | Path,
    video_name: str = "video.mp4",
) -> None:
    results_dir  = Path(results_dir)
    insights_dir = Path(insights_dir)
    insights_dir.mkdir(parents=True, exist_ok=True)

    analytics_path = results_dir / OutputFiles.ANALYTICS
    tracking_path  = results_dir / OutputFiles.TRACKING

    if not analytics_path.exists():
        raise FileNotFoundError(
            f"{OutputFiles.ANALYTICS} not found in {results_dir}. "
            "The pipeline may not have completed successfully."
        )

    df = pd.read_csv(analytics_path)

    required_cols = {
        AnalyticsCSVColumns.FRAME,
        AnalyticsCSVColumns.OBJECT_ID,
        AnalyticsCSVColumns.TEAM,
        AnalyticsCSVColumns.SPEED_KMH,
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{OutputFiles.ANALYTICS} is missing columns: {missing}")

    df["team_id"] = df[AnalyticsCSVColumns.TEAM].apply(_map_team)
    total_frames  = int(df[AnalyticsCSVColumns.FRAME].nunique()) if not df.empty else 0

    # ── Player stats ──────────────────────────────────────────────────────────
    player_stats: list[dict] = []
    if not df.empty:
        for (obj_id, team_id), group in df.groupby(
            [AnalyticsCSVColumns.OBJECT_ID, "team_id"], dropna=True
        ):
            if pd.isna(obj_id) or str(obj_id).strip() == "" or team_id == -2:
                continue
            speeds = group[AnalyticsCSVColumns.SPEED_KMH].dropna()
            player_stats.append({
                "object_id":      int(float(obj_id)),
                "team_id":        int(team_id),
                "top_speed_km_h": float(speeds.max())  if not speeds.empty else 0.0,
                "avg_speed_km_h": float(speeds.mean()) if not speeds.empty else 0.0,
                "total_frames":   int(len(group)),
                "poss_pct":       0.0,
            })
    player_df = pd.DataFrame(player_stats)

    # ── Possession counts ─────────────────────────────────────────────────────
    poss_counts: dict[int, int] = {0: 0, 1: 0}
    if not df.empty and "class" in df.columns:
        ball_frames   = df[df["class"] == "ball"][["frame", "x_m", "y_m"]].copy()
        player_frames = df[df["class"] == "player"][
            ["frame", "object_id", "team_id", "x_m", "y_m"]
        ].copy()

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

    if not player_df.empty and not poss_df.empty:
        for _, row in poss_df.iterrows():
            mask = player_df["team_id"] == row["team_id"]
            n    = mask.sum()
            if n > 0:
                player_df.loc[mask, "poss_pct"] = round(row["possession_pct"] / n, 2)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    player_df.to_csv(insights_dir / OutputFiles.PLAYER_SUMMARY,    index=False)
    poss_df.to_csv(  insights_dir / OutputFiles.POSSESSION_SUMMARY, index=False)

    if tracking_path.exists():
        track_df = pd.read_csv(tracking_path)
        rename_map: dict[str, str] = {}
        if TrackingCSVColumns.FRAME    in track_df.columns: rename_map[TrackingCSVColumns.FRAME]    = "frame_id"
        if TrackingCSVColumns.TRACK_ID in track_df.columns: rename_map[TrackingCSVColumns.TRACK_ID] = "object_id"
        if TrackingCSVColumns.CENTER_X in track_df.columns: rename_map[TrackingCSVColumns.CENTER_X] = "cx"
        if TrackingCSVColumns.CENTER_Y in track_df.columns: rename_map[TrackingCSVColumns.CENTER_Y] = "cy"
        track_df.rename(columns=rename_map).to_csv(insights_dir / "tracking_enriched.csv", index=False)

    # ── Pipeline summary JSON ─────────────────────────────────────────────────
    resolution = _read_resolution(results_dir / OutputFiles.ANNOTATED_VIDEO)
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
    with (insights_dir / "pipeline_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"[post_process] Done — {total_frames} frames, "
        f"{len(player_df)} players, saved to {insights_dir}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-process raw pipeline outputs into player/possession insights"
    )
    parser.add_argument("--results",  default="results",      help="Pipeline results directory")
    parser.add_argument("--insights", default="data/insights", help="Destination directory for insight files")
    parser.add_argument("--video",    default="video.mp4",    help="Video filename for the summary JSON")
    args = parser.parse_args(argv)

    try:
        post_process(
            results_dir=args.results,
            insights_dir=args.insights,
            video_name=args.video,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
