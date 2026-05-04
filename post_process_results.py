import pandas as pd
import json
import os
from pathlib import Path

def post_process(results_dir, insights_dir, video_name="video.mp4"):
    results_dir = Path(results_dir)
    insights_dir = Path(insights_dir)
    insights_dir.mkdir(parents=True, exist_ok=True)
    
    analytics_path = results_dir / "analytics.csv"
    tracking_path = results_dir / "tracking_output.csv"
    
    if not analytics_path.exists():
        alt_path = Path("dashboard") / "results" / "analytics.csv"
        if alt_path.exists():
            analytics_path = alt_path
            tracking_path = Path("dashboard") / "results" / "tracking_output.csv"
        else:
            return

    df = pd.read_csv(analytics_path)
    
    for col in ['frame', 'object_id', 'team', 'speed_kmh']:
        if col not in df.columns:
            return

    def map_team(team_str):
        t_str = str(team_str).lower()
        if "team 0" in t_str: return 0
        if "team 1" in t_str: return 1
        if "referee" in t_str: return -2
        return -1

    df['team_id'] = df['team'].apply(map_team)
    total_frames = int(df['frame'].nunique()) if not df.empty else 0
    
    player_stats = []
    if not df.empty:
        for (obj_id, team_id), group in df.groupby(['object_id', 'team_id'], dropna=True):
            if pd.isna(obj_id) or str(obj_id) == "" or team_id == -2: 
                continue
            
            speeds = group['speed_kmh'].dropna()
            player_stats.append({
                "object_id": int(float(obj_id)),
                "team_id": int(team_id),
                "top_speed_km_h": float(speeds.max()) if not speeds.empty else 0.0,
                "avg_speed_km_h": float(speeds.mean()) if not speeds.empty else 0.0,
                "total_frames": int(len(group)),
                "poss_pct": 0.0 
            })
    
    player_df = pd.DataFrame(player_stats)
    
    poss_df = pd.DataFrame([
        {"team_id": 0, "possession_pct": 50.0},
        {"team_id": 1, "possession_pct": 50.0}
    ])
    
    player_df.to_csv(insights_dir / "player_summary.csv", index=False)
    poss_df.to_csv(insights_dir / "possession_summary.csv", index=False)
    
    if tracking_path.exists():
        track_df = pd.read_csv(tracking_path)
        track_df = track_df.rename(columns={
            "frame": "frame_id",
            "track_id": "object_id",
            "center_x": "cx",
            "center_y": "cy"
        })
        track_df.to_csv(insights_dir / "tracking_enriched.csv", index=False)
    
    summary = {
        "video": video_name,
        "total_frames": total_frames,
        "resolution": "1920x1080",
        "replays_detected": 0
    }
    with open(insights_dir / "pipeline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    post_process("results", "data/insights")
