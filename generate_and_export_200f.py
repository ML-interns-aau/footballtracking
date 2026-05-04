"""
Run this script once to generate:
  data/dummy_detections_200f.csv
  results/sample_tracking_output_200f.csv

Usage (from project root):
    python generate_and_export_200f.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Make sure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.tracking_csv_builder import TrackingCSVBuilder
from src.homography.pitch_mapping import PitchMapping

# ------------------------------------------------------------------
# 1. Generate 200-frame synthetic detections
# ------------------------------------------------------------------
def generate_detections(num_frames: int = 200) -> pd.DataFrame:
    objects = [
        [0,  1,  100, 200,  2.0,  0.5],   # Team 0 Player 1
        [0,  2,  300, 180,  1.5,  0.2],   # Team 0 Player 2
        [1,  3,  800, 190,  1.0,  0.3],   # Team 1 Player 1
        [1,  4,  950, 210,  0.8, -0.1],   # Team 1 Player 2
        [2,  5,  530, 300,  0.5,  0.1],   # Referee
        [32, 99, 520, 380,  5.0,  2.0],   # Ball
    ]
    rng = np.random.default_rng(42)
    records = []
    for frame in range(1, num_frames + 1):
        for obj in objects:
            team_id, player_id, x, y, vx, vy = obj
            nx = x + vx * (frame - 1) + rng.normal(0, 0.5)
            ny = y + vy * (frame - 1) + rng.normal(0, 0.5)
            records.append({
                "frame":      frame,
                "team_id":    int(team_id),
                "player_id":  int(player_id),
                "bb_left":    round(nx, 2),
                "bb_top":     round(ny, 2),
                "bb_width":   60  if team_id != 32 else 22,
                "bb_height":  120 if team_id != 32 else 22,
                "confidence": round(0.9 + rng.uniform(0, 0.09), 4),
                "extra":      -1,
            })
    return pd.DataFrame(records)


def main():
    # ---- Paths ----
    root       = Path(__file__).resolve().parent
    data_dir   = root / "data"
    results_dir= root / "results"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    det_path = data_dir / "dummy_detections_200f.csv"
    out_path = results_dir / "sample_tracking_output_200f.csv"

    # ---- Generate detections ----
    print("[1/3] Generating 200-frame synthetic detections ...")
    df = generate_detections(200)
    df.to_csv(det_path, index=False)
    print(f"      Saved -> {det_path}  ({len(df)} rows)")

    # ---- Pitch mapper (standard 1920x1080 homography) ----
    print("[2/3] Setting up pitch mapper ...")
    pitch_mapper = PitchMapping(
        src_points=[[0, 1080], [1920, 1080], [1440, 324], [480, 324]],
        dst_points=[[0, 68],   [105, 68],    [105, 0],    [0, 0]],
    )

    # ---- Build structured CSV ----
    print("[3/3] Building tracking CSV ...")
    builder = TrackingCSVBuilder(pitch_mapper=pitch_mapper, fps=25.0)
    builder.load_from_csv(str(det_path))
    builder.finalize_and_write(str(out_path))
    print(f"\nDone! Sample CSV -> {out_path}")


if __name__ == "__main__":
    main()
