"""Generate 200-frame synthetic detections and export a tracking CSV.

Useful for smoke-testing the TrackingCSVBuilder without needing a real video.

Usage:
    python tools/generate_and_export_200f.py
    python tools/generate_and_export_200f.py --frames 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so src.* imports resolve from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.homography.pitch_mapping import PitchMapping
from src.exporters.tracking_csv_builder import TrackingCSVBuilder

# ---------------------------------------------------------------------------
# Synthetic objects: [team_id, player_id, x0, y0, vx, vy]
# ---------------------------------------------------------------------------

_OBJECTS: list[list[float]] = [
    [0,   1,  100, 200,  2.0,  0.5],
    [0,   2,  300, 180,  1.5,  0.2],
    [1,   3,  800, 190,  1.0,  0.3],
    [1,   4,  950, 210,  0.8, -0.1],
    [2,   5,  530, 300,  0.5,  0.1],
    [32, 99,  520, 380,  5.0,  2.0],
]

# Default homography — maps a 1920×1080 frame onto a 105×68 m pitch
_DEFAULT_SRC_PTS = [[0, 1080], [1920, 1080], [1440, 324], [480, 324]]
_DEFAULT_DST_PTS = [[0, 68],   [105, 68],    [105, 0],    [0, 0]]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def generate_detections(num_frames: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for frame in range(1, num_frames + 1):
        for obj in _OBJECTS:
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic detections and build a tracking CSV"
    )
    parser.add_argument("--frames", type=int, default=200,
                        help="Number of frames to generate (default: 200)")
    parser.add_argument("--fps",    type=float, default=25.0,
                        help="Frames per second used in the CSV builder (default: 25.0)")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    data_dir     = project_root / "data"
    results_dir  = project_root / "results"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    det_path = data_dir    / "dummy_detections_200f.csv"
    out_path = results_dir / "sample_tracking_output_200f.csv"

    print(f"[1/3] Generating {args.frames}-frame synthetic detections ...")
    df = generate_detections(args.frames)
    df.to_csv(det_path, index=False)
    print(f"      Saved -> {det_path}  ({len(df)} rows)")

    print("[2/3] Setting up pitch mapper ...")
    pitch_mapper = PitchMapping(
        src_points=_DEFAULT_SRC_PTS,
        dst_points=_DEFAULT_DST_PTS,
    )

    print("[3/3] Building tracking CSV ...")
    builder = TrackingCSVBuilder(pitch_mapper=pitch_mapper, fps=args.fps)
    builder.load_from_csv(str(det_path))
    builder.finalize_and_write(str(out_path))
    print(f"\nDone! Sample CSV -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
