"""Football Analytics Pipeline — main entry point.

Runs the full tracking pipeline on a pre-processed video and writes all
outputs (annotated video, CSVs, heatmaps, analytics JSON) to a single
game-specific output directory.

Usage:
    python main.py --input data/processed/match.mp4 --output_dir results/
    python main.py --input ... --max_frames 500 --target_fps 10 --conf 0.3
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import argparse

# ── Project — config ──────────────────────────────────────────────────────────
from app.config import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_IMGSZ,
    DEFAULT_RESIZE_W,
    DEFAULT_TARGET_FPS,
    MODEL_PATH,
)

# ── Project — pipeline runner ─────────────────────────────────────────────────
from src.pipeline_runner import FootballPipelineRunner


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    runner = FootballPipelineRunner(args)
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Analytics Pipeline")
    parser.add_argument("--input",        type=str,   required=True)
    parser.add_argument("--output_dir",   type=str,   default="results")
    parser.add_argument("--max_frames",   type=int,   default=0)
    parser.add_argument("--target_fps",   type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--resize_width", type=int,   default=DEFAULT_RESIZE_W)
    parser.add_argument("--conf",         type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou",          type=float, default=DEFAULT_IOU)
    parser.add_argument("--imgsz",        type=int,   default=DEFAULT_IMGSZ)
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--model_path",   type=str,   default=MODEL_PATH)
    parser.add_argument("--game_id",      type=str,   default=None,        help="Game ID for folder organisation")
    parser.add_argument("--match-id",     type=str,   default="match_001", help="Identifier for the match")
    parser.add_argument("--home-team",    type=str,   default="Home Team", help="Home team name")
    parser.add_argument("--away-team",    type=str,   default="Away Team", help="Away team name")
    _args = parser.parse_args()
    main(_args)
