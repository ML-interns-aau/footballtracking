# app/config.py
"""Shared paths, constants, and runtime configuration."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data directories ─────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
INSIGHTS_DIR    = os.path.join(DATA_DIR, "insights")

# ── Model ────────────────────────────────────────────────────────────────────
_default_model = os.path.join(PROJECT_ROOT, "models", "best.pt")
# If a compatible weight exists at the project root (used by main.py), prefer it.
_root_model = os.path.join(PROJECT_ROOT, "yolo8m_fixed.pt")
MODEL_PATH = _root_model if os.path.exists(_root_model) else _default_model

# ── Video formats ────────────────────────────────────────────────────────────
VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv"]

# ── Detection classes ────────────────────────────────────────────────────────
PLAYER_CLASS_IDS = {1, 2, 3}
BALL_CLASS_ID    = 0
CLASS_LABELS     = {0: "Ball", 1: "GK", 2: "Player", 3: "Referee"}
TEAM_LABELS      = {0: "Team A", 1: "Team B", -1: "Unassigned"}

# ── Pipeline defaults ────────────────────────────────────────────────────────
DEFAULT_CONF       = 0.35
DEFAULT_IOU        = 0.45
DEFAULT_IMGSZ      = 1280
DEFAULT_TARGET_FPS = 15
DEFAULT_RESIZE_W   = 1280

# ── Ensure directories exist ────────────────────────────────────────────────
for _d in [RAW_DIR, PROCESSED_DIR, ANNOTATIONS_DIR, INSIGHTS_DIR]:
    os.makedirs(_d, exist_ok=True)
