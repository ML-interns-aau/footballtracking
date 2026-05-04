# app/config.py
"""Shared paths, constants, and runtime configuration."""

import os
from pathlib import Path

# Import centralized configuration (with fallback for imports during setup)
try:
    from src.config import CONFIG
    _config_loaded = True
except ImportError:
    _config_loaded = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data directories ─────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
INSIGHTS_DIR    = os.path.join(DATA_DIR, "insights")

# ── Model ────────────────────────────────────────────────────────────────────
_default_model = os.path.join(PROJECT_ROOT, "models", "yolov8m_fixed.pt")
# If a compatible weight exists at the project root (used by main.py), prefer it.
_root_model = os.path.join(PROJECT_ROOT, "yolov8m_fixed.pt")
MODEL_PATH = _root_model if os.path.exists(_root_model) else _default_model

# ── Video formats ────────────────────────────────────────────────────────────
VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv"]

# ── Detection classes ────────────────────────────────────────────────────────
PLAYER_CLASS_IDS = {1, 2, 3}
BALL_CLASS_ID    = 0
CLASS_LABELS     = {0: "Ball", 1: "GK", 2: "Player", 3: "Referee"}
TEAM_LABELS      = {0: "Team A", 1: "Team B", -1: "Unassigned"}

# ── Pipeline defaults ────────────────────────────────────────────────────────
# Use centralized configuration if available, fall back to hardcoded values
defaults = CONFIG.get_dict('detection') if _config_loaded else {}
video_defaults = CONFIG.get_dict('video') if _config_loaded else {}

DEFAULT_CONF       = defaults.get('confidence_threshold', 0.35)
DEFAULT_IOU        = defaults.get('iou_threshold', 0.45)
DEFAULT_IMGSZ      = defaults.get('image_size', 1280)
DEFAULT_TARGET_FPS = video_defaults.get('target_fps', 15)
DEFAULT_RESIZE_W   = video_defaults.get('resize_width', 1280)

# ── Ensure directories exist ────────────────────────────────────────────────
for _d in [RAW_DIR, PROCESSED_DIR, ANNOTATIONS_DIR, INSIGHTS_DIR]:
    os.makedirs(_d, exist_ok=True)


def get_game_list():
    """Get list of all game folders in insights directory."""
    import json
    
    games = []
    insights_path = Path(INSIGHTS_DIR)
    
    if not insights_path.exists():
        return games
    
    for game_folder in insights_path.iterdir():
        if game_folder.is_dir():
            # Try to load game summary
            summary_path = game_folder / "game_summary.json"
            game_info = {
                "game_id": game_folder.name,
                "path": str(game_folder),
                "created": None,
                "video_name": None,
                "status": "Unknown"
            }
            
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                        game_info.update({
                            "video_name": summary.get("video_name"),
                            "created": summary.get("created"),
                            "status": summary.get("status", "Completed")
                        })
                except:
                    pass
            else:
                # Check if analytics.json exists as fallback
                analytics_path = game_folder / "analytics.json"
                if analytics_path.exists():
                    game_info["status"] = "Completed"
                else:
                    game_info["status"] = "Incomplete"
            
            games.append(game_info)
    
    # Sort by creation time (newest first)
    games.sort(key=lambda x: x.get("created", ""), reverse=True)
    return games


def create_game_folder(video_name: str) -> str:
    """Create a new game folder and return the game ID."""
    from src.pipeline.output_schema import OutputPathResolver
    import json
    from datetime import datetime
    
    game_id = OutputPathResolver.generate_game_id(video_name)
    game_path = Path(INSIGHTS_DIR) / game_id
    game_path.mkdir(parents=True, exist_ok=True)
    
    # Create game summary
    summary = {
        "game_id": game_id,
        "video_name": video_name,
        "created": datetime.now().isoformat(),
        "status": "Processing",
        "files": []
    }
    
    with open(game_path / "game_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return game_id


def update_game_status(game_id: str, status: str, **kwargs):
    """Update game status and metadata."""
    import json
    from datetime import datetime
    
    game_path = Path(INSIGHTS_DIR) / game_id
    summary_path = game_path / "game_summary.json"
    
    if not summary_path.exists():
        return False
    
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        
        summary["status"] = status
        summary["updated"] = datetime.now().isoformat()
        summary.update(kwargs)
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        return True
    except:
        return False
