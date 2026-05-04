"""Output schema definitions - Single source of truth for pipeline outputs.

This module defines the structure, naming conventions, and validation
for all pipeline output files including CSVs, JSON, and media files.
"""

from dataclasses import dataclass, field, asdict
from typing import TypedDict, NotRequired, Literal, List, Dict, Any, Optional
from enum import StrEnum
from pathlib import Path
import json
import csv
import pandas as pd


# =============================================================================
# Output File Names (Constants)
# =============================================================================

class OutputFiles:
    """Standardized output file names."""
    
    # CSV outputs
    TRACKING = "tracking_output.csv"
    ANALYTICS = "analytics.csv"
    PLAYER_SUMMARY = "player_summary.csv"
    POSSESSION_SUMMARY = "possession_summary.csv"
    
    # JSON outputs
    ANALYTICS_JSON = "analytics.json"
    METADATA = "metadata.json"
    
    # Media outputs
    ANNOTATED_VIDEO = "annotated_football_analysis.mp4"
    
    # Image outputs
    HEATMAP_TEAM_0 = "team_0_heatmap.png"
    HEATMAP_TEAM_1 = "team_1_heatmap.png"
    
    # Directory patterns
    HEATMAP_DIR = "heatmaps"


# =============================================================================
# CSV Column Schemas
# =============================================================================

class TrackingCSVColumns:
    """Column names for tracking_output.csv - aligned with results_page expectations."""
    
    FRAME = "frame"
    TRACK_ID = "track_id"  # Maps to tracker_id
    TEAM_ID = "team_id"
    PLAYER_ID = "player_id"
    
    # Bounding box (pixels)
    BB_LEFT = "bb_left"
    BB_TOP = "bb_top"
    BB_WIDTH = "bb_width"
    BB_HEIGHT = "bb_height"
    
    # Center point (pixels)
    CENTER_X = "center_x"  # Results page expects 'center_x'
    CENTER_Y = "center_y"  # Results page expects 'center_y'
    
    # Motion features
    VELOCITY_X = "velocity_x"
    VELOCITY_Y = "velocity_y"
    SPEED = "speed"  # km/h
    ACCELERATION = "acceleration"
    DIRECTION = "direction"
    
    # Pitch coordinates (meters)
    PITCH_X = "pitch_x"
    PITCH_Y = "pitch_y"
    
    # Classification
    IS_BALL = "is_ball"
    POSSESSION = "possession"
    ROLE = "role"
    IS_GOALKEEPER = "is_goalkeeper"
    TEAM_SIDE = "team_side"
    
    # Context
    DISTANCE_TO_BALL = "distance_to_ball"
    IN_POSSESSION_ZONE = "in_possession_zone"
    
    # Quality
    CONFIDENCE = "confidence"
    EXTRA = "extra"
    
    @classmethod
    def all_columns(cls) -> List[str]:
        """Return all column names in order."""
        return [
            cls.FRAME, cls.TRACK_ID, cls.TEAM_ID, cls.PLAYER_ID,
            cls.BB_LEFT, cls.BB_TOP, cls.BB_WIDTH, cls.BB_HEIGHT,
            cls.CENTER_X, cls.CENTER_Y,
            cls.VELOCITY_X, cls.VELOCITY_Y, cls.SPEED, cls.ACCELERATION, cls.DIRECTION,
            cls.PITCH_X, cls.PITCH_Y,
            cls.IS_BALL, cls.POSSESSION, cls.ROLE, cls.IS_GOALKEEPER, cls.TEAM_SIDE,
            cls.DISTANCE_TO_BALL, cls.IN_POSSESSION_ZONE,
            cls.CONFIDENCE, cls.EXTRA,
        ]


class AnalyticsCSVColumns:
    """Column names for analytics.csv - frame-by-frame object data."""
    
    FRAME = "frame"
    OBJECT_ID = "object_id"
    CLASS = "class"
    TEAM = "team"
    X_M = "x_m"
    Y_M = "y_m"
    SPEED_KMH = "speed_kmh"
    DISTANCE_M = "distance_m"
    
    @classmethod
    def all_columns(cls) -> List[str]:
        return [cls.FRAME, cls.OBJECT_ID, cls.CLASS, cls.TEAM, 
                cls.X_M, cls.Y_M, cls.SPEED_KMH, cls.DISTANCE_M]


class PlayerSummaryCSVColumns:
    """Column names for player_summary.csv - aggregated per-player stats."""
    
    OBJECT_ID = "object_id"
    TEAM_ID = "team_id"
    CLASS_ID = "class_id"
    TOTAL_FRAMES = "total_frames"
    TOP_SPEED_KM_H = "top_speed_km_h"
    AVG_SPEED_KM_H = "avg_speed_km_h"
    TOTAL_DISTANCE_M = "total_distance_m"
    POSS_PCT = "poss_pct"
    ROLE = "role"
    TEAM = "team"
    
    @classmethod
    def all_columns(cls) -> List[str]:
        return [
            cls.OBJECT_ID, cls.TEAM_ID, cls.CLASS_ID, cls.TOTAL_FRAMES,
            cls.TOP_SPEED_KM_H, cls.AVG_SPEED_KM_H, cls.TOTAL_DISTANCE_M,
            cls.POSS_PCT, cls.ROLE, cls.TEAM,
        ]


class PossessionSummaryCSVColumns:
    """Column names for possession_summary.csv - team-level possession."""
    
    TEAM_ID = "team_id"
    POSSESSION_PCT = "possession_pct"
    TOTAL_FRAMES = "total_frames"
    
    @classmethod
    def all_columns(cls) -> List[str]:
        return [cls.TEAM_ID, cls.POSSESSION_PCT, cls.TOTAL_FRAMES]


# =============================================================================
# JSON Structure Definitions (TypedDict)
# =============================================================================

class PassRecord(TypedDict):
    """Single pass event record."""
    passer_id: int
    receiver_id: int
    start_frame: int
    end_frame: int


class FrameObject(TypedDict):
    """Object within a frame."""
    id: str | int
    team: str
    x_m: float
    y_m: float
    speed: float
    distance: NotRequired[float]

# Add reserved keyword field after class definition
FrameObject.__annotations__["class"] = str


class FrameData(TypedDict):
    """Single frame entry."""
    frame: int
    objects: List[FrameObject]


class AnalyticsMetadata(TypedDict):
    """Metadata section for analytics.json."""
    video: str
    total_frames: int
    fps: float
    resolution: str
    replays_detected: int
    replay_frames_skipped: int
    total_passes: int
    processing_timestamp: NotRequired[str]


class AnalyticsJSON(TypedDict):
    """Complete analytics.json structure."""
    metadata: AnalyticsMetadata
    passes: List[PassRecord]
    frames: List[FrameData]


class MetadataJSON(TypedDict):
    """Processing metadata for the run."""
    video_name: str
    video_path: str
    output_dir: str
    processed_at: str
    config: Dict[str, Any]


# =============================================================================
# Dataclass Row Definitions (for type-safe construction)
# =============================================================================

@dataclass
class TrackingRow:
    """Type-safe row for tracking_output.csv."""
    frame_id: int
    object_id: int
    team_id: int = -1
    player_id: int = -1
    bb_left: float = 0.0
    bb_top: float = 0.0
    bb_width: float = 0.0
    bb_height: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    direction: float = 0.0
    pitch_x: float = 0.0
    pitch_y: float = 0.0
    is_ball: int = 0
    possession: int = 0
    role: str = ""
    is_goalkeeper: int = 0
    team_side: str = ""
    distance_to_ball: float = 0.0
    in_possession_zone: int = 0
    confidence: float = 0.0
    extra: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalyticsRow:
    """Type-safe row for analytics.csv."""
    frame: int
    object_id: str
    class_: str  # 'class' is reserved keyword
    team: str
    x_m: float
    y_m: float
    speed_kmh: float
    distance_m: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "object_id": self.object_id,
            "class": self.class_,
            "team": self.team,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "speed_kmh": self.speed_kmh,
            "distance_m": self.distance_m,
        }


@dataclass
class PlayerSummaryRow:
    """Type-safe row for player_summary.csv."""
    object_id: int
    team_id: int
    class_id: int
    total_frames: int
    top_speed_km_h: float
    avg_speed_km_h: float
    total_distance_m: float
    poss_pct: float
    role: str
    team: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PossessionSummaryRow:
    """Type-safe row for possession_summary.csv."""
    team_id: int
    possession_pct: float
    total_frames: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Validation Functions
# =============================================================================

class SchemaValidator:
    """Validation utilities for output files."""
    
    @staticmethod
    def validate_tracking_csv(df: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate tracking_output.csv schema.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        required = TrackingCSVColumns.all_columns()
        
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        
        # Type checks for key columns
        if TrackingCSVColumns.FRAME in df.columns:
            if not pd.api.types.is_integer_dtype(df[TrackingCSVColumns.FRAME]):
                errors.append(f"'{TrackingCSVColumns.FRAME}' must be integer")
        
        if TrackingCSVColumns.TRACK_ID in df.columns:
            if not pd.api.types.is_numeric_dtype(df[TrackingCSVColumns.TRACK_ID]):
                errors.append(f"'{TrackingCSVColumns.TRACK_ID}' must be numeric")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_analytics_csv(df: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate analytics.csv schema."""
        errors = []
        required = AnalyticsCSVColumns.all_columns()
        
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_analytics_json(data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate analytics.json structure."""
        errors = []
        
        if "metadata" not in data:
            errors.append("Missing 'metadata' section")
        else:
            meta = data["metadata"]
            required_meta = ["video", "total_frames", "fps", "resolution"]
            missing_meta = set(required_meta) - set(meta.keys())
            if missing_meta:
                errors.append(f"Missing metadata fields: {sorted(missing_meta)}")
        
        if "passes" not in data:
            errors.append("Missing 'passes' array")
        elif not isinstance(data["passes"], list):
            errors.append("'passes' must be a list")
        
        if "frames" not in data:
            errors.append("Missing 'frames' array")
        elif not isinstance(data["frames"], list):
            errors.append("'frames' must be a list")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_player_summary_csv(df: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate player_summary.csv schema."""
        errors = []
        required = PlayerSummaryCSVColumns.all_columns()
        
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_possession_summary_csv(df: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate possession_summary.csv schema."""
        errors = []
        required = PossessionSummaryCSVColumns.all_columns()
        
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        
        # Check team_id values are valid
        if PossessionSummaryCSVColumns.TEAM_ID in df.columns:
            invalid_teams = df[~df[PossessionSummaryCSVColumns.TEAM_ID].isin([0, 1, -1])]
            if len(invalid_teams) > 0:
                errors.append(f"Invalid team_id values found: {invalid_teams[PossessionSummaryCSVColumns.TEAM_ID].unique()}")
        
        return len(errors) == 0, errors


# =============================================================================
# Path Resolution Utilities
# =============================================================================

class OutputPathResolver:
    """Resolve output file paths consistently with game-specific folder support."""
    
    def __init__(self, output_dir: Path | str, game_id: str | None = None):
        self.output_dir = Path(output_dir)
        self.game_id = game_id
        
        # If game_id is provided, create game-specific subdirectory
        if self.game_id:
            self.game_dir = self.output_dir / self.game_id
            self.game_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.game_dir = self.output_dir
    
    def tracking_csv(self) -> Path:
        return self.game_dir / OutputFiles.TRACKING
    
    def analytics_csv(self) -> Path:
        return self.game_dir / OutputFiles.ANALYTICS
    
    def player_summary_csv(self) -> Path:
        return self.game_dir / OutputFiles.PLAYER_SUMMARY
    
    def possession_summary_csv(self) -> Path:
        return self.game_dir / OutputFiles.POSSESSION_SUMMARY
    
    def analytics_json(self) -> Path:
        return self.game_dir / OutputFiles.ANALYTICS_JSON
    
    def metadata_json(self) -> Path:
        return self.game_dir / OutputFiles.METADATA
    
    def annotated_video(self) -> Path:
        return self.game_dir / OutputFiles.ANNOTATED_VIDEO
    
    def heatmap_path(self, team_id: int) -> Path:
        """Get heatmap path for a team."""
        filename = f"team_{team_id}_heatmap.png"
        heatmap_dir = self.game_dir / OutputFiles.HEATMAP_DIR
        heatmap_dir.mkdir(exist_ok=True)
        return heatmap_dir / filename
    
    def all_expected_files(self) -> List[Path]:
        """Return list of all expected output files (excluding heatmaps)."""
        return [
            self.tracking_csv(),
            self.analytics_csv(),
            self.player_summary_csv(),
            self.possession_summary_csv(),
            self.analytics_json(),
            self.annotated_video(),
        ]
    
    @staticmethod
    def generate_game_id(video_name: str) -> str:
        """Generate a game ID from video filename."""
        import re
        # Remove extension and clean up name
        base_name = Path(video_name).stem
        # Remove special characters and replace with underscores
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
        # Add timestamp for uniqueness
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{clean_name}_{timestamp}"
    
    def get_game_summary_path(self) -> Path:
        """Get path to game summary file (metadata about the game)."""
        return self.game_dir / "game_summary.json"


# =============================================================================
# Helper Functions for Writers
# =============================================================================

def write_csv_headers(filepath: Path, columns: List[str]) -> None:
    """Initialize CSV with headers."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)


def append_csv_row(filepath: Path, row: Dict[str, Any], columns: List[str]) -> None:
    """Append a single row to CSV maintaining column order."""
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in columns])


def write_json_atomic(filepath: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """Write JSON atomically (temp file + rename) to prevent corruption."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path = filepath.with_suffix('.tmp')
    
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    
    temp_path.replace(filepath)


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    # Example: Create tracking row
    row = TrackingRow(
        frame_id=1,
        object_id=42,
        team_id=0,
        cx=100.5,
        cy=200.3,
        speed=12.5,
    )
    print("Tracking row:", row.to_dict())
    
    # Example: Validate a DataFrame
    import pandas as pd
    
    df = pd.DataFrame({
        TrackingCSVColumns.FRAME: [1, 2],
        TrackingCSVColumns.TRACK_ID: [1, 2],
        TrackingCSVColumns.TEAM_ID: [0, 1],
        TrackingCSVColumns.CX: [100.0, 200.0],
        TrackingCSVColumns.CY: [50.0, 75.0],
    })
    
    is_valid, errors = SchemaValidator.validate_tracking_csv(df)
    print(f"Valid: {is_valid}, Errors: {errors}")
    
    # Example: Path resolution
    resolver = OutputPathResolver("/tmp/test_output")
    print("Tracking CSV:", resolver.tracking_csv())
