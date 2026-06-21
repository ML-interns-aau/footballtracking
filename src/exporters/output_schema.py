from dataclasses import dataclass, field, asdict
import sys
from typing import TypedDict, Literal, List, Dict, Any, Optional

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired
from pathlib import Path
import json
import csv
import pandas as pd
class OutputFiles:
    TRACKING = "tracking_output.csv"
    ANALYTICS = "analytics.csv"
    PLAYER_SUMMARY = "player_summary.csv"
    POSSESSION_SUMMARY = "possession_summary.csv"
    ANALYTICS_JSON = "analytics.json"
    METADATA = "metadata.json"
    ANNOTATED_VIDEO = "annotated_football_analysis.mp4"
    HEATMAP_TEAM_0 = "team_0_heatmap.png"
    HEATMAP_TEAM_1 = "team_1_heatmap.png"
    HEATMAP_DIR = "heatmaps"
class TrackingCSVColumns:
    FRAME = "frame"
    TRACK_ID = "track_id"
    TEAM_ID = "team_id"
    PLAYER_ID = "player_id"
    BB_LEFT = "bb_left"
    BB_TOP = "bb_top"
    BB_WIDTH = "bb_width"
    BB_HEIGHT = "bb_height"
    CENTER_X = "center_x"
    CENTER_Y = "center_y"
    VELOCITY_X = "velocity_x"
    VELOCITY_Y = "velocity_y"
    SPEED = "speed"
    ACCELERATION = "acceleration"
    DIRECTION = "direction"
    PITCH_X = "pitch_x"
    PITCH_Y = "pitch_y"
    IS_BALL = "is_ball"
    POSSESSION = "possession"
    ROLE = "role"
    IS_GOALKEEPER = "is_goalkeeper"
    TEAM_SIDE = "team_side"
    DISTANCE_TO_BALL = "distance_to_ball"
    IN_POSSESSION_ZONE = "in_possession_zone"
    CONFIDENCE = "confidence"
    EXTRA = "extra"
    @classmethod
    def all_columns(cls) -> List[str]:
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
    TEAM_ID = "team_id"
    POSSESSION_PCT = "possession_pct"
    TOTAL_FRAMES = "total_frames"
    @classmethod
    def all_columns(cls) -> List[str]:
        return [cls.TEAM_ID, cls.POSSESSION_PCT, cls.TOTAL_FRAMES]
class PassValidation(TypedDict):
    same_team: bool
    distance_valid: bool
    speed_valid: bool

class PassRecord(TypedDict):
    pass_id: int
    event_type: str
    passer_id: int | None
    receiver_id: int | None
    passer_team: str | None
    receiver_team: str | None
    successful: bool
    intercepted: bool
    start_frame: int
    end_frame: int
    duration_frames: int
    distance_m: float
    ball_speed_kmh: float
    pass_type: str
    start_xy: list[float]
    end_xy: list[float]
    timestamp: float
    validation: PassValidation
class FrameObject(TypedDict):
    id: str | int
    team: str
    x_m: float
    y_m: float
    speed: float
    distance: NotRequired[float]
FrameObject.__annotations__["class"] = str
class FrameData(TypedDict):
    frame: int
    objects: List[FrameObject]
class AnalyticsMetadata(TypedDict):
    video: str
    total_frames: int
    fps: float
    resolution: str
    replays_detected: int
    replay_frames_skipped: int
    total_passes: int
    processing_timestamp: NotRequired[str]
class AnalyticsJSON(TypedDict):
    metadata: AnalyticsMetadata
    passes: List[PassRecord]
    frames: List[FrameData]
class MetadataJSON(TypedDict):
    video_name: str
    video_path: str
    output_dir: str
    processed_at: str
    config: Dict[str, Any]
@dataclass
class TrackingRow:
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
    frame: int
    object_id: str
    class_: str
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
    team_id: int
    possession_pct: float
    total_frames: int
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
class SchemaValidator:
    @staticmethod
    def validate_tracking_csv(df: pd.DataFrame) -> tuple[bool, List[str]]:
        errors = []
        required = TrackingCSVColumns.all_columns()
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        if TrackingCSVColumns.FRAME in df.columns:
            if not pd.api.types.is_integer_dtype(df[TrackingCSVColumns.FRAME]):
                errors.append(f"'{TrackingCSVColumns.FRAME}' must be integer")
        if TrackingCSVColumns.TRACK_ID in df.columns:
            if not pd.api.types.is_numeric_dtype(df[TrackingCSVColumns.TRACK_ID]):
                errors.append(f"'{TrackingCSVColumns.TRACK_ID}' must be numeric")
        return len(errors) == 0, errors
    @staticmethod
    def validate_analytics_csv(df: pd.DataFrame) -> tuple[bool, List[str]]:
        errors = []
        required = AnalyticsCSVColumns.all_columns()
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        return len(errors) == 0, errors
    @staticmethod
    def validate_analytics_json(data: Dict[str, Any]) -> tuple[bool, List[str]]:
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
        errors = []
        required = PlayerSummaryCSVColumns.all_columns()
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        return len(errors) == 0, errors
    @staticmethod
    def validate_possession_summary_csv(df: pd.DataFrame) -> tuple[bool, List[str]]:
        errors = []
        required = PossessionSummaryCSVColumns.all_columns()
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        if PossessionSummaryCSVColumns.TEAM_ID in df.columns:
            invalid_teams = df[~df[PossessionSummaryCSVColumns.TEAM_ID].isin([0, 1, -1])]
            if len(invalid_teams) > 0:
                errors.append(f"Invalid team_id values found: {invalid_teams[PossessionSummaryCSVColumns.TEAM_ID].unique()}")
        return len(errors) == 0, errors
class OutputPathResolver:
    def __init__(self, output_dir: Path | str, game_id: str | None = None):
        self.output_dir = Path(output_dir)
        self.game_id = game_id
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
        filename = f"team_{team_id}_heatmap.png"
        heatmap_dir = self.game_dir / OutputFiles.HEATMAP_DIR
        heatmap_dir.mkdir(exist_ok=True)
        return heatmap_dir / filename
    def all_expected_files(self) -> List[Path]:
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
        import re
        base_name = Path(video_name).stem
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{clean_name}_{timestamp}"
    def get_game_summary_path(self) -> Path:
        return self.game_dir / "game_summary.json"
def write_csv_headers(filepath: Path, columns: List[str]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
def append_csv_row(filepath: Path, row: Dict[str, Any], columns: List[str]) -> None:
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in columns])
def write_json_atomic(filepath: Path, data: Dict[str, Any], indent: int = 2) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path = filepath.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    temp_path.replace(filepath)
if __name__ == "__main__":
    row = TrackingRow(
        frame_id=1,
        object_id=42,
        team_id=0,
        cx=100.5,
        cy=200.3,
        speed=12.5,
    )
    print("Tracking row:", row.to_dict())
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
    resolver = OutputPathResolver("/tmp/test_output")
    print("Tracking CSV:", resolver.tracking_csv())