"""Football tracking pipeline — public API.

Import the components you need directly from this package:

    from src.pipeline import (
        FootballDetector,
        FootballTracker,
        TeamClassifier,
        BallTracker,
        CameraMotionEstimator,
        SpeedEstimator,
        HeatmapAnalyzer,
        PipelineVisualizer,
        DataExporter,
        EventsDetector,
        TrackingCSVBuilder,
        PlayerSummaryCSVBuilder,
        PossessionSummaryCSVBuilder,
        OutputFiles,
        OutputPathResolver,
    )
"""

from src.pipeline.ball_tracker import BallTracker
from src.pipeline.camera_motion import CameraMotionEstimator
from src.pipeline.data_exporter import DataExporter
from src.pipeline.detector import FootballDetector
from src.pipeline.events import EventsDetector
from src.pipeline.heatmap_analyzer import HeatmapAnalyzer
from src.pipeline.output_schema import OutputFiles, OutputPathResolver
from src.pipeline.player_summary_csv_builder import PlayerSummaryCSVBuilder
from src.pipeline.possession_summary_csv_builder import PossessionSummaryCSVBuilder
from src.pipeline.speed_estimator import SpeedEstimator
from src.pipeline.team_classifier import TeamClassifier
from src.pipeline.tracker import FootballTracker
from src.pipeline.tracking_csv_builder import TrackingCSVBuilder
from src.pipeline.visualizer import PipelineVisualizer

__all__ = [
    "BallTracker",
    "CameraMotionEstimator",
    "DataExporter",
    "EventsDetector",
    "FootballDetector",
    "FootballTracker",
    "HeatmapAnalyzer",
    "OutputFiles",
    "OutputPathResolver",
    "PipelineVisualizer",
    "PlayerSummaryCSVBuilder",
    "PossessionSummaryCSVBuilder",
    "SpeedEstimator",
    "TeamClassifier",
    "TrackingCSVBuilder",
]
