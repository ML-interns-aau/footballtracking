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

from src.engine.ball_tracker import BallTracker
from src.analytics.camera_motion import CameraMotionEstimator
from src.exporters.data_exporter import DataExporter
from src.engine.detector import FootballDetector
from src.analytics.events import EventsDetector
from src.analytics.heatmap_analyzer import HeatmapAnalyzer
from src.exporters.output_schema import OutputFiles, OutputPathResolver
from src.exporters.player_summary_csv_builder import PlayerSummaryCSVBuilder
from src.exporters.possession_summary_csv_builder import PossessionSummaryCSVBuilder
from src.analytics.speed_estimator import SpeedEstimator
from src.engine.team_classifier import TeamClassifier
from src.engine.tracker import FootballTracker
from src.exporters.tracking_csv_builder import TrackingCSVBuilder
from src.visualization.visualizer import PipelineVisualizer

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
