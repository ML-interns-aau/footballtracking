"""Observability package — logging, timing metrics, run summaries, errors, preflight, job status."""

from .logger import get_logger, setup_logging
from .metrics import PipelineMetrics
from .run_summary import write_run_summary
from .errors import (
    PipelineError,
    ConfigurationError,
    ModelNotFoundError,
    VideoError,
    VideoOpenError,
    VideoCorruptError,
    VideoEmptyError,
    DetectionError,
    EmptyDetectionsWarning,
    TrackingError,
    ExportError,
    PipelineTimeoutError,
)
from .preflight import run_preflight
from .job_status import JobStatus, read_job_status

__all__ = [
    "get_logger", "setup_logging",
    "PipelineMetrics",
    "write_run_summary",
    "PipelineError", "ConfigurationError", "ModelNotFoundError",
    "VideoError", "VideoOpenError", "VideoCorruptError", "VideoEmptyError",
    "DetectionError", "EmptyDetectionsWarning",
    "TrackingError", "ExportError", "PipelineTimeoutError",
    "run_preflight",
    "JobStatus", "read_job_status",
]
