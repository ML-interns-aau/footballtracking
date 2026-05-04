"""
Custom exception hierarchy for the Football Analytics Pipeline.

Raise these instead of generic exceptions so callers can distinguish
recoverable warnings from hard failures and surface user-friendly messages.

Hierarchy
---------
PipelineError                   # base — all pipeline failures
├── ConfigurationError          # bad args / missing config
│   └── ModelNotFoundError      # model weights file missing
├── VideoError                  # video I/O problems
│   ├── VideoOpenError          # cannot open video file
│   ├── VideoCorruptError       # video opened but unreadable frames
│   └── VideoEmptyError         # zero frames in video
├── DetectionError              # YOLO inference problems
│   └── EmptyDetectionsWarning  # non-fatal — zero detections this run
├── TrackingError               # tracker / team-classifier failures
├── ExportError                 # CSV / heatmap write failures
└── TimeoutError                # run exceeded wall-time limit
"""

from __future__ import annotations


class PipelineError(RuntimeError):
    """Base class for all pipeline errors."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def user_message(self) -> str:
        """Return a short, UI-safe description."""
        msg = str(self)
        if self.hint:
            msg += f"  →  {self.hint}"
        return msg


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigurationError(PipelineError):
    """Bad arguments or missing required configuration."""


class ModelNotFoundError(ConfigurationError):
    """Model weights file does not exist at the expected path."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Model weights not found: {path}",
            hint=(
                "Place yolov8m_fixed.pt in the models/ directory. "
                "Run `python download_model.py` to download it automatically."
            ),
        )
        self.path = path


# ── Video ─────────────────────────────────────────────────────────────────────

class VideoError(PipelineError):
    """Base class for all video I/O failures."""


class VideoOpenError(VideoError):
    """Cannot open the video file (missing, corrupt, or unsupported format)."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Cannot open video file: {path}",
            hint="Ensure the file exists and is a supported format (mp4, avi, mov, mkv).",
        )
        self.path = path


class VideoCorruptError(VideoError):
    """Video opened but frames cannot be decoded."""

    def __init__(self, path: str, frame_idx: int = 0) -> None:
        super().__init__(
            f"Video file appears corrupt (failed at frame {frame_idx}): {path}",
            hint="Re-encode the video with ffmpeg: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`",
        )
        self.path = path
        self.frame_idx = frame_idx


class VideoEmptyError(VideoError):
    """Video has zero readable frames."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Video contains zero frames: {path}",
            hint="The file may be empty or the codec is not supported on this system.",
        )
        self.path = path


# ── Detection ─────────────────────────────────────────────────────────────────

class DetectionError(PipelineError):
    """YOLO inference failure."""


class EmptyDetectionsWarning(DetectionError):
    """Non-fatal — entire run produced zero detections."""

    def __init__(self, frames_processed: int) -> None:
        super().__init__(
            f"No objects were detected in any of the {frames_processed} processed frames.",
            hint=(
                "Lower the confidence threshold (--conf) or verify the model is trained "
                "for football footage. Check that the input video contains visible players."
            ),
        )
        self.frames_processed = frames_processed


# ── Tracking ─────────────────────────────────────────────────────────────────

class TrackingError(PipelineError):
    """Tracker or team-classifier failure."""


# ── Export ────────────────────────────────────────────────────────────────────

class ExportError(PipelineError):
    """Failed to write output files (CSV, heatmap, annotated video)."""

    def __init__(self, path: str, reason: str = "") -> None:
        super().__init__(
            f"Export failed for {path}" + (f": {reason}" if reason else ""),
            hint="Check available disk space and write permissions on the output directory.",
        )
        self.path = path


# ── Timeout ───────────────────────────────────────────────────────────────────

class PipelineTimeoutError(PipelineError):
    """Pipeline exceeded the configured wall-time limit."""

    def __init__(self, limit_s: float, elapsed_s: float) -> None:
        super().__init__(
            f"Pipeline timed out after {elapsed_s:.0f}s (limit: {limit_s:.0f}s).",
            hint=(
                "Process fewer frames (--max_frames), use a GPU, "
                "or increase the timeout with --timeout_s."
            ),
        )
        self.limit_s   = limit_s
        self.elapsed_s = elapsed_s
