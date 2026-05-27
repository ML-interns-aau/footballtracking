from __future__ import annotations
class PipelineError(RuntimeError):
    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint
    def user_message(self) -> str:
        msg = str(self)
        if self.hint:
            msg += f"  →  {self.hint}"
        return msg
class ConfigurationError(PipelineError):
class ModelNotFoundError(ConfigurationError):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"Model weights not found: {path}",
            hint=(
                "Place yolov8m_fixed.pt in the models/ directory. "
                "Run `python download_model.py` to download it automatically."
            ),
        )
        self.path = path
class VideoError(PipelineError):
class VideoOpenError(VideoError):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"Cannot open video file: {path}",
            hint="Ensure the file exists and is a supported format (mp4, avi, mov, mkv).",
        )
        self.path = path
class VideoCorruptError(VideoError):
    def __init__(self, path: str, frame_idx: int = 0) -> None:
        super().__init__(
            f"Video file appears corrupt (failed at frame {frame_idx}): {path}",
            hint="Re-encode the video with ffmpeg: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`",
        )
        self.path = path
        self.frame_idx = frame_idx
class VideoEmptyError(VideoError):
    def __init__(self, path: str) -> None:
        super().__init__(
            f"Video contains zero frames: {path}",
            hint="The file may be empty or the codec is not supported on this system.",
        )
        self.path = path
class DetectionError(PipelineError):
class EmptyDetectionsWarning(DetectionError):
    def __init__(self, frames_processed: int) -> None:
        super().__init__(
            f"No objects were detected in any of the {frames_processed} processed frames.",
            hint=(
                "Lower the confidence threshold (--conf) or verify the model is trained "
                "for football footage. Check that the input video contains visible players."
            ),
        )
        self.frames_processed = frames_processed
class TrackingError(PipelineError):
class ExportError(PipelineError):
    def __init__(self, path: str, reason: str = "") -> None:
        super().__init__(
            f"Export failed for {path}" + (f": {reason}" if reason else ""),
            hint="Check available disk space and write permissions on the output directory.",
        )
        self.path = path
class PipelineTimeoutError(PipelineError):
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