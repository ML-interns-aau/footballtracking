from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    timestamp_seconds: float
    frame: np.ndarray


class VideoLoader:
    def __init__(self, video_path: str, default_fps: float = 30.0) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.video_path = self._resolve_path(video_path)
        self.default_fps = float(default_fps)

    def _resolve_path(self, candidate: str | Path) -> Path:
        path = Path(candidate)
        if path.is_absolute():
            return path
        root_relative = (self.project_root / path).resolve()
        if root_relative.exists():
            return root_relative
        return (Path.cwd() / path).resolve()

    def validate_video_path(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

    def open_capture(self) -> cv2.VideoCapture:
        self.validate_video_path()
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        return cap

    def get_metadata(self) -> VideoMetadata:
        cap = self.open_capture()
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if fps <= 0:
                fps = self.default_fps
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_seconds = (frame_count / fps) if fps > 0 else 0.0
            return VideoMetadata(
                path=str(self.video_path),
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                duration_seconds=duration_seconds,
            )
        finally:
            cap.release()

    def iter_sampled_frames(
        self, target_sample_fps: float = 15.0
    ) -> Iterator[FramePacket]:
        if target_sample_fps <= 0:
            raise ValueError("target_sample_fps must be > 0")

        cap = self.open_capture()
        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if source_fps <= 0:
                source_fps = self.default_fps
            skip_interval = max(1, round(source_fps / target_sample_fps))

            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None:
                    frame_index += 1
                    continue
                if frame_index % skip_interval == 0:
                    yield FramePacket(
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / source_fps,
                        frame=frame,
                    )
                frame_index += 1
        finally:
            cap.release()
