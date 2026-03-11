from __future__ import annotations

from dataclasses import replace
from typing import Any

try:
    from .enhancer import FrameEnhancer
    from .pipeline import ExtractionConfig, FrameExtractionPipeline
except ImportError:
    from enhancer import FrameEnhancer
    from pipeline import ExtractionConfig, FrameExtractionPipeline


class FrameExtractor:
    """
    Backward-compatible wrapper around the new modular extraction pipeline.
    """

    def __init__(
        self,
        video_path: str = "data/raw/114.mp4",
        output_dir: str = "data/interim/114_frames",
        target_sample_fps: float = 15,
        default_fps: float = 30,
        jpeg_quality: int = 90,
        enhancer: FrameEnhancer | None = None,
        **pipeline_overrides: Any,
    ) -> None:
        base_config = ExtractionConfig(
            video_path=video_path,
            output_dir=output_dir,
            target_sample_fps=target_sample_fps,
            default_fps=default_fps,
            jpeg_quality=jpeg_quality,
        )
        self.config = replace(base_config, **pipeline_overrides)
        self.pipeline = FrameExtractionPipeline(config=self.config, enhancer=enhancer)

    def run(self) -> dict:
        return self.pipeline.run()


if __name__ == "__main__":
    FrameExtractor().run()
