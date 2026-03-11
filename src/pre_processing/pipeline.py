from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from .enhancer import FrameEnhancer
    from .normalization import FrameNormalizer
    from .video_loader import VideoLoader
except ImportError:
    from enhancer import FrameEnhancer
    from normalization import FrameNormalizer
    from video_loader import VideoLoader


@dataclass(frozen=True)
class ExtractionConfig:
    video_path: str = "data/raw/114.mp4"
    output_dir: str = "data/interim/114_frames"
    target_sample_fps: float = 15.0
    default_fps: float = 30.0
    jpeg_quality: int = 90
    min_blur_score: float = 80.0
    min_brightness: float = 15.0
    max_brightness: float = 240.0
    enable_quality_filter: bool = True
    enable_enhancement: bool = False
    quality_check_downscale: float = 0.5
    write_workers: int = 2
    max_pending_writes: int = 128
    save_run_metadata: bool = True


class FrameExtractionPipeline:
    def __init__(
        self,
        config: ExtractionConfig | None = None,
        loader: VideoLoader | None = None,
        normalizer: FrameNormalizer | None = None,
        enhancer: FrameEnhancer | None = None,
    ) -> None:
        self.config = config or ExtractionConfig()
        self.project_root = Path(__file__).resolve().parents[2]
        self.loader = loader or VideoLoader(
            video_path=self.config.video_path,
            default_fps=self.config.default_fps,
        )
        self.normalizer = normalizer or FrameNormalizer()
        self.enhancer = enhancer or FrameEnhancer()
        self.output_dir = self._resolve_output_dir(self.config.output_dir)

    def _resolve_output_dir(self, candidate: str | Path) -> Path:
        path = Path(candidate)
        if path.is_absolute():
            return path
        root_relative = (self.project_root / path).resolve()
        cwd_relative = (Path.cwd() / path).resolve()
        if root_relative.parent.exists():
            return root_relative
        return cwd_relative

    def _is_frame_usable(self, frame: np.ndarray) -> tuple[bool, float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        downscale = self.config.quality_check_downscale
        if 0 < downscale < 1:
            gray = cv2.resize(
                gray,
                (0, 0),
                fx=downscale,
                fy=downscale,
                interpolation=cv2.INTER_AREA,
            )
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(gray.mean())

        usable = (
            blur_score >= self.config.min_blur_score
            and self.config.min_brightness <= brightness <= self.config.max_brightness
        )
        return usable, float(blur_score), brightness

    def _write_frame_task(self, output_path: Path, frame: np.ndarray) -> bool:
        return cv2.imwrite(
            str(output_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
        )

    def _write_frame_task_named(
        self, output_path: Path, frame: np.ndarray, output_name: str
    ) -> tuple[bool, str]:
        return self._write_frame_task(output_path, frame), output_name

    def _drain_pending_writes(
        self,
        pending: list[Future],
        required_max: int,
        failed_writes: list[str],
    ) -> None:
        while len(pending) > required_max:
            future = pending.pop(0)
            ok, output_name = future.result()
            if not ok:
                failed_writes.append(output_name)

    def run(self) -> dict:
        started_at = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = self.loader.get_metadata()

        print(
            f"Processing {metadata.path} ({metadata.width}x{metadata.height}, "
            f"{metadata.fps:.2f} FPS, {metadata.frame_count} frames)"
        )

        saved_count = 0
        dropped_quality_count = 0
        sampled_count = 0

        pending_writes: list[Future] = []
        failed_writes: list[str] = []
        max_pending = max(1, self.config.max_pending_writes)
        write_workers = max(1, self.config.write_workers)

        with ThreadPoolExecutor(max_workers=write_workers) as writer_pool:
            for packet in self.loader.iter_sampled_frames(
                target_sample_fps=self.config.target_sample_fps
            ):
                sampled_count += 1
                normalized = self.normalizer.normalize(packet.frame)
                usable, blur_score, brightness = True, 0.0, 0.0
                if self.config.enable_quality_filter:
                    # Gate low-quality frames early to avoid expensive enhancement work.
                    usable, blur_score, brightness = self._is_frame_usable(normalized)
                    if not usable:
                        dropped_quality_count += 1
                        continue
                output_frame = (
                    self.enhancer.enhance(normalized)
                    if self.config.enable_enhancement
                    else normalized
                )

                output_name = (
                    f"frame_{saved_count:06d}_src_{packet.frame_index:06d}"
                    f"_t_{packet.timestamp_seconds:09.3f}.jpg"
                )
                output_path = self.output_dir / output_name
                pending_writes.append(
                    writer_pool.submit(
                        self._write_frame_task_named,
                        output_path,
                        output_frame,
                        output_name,
                    )
                )
                saved_count += 1

                self._drain_pending_writes(
                    pending=pending_writes,
                    required_max=max_pending,
                    failed_writes=failed_writes,
                )

                if saved_count % 100 == 0:
                    print(
                        "Saved "
                        f"{saved_count} frames (latest blur={blur_score:.1f}, "
                        f"brightness={brightness:.1f})"
                    )

            self._drain_pending_writes(
                pending=pending_writes,
                required_max=0,
                failed_writes=failed_writes,
            )
        if failed_writes:
            saved_count -= len(failed_writes)
            print(f"Warning: failed to write {len(failed_writes)} frames.")

        elapsed_seconds = time.perf_counter() - started_at
        throughput_fps = sampled_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
        result = {
            "video_metadata": asdict(metadata),
            "config": asdict(self.config),
            "sampled_frames": sampled_count,
            "saved_frames": saved_count,
            "dropped_low_quality_frames": dropped_quality_count,
            "failed_writes": len(failed_writes),
            "elapsed_seconds": elapsed_seconds,
            "throughput_fps": throughput_fps,
            "output_dir": str(self.output_dir),
        }
        if self.config.save_run_metadata:
            metadata_path = self.output_dir / "run_metadata.json"
            metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print(
            f"Finished. Saved {saved_count} frames, dropped {dropped_quality_count} "
            f"low-quality frames in {elapsed_seconds:.2f}s "
            f"({throughput_fps:.2f} sampled FPS)."
        )
        return result
