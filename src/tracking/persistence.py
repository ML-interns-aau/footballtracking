from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TrackingPersistence:
    def __init__(self, config, input_source: str, run_name: str | None = None) -> None:
        self.config = config
        self.input_source = input_source
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_root = Path(__file__).resolve().parents[2]
        self.run_dir = self._resolve_output_dir(config.OUTPUT_ROOT_DIR) / self.run_name
        self.video_dir = self.run_dir / "video"
        self.tracks_dir = self.run_dir / "tracks"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.tracks_dir.mkdir(parents=True, exist_ok=True)

        self.video_output_path = self.video_dir / "annotated.mp4"
        self.jsonl_path = self.tracks_dir / "tracks.jsonl"
        self.csv_path = self.tracks_dir / "tracks.csv"
        self.meta_path = self.run_dir / "run_metadata.json"

        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "frame_index",
                "timestamp_seconds",
                "track_id",
                "role",
                "jersey_label",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "bottom_x",
                "bottom_y",
            ],
        )
        self._csv_writer.writeheader()
        self._jsonl_file = self.jsonl_path.open("w", encoding="utf-8")

    def _resolve_output_dir(self, candidate: str) -> Path:
        path = Path(candidate)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def write_frame(
        self,
        frame_index: int,
        timestamp_seconds: float,
        frame_output: Any,
    ) -> None:
        payload = {
            "frame_index": frame_index,
            "timestamp_seconds": timestamp_seconds,
            "field_keypoints": frame_output.field_keypoints,
            "detections": [],
        }
        for det in frame_output.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            bx, by = det.bottom_center
            row = {
                "frame_index": frame_index,
                "timestamp_seconds": round(timestamp_seconds, 3),
                "track_id": det.track_id,
                "role": det.role,
                "jersey_label": det.jersey_label,
                "confidence": round(float(det.confidence), 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "bottom_x": bx,
                "bottom_y": by,
            }
            payload["detections"].append(row)
            self._csv_writer.writerow(row)

        self._jsonl_file.write(json.dumps(payload) + "\n")

    def write_summary(self, summary: dict) -> None:
        self.meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def close(self) -> None:
        self._jsonl_file.close()
        self._csv_file.close()
