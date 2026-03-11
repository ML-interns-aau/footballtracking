from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import cv2
import numpy as np


@dataclass
class ClassifiedTrack:
    track_id: int
    role: str
    jersey_label: str
    color_bgr: tuple[int, int, int]
    bbox_xyxy: tuple[int, int, int, int]
    bottom_center: tuple[int, int]
    confidence: float

class JerseyRoleClassifier:
    def __init__(self, config) -> None:
        self.config = config
        self.palette: Dict[str, np.ndarray] = {
            "team_1": np.array(self.config.TEAM1_COLOR, dtype=np.float32),
            "team_2": np.array(self.config.TEAM2_COLOR, dtype=np.float32),
            "goalkeeper": np.array(self.config.GOALKEEPER_COLOR, dtype=np.float32),
            "referee": np.array(self.config.REFEREE_COLOR, dtype=np.float32),
        }
        
    def _extract_jersey_patch(
        self, frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int]
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        h = max(1, y2 - y1)
        torso_top = y1 + int(0.2 * h)
        torso_bottom = y1 + int(0.6 * h)
        
        x1 = max(0, x1)
        y1 = max(0, torso_top)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], torso_bottom)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    def _dominant_bgr(self, patch: np.ndarray) -> np.ndarray:
        if patch.size == 0:
            return np.array((128, 128, 128), dtype=np.float32)
        sample = patch.reshape(-1, 3).astype(np.float32)
        return np.median(sample, axis=0)

    def _nearest_jersey_label(self, color_bgr: np.ndarray) -> str:
        best = "unknown"
        best_dist = float("inf")
        for label, anchor in self.palette.items():
            dist = float(np.linalg.norm(color_bgr - anchor))
            if dist < best_dist:
                best_dist = dist
                best = label
        return best

    def _role_from_class_and_jersey(self, class_name: str, jersey_label: str) -> str:
        if class_name in {"sports ball", "ball"}:
            return "ball"
        if class_name != "person":
            return class_name
        if jersey_label == "referee":
            return "referee"
        if jersey_label == "goalkeeper":
            return "goalkeeper"
        return "player"

    def classify(self, frame: np.ndarray, detections: List) -> List[ClassifiedTrack]:
        output: List[ClassifiedTrack] = []
        for det in detections:
            if det.class_id in self.config.BALL_CLASS_IDS or det.class_name in {"sports ball", "ball"}:
                output.append(
                    ClassifiedTrack(
                        track_id=det.track_id,
                        role="ball",
                        jersey_label="ball",
                        color_bgr=(0, 255, 255),
                        bbox_xyxy=det.bbox_xyxy,
                        bottom_center=det.bottom_center,
                        confidence=det.confidence,
                    )
                )
                continue

            if det.class_name != "person":
                output.append(
                    ClassifiedTrack(
                        track_id=det.track_id,
                        role=det.class_name,
                        jersey_label=det.class_name,
                        color_bgr=(180, 180, 180),
                        bbox_xyxy=det.bbox_xyxy,
                        bottom_center=det.bottom_center,
                        confidence=det.confidence,
                    )
                )
                continue

            jersey_patch = self._extract_jersey_patch(frame, det.bbox_xyxy)
            dominant = self._dominant_bgr(jersey_patch)
            jersey_label = self._nearest_jersey_label(dominant)
            role = self._role_from_class_and_jersey(det.class_name, jersey_label)
            color_bgr = tuple(int(v) for v in self.palette.get(jersey_label, dominant))

            output.append(
                ClassifiedTrack(
                    track_id=det.track_id,
                    role=role,
                    jersey_label=jersey_label,
                    color_bgr=color_bgr,
                    bbox_xyxy=det.bbox_xyxy,
                    bottom_center=det.bottom_center,
                    confidence=det.confidence,
                )
            )
        return output
