from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import cv2

try:
    from .classifier import ClassifiedTrack, JerseyRoleClassifier
    from .detection import FieldKeypointDetector, MultiObjectDetector
    from .utils import Annotators
except ImportError:
    from classifier import ClassifiedTrack, JerseyRoleClassifier
    from detection import FieldKeypointDetector, MultiObjectDetector
    from utils import Annotators


@dataclass
class FrameTrackingOutput:
    detections: List[ClassifiedTrack]
    field_keypoints: list[tuple[int, int]]
    annotated_frame: Any


class TrackingOrchestrator:
    def __init__(self, config) -> None:
        self.config = config
        self.detector = MultiObjectDetector(config)
        self.classifier = JerseyRoleClassifier(config)
        self.keypoint_detector = FieldKeypointDetector(config)

    def _draw_tracks(self, frame, tracks: List[ClassifiedTrack]) -> None:
        for t in tracks:
            x1, y1, x2, y2 = t.bbox_xyxy
            if t.role == "ball":
                Annotators.draw_triangle(frame, t.bottom_center, size=8, color=(0, 255, 255), thickness=2)
            else:
                center = (int((x1 + x2) / 2), int(y2))
                axes = (max(8, int((x2 - x1) * 0.45)), max(4, int((y2 - y1) * 0.12)))
                Annotators.draw_ellipse(frame, center, axes, t.color_bgr, thickness=2)
                Annotators.draw_edge_box(frame, t.bbox_xyxy, t.color_bgr, edge=10, thickness=2)

            cv2.putText(
                frame,
                f"{t.role}:{t.track_id}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                t.color_bgr,
                2,
            )

    def _draw_keypoints(self, frame, keypoints: list[tuple[int, int]]) -> None:
        for p in keypoints:
            Annotators.draw_vertex(frame, p, color=(255, 255, 255), radius=2)

    def process_frame(self, frame):
        detections = self.detector.detect(frame)
        classified = self.classifier.classify(frame, detections)
        kp_info = self.keypoint_detector.detect(frame) if self.config.ENABLE_FIELD_WARP else {"keypoints": []}

        annotated = frame.copy()
        self._draw_tracks(annotated, classified)
        self._draw_keypoints(annotated, kp_info["keypoints"])
        return FrameTrackingOutput(
            detections=classified,
            field_keypoints=kp_info["keypoints"],
            annotated_frame=annotated,
        )
