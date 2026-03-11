from __future__ import annotations

import base64
from dataclasses import dataclass
import time
from typing import List

import cv2
import numpy as np
import requests
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class TrackDetection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]

    @property
    def bottom_center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return int((x1 + x2) / 2), int(y2)


class MultiObjectDetector:
    def __init__(self, config) -> None:
        self.config = config
        self.backend = str(getattr(self.config, "DETECTOR_BACKEND", "ultralytics")).lower()
        self._tracker = _CentroidTracker()

        if self.backend == "roboflow":
            model_id = str(getattr(self.config, "ROBOFLOW_MODEL_ID", "")).strip()
            api_key = str(getattr(self.config, "ROBOFLOW_API_KEY", "")).strip()
            if not model_id or not api_key:
                raise RuntimeError(
                    "Roboflow backend selected, but ROBOFLOW_MODEL_ID/ROBOFLOW_API_KEY is missing."
                )
            self.model = _RoboflowDetector(config)
            return

        if YOLO is None:
            raise RuntimeError(
                "ultralytics is required for tracking. Install with `pip install ultralytics`."
            )
        self.model = YOLO(self.config.MODEL_PATH)

    def detect(self, frame: np.ndarray) -> List[TrackDetection]:
        if self.backend == "roboflow":
            return self.model.detect(frame, self._tracker)

        class_filter = self.config.CLASS_FILTER if self.config.CLASS_FILTER else None
        result = self.model.track(
            source=frame,
            persist=True,
            conf=self.config.CONFIDENCE,
            iou=self.config.IOU_THRESHOLD,
            tracker=self.config.TRACKER_TYPE,
            device=self.config.DEVICE,
            imgsz=self.config.INFERENCE_IMGSZ,
            max_det=self.config.MAX_DETECTIONS,
            classes=class_filter,
            half=bool(self.config.USE_HALF_PRECISION and self.config.DEVICE == "cuda"),
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        names = result.names if hasattr(result, "names") else {}
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.arange(len(boxes), dtype=int)

        detections: List[TrackDetection] = []
        for box, class_id, conf, track_id in zip(boxes, cls_ids, confs, track_ids):
            class_name = names.get(int(class_id), str(class_id))
            detections.append(
                TrackDetection(
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=str(class_name),
                    confidence=float(conf),
                    bbox_xyxy=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                )
            )
        return detections


class _CentroidTracker:
    def __init__(self, distance_threshold: float = 60.0) -> None:
        self.distance_threshold = float(distance_threshold)
        self.previous: dict[int, tuple[int, int]] = {}
        self.next_id = 1

    def assign(self, centers: list[tuple[int, int]]) -> list[int]:
        if not centers:
            self.previous = {}
            return []

        remaining_prev = dict(self.previous)
        ids: list[int] = []
        for center in centers:
            best_id = None
            best_distance = float("inf")
            for track_id, prev_center in remaining_prev.items():
                dist = float(np.linalg.norm(np.array(center) - np.array(prev_center)))
                if dist < best_distance:
                    best_distance = dist
                    best_id = track_id

            if best_id is not None and best_distance <= self.distance_threshold:
                ids.append(best_id)
                remaining_prev.pop(best_id, None)
            else:
                ids.append(self.next_id)
                self.next_id += 1

        self.previous = {track_id: center for track_id, center in zip(ids, centers)}
        return ids


class _RoboflowDetector:
    def __init__(self, config) -> None:
        self.config = config
        self.model_id = str(self.config.ROBOFLOW_MODEL_ID).strip()
        self.api_key = str(self.config.ROBOFLOW_API_KEY).strip()
        self.url = f"https://detect.roboflow.com/{self.model_id}"
        self.session = requests.Session()
        self.max_infer_size = max(0, int(getattr(self.config, "ROBOFLOW_MAX_INFER_SIZE", 640)))
        self.jpeg_quality = max(40, min(95, int(getattr(self.config, "ROBOFLOW_JPEG_QUALITY", 75))))
        self.infer_stride = max(1, int(getattr(self.config, "ROBOFLOW_INFER_EVERY_N_FRAMES", 2)))
        self.connect_timeout = float(getattr(self.config, "ROBOFLOW_CONNECT_TIMEOUT_SECONDS", 5))
        self.read_timeout = float(getattr(self.config, "ROBOFLOW_READ_TIMEOUT_SECONDS", 20))
        self.max_retries = max(0, int(getattr(self.config, "ROBOFLOW_MAX_RETRIES", 2)))
        self.retry_backoff = max(0.0, float(getattr(self.config, "ROBOFLOW_RETRY_BACKOFF_SECONDS", 0.7)))
        self._frame_counter = 0
        self._last_predictions: list[dict] = []
        self._last_scale = 1.0
        self._network_warned = False

    @staticmethod
    def _canonical_class_name(raw_name: str) -> str:
        name = raw_name.strip().lower().replace("-", " ").replace("_", " ")
        if "ball" in name:
            return "sports ball"
        if "player" in name or "person" in name:
            return "person"
        if "refree" in name or "referee" in name:
            return "referee"
        return name

    def _prepare_for_infer(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        longest = max(h, w)
        if self.max_infer_size <= 0 or longest <= self.max_infer_size:
            return frame, 1.0
        scale = self.max_infer_size / float(longest)
        resized = cv2.resize(
            frame,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _infer(self, frame: np.ndarray) -> tuple[list[dict], float]:
        infer_frame, scale = self._prepare_for_infer(frame)
        ok, encoded = cv2.imencode(
            ".jpg",
            infer_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return [], scale
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.url,
                    params={
                        "api_key": self.api_key,
                        "confidence": int(float(self.config.CONFIDENCE) * 100),
                        "overlap": int(getattr(self.config, "ROBOFLOW_OVERLAP", 30)),
                        "format": "json",
                    },
                    data=image_b64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=(self.connect_timeout, self.read_timeout),
                )
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise
                if self.retry_backoff > 0:
                    time.sleep(self.retry_backoff * (attempt + 1))
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Roboflow request failed before sending.")
        if response.status_code in {401, 403}:
            body = (response.text or "").strip()
            raise RuntimeError(
                f"Roboflow auth/access error (HTTP {response.status_code}) for model "
                f"'{self.model_id}'. Response: {body[:500]}"
            )
        response.raise_for_status()
        payload = response.json()
        return payload.get("predictions", []) or [], scale

    def detect(self, frame: np.ndarray, tracker: _CentroidTracker) -> List[TrackDetection]:
        self._frame_counter += 1
        if self._frame_counter % self.infer_stride == 0 or not self._last_predictions:
            try:
                predictions, scale = self._infer(frame)
                self._last_predictions = predictions
                self._last_scale = scale
                self._network_warned = False
            except requests.exceptions.RequestException as exc:
                predictions = self._last_predictions
                scale = self._last_scale
                if not self._network_warned:
                    print(f"[tracking] Roboflow request issue ({type(exc).__name__}); reusing last detections.")
                    self._network_warned = True
        else:
            predictions = self._last_predictions
            scale = self._last_scale
        if not predictions:
            return []

        inv_scale = 1.0 / scale if scale > 0 else 1.0
        detections_raw: list[tuple[int, int, int, int, int, str, float, tuple[int, int]]] = []
        for pred in predictions:
            x = float(pred.get("x", 0)) * inv_scale
            y = float(pred.get("y", 0)) * inv_scale
            w = float(pred.get("width", 0)) * inv_scale
            h = float(pred.get("height", 0)) * inv_scale
            x1 = max(0, int(round(x - w / 2)))
            y1 = max(0, int(round(y - h / 2)))
            x2 = max(x1 + 1, int(round(x + w / 2)))
            y2 = max(y1 + 1, int(round(y + h / 2)))
            class_name = self._canonical_class_name(str(pred.get("class", "unknown")))
            if class_name == "person":
                class_id = int(self.config.CLASS_PERSON)
            elif class_name == "sports ball":
                class_id = int(next(iter(self.config.BALL_CLASS_IDS), 32))
            else:
                class_id = -1
            confidence = float(pred.get("confidence", 0.0))
            center = (int((x1 + x2) / 2), int(y2))
            detections_raw.append((x1, y1, x2, y2, class_id, class_name, confidence, center))

        centers = [d[-1] for d in detections_raw]
        track_ids = tracker.assign(centers)
        detections: List[TrackDetection] = []
        for track_id, det in zip(track_ids, detections_raw):
            x1, y1, x2, y2, class_id, class_name, confidence, _ = det
            detections.append(
                TrackDetection(
                    track_id=int(track_id),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return detections


class FieldKeypointDetector:
    def __init__(self, config) -> None:
        self.config = config

    def _estimate_field_quad(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 30, 30), (95, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        if not contours:
            return np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

        biggest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(biggest)
        box = cv2.boxPoints(rect).astype(np.float32)
        s = box.sum(axis=1)
        d = np.diff(box, axis=1)
        ordered = np.array(
            [box[np.argmin(s)], box[np.argmin(d)], box[np.argmax(s)], box[np.argmax(d)]],
            dtype=np.float32,
        )
        return ordered

    def detect(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        src_quad = self._estimate_field_quad(frame)
        dst_quad = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        transform = cv2.getPerspectiveTransform(src_quad, dst_quad)
        inverse_transform = cv2.getPerspectiveTransform(dst_quad, src_quad)

        warped = cv2.warpPerspective(frame, transform, (w, h))
        template = self.config.FIELD_KEYPOINT_TEMPLATE
        warped_points = np.float32([[x * (w - 1), y * (h - 1)] for x, y in template]).reshape(-1, 1, 2)
        original_points = cv2.perspectiveTransform(warped_points, inverse_transform).reshape(-1, 2)

        keypoints = [(int(p[0]), int(p[1])) for p in original_points]
        return {
            "warp_matrix": transform,
            "inverse_warp_matrix": inverse_transform,
            "warped_frame": warped,
            "keypoints": keypoints,
        }
