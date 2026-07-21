import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import pathlib
import platform
from collections import deque

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath


class FootballDetector:
    """YOLO-based player/ball detector, with decoy gating on the ball class.

    A generic COCO-trained model happily calls any round object a "sports
    ball" — sideline spare balls, sponsor logos on perimeter LED boards, etc.
    The gates below reject ball-class candidates that fall outside the pitch
    surface or above it (the advertising/crowd band), without touching the
    person class or raising confidence (which would just drop the small,
    blurry real ball along with the decoys).
    """

    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        conf: float = 0.30,
        iou: float = 0.40,
        device: str = "cpu",
        imgsz: int = 960,
        use_pitch_gate: bool = True,
        use_led_gate: bool = True,
        ball_weights: str | None = None,
        mask_history: int = 5,
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.CLASS_NAMES_DICT = self.model.model.names
        self.conf   = conf
        self.iou    = iou
        self.device = device
        self.imgsz  = imgsz

        self.use_pitch_gate = use_pitch_gate
        self.use_led_gate   = use_led_gate
        self._mask_history: deque = deque(maxlen=mask_history)
        self._smoothed_mask: np.ndarray | None = None

        # Optional football/SoccerNet-finetuned checkpoint dedicated to the ball
        # class; COCO YOLO still handles `person`. Ball detections from this
        # model are merged in before gating. The checkpoint may be single-class
        # (ball only) or multi-class (e.g. Player/Ball/Referee) -- look up the
        # "ball" class by name rather than assuming id 0, so multi-class
        # checkpoints don't get their player/referee detections mislabeled.
        self.ball_model = None
        self.ball_model_ball_class_id = 0
        if ball_weights:
            self.ball_model = YOLO(ball_weights)
            self.ball_model.to(device)
            ball_names = {cid: name.lower() for cid, name in self.ball_model.model.names.items()}
            matches = [cid for cid, name in ball_names.items() if name == "ball"]
            if matches:
                self.ball_model_ball_class_id = matches[0]
            elif len(ball_names) > 1:
                raise ValueError(
                    f"--ball_weights model has multiple classes {ball_names} but none named "
                    "'ball'; cannot tell which class to use."
                )

        self.last_gate_stats = {"pitch_rejected": 0, "led_rejected": 0}

    def detect(self, frame: np.ndarray) -> sv.Detections:
        results = self.model(
            frame,
            classes=[0, 32],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            agnostic_nms=True,
            verbose=False,
            device=self.device,
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        if self.ball_model is not None:
            detections = self._merge_ball_weights(frame, detections)

        if detections.class_id is not None and (self.use_pitch_gate or self.use_led_gate):
            detections = self._gate_ball_candidates(frame, detections)

        return detections

    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        detections = self.detect(frame)
        return detections[detections.class_id == 0]

    def detect_ball(self, frame: np.ndarray) -> sv.Detections:
        detections = self.detect(frame)
        return detections[detections.class_id == 32]

    # ------------------------------------------------------------------ #
    # Decoy gating
    # ------------------------------------------------------------------ #

    def _compute_pitch_mask(self, frame: np.ndarray) -> np.ndarray:
        """HSV green segmentation -> morphology -> largest CC -> temporal smoothing."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([30, 30, 30], dtype=np.uint8)
        upper = np.array([95, 255, 255], dtype=np.uint8)
        raw_mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if num_labels <= 1:
            largest_cc = closed
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = 1 + int(np.argmax(areas))
            largest_cc = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        self._mask_history.append(largest_cc)
        if self._smoothed_mask is None or self._smoothed_mask.shape != largest_cc.shape:
            self._smoothed_mask = largest_cc.astype(np.float32)
        else:
            # Running EMA rather than a hard union/intersection, so a single
            # noisy frame can't flip the mask but real changes still settle in.
            self._smoothed_mask = 0.6 * self._smoothed_mask + 0.4 * largest_cc.astype(np.float32)

        return (self._smoothed_mask > 127).astype(np.uint8) * 255

    @staticmethod
    def _pitch_top_boundary(pitch_mask: np.ndarray) -> float:
        cols_with_pitch = np.any(pitch_mask > 0, axis=0)
        if not np.any(cols_with_pitch):
            return 0.0
        top_rows = np.argmax(pitch_mask > 0, axis=0).astype(np.float64)
        top_rows = top_rows[cols_with_pitch]
        return float(np.percentile(top_rows, 10))

    def _gate_ball_candidates(self, frame: np.ndarray, detections: sv.Detections) -> sv.Detections:
        ball_mask = detections.class_id == 32
        if not np.any(ball_mask):
            return detections

        pitch_mask = self._compute_pitch_mask(frame)
        top_boundary = self._pitch_top_boundary(pitch_mask) if self.use_led_gate else 0.0
        h, w = pitch_mask.shape[:2]

        keep = np.ones(len(detections), dtype=bool)
        for i in np.where(ball_mask)[0]:
            x1, y1, x2, y2 = detections.xyxy[i]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cxi = int(np.clip(cx, 0, w - 1))
            cyi = int(np.clip(cy, 0, h - 1))

            if self.use_pitch_gate and pitch_mask[cyi, cxi] == 0:
                keep[i] = False
                self.last_gate_stats["pitch_rejected"] += 1
                continue

            if self.use_led_gate and cy < top_boundary - 5:
                keep[i] = False
                self.last_gate_stats["led_rejected"] += 1

        return detections[keep]

    def _merge_ball_weights(self, frame: np.ndarray, detections: sv.Detections) -> sv.Detections:
        results = self.ball_model(
            frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            verbose=False, device=self.device,
        )[0]
        ball_dets = sv.Detections.from_ultralytics(results)
        if ball_dets.class_id is not None:
            ball_dets = ball_dets[ball_dets.class_id == self.ball_model_ball_class_id]
        if len(ball_dets) == 0:
            return detections
        ball_dets.class_id = np.full(len(ball_dets), 32)
        non_ball = detections[detections.class_id != 32] if detections.class_id is not None else detections
        return sv.Detections.merge([non_ball, ball_dets])
