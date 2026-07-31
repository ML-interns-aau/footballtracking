import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import pathlib
import platform
from collections import deque

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath


def _tile_bounds(
    width: int, height: int, grid_rows: int, grid_cols: int, overlap_frac: float,
) -> list[tuple[int, int, int, int]]:
    tile_w = width / grid_cols
    tile_h = height / grid_rows
    overlap_x = tile_w * overlap_frac
    overlap_y = tile_h * overlap_frac
    tiles = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            x1 = max(0, int(c * tile_w - overlap_x))
            y1 = max(0, int(r * tile_h - overlap_y))
            x2 = min(width, int((c + 1) * tile_w + overlap_x))
            y2 = min(height, int((r + 1) * tile_h + overlap_y))
            tiles.append((x1, y1, x2, y2))
    return tiles


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
        gate_offpitch_players: bool = False,
        ball_conf: float | None = None,
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.CLASS_NAMES_DICT = self.model.model.names
        self.conf   = conf
        self.iou    = iou
        self.device = device
        self.imgsz  = imgsz
        self.ball_conf = ball_conf if ball_conf is not None else conf

        self.use_pitch_gate = use_pitch_gate
        self.use_led_gate   = use_led_gate
        self.gate_offpitch_players = gate_offpitch_players
        self._mask_history: deque = deque(maxlen=mask_history)
        self._smoothed_mask: np.ndarray | None = None

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

        self.last_gate_stats = {
            "pitch_rejected": 0, "led_rejected": 0, "offpitch_player_rejected": 0,
            "above_pitch_tagged": 0,
        }

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

        needs_pitch_mask = self.use_pitch_gate or self.use_led_gate or self.gate_offpitch_players
        pitch_mask = self._compute_pitch_mask(frame) if (needs_pitch_mask and detections.class_id is not None) else None

        if detections.class_id is not None and (self.use_pitch_gate or self.use_led_gate):
            detections = self._gate_ball_candidates(frame, detections, pitch_mask)

        if detections.class_id is not None and self.gate_offpitch_players:
            detections = self._gate_offpitch_players(detections, pitch_mask)

        return detections

    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        detections = self.detect(frame)
        return detections[detections.class_id == 0]

    def detect_ball(self, frame: np.ndarray) -> sv.Detections:
        detections = self.detect(frame)
        return detections[detections.class_id == 32]

    def detect_ball_tiled(
        self, frame: np.ndarray, grid_rows: int = 2, grid_cols: int = 3, overlap_frac: float = 0.2,
    ) -> sv.Detections:
        """Runs ball-class detection on overlapping crops of the frame and merges
        the results back to full-frame coordinates with NMS.

        A corner-kick ball in flight is often only ~5px across at full-frame
        imgsz -- near the detector's floor for a small object, since YOLO
        resizes the whole frame down to imgsz regardless of how much of it is
        pitch. Cropping to a tile before that resize effectively multiplies the
        ball's pixel footprint, recovering detections a full-frame pass misses
        entirely. Player detection doesn't need this (players are already
        large enough at full-frame imgsz), so this only ever supplies extra
        ball candidates alongside the normal detect() pass.
        """
        h, w = frame.shape[:2]
        tiles = _tile_bounds(w, h, grid_rows, grid_cols, overlap_frac)
        xyxy_list: list[tuple[float, float, float, float]] = []
        conf_list: list[float] = []
        for x1, y1, x2, y2 in tiles:
            tile_img = frame[y1:y2, x1:x2]
            if tile_img.size == 0:
                continue
            if self.ball_model is not None:
                results = self.ball_model(
                    tile_img, conf=self.ball_conf, iou=self.iou, imgsz=self.imgsz,
                    verbose=False, device=self.device,
                )[0]
                dets = sv.Detections.from_ultralytics(results)
                if dets.class_id is not None:
                    dets = dets[dets.class_id == self.ball_model_ball_class_id]
            else:
                results = self.model(
                    tile_img, classes=[32], conf=self.ball_conf, iou=self.iou, imgsz=self.imgsz,
                    agnostic_nms=True, verbose=False, device=self.device,
                )[0]
                dets = sv.Detections.from_ultralytics(results)
            for i in range(len(dets)):
                bx1, by1, bx2, by2 = dets.xyxy[i]
                conf = float(dets.confidence[i]) if dets.confidence is not None else 1.0
                xyxy_list.append((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1))
                conf_list.append(conf)

        if not xyxy_list:
            return sv.Detections.empty()

        merged = sv.Detections(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            confidence=np.array(conf_list, dtype=np.float32),
            class_id=np.full(len(xyxy_list), 32),
        )
        return merged.with_nms(threshold=self.iou, class_agnostic=True)


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

    def _gate_ball_candidates(self, frame: np.ndarray, detections: sv.Detections, pitch_mask: np.ndarray) -> sv.Detections:
        ball_mask = detections.class_id == 32
        if not np.any(ball_mask):
            return detections

        top_boundary = self._pitch_top_boundary(pitch_mask) if self.use_led_gate else 0.0
        h, w = pitch_mask.shape[:2]

        above_pitch = np.zeros(len(detections), dtype=bool)
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
                above_pitch[i] = True
                self.last_gate_stats["above_pitch_tagged"] += 1

        gated = detections[keep]
        gated.data["above_pitch"] = above_pitch[keep]
        return gated

    def _gate_offpitch_players(self, detections: sv.Detections, pitch_mask: np.ndarray) -> sv.Detections:
        """Rejects person-class detections whose feet (bbox bottom-center) fall
        outside the pitch surface -- bench staff, subs, ball boys, and sideline
        officials, which a plain person detector doesn't distinguish from
        on-field players.

        The HSV green segmentation is unreliable right at the pitch boundary
        (goal frame/netting shadow, corner arc, touchline paint) -- the same
        limitation documented above for the ball gate. Dilating the mask to
        compensate doesn't work: the perimeter walkway/technical area sits
        immediately adjacent to the pitch edge in image space, so any dilation
        wide enough to bridge the corner-arc color gap just as readily bridges
        into that walkway, wrongly re-admitting bench staff. Instead, reuse the
        top-boundary cue the LED gate already computes for the ball: reject a
        player only when BOTH signals agree they're off-pitch -- outside the
        green mask AND above the pitch's visible top edge (the crowd/technical-
        area band). A corner-taker standing on a real, if color-ambiguous,
        patch of grass is below that boundary and still passes; technical-area
        staff standing above it fail both.
        """
        player_mask = detections.class_id == 0
        if not np.any(player_mask):
            return detections

        top_boundary = self._pitch_top_boundary(pitch_mask)
        h, w = pitch_mask.shape[:2]
        keep = np.ones(len(detections), dtype=bool)
        for i in np.where(player_mask)[0]:
            x1, y1, x2, y2 = detections.xyxy[i]
            foot_x, foot_y = (x1 + x2) / 2.0, y2
            fxi = int(np.clip(foot_x, 0, w - 1))
            fyi = int(np.clip(foot_y, 0, h - 1))
            off_color = pitch_mask[fyi, fxi] == 0
            off_boundary = foot_y < top_boundary - 5
            if off_color and off_boundary:
                keep[i] = False
                self.last_gate_stats["offpitch_player_rejected"] += 1

        return detections[keep]

    def _merge_ball_weights(self, frame: np.ndarray, detections: sv.Detections) -> sv.Detections:
        results = self.ball_model(
            frame, conf=self.ball_conf, iou=self.iou, imgsz=self.imgsz,
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
