# Phase 5 — Integration: Wire the Fine-Tuned YOLO Into the Pipeline
# File: .github/instructions/05-integration.instructions.md
# Apply to: detector.py, team_classifier.py, and the main inference entry point

## Goal

Replace the two-stage pipeline (generic YOLO detector + HSV TeamClassifier) with
a single fine-tuned YOLO model that directly outputs team-aware class labels.
The fine-tuned model predicts 5 classes in one forward pass:

```
0: player_left        → team A outfield player
1: player_right       → team B outfield player
2: goalkeeper_left    → team A goalkeeper
3: goalkeeper_right   → team B goalkeeper
4: referee            → referee (any role)
```

---

## Architecture After Integration

```
Before:
  frame → YOLOv11 (person/gk/ref) → TeamClassifier (HSV KMeans) → team labels

After:
  frame → Fine-tuned YOLO (5 classes) → team labels directly
```

The `TeamClassifier` HSV pipeline becomes a **fallback only** — used when the
fine-tuned model is not available or when confidence is below threshold.

---

## Changes to `detector.py`

### Add a `FineTunedDetector` class

Add this class alongside the existing `FootballDetector`. Do not modify the existing
class — it remains as the fallback.

```python
class FineTunedDetector:
    """
    Wraps the fine-tuned YOLO model that predicts team-aware classes directly.

    Class map:
        0: player_left
        1: player_right
        2: goalkeeper_left
        3: goalkeeper_right
        4: referee

    The original 'person' / 'goalkeeper' / 'referee' generic detector is replaced
    by this model when a fine-tuned checkpoint is available.
    """

    CLASS_NAMES = [
        "player_left",
        "player_right",
        "goalkeeper_left",
        "goalkeeper_right",
        "referee",
    ]

    # Map predicted class_id → (role, team) for downstream consumers
    CLASS_TO_ROLE_TEAM: dict[int, tuple[str, str | None]] = {
        0: ("player",     "left"),
        1: ("player",     "right"),
        2: ("goalkeeper", "left"),
        3: ("goalkeeper", "right"),
        4: ("referee",    None),
    }

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.35,
        iou_threshold:  float = 0.45,
        device: str = "auto",
    ):
        from ultralytics import YOLO
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Fine-tuned model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.conf  = conf_threshold
        self.iou   = iou_threshold

        # Device selection
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Runs inference on a single BGR frame.

        Returns:
            List of detection dicts, one per detected object:
            {
                "bbox":       [x1, y1, x2, y2],   # absolute pixels
                "confidence": float,
                "class_id":   int,                  # 0–4
                "class_name": str,                  # e.g. "player_left"
                "role":       str,                  # "player"|"goalkeeper"|"referee"
                "team":       str | None,           # "left"|"right"|None
            }
        """
        results = self.model.predict(
            source  = frame,
            conf    = self.conf,
            iou     = self.iou,
            device  = self.device,
            verbose = False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes      = result.boxes.xyxy.cpu().numpy()    # (N, 4)
            confs      = result.boxes.conf.cpu().numpy()    # (N,)
            class_ids  = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for bbox, conf, cls_id in zip(boxes, confs, class_ids):
                role, team = self.CLASS_TO_ROLE_TEAM.get(cls_id, ("unknown", None))
                detections.append({
                    "bbox":       bbox.tolist(),
                    "confidence": float(conf),
                    "class_id":   cls_id,
                    "class_name": self.CLASS_NAMES[cls_id],
                    "role":       role,
                    "team":       team,
                })

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """
        Runs inference on a batch of frames. More efficient than calling
        detect() in a loop for video processing.
        """
        results = self.model.predict(
            source  = frames,
            conf    = self.conf,
            iou     = self.iou,
            device  = self.device,
            stream  = True,
            verbose = False,
        )
        return [self._parse_result(r) for r in results]

    def _parse_result(self, result) -> list[dict]:
        detections = []
        if result.boxes is None:
            return detections
        boxes     = result.boxes.xyxy.cpu().numpy()
        confs     = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        for bbox, conf, cls_id in zip(boxes, confs, class_ids):
            role, team = self.CLASS_TO_ROLE_TEAM.get(cls_id, ("unknown", None))
            detections.append({
                "bbox":       bbox.tolist(),
                "confidence": float(conf),
                "class_id":   cls_id,
                "class_name": self.CLASS_NAMES[cls_id],
                "role":       role,
                "team":       team,
            })
        return detections
```

---

## Changes to `team_classifier.py`

### Add a `ModelBasedTeamClassifier` wrapper

Add this class. It wraps `FineTunedDetector` and provides the same interface as
the existing `TeamClassifier` so callers don't need to change.

```python
class ModelBasedTeamClassifier:
    """
    Drop-in replacement for TeamClassifier that uses the fine-tuned YOLO model
    instead of HSV color clustering.

    Interface is identical to TeamClassifier for backward compatibility:
        results = classifier.classify_players(frame, detections, tracker_ids)

    Falls back to the legacy HSV TeamClassifier if the model is unavailable or
    if confidence is below min_conf_fallback.
    """

    def __init__(
        self,
        model_path: str | Path,
        fallback_classifier: "TeamClassifier | None" = None,
        min_conf_fallback: float = 0.30,
    ):
        self.min_conf_fallback = min_conf_fallback
        self.fallback = fallback_classifier

        try:
            self.detector = FineTunedDetector(model_path)
            self._model_available = True
        except (FileNotFoundError, ImportError) as e:
            logging.warning(f"Fine-tuned model unavailable: {e}. Using fallback classifier.")
            self._model_available = False

    def classify_players(
        self,
        frame: np.ndarray,
        detections: list,   # list of (bbox, tracker_id) or detection dicts
        tracker_ids: list | None = None,
    ) -> dict[int, str]:
        """
        Returns a dict mapping tracker_id → team label.

        Team labels: "left", "right", "referee", "goalkeeper_left", "goalkeeper_right"
        """
        if not self._model_available:
            if self.fallback:
                return self.fallback.classify_players(frame, detections, tracker_ids)
            return {}

        # Run model inference on the full frame
        model_detections = self.detector.detect(frame)

        # Match model detections to tracker IDs by bbox IoU
        results: dict[int, str] = {}
        if tracker_ids is not None:
            for track_id, det in zip(tracker_ids, detections):
                # Find the best-matching model detection for this tracked bbox
                best_match = _match_detection_by_iou(det, model_detections)
                if best_match and best_match["confidence"] >= self.min_conf_fallback:
                    label = best_match["team"] or best_match["role"]
                    results[track_id] = label
                elif self.fallback:
                    # Low confidence — defer to color fallback for this player
                    fallback_result = self.fallback.classify_players(
                        frame, [det], [track_id]
                    )
                    results.update(fallback_result)

        return results


def _match_detection_by_iou(
    tracked_det,
    model_detections: list[dict],
    iou_threshold: float = 0.4,
) -> dict | None:
    """
    Finds the model detection with the highest IoU overlap with tracked_det.
    Returns None if no match above iou_threshold.
    """
    if not model_detections:
        return None

    tracked_bbox = tracked_det["bbox"] if isinstance(tracked_det, dict) else tracked_det

    best_iou  = iou_threshold
    best_det  = None

    for model_det in model_detections:
        iou = _compute_iou(tracked_bbox, model_det["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_det = model_det

    return best_det


def _compute_iou(bbox_a: list, bbox_b: list) -> float:
    """Computes IoU between two bboxes in [x1,y1,x2,y2] format."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / (union_area + 1e-6)
```

---

## Wiring It Into the Main Pipeline

In your main inference entry point (wherever `TeamClassifier` is instantiated):

```python
from pathlib import Path
from team_classifier import TeamClassifier, ModelBasedTeamClassifier

FINETUNED_MODEL_PATH = Path("runs/finetune/soccernet_team_classifier/weights/best.pt")

# Legacy fallback (phase 1 improvements still apply)
legacy_classifier = TeamClassifier(...)

# Preferred: model-based classifier with fallback
if FINETUNED_MODEL_PATH.exists():
    classifier = ModelBasedTeamClassifier(
        model_path          = FINETUNED_MODEL_PATH,
        fallback_classifier = legacy_classifier,
        min_conf_fallback   = 0.35,
    )
    print("Using fine-tuned model for team classification.")
else:
    classifier = legacy_classifier
    print("Fine-tuned model not found. Using HSV color classifier.")
```

This means the system degrades gracefully — it uses the model if available,
and falls back to the improved HSV classifier from phase 1 if not.

---

## Evaluation Script

Create `scripts/evaluate_classifier.py` to compare the two approaches:

```python
"""
Evaluates team classification accuracy on a labeled video clip.

Usage:
    python scripts/evaluate_classifier.py \
        --clip data/SoccerNetGS/valid/SNGS-004 \
        --model runs/finetune/soccernet_team_classifier/weights/best.pt

Outputs:
    - Per-class accuracy
    - Confusion matrix
    - Team flip rate (switches per 1000 frames)
    - Coverage (fraction of frames with valid classification)
"""
```

Key metrics to compute and print:
1. **Per-class accuracy**: correct / total for each of the 5 classes
2. **Goalkeeper detection rate**: recall for classes 2 and 3 specifically
3. **Team flip rate**: number of tracker_id team changes per 1000 frames
4. **Coverage**: fraction of detected players that received a team label
5. **Confusion matrix**: 5×5 matrix showing misclassifications

---

## Deployment Checklist

Before using the fine-tuned model in production:

- [ ] `best.pt` exists at `runs/finetune/soccernet_team_classifier/weights/best.pt`
- [ ] `mAP50 > 0.75` on the validation set (check `results.csv`)
- [ ] Goalkeeper mAP > 0.55 (lower threshold due to class imbalance)
- [ ] Referee mAP > 0.80 (referees have distinctive appearance)
- [ ] Team flip rate on a 60-second test clip is < 5 flips per 1000 frames
- [ ] Inference runs at ≥ 25 FPS on target hardware (check with `detect_batch`)
- [ ] Fallback to HSV classifier works correctly when model returns low confidence
