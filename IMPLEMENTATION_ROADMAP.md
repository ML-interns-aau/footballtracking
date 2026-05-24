# Practical Implementation Roadmap: Code-First Guide

This document maps the masterclass concepts directly to your codebase and provides concrete implementation steps.

---

## Quick Start: What to Do This Week

### Week 1 Actions

**Monday-Wednesday: Detection**
1. Export 100 representative frames from your test videos
2. Annotate 5 classes: player, goalkeeper, referee, ball, staff
3. Create `datasets/football/` structure with train/val/test splits
4. Start YOLOv11 fine-tuning on your hardware

**Thursday-Friday: Setup**
1. Create `src/improvements/` directory structure
2. Set up validation framework
3. Establish baselines on current system

### Week 2-4: Sprints

**Sprint 1: Detection** (Weeks 2-4)
- Build annotation pipeline
- Train YOLOv11m
- Benchmark vs. current YOLOv8m
- Deploy and test

**Sprint 2: Ball Tracking** (Weeks 5-7)
- Build ensemble architecture
- Implement fusion logic
- Integrate into pipeline
- Test on difficult sequences

**Sprint 3: Team Classification** (Weeks 8-10)
- Upgrade to multi-region color extraction
- Add CNN classifier
- Implement temporal locking
- Validate consistency

**Sprint 4: Validation Framework** (Weeks 11-12)
- Build ground truth extractor
- Implement all 4 validation metrics
- Create QA dashboard
- Deploy monitoring

---

## 1. Detection Improvement - Immediate Implementation

### Step 1.1: Dataset Preparation

```bash
# Create directory structure
mkdir -p datasets/football/{train,val,test}/{images,labels}
mkdir -p datasets/football/src_frames
mkdir -p models/yolov11
```

### Step 1.2: Frame Extraction & Annotation

Create `scripts/prepare_dataset.py`:

```python
import cv2
import json
from pathlib import Path

class FootballDatasetPreparer:
    """Extract frames from videos for annotation."""
    
    def extract_frames(self, video_path, output_dir, frame_interval=2):
        """Extract 1 frame every N frames."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Save frame with metadata
                frame_name = f"{Path(video_path).stem}_{frame_count:06d}.jpg"
                cv2.imwrite(str(output_dir / frame_name), frame)
                
                # Save metadata
                metadata = {
                    'video_source': str(video_path),
                    'frame_index': frame_count,
                    'timestamp_s': frame_count / fps,
                    'fps': fps,
                    'resolution': [frame.shape[1], frame.shape[0]]
                }
                with open(output_dir / f"{frame_name.replace('.jpg', '.json')}", 'w') as f:
                    json.dump(metadata, f)
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {frame_count // frame_interval} frames to {output_dir}")
    
    def split_dataset(self, frames_dir, train_ratio=0.7, val_ratio=0.15):
        """Split annotated frames into train/val/test."""
        import random
        from shutil import copy2
        
        frames_dir = Path(frames_dir)
        all_images = sorted(frames_dir.glob('*.jpg'))
        random.shuffle(all_images)
        
        n = len(all_images)
        train_split = int(n * train_ratio)
        val_split = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': all_images[:train_split],
            'val': all_images[train_split:val_split],
            'test': all_images[val_split:]
        }
        
        for split_name, images in splits.items():
            split_dir_img = Path(f"datasets/football/{split_name}/images")
            split_dir_lbl = Path(f"datasets/football/{split_name}/labels")
            split_dir_img.mkdir(parents=True, exist_ok=True)
            split_dir_lbl.mkdir(parents=True, exist_ok=True)
            
            for img in images:
                copy2(img, split_dir_img / img.name)
        
        print(f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

# Usage
if __name__ == '__main__':
    preparer = FootballDatasetPreparer()
    # Extract frames from your videos
    preparer.extract_frames('data/raw/sample_match.mp4', 'datasets/football/src_frames', frame_interval=50)
    # After manual annotation in Roboflow/LabelImg, split dataset
    preparer.split_dataset('datasets/football/annotated_frames')
```

### Step 1.3: Dataset Configuration

Create `datasets/football/dataset.yaml`:

```yaml
path: /path/to/footballtracking/datasets/football
train: train/images
val: val/images
test: test/images

nc: 5  # number of classes
names: ['player', 'goalkeeper', 'referee', 'ball', 'staff']

# Class indices
player: 0
goalkeeper: 1
referee: 2
ball: 3
staff: 4
```

### Step 1.4: Fine-tune YOLOv11

Create `src/improvements/train_detection.py`:

```python
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class FootballYOLOv11Trainer:
    """Fine-tune YOLOv11 on football dataset."""
    
    def __init__(self, model_size='m'):
        # Options: n, s, m, l, x
        self.model = YOLO(f'yolov11{model_size}.pt')
        self.model_size = model_size
    
    def train(self, data_yaml, epochs=100, batch_size=32, device='0'):
        """Fine-tune on your football dataset."""
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=960,              # Image size
            batch=batch_size,
            device=device,
            patience=20,            # Early stopping
            
            # Optimizer
            optimizer='SGD',
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.9,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            
            # Callbacks & logging
            save=True,
            save_period=10,
            workers=8,
            verbose=True,
        )
        
        logger.info(f"Training complete. Results: {results}")
        return results
    
    def evaluate_test_set(self, data_yaml, conf=0.35, iou=0.6, device='0'):
        """Evaluate on test set with detailed metrics."""
        metrics = self.model.val(
            data=data_yaml,
            conf=conf,
            iou=iou,
            device=device,
            split='test',  # Evaluate on test set only
        )
        
        logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def export_for_production(self, export_format='onnx'):
        """Export model for production deployment."""
        export_path = self.model.export(
            format=export_format,
            half=True,              # FP16 quantization
            opset=13,
            imgsz=960,
        )
        logger.info(f"Model exported to: {export_path}")
        return export_path

# Usage
if __name__ == '__main__':
    trainer = FootballYOLOv11Trainer(model_size='m')
    
    # Train
    trainer.train(
        data_yaml='datasets/football/dataset.yaml',
        epochs=100,
        batch_size=32,
        device='0'
    )
    
    # Evaluate
    trainer.evaluate_test_set('datasets/football/dataset.yaml')
    
    # Export
    trainer.export_for_production('onnx')
```

### Step 1.5: Update Detector in Pipeline

Modify `src/pipeline/detector.py`:

```python
import supervision as sv
from ultralytics import YOLO
import numpy as np
import pathlib
import platform
from pathlib import Path

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

class FootballDetectorV2:
    """
    Production detector with YOLOv11 and context-aware filtering.
    Replaces the current YOLOv8m detector.
    """
    
    CLASS_NAMES = {
        0: 'player',
        1: 'goalkeeper',
        2: 'referee',
        3: 'ball',
        4: 'staff'
    }
    
    def __init__(
        self,
        model_path: str = "models/yolov11/best.pt",
        conf: float = 0.35,
        iou: float = 0.35,
        device: str = "cuda:0",
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.CLASS_NAMES_DICT = self.model.model.names
        self.conf = conf
        self.iou = iou
        self.device = device
        
        # Tracking state for context-aware filtering
        self.frames_since_ball_detection = 0
        self.last_player_count = 0
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run inference with all classes."""
        results = self.model(
            frame,
            classes=[0, 1, 2, 3, 4],  # All football classes
            conf=self.conf,
            iou=self.iou,
            imgsz=960,
            agnostic_nms=True,
            verbose=False,
            device=self.device,
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        
        # Apply post-processing
        detections = self._post_process(detections)
        
        return detections
    
    def _post_process(self, detections: sv.Detections) -> sv.Detections:
        """Post-processing to fix common detection errors."""
        
        # 1. Ensure only 1 ball (keep highest confidence)
        ball_mask = detections.class_id == 3
        if ball_mask.sum() > 1:
            ball_dets = detections[ball_mask]
            best_ball_idx = np.argmax(ball_dets.confidence)
            
            # Keep best ball + non-ball detections
            non_ball = detections[~ball_mask]
            best_ball = sv.Detections(
                xyxy=ball_dets.xyxy[best_ball_idx:best_ball_idx+1],
                confidence=ball_dets.confidence[best_ball_idx:best_ball_idx+1],
                class_id=ball_dets.class_id[best_ball_idx:best_ball_idx+1],
            )
            
            # Combine
            if len(non_ball) > 0:
                detections = sv.Detections(
                    xyxy=np.vstack([non_ball.xyxy, best_ball.xyxy]),
                    confidence=np.hstack([non_ball.confidence, best_ball.confidence]),
                    class_id=np.hstack([non_ball.class_id, best_ball.class_id]),
                )
            else:
                detections = best_ball
            
            self.frames_since_ball_detection = 0
        else:
            self.frames_since_ball_detection += 1
        
        # 2. Filter low-confidence staff (< 0.5 staff ok)
        staff_mask = detections.class_id == 4
        if staff_mask.sum() > 2:
            staff_dets = detections[staff_mask]
            top_staff_idx = np.argsort(-staff_dets.confidence)[:2]
            
            keep_staff = sv.Detections(
                xyxy=staff_dets.xyxy[top_staff_idx],
                confidence=staff_dets.confidence[top_staff_idx],
                class_id=staff_dets.class_id[top_staff_idx],
            )
            other_dets = detections[~staff_mask]
            
            if len(other_dets) > 0:
                detections = sv.Detections(
                    xyxy=np.vstack([other_dets.xyxy, keep_staff.xyxy]),
                    confidence=np.hstack([other_dets.confidence, keep_staff.confidence]),
                    class_id=np.hstack([other_dets.class_id, keep_staff.class_id]),
                )
            else:
                detections = keep_staff
        
        # 3. Sanity check: player count
        player_mask = (detections.class_id == 0) | (detections.class_id == 1)
        player_count = player_mask.sum()
        
        if player_count > 30:
            # Too many "people" - likely false positives
            # Filter to keep only high-confidence players
            player_dets = detections[player_mask]
            top_players_idx = np.argsort(-player_dets.confidence)[:25]
            
            keep_players = sv.Detections(
                xyxy=player_dets.xyxy[top_players_idx],
                confidence=player_dets.confidence[top_players_idx],
                class_id=player_dets.class_id[top_players_idx],
            )
            other_dets = detections[~player_mask]
            
            if len(other_dets) > 0:
                detections = sv.Detections(
                    xyxy=np.vstack([keep_players.xyxy, other_dets.xyxy]),
                    confidence=np.hstack([keep_players.confidence, other_dets.confidence]),
                    class_id=np.hstack([keep_players.class_id, other_dets.class_id]),
                )
            else:
                detections = keep_players
        
        self.last_player_count = player_count
        return detections
    
    def detect_players(self, frame: np.ndarray) -> sv.Detections:
        """Return only player + goalkeeper detections."""
        detections = self.detect(frame)
        return detections[(detections.class_id == 0) | (detections.class_id == 1)]
    
    def detect_ball(self, frame: np.ndarray) -> sv.Detections:
        """Return only ball detections."""
        detections = self.detect(frame)
        return detections[detections.class_id == 3]
    
    def detect_referees(self, frame: np.ndarray) -> sv.Detections:
        """Return only referee detections."""
        detections = self.detect(frame)
        return detections[detections.class_id == 2]
```

---

## 2. Ball Tracking - Multi-Model Ensemble

### Step 2.1: Build Ensemble Architecture

Create `src/improvements/ball_tracker_ensemble.py`:

```python
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass

@dataclass
class BallTrackingResult:
    position: np.ndarray
    confidence: float
    is_valid: bool
    tracker_source: str  # 'detection', 'kalman', 'optical_flow', 'ensemble'

class EnhancedKalmanBallTracker:
    """Kalman filter with adaptive noise."""
    
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self._initialised = False
        self._missed_count = 0
        self.last_position = None
    
    def predict(self, detection_confidence=0.5):
        """Predict with adaptive noise."""
        # Increase process noise if low detection confidence
        adaptive_q = self.kf.processNoiseCov.copy()
        if detection_confidence < 0.5:
            adaptive_q *= 5
        self.kf.processNoiseCov = adaptive_q
        
        if not self._initialised:
            return None, 0.0
        
        prediction = self.kf.predict()
        return prediction[:2].flatten(), 0.6
    
    def update(self, detection_pos, detection_confidence=0.9):
        """Update with detection."""
        if detection_confidence < 0.3:
            return
        
        if not self._initialised:
            self.kf.statePost = np.array([
                [float(detection_pos[0])],
                [float(detection_pos[1])],
                [0],
                [0]
            ], dtype=np.float32)
            self._initialised = True
            return
        
        measurement = np.array([
            [float(detection_pos[0])],
            [float(detection_pos[1])]
        ], dtype=np.float32)
        
        self.kf.correct(measurement)
        self.last_position = detection_pos
    
    def reset(self):
        """Reset tracker."""
        self._initialised = False
        self._missed_count = 0
        self.last_position = None

class OpticalFlowBallTracker:
    """Optical flow-based tracking."""
    
    def __init__(self, initial_pos):
        self.last_pos = initial_pos
        self.last_frame_gray = None
        self.max_distance = 80  # pixels per frame
    
    def track(self, frame, prev_frame):
        """Track ball using optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.last_frame_gray is None:
            self.last_frame_gray = gray
            return self.last_pos, 0.3
        
        # Dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.last_frame_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, n8=True, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        x, y = int(self.last_pos[0]), int(self.last_pos[1])
        h, w = gray.shape
        
        if 0 <= x < w and 0 <= y < h:
            flow_vec = flow[y, x]
            estimated_pos = self.last_pos + flow_vec
            
            # Confidence based on flow magnitude
            magnitude = np.linalg.norm(flow_vec)
            confidence = min(0.7, magnitude / 50)
            
            # Check if movement is reasonable
            if magnitude <= self.max_distance:
                self.last_pos = estimated_pos
            else:
                confidence *= 0.5  # Suspicious movement
        else:
            confidence = 0.1
        
        self.last_frame_gray = gray
        return self.last_pos, confidence

class BallPhysicsModel:
    """Validate ball movements against physics."""
    
    PITCH_WIDTH_M = 68
    PITCH_HEIGHT_M = 105
    MAX_BALL_SPEED_MPS = 40
    MAX_SPEED_PX_FRAME = 150  # ~60 m/s in pixels
    
    def __init__(self, pitch_mapper=None):
        self.pitch_mapper = pitch_mapper
    
    def is_plausible(self, pos1_px, pos2_px, time_delta_s=0.04):
        """Check if trajectory is physically plausible."""
        
        # Check 1: Unrealistic pixel jump
        distance_px = np.linalg.norm(pos2_px - pos1_px)
        if distance_px > self.MAX_SPEED_PX_FRAME:
            return False, 0.3
        
        # Check 2: Within pitch (if we can map)
        if self.pitch_mapper:
            try:
                pos2_pitch = self.pitch_mapper.pixel_to_pitch(pos2_px)
                if not self._is_within_pitch(pos2_pitch):
                    return False, 0.5
            except:
                pass
        
        return True, 1.0
    
    def _is_within_pitch(self, pitch_pos):
        """Check if position is on valid pitch."""
        x, y = pitch_pos
        return -5 <= x <= self.PITCH_WIDTH_M + 5 and \
               -5 <= y <= self.PITCH_HEIGHT_M + 5

class BallTrackerEnsemble:
    """Ensemble of ball trackers with fusion."""
    
    def __init__(self, pitch_mapper=None):
        self.kalman_tracker = EnhancedKalmanBallTracker()
        self.optical_flow_tracker = None
        self.physics_model = BallPhysicsModel(pitch_mapper)
        
        self.last_pos = None
        self.trail = deque(maxlen=30)
        self.occluded_frames = 0
        self.max_occlusion_frames = 30
    
    def update(self, frame, prev_frame, detection=None, detection_conf=0.5):
        """
        Update with multiframe detection.
        Returns BallTrackingResult.
        """
        
        # 1. Kalman prediction
        if detection is not None and detection_conf > 0.3:
            # Update Kalman with detection
            self.kalman_tracker.update(detection, detection_conf)
            self.last_pos = detection
            self.occluded_frames = 0
            self.trail.append(detection)
            
            return BallTrackingResult(
                position=detection,
                confidence=detection_conf,
                is_valid=True,
                tracker_source='detection'
            )
        
        # 2. Kalman prediction (no detection)
        kalman_pred, kalman_conf = self.kalman_tracker.predict(detection_conf)
        
        # 3. Optical flow
        if self.last_pos is not None and self.optical_flow_tracker is None:
            self.optical_flow_tracker = OpticalFlowBallTracker(self.last_pos)
        
        of_pred, of_conf = (None, 0.0)
        if self.optical_flow_tracker:
            of_pred, of_conf = self.optical_flow_tracker.track(frame, prev_frame)
        
        # 4. Ensemble vote
        predictions = {}
        confidences = {}
        
        if kalman_pred is not None:
            predictions['kalman'] = kalman_pred
            confidences['kalman'] = kalman_conf * 0.5
        
        if of_pred is not None:
            predictions['optical_flow'] = of_pred
            confidences['optical_flow'] = of_conf * 0.3
        
        if not predictions:
            # No predictions available
            self.occluded_frames += 1
            if self.occluded_frames > self.max_occlusion_frames:
                self.kalman_tracker.reset()
            
            return BallTrackingResult(
                position=self.last_pos,
                confidence=0.0,
                is_valid=False,
                tracker_source='none'
            )
        
        # Weighted average
        total_conf = sum(confidences.values())
        final_pos = np.average(
            [p for p in predictions.values()],
            axis=0,
            weights=[confidences[k] for k in predictions.keys()]
        )
        
        # Physics check
        if self.last_pos is not None:
            is_plausible, physics_conf = self.physics_model.is_plausible(
                self.last_pos, final_pos
            )
            if not is_plausible:
                # Fallback to last known position
                final_pos = self.last_pos
                total_conf *= physics_conf
        
        self.last_pos = final_pos
        self.trail.append(final_pos)
        self.occluded_frames += 1
        
        return BallTrackingResult(
            position=final_pos,
            confidence=min(1.0, total_conf),
            is_valid=True,
            tracker_source='ensemble'
        )
```

### Step 2.2: Integrate into Pipeline

Modify `src/pipeline/data_exporter.py` to use ensemble tracker. Replace ball tracking initialization:

```python
from src.improvements.ball_tracker_ensemble import BallTrackerEnsemble

class DataExporter:
    def __init__(self, ...):
        # Replace: self.ball_tracker = BallTracker()
        self.ball_tracker = BallTrackerEnsemble(pitch_mapper=self.pitch_mapper)
        ...
    
    def process_frame(self, frame_idx, frame, detections, prev_frame=None):
        """Process single frame with ensemble ball tracking."""
        
        # ... existing code ...
        
        # Ball tracking (now ensemble)
        ball_detections = detections.filter(lambda x: x.class_id == 3)
        
        if len(ball_detections) > 0:
            ball_conf = ball_detections[0].confidence
            ball_pos = ball_detections[0].center
        else:
            ball_conf = 0.0
            ball_pos = None
        
        # Update ensemble tracker
        ball_result = self.ball_tracker.update(
            frame, prev_frame, detection=ball_pos, detection_conf=ball_conf
        )
        
        if ball_result.is_valid:
            ball_position_px = ball_result.position
            ball_confidence = ball_result.confidence
        else:
            ball_position_px = None
            ball_confidence = 0.0
        
        # ... rest of processing ...
```

---

## 3. Team Classification - Upgrade

### Step 3.1: Enhanced Color Extraction

Create `src/improvements/team_classifier_v2.py`:

```python
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import deque, defaultdict

class EnhancedTeamClassifier:
    """
    Improved team classifier with:
    - Multi-region jersey sampling
    - Histogram-based color matching
    - Temporal consistency
    - Goalkeeper detection
    """
    
    REFEREE_ID = -2
    UNKNOWN_ID = -1
    
    _GREEN_LOWER = np.array([35, 40, 40])
    _GREEN_UPPER = np.array([85, 255, 255])
    
    JERSEY_REGIONS = {
        'torso': (0.3, 0.7),
        'arm': (0.2, 0.5),
        'shoulder': (0.1, 0.4),
    }
    
    def __init__(self, n_teams=2, history_len=20):
        self.n_teams = n_teams
        self.history_len = history_len
        
        self.kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=42)
        self.team_histograms = {}
        self.is_fitted = False
        
        # Temporal smoothing
        self.vote_history = defaultdict(lambda: deque(maxlen=history_len))
        self.locked_teams = {}
        self.lock_frame_count = {}
    
    def _extract_multi_region_hsv(self, frame, bbox):
        """Extract HSV from multiple body regions."""
        samples = []
        weights = []
        
        x1, y1, x2, y2 = map(int, bbox)
        h_total = y2 - y1
        w_total = x2 - x1
        
        for region_name, (y_start, y_end) in self.JERSEY_REGIONS.items():
            y_start_px = int(y1 + h_total * y_start)
            y_end_px = int(y1 + h_total * y_end)
            crop = frame[y_start_px:y_end_px, x1:x2]
            
            if crop.size == 0:
                continue
            
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            pixels = hsv.reshape(-1, 3).astype(np.float32)
            
            # Filter green grass
            mask_h = (pixels[:, 0] >= self._GREEN_LOWER[0]) & (pixels[:, 0] <= self._GREEN_UPPER[0])
            mask_s = (pixels[:, 1] >= self._GREEN_LOWER[1]) & (pixels[:, 1] <= self._GREEN_UPPER[1])
            mask_v = (pixels[:, 2] >= self._GREEN_LOWER[2]) & (pixels[:, 2] <= self._GREEN_UPPER[2])
            green_mask = mask_h & mask_s & mask_v
            non_green = pixels[~green_mask]
            
            if len(non_green) < 10:
                non_green = pixels
            
            # Filter dark pixels (shadows)
            bright_mask = non_green[:, 2] > 30
            bright = non_green[bright_mask]
            if len(bright) < 10:
                bright = non_green
            
            if len(bright) > 0:
                dominant_hsv = bright.mean(axis=0)
                saturation = dominant_hsv[1]
                samples.append(dominant_hsv)
                weights.append(saturation)
        
        if not samples:
            return np.zeros(3), 0.0
        
        weights = np.array(weights)
        weights /= weights.sum()
        result = np.average(samples, axis=0, weights=weights)
        
        return result, weights.max()
    
    def fit(self, player_hsv_samples, player_team_labels):
        """Fit KMeans on training data."""
        self.kmeans.fit(player_hsv_samples)
        
        # Build histograms for each team
        for team_id in range(self.n_teams):
            team_samples = player_hsv_samples[player_team_labels == team_id]
            hist = cv2.calcHist(
                [team_samples.astype(np.uint8)],
                [0, 1], None, [8, 8], [0, 180, 0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            self.team_histograms[team_id] = hist
        
        self.is_fitted = True
    
    def classify_player(self, player_hsv, player_id, nearby_players_hsv=None):
        """Classify player with consistency checking."""
        
        # Method 1: Histogram matching
        team_scores = {}
        for team_id, hist in self.team_histograms.items():
            player_hist = cv2.calcHist(
                [player_hsv.astype(np.uint8).reshape(1, -1)],
                [0, 1], None, [8, 8], [0, 180, 0, 256]
            )
            player_hist = cv2.normalize(player_hist, player_hist).flatten()
            score = cv2.compareHist(player_hist, hist, cv2.HISTCMP_CORREL)
            team_scores[team_id] = score
        
        suggested_team = max(team_scores, key=team_scores.get)
        confidence = team_scores[suggested_team]
        
        # Method 2: Check for goalkeeper (distinctly different color)
        is_gk, gk_conf = self._check_goalkeeper(
            player_hsv, suggested_team, nearby_players_hsv
        )
        if is_gk:
            suggested_team = self.REFEREE_ID  # Mark as special (goalkeeper)
            confidence = gk_conf
        
        # Method 3: Apply temporal consistency
        final_team, final_conf = self._apply_temporal_consistency(
            player_id, suggested_team, confidence
        )
        
        return final_team, final_conf
    
    def _check_goalkeeper(self, player_hsv, team_id, nearby_players_hsv=None):
        """Detect goalkeeper by color distinctiveness."""
        if nearby_players_hsv is None or len(nearby_players_hsv) == 0:
            return False, 0.2
        
        # Calculate color distance to teammates
        distances = [
            np.linalg.norm(player_hsv - other) for other in nearby_players_hsv
        ]
        avg_distance = np.mean(distances)
        
        # GK has different jersey color (distance > threshold)
        if avg_distance > 40:
            return True, 0.85
        
        return False, 0.3
    
    def _apply_temporal_consistency(self, player_id, suggested_team, confidence):
        """Apply temporal smoothing to reduce flickering."""
        
        history = self.vote_history[player_id]
        history.append(suggested_team)
        
        # Check if locked
        if player_id in self.locked_teams:
            locked_team = self.locked_teams[player_id]
            
            if suggested_team == locked_team:
                self.lock_frame_count[player_id] = self.history_len
            else:
                # Try to switch if very confident
                if confidence > 0.85:
                    self.lock_frame_count[player_id] -= 1
                    if self.lock_frame_count[player_id] <= 0:
                        del self.locked_teams[player_id]
                        return suggested_team, confidence
                else:
                    return locked_team, 0.7
        else:
            # Lock if confident enough
            if confidence > 0.75:
                self.locked_teams[player_id] = suggested_team
                self.lock_frame_count[player_id] = self.history_len
        
        return suggested_team, confidence
```

### Step 3.2: Update Main Pipeline

Update `main.py` to use new classifier:

```python
from src.improvements.team_classifier_v2 import EnhancedTeamClassifier

# In main processing loop
team_classifier = EnhancedTeamClassifier(n_teams=2, history_len=20)

# First pass: collect training data
# Second pass: fit and classify
for frame_idx, frame in enumerate(frames):
    # Detection
    detections = detector.detect(frame)
    players = detections[detections.class_id.isin([0, 1])]  # Players + GKs
    
    # Extract HSV for each player
    player_hsv_list = []
    for player_det in players:
        hsv, conf = team_classifier._extract_multi_region_hsv(frame, player_det.xyxy[0])
        player_hsv_list.append(hsv)
    
    # Classify teams
    for i, (player_det, player_hsv) in enumerate(zip(players, player_hsv_list)):
        team_id, confidence = team_classifier.classify_player(
            player_hsv, player_id=i
        )
        # Use team_id in tracking
```

---

## 4. Data Validation Framework

### Step 4.1: Create Validation System

Create `src/improvements/validation_framework.py`:

```python
import json
import numpy as np
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

class AnalyticsValidator:
    """Comprehensive analytics validation."""
    
    def __init__(self, generated_json_path, ground_truth_json_path=None):
        with open(generated_json_path, 'r') as f:
            self.generated = json.load(f)
        
        self.ground_truth = None
        if ground_truth_json_path:
            with open(ground_truth_json_path, 'r') as f:
                self.ground_truth = json.load(f)
        
        self.validation_report = {
            'video_id': self.generated.get('video_id'),
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
    
    def validate_possession_accuracy(self):
        """Validate possession changes."""
        if not self.ground_truth:
            return None
        
        generated_possessions = self.generated.get('possession_timeline', [])
        gt_possessions = self.ground_truth.get('possession_events', [])
        
        matches = 0
        for gt_event in gt_possessions:
            for gen_event in generated_possessions:
                if abs(gen_event['frame'] - gt_event['frame']) < 2:
                    if gen_event['team'] == gt_event['team']:
                        matches += 1
                    break
        
        accuracy = matches / len(gt_possessions) if gt_possessions else 0.0
        
        return {
            'accuracy': float(accuracy),
            'matches': matches,
            'total_gt': len(gt_possessions),
            'metric_name': 'Possession_Accuracy'
        }
    
    def validate_player_team_consistency(self):
        """Check for team switching."""
        inconsistencies = []
        
        for tracking_frame in self.generated.get('tracking', []):
            player_teams = {}
            for player in tracking_frame.get('players', []):
                pid = player['track_id']
                team = player['team']
                
                if pid not in player_teams:
                    player_teams[pid] = set()
                player_teams[pid].add(team)
            
            for pid, teams in player_teams.items():
                if len(teams) > 1:
                    inconsistencies.append({
                        'player_id': pid,
                        'frame': tracking_frame['frame'],
                        'teams': list(teams)
                    })
        
        consistency = 1.0 if not inconsistencies else 0.8
        
        return {
            'consistency_score': float(consistency),
            'inconsistencies': inconsistencies[:10],
            'metric_name': 'Player_Team_Consistency'
        }
    
    def validate_ball_continuity(self):
        """Check for unrealistic ball jumps."""
        ball_positions = []
        
        for frame_data in self.generated.get('tracking', []):
            if 'ball' in frame_data and frame_data['ball']:
                ball_positions.append({
                    'frame': frame_data['frame'],
                    'pos': frame_data['ball']['position']
                })
        
        jumps = []
        for i in range(1, len(ball_positions)):
            prev_pos = np.array(ball_positions[i-1]['pos'])
            curr_pos = np.array(ball_positions[i]['pos'])
            distance = np.linalg.norm(curr_pos - prev_pos)
            
            if distance > 150:
                jumps.append({
                    'frame': ball_positions[i]['frame'],
                    'distance': float(distance)
                })
        
        continuity = 1.0 - (len(jumps) / len(ball_positions)) if ball_positions else 0.0
        
        return {
            'continuity_score': float(continuity),
            'suspicious_jumps': len(jumps),
            'metric_name': 'Ball_Continuity'
        }
    
    def validate_speed_realism(self):
        """Check for physically unrealistic speeds."""
        unrealistic_speeds = []
        
        for frame_data in self.generated.get('tracking', []):
            for player in frame_data.get('players', []):
                speed = player.get('speed_kmh', 0)
                if speed > 45:
                    unrealistic_speeds.append({
                        'player_id': player['track_id'],
                        'frame': frame_data['frame'],
                        'speed': speed
                    })
        
        realism = 1.0 if not unrealistic_speeds else 0.7
        
        return {
            'realism_score': float(realism),
            'flags': len(unrealistic_speeds),
            'metric_name': 'Speed_Realism'
        }
    
    def generate_report(self):
        """Generate full validation report."""
        report = self.validation_report.copy()
        
        metrics = {
            'possession': self.validate_possession_accuracy(),
            'player_consistency': self.validate_player_team_consistency(),
            'ball_continuity': self.validate_ball_continuity(),
            'speed': self.validate_speed_realism(),
        }
        
        report['metrics'] = {k: v for k, v in metrics.items() if v}
        
        # Overall score
        valid_scores = [v['accuracy'] if 'accuracy' in v else v.get('consistency_score', v.get('continuity_score', v.get('realism_score', 0))) 
                       for v in report['metrics'].values()]
        overall = np.mean(valid_scores) if valid_scores else 0.0
        
        report['overall_quality_score'] = float(overall)
        report['quality_tier'] = 'PRODUCTION_READY' if overall >= 0.95 else \
                                  'GOOD' if overall >= 0.85 else \
                                  'ACCEPTABLE' if overall >= 0.70 else 'NEEDS_IMPROVEMENT'
        
        return report
    
    def save_report(self, output_path):
        """Save validation report."""
        report = self.generate_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        return report
```

### Step 4.2: Integrate into Pipeline

Add to `main.py`:

```python
from src.improvements.validation_framework import AnalyticsValidator

# After pipeline completes
if __name__ == '__main__':
    # ... run pipeline ...
    
    # Validate output
    validator = AnalyticsValidator(
        'results/analytics.json',
        ground_truth_json_path=None  # Add when available
    )
    report = validator.save_report('results/validation_report.json')
    
    print(f"Quality Score: {report['overall_quality_score']*100:.1f}%")
    print(f"Tier: {report['quality_tier']}")
    
    if report['quality_tier'] not in ['PRODUCTION_READY', 'GOOD']:
        print("⚠️ WARNING: Output quality below acceptable threshold")
        print("Metrics:", report['metrics'])
```

---

## Testing & Iteration Strategy

### Week-by-Week Test Plan

```
Week 1-2: Detection
  - Train YOLOv11 on annotated dataset
  - Test on 100 unseen frames
  - Compare mAP vs. current YOLOv8m
  - Benchmark speed (fps, memory)

Week 3-4: Ball Tracking
  - Test ensemble on 10 difficult sequences
  - Measure occlusion handling
  - Compare drift vs. original Kalman
  - Validate physics filtering

Week 5-6: Team Classification
  - Test consistency on 10 full matches
  - Measure team-switching frequency
  - Validate goalkeeper detection
  - Benchmark vs. original HSV approach

Week 7: Integration & Validation
  - Run end-to-end on 5 test videos
  - Generate validation reports
  - Compare all metrics
  - Document improvements
```

### Success Criteria

```
Detection: mAP50 > 95% (players), > 96% (ball)
Ball Tracking: > 98% continuity, < 10 jumps/hour
Team Class: > 99% consistency within matches
Validation: All 4 metrics operational, < 5% false anomalies
Speed: < 8 min per 90-min match (GPU)
```

---

## Next Steps

1. **This Week**: Start data collection & annotation
2. **Week 2**: Train YOLOv11, compare with YOLOv8m
3. **Week 3**: Build ball tracker ensemble
4. **Week 4**: Upgrade team classifier
5. **Week 5**: Build validation framework
6. **Week 6**: Integration testing
7. **Week 7**: Production readiness review

Good luck! Start with detection—it's the foundation for everything else.

