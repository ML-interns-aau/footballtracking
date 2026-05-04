import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from collections import defaultdict, deque
from math import ceil


class TeamClassifier:
    """
    Robust team classifier with:
    - HSV colour space for lighting-robust jersey colour extraction
    - Dynamic outlier detection for referees (no hardcoded colors!)
    - Green grass pixel exclusion before colour sampling
    - Per-player voting history to prevent label flickering
    - Periodic re-fitting to adapt to camera angle changes
    """

    REFEREE_ID = -2   # sentinel team id for referees
    UNKNOWN_ID = -1   # team not yet assigned

    # HSV thresholds for "green grass" exclusion
    _GREEN_LOWER = np.array([35, 40, 40])
    _GREEN_UPPER = np.array([85, 255, 255])

    def __init__(self, n_teams: int = 2, history_len: int = 15, refit_interval: int = 150):
        self.n_teams = n_teams
        self.history_len = history_len
        self.refit_interval = refit_interval

        self.kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=42)
        self.is_fitted = False
        self.outlier_threshold = 100.0  # Dynamic distance threshold for referees

        # Per player: rolling deque of team-id votes
        self.vote_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=history_len))
        self.locked_teams: dict[int, int] = {}
        self.switch_votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(6, history_len // 2)))
        self.team_lock_min_votes = max(6, history_len // 2)
        self.frame_count = 0

    def _get_jersey_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Return the upper-torso crop (top 50 % of bounding box)."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        crop_h = crop.shape[0]
        return crop[:max(1, crop_h // 2), :]

    def _extract_dominant_hsv(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Extract dominant jersey colour in HSV, excluding green grass pixels.
        Returns a 3-element HSV vector.
        """
        if bgr_crop.size == 0:
            return np.zeros(3)

        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Exclude green grass pixels
        mask_h = (pixels[:, 0] >= self._GREEN_LOWER[0]) & (pixels[:, 0] <= self._GREEN_UPPER[0])
        mask_s = (pixels[:, 1] >= self._GREEN_LOWER[1]) & (pixels[:, 1] <= self._GREEN_UPPER[1])
        mask_v = (pixels[:, 2] >= self._GREEN_LOWER[2]) & (pixels[:, 2] <= self._GREEN_UPPER[2])
        green_mask = mask_h & mask_s & mask_v
        non_green = pixels[~green_mask]

        if len(non_green) < 5:
            non_green = pixels  # fall back to all pixels

        # Exclude very dark pixels (shadows)
        bright_mask = non_green[:, 2] > 30
        bright = non_green[bright_mask]
        if len(bright) < 5:
            bright = non_green

        # Single-cluster KMeans → dominant colour
        km = KMeans(n_clusters=1, n_init=3, random_state=42)
        km.fit(bright)
        return km.cluster_centers_[0]

    def fit_teams(self, frame: np.ndarray, detections: sv.Detections):
        player_mask = detections.class_id == 0
        player_detections = detections[player_mask]
        if len(player_detections) < 6:
            return

        colors = []
        for bbox in player_detections.xyxy:
            crop = self._get_jersey_crop(frame, bbox)
            if crop.size == 0:
                continue
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            domcol = self._extract_dominant_hsv(crop)
            colors.append(domcol)

        if len(colors) < self.n_teams:
            return

        labels = self.kmeans.fit_predict(colors)
        
        # Calculate dynamic outlier threshold based on how spread out the teams are
        centers = self.kmeans.cluster_centers_
        dists = np.linalg.norm(colors - centers[labels], axis=1)
        self.outlier_threshold = np.percentile(dists, 85) * 2.0  # 85th percentile distance x 2 multiplier
        self.is_fitted = True

    def assign_teams(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        team_ids = np.full(len(detections), self.UNKNOWN_ID, dtype=int)

        if not self.is_fitted or (self.frame_count > 0 and self.frame_count % self.refit_interval == 0):
            self.fit_teams(frame, detections)

        self.frame_count += 1

        if not self.is_fitted:
            return team_ids

        for i, (bbox, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
            if class_id != 0:
                continue

            tracker_id = (int(detections.tracker_id[i]) if detections.tracker_id is not None else None)

            crop = self._get_jersey_crop(frame, bbox)
            if crop.size == 0:
                continue

            domcol = self._extract_dominant_hsv(crop)
            
            # 1. Measure distance to both team centers
            dists = np.linalg.norm(self.kmeans.cluster_centers_ - domcol, axis=1)
            min_dist = np.min(dists)
            
            # 2. Outlier check (Referee / Goalkeeper)
            if min_dist > self.outlier_threshold:
                raw_team = self.REFEREE_ID
            else:
                raw_team = int(np.argmin(dists))

            # 3. Voting and locking
            if tracker_id is not None:
                self.vote_history[tracker_id].append(raw_team)
                history = list(self.vote_history[tracker_id])
                
                voted_team = max(set(history), key=history.count)
                
                if voted_team == self.REFEREE_ID:
                    team_ids[i] = self.REFEREE_ID
                else:
                    if tracker_id not in self.locked_teams:
                        if len(history) >= self.team_lock_min_votes and history.count(voted_team) >= ceil(0.7 * len(history)):
                            self.locked_teams[tracker_id] = voted_team

                    if tracker_id in self.locked_teams:
                        current_team = self.locked_teams[tracker_id]
                        if voted_team != current_team:
                            self.switch_votes[tracker_id].append(voted_team)
                            votes = list(self.switch_votes[tracker_id])
                            if len(votes) >= 6 and votes.count(voted_team) >= 5:
                                self.locked_teams[tracker_id] = voted_team
                                self.switch_votes[tracker_id].clear()
                                team_ids[i] = voted_team
                            else:
                                team_ids[i] = current_team
                        else:
                            self.switch_votes[tracker_id].clear()
                            team_ids[i] = current_team
                    else:
                        team_ids[i] = voted_team
            else:
                team_ids[i] = raw_team

        return team_ids
