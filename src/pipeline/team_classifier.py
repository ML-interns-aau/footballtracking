import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from collections import defaultdict, deque
from math import ceil


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Gray-world color constancy normalization.
    Scales each BGR channel so the per-channel mean equals the global mean.
    Reduces white-balance and lighting drift across frames.
    """
    frame = frame.astype(np.float32)
    mean_per_channel = frame.mean(axis=(0, 1))
    global_mean = mean_per_channel.mean()
    scale = global_mean / (mean_per_channel + 1e-6)
    normalized = np.clip(frame * scale, 0, 255)
    return normalized.astype(np.uint8)


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

    MIN_PIXELS_FOR_KMEANS = 30

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

    def _get_jersey_crop(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Returns a tight torso crop: vertically from 15% to 55% of the bbox height,
        horizontally inset by 10% on each side. Returns `None` for degenerate boxes.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        top = y1 + int(0.15 * h)
        bottom = y1 + int(0.55 * h)
        left = x1 + int(0.10 * w)
        right = x2 - int(0.10 * w)

        # Guard against degenerate boxes
        if bottom <= top or right <= left:
            return None

        # Clip to frame bounds
        H, W = frame.shape[:2]
        top = max(0, min(H, top))
        bottom = max(0, min(H, bottom))
        left = max(0, min(W, left))
        right = max(0, min(W, right))

        if bottom <= top or right <= left:
            return None

        return frame[top:bottom, left:right]

    # (Old upper-half crop removed; using tighter torso crop above.)

    def _compute_grass_mask(self, hsv_pixels: np.ndarray) -> np.ndarray:
        """
        Returns a boolean mask of pixels that look like the current field grass.
        hsv_pixels: shape (N, 3) in OpenCV HSV scale (H:0-179, S:0-255, V:0-255)
        """
        if hsv_pixels.size == 0:
            return np.zeros(0, dtype=bool)

        h = hsv_pixels[:, 0]
        s = hsv_pixels[:, 1]
        v = hsv_pixels[:, 2]
        candidate_grass = (h >= 35) & (h <= 85) & (s > 40) & (v > 40)
        if candidate_grass.sum() < 10:
            # Fallback: very wide green range if no grass found
            return (h >= 30) & (h <= 90)

        grass_median = np.median(hsv_pixels[candidate_grass], axis=0)
        gh, gs, gv = grass_median
        mask = (
            (np.abs(h.astype(int) - int(gh)) < 15) &
            (np.abs(s.astype(int) - int(gs)) < 40) &
            (np.abs(v.astype(int) - int(gv)) < 40)
        )
        return mask

    def _extract_dominant_lab(self, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Extract dominant jersey colour in Lab space, excluding grass and dark pixels.
        Returns a 3-element Lab vector, or None when insufficient pixels.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return None

        # Compute HSV pixels for grass masking
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv.reshape(-1, 3).astype(np.int32)

        grass_mask = self._compute_grass_mask(hsv_pixels)

        # Convert to Lab and flatten
        lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2Lab)
        lab_pixels = lab.reshape(-1, 3).astype(np.float32)

        # Exclude grass pixels
        if grass_mask.shape[0] != lab_pixels.shape[0]:
            # Fallback: if shapes mismatch, don't mask
            valid = lab_pixels
        else:
            valid = lab_pixels[~grass_mask]

        # Exclude very dark pixels based on L channel
        if valid.size == 0:
            return None
        bright_mask = valid[:, 0] >= 20
        bright = valid[bright_mask]
        if bright.size == 0:
            return None

        if len(bright) < self.MIN_PIXELS_FOR_KMEANS:
            return None

        km = KMeans(n_clusters=1, n_init=3, random_state=42)
        km.fit(bright)
        return km.cluster_centers_[0]

    def fit_teams(self, frame: np.ndarray, detections: sv.Detections):
        # Normalize frame once per-frame
        frame = normalize_frame(frame)

        player_mask = detections.class_id == 0
        player_detections = detections[player_mask]
        if len(player_detections) < 6:
            return

        colors = []
        for bbox in player_detections.xyxy:
            crop = self._get_jersey_crop(frame, bbox)
            if crop is None:
                continue
            domcol = self._extract_dominant_lab(crop)
            if domcol is None:
                continue
            colors.append(domcol)

        if len(colors) < self.n_teams:
            return

        colors = np.array(colors)
        labels = self.kmeans.fit_predict(colors)

        # Calculate dynamic outlier threshold based on how spread out the teams are
        centers = self.kmeans.cluster_centers_
        dists = np.linalg.norm(colors - centers[labels], axis=1)
        self.outlier_threshold = np.percentile(dists, 85) * 2.0
        self.is_fitted = True

    def assign_teams(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        team_ids = np.full(len(detections), self.UNKNOWN_ID, dtype=int)

        # Normalize once per frame
        frame = normalize_frame(frame)

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
            if crop is None:
                continue

            domcol = self._extract_dominant_lab(crop)
            if domcol is None:
                continue
            
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
