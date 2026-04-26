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
    - Automatic referee detection via low-saturation (black/grey) jersey filtering
    - Green grass pixel exclusion before colour sampling
    - Per-player voting history (last 15 frames) to prevent label flickering
    - Periodic re-fitting (every 150 frames) to adapt to camera angle changes
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

        # KMeans trained on jersey colours (teams only, referees excluded after fit)
        self.kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=42)
        self.is_fitted = False

        # Per player: rolling deque of team-id votes
        self.vote_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=history_len))
        # Permanently locked referees — once confirmed, never re-classified as a player
        self.locked_referees: set[int] = set()
        # Team lock per tracker id to resist short-term team swaps during occlusion.
        self.locked_teams: dict[int, int] = {}
        self.switch_votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(6, history_len // 2)))
        self.team_lock_min_votes = max(6, history_len // 2)
        self.frame_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_jersey_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Return the upper-torso crop (top 50 % of bounding box)."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        crop_h = crop.shape[0]
        return crop[:max(1, crop_h // 2), :]

    def _is_referee(self, hsv_crop: np.ndarray) -> bool:
        """
        Referees typically wear black, dark grey, or fluorescent yellow/orange.
        Detection via pixel-ratio heuristics:
          - Achromatic + dark kit ratio (black/grey referee shirts)
          - OR fluorescent yellow/lime ratio (common referee jerseys)
        """
        if hsv_crop.size == 0:
            return False
        h = hsv_crop[:, :, 0]
        s = hsv_crop[:, :, 1]
        v = hsv_crop[:, :, 2]

        achromatic_ratio = float(np.mean(s < 45))
        dark_ratio = float(np.mean(v < 80))

        # Fluorescent yellow/lime referee kit in HSV space
        fluorescent_ratio = float(np.mean((h >= 18) & (h <= 42) & (s > 90) & (v > 140)))

        if achromatic_ratio > 0.58 and dark_ratio > 0.25:
            return True
        if fluorescent_ratio > 0.35:
            return True
        return False

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_teams(self, frame: np.ndarray, detections: sv.Detections):
        """
        Fit 2-cluster KMeans on jersey colours of non-referee players.
        Requires ≥ 6 player detections; skips if fewer.
        """
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
            if self._is_referee(hsv_crop):
                continue  # exclude referees from team fitting
            domcol = self._extract_dominant_hsv(crop)
            colors.append(domcol)

        if len(colors) < 4:
            return  # not enough non-referee players

        self.kmeans.fit(colors)
        self.is_fitted = True

    def assign_teams(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Assign team IDs for every detection:
          - Ball (class 32) → UNKNOWN_ID
          - Referee         → REFEREE_ID (-2)
          - Player          → 0 or 1 (stable via voting history)

        Returns int array of shape (n_detections,).
        """
        team_ids = np.full(len(detections), self.UNKNOWN_ID, dtype=int)

        # Fit on first call (or periodically refit)
        if not self.is_fitted or (self.frame_count > 0 and self.frame_count % self.refit_interval == 0):
            self.fit_teams(frame, detections)

        self.frame_count += 1

        if not self.is_fitted:
            return team_ids

        for i, (bbox, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
            if class_id != 0:
                continue  # ball or other → leave as UNKNOWN_ID

            tracker_id = (
                int(detections.tracker_id[i])
                if detections.tracker_id is not None
                else None
            )

            # Permanently locked referees — skip re-classification entirely
            if tracker_id is not None and tracker_id in self.locked_referees:
                team_ids[i] = self.REFEREE_ID
                continue

            crop = self._get_jersey_crop(frame, bbox)
            if crop.size == 0:
                continue

            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Referee check
            if self._is_referee(hsv_crop):
                team_ids[i] = self.REFEREE_ID
                if tracker_id is not None:
                    self.vote_history[tracker_id].append(self.REFEREE_ID)
                    # Lock as referee once majority of history votes REFEREE_ID
                    history = list(self.vote_history[tracker_id])
                    if history.count(self.REFEREE_ID) > len(history) // 2:
                        self.locked_referees.add(tracker_id)
                        self.locked_teams.pop(tracker_id, None)
                        self.switch_votes.pop(tracker_id, None)
                continue

            # Team classification
            domcol = self._extract_dominant_hsv(crop)
            raw_team = int(self.kmeans.predict([domcol])[0])

            if tracker_id is not None:
                # Only append non-referee votes if not already a locked referee
                self.vote_history[tracker_id].append(raw_team)
                history = list(self.vote_history[tracker_id])
                # If more than half of votes are REFEREE_ID, lock and override
                if history.count(self.REFEREE_ID) > len(history) // 2:
                    self.locked_referees.add(tracker_id)
                    self.locked_teams.pop(tracker_id, None)
                    self.switch_votes.pop(tracker_id, None)
                    team_ids[i] = self.REFEREE_ID
                else:
                    # Majority vote among non-referee votes only
                    player_votes = [v for v in history if v != self.REFEREE_ID]
                    if player_votes:
                        voted_team = max(set(player_votes), key=player_votes.count)

                        # Lock team assignment once enough consistent votes are seen.
                        if tracker_id not in self.locked_teams:
                            if (
                                len(player_votes) >= self.team_lock_min_votes
                                and player_votes.count(voted_team) >= ceil(0.7 * len(player_votes))
                            ):
                                self.locked_teams[tracker_id] = voted_team

                        # For already-locked players, only allow switching after repeated opposite votes.
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
            else:
                team_ids[i] = raw_team

        return team_ids
