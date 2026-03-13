import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

class TeamClassifier:
    def __init__(self, n_colors: int = 2):
        """Initialize the team classifier using KMeans.
        
        Args:
            n_colors: Expected number of teams (usually 2).
        """
        self.kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        self.team_colors = {}
        self.player_team_dict = {} # Maps tracker_id to team_id
        self.is_fitted = False
        
    def _get_player_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract the upper body of the player for color analysis.
        
        Args:
            frame: Video frame
            bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, bbox)
        # Add bounds checking
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        
        # We only want the upper half of the crop (usually the jersey)
        crop_h = crop.shape[0]
        upper_half = crop[:max(1, crop_h // 2), :]
        return upper_half
        
    def _extract_dominant_color(self, img_crop: np.ndarray) -> np.ndarray:
        """Get the dominant RGB color of an image crop using KMeans."""
        if img_crop.size == 0:
            return np.zeros(3)
            
        # Reshape to a list of pixels and convert to RGB
        pixels = img_crop.reshape((-1, 3))
        # Exclude black/dark background pixels (simple heuristic)
        mask = np.sum(pixels, axis=1) > 50
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) == 0:
            return np.mean(pixels, axis=0) # fallback
            
        # Use a quick 1-cluster kmeans to find the average dominant color instead of mean
        # Mean can be skewed by noisy backgrounds
        kmeans = KMeans(n_clusters=1, n_init=3, random_state=42)
        kmeans.fit(filtered_pixels)
        return kmeans.cluster_centers_[0]

    def fit_teams(self, frame: np.ndarray, detections: sv.Detections):
        """Fit the KMeans model to find the two main team colors from the frame.
        Should be called once on a frame where many players are visible.
        """
        player_detections = detections[detections.class_id == 0]
        if len(player_detections) < 10:
            return # Not enough players to confidently fit teams
            
        colors = []
        for bbox in player_detections.xyxy:
            crop = self._get_player_crop(frame, bbox)
            color = self._extract_dominant_color(crop)
            colors.append(color)
            
        self.kmeans.fit(colors)
        self.is_fitted = True
        
    def assign_teams(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Assign Team ID to each tracked player detection.
        
        Args:
            frame: Video frame
            detections: supervision Detections object
            
        Returns:
            np.ndarray: Array of team IDs corresponding to the detections.
            0 = Team 1, 1 = Team 2, -1 = Unassigned/Unknown
        """
        team_ids = np.full(len(detections), -1, dtype=int)
        
        if not self.is_fitted:
            self.fit_teams(frame, detections)
            if not self.is_fitted:
                return team_ids
                
        for i, (bbox, class_id, tracker_id) in enumerate(zip(detections.xyxy, detections.class_id, detections.tracker_id)):
            # Only classify players
            if class_id != 0 or tracker_id is None:
                continue
                
            # If we already know this player's team, use that to save compute
            if tracker_id in self.player_team_dict:
                team_ids[i] = self.player_team_dict[tracker_id]
                continue
                
            crop = self._get_player_crop(frame, bbox)
            color = self._extract_dominant_color(crop)
            
            # Predict team (0 or 1)
            team_id = self.kmeans.predict([color])[0]
            team_ids[i] = team_id
            self.player_team_dict[tracker_id] = team_id
            
        return team_ids
