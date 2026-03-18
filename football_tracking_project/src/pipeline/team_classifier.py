import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

class TeamClassifier:
    def __init__(self, n_colors: int = 2):
        self.kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        self.team_colors = {}
        self.player_team_dict = {}
        self.is_fitted = False
        
    def _get_player_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        
        crop_h = crop.shape[0]
        upper_half = crop[:max(1, crop_h // 2), :]
        return upper_half
        
    def _extract_dominant_color(self, img_crop: np.ndarray) -> np.ndarray:
        if img_crop.size == 0:
            return np.zeros(3)
            
        pixels = img_crop.reshape((-1, 3))
        mask = np.sum(pixels, axis=1) > 50
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) == 0:
            return np.mean(pixels, axis=0)
            
        kmeans = KMeans(n_clusters=1, n_init=3, random_state=42)
        kmeans.fit(filtered_pixels)
        return kmeans.cluster_centers_[0]

    def fit_teams(self, frame: np.ndarray, detections: sv.Detections):
        player_detections = detections[detections.class_id == 0]
        if len(player_detections) < 10:
            return
            
        colors = []
        for bbox in player_detections.xyxy:
            crop = self._get_player_crop(frame, bbox)
            color = self._extract_dominant_color(crop)
            colors.append(color)
            
        self.kmeans.fit(colors)
        self.is_fitted = True
        
    def assign_teams(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        team_ids = np.full(len(detections), -1, dtype=int)
        
        if not self.is_fitted:
            self.fit_teams(frame, detections)
            if not self.is_fitted:
                return team_ids
                
        for i, (bbox, class_id, tracker_id) in enumerate(zip(detections.xyxy, detections.class_id, detections.tracker_id)):
            if class_id != 0 or tracker_id is None:
                continue
                
            if tracker_id in self.player_team_dict:
                team_ids[i] = self.player_team_dict[tracker_id]
                continue
                
            crop = self._get_player_crop(frame, bbox)
            color = self._extract_dominant_color(crop)
            
            team_id = self.kmeans.predict([color])[0]
            team_ids[i] = team_id
            self.player_team_dict[tracker_id] = team_id
            
        return team_ids
