import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

class HeatmapAnalyzer:
    def __init__(self, pitch_width: int = 105, pitch_height: int = 68, resolution: float = 1.0):
        """Initialize the heatmap analyzer.
        
        Args:
            pitch_width: Length of the pitch in meters
            pitch_height: Width of the pitch in meters
            resolution: Granularity of the heatmap cells in meters (e.g., 1.0 = 1x1m cells)
        """
        self.w = pitch_width
        self.h = pitch_height
        self.res = resolution
        
        # Grid dimensions
        self.grid_x = int(self.w / self.res)
        self.grid_y = int(self.h / self.res)
        
        # Store historical points per player and team
        # team_id -> [(x_m, y_m), ...]
        self.team_points = {0: [], 1: []}
        
        # player_id -> [(x_m, y_m), ...]
        self.player_points = {}

    def add_point(self, tracker_id: int, team_id: int, x_m: float, y_m: float):
        """Add a real-world position to the analyzer for generation later."""
        if x_m < 0 or x_m > self.w or y_m < 0 or y_m > self.h:
            return 
            
        point = (x_m, y_m)
        
        if tracker_id not in self.player_points:
            self.player_points[tracker_id] = []
            
        self.player_points[tracker_id].append(point)
        
        if team_id in self.team_points:
            self.team_points[team_id].append(point)

    def _generate_density_map(self, points: list[tuple[float, float]]) -> np.ndarray:
        """Create a 2D density array from a list of meter coordinates."""
        density = np.zeros((self.grid_y, self.grid_x), dtype=np.float32)
        
        for x, y in points:
            grid_px = int(x / self.res)
            grid_py = int(y / self.res)
            
            # bounds check
            grid_px = min(max(grid_px, 0), self.grid_x - 1)
            grid_py = min(max(grid_py, 0), self.grid_y - 1)
            
            density[grid_py, grid_px] += 1
            
        # Apply Gaussian blur to smooth the heatmap
        density = cv2.GaussianBlur(density, (9, 9), 0)
        
        # Normalize to 0-255
        if np.max(density) > 0:
            density = (density / np.max(density) * 255).astype(np.uint8)
        else:
            density = density.astype(np.uint8)
            
        return density

    def save_team_heatmap(self, team_id: int, output_path: str):
        """Generate and save a matplotlib heatmap image for a specific team."""
        pts = self.team_points.get(team_id, [])
        if len(pts) == 0:
            return
            
        density = self._generate_density_map(pts)
        
        plt.figure(figsize=(10, 6.5))
        # Draw a generic green pitch background
        plt.gca().set_facecolor('#2d402b') 
        
        # Overlay the heatmap
        plt.imshow(density, cmap='hot', interpolation='nearest', alpha=0.7, 
                   extent=[0, self.w, self.h, 0])
                   
        plt.title(f"Team {team_id} Activity Heatmap")
        plt.xlabel("Meters")
        plt.ylabel("Meters")
        plt.xlim(0, self.w)
        plt.ylim(self.h, 0) # y-axis inverted (0 is top left usually in CV, pitch drawing handles this)
        
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved team {team_id} heatmap to {output_path}")
