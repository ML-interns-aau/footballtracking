"""Player Summary CSV Builder - Aggregated per-player statistics.

This module builds player_summary.csv with aggregated statistics for each
player including speed metrics, distance covered, and possession data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import pandas as pd
from pathlib import Path

from src.pipeline.output_schema import (
    OutputFiles,
    PlayerSummaryCSVColumns,
    write_csv_headers,
)


@dataclass
class PlayerStats:
    """Statistics for a single player."""
    object_id: int
    team_id: int
    class_id: int
    total_frames: int = 0
    top_speed_km_h: float = 0.0
    avg_speed_km_h: float = 0.0
    total_distance_m: float = 0.0
    possession_frames: int = 0
    role: str = "player"
    team: str = "Unknown"
    
    @property
    def possession_pct(self) -> float:
        """Calculate possession percentage."""
        if self.total_frames == 0:
            return 0.0
        return (self.possession_frames / self.total_frames) * 100.0
    
    def update_speed(self, speed_km_h: float) -> None:
        """Update speed statistics."""
        self.total_frames += 1
        self.top_speed_km_h = max(self.top_speed_km_h, speed_km_h)
        # Rolling average calculation
        self.avg_speed_km_h = ((self.avg_speed_km_h * (self.total_frames - 1)) + speed_km_h) / self.total_frames
    
    def update_distance(self, distance_m: float) -> None:
        """Update total distance covered."""
        self.total_distance_m += distance_m
    
    def update_possession(self, has_possession: bool) -> None:
        """Update possession statistics."""
        if has_possession:
            self.possession_frames += 1


class PlayerSummaryCSVBuilder:
    """Builds player_summary.csv with aggregated player statistics."""
    
    def __init__(self):
        self.player_stats: Dict[int, PlayerStats] = {}
        self.total_frames: int = 0
    
    def add_frame(self, frame_idx: int, tracked_objects: List[Dict]) -> None:
        """Add frame data and update player statistics.
        
        Args:
            frame_idx: Frame number
            tracked_objects: List of tracked object dictionaries with keys:
                - tracker_id: int (required)
                - team_id: int
                - class_id: int
                - speed_km_h: float
                - distance_m: float
                - possession: bool
                - team: str
                - role: str
        """
        self.total_frames = max(self.total_frames, frame_idx + 1)
        
        for obj in tracked_objects:
            tracker_id = obj.get('tracker_id')
            if tracker_id is None:
                continue
            
            # Initialize player stats if not exists
            if tracker_id not in self.player_stats:
                self.player_stats[tracker_id] = PlayerStats(
                    object_id=tracker_id,
                    team_id=obj.get('team_id', -1),
                    class_id=obj.get('class_id', 0),
                    team=obj.get('team', 'Unknown'),
                    role=obj.get('role', 'player')
                )
            
            stats = self.player_stats[tracker_id]
            
            # Update statistics
            speed = obj.get('speed_km_h', 0.0)
            distance = obj.get('distance_m', 0.0)
            has_possession = obj.get('possession', False)
            
            stats.update_speed(speed)
            stats.update_distance(distance)
            stats.update_possession(has_possession)
    
    def finalize_and_write(self, output_path: Path) -> None:
        """Finalize statistics and write to CSV file.
        
        Args:
            output_path: Path to output CSV file
        """
        if not self.player_stats:
            print("[PlayerSummaryCSVBuilder] No player data to export")
            return
        
        # Convert to DataFrame
        rows = []
        for player_id, stats in self.player_stats.items():
            # Skip ball and non-player objects
            if stats.class_id not in [0, 1, 2, 3]:  # Player, GK, Referee classes
                continue
            
            row = {
                PlayerSummaryCSVColumns.OBJECT_ID: stats.object_id,
                PlayerSummaryCSVColumns.TEAM_ID: stats.team_id,
                PlayerSummaryCSVColumns.CLASS_ID: stats.class_id,
                PlayerSummaryCSVColumns.TOTAL_FRAMES: stats.total_frames,
                PlayerSummaryCSVColumns.TOP_SPEED_KM_H: round(stats.top_speed_km_h, 2),
                PlayerSummaryCSVColumns.AVG_SPEED_KM_H: round(stats.avg_speed_km_h, 2),
                PlayerSummaryCSVColumns.TOTAL_DISTANCE_M: round(stats.total_distance_m, 2),
                PlayerSummaryCSVColumns.POSS_PCT: round(stats.possession_pct, 2),
                PlayerSummaryCSVColumns.ROLE: stats.role,
                PlayerSummaryCSVColumns.TEAM: stats.team,
            }
            rows.append(row)
        
        if not rows:
            print("[PlayerSummaryCSVBuilder] No valid player rows to export")
            return
        
        # Create DataFrame and sort
        df = pd.DataFrame(rows)
        df = df.sort_values([PlayerSummaryCSVColumns.TEAM_ID, PlayerSummaryCSVColumns.OBJECT_ID])
        
        # Write CSV with headers
        write_csv_headers(output_path, PlayerSummaryCSVColumns.all_columns())
        
        # Write data
        df.to_csv(output_path, index=False, mode='a', header=False)
        
        print(f"[PlayerSummaryCSVBuilder] Exported {len(rows)} players to {output_path}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for debugging."""
        if not self.player_stats:
            return {"total_players": 0, "total_frames": self.total_frames}
        
        speeds = [s.avg_speed_km_h for s in self.player_stats.values()]
        distances = [s.total_distance_m for s in self.player_stats.values()]
        
        return {
            "total_players": len(self.player_stats),
            "total_frames": self.total_frames,
            "avg_speed_all_players": sum(speeds) / len(speeds) if speeds else 0,
            "total_distance_all_players": sum(distances),
        }
