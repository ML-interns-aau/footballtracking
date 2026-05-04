"""Possession Summary CSV Builder - Team-level possession statistics.

This module builds possession_summary.csv with team possession percentages
and frame counts.
"""

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from pathlib import Path

from src.pipeline.output_schema import (
    OutputFiles,
    PossessionSummaryCSVColumns,
    write_csv_headers,
)


@dataclass
class TeamPossessionStats:
    """Possession statistics for a single team."""
    team_id: int
    possession_frames: int = 0
    total_frames: int = 0
    
    @property
    def possession_pct(self) -> float:
        """Calculate possession percentage."""
        if self.total_frames == 0:
            return 0.0
        return (self.possession_frames / self.total_frames) * 100.0
    
    def update_possession(self, has_possession: bool) -> None:
        """Update possession statistics for a frame."""
        self.total_frames += 1
        if has_possession:
            self.possession_frames += 1


class PossessionSummaryCSVBuilder:
    """Builds possession_summary.csv with team possession statistics."""
    
    def __init__(self):
        self.team_stats: Dict[int, TeamPossessionStats] = {}
        self.total_frames: int = 0
    
    def add_frame(self, frame_idx: int, tracked_objects: List[Dict]) -> None:
        """Add frame data and update team possession statistics.
        
        Args:
            frame_idx: Frame number
            tracked_objects: List of tracked object dictionaries with keys:
                - team_id: int
                - possession: bool (whether object has ball possession)
        """
        self.total_frames = max(self.total_frames, frame_idx + 1)
        
        # Track possession per team for this frame
        team_possession_this_frame: Dict[int, bool] = {}
        
        for obj in tracked_objects:
            team_id = obj.get('team_id', -1)
            has_possession = obj.get('possession', False)
            
            # Skip invalid team IDs
            if team_id < 0:
                continue
            
            # Initialize team stats if not exists
            if team_id not in self.team_stats:
                self.team_stats[team_id] = TeamPossessionStats(team_id=team_id)
            
            # Track if team has possession in this frame
            if has_possession:
                team_possession_this_frame[team_id] = True
        
        # Update all teams for this frame
        for team_id, stats in self.team_stats.items():
            has_possession = team_possession_this_frame.get(team_id, False)
            stats.update_possession(has_possession)
    
    def finalize_and_write(self, output_path: Path) -> None:
        """Finalize statistics and write to CSV file.
        
        Args:
            output_path: Path to output CSV file
        """
        if not self.team_stats:
            print("[PossessionSummaryCSVBuilder] No team data to export")
            return
        
        # Convert to DataFrame
        rows = []
        for team_id, stats in self.team_stats.items():
            # Only include valid team IDs (0, 1 for main teams)
            if team_id not in [0, 1]:
                continue
            
            row = {
                PossessionSummaryCSVColumns.TEAM_ID: stats.team_id,
                PossessionSummaryCSVColumns.POSSESSION_PCT: round(stats.possession_pct, 2),
                PossessionSummaryCSVColumns.TOTAL_FRAMES: stats.total_frames,
            }
            rows.append(row)
        
        if not rows:
            print("[PossessionSummaryCSVBuilder] No valid team rows to export")
            return
        
        # Create DataFrame and sort
        df = pd.DataFrame(rows)
        df = df.sort_values(PossessionSummaryCSVColumns.TEAM_ID)
        
        # Write CSV with headers
        write_csv_headers(output_path, PossessionSummaryCSVColumns.all_columns())
        
        # Write data
        df.to_csv(output_path, index=False, mode='a', header=False)
        
        print(f"[PossessionSummaryCSVBuilder] Exported {len(rows)} teams to {output_path}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for debugging."""
        if not self.team_stats:
            return {"total_teams": 0, "total_frames": self.total_frames}
        
        possession_pcts = [s.possession_pct for s in self.team_stats.values()]
        
        return {
            "total_teams": len(self.team_stats),
            "total_frames": self.total_frames,
            "team_possessions": {team_id: round(stats.possession_pct, 2) 
                               for team_id, stats in self.team_stats.items()},
            "total_possession_pct": sum(possession_pcts),
        }
