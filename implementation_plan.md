# Football Tracking Improvements Plan

This plan addresses the two major architectural improvements requested: replacing the hardcoded pitch mapping with a configurable camera calibration system, and overhauling the possession and pass detection logic to be much more robust.

## User Review Required
Please review the proposed approach for camera calibration. I am proposing a JSON-based configuration file mapping video names to specific pitch coordinates, along with a helper script to generate these coordinates. Let me know if you would prefer a fully automated approach (which would require integrating a new pitch-segmentation deep learning model) or if the semi-automated configuration approach is sufficient for your needs.

## Open Questions
1. **Calibration Tooling:** Would you like me to build a small standalone Python script (`scripts/calibrate.py`) that opens a video frame and lets you click 4 points on the field to automatically generate the calibration JSON for a specific video?
2. **Possession Parameters:** I plan to use thresholds like `POSSESSION_RADIUS = 1.5 meters` and `MIN_FRAMES_POSSESSION = 5 frames` (to prevent flickering). Do these sound like reasonable defaults for your standard 25fps video?

## Proposed Changes

---

### Calibration System (Pitch Mapping)
The current `PitchMapper` hardcodes 4 source points based on the video width and height. We will move to a configuration-based approach.

#### [NEW] `data/calibrations.json`
- A JSON file storing video-specific calibration matrices.
- Format: `{"video_name.mp4": {"src_pts": [[x1,y1], ...], "dst_pts": [[x1,y1], ...]}}`

#### [MODIFY] `src/pipeline/pitch_mapper.py`
- Modify `PitchMapper` initialization to accept an optional `video_name` or direct `src_pts`/`dst_pts`.
- Add a class method `PitchMapper.from_config(video_name, config_path, default_width, default_height)` to load points from `calibrations.json`, falling back to the current hardcoded defaults (and logging a warning) if the video isn't found.

#### [MODIFY] `main.py`
- Update the initialization of `PitchMapper` to use the new `from_config` method, passing the input video filename.

---

### Advanced Possession & Pass Detection
The current logic in `DataExporter.update_passes` instantly assigns possession to the closest player within 1.0m. We will refactor this to use temporal smoothing and ball velocity.

#### [MODIFY] `src/pipeline/data_exporter.py`
- Enhance `DataExporter` state variables to track:
  - `current_possessor_id`
  - `possession_frames` (how long the current player has had the ball)
  - `last_possessor_id` (for pass tracking)
  - `loose_ball_frames` (how long the ball has been away from the possessor)
- Update `update_passes(self, frame_idx, ball_pos, player_positions, ball_speed_kmh)` signature to accept ball speed.
- **New Logic Flow**:
  1. Find the closest player to the ball.
  2. **Temporal Smoothing (Gain Possession)**: If the closest player is within `POSSESSION_RADIUS` (e.g., 1.5m), increment their "contact frames". If contact frames > `MIN_POSSESSION_FRAMES` (e.g., 5), they gain possession.
  3. **Temporal Smoothing (Lose Possession)**: If the ball is further than `POSSESSION_RADIUS` from the *current* possessor, increment `loose_ball_frames`. If `loose_ball_frames` > `MIN_LOOSE_FRAMES` OR `ball_speed_kmh` > `PASS_SPEED_THRESHOLD` (e.g., 20 km/h), the player loses possession.
  4. **Pass Detection**: If possession changes from Player A to Player B, and there was a period of "loose ball" with high speed in between, log a valid pass.

#### [MODIFY] `main.py`
- Pass `ball_speed_kmh` into `data_exporter.update_passes()` inside the main processing loop.

---

## Verification Plan

### Automated Tests
- Run `main.py` on a sample video slice (`--max_frames 100`).
- Inspect the output `analytics.json` to verify that possession events do not flicker rapidly between frames.
- Ensure `analytics.csv` correctly records the possession data.

### Manual Verification
- View the annotated output video. Verify that possession assignments look natural (not jumping to a player when the ball is mid-air passing over them).
- Check the console logs for "Pass detected" and verify they correlate with actual passing events rather than just players running near each other.
