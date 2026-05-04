# Football Analytics - Configuration Management Guide

## Overview

This project now uses centralized configuration management via YAML files. All pipeline parameters, thresholds, and settings are configurable without modifying code.

---

## Quick Start

### 1. View Current Configuration
```python
from src.config import CONFIG
CONFIG.print_summary()
```

### 2. Get Configuration Values
```python
from src.config import CONFIG

# Single values with defaults
conf_threshold = CONFIG.get('detection.confidence_threshold', 0.30)
pitch_width = CONFIG.get('pitch.width_m', 105.0)

# Entire sections
detection_config = CONFIG.get_dict('detection')
```

### 3. Use Environment-Specific Configs

**Development (fast testing):**
```powershell
$env:FOOTBALL_ENV='dev'
python main.py --input video.mp4
```

**Production (full analysis):**
```powershell
$env:FOOTBALL_ENV='prod'
python main.py --input video.mp4
```

**Default (normal settings):**
```powershell
python main.py --input video.mp4
```

---

## Configuration Files

### File Structure
```
configs/
├── pipeline.yaml          # Base configuration (always loaded first)
├── pipeline.dev.yaml      # Development overrides
├── pipeline.prod.yaml     # Production overrides
├── homography.json        # Pitch calibration points
└── config.yaml            # Legacy (empty, can be removed)
```

### Configuration Inheritance
1. **Base** (`pipeline.yaml`) - Always loaded
2. **Environment** (`pipeline.{env}.yaml`) - Overrides base if env var set
3. **Defaults** (in code) - Fallback if files missing

### Example: How Dev Config Overrides Work

**pipeline.yaml:**
```yaml
detection:
  confidence_threshold: 0.30
  image_size: 1280
```

**pipeline.dev.yaml:**
```yaml
detection:
  confidence_threshold: 0.25  # Overrides 0.30
  image_size: 640             # Overrides 1280
```

When `FOOTBALL_ENV=dev`:
- confidence_threshold = **0.25**
- image_size = **640**

---

## Configuration Reference

### Detection Settings
| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `detection.confidence_threshold` | 0.30 | 0.0-1.0 | Minimum confidence for detections |
| `detection.iou_threshold` | 0.40 | 0.0-1.0 | IoU threshold for NMS |
| `detection.image_size` | 1280 | >0 | Input image size for YOLO |

### Tracking Settings
| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `tracking.track_threshold` | 0.20 | 0.0-1.0 | Detection threshold for tracking |
| `tracking.track_buffer` | 60 | >0 | Frames to keep lost tracks |
| `tracking.match_threshold` | 0.80 | 0.0-1.0 | IoU threshold for matching |

### Team Classification
| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `team_classification.n_teams` | 2 | >0 | Number of teams |
| `team_classification.history_length` | 15 | >0 | Frames for color histogram |
| `team_classification.refit_interval` | 150 | >0 | Frames between K-means refit |

### Ball Tracking
| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `ball.max_trail_length` | 25 | >0 | Past positions to keep |
| `ball.max_missed_frames` | 30 | >0 | Frames before ball track lost |

### Pitch Dimensions
| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `pitch.width_m` | 105.0 | >0 | Pitch width in meters |
| `pitch.height_m` | 68.0 | >0 | Pitch height in meters |

### Video Processing
| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `video.target_fps` | 15 | ≥0 | Target FPS (0 = source FPS) |
| `video.resize_width` | 1280 | ≥0 | Resize width (0 = no resize) |
| `video.codec_preference` | "avc1" | - | Primary video codec |
| `video.codec_fallback` | "mp4v" | - | Fallback codec |

### Speed Estimation
| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `speed.window_size` | 8 | >0 | Frames for rolling average |
| `speed.min_frames_for_speed` | 3 | >0 | Min frames before speed calc |

### Possession Detection
| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `possession.max_distance_m` | 2.0 | >0 | Max distance for possession claim |

### Heatmap Generation
| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `heatmap.bins_x` | 105 | >0 | X-axis bins (1 per meter) |
| `heatmap.bins_y` | 68 | >0 | Y-axis bins (1 per meter) |

### Performance Settings
| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `performance.max_frames_default` | 0 | ≥0 | Max frames (0 = unlimited) |
| `performance.frame_batch_size` | 1 | >0 | Frames between UI updates |
| `performance.memory_warning_mb` | 2048 | >0 | Memory warning threshold |

### Output Settings
| Key | Default | Options | Description |
|-----|---------|---------|-------------|
| `output.save_intermediate` | false | bool | Save debug frames |
| `output.video_format` | "mp4" | "mp4" | Output video format |
| `output.csv_format` | "csv" | "csv"/"parquet" | Output CSV format |

---

## Common Use Cases

### 1. Tuning Detection Confidence
Edit `configs/pipeline.yaml`:
```yaml
detection:
  confidence_threshold: 0.25  # Lower = more detections, more false positives
  confidence_threshold: 0.40  # Higher = fewer detections, more precise
```

### 2. Faster Processing for Testing
Edit `configs/pipeline.dev.yaml`:
```yaml
detection:
  image_size: 640  # Smaller = faster

performance:
  max_frames_default: 100  # Only process 100 frames
```

### 3. Longer Ball Trail for Analysis
Edit `configs/pipeline.yaml`:
```yaml
ball:
  max_trail_length: 50  # Keep more history
```

### 4. Stricter Team Classification
Edit `configs/pipeline.yaml`:
```yaml
team_classification:
  history_length: 25  # More frames for stability
  refit_interval: 300  # Less frequent refitting
```

---

## Validation

Configuration is automatically validated on startup. Invalid configs will raise errors with helpful messages.

### Manual Validation
```python
from src.config import CONFIG

# Check without raising
is_valid, errors = CONFIG.validate(raise_on_error=False)
if not is_valid:
    print(f"Configuration errors: {errors}")

# Force raise on error
CONFIG.validate(raise_on_error=True)  # Raises ValueError if invalid
```

### Validation Rules
- Required keys must exist
- Values must be within valid ranges
- Types are checked (e.g., confidence must be 0.0-1.0)

---

## Troubleshooting

### "Missing required configuration: detection.confidence_threshold"
Your YAML file is missing a required key. Check `configs/pipeline.yaml` has all sections from the reference above.

### "detection.confidence_threshold = 1.5, must be <= 1.0"
Your value is out of range. Confidence thresholds must be between 0.0 and 1.0.

### Configuration Not Loading
1. Check file exists: `configs/pipeline.yaml`
2. Check YAML syntax (use online YAML validator)
3. Check file encoding (should be UTF-8)
4. Run `CONFIG.print_summary()` to see loaded path

### Changes Not Taking Effect
1. Restart Python/Streamlit (config is cached)
2. Or call `CONFIG.reload()` to refresh
3. Check environment variable: `echo $env:FOOTBALL_ENV`

---

## Migration from Hardcoded Values

If you previously modified hardcoded values in code:

1. Find the old value in `main.py` or pipeline files
2. Add it to `configs/pipeline.yaml` with the same value
3. Remove the old hardcoded value (optional - defaults are fallback)

Example:
```python
# Old hardcoded in main.py
tracker = FootballTracker(track_buffer=100)  # Was 60

# New in configs/pipeline.yaml
tracking:
  track_buffer: 100  # Now configurable!
```

---

## API Reference

### ConfigLoader Class

#### `load(config_path=None)`
Load configuration from file. Uses environment-specific config if `FOOTBALL_ENV` set.

#### `reload(config_path=None)`
Force reload configuration without restarting.

#### `get(key, default=None)`
Get configuration value using dot notation.
```python
CONFIG.get('detection.confidence_threshold', 0.30)
```

#### `get_dict(key)`
Get a configuration section as dictionary.
```python
detection = CONFIG.get_dict('detection')
# Returns: {'confidence_threshold': 0.30, 'iou_threshold': 0.40, ...}
```

#### `validate(raise_on_error=True)`
Validate all configuration values.
```python
is_valid, errors = CONFIG.validate(raise_on_error=False)
```

#### `get_environment()`
Return current environment name.
```python
env = CONFIG.get_environment()  # 'dev', 'prod', or 'default'
```

#### `print_summary()`
Print configuration summary for debugging.
```python
CONFIG.print_summary()
```

---

## Files Modified During Implementation

1. **Created:**
   - `configs/pipeline.yaml` - Base configuration
   - `configs/pipeline.dev.yaml` - Development overrides
   - `configs/pipeline.prod.yaml` - Production overrides
   - `src/config/config_loader.py` - Configuration loader
   - `src/config/__init__.py` - Package exports

2. **Modified:**
   - `main.py` - Updated all component initializations to use CONFIG
   - `app/config.py` - Updated defaults to use CONFIG
   - `requirements.txt` - Added `pyyaml>=6.0`

---

## Summary

✅ All pipeline parameters now configurable via YAML
✅ Environment-specific configs (dev/prod/default)
✅ Automatic validation on startup
✅ Backward compatible (defaults work without config file)
✅ No breaking changes to existing code
