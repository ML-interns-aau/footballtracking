# Phase 1 — Color Classifier Improvements
# File: .github/instructions/01-color-classifier-improvements.instructions.md
# Apply to: team_classifier.py

## Goal

Improve `TeamClassifier` accuracy without any model training or new dependencies.
All changes are drop-in replacements to the existing color pipeline.
Dependencies: numpy, opencv-python, scikit-learn (already present).

---

## Change 1 — Gray-World Frame Normalization

Add this as a **module-level function** in `team_classifier.py`, above the class definition.
Call it once per frame at the top of the per-frame entry point, before any crop is taken.

```python
def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Gray-world color constancy. Scales each BGR channel so its mean equals
    the global mean across all channels. Reduces white-balance and lighting
    drift between frames and across different stadium lighting conditions.

    Args:
        frame: BGR uint8 image, shape (H, W, 3)
    Returns:
        Normalized BGR uint8 image, same shape
    """
    frame_f = frame.astype(np.float32)
    mean_per_channel = frame_f.mean(axis=(0, 1))   # shape (3,) — B, G, R means
    global_mean = mean_per_channel.mean()           # scalar
    scale = global_mean / (mean_per_channel + 1e-6) # per-channel scale factors
    normalized = np.clip(frame_f * scale, 0, 255)
    return normalized.astype(np.uint8)
```

Usage in the per-frame entry point:
```python
def classify_players(self, frame: np.ndarray, detections: list, ...) -> dict:
    frame = normalize_frame(frame)   # ← add this as the first line
    ...
```

**Do not** call `normalize_frame` inside `_get_jersey_crop()`. Normalize once at the
frame level, not per crop — per-crop normalization changes relative colors between players.

---

## Change 2 — Dynamic Grass Mask

Delete `_GREEN_LOWER` and `_GREEN_UPPER` class constants entirely.

Add this method to `TeamClassifier`. It replaces every use of the old fixed mask.

```python
def _compute_grass_mask(self, hsv_pixels: np.ndarray) -> np.ndarray:
    """
    Dynamically computes a boolean mask of grass-colored pixels based on
    the current frame's actual field color, instead of hardcoded HSV bounds.

    This handles turf color variation across stadiums and lighting conditions.

    Args:
        hsv_pixels: shape (N, 3), OpenCV HSV scale (H: 0–179, S: 0–255, V: 0–255)
    Returns:
        Boolean mask of shape (N,) — True where pixel is grass-colored
    """
    h = hsv_pixels[:, 0].astype(int)
    s = hsv_pixels[:, 1].astype(int)
    v = hsv_pixels[:, 2].astype(int)

    # Step 1: identify candidate grass pixels using broad green hue range
    candidate_mask = (h >= 35) & (h <= 85) & (s > 40) & (v > 40)

    if candidate_mask.sum() < 10:
        # Fallback to a wide static range if no grass found in crop
        # (e.g. extreme close-up or full occlusion)
        return (h >= 30) & (h <= 90)

    # Step 2: compute the median HSV of candidate grass pixels as the
    # "current field color" prototype
    grass_median = np.median(hsv_pixels[candidate_mask], axis=0)
    gh, gs, gv = int(grass_median[0]), int(grass_median[1]), int(grass_median[2])

    # Step 3: build the exclusion mask as pixels within tolerance of that median
    hue_tolerance = 15   # hue units (OpenCV 0–179 scale)
    sat_tolerance = 40
    val_tolerance = 40

    grass_mask = (
        (np.abs(h - gh) < hue_tolerance) &
        (np.abs(s - gs) < sat_tolerance) &
        (np.abs(v - gv) < val_tolerance)
    )
    return grass_mask
```

In `_get_dominant_color()` (or equivalent), replace the old mask call pattern with:
```python
# Convert crop to HSV and flatten
hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
hsv_pixels = hsv_crop.reshape(-1, 3)

# Dynamic grass removal
grass_mask = self._compute_grass_mask(hsv_pixels)
valid_hsv = hsv_pixels[~grass_mask]
```

---

## Change 3 — Switch to Lab Color Space (Highest Impact)

This is the most impactful single change. Lab is perceptually uniform — equal numeric
distances correspond to equal perceived color differences. HSV is not: it wraps around
at hue=0/180, and saturation instability at low values makes similar colors appear
numerically distant.

Replace the dominant color extraction in `_get_dominant_color()` as follows:

```python
def _get_dominant_color(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """
    Extracts the dominant jersey color in Lab color space.
    Uses dynamic grass masking (via HSV) to remove turf pixels before clustering.

    Returns:
        Lab color vector shape (3,), or None if crop is invalid/too small
    """
    crop = self._get_jersey_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None

    # --- Grass exclusion (done in HSV) ---
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv_crop.reshape(-1, 3)
    grass_mask = self._compute_grass_mask(hsv_pixels)

    # --- Dominant color extraction (done in Lab) ---
    lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    lab_pixels = lab_crop.reshape(-1, 3).astype(np.float32)

    # Apply grass mask (same pixel indices as HSV)
    valid_lab = lab_pixels[~grass_mask]

    # Exclude very dark pixels (L < 20 in OpenCV Lab scale 0–255)
    # These are shadows, not jersey fabric
    brightness_mask = valid_lab[:, 0] > 20
    valid_lab = valid_lab[brightness_mask]

    if len(valid_lab) < MIN_PIXELS_FOR_KMEANS:
        return None

    # 1-cluster KMeans gives the dominant Lab color
    kmeans = KMeans(n_clusters=1, n_init=1, random_state=0)
    kmeans.fit(valid_lab)
    return kmeans.cluster_centers_[0]   # shape (3,) — Lab vector
```

Update `fit_teams()` to cluster in Lab space. The cluster centers and all distance
calculations remain structurally identical — only the feature space changes.
**Never mix HSV and Lab vectors in the same KMeans call.**

---

## Change 4 — Tighter Jersey Crop

Replace `_get_jersey_crop()` with this version. It targets the chest/jersey area
rather than the entire top 50% of the bbox, which includes too much face and hair
for small/distant players.

```python
MIN_PIXELS_FOR_KMEANS = 30   # module-level constant

def _get_jersey_crop(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """
    Crops the torso/jersey region from a player bounding box.

    Vertical range: 15%–55% of bbox height (skips face/hair, stops before shorts)
    Horizontal range: inset by 10% on each side (reduces background contamination)

    Args:
        frame: full BGR frame
        bbox: (x1, y1, x2, y2) in absolute pixel coordinates
    Returns:
        BGR crop array, or None if the bbox is too small/degenerate
    """
    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1
    w = x2 - x1

    top    = y1 + int(0.15 * h)
    bottom = y1 + int(0.55 * h)
    left   = x1 + int(0.10 * w)
    right  = x2 - int(0.10 * w)

    # Guard: degenerate box (very small/distant player)
    if bottom <= top or right <= left or (bottom - top) * (right - left) < MIN_PIXELS_FOR_KMEANS:
        return None

    return frame[top:bottom, left:right]
```

---

## Testing This Phase

After implementing all four changes, run a visual sanity check before any metric
evaluation. Create a quick debug script `scripts/debug_classifier.py`:

```python
"""
Debug script: visualize jersey crops and their dominant Lab colors on a video clip.
Usage: python scripts/debug_classifier.py --video path/to/clip.mp4 --frames 60
"""
```

The script should:
1. Run `normalize_frame()` on each frame and display side-by-side (original vs normalized)
2. Extract jersey crops for each detection and display them in a grid
3. Print the dominant Lab vector for each player per frame
4. Display cluster centers (team colors) as colored rectangles labeled "team_left" / "team_right"

Acceptance criteria before moving to phase 2:
- [ ] Normalized frames look visually consistent across lit/shadowed areas
- [ ] Crops hit jersey chest area, not face or background, for most players
- [ ] Cluster centers are stable across ≥30 consecutive frames (print them — they should not jump)
- [ ] Team flip rate is lower than before the changes
