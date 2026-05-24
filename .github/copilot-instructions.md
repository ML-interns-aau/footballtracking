# Copilot Instructions — Team Classifier One-Day Improvement Sprint

## Project Context

This project is a football (soccer) video analysis system. The `TeamClassifier` class
(in `team_classifier.py`) assigns detected players to teams using HSV color clustering.
The goal of this sprint is to significantly improve classification accuracy **without**
fine-tuning or training any model. All changes are algorithmic and drop-in.

The detector lives in `detector.py`. The classifier is called per-frame with person
bounding boxes and optional tracker IDs.

---

## Sprint Goal (One Day)

Improve team classification accuracy by fixing the root causes of error:

1. Lighting/white-balance drift → **gray-world color normalization**
2. Hardcoded grass exclusion failing on different turf colors → **dynamic grass mask**
3. HSV being perceptually non-uniform → **switch dominant color extraction to Lab color space**
4. Poor jersey crops on distant/small players → **tighten the bbox crop**

No model training. No new dependencies beyond what numpy/sklearn/opencv already provide.

---

## File-by-File Instructions

### `team_classifier.py` — Primary Target

#### 1. Add a `normalize_frame()` utility (gray-world assumption)

Add this as a **module-level function** (not a method), called once per frame before
any crop is extracted:

```python
def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Gray-world color constancy normalization.
    Scales each BGR channel so the per-channel mean equals the global mean.
    Reduces white-balance and lighting drift across frames.
    """
    frame = frame.astype(np.float32)
    mean_per_channel = frame.mean(axis=(0, 1))   # shape (3,)
    global_mean = mean_per_channel.mean()
    scale = global_mean / (mean_per_channel + 1e-6)
    normalized = np.clip(frame * scale, 0, 255)
    return normalized.astype(np.uint8)
```

Call it at the top of `classify_players()` (or whatever the per-frame entry point is)
**before** any crop is taken:

```python
frame = normalize_frame(frame)
```

Do not apply it inside `_get_jersey_crop()` — normalize once at the frame level, not
per crop.

---

#### 2. Replace the fixed green mask with a dynamic one

Delete `_GREEN_LOWER` and `_GREEN_UPPER` class constants.

Add a method `_compute_grass_mask()` that samples the field color from the current
frame each time it is called. The field is the dominant green-ish region. Strategy:

- Convert frame to HSV.
- Keep pixels where `H` is in `[35, 85]` (broad green hue range in OpenCV 0–179 scale)
  and `S > 40` and `V > 40`.
- Compute the **median HSV** of those pixels as the "current grass color".
- Build the exclusion mask as pixels within ±15 H, ±40 S, ±40 V of that median.

```python
def _compute_grass_mask(self, hsv_pixels: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask of pixels that look like the current field grass.
    hsv_pixels: shape (N, 3) in OpenCV HSV scale (H: 0-179, S: 0-255, V: 0-255).
    """
    h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
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
```

Replace every use of the old `_GREEN_LOWER`/`_GREEN_UPPER` mask with a call to
`_compute_grass_mask()` on the flattened HSV pixels of each crop.

---

#### 3. Switch dominant-color extraction to Lab color space

This is the highest-impact single change. Lab is perceptually uniform — similar
jersey colors are numerically close, and the hue instability of HSV near low
saturation is eliminated.

Replace the dominant color extraction logic in `_get_dominant_color()` (or equivalent)
as follows:

- Convert the crop from BGR to **Lab** using `cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)`.
- Flatten to `(N, 3)`.
- Apply the dynamic grass mask (convert the same crop to HSV first, compute mask,
  then apply it to the Lab pixels — use the same pixel indices).
- Exclude very dark pixels: drop rows where `L < 20` (Lab L channel, OpenCV scale 0–255,
  so threshold is `~20`).
- Run `KMeans(n_clusters=1)` on the remaining Lab pixels to get the dominant Lab vector.
- Return this Lab vector as the player's color feature.

Update `fit_teams()` to cluster in Lab space instead of HSV. The cluster centers and
distance calculations all remain the same — just the feature space changes.

**Do not** mix HSV and Lab vectors. Pick one (Lab) and use it end-to-end for both
fitting and inference.

---

#### 4. Tighten the jersey crop

In `_get_jersey_crop()`, the current crop takes the top 50% of the bounding box.
This is too generous on small/distant players and includes too much face, hair,
and background.

Change it to:

```python
def _get_jersey_crop(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Returns a tight torso crop: vertically from 15% to 55% of the bbox height,
    horizontally inset by 10% on each side.
    This targets the jersey chest area and reduces face/hair/background contamination.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1
    w = x2 - x1

    top    = y1 + int(0.15 * h)
    bottom = y1 + int(0.55 * h)
    left   = x1 + int(0.10 * w)
    right  = x2 - int(0.10 * w)

    # Guard against degenerate boxes (very small detections)
    if bottom <= top or right <= left:
        return None

    return frame[top:bottom, left:right]
```

Return `None` for degenerate crops and skip them in the caller — do not pass empty
arrays to KMeans.

---

#### 5. Guard against KMeans receiving too-few pixels

After applying the grass mask and dark-pixel exclusion, the remaining pixel count can
drop to near zero for small crops. Add a guard:

```python
MIN_PIXELS_FOR_KMEANS = 30

if len(valid_pixels) < MIN_PIXELS_FOR_KMEANS:
    return None  # skip this player for this frame
```

Handle `None` returns in `fit_teams()` by filtering them out before clustering.

---

### `detector.py` — Minor Touch

No structural changes needed. Just make sure the bounding boxes returned are in
`(x1, y1, x2, y2)` absolute pixel format before passing to `TeamClassifier`. If they
are normalized `[0, 1]`, denormalize with frame width/height before the crop step.

---

## What NOT to Do Today

- Do **not** add any new deep learning model or pretrained CNN (that is the next sprint).
- Do **not** change the voting/locking logic — it is working well and will benefit
  automatically from the cleaner upstream signal.
- Do **not** change `refit_interval` or cluster count — keep those stable while
  testing the color pipeline changes.
- Do **not** add new pip dependencies. Everything here uses `numpy`, `opencv-python`,
  and `scikit-learn`, which are already present.

---

## Testing Checklist

After each change, verify on a short video clip (30–60 seconds is enough):

- [ ] Frame normalization: jersey colors look consistent across shadowed/lit regions
- [ ] Dynamic grass mask: players near the sideline are not partially masked out
- [ ] Lab clustering: `fit_teams()` produces stable cluster centers across frames
      (print them — they should not jump around)
- [ ] Tighter crop: visually inspect a few crops to confirm they hit chest/jersey
      and not face/background
- [ ] No crashes on small detections (degenerate bbox guard works)
- [ ] Overall: per-tracker team assignment flips less often than before

Run your existing evaluation script (if any) before and after and compare flip-rate
and per-frame accuracy.

---

## Suggested Implementation Order

1. `normalize_frame()` — 20 min, isolated, easy to test
2. Tighter `_get_jersey_crop()` — 15 min, visual sanity check with `cv2.imshow`
3. Dynamic `_compute_grass_mask()` — 30 min, replace old constants
4. Lab color space swap in `_get_dominant_color()` + `fit_teams()` — 45 min, most impactful
5. Degenerate crop guard — 10 min, defensive coding
6. End-to-end test on a full clip — remaining time

Total estimated time: ~2–3 hours of focused coding + testing time.
