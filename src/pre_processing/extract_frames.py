import cv2
import numpy as np
import os

def enhance_frame_production(frame):
    # 1. Resize first to target AI input (e.g., 1280x720)
    # Processing fewer pixels is the #1 rule of real-time CV 
    frame = cv2.resize(frame, (1280, 720))

    # 2. Sharpening via Laplacian (Much faster than NlMeans)
    # This makes the edges of the ball and boots much sharper 
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(frame, -1, kernel)

    # 3. CLAHE on the L channel (Essential for varying stadium light)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# --- Main Logic ---
video_path = "data/raw/1.mp4"

# 1) Ensure input exists and can be opened
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {video_path}")

# 2) Derive FPS and sampling interval (defensive defaults)
original_fps = cap.get(cv2.CAP_PROP_FPS) or 0
if original_fps <= 0:
    print("Warning: couldn't read FPS from video; defaulting to 30")
    original_fps = 30

skip_interval = max(1, round(original_fps / 15))

saved_count = 0
frame_idx = 0

# 3) Checking if output directory exists.
out_dir = os.path.join("data", "interim", "1_mp4_frames")
os.makedirs(out_dir, exist_ok=True)

print(f"Reading {video_path} @ {original_fps} FPS, saving every {skip_interval} frames into {out_dir}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame is None:
        frame_idx += 1
        continue

    if frame_idx % skip_interval == 0:
        processed = enhance_frame_production(frame)
        out_path = os.path.join(out_dir, f"frame_{saved_count:05d}.jpg")
        ok = cv2.imwrite(out_path, processed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            print(f"Failed to write frame to {out_path}")
        else:
            saved_count += 1
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} frames...")

    frame_idx += 1

cap.release()
print(f"Finished. Saved {saved_count} frames to {out_dir}")