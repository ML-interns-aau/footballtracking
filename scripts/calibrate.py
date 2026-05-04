import cv2
import json
import argparse
from pathlib import Path

# Setup global variables
points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)}: ({x}, {y})")

def main():
    parser = argparse.ArgumentParser(description="Calibrate camera for a specific video")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--config", default="data/calibrations.json", help="Path to calibrations.json")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read the first frame.")
        return

    print("--- Calibration Tool ---")
    print("Click on 4 points in the video frame in the following order:")
    print("1. Bottom Left (e.g., [0, 68] on the standard pitch)")
    print("2. Bottom Right (e.g., [105, 68])")
    print("3. Top Right (e.g., [105, 0])")
    print("4. Top Left (e.g., [0, 0])")
    print("Press 'q' to quit without saving.")

    cv2.imshow("Calibration", frame)
    cv2.setMouseCallback("Calibration", click_event)

    while True:
        temp_frame = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(temp_frame, (pt[0], pt[1]), 5, (0, 0, 255), -1)
            cv2.putText(temp_frame, str(i + 1), (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if i > 0:
                cv2.line(temp_frame, tuple(points[i - 1]), tuple(pt), (0, 255, 0), 2)
        
        if len(points) == 4:
            cv2.line(temp_frame, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)

        cv2.imshow("Calibration", temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Calibration cancelled.")
            break
        elif len(points) == 4:
            print("\n4 points collected. Saving to config...")
            cv2.waitKey(1000) # Wait a sec so user can see the lines
            
            config_path = Path(args.config)
            config_data = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            
            # The standard destination points for a 105x68m pitch
            dst_pts = [
                [0, 68],
                [105, 68],
                [105, 0],
                [0, 0]
            ]
            
            video_name = video_path.name
            config_data[video_name] = {
                "src_pts": points,
                "dst_pts": dst_pts
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            print(f"Calibration for {video_name} saved to {config_path}!")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
