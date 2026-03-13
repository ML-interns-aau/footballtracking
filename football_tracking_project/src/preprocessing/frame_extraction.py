"""Extract frames from a video at a target FPS.

Usage:
	python src/preprocessing/frame_extraction.py \
		--input data/raw_videos/clean-match.mp4 \
		--output data/frames \
		--fps 15
"""



import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, target_fps: float) -> int:
	if target_fps <= 0:
		raise ValueError("target_fps must be greater than 0")

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise FileNotFoundError(f"Unable to open video: {video_path}")

	source_fps = cap.get(cv2.CAP_PROP_FPS)
	if not source_fps or source_fps <= 0:
		source_fps = 30.0

	output_dir.mkdir(parents=True, exist_ok=True)

	frame_index: int = 0
	saved_count: int = 0

	# Save roughly every Nth frame to reach the desired sampling frequency.
	frame_step = max(1, round(source_fps / target_fps))

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		if frame_index % frame_step == 0:
			frame_name = output_dir / f"frame_{saved_count:06d}.png"
			cv2.imwrite(str(frame_name), frame)
			saved_count += 1

		frame_index += 1

	cap.release()
	return saved_count


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Extract frames from video")
	parser.add_argument("--input", type=Path, required=True, help="Path to input video")
	parser.add_argument("--output", type=Path, required=True, help="Output frames directory")
	parser.add_argument("--fps", type=float, default=15.0, help="Target extraction FPS")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	saved = extract_frames(args.input, args.output, args.fps)
	print(f"Saved {saved} frames to {args.output}")


if __name__ == "__main__":
	main()

