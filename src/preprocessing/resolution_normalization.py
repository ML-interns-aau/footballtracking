from __future__ import annotations

import cv2
from pathlib import Path


def resize_frame(frame, resize_width: int):
	"""Resize `frame` to `resize_width` while preserving aspect ratio.

	If `resize_width` is falsy or matches the current width, the original
	frame is returned.
	"""
	if not resize_width or resize_width <= 0:
		return frame

	h, w = frame.shape[:2]
	if w == resize_width:
		return frame

	scale = resize_width / float(w)
	new_h = max(1, int(round(h * scale)))
	return cv2.resize(frame, (resize_width, new_h))


def preprocess_video(input_path: str | Path, output_path: str | Path, target_fps: float, resize_width: int, progress_callback=None):
	"""Normalize FPS and optionally resize a video.

	The output video is written to `output_path`. When provided,
	`progress_callback(current, total)` is called after each written frame.
	"""
	input_path = Path(input_path)
	output_path = Path(output_path)

	if target_fps <= 0:
		raise ValueError("target_fps must be greater than 0")

	cap = cv2.VideoCapture(str(input_path))
	if not cap.isOpened():
		raise FileNotFoundError(f"Unable to open video: {input_path}")

	source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	frame_step = max(1, round(source_fps / target_fps))
	out_width = resize_width if resize_width and resize_width > 0 else source_width
	if out_width != source_width:
		scale = out_width / float(source_width)
		out_height = max(1, int(round(source_height * scale)))
	else:
		out_height = source_height

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(output_path), fourcc, float(target_fps), (out_width, out_height))
	if not writer.isOpened():
		cap.release()
		raise RuntimeError(f"Unable to open output video writer: {output_path}")

	written = 0
	frame_index = 0
	estimated_total = max(1, (total_frames + frame_step - 1) // frame_step) if total_frames > 0 else 1

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		if frame_index % frame_step == 0:
			if out_width != source_width or out_height != source_height:
				frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
			writer.write(frame)
			written += 1
			if progress_callback is not None:
				progress_callback(written, estimated_total)

		frame_index += 1

	cap.release()
	writer.release()
	return output_path
