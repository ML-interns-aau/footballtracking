import cv2
import os

try:
    from . import enhance_frame_extraction as ef
except ImportError:
    import enhance_frame_extraction as ef

class FrameExtractor:
    def __init__(
        self,
        video_path="data/raw/1.mp4",
        output_dir=os.path.join("data", "interim", "1_mp4_frames"),
        target_sample_fps=15,
        default_fps=30,
        jpeg_quality=95,
        enhancer=None,
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.target_sample_fps = target_sample_fps
        self.default_fps = default_fps
        self.jpeg_quality = jpeg_quality
        self.enhancer = enhancer or ef.FrameEnhancer()

    def _validate_video_path(self):
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

    def _get_video_capture(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        return cap

    def _get_original_fps(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0:
            print(
                f"Warning: couldn't read FPS from video; defaulting to {self.default_fps}"
            )
            return self.default_fps
        return fps

    def _save_frame(self, frame, saved_count):
        processed = self.enhancer.enhance(frame)
        out_path = os.path.join(self.output_dir, f"frame_{saved_count:05d}.jpg")
        ok = cv2.imwrite(
            out_path, processed, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            print(f"Failed to write frame to {out_path}")
            return False
        return True

    def run(self):
        self._validate_video_path()
        os.makedirs(self.output_dir, exist_ok=True)

        cap = self._get_video_capture()
        try:
            original_fps = self._get_original_fps(cap)
            skip_interval = max(1, round(original_fps / self.target_sample_fps))

            print(
                f"Reading {self.video_path} @ {original_fps} FPS, "
                f"saving every {skip_interval} frames into {self.output_dir}"
            )

            saved_count = 0
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame is None:
                    frame_idx += 1
                    continue

                if frame_idx % skip_interval == 0:
                    saved_ok = self._save_frame(frame, saved_count)
                    if saved_ok:
                        saved_count += 1
                        if saved_count % 100 == 0:
                            print(f"Saved {saved_count} frames...")

                frame_idx += 1
        finally:
            cap.release()

        print(f"Finished. Saved {saved_count} frames to {self.output_dir}")


if __name__ == "__main__":
    FrameExtractor().run()
