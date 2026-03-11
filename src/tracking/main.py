import cv2
from datetime import datetime
from pathlib import Path
import time

try:
    from .config import Config
    from .persistence import TrackingPersistence
    from .tracker import BaseTracker
    from .utils import VideoUtils
except ImportError:
    from config import Config
    from persistence import TrackingPersistence
    from tracker import BaseTracker
    from utils import VideoUtils


def run_pipeline(input_source: str | None = None, output_dest: str | None = None):
    def _format_seconds(seconds: float) -> str:
        total = max(0, int(seconds))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    cfg = Config()
    project_root = Path(__file__).resolve().parents[2]
    input_source = input_source or cfg.INPUT_VIDEO
    input_path = Path(input_source)
    if not input_path.is_absolute():
        root_candidate = (project_root / input_path).resolve()
        cwd_candidate = (Path.cwd() / input_path).resolve()
        input_path = root_candidate if root_candidate.exists() else cwd_candidate

    tracker = BaseTracker(cfg)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video source: {input_path}. "
            f"Available sample: {project_root / 'data/raw/114.mp4'}"
        )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    persistence = TrackingPersistence(cfg, input_source=str(input_path), run_name=run_name)
    if output_dest is not None:
        persistence.video_output_path = persistence._resolve_output_dir(output_dest)

    writer = VideoUtils.get_writer(cap, str(persistence.video_output_path))
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    started_at = time.perf_counter()
    last_report_at = started_at
    report_interval_seconds = 2.0

    print(
        f"Processing video: {input_path} | source_fps={fps:.2f} "
        f"| total_frames={total_frames if total_frames > 0 else 'unknown'}"
    )

    try:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # 1. Get tracking/classification/keypoint results.
                result = tracker.get_tracks(frame)

                # 2. Extract bottom-center coordinates.
                centers = tracker.extract_coords(result)

                # 3. Visualize.
                annotated_frame = result.annotated_frame
                annotated_frame = VideoUtils.draw_custom_ui(annotated_frame, centers)
                writer.write(annotated_frame)

                if cfg.SAVE_TRACKING_EXPORTS:
                    persistence.write_frame(
                        frame_index=frame_idx,
                        timestamp_seconds=frame_idx / fps,
                        frame_output=result,
                    )

                if cfg.SHOW_PREVIEW:
                    cv2.imshow("Tracking System", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_idx += 1

                now = time.perf_counter()
                if now - last_report_at >= report_interval_seconds:
                    elapsed = now - started_at
                    proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                    if total_frames > 0 and proc_fps > 0:
                        pct = (frame_idx / total_frames) * 100.0
                        remaining = max(0, total_frames - frame_idx)
                        eta = remaining / proc_fps
                        print(
                            f"[progress] {frame_idx}/{total_frames} ({pct:.1f}%) "
                            f"| speed={proc_fps:.2f} fps | elapsed={_format_seconds(elapsed)} "
                            f"| eta={_format_seconds(eta)}"
                        )
                    else:
                        print(
                            f"[progress] frames={frame_idx} | speed={proc_fps:.2f} fps "
                            f"| elapsed={_format_seconds(elapsed)}"
                        )
                    last_report_at = now
        except KeyboardInterrupt:
            print("Interrupted by user. Finalizing outputs...")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        persistence.write_summary(
            {
                "input_source": input_source,
                "resolved_input_source": str(input_path),
                "frames_processed": frame_idx,
                "fps": fps,
                "video_output_path": str(persistence.video_output_path),
                "tracks_jsonl_path": str(persistence.jsonl_path),
                "tracks_csv_path": str(persistence.csv_path),
            }
        )
        persistence.close()

    print(f"Output video: {persistence.video_output_path}")
    print(f"Tracking JSONL: {persistence.jsonl_path}")
    print(f"Tracking CSV: {persistence.csv_path}")

if __name__ == "__main__":
    run_pipeline()
