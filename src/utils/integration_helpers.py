from __future__ import annotations
import argparse
from typing import Union
Number = Union[int, float]
DEFAULT_FPS: float = 25.0
def kmh_to_ms(kmh: Number) -> float:
    if isinstance(kmh, bool) or not isinstance(kmh, (int, float)):
        raise TypeError(f"kmh must be int or float, got {type(kmh).__name__}")
    return float(kmh) / 3.6
def format_game_clock(seconds: Number, fmt: str = "MM:SS") -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TypeError(f"seconds must be int or float, got {type(seconds).__name__}")
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if fmt == "MM:SS":
        total_minutes = hours * 60 + minutes
        return f"{total_minutes:02d}:{secs:02d}"
    if fmt == "HH:MM:SS":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    if fmt == "M:SS":
        total_minutes = hours * 60 + minutes
        return f"{total_minutes}:{secs:02d}"
    raise ValueError(f"Unsupported fmt {fmt!r}. Use 'MM:SS', 'HH:MM:SS', or 'M:SS'.")
def timestamp_from_frame(frame_idx: int, fps: float = DEFAULT_FPS) -> float:
    if isinstance(frame_idx, bool) or not isinstance(frame_idx, int):
        raise TypeError(f"frame_idx must be int, got {type(frame_idx).__name__}")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise TypeError(f"fps must be int or float, got {type(fps).__name__}")
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be >= 0, got {frame_idx}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return float(frame_idx) / float(fps)
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="integration_helpers",
        description="Quick CLI for unit conversion, clock formatting, and frame timestamps.",
    )
    p.add_argument("--speed-kmh", type=float, default=None,
                   help="Speed in km/h to convert to m/s.")
    p.add_argument("--seconds", type=float, default=None,
                   help="Seconds to format as a game clock.")
    p.add_argument("--clock-fmt", type=str, default="MM:SS",
                   choices=["MM:SS", "HH:MM:SS", "M:SS"],
                   help="Clock format string.")
    p.add_argument("--frame", type=int, default=None,
                   help="Frame index to convert to a timestamp (seconds).")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help=f"Frames per second (default: {DEFAULT_FPS}).")
    return p
def _main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    printed = False
    if args.speed_kmh is not None:
        ms = kmh_to_ms(args.speed_kmh)
        print(f"{args.speed_kmh:g} km/h = {ms:.4f} m/s")
        printed = True
    if args.seconds is not None:
        print(f"{args.seconds:g}s -> {format_game_clock(args.seconds, args.clock_fmt)}")
        printed = True
    if args.frame is not None:
        ts = timestamp_from_frame(args.frame, args.fps)
        print(f"frame {args.frame} @ {args.fps:g} fps -> {ts:.4f}s "
              f"({format_game_clock(ts)})")
        printed = True
    if not printed:
        _build_parser().print_help()
        return 1
    return 0
if __name__ == "__main__":
    raise SystemExit(_main())