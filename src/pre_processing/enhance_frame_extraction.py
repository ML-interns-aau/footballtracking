from .enhancer import EnhancerConfig, FrameEnhancer


def enhance_frame_production(frame):
    """Backward-compatible wrapper."""
    return FrameEnhancer(EnhancerConfig()).enhance(frame)
