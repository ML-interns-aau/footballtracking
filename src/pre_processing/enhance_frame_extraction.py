import cv2
import numpy as np


class FrameEnhancer:
    def __init__(
        self,
        target_size=(1280, 720),
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        sharpen_kernel=None,
    ):
        self.target_size = target_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.sharpen_kernel = (
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            if sharpen_kernel is None
            else sharpen_kernel
        )
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

    def enhance(self, frame):
        # Resize to target AI input before applying heavier operations.
        frame = cv2.resize(frame, self.target_size)
        sharpened = cv2.filter2D(frame, -1, self.sharpen_kernel)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance_frame_production(frame):
    """Backward-compatible wrapper."""
    return FrameEnhancer().enhance(frame)
