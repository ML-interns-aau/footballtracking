from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class EnhancerConfig:
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)
    sharpen_amount: float = 1.0
    enable_adaptive_clahe: bool = True
    clahe_skip_std_threshold: float = 42.0
    enable_denoise: bool = False
    denoise_strength: int = 3


class FrameEnhancer:
    def __init__(
        self,
        config: EnhancerConfig | None = None,
        sharpen_kernel: np.ndarray | None = None,
    ) -> None:
        self.config = config or EnhancerConfig()
        self.sharpen_kernel = (
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
            if sharpen_kernel is None
            else sharpen_kernel
        )
        self.sharpen_kernel = self.sharpen_kernel * self.config.sharpen_amount
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=self.config.tile_grid_size,
        )

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty")

        working = frame
        if self.config.enable_denoise:
            working = cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                self.config.denoise_strength,
                self.config.denoise_strength,
                7,
                21,
            )

        sharpened = cv2.filter2D(working, -1, self.sharpen_kernel)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        if (
            not self.config.enable_adaptive_clahe
            or float(l_channel.std()) < self.config.clahe_skip_std_threshold
        ):
            lab[:, :, 0] = self._clahe.apply(l_channel)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
