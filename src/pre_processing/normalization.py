from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class NormalizationConfig:
    target_size: Tuple[int, int] = (1280, 720)
    apply_gray_world_white_balance: bool = False
    gamma: float = 1.0


class FrameNormalizer:
    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self.config = config or NormalizationConfig()
        self._gamma_table = self._build_gamma_table(self.config.gamma)

    def _build_gamma_table(self, gamma: float) -> np.ndarray | None:
        if gamma <= 0 or abs(gamma - 1.0) < 1e-6:
            return None
        inv_gamma = 1.0 / gamma
        return np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)],
            dtype=np.uint8,
        )

    def _apply_gray_world_white_balance(self, frame: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(frame.astype(np.float32))
        avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        if avg_b > 0:
            b *= avg_gray / avg_b
        if avg_g > 0:
            g *= avg_gray / avg_g
        if avg_r > 0:
            r *= avg_gray / avg_r
        balanced = cv2.merge((b, g, r))
        return np.clip(balanced, 0, 255).astype(np.uint8)

    def _apply_gamma(self, frame: np.ndarray, gamma: float) -> np.ndarray:
        if self._gamma_table is None:
            return frame
        return cv2.LUT(frame, self._gamma_table)

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty")

        normalized = cv2.resize(frame, self.config.target_size, interpolation=cv2.INTER_AREA)
        if self.config.apply_gray_world_white_balance:
            normalized = self._apply_gray_world_white_balance(normalized)
        normalized = self._apply_gamma(normalized, self.config.gamma)
        return normalized
