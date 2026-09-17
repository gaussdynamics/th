"""Centralized color and normalization mappings for animation channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import BLUE_B, BLUE_E, GREEN_B, RED_B, WHITE, color_gradient


@dataclass(frozen=True)
class ChannelNorm:
    vmin: float
    vmax: float
    diverging: bool = False

    def normalize(self, value: float) -> float:
        if self.vmax <= self.vmin:
            return 0.5
        return max(0.0, min(1.0, (value - self.vmin) / (self.vmax - self.vmin)))


class ColorMapping:
    """Consistent mapping between physical/tensor quantities and colors."""

    def __init__(self, tensor_slice: np.ndarray, coupler_forces: np.ndarray, velocities: np.ndarray) -> None:
        force_abs = float(np.max(np.abs(coupler_forces))) if coupler_forces.size else 1.0
        vel_abs = float(np.max(np.abs(velocities))) if velocities.size else 1.0
        self.force_norm = ChannelNorm(vmin=-force_abs, vmax=force_abs, diverging=True)
        self.velocity_norm = ChannelNorm(vmin=-vel_abs, vmax=vel_abs, diverging=True)
        tensor_min = float(np.min(tensor_slice))
        tensor_max = float(np.max(tensor_slice))
        self.tensor_norm = ChannelNorm(vmin=tensor_min, vmax=tensor_max, diverging=False)

        self._diverging_colors = color_gradient([BLUE_E, WHITE, RED_B], 33)
        self._sequential_colors = color_gradient([BLUE_B, GREEN_B], 33)

    def diverging_color(self, value: float, norm: ChannelNorm | None = None):
        n = self.force_norm if norm is None else norm
        alpha = n.normalize(value)
        return self._diverging_colors[int(round(alpha * (len(self._diverging_colors) - 1)))]

    def sequential_color(self, value: float, norm: ChannelNorm | None = None):
        n = self.tensor_norm if norm is None else norm
        alpha = n.normalize(value)
        return self._sequential_colors[int(round(alpha * (len(self._sequential_colors) - 1)))]
