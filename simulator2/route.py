"""Route profile and curvature proxy force (notebook-aligned)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RouteProfile:
    """Piecewise-linear ``sin(theta(x))`` and optional ``kappa(x)`` along track."""

    s_nodes_m: np.ndarray
    sin_theta_nodes: np.ndarray
    kappa_nodes: Optional[np.ndarray] = None

    def sin_theta_at(self, x: float) -> float:
        return float(np.interp(x, self.s_nodes_m, self.sin_theta_nodes))

    def kappa_at(self, x: float) -> float:
        if self.kappa_nodes is None:
            return 0.0
        return float(np.interp(x, self.s_nodes_m, self.kappa_nodes))


def curvature_force_longitudinal(
    m_kg: float, v: float, x: float, route: RouteProfile, k_scale: float
) -> float:
    """Longitudinal curvature resistance proxy (same as notebook)."""
    if k_scale == 0.0:
        return 0.0
    kap = route.kappa_at(x)
    mag = k_scale * m_kg * v * v * abs(kap)
    return 0.0 if abs(v) < 1e-9 else mag * np.sign(v)
