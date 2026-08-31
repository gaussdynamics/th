"""Route profile and curvature proxy force (notebook-aligned).

``RouteProfile`` is the arc-length route field the dynamics sample at each
vehicle position. ``v_max_nodes`` (the speed limit field) was added for the
real-terrain pipeline; it is *not* read by the open-loop dynamics RHS — it is
exogenous information for the controller / speed-limit logic — so adding it does
not change any existing simulation result. ``RouteProfile.from_tensor_npz`` loads
the route tensor emitted by the route-generation pipeline
(``route_s``/``route_sin_theta``/``route_kappa``/``route_vmax``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RouteProfile:
    """Piecewise-linear ``sin(theta(x))``, optional ``kappa(x)`` and ``v_max(x)``."""

    s_nodes_m: np.ndarray
    sin_theta_nodes: np.ndarray
    kappa_nodes: Optional[np.ndarray] = None
    v_max_nodes: Optional[np.ndarray] = None

    def sin_theta_at(self, x: float) -> float:
        return float(np.interp(x, self.s_nodes_m, self.sin_theta_nodes))

    def kappa_at(self, x: float) -> float:
        if self.kappa_nodes is None:
            return 0.0
        return float(np.interp(x, self.s_nodes_m, self.kappa_nodes))

    def v_max_at(self, x: float) -> float:
        """Speed limit (m/s) at chainage ``x``; ``+inf`` if no limit field present."""
        if self.v_max_nodes is None:
            return float("inf")
        return float(np.interp(x, self.s_nodes_m, self.v_max_nodes))

    @classmethod
    def from_tensor_npz(cls, path) -> "RouteProfile":
        """Build a RouteProfile from a route-tensor ``.npz``.

        Expects the named arrays the route-generation pipeline emits:
        ``route_s`` (m), ``route_sin_theta``, ``route_kappa`` (1/m),
        ``route_vmax`` (m/s). ``route_kappa``/``route_vmax`` are optional.
        Works on both ``route_profiles/<name>.npz`` and the export bundle's
        ``routes/<name>.npz`` (same schema).
        """
        with np.load(path) as d:
            if "route_s" not in d or "route_sin_theta" not in d:
                raise ValueError(
                    f"{path!r} is not a route tensor (need route_s, route_sin_theta; "
                    f"found {list(d.keys())})."
                )
            s = np.asarray(d["route_s"], dtype=float)
            sin_theta = np.asarray(d["route_sin_theta"], dtype=float)
            kappa = np.asarray(d["route_kappa"], dtype=float) if "route_kappa" in d else None
            v_max = np.asarray(d["route_vmax"], dtype=float) if "route_vmax" in d else None
        return cls(s_nodes_m=s, sin_theta_nodes=sin_theta,
                   kappa_nodes=kappa, v_max_nodes=v_max)


def load_route_profile(path) -> RouteProfile:
    """Convenience wrapper for :meth:`RouteProfile.from_tensor_npz`."""
    return RouteProfile.from_tensor_npz(path)


def curvature_force_longitudinal(
    m_kg: float, v: float, x: float, route: RouteProfile, k_scale: float
) -> float:
    """Longitudinal curvature resistance proxy (same as notebook)."""
    if k_scale == 0.0:
        return 0.0
    kap = route.kappa_at(x)
    mag = k_scale * m_kg * v * v * abs(kap)
    return 0.0 if abs(v) < 1e-9 else mag * np.sign(v)
