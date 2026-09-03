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

from .constants import GRAVITY_MPS2


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


# Curvature resistance models. ``proxy_v2`` is the original notebook proxy and
# remains the default so existing results, fixtures and Chapter 4 are unchanged.
CURVATURE_MODELS = ("proxy_v2", "roeckl", "linear")

#: Tightest radius the empirical formulae are evaluated at (m). Roeckl's
#: denominators vanish at R = 55 m / 30 m; real track does not go near that
#: (the sharpest curve in the current corridor set is 80 m), but the RHS must
#: not be able to produce an infinity on a noisy curvature field.
MIN_CURVE_RADIUS_M = 60.0

#: AREMA: 0.8 lb per ton per degree of curve, with D = 1746.4 * kappa, giving
#: specific resistance 0.0004 * 1746.4 * kappa as a fraction of weight.
_AREMA_COEFF = 0.0004 * 1746.4  # ~0.6986, dimensionless


def curvature_specific_resistance(kappa: np.ndarray | float, model: str) -> np.ndarray:
    """Curve resistance as a fraction of vehicle weight, given curvature.

    ``roeckl`` is the standard two-branch empirical formula, in N/kN::

        w_c = 650 / (R - 55)    R >= 300 m
        w_c = 500 / (R - 30)    R <  300 m

    rewritten in terms of ``kappa = 1/R`` so ``kappa = 0`` (tangent track) is
    not a division by zero. Note it is genuinely **discontinuous at R = 300 m**
    -- 2.65 vs 1.85 N/kN -- which is a property of the published formula, not of
    this implementation. ``linear`` is the AREMA form, ``w_c ~ 0.6986 * kappa``,
    which tracks Roeckl within about 15 % over 300-2000 m and is smooth, so it
    is the better choice if gradient quality through the RHS matters.

    Both are **speed-independent**, which is the substantive difference from
    ``proxy_v2``: real curve resistance does not scale with ``v^2``.
    """
    kap = np.abs(np.asarray(kappa, dtype=float))
    kap = np.minimum(kap, 1.0 / MIN_CURVE_RADIUS_M)
    if model == "linear":
        return _AREMA_COEFF * kap
    if model == "roeckl":
        wide = 1e-3 * 650.0 * kap / (1.0 - 55.0 * kap)
        tight = 1e-3 * 500.0 * kap / (1.0 - 30.0 * kap)
        return np.where(kap <= 1.0 / 300.0, wide, tight)
    raise ValueError(f"{model!r} has no specific-resistance form")


def curvature_force_longitudinal(
    m_kg: float,
    v: float,
    x: float,
    route: RouteProfile,
    k_scale: float,
    model: str = "proxy_v2",
) -> float:
    """Longitudinal curvature resistance, opposing the direction of travel.

    ``model="proxy_v2"`` (default) is the original notebook proxy
    ``k * m * v^2 * |kappa|``. Its ``k`` is uncalibrated and its ``v^2`` shape
    does not match any standard curve-resistance model, so no single ``k`` is
    correct across the speed range -- one matching Roeckl at 15 m/s is 9x too
    small at 5 m/s and 2.8x too large at 25 m/s.

    ``model="roeckl"`` and ``model="linear"`` use the speed-independent
    empirical forms, where ``k_scale`` becomes a dimensionless multiplier on the
    standard formula: **1.0 is the textbook value**, which is what makes them
    calibrated rather than tunable.
    """
    if k_scale == 0.0:
        return 0.0
    kap = route.kappa_at(x)
    if model == "proxy_v2":
        mag = k_scale * m_kg * v * v * abs(kap)
    else:
        mag = k_scale * m_kg * GRAVITY_MPS2 * float(
            curvature_specific_resistance(kap, model)
        )
    return 0.0 if abs(v) < 1e-9 else mag * np.sign(v)
