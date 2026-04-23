"""Longitudinal resistance, grade, and nonlinear slack coupler forces."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .constants import GRAVITY_MPS2


def davis_resistance_longitudinal(
    v: np.ndarray | float, A: float, B: float, C: float
) -> np.ndarray:
    """Longitudinal resistance [N] opposing motion (vectorized)."""
    v = np.asarray(v, dtype=float)
    sgn = np.where(np.abs(v) < 1e-9, 0.0, np.sign(v))
    mag = A + B * np.abs(v) + C * v * v
    return mag * sgn


def grade_force(
    m_kg: float, x_m: float, theta_at_x: Callable[[float], float]
) -> float:
    """Grade resistance ``G = m g sin(theta(x))``; ``theta`` in radians."""
    th = float(theta_at_x(x_m))
    return m_kg * GRAVITY_MPS2 * np.sin(th)


def coupler_force_slack_asymmetric(
    delta: np.ndarray,
    delta_dot: np.ndarray,
    slack_half: float,
    k_draft: float,
    c_draft: float,
    k_buff: float,
    c_buff: float,
) -> np.ndarray:
    """Force on rear car; vectorized (notebook implementation)."""
    delta = np.asarray(delta, float)
    delta_dot = np.asarray(delta_dot, float)
    s = float(slack_half)
    F = np.zeros_like(delta)
    mask_d = delta > s
    e_d = delta[mask_d] - s
    F[mask_d] = k_draft * e_d + c_draft * delta_dot[mask_d]
    mask_b = delta < -s
    e_b = delta[mask_b] + s
    F[mask_b] = k_buff * e_b + c_buff * delta_dot[mask_b]
    return F


def traction_ramp(
    t: float, t0: float = 5.0, t1: float = 20.0, F_max: float = 180_000.0
) -> float:
    """Smooth S-shaped traction command in time (Stage 1 notebook)."""
    if t < t0:
        return 0.0
    if t > t1:
        return F_max
    u = (t - t0) / (t1 - t0)
    return F_max * (3 * u * u - 2 * u * u * u)
