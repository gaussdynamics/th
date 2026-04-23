"""Train ODE right-hand sides: classical and tensorized (extended model)."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .constants import DEFAULT_V_EPS
from .forces import (
    coupler_force_slack_asymmetric,
    davis_resistance_longitudinal,
    grade_force,
)
from .params import CouplerParameters, VehicleParameters
from .route import RouteProfile, curvature_force_longitudinal
from .state_schema import EdgeChannel, NodeChannel
from .tensor_state import compute_edge_tensor_from_nodes, unpack_to_tensor_state


def train_rhs_n(
    t: float,
    y: np.ndarray,
    vehicles: Sequence[VehicleParameters],
    couplers: Sequence[CouplerParameters],
    route: RouteProfile,
    F_trac_cmd: Callable[[float, int], float],
    F_brk_cmd: Callable[[float, int], float],
    k_curv_scale: float,
) -> np.ndarray:
    """RHS for ``y = [x (N), v (N)]`` (notebook ``train_rhs_n``)."""
    n = len(vehicles)
    x = y[0:n]
    v = y[n : 2 * n]
    dv = np.zeros(n, dtype=float)
    f_couple = np.zeros(n - 1, dtype=float)
    for j in range(n - 1):
        c = couplers[j]
        dlt = (x[j] - x[j + 1]) - c.L0_m
        ddv = v[j] - v[j + 1]
        f_couple[j] = float(
            coupler_force_slack_asymmetric(
                np.array([dlt]),
                np.array([ddv]),
                c.slack_half_m,
                c.k_draft,
                c.c_draft,
                c.k_buff,
                c.c_buff,
            )[0]
        )
    for i in range(n):
        vp = vehicles[i]
        fi_in = f_couple[i - 1] if i > 0 else 0.0
        fi_out = f_couple[i] if i < n - 1 else 0.0
        r = float(
            davis_resistance_longitudinal(v[i], vp.davis_A, vp.davis_B, vp.davis_C)
        )
        g = grade_force(vp.mass_kg, float(x[i]), route.sin_theta_at)
        ccur = curvature_force_longitudinal(
            vp.mass_kg, float(v[i]), float(x[i]), route, k_curv_scale
        )
        f_tr = min(F_trac_cmd(t, i), vp.F_trac_max_N) if vp.can_traction else 0.0
        f_tr = float(f_tr)
        f_br = min(F_brk_cmd(t, i), vp.F_brk_max_N)
        f_br = float(max(f_br, 0.0))
        dv[i] = (f_tr - f_br - r - g - ccur + fi_in - fi_out) / vp.mass_kg
    return np.concatenate([v, dv])


def train_rhs_extended(
    t: float,
    y: np.ndarray,
    vehicles: Sequence[VehicleParameters],
    couplers: Sequence[CouplerParameters],
    route: RouteProfile,
    u_trac_cmd: Callable[[float, int], float],
    u_brk_cmd: Callable[[float, int], float],
    k_curv_scale: float,
    tau_brk: float,
    tau_trac: float,
    p_max_w: float,
    v_eps: float = DEFAULT_V_EPS,
) -> np.ndarray:
    """Extended RHS with first-order traction/brake buildup and power cap (notebook)."""
    n = len(vehicles)
    x = y[0:n]
    v = y[n : 2 * n]
    z_brk = y[2 * n : 3 * n]
    z_trac = y[3 * n : 4 * n]
    dx = v
    f_couple = np.zeros(n - 1, dtype=float)
    for j in range(n - 1):
        c = couplers[j]
        dlt = (x[j] - x[j + 1]) - c.L0_m
        ddv = v[j] - v[j + 1]
        f_couple[j] = float(
            coupler_force_slack_asymmetric(
                np.array([dlt]),
                np.array([ddv]),
                c.slack_half_m,
                c.k_draft,
                c.c_draft,
                c.k_buff,
                c.c_buff,
            )[0]
        )
    dv = np.zeros(n, dtype=float)
    dz_brk = np.zeros(n, dtype=float)
    dz_trac = np.zeros(n, dtype=float)
    for i in range(n):
        vp = vehicles[i]
        ut = u_trac_cmd(t, i) if vp.can_traction else 0.0
        ub = u_brk_cmd(t, i)
        dz_trac[i] = (float(ut) - float(z_trac[i])) / tau_trac
        dz_brk[i] = (float(ub) - float(z_brk[i])) / tau_brk
        fi_in = f_couple[i - 1] if i > 0 else 0.0
        fi_out = f_couple[i] if i < n - 1 else 0.0
        r = float(
            davis_resistance_longitudinal(v[i], vp.davis_A, vp.davis_B, vp.davis_C)
        )
        g = grade_force(vp.mass_kg, float(x[i]), route.sin_theta_at)
        ccur = curvature_force_longitudinal(
            vp.mass_kg, float(v[i]), float(x[i]), route, k_curv_scale
        )
        zt = float(max(z_trac[i], 0.0))
        zb = float(max(z_brk[i], 0.0))
        f_tr = 0.0
        if vp.can_traction:
            f_tr = min(zt, p_max_w / max(abs(float(v[i])), v_eps))
        f_br = min(zb, vp.F_brk_max_N)
        dv[i] = (f_tr - f_br - r - g - ccur + fi_in - fi_out) / vp.mass_kg
    return np.concatenate([dx, dv, dz_brk, dz_trac])


def train_rhs_tensorized(
    t: float,
    y: np.ndarray,
    vehicles: Sequence[VehicleParameters],
    couplers: Sequence[CouplerParameters],
    route: RouteProfile,
    u_trac_cmd: Callable[[float, int], float],
    u_brk_cmd: Callable[[float, int], float],
    k_curv_scale: float,
    tau_brk: float,
    tau_trac: float,
    p_max_w: float,
    v_eps: float = DEFAULT_V_EPS,
) -> np.ndarray:
    """Same dynamics as ``train_rhs_extended``, expressed via ``H`` and ``E`` tensors.

    Steps: unpack ``y`` → ``TensorTrainState``; rebuild ``E`` from nodes; compute actuator
    derivatives; assemble per-vehicle forces using coupler forces from ``E``; pack
    derivatives of ``(x, v, z_brk, z_trac)``.
    """
    state = unpack_to_tensor_state(y, vehicles, couplers)
    h = state.H
    # Refresh edge kinematics and forces from current node tensor
    e = compute_edge_tensor_from_nodes(h, couplers)
    n = len(vehicles)
    f_cpl = e[:, EdgeChannel.F_CPL]
    dv = np.zeros(n, dtype=float)
    dz_brk = np.zeros(n, dtype=float)
    dz_trac = np.zeros(n, dtype=float)
    dx = h[:, NodeChannel.V].copy()

    for i in range(n):
        vp = vehicles[i]
        ut = u_trac_cmd(t, i) if vp.can_traction else 0.0
        ub = u_brk_cmd(t, i)
        z_trac_i = h[i, NodeChannel.Z_TRAC]
        z_brk_i = h[i, NodeChannel.Z_BRK]
        dz_trac[i] = (float(ut) - float(z_trac_i)) / tau_trac
        dz_brk[i] = (float(ub) - float(z_brk_i)) / tau_brk

        fi_in = f_cpl[i - 1] if i > 0 else 0.0
        fi_out = f_cpl[i] if i < n - 1 else 0.0

        vi = float(h[i, NodeChannel.V])
        xi = float(h[i, NodeChannel.X])
        r = float(
            davis_resistance_longitudinal(
                vi,
                h[i, NodeChannel.DAVIS_A],
                h[i, NodeChannel.DAVIS_B],
                h[i, NodeChannel.DAVIS_C],
            )
        )
        g = grade_force(float(h[i, NodeChannel.MASS_KG]), xi, route.sin_theta_at)
        ccur = curvature_force_longitudinal(
            float(h[i, NodeChannel.MASS_KG]), vi, xi, route, k_curv_scale
        )
        zt = float(max(z_trac_i, 0.0))
        zb = float(max(z_brk_i, 0.0))
        f_tr = 0.0
        if vp.can_traction:
            f_tr = min(zt, p_max_w / max(abs(vi), v_eps))
        f_br = min(zb, vp.F_brk_max_N)
        dv[i] = (f_tr - f_br - r - g - ccur + fi_in - fi_out) / vp.mass_kg

    dy_dyn = np.concatenate([dx, dv, dz_brk, dz_trac])
    return dy_dyn
