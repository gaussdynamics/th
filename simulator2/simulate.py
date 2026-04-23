"""Time integration: classical ``simulate_train`` and tensorized rollout."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .forces import coupler_force_slack_asymmetric
from .io_types import ExtendedTrainScenario, TensorSimulationResult
from .params import TrainSimulationConfig
from .rhs import train_rhs_tensorized
from .state_schema import D_EDGE, D_NODE, edge_channel_dict, node_channel_dict
from .tensor_state import compute_edge_tensor_from_nodes, unpack_to_tensor_state


def simulate_train(
    y0: np.ndarray,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    cfg: TrainSimulationConfig,
) -> Dict[str, Any]:
    """Integrate generic RHS and post-process coupler forces (notebook API)."""
    sol = solve_ivp(
        fun=rhs,
        t_span=cfg.t_span,
        y0=y0,
        t_eval=cfg.t_eval,
        rtol=cfg.rtol,
        atol=cfg.atol,
        method=cfg.method,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    n = len(cfg.vehicles)
    n_state = sol.y.shape[0]
    if n_state < 2 * n:
        raise ValueError("State too small for vehicle count")
    xh = sol.y[0:n, :]
    vh = sol.y[n : 2 * n, :]
    fhist = np.zeros((n - 1, sol.t.size))
    for k in range(sol.t.size):
        x = xh[:, k]
        v = vh[:, k]
        for j in range(n - 1):
            c = cfg.couplers[j]
            dlt = (x[j] - x[j + 1]) - c.L0_m
            ddv = v[j] - v[j + 1]
            fhist[j, k] = coupler_force_slack_asymmetric(
                np.array([dlt]),
                np.array([ddv]),
                c.slack_half_m,
                c.k_draft,
                c.c_draft,
                c.k_buff,
                c.c_buff,
            )[0]
    return {"sol": sol, "F_coupler": fhist, "N": n}


def simulate_train_tensorized(
    scenario: ExtendedTrainScenario,
) -> TensorSimulationResult:
    """Integrate extended dynamics with ``train_rhs_tensorized`` and record ``H_hist``, ``E_hist``."""
    vehicles = scenario.vehicles
    couplers = scenario.couplers
    n = len(vehicles)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return train_rhs_tensorized(
            t,
            y,
            vehicles,
            couplers,
            scenario.route,
            scenario.u_trac_cmd,
            scenario.u_brk_cmd,
            scenario.k_curv_scale,
            scenario.tau_brk_s,
            scenario.tau_trac_s,
            scenario.p_max_w,
        )

    sol = solve_ivp(
        fun=rhs,
        t_span=scenario.t_span,
        y0=scenario.y0,
        t_eval=scenario.t_eval,
        rtol=scenario.rtol,
        atol=scenario.atol,
        method=scenario.method,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    t_points = sol.t
    nt = t_points.size
    h_hist = np.zeros((nt, n, D_NODE), dtype=float)
    e_hist = np.zeros((nt, max(0, n - 1), D_EDGE), dtype=float)

    for k in range(nt):
        yk = sol.y[:, k]
        state = unpack_to_tensor_state(yk, vehicles, couplers)
        h_hist[k] = state.H
        e_hist[k] = compute_edge_tensor_from_nodes(state.H, couplers)

    metadata = {
        "N": n,
        "d_node": D_NODE,
        "d_edge": D_EDGE,
        "node_channels": node_channel_dict(),
        "edge_channels": edge_channel_dict(),
        **scenario.extra_metadata,
    }

    return TensorSimulationResult(
        t=t_points,
        y_flat=sol.y,
        H_hist=h_hist,
        E_hist=e_hist,
        sol=sol,
        metadata=metadata,
    )
