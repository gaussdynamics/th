"""Tensor train state ``H`` (nodes) and ``E`` (edges), pack/unpack, and padding masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .forces import coupler_force_slack_asymmetric
from .params import CouplerParameters, VehicleParameters
from .state_schema import D_EDGE, D_NODE, EdgeChannel, NodeChannel


@dataclass
class TensorTrainState:
    """First-class tensorized train state (NumPy).

    ``H`` holds per-vehicle features; ``E`` holds per-coupler features between
    consecutive vehicles (same ordering as the notebook).
    """

    H: np.ndarray
    E: np.ndarray


def build_node_tensor(
    vehicles: Sequence[VehicleParameters],
    x: np.ndarray,
    v: np.ndarray,
    z_brk: np.ndarray,
    z_trac: np.ndarray,
) -> np.ndarray:
    """Assemble ``H`` with shape ``[N, d_node]`` from dynamic states and static parameters."""
    n = len(vehicles)
    H = np.zeros((n, D_NODE), dtype=float)
    H[:, NodeChannel.X] = x
    H[:, NodeChannel.V] = v
    H[:, NodeChannel.Z_BRK] = z_brk
    H[:, NodeChannel.Z_TRAC] = z_trac
    for i, vp in enumerate(vehicles):
        H[i, NodeChannel.MASS_KG] = vp.mass_kg
        H[i, NodeChannel.DAVIS_A] = vp.davis_A
        H[i, NodeChannel.DAVIS_B] = vp.davis_B
        H[i, NodeChannel.DAVIS_C] = vp.davis_C
        H[i, NodeChannel.CAN_TRACTION] = 1.0 if vp.can_traction else 0.0
        H[i, NodeChannel.F_TRAC_MAX_N] = vp.F_trac_max_N
        H[i, NodeChannel.F_BRK_MAX_N] = vp.F_brk_max_N
    return H


def build_edge_tensor(
    couplers: Sequence[CouplerParameters],
    delta: np.ndarray,
    delta_dot: np.ndarray,
    F_cpl: np.ndarray,
) -> np.ndarray:
    """Assemble ``E`` with shape ``[N-1, d_edge]``."""
    m = len(couplers)
    E = np.zeros((m, D_EDGE), dtype=float)
    E[:, EdgeChannel.DELTA] = delta
    E[:, EdgeChannel.DELTA_DOT] = delta_dot
    E[:, EdgeChannel.F_CPL] = F_cpl
    for j, c in enumerate(couplers):
        E[j, EdgeChannel.L0_M] = c.L0_m
        E[j, EdgeChannel.SLACK_HALF_M] = c.slack_half_m
        E[j, EdgeChannel.K_DRAFT] = c.k_draft
        E[j, EdgeChannel.C_DRAFT] = c.c_draft
        E[j, EdgeChannel.K_BUFF] = c.k_buff
        E[j, EdgeChannel.C_BUFF] = c.c_buff
    return E


def compute_edge_tensor_from_nodes(
    H: np.ndarray, couplers: Sequence[CouplerParameters]
) -> np.ndarray:
    """Kinematic coupler extensions and algebraic coupler forces from node positions/speeds."""
    n = H.shape[0]
    if n < 2:
        return np.zeros((0, D_EDGE), dtype=float)
    x = H[:, NodeChannel.X]
    v = H[:, NodeChannel.V]
    m = n - 1
    delta = np.zeros(m, dtype=float)
    delta_dot = np.zeros(m, dtype=float)
    F_cpl = np.zeros(m, dtype=float)
    for j in range(m):
        c = couplers[j]
        dlt = (x[j] - x[j + 1]) - c.L0_m
        ddv = v[j] - v[j + 1]
        delta[j] = dlt
        delta_dot[j] = ddv
        F_cpl[j] = float(
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
    return build_edge_tensor(couplers, delta, delta_dot, F_cpl)


def unpack_to_tensor_state(
    y: np.ndarray,
    vehicles: Sequence[VehicleParameters],
    couplers: Sequence[CouplerParameters],
) -> TensorTrainState:
    """Unpack extended flat state ``y`` of length ``4N`` into ``TensorTrainState``."""
    n = len(vehicles)
    if y.shape[0] != 4 * n:
        raise ValueError(f"Expected y of length {4*n}, got {y.shape[0]}")
    x = y[0:n]
    v = y[n : 2 * n]
    z_brk = y[2 * n : 3 * n]
    z_trac = y[3 * n : 4 * n]
    H = build_node_tensor(vehicles, x, v, z_brk, z_trac)
    E = compute_edge_tensor_from_nodes(H, couplers)
    return TensorTrainState(H=H, E=E)


def extract_dynamic_node_tensor(state: TensorTrainState) -> np.ndarray:
    """Dynamic node features only: ``[N, 4]`` columns ``x, v, z_brk, z_trac``."""
    H = state.H
    out = np.stack(
        [
            H[:, NodeChannel.X],
            H[:, NodeChannel.V],
            H[:, NodeChannel.Z_BRK],
            H[:, NodeChannel.Z_TRAC],
        ],
        axis=1,
    )
    return out


def pack_dynamic_state_from_tensor_state(state: TensorTrainState) -> np.ndarray:
    """Flatten dynamic state for ``solve_ivp``: ``[x; v; z_brk; z_trac]``."""
    dyn = extract_dynamic_node_tensor(state)
    return np.concatenate(
        [dyn[:, 0], dyn[:, 1], dyn[:, 2], dyn[:, 3]], dtype=float
    )


def make_node_mask(n: int, n_max: int) -> np.ndarray:
    """Boolean mask ``[n_max]`` with first ``n`` entries True (padding for future batching)."""
    m = np.zeros(n_max, dtype=bool)
    m[: min(n, n_max)] = True
    return m


def make_edge_mask(n: int, n_max: int) -> np.ndarray:
    """Boolean mask for edges ``[n_max - 1]``; valid couplers are ``0 .. n-2``."""
    if n_max < 2:
        return np.zeros(max(0, n_max - 1), dtype=bool)
    m = np.zeros(n_max - 1, dtype=bool)
    n_edge = max(0, n - 1)
    m[: min(n_edge, n_max - 1)] = True
    return m
