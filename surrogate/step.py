"""The update rule: what the network predicts, and how a state advances.

The network predicts two scalars per vehicle over one output step ``h``:

``abar``
    Mean acceleration. ``v`` follows by definition.

``ddelta_corr`` (per coupler, not per vehicle)
    What the trapezoid misses on *shape*. The kinematic estimate carries the
    consist's bulk motion to 0.7 mm per step but is off by ~29 mm on the shape
    of the train -- 2.1x the coupler half-slack -- because velocity completes
    close to a full oscillation within one step, and the endpoints of an
    oscillation do not pin down its integral.

    This correction is predicted in the *relative* coordinate because it does
    not survive in the absolute one. ``state`` is float32 and ``x`` runs to
    ~15 km, where the float32 step is 0.977 mm; the correction has a std of
    0.753 mm there, so a per-vehicle position target is mostly rounding error
    (signal/quantization 0.77, and the first smoke run duly learned nothing
    from it). The same quantity read off ``edge_dynamic``, on a 66 mm scale,
    has a ratio of 117,713. See ``SURROGATE_FORMULATION_NOTE.md``.

``z_brk`` and ``z_trac`` are not predicted at all. They obey
``zdot = (u - z)/tau`` with ``u`` piecewise-linear and known in advance, which
is exactly integrable; :func:`actuator_step` is that solution, verified against
the corpus at 3.6e-3 N on a 9.2e4 N scale (float32 storage noise, and 100x
better than an Euler step of the same lag).
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Channel order of ``state[..., 4]`` as ``simulator2.dataset`` writes it.
X, V, Z_BRK, Z_TRAC = 0, 1, 2, 3


def actuator_step(z: Tensor, u0: Tensor, u1: Tensor, tau: float, h: float) -> Tensor:
    """Exact solution of ``zdot = (u - z)/tau`` for ``u`` linear on ``[0, h]``.

    Integrating ``zdot + z/tau = u/tau`` with ``u(t) = u0 + (u1-u0) t/h``::

        z(h) = e^-r z0 + u0 (1 - e^-r) + (u1 - u0) [1 - tau (1 - e^-r)/h]

    with ``r = h/tau``. Linear interpolation of the command between output rows
    is exactly what the reference RHS does, so this is not an approximation of
    the generator -- it is the generator's own actuator model, solved.
    """
    e = torch.exp(torch.tensor(-h / tau, dtype=z.dtype, device=z.device))
    return e * z + u0 * (1.0 - e) + (u1 - u0) * (1.0 - tau * (1.0 - e) / h)


def node_targets(state_k: Tensor, state_k1: Tensor, h: float) -> Tensor:
    """Mean acceleration over the step -- the node head's target."""
    return (state_k1[..., V] - state_k[..., V]) / h


def edge_targets(
    delta_k: Tensor, delta_k1: Tensor, state_k: Tensor, state_k1: Tensor, h: float
) -> Tensor:
    """What the trapezoid misses on coupler stretch -- the edge head's target.

    ``d(delta)/dt = delta_dot`` exactly, so this mirrors the node case:
    trapezoid from the endpoint closing rates, then the residual. ``delta_k``
    and ``delta_k1`` must come from ``edge_dynamic``, not from differencing
    ``state[..., X]`` -- see the module docstring.
    """
    ddot_k = state_k[..., V][..., :-1] - state_k[..., V][..., 1:]
    ddot_k1 = state_k1[..., V][..., :-1] - state_k1[..., V][..., 1:]
    return delta_k1 - (delta_k + h * 0.5 * (ddot_k + ddot_k1))


def reconstruct_x(x_centroid: Tensor, delta: Tensor, l0: Tensor) -> Tensor:
    """Absolute positions from the consist centroid and the coupler stretches.

    ``delta_j = (x_j - x_{j+1}) - L0_j``, so walking backwards from any vehicle
    fixes every other. The centroid is used as the anchor rather than the lead
    vehicle because the trapezoid is accurate on *bulk* motion (0.7 mm per step)
    and not on any individual vehicle (~29 mm), and the centroid is the bulk.
    """
    gap = delta + l0                                   # [..., N-1]
    offs = torch.cumsum(gap, dim=-1)
    offs = torch.cat([torch.zeros_like(offs[..., :1]), offs], dim=-1)  # [..., N]
    offs = -offs
    return x_centroid[..., None] + offs - offs.mean(dim=-1, keepdim=True)


def advance(
    state: Tensor,
    delta: Tensor,
    abar: Tensor,
    ddelta_corr: Tensor,
    u0: Tensor,
    u1: Tensor,
    l0: Tensor,
    *,
    tau_brk: float,
    tau_trac: float,
    h: float,
) -> tuple[Tensor, Tensor]:
    """One output step, returning ``(state_k1, delta_k1)``.

    ``state`` ``[..., N, 4]``; ``delta``, ``ddelta_corr``, ``l0`` ``[..., N-1]``;
    ``abar`` ``[..., N]``; ``u0``/``u1`` ``[..., N, 2]`` as ``(u_trac, u_brk)``.
    """
    v = state[..., V]
    v1 = v + h * abar

    ddot = v[..., :-1] - v[..., 1:]
    ddot1 = v1[..., :-1] - v1[..., 1:]
    delta1 = delta + h * 0.5 * (ddot + ddot1) + ddelta_corr

    # Bulk motion by trapezoid on the centroid; shape from the new deltas.
    c = state[..., X].mean(dim=-1)
    c1 = c + h * 0.5 * (v.mean(dim=-1) + v1.mean(dim=-1))
    x1 = reconstruct_x(c1, delta1, l0)

    zb1 = actuator_step(state[..., Z_BRK], u0[..., 1], u1[..., 1], tau_brk, h)
    zt1 = actuator_step(state[..., Z_TRAC], u0[..., 0], u1[..., 0], tau_trac, h)
    return torch.stack([x1, v1, zb1, zt1], dim=-1), delta1


def coupler_state(state: Tensor, edge_static: Tensor) -> Tensor:
    """``[delta, delta_dot, F_cpl]`` per coupler, from the node state.

    These live in ``edge_dynamic`` in the corpus but are *derived*, so handing
    them to the network is free rather than leakage: at inference they are
    computed the same way. It hands the model the slack deadband instead of
    making it rediscover the discontinuity from raw positions.

    ``edge_static`` columns are ``L0, slack_half, k_draft, c_draft, k_buff,
    c_buff``.
    """
    x, v = state[..., X], state[..., V]
    delta = (x[..., :-1] - x[..., 1:]) - edge_static[..., 0]
    delta_dot = v[..., :-1] - v[..., 1:]
    slack = edge_static[..., 1]
    zero = torch.zeros_like(delta)
    f_draft = edge_static[..., 2] * (delta - slack) + edge_static[..., 3] * delta_dot
    f_buff = edge_static[..., 4] * (delta + slack) + edge_static[..., 5] * delta_dot
    f = torch.where(delta > slack, f_draft, zero)
    f = torch.where(delta < -slack, f_buff, f)
    return torch.stack([delta, delta_dot, f], dim=-1)
