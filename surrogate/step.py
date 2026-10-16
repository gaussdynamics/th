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
better than an Euler step of the same lag). Under the v3 air brake the lag is
per vehicle and switches between application and release time constants;
:func:`brake_step` handles that, and matches a v3 corpus to 3.5e-4 N rms.

**The draft gear state** (v3 corpora) is not predicted either. Under the
friction draft gear each coupler carries a gear force ``z``; the coupler force
is ``z`` clamped to the loading/unloading envelope at the current deflection,
which :class:`GearLaw` evaluates exactly. ``z`` advances by the elastic return
map on the predicted stretch, :meth:`GearLaw.advance`, which misses the stored
next force by 1.7 kN rms on a 219 kN scale given the true stretch. Without the
state, the best stateless predictor of the force from stretch and closing rate
misses by 52 kN rms (``SIMULATOR_VALIDATION_NOTE.md``, 2026-10-12).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

#: Channel order of ``state[..., 4]`` as ``simulator2.dataset`` writes it.
X, V, Z_BRK, Z_TRAC = 0, 1, 2, 3


@dataclass(frozen=True)
class GearLaw:
    """The friction draft gear of ``simulator2.draft_gear``, on carried state.

    ``table`` is the per-gear ``(deflection, loading, unloading)`` table; the
    constants are the RHS's (``TorchScenarioBatch.draft_gear_*``), which the v3
    corpus does not store because they are the module defaults.
    """

    table: Tensor
    n_gears: float
    k_transition: float
    k_lock: float

    @classmethod
    def default(cls, device: str | torch.device = "cpu",
                dtype: torch.dtype = torch.float32) -> "GearLaw":
        from simulator2.draft_gear import (DEFAULT_K_LOCK, DEFAULT_K_TRANSITION,
                                           N_GEARS, load_table)
        return cls(torch.as_tensor(load_table(), dtype=dtype, device=device),
                   float(N_GEARS), DEFAULT_K_TRANSITION, DEFAULT_K_LOCK)

    def deflection(self, delta: Tensor, slack_half: Tensor) -> Tensor:
        """Connection deflection beyond the coupler slack, zero inside it."""
        zero = torch.zeros_like(delta)
        return torch.where(delta > slack_half, delta - slack_half,
                           torch.where(delta < -slack_half, delta + slack_half, zero))

    def envelope(self, delta: Tensor, slack_half: Tensor) -> tuple[Tensor, Tensor]:
        """Force bounds ``(lo, hi)`` at stretch ``delta``; both zero in the slack."""
        from simulator2.draft_gear import _envelope_torch
        e = self.deflection(delta, slack_half)
        lo, hi, _, _ = _envelope_torch(e, self.table.to(e.dtype), self.n_gears, self.k_lock)
        return lo, hi

    def advance(self, z: Tensor, delta0: Tensor, delta1: Tensor,
                slack_half: Tensor) -> Tensor:
        """Elastic return map over one step: stick at ``k_t``, slide on the envelope.

        Exact when the deflection moves monotonically within the step; the
        corpus's 0.25 s step is not always that, which is the 1.7 kN floor.
        """
        e0 = self.deflection(delta0, slack_half)
        e1 = self.deflection(delta1, slack_half)
        lo, hi = self.envelope(delta1, slack_half)
        return torch.minimum(torch.maximum(z + self.k_transition * (e1 - e0), lo), hi)


def actuator_step(z: Tensor, u0: Tensor, u1: Tensor, tau: float | Tensor,
                  h: float) -> Tensor:
    """Exact solution of ``zdot = (u - z)/tau`` for ``u`` linear on ``[0, h]``.

    Integrating ``zdot + z/tau = u/tau`` with ``u(t) = u0 + (u1-u0) t/h``::

        z(h) = e^-r z0 + u0 (1 - e^-r) + (u1 - u0) [1 - tau (1 - e^-r)/h]

    with ``r = h/tau``. Linear interpolation of the command between output rows
    is exactly what the reference RHS does, so this is not an approximation of
    the generator -- it is the generator's own actuator model, solved.
    ``tau`` may be a tensor broadcasting against ``z`` (one lag per vehicle).
    """
    e = torch.exp(torch.as_tensor(-h / tau, dtype=z.dtype, device=z.device))
    return e * z + u0 * (1.0 - e) + (u1 - u0) * (1.0 - tau * (1.0 - e) / h)


def brake_step(z: Tensor, u0: Tensor, u1: Tensor, tau_apply: Tensor,
               tau_release: Tensor, h: float) -> Tensor:
    """:func:`actuator_step` with the v3 air brake's two time constants.

    The RHS uses the release constant while the command is below the cylinder
    force. Choosing it once per step from the mid-step command reproduces a v3
    corpus to float32 storage noise (3.5e-4 N rms; sub-stepping 5x or 25x
    changes nothing), because the 0.25 s application lag never lets the
    switch fall inside a step.
    """
    tau = torch.where(0.5 * (u0 + u1) < z, tau_release, tau_apply)
    return actuator_step(z, u0, u1, tau, h)


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
    tau_brk: float | Tensor,
    tau_trac: float,
    h: float,
    tau_brk_release: Tensor | None = None,
    gear: Tensor | None = None,
    law: GearLaw | None = None,
    slack_half: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """One output step, returning ``(state_k1, delta_k1, gear_k1)``.

    ``state`` ``[..., N, 4]``; ``delta``, ``ddelta_corr``, ``l0`` ``[..., N-1]``;
    ``abar`` ``[..., N]``; ``u0``/``u1`` ``[..., N, 2]`` as ``(u_trac, u_brk)``.
    v3 corpora pass ``tau_brk``/``tau_brk_release`` per vehicle and the gear
    state ``gear`` ``[..., N-1]`` with its ``law`` and ``slack_half``; legacy
    corpora pass neither and get ``gear_k1 = None``.
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

    if tau_brk_release is None:
        zb1 = actuator_step(state[..., Z_BRK], u0[..., 1], u1[..., 1], tau_brk, h)
    else:
        zb1 = brake_step(state[..., Z_BRK], u0[..., 1], u1[..., 1], tau_brk,
                         tau_brk_release, h)
    zt1 = actuator_step(state[..., Z_TRAC], u0[..., 0], u1[..., 0], tau_trac, h)
    gear1 = None if gear is None else law.advance(gear, delta, delta1, slack_half)
    return torch.stack([x1, v1, zb1, zt1], dim=-1), delta1, gear1


def coupler_features(delta: Tensor, v: Tensor, edge_static: Tensor,
                     gear: Tensor | None = None, law: GearLaw | None = None) -> Tensor:
    """``[delta, delta_dot, F_cpl]`` from a *carried* stretch and the velocities.

    The rollout propagates ``delta`` directly rather than re-deriving it from
    positions, so this is the form the model sees at inference. Deriving it
    from ``x`` instead would reintroduce the ~1 mm float32 quantization that
    the edge formulation exists to avoid -- and at coupler stiffness that is up
    to 17.8 kN of noise on ``F_cpl``.

    With a gear state (v3) the force is the gear law's, and the gear state and
    its envelope ``lo``/``hi`` follow: ``[delta, delta_dot, F_cpl, z, lo, hi]``.
    Where ``z`` sits between ``lo`` and ``hi`` is where the gear is in its
    hysteresis loop, which stretch and closing rate alone do not tell.
    """
    delta_dot = v[..., :-1] - v[..., 1:]
    if gear is not None:
        lo, hi = law.envelope(delta, edge_static[..., 1])
        f = torch.minimum(torch.maximum(gear, lo), hi)
        return torch.stack([delta, delta_dot, f, gear, lo, hi], dim=-1)
    slack = edge_static[..., 1]
    zero = torch.zeros_like(delta)
    f_draft = edge_static[..., 2] * (delta - slack) + edge_static[..., 3] * delta_dot
    f_buff = edge_static[..., 4] * (delta + slack) + edge_static[..., 5] * delta_dot
    f = torch.where(delta > slack, f_draft, zero)
    f = torch.where(delta < -slack, f_buff, f)
    return torch.stack([delta, delta_dot, f], dim=-1)


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
    return coupler_features(delta, v, edge_static)
