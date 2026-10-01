"""Friction draft gear for the vehicle connections.

Replaces the Kelvin-Voigt coupler (``forces.coupler_force_slack_asymmetric``)
when a batch sets ``coupler_model = "draft_gear"``. Written for the
international LTD benchmark (Spiryagin, Wu & Cole 2017, section 3.1), whose
draft gear is specified only by a drop-hammer cross plot (their Fig. 3),
digitized into ``data/draft_gear_benchmark.csv`` by
``scripts/digitize_draft_gear.py``.

**Connection.** A vehicle connection is the coupler (or drawbar) slack plus
two draft gears in series, one in each vehicle's end (benchmark Fig. 2). With
equal gears in series each carries the connection force and deflects half the
travel. The benchmark coupler has 10 mm total slack (``slack_half = 5 mm``);
a drawbar has none. The gear closes in both draft and buff (the yoke
arrangement), so the characteristic is odd in the connection deflection
``e``. Beyond full travel the gear is solid and ``k_lock`` takes over.

**The model in use: stateful friction** (:func:`draft_gear_stateful_torch`).
Each connection carries its gear force ``z`` as a state. Between the loading
curve ``F_L`` and the unloading curve ``F_U`` the gear sticks and ``z`` moves
elastically with the transitional stiffness, ``dz/dt = k_t de/dt``; at a curve
it slides along it. A gear held still keeps its force, as friction gear does.
Both curves start at zero, so there is no force jump at the slack edge.

That does **not** make it converge better than Kelvin-Voigt. ``dz/dt``
switches between stick and slip, and the curve slopes jump at every table
knot, so while couplers are in contact the RHS is discontinuous almost
everywhere: at ``dt = 0.02`` the median velocity error is ~1e-4 m/s against
~1e-6 for Kelvin-Voigt (``scripts/verify_simulator_convergence.py
--draft-gear``). Energy closes and dissipation converges; trajectories are
first-order accurate at best.

``k_t`` is the one modelling choice. The unloading drop in the drop-hammer
plot implies ~7e8 N/m per connection; the default 1e8 keeps RK4 at
``dt = 0.02`` stable on 26 t empty wagons, and captures 86-90 % of the loop
area as a result (the elastic line cuts off the corner of the loop). Use the
drop-hammer value where the step allows it, as the benchmark trains do.

**Rejected: the velocity-switch model** (:func:`draft_gear_force`,
:func:`draft_gear_force_torch`). The stateless look-up-table form of the
review the benchmark cites (Wu et al. 2014, VSD 52(6)),

    F = F_U(g) + (F_L(g) - F_U(g)) * (1 + tanh(sign(e) * de/dt / v_t)) / 2

Kept for the record and its tests, not wired into the RHS. Its switch acts as
a damper of ``(F_L - F_U) / 2 v_t``, ~4e7 N s/m at ``v_t = 5 mm/s``: at the
production step RK4 manufactured ~30x the true dissipation, and even when
converged the dissipation depended on ``v_t``, which has no physical meaning.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

BENCHMARK_TABLE = Path(__file__).parent / "data" / "draft_gear_benchmark.csv"

#: Closing-velocity width of the rejected velocity-switch model [m/s].
DEFAULT_V_TRANSITION = 0.005
#: Connection stiffness once a gear is solid [N/m]. Only reached in severe
#: impacts; like ``k_t`` it is softer than the steel it stands for, so that
#: the production step stays stable.
DEFAULT_K_LOCK = 5.0e8
#: Gears per connection: one at each vehicle end.
N_GEARS = 2


def load_table(path: Path = BENCHMARK_TABLE) -> np.ndarray:
    """``(K, 3)``: per-gear deflection [m], loading force [N], unloading [N]."""
    with open(path) as fh:
        rows = [[float(v) for v in r] for r in list(csv.reader(fh))[1:]]
    t = np.asarray(rows)
    return np.column_stack([t[:, 0] * 1e-3, t[:, 1] * 1e3, t[:, 2] * 1e3])


def draft_gear_force(delta, delta_dot, slack_half, table: np.ndarray,
                     n_gears: int = N_GEARS,
                     v_transition: float = DEFAULT_V_TRANSITION,
                     k_lock: float = DEFAULT_K_LOCK) -> np.ndarray:
    """Connection force [N], positive in draft (tension), NumPy reference.

    ``delta`` is the connection extension ``x_front - x_rear - L0``, as in
    ``forces.coupler_force_slack_asymmetric``.
    """
    delta = np.asarray(delta, float)
    ddot = np.asarray(delta_dot, float)
    s = np.asarray(slack_half, float)
    e = np.where(delta > s, delta - s, np.where(delta < -s, delta + s, 0.0))
    sgn = np.sign(e)
    g = np.abs(e) / n_gears
    g_max = table[-1, 0]
    gc = np.minimum(g, g_max)
    f_l = np.interp(gc, table[:, 0], table[:, 1])
    f_u = np.interp(gc, table[:, 0], table[:, 2])
    sigma = 0.5 * (1.0 + np.tanh(sgn * ddot / v_transition))
    mag = f_u + (f_l - f_u) * sigma
    mag = mag + k_lock * np.maximum(np.abs(e) - n_gears * g_max, 0.0)
    return sgn * mag


#: Elastic stiffness of a connection while its gears stick [N/m]. The
#: unloading drop in the benchmark plot implies ~7e8 per connection (two gears
#: in series); 1e8 keeps RK4 at dt = 0.02 stable down to 26 t empty wagons.
DEFAULT_K_TRANSITION = 1.0e8


def _envelope_torch(e: Tensor, table: Tensor, n_gears: float, k_lock: float):
    """Signed force bounds ``lo <= F <= hi`` at connection deflection ``e``
    and their slopes ``d lo/de``, ``d hi/de``. In the slack (``e = 0``) both
    bounds are zero."""
    xp = table[:, 0].contiguous()
    g_max = xp[-1]
    g = e.abs() / n_gears
    gc = torch.minimum(g, g_max)
    idx = torch.searchsorted(xp, gc.detach().reshape(-1), right=True)
    idx = idx.clamp(1, xp.numel() - 1).reshape(gc.shape)
    dx = xp[idx] - xp[idx - 1]
    w = (gc - xp[idx - 1]) / dx
    f_l = table[idx - 1, 1] + w * (table[idx, 1] - table[idx - 1, 1])
    f_u = table[idx - 1, 2] + w * (table[idx, 2] - table[idx - 1, 2])
    # Slopes with respect to connection deflection |e| (gears in series).
    s_l = (table[idx, 1] - table[idx - 1, 1]) / dx / n_gears
    s_u = (table[idx, 2] - table[idx - 1, 2]) / dx / n_gears
    over = (e.abs() - n_gears * g_max).clamp_min(0.0)
    solid = over > 0
    f_l = f_l + k_lock * over
    f_u = f_u + k_lock * over
    s_l = torch.where(solid, torch.full_like(s_l, k_lock), s_l)
    s_u = torch.where(solid, torch.full_like(s_u, k_lock), s_u)
    pos = e >= 0
    # For e < 0 the characteristic is mirrored: F in [-F_L, -F_U], and
    # d(-F(|e|))/de = F'(|e|), so the slopes keep their sign.
    lo = torch.where(pos, f_u, -f_l)
    hi = torch.where(pos, f_l, -f_u)
    dlo = torch.where(pos, s_u, s_l)
    dhi = torch.where(pos, s_l, s_u)
    return lo, hi, dlo, dhi


def draft_gear_stateful_torch(
    delta: Tensor, delta_dot: Tensor, z: Tensor, slack_half: Tensor, table: Tensor,
    n_gears: float = N_GEARS, k_transition: float = DEFAULT_K_TRANSITION,
    k_lock: float = DEFAULT_K_LOCK,
) -> tuple[Tensor, Tensor]:
    """Stateful friction draft gear: connection force and ``dz/dt``.

    ``z`` is the gear force state [N], positive in draft. Between the loading
    and unloading curves the gears stick and ``z`` moves elastically,
    ``dz/dt = k_t de/dt``. At a curve, if the elastic rate would carry ``z``
    past it, ``z`` slides along the curve instead. The output is ``z``
    clamped to the envelope, which also absorbs integration drift.

    Unlike the velocity switch this has no closing-speed parameter: a gear
    held still keeps whatever force it had, as friction gear does.
    """
    zero = torch.zeros_like(delta)
    outside = delta.abs() > slack_half
    e = torch.where(delta > slack_half, delta - slack_half,
                    torch.where(delta < -slack_half, delta + slack_half, zero))
    e_dot = torch.where(outside, delta_dot, zero)
    lo, hi, dlo, dhi = _envelope_torch(e, table, n_gears, k_lock)
    r = k_transition * e_dot
    r_hi, r_lo = dhi * e_dot, dlo * e_dot
    dz = torch.where((z >= hi) & (r > r_hi), r_hi,
                     torch.where((z <= lo) & (r < r_lo), r_lo, r))
    force = torch.minimum(torch.maximum(z, lo), hi)
    return force, dz


def draft_gear_initial_state(delta, slack_half, table: np.ndarray,
                             n_gears: int = N_GEARS) -> np.ndarray:
    """A neutral starting force: the envelope midpoint at the initial
    deflection (zero for a connection in its slack)."""
    delta = np.asarray(delta, float)
    s = np.asarray(slack_half, float)
    e = np.where(delta > s, delta - s, np.where(delta < -s, delta + s, 0.0))
    g = np.minimum(np.abs(e) / n_gears, table[-1, 0])
    mid = 0.5 * (np.interp(g, table[:, 0], table[:, 1])
                 + np.interp(g, table[:, 0], table[:, 2]))
    return np.sign(e) * mid


def draft_gear_force_torch(delta: Tensor, delta_dot: Tensor, slack_half: Tensor,
                           table: Tensor, n_gears: float = N_GEARS,
                           v_transition: float = DEFAULT_V_TRANSITION,
                           k_lock: float = DEFAULT_K_LOCK) -> Tensor:
    """Same law as :func:`draft_gear_force`, batched and differentiable.

    ``table`` is ``(K, 3)`` in SI units on the batch's device.
    """
    zero = torch.zeros_like(delta)
    e = torch.where(delta > slack_half, delta - slack_half,
                    torch.where(delta < -slack_half, delta + slack_half, zero))
    sgn = torch.sign(e)
    g = e.abs() / n_gears
    xp = table[:, 0].contiguous()
    g_max = xp[-1]
    gc = torch.minimum(g, g_max)
    idx = torch.searchsorted(xp, gc.detach().reshape(-1), right=True)
    idx = idx.clamp(1, xp.numel() - 1).reshape(gc.shape)
    x0, x1 = xp[idx - 1], xp[idx]
    w = (gc - x0) / (x1 - x0)
    f_l = table[idx - 1, 1] + w * (table[idx, 1] - table[idx - 1, 1])
    f_u = table[idx - 1, 2] + w * (table[idx, 2] - table[idx - 1, 2])
    sigma = 0.5 * (1.0 + torch.tanh(sgn * delta_dot / v_transition))
    mag = f_u + (f_l - f_u) * sigma
    mag = mag + k_lock * (e.abs() - n_gears * g_max).clamp_min(0.0)
    return sgn * mag
