"""Batched, GPU-resident, differentiable torch port of the N-vehicle RHS.

The NumPy path (:mod:`simulator2.rhs`) is the oracle for this module and is not
imported here: everything is rebuilt from the scenario tensors so that a shared
bug cannot make the two implementations agree. See ``TORCH_PORT_SPEC.md`` for
the design rationale and ``tests/test_torch_rhs.py`` for the acceptance
contract.

Layout, mirroring the NumPy path exactly:

``y``           ``(B, 4N)`` flat, ordered ``[x(N), v(N), z_brk(N), z_trac(N)]``
``node_static`` ``(B, N, 7)``   see :class:`NodeStatic`
``edge_static`` ``(B, N-1, 6)`` see :class:`EdgeStatic`

Ragged consists are handled by padding to ``N = max(N_i)`` plus a boolean
``node_mask``; padded vehicles contribute exactly zero force and zero
acceleration. Disjoint-union graph batching is deliberately not used here -- it
belongs to the GNN dataloader later, whereas the integrator wants a dense,
bucketable batch.

Two properties of the NumPy model are reproduced deliberately rather than
cleaned up, because the spec locks the physics (section 4.2, 4.8):

* the grade term evaluates ``sin()`` on a field that already holds
  ``sin(theta)`` -- see :func:`_grade_force`;
* Davis resistance uses a hard ``sign(v)`` with a ``1e-9`` threshold, which is
  a genuine discontinuity at ``v = 0``.

Both are noted in ``TORCH_PORT_REPORT.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .constants import (
    DEFAULT_C_BUFF,
    DEFAULT_C_DRAFT,
    DEFAULT_DAVIS_A_CAR,
    DEFAULT_DAVIS_A_LOCO,
    DEFAULT_DAVIS_B_CAR,
    DEFAULT_DAVIS_B_LOCO,
    DEFAULT_DAVIS_C_CAR,
    DEFAULT_DAVIS_C_LOCO,
    DEFAULT_K_BUFF,
    DEFAULT_K_DRAFT,
    DEFAULT_M_CAR_KG,
    DEFAULT_M_LOCO_KG,
    DEFAULT_P_MAX_W,
    DEFAULT_SLACK_HALF_M,
    DEFAULT_TAU_BRK_S,
    DEFAULT_TAU_TRAC_S,
    DEFAULT_V_BRAKE_EPS,
    DEFAULT_V_EPS,
    GRAVITY_MPS2,
)

__all__ = ["TorchScenarioBatch", "torch_rhs", "rollout_rk4"]


# Threshold below which Davis resistance and the curvature proxy are treated as
# exactly zero rather than taking a sign. Matches ``forces.davis_resistance_
# longitudinal`` and ``route.curvature_force_longitudinal``.
_V_SIGN_EPS = 1e-9


class NodeStatic(IntEnum):
    """Columns of ``node_static``; the layout the fixtures are written in.

    Distinct from :class:`simulator2.state_schema.NodeChannel`, which indexes
    the 11-channel ``H`` tensor with the dynamic state interleaved. Here the
    dynamic state lives in ``y`` and only the static per-vehicle parameters are
    carried alongside.
    """

    MASS_KG = 0
    DAVIS_A = 1
    DAVIS_B = 2
    DAVIS_C = 3
    CAN_TRACTION = 4
    F_TRAC_MAX_N = 5
    F_BRK_MAX_N = 6


class EdgeStatic(IntEnum):
    """Columns of ``edge_static``."""

    L0_M = 0
    SLACK_HALF_M = 1
    K_DRAFT = 2
    C_DRAFT = 3
    K_BUFF = 4
    C_BUFF = 5


D_NODE_STATIC = len(NodeStatic)
D_EDGE_STATIC = len(EdgeStatic)


# ---------------------------------------------------------------------------
# batched interpolation helpers
# ---------------------------------------------------------------------------

def _interp_batched(xp: Tensor, fp: Tensor, x: Tensor) -> Tensor:
    """Batched piecewise-linear interpolation with ``numpy.interp`` semantics.

    ``xp``, ``fp`` are ``(B, S)`` with ``xp`` strictly increasing along dim 1;
    ``x`` is ``(B, M)``. Values outside ``[xp[0], xp[-1]]`` clamp to the nearest
    endpoint, exactly as :func:`numpy.interp` does.

    Uses ``searchsorted`` + ``gather``, never a Python loop over vehicles, so
    the cost is independent of ``N``. Differentiable in ``x``.
    """
    n_nodes = xp.shape[1]
    idx = torch.searchsorted(xp, x.contiguous(), right=True)
    idx = idx.clamp(1, n_nodes - 1)

    x0 = torch.gather(xp, 1, idx - 1)
    x1 = torch.gather(xp, 1, idx)
    f0 = torch.gather(fp, 1, idx - 1)
    f1 = torch.gather(fp, 1, idx)

    span = x1 - x0
    # A degenerate (repeated) node would divide by zero; fall back to the left
    # value there rather than emitting a NaN that would poison the whole batch.
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    w = torch.where(span > 0, (x - x0) / safe_span, torch.zeros_like(span))
    w = w.clamp(0.0, 1.0)  # clamping here is what reproduces numpy's flat ends
    return f0 + w * (f1 - f0)


def _interp_commands(
    cmd_t: Tensor, values: Tensor, t: float
) -> Tensor:
    """Sample a materialized command field ``(B, T, N)`` at scalar time ``t``.

    Commands are materialized in time per ``DATA_SCHEMA.md`` section E rather
    than passed as Python callables, which would serialise the batch.
    """
    b, n_t = cmd_t.shape
    tq = torch.full((b, 1), float(t), dtype=cmd_t.dtype, device=cmd_t.device)
    idx = torch.searchsorted(cmd_t, tq, right=True).clamp(1, n_t - 1)

    t0 = torch.gather(cmd_t, 1, idx - 1)
    t1 = torch.gather(cmd_t, 1, idx)
    span = t1 - t0
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    w = torch.where(span > 0, (tq - t0) / safe_span, torch.zeros_like(span))
    w = w.clamp(0.0, 1.0)

    n_veh = values.shape[2]
    gather_lo = (idx - 1).unsqueeze(-1).expand(b, 1, n_veh)
    gather_hi = idx.unsqueeze(-1).expand(b, 1, n_veh)
    v_lo = torch.gather(values, 1, gather_lo).squeeze(1)
    v_hi = torch.gather(values, 1, gather_hi).squeeze(1)
    return v_lo + w * (v_hi - v_lo)


# ---------------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------------

@dataclass
class TorchScenarioBatch:
    """A padded batch of scenarios, all tensors resident on one device.

    Every field is ``(B, ...)``. Scalars that vary per scenario are kept as
    ``(B, 1)`` so they broadcast against per-vehicle quantities without any
    reshaping in the hot path.
    """

    y0: Tensor              # (B, 4N)
    node_mask: Tensor       # (B, N)   bool
    edge_mask: Tensor       # (B, N-1) bool
    node_static: Tensor     # (B, N, 7)
    edge_static: Tensor     # (B, N-1, 6)

    cmd_t: Tensor           # (B, T_cmd)
    u_trac: Tensor          # (B, T_cmd, N)
    u_brk: Tensor           # (B, T_cmd, N)

    route_s: Tensor         # (B, S)
    route_sin_theta: Tensor  # (B, S)
    route_kappa: Tensor     # (B, S)

    tau_brk_s: Tensor       # (B, 1)
    tau_trac_s: Tensor      # (B, 1)
    p_max_w: Tensor         # (B, 1)
    k_curv_scale: Tensor    # (B, 1)
    v_eps: Tensor           # (B, 1)
    v_brake_eps: Tensor     # (B, 1)
    brake_opposes_motion: Tensor  # (B, 1) bool

    def __post_init__(self) -> None:
        # Padded vehicles carry mass 0, which would divide to NaN before the
        # mask is applied -- and NaN * 0 is still NaN, so the mask alone cannot
        # save us. Substitute a harmless unit mass in the padded slots.
        mass = self.node_static[..., NodeStatic.MASS_KG]
        self._mass_safe = torch.where(self.node_mask, mass, torch.ones_like(mass))
        self._node_maskf = self.node_mask.to(self.y0.dtype)
        self._edge_maskf = self.edge_mask.to(self.y0.dtype)

    # -- properties ---------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return int(self.node_mask.shape[0])

    @property
    def n_vehicles(self) -> int:
        """Padded width ``N``; per-scenario counts are ``node_mask.sum(1)``."""
        return int(self.node_mask.shape[1])

    @property
    def device(self) -> torch.device:
        return self.y0.device

    @property
    def dtype(self) -> torch.dtype:
        return self.y0.dtype

    # -- construction -------------------------------------------------------

    @classmethod
    def from_fixtures(
        cls,
        paths: Sequence[Path | str],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "TorchScenarioBatch":
        """Build a batch from reference ``.npz`` fixtures.

        The fixtures are self-describing, so nothing here needs the NumPy
        simulator. Scenarios may differ in ``N``, in command-grid length and in
        route-grid length; each is padded to the batch maximum.
        """
        raw = []
        for p in paths:
            with np.load(Path(p)) as d:
                raw.append({k: d[k] for k in d.files})

        b = len(raw)
        n_max = max(int(r["N"]) for r in raw)
        t_cmd_max = max(int(r["t"].shape[0]) for r in raw)
        s_max = max(int(r["route_s"].shape[0]) for r in raw)

        dev = torch.device(device)
        f_kw = {"dtype": dtype, "device": dev}

        y0 = torch.zeros(b, 4 * n_max, **f_kw)
        node_mask = torch.zeros(b, n_max, dtype=torch.bool, device=dev)
        edge_mask = torch.zeros(b, max(n_max - 1, 0), dtype=torch.bool, device=dev)
        node_static = torch.zeros(b, n_max, D_NODE_STATIC, **f_kw)
        edge_static = torch.zeros(b, max(n_max - 1, 0), D_EDGE_STATIC, **f_kw)
        cmd_t = torch.zeros(b, t_cmd_max, **f_kw)
        u_trac = torch.zeros(b, t_cmd_max, n_max, **f_kw)
        u_brk = torch.zeros(b, t_cmd_max, n_max, **f_kw)
        route_s = torch.zeros(b, s_max, **f_kw)
        route_sin_theta = torch.zeros(b, s_max, **f_kw)
        route_kappa = torch.zeros(b, s_max, **f_kw)

        def _scalar(name: str) -> Tensor:
            return torch.tensor(
                [[float(r[name])] for r in raw], **f_kw
            )

        for i, r in enumerate(raw):
            n = int(r["N"])
            y = np.asarray(r["y0"], dtype=np.float64)
            # y is [x(n), v(n), z_brk(n), z_trac(n)]; re-block it into the
            # padded width so every scenario shares one channel layout.
            for blk in range(4):
                seg = torch.as_tensor(y[blk * n : (blk + 1) * n], **f_kw)
                y0[i, blk * n_max : blk * n_max + n] = seg

            node_mask[i, :n] = True
            if n_max > 1:
                edge_mask[i, : max(n - 1, 0)] = True
            node_static[i, :n] = torch.as_tensor(r["node_static"], **f_kw)
            if n > 1 and n_max > 1:
                edge_static[i, : n - 1] = torch.as_tensor(r["edge_static"], **f_kw)

            n_t = int(r["t"].shape[0])
            cmd_t[i, :n_t] = torch.as_tensor(r["t"], **f_kw)
            # Pad the command grid with strictly increasing times beyond the
            # end so searchsorted stays well-defined; the clamped interpolation
            # then just holds the final command.
            if n_t < t_cmd_max:
                last = float(r["t"][-1])
                cmd_t[i, n_t:] = torch.arange(
                    1, t_cmd_max - n_t + 1, **f_kw
                ) + last
                u_trac[i, n_t:, :n] = torch.as_tensor(r["u_trac"][-1], **f_kw)
                u_brk[i, n_t:, :n] = torch.as_tensor(r["u_brk"][-1], **f_kw)
            u_trac[i, :n_t, :n] = torch.as_tensor(r["u_trac"], **f_kw)
            u_brk[i, :n_t, :n] = torch.as_tensor(r["u_brk"], **f_kw)

            n_s = int(r["route_s"].shape[0])
            route_s[i, :n_s] = torch.as_tensor(r["route_s"], **f_kw)
            route_sin_theta[i, :n_s] = torch.as_tensor(r["route_sin_theta"], **f_kw)
            kap = r["route_kappa"]
            if kap is not None and np.ndim(kap) == 1:
                route_kappa[i, :n_s] = torch.as_tensor(kap, **f_kw)
            if n_s < s_max:
                last_s = float(r["route_s"][-1])
                route_s[i, n_s:] = torch.arange(1, s_max - n_s + 1, **f_kw) + last_s
                route_sin_theta[i, n_s:] = route_sin_theta[i, n_s - 1]
                route_kappa[i, n_s:] = route_kappa[i, n_s - 1]

        bom = torch.tensor(
            [[bool(r["brake_opposes_motion"])] for r in raw],
            dtype=torch.bool, device=dev,
        )

        return cls(
            y0=y0, node_mask=node_mask, edge_mask=edge_mask,
            node_static=node_static, edge_static=edge_static,
            cmd_t=cmd_t, u_trac=u_trac, u_brk=u_brk,
            route_s=route_s, route_sin_theta=route_sin_theta,
            route_kappa=route_kappa,
            tau_brk_s=_scalar("tau_brk_s"), tau_trac_s=_scalar("tau_trac_s"),
            p_max_w=_scalar("p_max_w"), k_curv_scale=_scalar("k_curv_scale"),
            v_eps=_scalar("v_eps"), v_brake_eps=_scalar("v_brake_eps"),
            brake_opposes_motion=bom,
        )

    @classmethod
    def synthetic(
        cls,
        batch_size: int = 8,
        n_vehicles: int = 60,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        duration_s: float = 600.0,
        seed: int = 0,
    ) -> "TorchScenarioBatch":
        """A plausible full-size batch for throughput work.

        Correctness is established against the fixtures, which are small by
        necessity (generating large references with the NumPy path is
        impractical -- see ``TORCH_PORT_SPEC.md`` section 9). This builds
        realistic consists at realistic ``N`` so performance can be measured at
        the size the dataset actually needs, without pretending to be a
        validation case.
        """
        dev = torch.device(device)
        f_kw = {"dtype": dtype, "device": dev}
        rng = np.random.default_rng(seed)
        b, n = batch_size, n_vehicles

        n_loco = max(1, n // 40)  # a head-end pair on a long consist
        is_loco = np.zeros(n, dtype=bool)
        is_loco[:n_loco] = True

        node = np.zeros((b, n, D_NODE_STATIC), dtype=np.float64)
        for i in range(b):
            jitter = rng.uniform(0.9, 1.1, size=n)
            node[i, :, NodeStatic.MASS_KG] = np.where(
                is_loco, DEFAULT_M_LOCO_KG, DEFAULT_M_CAR_KG * jitter
            )
            node[i, :, NodeStatic.DAVIS_A] = np.where(
                is_loco, DEFAULT_DAVIS_A_LOCO, DEFAULT_DAVIS_A_CAR
            )
            node[i, :, NodeStatic.DAVIS_B] = np.where(
                is_loco, DEFAULT_DAVIS_B_LOCO, DEFAULT_DAVIS_B_CAR
            )
            node[i, :, NodeStatic.DAVIS_C] = np.where(
                is_loco, DEFAULT_DAVIS_C_LOCO, DEFAULT_DAVIS_C_CAR
            )
            node[i, :, NodeStatic.CAN_TRACTION] = is_loco.astype(np.float64)
            node[i, :, NodeStatic.F_TRAC_MAX_N] = np.where(is_loco, 600_000.0, 0.0)
            node[i, :, NodeStatic.F_BRK_MAX_N] = 250_000.0

        edge = np.zeros((b, n - 1, D_EDGE_STATIC), dtype=np.float64)
        edge[:, :, EdgeStatic.L0_M] = 20.0
        edge[:, :, EdgeStatic.SLACK_HALF_M] = DEFAULT_SLACK_HALF_M
        edge[:, :, EdgeStatic.K_DRAFT] = DEFAULT_K_DRAFT
        edge[:, :, EdgeStatic.C_DRAFT] = DEFAULT_C_DRAFT
        edge[:, :, EdgeStatic.K_BUFF] = DEFAULT_K_BUFF
        edge[:, :, EdgeStatic.C_BUFF] = DEFAULT_C_BUFF

        # Rolling route, 30 km, grades within the pipeline's 4 % clip.
        s_nodes = 3001
        s = np.linspace(0.0, 30_000.0, s_nodes)
        sin_theta = np.zeros((b, s_nodes))
        kappa = np.zeros((b, s_nodes))
        for i in range(b):
            phase = rng.uniform(0.0, 2 * math.pi, size=3)
            sin_theta[i] = (
                0.012 * np.sin(2 * math.pi * s / 8_000.0 + phase[0])
                + 0.006 * np.sin(2 * math.pi * s / 2_500.0 + phase[1])
            )
            kappa[i] = 1.0e-3 * np.abs(np.sin(2 * math.pi * s / 4_000.0 + phase[2]))

        # A modulated traction/brake profile: enough command movement to keep
        # the couplers working, which is the expensive case for the solver.
        n_t = int(duration_s / 0.5) + 1
        t_cmd = np.linspace(0.0, duration_s, n_t)
        u_trac = np.zeros((b, n_t, n))
        u_brk = np.zeros((b, n_t, n))
        for i in range(b):
            ph = rng.uniform(0.0, 2 * math.pi)
            duty = 0.5 + 0.4 * np.sin(2 * math.pi * t_cmd / 90.0 + ph)
            trac = np.clip(duty, 0.0, 1.0)
            brk = np.clip(-duty, 0.0, 1.0)
            u_trac[i, :, :] = (trac[:, None] * node[i, :, NodeStatic.F_TRAC_MAX_N])
            u_brk[i, :, :] = (brk[:, None] * node[i, :, NodeStatic.F_BRK_MAX_N])

        # Vehicles are ordered front to back, so x decreases with index.
        x_lead = 1_000.0
        x0 = x_lead - np.cumsum(
            np.concatenate([[0.0], edge[0, :, EdgeStatic.L0_M]])
        )
        y0 = np.zeros((b, 4 * n))
        for i in range(b):
            y0[i, 0:n] = x0
            y0[i, n : 2 * n] = 15.0
            y0[i, 2 * n : 3 * n] = u_brk[i, 0]   # actuators settled at t = 0
            y0[i, 3 * n : 4 * n] = u_trac[i, 0]

        ones = torch.ones(b, 1, **f_kw)
        return cls(
            y0=torch.as_tensor(y0, **f_kw),
            node_mask=torch.ones(b, n, dtype=torch.bool, device=dev),
            edge_mask=torch.ones(b, n - 1, dtype=torch.bool, device=dev),
            node_static=torch.as_tensor(node, **f_kw),
            edge_static=torch.as_tensor(edge, **f_kw),
            cmd_t=torch.as_tensor(np.tile(t_cmd, (b, 1)), **f_kw),
            u_trac=torch.as_tensor(u_trac, **f_kw),
            u_brk=torch.as_tensor(u_brk, **f_kw),
            route_s=torch.as_tensor(np.tile(s, (b, 1)), **f_kw),
            route_sin_theta=torch.as_tensor(sin_theta, **f_kw),
            route_kappa=torch.as_tensor(kappa, **f_kw),
            tau_brk_s=ones * DEFAULT_TAU_BRK_S,
            tau_trac_s=ones * DEFAULT_TAU_TRAC_S,
            p_max_w=ones * DEFAULT_P_MAX_W,
            k_curv_scale=torch.zeros(b, 1, **f_kw),
            v_eps=ones * DEFAULT_V_EPS,
            v_brake_eps=ones * DEFAULT_V_BRAKE_EPS,
            brake_opposes_motion=torch.ones(b, 1, dtype=torch.bool, device=dev),
        )

    def to(self, device=None, dtype=None) -> "TorchScenarioBatch":
        """Move / cast every tensor field, preserving bool masks."""
        def _cast(v: Tensor) -> Tensor:
            if v.dtype == torch.bool:
                return v.to(device=device) if device is not None else v
            return v.to(device=device, dtype=dtype)

        return TorchScenarioBatch(**{
            f: _cast(getattr(self, f)) for f in self.__dataclass_fields__
        })


# ---------------------------------------------------------------------------
# right-hand side
# ---------------------------------------------------------------------------

def _grade_force(mass: Tensor, sin_theta_field: Tensor) -> Tensor:
    """``m g sin(theta(x))`` as the NumPy path computes it.

    ``forces.grade_force`` calls ``np.sin`` on the value returned by
    ``RouteProfile.sin_theta_at``, which is already ``sin(theta)``. The extra
    ``sin`` is therefore applied to a sine. It is reproduced here because the
    NumPy path is the oracle and the fixtures encode it; at the grades in play
    (``|sin theta| <= 0.04``) the two differ by about 1e-5 relative. Flagged in
    the port report rather than silently corrected.
    """
    return mass * GRAVITY_MPS2 * torch.sin(sin_theta_field)


def torch_rhs(
    t: float,
    y: Tensor,
    batch: TorchScenarioBatch,
    *,
    brake_opposes_motion: bool | None = None,
) -> Tensor:
    """Batched RHS. ``y`` is ``(B, 4N)``; returns ``dy`` of the same shape.

    ``brake_opposes_motion=None`` uses the per-scenario value carried in
    ``batch``; an explicit bool overrides it for the whole batch.

    No in-place mutation, no ``.item()``, no data-dependent Python branching
    over batch elements -- the whole thing has to stay inside autograd.
    """
    n = batch.n_vehicles
    x = y[:, 0:n]
    v = y[:, n : 2 * n]
    z_brk = y[:, 2 * n : 3 * n]
    z_trac = y[:, 3 * n : 4 * n]

    ns = batch.node_static
    mass = ns[..., NodeStatic.MASS_KG]
    davis_a = ns[..., NodeStatic.DAVIS_A]
    davis_b = ns[..., NodeStatic.DAVIS_B]
    davis_c = ns[..., NodeStatic.DAVIS_C]
    can_traction = ns[..., NodeStatic.CAN_TRACTION]
    f_brk_max = ns[..., NodeStatic.F_BRK_MAX_N]

    # -- coupler forces (B, N-1) -------------------------------------------
    es = batch.edge_static
    delta = (x[:, :-1] - x[:, 1:]) - es[..., EdgeStatic.L0_M]
    delta_dot = v[:, :-1] - v[:, 1:]
    slack = es[..., EdgeStatic.SLACK_HALF_M]

    zeros_e = torch.zeros_like(delta)
    f_draft = es[..., EdgeStatic.K_DRAFT] * (delta - slack) \
        + es[..., EdgeStatic.C_DRAFT] * delta_dot
    f_buff = es[..., EdgeStatic.K_BUFF] * (delta + slack) \
        + es[..., EdgeStatic.C_BUFF] * delta_dot
    # The exact piecewise deadband, kept as-is: softening it would change the
    # physics documented in Chapter 4 (spec section 4.8).
    f_cpl = torch.where(delta > slack, f_draft, zeros_e)
    f_cpl = torch.where(delta < -slack, f_buff, f_cpl)
    f_cpl = f_cpl * batch._edge_maskf

    f_in = torch.nn.functional.pad(f_cpl, (1, 0))   # force from the vehicle ahead
    f_out = torch.nn.functional.pad(f_cpl, (0, 1))  # force to the vehicle behind

    # -- commands -----------------------------------------------------------
    u_trac = _interp_commands(batch.cmd_t, batch.u_trac, t)
    u_brk = _interp_commands(batch.cmd_t, batch.u_brk, t)
    u_trac = torch.where(can_traction > 0, u_trac, torch.zeros_like(u_trac))

    # -- resistances --------------------------------------------------------
    abs_v = v.abs()
    # Hard sign with a 1e-9 threshold: a genuine discontinuity at v = 0 in the
    # existing model, reproduced exactly rather than smoothed.
    sgn_v = torch.where(abs_v < _V_SIGN_EPS, torch.zeros_like(v), torch.sign(v))
    r_davis = (davis_a + davis_b * abs_v + davis_c * v * v) * sgn_v

    sin_theta = _interp_batched(batch.route_s, batch.route_sin_theta, x)
    f_grade = _grade_force(mass, sin_theta)

    kappa = _interp_batched(batch.route_s, batch.route_kappa, x)
    f_curv = batch.k_curv_scale * mass * v * v * kappa.abs() * sgn_v

    # -- actuators ----------------------------------------------------------
    zt = z_trac.clamp_min(0.0)
    zb = z_brk.clamp_min(0.0)

    power_cap = batch.p_max_w / abs_v.clamp_min(batch.v_eps)
    f_trac = torch.where(
        can_traction > 0, torch.minimum(zt, power_cap), torch.zeros_like(zt)
    )

    f_brake = torch.minimum(zb, f_brk_max)
    if brake_opposes_motion is None:
        bom = batch.brake_opposes_motion
    else:
        bom = torch.full_like(batch.brake_opposes_motion, bool(brake_opposes_motion))
    f_brake = torch.where(
        bom, f_brake * torch.tanh(v / batch.v_brake_eps), f_brake
    )

    # -- assemble -----------------------------------------------------------
    net = f_trac - f_brake - r_davis - f_grade - f_curv + f_in - f_out
    dv = net / batch._mass_safe

    dx = v
    dz_brk = (u_brk - z_brk) / batch.tau_brk_s
    dz_trac = (u_trac - z_trac) / batch.tau_trac_s

    m = batch._node_maskf
    return torch.cat([dx * m, dv * m, dz_brk * m, dz_trac * m], dim=1)


# ---------------------------------------------------------------------------
# fixed-step RK4 rollout
# ---------------------------------------------------------------------------

def rollout_rk4(
    batch: TorchScenarioBatch,
    y0: Tensor,
    t_grid: Tensor,
    dt: float = 0.02,
) -> tuple[Tensor, Tensor]:
    """Fixed-step RK4, sampled onto ``t_grid``.

    Returns ``(t_grid, y_out)`` with ``y_out`` of shape ``(B, T, 4N)``.

    ``t_grid`` is generally coarser than ``dt``. Rather than integrating on a
    global ``dt`` lattice and interpolating -- which would add a second-order
    sampling error on top of RK4's fourth-order one -- each output interval is
    subdivided into ``ceil(span / dt)`` equal substeps. Output times are then
    hit exactly, and the effective step is always ``<= dt``, so accuracy is at
    least what ``dt`` promises. Non-uniform output grids are handled naturally.

    Fixed-step is a locked decision (spec section 4.1): the coupler deadband
    makes adaptive stepping's cost vary 61x across driving regimes, and the
    measured Jacobian spectral radius (~21 rad/s at N=130) leaves dt=0.02
    roughly 7x inside the RK4 stability limit.
    """
    if t_grid.ndim != 1:
        raise ValueError(f"t_grid must be 1-D, got shape {tuple(t_grid.shape)}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # One host sync up front for the whole schedule, rather than one per step.
    t_np = t_grid.detach().cpu().numpy().astype(np.float64)

    y = y0
    outputs = [y]
    for j in range(len(t_np) - 1):
        t_start, t_end = float(t_np[j]), float(t_np[j + 1])
        span = t_end - t_start
        n_sub = max(1, int(math.ceil(abs(span) / dt - 1e-12)))
        h = span / n_sub
        for k in range(n_sub):
            y = _rk4_step(batch, y, t_start + k * h, h)
        outputs.append(y)

    return t_grid, torch.stack(outputs, dim=1)


def _rk4_step(batch: TorchScenarioBatch, y: Tensor, t: float, h: float) -> Tensor:
    """One classical RK4 step. Out-of-place throughout, for autograd."""
    k1 = torch_rhs(t, y, batch)
    k2 = torch_rhs(t + 0.5 * h, y + (0.5 * h) * k1, batch)
    k3 = torch_rhs(t + 0.5 * h, y + (0.5 * h) * k2, batch)
    k4 = torch_rhs(t + h, y + h * k3, batch)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
