"""Verification of the data-generating simulator against exact solutions.

Verification, not validation: this asks whether the code solves the equations
Chapter 4 writes down, not whether those equations describe a real train.
Everything here runs the production path -- ``torch_rhs`` advanced by the
fixed-step RK4 of ``rollout_rk4`` -- because that is what wrote the corpora,
and compares it with something that does not share its code: a closed form,
an eigenvalue problem, or a conservation law.

The cases, each chosen so that one force law is isolated and has an exact
answer:

* **coastdown** -- Davis resistance alone; ``m v' = -(A + B v + C v^2)`` has an
  arctangent solution when ``4AC > B^2``, which holds for every catalog entry.
* **constant grade** -- ``v' = -g sin(sin theta)``, linear in time. Also
  measures the double-sine quirk (``_grade_force``) against ``g sin theta``.
* **balancing speed** -- constant traction against Davis; the root of a
  quadratic, or of a cubic when the power cap binds.
* **actuator lag** -- ``z' = (u - z)/tau`` from rest, an exponential.
* **brake stop** -- constant brake with the ``tanh(v/eps)`` direction blend,
  resistance off. The blend makes the stop asymptotic, and the distance has
  a closed form: ``v0^2/(2a) + eps^2 pi^2/(12 a)``.
* **deadband oscillator** -- two masses, one undamped coupler with slack and
  asymmetric stiffness. The motion is piecewise harmonic with free flight
  through the gap, so the period and the buff amplitude are exact.
* **damped contact** -- the same pair with damping: a Kelvin-Voigt contact,
  whose coefficient of restitution is ``exp(-zeta pi / sqrt(1 - zeta^2))``.
* **chain modes** -- ``N`` equal masses, linear couplers without slack. The
  free-free chain has ``omega_j = 2 sqrt(k/m) sin(j pi / 2N)`` and cosine mode
  shapes; a mode released from rest must stay in that mode.
* **momentum** -- coupler forces are internal, so ``sum m_i v_i'`` must equal
  the sum of external forces to rounding, at any state.
* **energy audit** -- kinetic + grade potential + coupler spring energy,
  against the work done by traction, brakes, resistance and coupler damping,
  integrated alongside the state by the same RK4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .constants import (
    DEFAULT_V_BRAKE_EPS,
    DEFAULT_V_EPS,
    GRAVITY_MPS2,
)
from .draft_gear import draft_gear_stateful_torch
from .torch_rhs import (
    D_EDGE_STATIC,
    D_NODE_STATIC,
    EdgeStatic,
    NodeStatic,
    TorchScenarioBatch,
    _interp_batched,
    torch_rhs,
)

F64 = torch.float64


# ---------------------------------------------------------------------------
# a plain single-scenario batch
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """One scenario, specified directly. Arrays are per vehicle or per coupler.

    Commands are held constant over the run unless ``cmd_t`` / ``u_trac`` /
    ``u_brk`` are given as ``(T,)`` and ``(T, N)``. Actuators start settled
    at the first command unless ``z_trac0`` / ``z_brk0`` say otherwise.
    """

    mass: np.ndarray
    x0: np.ndarray
    v0: np.ndarray
    davis: tuple[float, float, float] = (0.0, 0.0, 0.0)
    can_traction: np.ndarray | None = None
    f_trac_max: float = 1e9
    f_brk_max: float = 1e9
    L0: float = 20.0
    slack: float = 0.0
    k_draft: float = 0.0
    c_draft: float = 0.0
    k_buff: float = 0.0
    c_buff: float = 0.0
    sin_theta: float | tuple[np.ndarray, np.ndarray] = 0.0
    u_trac: float | np.ndarray = 0.0
    u_brk: float | np.ndarray = 0.0
    cmd_t: np.ndarray | None = None
    z_trac0: np.ndarray | None = None
    z_brk0: np.ndarray | None = None
    tau_trac: float = 5.0
    tau_brk: float = 3.0
    tau_brk_node: np.ndarray | None = None
    tau_brk_release_node: np.ndarray | None = None
    p_max: float = 1e12
    v_eps: float = DEFAULT_V_EPS
    v_brake_eps: float = DEFAULT_V_BRAKE_EPS
    route_len: float = 200_000.0
    coupler_model: str = "kelvin_voigt"
    z_cpl0: np.ndarray | None = None
    draft_gear_table: np.ndarray | None = None
    draft_gear_k_transition: float = 1.0e8
    draft_gear_k_lock: float = 5.0e8
    grade_model: str = "double_sine"
    inertia_factor: np.ndarray | None = None
    standstill_model: str = "none"

    def batch(self) -> TorchScenarioBatch:
        n = len(self.mass)
        mass = np.asarray(self.mass, float)
        node = np.zeros((1, n, D_NODE_STATIC))
        node[0, :, NodeStatic.MASS_KG] = mass
        node[0, :, NodeStatic.DAVIS_A] = self.davis[0]
        node[0, :, NodeStatic.DAVIS_B] = self.davis[1]
        node[0, :, NodeStatic.DAVIS_C] = self.davis[2]
        trac = (np.ones(n, bool) if self.can_traction is None
                else np.asarray(self.can_traction, bool))
        node[0, :, NodeStatic.CAN_TRACTION] = trac
        node[0, :, NodeStatic.F_TRAC_MAX_N] = self.f_trac_max
        node[0, :, NodeStatic.F_BRK_MAX_N] = self.f_brk_max

        edge = np.zeros((1, max(n - 1, 0), D_EDGE_STATIC))
        edge[0, :, EdgeStatic.L0_M] = self.L0
        edge[0, :, EdgeStatic.SLACK_HALF_M] = self.slack
        edge[0, :, EdgeStatic.K_DRAFT] = self.k_draft
        edge[0, :, EdgeStatic.C_DRAFT] = self.c_draft
        edge[0, :, EdgeStatic.K_BUFF] = self.k_buff
        edge[0, :, EdgeStatic.C_BUFF] = self.c_buff

        if isinstance(self.sin_theta, tuple):
            s, st = (np.asarray(a, float) for a in self.sin_theta)
        else:
            s = np.linspace(-self.route_len, self.route_len, 3)
            st = np.full_like(s, float(self.sin_theta))

        if self.cmd_t is None:
            cmd_t = np.array([0.0, 1e7])
            ut = np.broadcast_to(np.asarray(self.u_trac, float), (n,))
            ub = np.broadcast_to(np.asarray(self.u_brk, float), (n,))
            u_trac = np.stack([ut, ut])
            u_brk = np.stack([ub, ub])
        else:
            cmd_t = np.asarray(self.cmd_t, float)
            u_trac = np.asarray(self.u_trac, float)
            u_brk = np.asarray(self.u_brk, float)

        z_t = u_trac[0] if self.z_trac0 is None else np.asarray(self.z_trac0, float)
        z_b = u_brk[0] if self.z_brk0 is None else np.asarray(self.z_brk0, float)
        parts = [self.x0, self.v0, z_b, z_t]
        if self.coupler_model == "draft_gear":
            from .draft_gear import draft_gear_initial_state
            x0 = np.asarray(self.x0, float)
            parts.append(draft_gear_initial_state(
                x0[:-1] - x0[1:] - self.L0, self.slack, self.draft_gear_table)
                if self.z_cpl0 is None else np.asarray(self.z_cpl0, float))
        y0 = np.concatenate(parts)[None, :]

        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=F64)  # noqa: E731
        one = torch.ones(1, 1, dtype=F64)
        return TorchScenarioBatch(
            y0=t(y0),
            node_mask=torch.ones(1, n, dtype=torch.bool),
            edge_mask=torch.ones(1, max(n - 1, 0), dtype=torch.bool),
            node_static=t(node), edge_static=t(edge),
            cmd_t=t(cmd_t[None]), u_trac=t(u_trac[None]), u_brk=t(u_brk[None]),
            route_s=t(s[None]), route_sin_theta=t(st[None]),
            route_kappa=torch.zeros(1, len(s), dtype=F64),
            tau_brk_s=one * self.tau_brk, tau_trac_s=one * self.tau_trac,
            p_max_w=one * self.p_max, k_curv_scale=torch.zeros(1, 1, dtype=F64),
            v_eps=one * self.v_eps, v_brake_eps=one * self.v_brake_eps,
            brake_opposes_motion=torch.ones(1, 1, dtype=torch.bool),
            coupler_model=self.coupler_model,
            draft_gear_table=(None if self.draft_gear_table is None
                              else t(self.draft_gear_table)),
            draft_gear_k_transition=self.draft_gear_k_transition,
            draft_gear_k_lock=self.draft_gear_k_lock,
            tau_brk_node=(None if self.tau_brk_node is None
                          else t(np.asarray(self.tau_brk_node, float)[None])),
            tau_brk_release_node=(
                None if self.tau_brk_release_node is None
                else t(np.asarray(self.tau_brk_release_node, float)[None])),
            grade_model=self.grade_model,
            inertia_factor=(None if self.inertia_factor is None
                            else t(np.asarray(self.inertia_factor, float)[None])),
            standstill_model=self.standstill_model,
        )


def rk4_fixed(batch: TorchScenarioBatch, t_end: float, dt: float,
              every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step RK4 on ``torch_rhs`` -- the production scheme, sampled every
    ``every`` steps. Returns ``(t, y)`` with ``y`` of shape ``(T, 4N)``.

    Equivalent to ``rollout_rk4`` on a uniform grid; written out so a test can
    run hundreds of thousands of steps without stacking every output.
    """
    n_steps = int(round(t_end / dt))
    y = batch.y0.clone()
    ts, ys = [0.0], [y[0].numpy().copy()]
    with torch.no_grad():
        for k in range(n_steps):
            t = k * dt
            k1 = torch_rhs(t, y, batch)
            k2 = torch_rhs(t + 0.5 * dt, y + 0.5 * dt * k1, batch)
            k3 = torch_rhs(t + 0.5 * dt, y + 0.5 * dt * k2, batch)
            k4 = torch_rhs(t + dt, y + dt * k3, batch)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if (k + 1) % every == 0:
                ts.append((k + 1) * dt)
                ys.append(y[0].numpy().copy())
    return np.asarray(ts), np.stack(ys)


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------

def coastdown_v(t: np.ndarray, v0: float, m: float, A: float, B: float,
                C: float) -> np.ndarray:
    """Speed under ``m v' = -(A + B v + C v^2)``, ``4AC > B^2``, until rest."""
    q = math.sqrt(4 * A * C - B * B)
    phi0 = math.atan((2 * C * v0 + B) / q)
    return (q * np.tan(phi0 - t * q / (2 * m)) - B) / (2 * C)


def coastdown_stop_time(v0: float, m: float, A: float, B: float, C: float) -> float:
    q = math.sqrt(4 * A * C - B * B)
    return 2 * m / q * (math.atan((2 * C * v0 + B) / q) - math.atan(B / q))


def balancing_speed(F: float, A: float, B: float, C: float,
                    p_max: float = math.inf, m_g_sin: float = 0.0) -> float:
    """Speed where traction equals Davis plus grade.

    Tractive force is ``min(F, p_max / v)`` (above ``v_eps``); the balance is
    the root of a quadratic if ``F`` binds and of a cubic if power binds.
    """
    v_f = (-B + math.sqrt(B * B - 4 * C * (A + m_g_sin - F))) / (2 * C)
    if F * v_f <= p_max:
        return v_f
    roots = np.roots([C, B, A + m_g_sin, -p_max])
    real = roots[np.abs(roots.imag) < 1e-9].real
    return float(real[real > 0].min())


def brake_stop_distance(v0: float, a: float, eps: float) -> float:
    """Distance to rest under ``v' = -a tanh(v/eps)``.

    ``int v dv / (a tanh(v/eps))`` from 0 to v0 is ``v0^2/(2a)`` plus
    ``(eps^2/a) int_0^{v0/eps} x (coth x - 1) dx``, whose integral tends to
    ``pi^2/12`` exponentially fast once ``v0 >> eps``.
    """
    return v0 * v0 / (2 * a) + eps * eps * math.pi ** 2 / (12 * a)


def deadband_oscillator(m1: float, m2: float, excess: float, slack_half: float,
                        k_draft: float, k_buff: float) -> dict:
    """Undamped two-mass coupler released from rest ``excess`` into draft.

    Relative coordinate, reduced mass ``mu``: a half-cycle in draft, free
    flight across the ``2 s`` gap, a half-cycle in buff, and back.
    """
    mu = m1 * m2 / (m1 + m2)
    w_d, w_b = math.sqrt(k_draft / mu), math.sqrt(k_buff / mu)
    u = excess * w_d                        # relative speed through the gap
    period = math.pi / w_d + math.pi / w_b + 2 * (2 * slack_half) / u
    return {"period": period, "buff_amplitude": u / w_b, "gap_speed": u,
            "omega_draft": w_d, "omega_buff": w_b}


def kelvin_voigt_restitution(m1: float, m2: float, k: float, c: float) -> float:
    """Exit/entry relative-speed ratio of a linear spring-dashpot contact that
    ends when penetration returns to zero."""
    mu = m1 * m2 / (m1 + m2)
    zeta = c / (2 * math.sqrt(k * mu))
    return math.exp(-zeta * math.pi / math.sqrt(1 - zeta * zeta))


def chain_modes(n: int, m: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies and shapes of a free-free chain of ``n`` equal masses."""
    j = np.arange(n)
    omega = 2 * math.sqrt(k / m) * np.sin(j * math.pi / (2 * n))
    i = np.arange(n)
    shapes = np.cos(np.outer(i + 0.5, j) * math.pi / n)   # column j is mode j
    return omega, shapes


def restitution_from_run(t: np.ndarray, y: np.ndarray, v_in: float,
                         slack_half: float, L0: float) -> float:
    """Relative exit speed over entry speed for the first draft contact in a
    two-vehicle run, with the exit located by linear interpolation."""
    d = y[:, 0] - y[:, 1] - L0
    dd = y[:, 2] - y[:, 3]
    k = np.where((d[:-1] > slack_half) & (d[1:] <= slack_half))[0][0]
    w = (d[k] - slack_half) / (d[k] - d[k + 1])
    return -float(dd[k] + w * (dd[k + 1] - dd[k])) / v_in


# ---------------------------------------------------------------------------
# a realistic consist, for the checks that need every force law at once
# ---------------------------------------------------------------------------

def realistic_case(n: int = 10, duration_s: float = 120.0,
                   coupler_model: str = "kelvin_voigt") -> Case:
    """A short consist on the default catalog parameters: one powered lead
    unit, rolling grades inside the 4 % clip, and a 60 s traction/brake cycle
    so that couplers run in, out and through the deadband.

    ``coupler_model="draft_gear"`` swaps in the benchmark draft gear with the
    benchmark's 10 mm total coupler slack; everything else is unchanged."""
    from .constants import (DEFAULT_C_BUFF, DEFAULT_C_DRAFT, DEFAULT_K_BUFF,
                            DEFAULT_K_DRAFT, DEFAULT_P_MAX_W, DEFAULT_SLACK_HALF_M)
    m = np.full(n, 1e5)
    m[0] = 1.3e5
    s = np.linspace(-5_000.0, 20_000.0, 2_501)
    st = 0.012 * np.sin(2 * np.pi * s / 3_000) + 0.005 * np.sin(2 * np.pi * s / 900)
    t = np.arange(0.0, duration_s + 0.5, 0.5)
    duty = np.sin(2 * np.pi * t / 60.0)
    u_tr = np.zeros((len(t), n))
    u_tr[:, 0] = np.clip(duty, 0, 1) * 300e3
    u_br = np.repeat((np.clip(-duty, 0, 1) * 60e3)[:, None], n, axis=1)
    return Case(
        mass=m, x0=-18.0 * np.arange(n), v0=np.full(n, 12.0),
        davis=(600.0, 12.0, 0.7), can_traction=np.arange(n) == 0,
        f_trac_max=300e3, f_brk_max=250e3, L0=18.0,
        slack=DEFAULT_SLACK_HALF_M if coupler_model == "kelvin_voigt" else 0.005,
        k_draft=DEFAULT_K_DRAFT, c_draft=DEFAULT_C_DRAFT,
        k_buff=DEFAULT_K_BUFF, c_buff=DEFAULT_C_BUFF,
        sin_theta=(s, st), cmd_t=t, u_trac=u_tr, u_brk=u_br, p_max=DEFAULT_P_MAX_W,
        **_coupler_kwargs(coupler_model),
    )


def _coupler_kwargs(coupler_model: str) -> dict:
    if coupler_model == "kelvin_voigt":
        return {}
    from .draft_gear import load_table
    return {"coupler_model": "draft_gear", "draft_gear_table": load_table()}


# ---------------------------------------------------------------------------
# force decomposition, momentum and energy
# ---------------------------------------------------------------------------

def forces(t: float, y: Tensor, batch: TorchScenarioBatch) -> dict[str, Tensor]:
    """Per-vehicle external forces and per-coupler forces at ``(t, y)``.

    Rebuilt from the force laws as Chapter 4 states them, independently of
    ``torch_rhs``; the momentum check compares the two. Curvature is left out
    because the verification cases run with ``k_curv_scale = 0``.
    """
    n = batch.n_vehicles
    x, v = y[:, :n], y[:, n:2 * n]
    z_brk, z_trac = y[:, 2 * n:3 * n], y[:, 3 * n:4 * n]
    ns, es = batch.node_static, batch.edge_static
    mass = ns[..., NodeStatic.MASS_KG]

    sgn = torch.where(v.abs() < 1e-9, torch.zeros_like(v), torch.sign(v))
    resist = (ns[..., NodeStatic.DAVIS_A] + ns[..., NodeStatic.DAVIS_B] * v.abs()
              + ns[..., NodeStatic.DAVIS_C] * v * v) * sgn
    sin_th = _interp_batched(batch.route_s, batch.route_sin_theta, x)
    grade = mass * GRAVITY_MPS2 * torch.sin(sin_th)
    cap = batch.p_max_w / v.abs().clamp_min(batch.v_eps)
    trac = torch.minimum(z_trac.clamp_min(0), cap) * (ns[..., NodeStatic.CAN_TRACTION] > 0)
    brake = torch.minimum(z_brk.clamp_min(0), ns[..., NodeStatic.F_BRK_MAX_N]) \
        * torch.tanh(v / batch.v_brake_eps)

    delta = x[:, :-1] - x[:, 1:] - es[..., EdgeStatic.L0_M]
    ddot = v[:, :-1] - v[:, 1:]
    s = es[..., EdgeStatic.SLACK_HALF_M]
    in_d, in_b = delta > s, delta < -s
    spring = torch.where(in_d, es[..., EdgeStatic.K_DRAFT] * (delta - s),
                         torch.where(in_b, es[..., EdgeStatic.K_BUFF] * (delta + s),
                                     torch.zeros_like(delta)))
    damper = torch.where(in_d, es[..., EdgeStatic.C_DRAFT] * ddot,
                         torch.where(in_b, es[..., EdgeStatic.C_BUFF] * ddot,
                                     torch.zeros_like(delta)))
    energy = torch.where(in_d, 0.5 * es[..., EdgeStatic.K_DRAFT] * (delta - s) ** 2,
                         torch.where(in_b, 0.5 * es[..., EdgeStatic.K_BUFF] * (delta + s) ** 2,
                                     torch.zeros_like(delta)))
    if batch.coupler_model == "draft_gear":
        # The gear model defines no potential, so the whole connection force
        # is booked as work into the couplers. Over a closed loading cycle
        # that work is what the gear dissipates.
        damper, _ = draft_gear_stateful_torch(
            delta, ddot, y[:, 4 * n:5 * n - 1], s, batch.draft_gear_table,
            k_transition=batch.draft_gear_k_transition,
            k_lock=batch.draft_gear_k_lock)
        spring = torch.zeros_like(delta)
        energy = torch.zeros_like(delta)
    return {"mass": mass, "v": v, "x": x, "resist": resist, "grade": grade,
            "trac": trac, "brake": brake, "spring": spring, "damper": damper,
            "ddot": ddot, "spring_energy": energy}


def momentum_residual(t: float, y: Tensor, batch: TorchScenarioBatch) -> float:
    """``sum m_i v_i'`` from ``torch_rhs`` minus the sum of external forces,
    relative to the largest force present."""
    n = batch.n_vehicles
    dv = torch_rhs(t, y, batch)[:, n:2 * n]
    f = forces(t, y, batch)
    ext = f["trac"] - f["brake"] - f["resist"] - f["grade"]
    lhs = (f["mass"] * dv).sum()
    scale = max(float(ext.abs().max()), float(f["spring"].abs().max()), 1.0)
    return float((lhs - ext.sum()).abs()) / scale


def grade_potential(route_s: np.ndarray, sin_theta: np.ndarray, *, double_sine: bool = True):
    """``Phi(x) = int sin(sin theta)`` (what the model applies) or
    ``int sin theta`` (true height), on a fine grid, as an interpolant."""
    s = np.linspace(route_s[0], route_s[-1], 200_001)
    st = np.interp(s, route_s, sin_theta)
    f = np.sin(st) if double_sine else st
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(s))])
    return lambda x: np.interp(x, s, phi)


def energy_audit(batch: TorchScenarioBatch, t_end: float, dt: float) -> dict:
    """Integrate the state and the four work accumulators with one RK4.

    ``E(t) - E(0)`` from the state must equal traction work minus brake,
    resistance and damper dissipation. Grade is on the energy side, as
    potential. Returns the residual relative to the energy throughput, i.e.
    the largest of the work terms in magnitude.
    """
    n = batch.n_vehicles
    ny = batch.y0.shape[1]     # 4N, plus N - 1 gear states under draft_gear

    def aug(t, ya):
        y = ya[:, :ny]
        f = forces(t, y, batch)
        v = f["v"]
        p = torch.stack([
            (f["trac"] * v).sum(1),
            (f["brake"] * v).sum(1),
            (f["resist"] * v).sum(1),
            (f["damper"] * f["ddot"]).sum(1),
        ], dim=1)
        return torch.cat([torch_rhs(t, y, batch), p], dim=1)

    s_np = batch.route_s[0].numpy()
    phi = grade_potential(s_np, batch.route_sin_theta[0].numpy())
    m = batch.node_static[0, :, NodeStatic.MASS_KG].numpy()

    def energy(y):
        f = forces(0.0, y, batch)
        ke = 0.5 * (m * y[0, n:2 * n].numpy() ** 2).sum()
        pe = GRAVITY_MPS2 * (m * phi(y[0, :n].numpy())).sum()
        return ke + pe + float(f["spring_energy"].sum())

    ya = torch.cat([batch.y0, torch.zeros(1, 4, dtype=F64)], dim=1)
    e0 = energy(batch.y0)
    n_steps = int(round(t_end / dt))
    worst = 0.0
    with torch.no_grad():
        for k in range(n_steps):
            t = k * dt
            k1 = aug(t, ya)
            k2 = aug(t + 0.5 * dt, ya + 0.5 * dt * k1)
            k3 = aug(t + 0.5 * dt, ya + 0.5 * dt * k2)
            k4 = aug(t + dt, ya + dt * k3)
            ya = ya + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            w = ya[0, ny:].numpy()
            resid = energy(ya[:, :ny]) - e0 - (w[0] - w[1] - w[2] - w[3])
            worst = max(worst, abs(resid))
    w = ya[0, ny:].numpy()
    throughput = float(np.abs(w).max())
    return {"work_traction": w[0], "work_brake": w[1], "work_resist": w[2],
            "work_damper": w[3], "energy_change": energy(ya[:, :ny]) - e0,
            "max_residual": worst, "rel_residual": worst / throughput}
