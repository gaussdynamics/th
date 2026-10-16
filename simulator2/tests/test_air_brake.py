"""The empirical air brake (:mod:`simulator2.air_brake`).

The brake pipe solution is checked against an independent finite-difference
solve and the diffusion time scale; the control valve against its closed
form at the head; the calibration against the digitized curves of
Spiryagin et al. (2017). The integrated response is checked exactly: the RHS
follows each vehicle's command through the actuator lag, uncoupled vehicles
lose exactly that impulse, and with couplers the train's momentum still
follows the summed brake force.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.linalg import solve_banded

from simulator2.air_brake import (
    AirBrakeParams,
    PHYSICAL_FIELDS,
    brake_capacity_n,
    cylinder_level,
    direct_release,
    onset_delay_s,
    pipe_modes,
    release_delay_s,
    require_sourced,
    unsourced,
    vehicle_commands,
)
from simulator2.constants import GRAVITY_MPS2 as G
from simulator2.verification import Case, rk4_fixed

P = AirBrakeParams()
DATA = Path(__file__).resolve().parents[1] / "data/spiryagin2017_air_brake"


# -- brake force ------------------------------------------------------------------

def test_capacity_is_fixed_per_car_and_capped_for_empty_cars() -> None:
    mass = np.array([P.car_design_gross_kg, 26_000.0, 60_000.0, 180_000.0])
    loco = np.array([False, False, False, True])
    f = brake_capacity_n(mass, loco, P)
    design = P.shoe_friction * P.net_braking_ratio * P.car_design_gross_kg * G
    assert f[0] == pytest.approx(design)                     # loaded car
    assert f[2] == pytest.approx(design)                     # part load: same shoe force
    # empty: shoe force at the AAR empty maximum on light weight
    assert f[1] == pytest.approx(P.shoe_friction * P.max_empty_braking_ratio * 26_000.0 * G)
    assert f[1] < design
    assert f[3] == pytest.approx(P.shoe_friction * P.net_braking_ratio * 180_000.0 * G)
    assert np.all(f <= P.adhesion * mass * G + 1e-9)


def test_capacity_is_capped_at_adhesion() -> None:
    p = replace(P, adhesion=0.02)  # below the loaded demand, 0.30 x 0.10
    mass = np.array([26_000.0, P.car_design_gross_kg])
    f = brake_capacity_n(mass, np.array([False, False]), p)
    np.testing.assert_allclose(f, 0.02 * mass * G)


def test_default_parameters_are_all_sourced() -> None:
    assert unsourced(P) == []
    require_sourced(P)
    with pytest.raises(ValueError, match="net_braking_ratio"):
        require_sourced(replace(P, sources={k: v for k, v in P.sources.items()
                                            if k != "net_braking_ratio"}))
    require_sourced(AirBrakeParams(sources={f: "test" for f in PHYSICAL_FIELDS}))


# -- brake pipe ---------------------------------------------------------------------

def _fd_pipe(t: np.ndarray, level: np.ndarray, x_out: np.ndarray, length: float,
             d: float, vent_tail: bool, dx: float = 5.0) -> np.ndarray:
    """Implicit-Euler finite differences on a finer grid: the reference."""
    n = int(round(length / dx)) + 1
    x = np.linspace(0.0, length, n)
    u, out = np.full(n, level[0]), [np.interp(x_out, x, np.full(n, level[0]))]
    for k in range(1, t.size):
        a = d * (t[k] - t[k - 1]) / (x[1] - x[0]) ** 2
        ab = np.zeros((3, n))
        ab[0, 1:], ab[1], ab[2, :-1] = -a, 1 + 2 * a, -a
        ab[2, -2] = -2 * a                        # mirror node at the far end
        rhs = u.copy()
        ab[1, 0], ab[0, 1], rhs[0] = 1.0, 0.0, level[k]
        if vent_tail:
            ab[1, -1], ab[2, -2], rhs[-1] = 1.0, 0.0, level[k]
        u = solve_banded((1, 1), ab, rhs)
        out.append(np.interp(x_out, x, u))
    return np.array(out)


@pytest.mark.parametrize("vent_tail", [False, True])
def test_pipe_modes_match_finite_differences(vent_tail: bool) -> None:
    t = np.arange(0.0, 120.0, 0.05)
    level = np.where(t < 1.0, 0.0, np.where(t < 40.0, 0.4, 1.0))
    x = np.array([0.0, 30.0, 700.0, 1300.0, 2000.0])
    tau, gain = pipe_modes(x, 2000.0, P.pipe_diffusivity_m2ps, vent_tail, dx_m=10.0)
    assert np.allclose(gain.sum(axis=1), 1.0)
    modal = np.zeros((t.size, x.size))
    for tk, g in zip(tau, gain.T):
        y = np.empty_like(t)
        y[0] = level[0]
        for k in range(1, t.size):              # same implicit-Euler step, per mode
            y[k] = level[k] if tk == 0 else (y[k - 1] + level[k] * 0.05 / tk) / (1 + 0.05 / tk)
        modal += np.outer(y, g)
    ref = _fd_pipe(t, level, x, 2000.0, P.pipe_diffusivity_m2ps, vent_tail)
    assert np.abs(modal - ref).max() < 0.01


def test_pipe_lag_scales_with_length_squared_and_venting() -> None:
    d = P.pipe_diffusivity_m2ps
    slow = {length: pipe_modes(np.zeros(1), length, d)[0].max() for length in (1000.0, 2000.0)}
    assert slow[2000.0] == pytest.approx(4 * 2000.0 ** 2 / (np.pi ** 2 * d), rel=1e-3)
    assert slow[2000.0] / slow[1000.0] == pytest.approx(4.0, rel=1e-3)
    both = pipe_modes(np.zeros(1), 2000.0, d, vent_tail=True)[0].max()
    assert both == pytest.approx(slow[1000.0], rel=1e-3)        # vented at both ends


# -- control valve and front --------------------------------------------------------

def test_head_cylinder_follows_inshot_then_service_rate() -> None:
    """At the exhaust the pipe follows the driver at once, so the head cylinder
    is the valve alone: inshot rate up to the inshot level, then the service
    rate, up to the level."""
    t = np.arange(0.0, 40.0, P.dt_s)
    for level in (125.0 / 350.0, 1.0):
        cyl = cylinder_level(t, np.where(t > 0.0, level, 0.0), np.zeros(1), P)[:, 0]
        t_in = P.inshot_frac / P.inshot_rate_per_s
        exact = np.where(t < t_in, P.inshot_rate_per_s * t,
                         P.inshot_frac + P.service_rate_per_s * (t - t_in))
        exact = np.minimum(np.clip(exact, 0.0, None), level)
        assert np.abs(cyl - exact).max() < P.inshot_rate_per_s * P.dt_s + 1e-9


def test_front_arrives_at_the_propagation_speed() -> None:
    t = np.arange(0.0, 20.0, 0.01)
    driver = np.where(t >= 1.0, 1.0, 0.0)
    x0 = -15.0 * np.arange(100)
    u, tau, tau_rel, f_max = vehicle_commands(t, driver, x0, np.full(100, 1e5),
                                              np.zeros(100, bool), P)
    for i in (0, 40, 99):
        onset = 1.0 + 15.0 * P.pipe_length_factor * i / P.propagation_mps
        assert np.all(u[t < onset - P.dt_s, i] == 0.0)
        assert np.all(u[t > onset + 2 * P.dt_s, i] > 0.0)
    assert np.all(tau == P.actuator_tau_s) and np.all(tau_rel == P.tau_release_s)


def test_cylinders_lag_the_pipe_and_saturate_with_distance() -> None:
    """Behind the head, cylinders are pipe-limited: they never exceed the pipe
    reduction, never fall while applied, and the lag grows less from mid-train
    to the tail than from the head to mid-train."""
    t = np.arange(0.0, 300.0, P.dt_s)
    level = np.where(t > 0.0, 1.0, 0.0)
    x = np.array([0.0, 1000.0, 2000.0])
    cyl = cylinder_level(t, level, x, P)
    tau, gain = pipe_modes(x, 2000.0, P.pipe_diffusivity_m2ps, dx_m=P.pipe_dx_m)
    a = np.exp(-P.dt_s / np.where(tau > 0, tau, 1.0))
    pipe = np.zeros((t.size, len(tau)))
    for k in range(1, t.size):
        pipe[k] = np.where(tau > 0, a * pipe[k - 1] + (1 - a) * level[k], level[k])
    w = np.maximum(pipe @ gain.T, P.quick_service_frac)
    assert np.all(cyl <= w + 1e-9)
    assert np.all(np.diff(cyl, axis=0) >= -1e-12)
    t50 = [t[np.argmax(cyl[:, i] >= 0.5)] for i in range(3)]
    assert t50[2] - t50[1] < t50[1] - t50[0]


def test_defaults_reproduce_spiryagin_full_service() -> None:
    """Regression on the calibration (scripts/calibrate_air_brake.py):
    Fig. 5.67 (b), head-end train, cylinders at 16, 996 and 1996 m."""
    assert 250.0 <= P.propagation_mps <= 300.0
    assert P.release_propagation_mps == pytest.approx(162.52)   # Wu et al. (2017)
    assert P.release_propagation_mps < P.propagation_mps
    d = pd.read_csv(DATA / "fig5_67b.csv")
    t = np.arange(0.0, 90.0 + P.dt_s / 2, P.dt_s)
    pos = (16, 996, 1996)
    cyl = cylinder_level(t, np.where(t > 0.0, 1.0, 0.0), np.array(pos, float), P,
                         length_m=2000.0) * 350.0
    for i, x in enumerate(pos):
        c = d[d.curve == f"BC{x}"]
        rms = np.sqrt(np.mean((np.interp(c.t_s, t, cyl[:, i]) - c.p_kPa) ** 2))
        assert rms < 30.0, (x, rms)


def test_release_travels_with_the_slower_release_wave() -> None:
    t = np.arange(0.0, 60.0, 0.01)
    driver = np.where((t >= 1.0) & (t < 30.0), 0.7, 0.0)
    x0 = -15.0 * np.arange(60)
    u, *_ = vehicle_commands(t, driver, x0, np.full(60, 1e5), np.zeros(60, bool), P)
    for i in (0, 20, 59):
        off = 30.0 + 15.0 * P.pipe_length_factor * i / P.release_propagation_mps
        held = (t > 20.0) & (t < off - P.dt_s)
        assert np.all(u[held, i] > 0.0)
        assert np.all(np.diff(u[held, i]) >= -1e-9)           # no partial release
        assert np.all(u[t >= off + P.dt_s, i] == 0.0)


def test_pipe_length_factor_stretches_the_pipe() -> None:
    """A pipe k times the train length: each vehicle brakes as if it sat k
    times further back, front and pipe lag alike."""
    t = np.arange(0.0, 120.0, 0.05)
    driver = np.where(t >= 1.0, 1.0, 0.0)
    x0 = -15.0 * np.arange(80)
    mass, loco = np.full(80, 1e5), np.zeros(80, bool)
    p2 = replace(P, pipe_length_factor=2.0)
    u, *_, f_max = vehicle_commands(t, driver, x0, mass, loco, p2)
    level = direct_release(t, driver, p2)
    ref = cylinder_level(t, level, 2.0 * 15.0 * np.arange(80), p2) * f_max[None, :]
    assert np.array_equal(u, ref)
    for i in (20, 79):
        onset = 1.0 + 2.0 * 15.0 * i / P.propagation_mps
        assert np.all(u[t < onset - P.dt_s, i] == 0.0)
        assert np.all(u[t > onset + 2 * P.dt_s, i] > 0.0)
    u1, *_ = vehicle_commands(t, driver, x0, mass, loco, P)
    assert u[:, -1].sum() < u1[:, -1].sum()                    # longer pipe, later tail


def test_onset_and_release_delays() -> None:
    d = np.array([0.0, 100.0, 500.0, 1000.0])
    assert onset_delay_s(d, P) == pytest.approx(d / P.propagation_mps)
    assert release_delay_s(d, P) == pytest.approx(d / P.release_propagation_mps)


# -- direct release ---------------------------------------------------------------

def _run(cmds: list[tuple[float, float]], t_end: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, t_end, 0.5)
    f = np.zeros_like(t)
    for t0, level in cmds:
        f[t >= t0] = level
    return t, direct_release(t, f, P)


def test_graduated_application_passes_through() -> None:
    t, out = _run([(10, 0.2), (20, 0.4), (30, 0.8)])
    assert out[t == 15][0] == 0.2 and out[t == 25][0] == 0.4 and out[t == 35][0] == 0.8


def test_a_small_dip_holds_the_peak() -> None:
    t, out = _run([(10, 0.5), (20, 0.5 - P.release_tol / 2)])
    assert np.all(out[t >= 20] == 0.5)


def test_any_real_reduction_releases_fully_until_recharged() -> None:
    t, out = _run([(10, 0.6), (40, 0.3)])
    assert np.all(out[(t >= 40) & (t < 40 + P.recharge_s)] == 0.0)
    assert np.all(out[t >= 40 + P.recharge_s] == 0.3)        # re-applies at the command


def test_reapplication_waits_for_recharge() -> None:
    t, out = _run([(10, 0.6), (40, 0.0), (50, 0.5)])
    ready = 40 + P.recharge_s
    assert np.all(out[(t >= 40) & (t < ready)] == 0.0)
    assert np.all(out[t >= ready] == 0.5)


def test_release_can_be_turned_off() -> None:
    t = np.arange(0.0, 50.0, 0.5)
    driver = np.where(t >= 10, 0.6, 0.0) - np.where(t >= 30, 0.3, 0.0)
    p = AirBrakeParams(direct_release=False)
    u, *_, f_max = vehicle_commands(t, driver, np.zeros(1), np.full(1, 1e5), np.zeros(1, bool), p)
    assert u[t == 35, 0][0] == pytest.approx(0.3 * f_max[0])


# -- integrated response ----------------------------------------------------------

def _braking_case(n: int, p: AirBrakeParams, **coupler) -> tuple[Case, np.ndarray, np.ndarray]:
    t = np.arange(0.0, 40.0, 0.005)
    driver = np.where(t >= 1.0, 1.0, 0.0)
    x0 = -15.0 * np.arange(n)
    mass = np.full(n, 100_000.0)
    u, tau, _, _ = vehicle_commands(t, driver, x0, mass, np.zeros(n, bool), p)
    case = Case(mass=mass, x0=x0, v0=np.full(n, 20.0), L0=15.0, cmd_t=t,
                u_trac=np.zeros_like(u), u_brk=u, f_brk_max=1e12, tau_brk_node=tau,
                **coupler)
    return case, t, u


def _lagged_impulse(t_cmd: np.ndarray, u: np.ndarray, tau: float,
                    t_out: np.ndarray) -> np.ndarray:
    """Exact ``int z`` for ``z' = (u - z) / tau``, ``u`` piecewise linear, ``z(0) = u(0)``.

    Per segment of slope ``s`` and length ``h``:
    ``z1 = u1 - tau s + (z0 - u0 + tau s) e^(-h/tau)`` and
    ``int z = (u0 + u1) h / 2 - tau (z1 - z0)``.
    """
    h = np.diff(t_cmd)[:, None]
    s = np.diff(u, axis=0) / h
    e = np.exp(-h / tau)
    z = np.empty_like(u)
    z[0] = u[0]
    for k in range(1, len(t_cmd)):
        z[k] = u[k] - tau * s[k - 1] + (z[k - 1] - u[k - 1] + tau * s[k - 1]) * e[k - 1]
    seg = (u[:-1] + u[1:]) * h / 2 - tau * np.diff(z, axis=0)
    cum = np.vstack([np.zeros((1, u.shape[1])), np.cumsum(seg, axis=0)])
    return np.stack([np.interp(t_out, t_cmd, cum[:, i]) for i in range(u.shape[1])], axis=1)


def test_uncoupled_vehicles_lose_exactly_their_lagged_command_impulse() -> None:
    n = 40
    case, t_cmd, u = _braking_case(n, P)
    t, y = rk4_fixed(case.batch(), 30.0, 0.005, every=200)
    exact = 20.0 - _lagged_impulse(t_cmd, u, P.actuator_tau_s, t) / 100_000.0
    assert np.abs(y[:, n:2 * n] - exact).max() < 1e-5


def test_coupled_train_momentum_follows_the_summed_brake_force() -> None:
    n = 40
    case, t_cmd, u = _braking_case(n, P, slack=0.0127, k_draft=9e6,
                                   c_draft=5e5, k_buff=12e6, c_buff=7e5)
    t, y = rk4_fixed(case.batch(), 30.0, 0.005, every=200)
    momentum = (100_000.0 * y[:, n:2 * n]).sum(axis=1)
    exact = n * 100_000.0 * 20.0 - _lagged_impulse(t_cmd, u, P.actuator_tau_s, t).sum(axis=1)
    assert np.abs(momentum - exact).max() / (n * 100_000.0 * 20.0) < 1e-7


def test_propagation_raises_the_run_in_force() -> None:
    """The point of F2: the head brakes first and the tail runs in. The same
    application, propagated, compresses the couplers harder than one that
    reaches every car at once."""
    n = 60
    kw = dict(slack=0.0127, k_draft=9e6, c_draft=5e5, k_buff=12e6, c_buff=7e5)
    peaks = []
    for p in (P, replace(P, propagation_mps=1e9, release_propagation_mps=1e8,
                         pipe_diffusivity_m2ps=1e12)):
        case, *_ = _braking_case(n, p, **kw)
        _, y = rk4_fixed(case.batch(), 20.0, 0.005, every=4)
        x = y[:, :n]
        delta = x[:, :-1] - x[:, 1:] - 15.0
        buff = np.clip(-(delta + 0.0127), 0.0, None) * 12e6
        peaks.append(buff.max())
    assert peaks[0] > 2.0 * peaks[1]


# -- the RHS hook -----------------------------------------------------------------

def test_uniform_tau_node_matches_the_scalar_path() -> None:
    n = 8
    base = dict(mass=np.full(n, 8e4), x0=-15.0 * np.arange(n), v0=np.full(n, 15.0),
                L0=15.0, slack=0.0127, k_draft=9e6, c_draft=5e5, k_buff=12e6,
                c_buff=7e5, u_brk=np.linspace(1e4, 9e4, n), z_brk0=np.zeros(n),
                f_brk_max=1e6, tau_brk=4.0)
    _, a = rk4_fixed(Case(**base).batch(), 10.0, 0.02)
    _, b = rk4_fixed(Case(**base, tau_brk_node=np.full(n, 4.0)).batch(), 10.0, 0.02)
    assert np.array_equal(a, b)


def test_cylinder_exhausts_with_the_release_time_constant() -> None:
    """An uncoupled vehicle braked to its command and released: the force
    builds with ``tau_apply`` and then decays as ``exp(-t / tau_release)``."""
    tau_a, tau_r, f = 4.0, 1.5, 5e4
    dt = 0.005
    t = np.arange(0.0, 40.0, dt)
    u = np.where(t < 20.0, f, 0.0)[:, None]
    case = Case(mass=np.full(1, 1e5), x0=np.zeros(1), v0=np.full(1, 30.0), cmd_t=t,
                u_trac=np.zeros_like(u), u_brk=u, z_brk0=np.zeros(1), f_brk_max=1e12,
                tau_brk_node=np.full(1, tau_a), tau_brk_release_node=np.full(1, tau_r))
    ts, y = rk4_fixed(case.batch(), 30.0, dt, every=100)
    z = y[:, 2]
    # the release edge reaches the RHS as a one-sample ramp centred at 20 - dt/2
    t_rel = 20.0 - dt / 2
    z_rel = f * (1.0 - np.exp(-t_rel / tau_a))
    up, down = ts < 19.9, ts > 20.1
    assert np.abs(z[up] - f * (1.0 - np.exp(-ts[up] / tau_a))).max() < 1e-6 * f
    assert np.abs(z[down] - z_rel * np.exp(-(ts[down] - t_rel) / tau_r)).max() < 1e-3 * f


def test_release_tau_equal_to_apply_tau_matches_no_release_tau() -> None:
    n = 8
    t = np.arange(0.0, 10.0, 0.02)
    u = np.where(t < 5.0, 1.0, 0.0)[:, None] * np.linspace(1e4, 9e4, n)[None]
    base = dict(mass=np.full(n, 8e4), x0=-15.0 * np.arange(n), v0=np.full(n, 15.0),
                L0=15.0, slack=0.0127, k_draft=9e6, c_draft=5e5, k_buff=12e6,
                c_buff=7e5, cmd_t=t, u_trac=np.zeros_like(u), u_brk=u,
                f_brk_max=1e6, tau_brk_node=np.full(n, 4.0))
    _, a = rk4_fixed(Case(**base).batch(), 10.0, 0.02)
    _, b = rk4_fixed(Case(**base, tau_brk_release_node=np.full(n, 4.0)).batch(), 10.0, 0.02)
    assert np.array_equal(a, b)


def test_tau_node_shape_is_checked() -> None:
    n = 4
    case = Case(mass=np.full(n, 8e4), x0=-15.0 * np.arange(n), v0=np.zeros(n),
                tau_brk_node=np.ones(n))
    batch = case.batch()
    fields = {f: getattr(batch, f) for f in batch.__dataclass_fields__}
    for name in ("tau_brk_node", "tau_brk_release_node"):
        with pytest.raises(ValueError, match=name):
            type(batch)(**{**fields, name: torch.ones(1, n + 1, dtype=torch.float64)})
