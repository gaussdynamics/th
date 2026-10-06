"""The empirical air brake (:mod:`simulator2.air_brake`).

Command-side laws are checked against their definitions; the integrated
response against closed forms: uncoupled vehicles brake exactly as
``F_max (1 - exp(-(t - d) / tau))`` from their own onset, and with couplers
the train's momentum still follows the summed brake force exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simulator2.air_brake import (
    WEI2017_APPLY_DELAY_S,
    WEI2017_RELEASE_DELAY_S,
    WEI2017_SPAN_M,
    WEI2017_TAU_APPLY_S,
    AirBrakeParams,
    PHYSICAL_FIELDS,
    brake_capacity_n,
    direct_release,
    onset_delay_s,
    release_delay_s,
    require_sourced,
    rise_tau_s,
    unsourced,
    vehicle_commands,
)
from simulator2.constants import GRAVITY_MPS2 as G
from simulator2.verification import Case, rk4_fixed

P = AirBrakeParams()


# -- brake force ------------------------------------------------------------------

def test_capacity_is_fixed_per_car_and_capped_at_adhesion() -> None:
    mass = np.array([P.car_design_gross_kg, 20_000.0, 60_000.0, 180_000.0])
    loco = np.array([False, False, False, True])
    f = brake_capacity_n(mass, loco, P)
    design = P.shoe_friction * P.net_braking_ratio * P.car_design_gross_kg * G
    assert f[0] == pytest.approx(design)                     # loaded car
    assert f[2] == pytest.approx(design)                     # part load: same shoe force
    assert f[1] == pytest.approx(P.adhesion * 20_000.0 * G)  # light empty: adhesion binds
    assert f[1] < design
    assert f[3] == pytest.approx(P.shoe_friction * P.net_braking_ratio * 180_000.0 * G)
    assert np.all(f <= P.adhesion * mass * G + 1e-9)


def test_force_parameters_are_still_flagged_as_placeholders() -> None:
    assert unsourced(P) == ["car_design_gross_kg", "net_braking_ratio",
                            "shoe_friction", "adhesion"]
    with pytest.raises(ValueError, match="net_braking_ratio"):
        require_sourced(P)
    require_sourced(AirBrakeParams(sources={f: "test" for f in PHYSICAL_FIELDS}))


def test_defaults_reproduce_the_wei_2017_measurements() -> None:
    """At the 116th car (112 car lengths behind the 4th) the defaults give
    back the measured onset, release and build-up figures."""
    d = np.array([0.0, WEI2017_SPAN_M])
    assert np.diff(onset_delay_s(d, P))[0] == pytest.approx(np.mean(WEI2017_APPLY_DELAY_S))
    assert np.diff(release_delay_s(d, P))[0] == pytest.approx(WEI2017_RELEASE_DELAY_S)
    assert rise_tau_s(d, P) == pytest.approx(WEI2017_TAU_APPLY_S)
    assert 150.0 < P.propagation_mps < 200.0
    assert P.release_propagation_mps < P.propagation_mps


# -- propagation and build-up -----------------------------------------------------

def test_onset_delay_and_rise_time_along_the_train() -> None:
    d = np.array([0.0, 100.0, 500.0, 1000.0])
    assert onset_delay_s(d, P) == pytest.approx(d / P.propagation_mps)
    assert release_delay_s(d, P) == pytest.approx(d / P.release_propagation_mps)
    tau = rise_tau_s(d, P)
    assert tau[0] == pytest.approx(P.tau_head_s)
    assert np.allclose(np.diff(tau) / np.diff(d), P.tau_per_m_s)
    # per metre, not per train: a short train's tail is not a long train's tail
    assert rise_tau_s(d[:2], P)[-1] == pytest.approx(tau[1])


def test_each_vehicle_sees_the_driver_command_shifted_by_its_delay() -> None:
    t = np.arange(0.0, 20.0, 0.01)
    driver = np.where(t >= 1.0, 0.7, 0.0)
    x0 = -15.0 * np.arange(60)
    mass = np.full(60, 100_000.0)
    u, tau, tau_rel, f_max = vehicle_commands(t, driver, x0, mass, np.zeros(60, bool), P)
    for i in (0, 20, 59):
        onset = 1.0 + 15.0 * i / P.propagation_mps
        assert np.all(u[t < onset - 0.01, i] == 0.0)
        assert np.allclose(u[t >= onset + 0.01, i], 0.7 * f_max[i])
    assert tau[0] == pytest.approx(P.tau_head_s)
    assert tau[-1] == pytest.approx(P.tau_head_s + P.tau_per_m_s * 15.0 * 59)
    assert np.all(tau_rel == P.tau_release_s)


def test_release_travels_with_the_slower_release_wave() -> None:
    t = np.arange(0.0, 60.0, 0.01)
    driver = np.where((t >= 1.0) & (t < 30.0), 0.7, 0.0)
    x0 = -15.0 * np.arange(60)
    u, *_, f_max = vehicle_commands(t, driver, x0, np.full(60, 1e5), np.zeros(60, bool), P)
    for i in (0, 20, 59):
        on = 1.0 + 15.0 * i / P.propagation_mps
        off = 30.0 + 15.0 * i / P.release_propagation_mps
        held = (t >= on + 0.01) & (t < off - 0.01)
        assert np.allclose(u[held, i], 0.7 * f_max[i])
        assert np.all(u[t >= off + 0.01, i] == 0.0)


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

def _braking_case(n: int, p: AirBrakeParams, **coupler) -> tuple[Case, np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0.0, 40.0, 0.005)
    driver = np.where(t >= 1.0, 1.0, 0.0)
    x0 = -15.0 * np.arange(n)
    mass = np.full(n, 100_000.0)
    u, tau, _, f_max = vehicle_commands(t, driver, x0, mass, np.zeros(n, bool), p)
    case = Case(mass=mass, x0=x0, v0=np.full(n, 20.0), L0=15.0, cmd_t=t,
                u_trac=np.zeros_like(u), u_brk=u, f_brk_max=1e12, tau_brk_node=tau,
                **coupler)
    # The step reaches the RHS as a linear ramp across one 5 ms command
    # sample, whose impulse equals a step at the ramp's midpoint.
    return case, tau, f_max, onset_delay_s(-x0, p) + 1.0 - 0.0025


def _impulse(t: np.ndarray, f_max: float, tau: float, onset: float) -> np.ndarray:
    """``int_0^t F_max (1 - exp(-(s - onset) / tau)) ds`` for ``s > onset``."""
    s = np.clip(t - onset, 0.0, None)
    return f_max * (s - tau * (1.0 - np.exp(-s / tau)))


def test_uncoupled_vehicles_follow_their_own_build_up_exactly() -> None:
    n = 40
    case, tau, f_max, onset = _braking_case(n, P)
    t, y = rk4_fixed(case.batch(), 30.0, 0.005, every=200)
    v = y[:, n:2 * n]
    for i in (0, 13, n - 1):
        exact = 20.0 - _impulse(t, f_max[i], tau[i], onset[i]) / 100_000.0
        assert np.abs(v[:, i] - exact).max() < 1e-5


def test_coupled_train_momentum_follows_the_summed_brake_force() -> None:
    n = 40
    case, tau, f_max, onset = _braking_case(n, P, slack=0.0127, k_draft=9e6,
                                            c_draft=5e5, k_buff=12e6, c_buff=7e5)
    t, y = rk4_fixed(case.batch(), 30.0, 0.005, every=200)
    momentum = (100_000.0 * y[:, n:2 * n]).sum(axis=1)
    exact = 40 * 100_000.0 * 20.0 - sum(_impulse(t, f_max[i], tau[i], onset[i]) for i in range(n))
    assert np.abs(momentum - exact).max() / (40 * 100_000.0 * 20.0) < 1e-7


def test_propagation_raises_the_run_in_force() -> None:
    """The point of F2: the head brakes first and the tail runs in. The same
    application, propagated, compresses the couplers harder than an
    instantaneous one."""
    n = 60
    kw = dict(slack=0.0127, k_draft=9e6, c_draft=5e5, k_buff=12e6, c_buff=7e5)
    peaks = []
    for c in (P.propagation_mps, 1e9):
        p = AirBrakeParams(propagation_mps=c, release_propagation_mps=min(c, 1e8),
                           tau_per_m_s=0.0)
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
