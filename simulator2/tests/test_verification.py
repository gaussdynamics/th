"""The data-generating simulator against exact solutions.

See ``simulator2/verification.py`` for the cases and ``SIMULATOR_VALIDATION_NOTE.md``
for what they establish. Every case runs the production scheme: ``torch_rhs``
under fixed-step RK4 at ``dt = 0.02 s``, in float64.

Where the model does something physically questionable but deliberate, the
test pins the behaviour down instead of skipping it (``test_..._characterized``),
so that changing the physics fails loudly and the note gets updated with it.

Timescales are shortened (light masses) where a case would otherwise need
hundreds of thousands of steps. The closed forms hold for any parameters; what
is being checked is the implementation of the law and its integration at the
production step.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from simulator2.constants import (
    DEFAULT_C_DRAFT,
    DEFAULT_K_BUFF,
    DEFAULT_K_DRAFT,
    DEFAULT_SLACK_HALF_M,
    GRAVITY_MPS2,
)
from simulator2.verification import (
    Case,
    balancing_speed,
    brake_stop_distance,
    chain_modes,
    coastdown_stop_time,
    coastdown_v,
    deadband_oscillator,
    energy_audit,
    kelvin_voigt_restitution,
    momentum_residual,
    realistic_case,
    restitution_from_run,
    rk4_fixed,
)

DT = 0.02  # the production step (scripts/build_dataset.py)


def _single(m: float = 1e5, v0: float = 0.0, **kw) -> Case:
    return Case(mass=np.array([m]), x0=np.array([0.0]), v0=np.array([v0]), **kw)


# ---------------------------------------------------------------------------
# one vehicle: each force law alone
# ---------------------------------------------------------------------------

def test_coastdown_matches_arctangent_solution() -> None:
    A, B, C, m, v0 = 600.0, 12.0, 0.7, 2e3, 25.0
    t_stop = coastdown_stop_time(v0, m, A, B, C)
    t, y = rk4_fixed(_single(m, v0, davis=(A, B, C)).batch(), 0.99 * t_stop, DT, every=10)
    err = np.abs(y[:, 1] - coastdown_v(t, v0, m, A, B, C)).max()
    assert err < 1e-9, err


def test_rest_behaviour_characterized() -> None:
    """Davis resistance takes a hard ``sign(v)``, so it cannot hold a vehicle
    at rest: once stopped, the speed chatters about zero instead of staying
    there. The creep is bounded by one step of the static term, ``A/m * dt``."""
    A, B, C, m, v0 = 600.0, 12.0, 0.7, 2e3, 25.0
    t_stop = coastdown_stop_time(v0, m, A, B, C)
    t, y = rk4_fixed(_single(m, v0, davis=(A, B, C)).batch(), t_stop + 20, DT)
    after = np.abs(y[t > t_stop + 1, 1])
    assert after.max() > 0.0, "vehicle came to exact rest -- update the note"
    assert after.max() < A / m * DT


def test_constant_grade_is_linear_in_time() -> None:
    st = 0.04
    t, y = rk4_fixed(_single(v0=20.0, sin_theta=st).batch(), 30.0, DT, every=50)
    assert np.abs(y[:, 1] - (20.0 - GRAVITY_MPS2 * math.sin(st) * t)).max() < 1e-9


def test_double_sine_grade_characterized() -> None:
    """``_grade_force`` takes ``sin`` of a field that already holds
    ``sin(theta)``. The resulting error is ``-st^2/6`` relative: 2.7e-4 at the
    4 % clip, not the "about 1e-5" of the port report (that is the 1 % value)."""
    st = 0.04
    t, y = rk4_fixed(_single(v0=20.0, sin_theta=st).batch(), 10.0, DT, every=500)
    decel = (20.0 - y[-1, 1]) / t[-1]
    rel = decel / (GRAVITY_MPS2 * st) - 1.0
    assert rel == pytest.approx(-st * st / 6, rel=1e-3)


def test_exact_grade_model_is_m_g_sin_theta() -> None:
    """F7 fixed (physics v3): the route field is sin(theta), used as is."""
    st = 0.04
    t, y = rk4_fixed(_single(v0=20.0, sin_theta=st, grade_model="exact").batch(),
                     30.0, DT, every=50)
    assert np.abs(y[:, 1] - (20.0 - GRAVITY_MPS2 * st * t)).max() < 1e-9


@pytest.mark.parametrize("force, p_max", [(20e3, 1e12), (150e3, 3.5e6)],
                         ids=["force-limited", "power-limited"])
def test_balancing_speed(force: float, p_max: float) -> None:
    v_b = balancing_speed(force, 800.0, 15.0, 0.8, p_max)
    case = _single(2e3, 0.99 * v_b, davis=(800.0, 15.0, 0.8), u_trac=force, p_max=p_max)
    _, y = rk4_fixed(case.batch(), 150.0, DT, every=7500)
    assert y[-1, 1] == pytest.approx(v_b, rel=1e-6)


def test_actuator_lag_is_exponential() -> None:
    case = _single(1e9, u_trac=1e5, z_trac0=np.array([0.0]))
    t, y = rk4_fixed(case.batch(), 20.0, DT)
    assert np.abs(y[:, 3] - 1e5 * (1 - np.exp(-t / 5.0))).max() < 1e-4  # N


def test_brake_stop_distance_with_direction_blend() -> None:
    """The ``tanh(v/eps)`` blend makes the stop asymptotic and adds
    ``eps^2 pi^2 / 12a`` to the textbook ``v0^2 / 2a``."""
    a, v0 = 2.0, 20.0
    _, y = rk4_fixed(_single(v0=v0, u_brk=a * 1e5).batch(), 40.0, DT, every=2000)
    assert y[-1, 0] == pytest.approx(brake_stop_distance(v0, a, 0.1), abs=1e-4)


# ---------------------------------------------------------------------------
# couplers
# ---------------------------------------------------------------------------

def _peak_times(t: np.ndarray, d: np.ndarray, above: float, dt: float) -> np.ndarray:
    i = np.where((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:]) & (d[1:-1] > above))[0] + 1
    y0, y1, y2 = d[i - 1], d[i], d[i + 1]
    return t[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * dt


@pytest.mark.parametrize("dt, tol_period, tol_amp",
                         [(DT, 2e-3, 5e-3), (0.001, 1e-5, 1e-4)],
                         ids=["production-dt", "fine-dt"])
def test_deadband_oscillator(dt: float, tol_period: float, tol_amp: float) -> None:
    """Undamped pair released into draft: half-cycle in draft, free flight
    across the slack, half-cycle in buff. Convergence is only about first
    order -- the deadband edges are kinks -- hence the looser production
    tolerance."""
    m, s, excess, L0 = 1e5, DEFAULT_SLACK_HALF_M, 0.01, 20.0
    exact = deadband_oscillator(m, m, excess, s, DEFAULT_K_DRAFT, DEFAULT_K_BUFF)
    case = Case(mass=np.array([m, m]), x0=np.array([0.0, -(L0 + s + excess)]),
                v0=np.zeros(2), L0=L0, slack=s,
                k_draft=DEFAULT_K_DRAFT, k_buff=DEFAULT_K_BUFF)
    t, y = rk4_fixed(case.batch(), 3 * exact["period"] + 0.1, dt)
    d = y[:, 0] - y[:, 1] - L0
    period = np.diff(_peak_times(t, d, s, dt)).mean()
    assert period / exact["period"] - 1 == pytest.approx(0, abs=tol_period)
    assert -(d.min() + s) / exact["buff_amplitude"] - 1 == pytest.approx(0, abs=tol_amp)


def _contact_run(dt: float):
    m, s, L0, v_in = 1e5, DEFAULT_SLACK_HALF_M, 20.0, 0.2
    case = Case(mass=np.array([m, m]), x0=np.array([0.0, -(L0 + s)]),
                v0=np.array([0.0, -v_in]), L0=L0, slack=s,
                k_draft=DEFAULT_K_DRAFT, c_draft=DEFAULT_C_DRAFT,
                k_buff=DEFAULT_K_BUFF, c_buff=7e5)
    t, y = rk4_fixed(case.batch(), 1.0, dt)
    return m, s, L0, v_in, t, y


def test_damped_contact_restitution() -> None:
    m, s, L0, v_in, t, y = _contact_run(0.001)
    e = restitution_from_run(t, y, v_in, s, L0)
    assert e == pytest.approx(
        kelvin_voigt_restitution(m, m, DEFAULT_K_DRAFT, DEFAULT_C_DRAFT), rel=5e-3)


def test_draft_contact_pushes_characterized() -> None:
    """A spring-dashpot contact that ends when penetration returns to zero
    goes compressive before it ends: the damper term outweighs the spring on
    the way out. The coupler pushes while still stretched past its slack --
    about a quarter of the peak pull -- which a slack knuckle cannot do."""
    _, s, L0, _, _, y = _contact_run(0.001)
    d = y[:, 0] - y[:, 1] - L0
    dd = y[:, 2] - y[:, 3]
    f = np.where(d > s, DEFAULT_K_DRAFT * (d - s) + DEFAULT_C_DRAFT * dd, 0.0)
    ratio = -f.min() / f.max()
    assert 0.2 < ratio < 0.3, ratio


@pytest.mark.parametrize("j", [1, 5, 9])
def test_chain_normal_modes(j: int) -> None:
    """A free-free chain of equal masses, linear couplers without slack. The
    mode must stay in its own shape to rounding, and its phase error must be
    RK4's own dispersion, ``omega * (omega h)^4 / 120`` per unit time."""
    n, m, k, a = 10, 1e5, DEFAULT_K_DRAFT, 0.01
    omega, shapes = chain_modes(n, m, k)
    x_eq = -20.0 * np.arange(n)
    case = Case(mass=np.full(n, m), x0=x_eq + a * shapes[:, j], v0=np.zeros(n),
                L0=20.0, slack=0.0, k_draft=k, k_buff=k)
    t_end = 10 * 2 * math.pi / omega[j]
    t, y = rk4_fixed(case.batch(), t_end, DT)
    u = y[:, :n] - x_eq
    phi = shapes[:, j]
    proj = u @ phi / (phi @ phi)
    assert np.abs(u - np.outer(proj, phi)).max() / a < 1e-8   # no leakage

    predicted_phase = omega[j] * (omega[j] * DT) ** 4 / 120 * t_end
    err = np.abs(u - a * np.outer(np.cos(omega[j] * t), phi)).max() / a
    assert err < 2 * predicted_phase + 1e-6


# ---------------------------------------------------------------------------
# conservation, on every force law at once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def realistic():
    case = realistic_case(duration_s=60.0)
    batch = case.batch()
    t, y = rk4_fixed(batch, 60.0, DT, every=50)
    return batch, t, y


def test_realistic_case_exercises_the_deadband(realistic) -> None:
    """Guard on the fixture: the conservation checks mean little unless the
    couplers actually enter and leave contact."""
    _, _, y = realistic
    d = y[:, :9] - y[:, 1:10] - 18.0
    contact = np.abs(d) > DEFAULT_SLACK_HALF_M
    assert 0.2 < contact.mean() < 0.95
    assert np.any(d > DEFAULT_SLACK_HALF_M) and np.any(d < -DEFAULT_SLACK_HALF_M)


def test_coupler_forces_are_internal(realistic) -> None:
    batch, t, y = realistic
    worst = max(momentum_residual(tt, torch.as_tensor(yy[None]), batch)
                for tt, yy in zip(t, y))
    assert worst < 1e-12, worst


def test_energy_balance_closes(realistic) -> None:
    """Kinetic + grade potential + coupler spring energy must change by
    exactly the traction work less brake, resistance and damper dissipation."""
    batch, _, _ = realistic
    audit = energy_audit(batch, 60.0, DT)
    assert audit["rel_residual"] < 1e-5, audit
    assert audit["work_damper"] > 0.0


# -- F6: standstill holding (physics v3) ------------------------------------------

_R_START = 0.0343        # TorchScenarioBatch.standstill_resistance_npkg


def _standstill(**kw) -> Case:
    base = dict(mass=np.full(1, 1e5), x0=np.zeros(1), v0=np.zeros(1),
                grade_model="exact", standstill_model="breakaway", f_brk_max=1e6,
                davis=(1000.0, 10.0, 1.0))
    return Case(**{**base, **kw})


def test_braked_vehicle_holds_on_a_grade_within_its_static_limit() -> None:
    """Brake 30 kN + starting resistance 3.4 kN against a 1 % grade (9.8 kN):
    held, at the production step, with no drift."""
    t, y = rk4_fixed(_standstill(sin_theta=0.01, u_brk=30e3).batch(), 120.0, DT, every=500)
    assert np.abs(y[:, 1]).max() < 1e-9 and np.abs(y[:, 0]).max() < 1e-9


def test_breakaway_beyond_the_static_limit() -> None:
    """Grade force above brake + starting resistance: the vehicle rolls back,
    at first against the full static limit."""
    m, st, brk = 1e5, 0.02, 10e3
    case = _standstill(sin_theta=st, u_brk=brk)
    t, y = rk4_fixed(case.batch(), 0.5, 0.001, every=10)
    f_static = brk + _R_START * m
    a0 = -(m * GRAVITY_MPS2 * st - f_static) / m
    early = t < 0.3                                   # still inside the 0.1 m/s window
    assert np.abs(y[early, 1] - a0 * t[early]).max() < 1e-9


def test_unbraked_vehicle_rests_without_chatter() -> None:
    """Legacy chatters about v = 0 under the hard sign (characterized above);
    breakaway holds a vehicle at rest on the flat exactly."""
    t, y = rk4_fixed(_standstill().batch(), 60.0, DT, every=50)
    assert np.abs(y[:, 1]).max() == 0.0


def test_braked_stop_stays_stopped_on_a_grade() -> None:
    """A braked vehicle stops on an upgrade and stays there. Legacy never
    stands still: the tanh blend balances the grade at a steady backward
    creep (~3 cm/s here, about 2 m a minute)."""
    kw = dict(v0=np.full(1, 5.0), sin_theta=0.01, u_brk=30e3)
    _, y3 = rk4_fixed(_standstill(**kw).batch(), 60.0, DT, every=50)
    _, yl = rk4_fixed(_standstill(standstill_model="none", **kw).batch(), 60.0, DT, every=50)
    assert y3[-1, 1] == pytest.approx(0.0, abs=1e-9) and y3[:, 1].min() > -1e-6
    assert -0.1 < yl[-1, 1] < -0.01 and yl[-1, 0] < yl[-10, 0]    # legacy creeps back


def test_stiction_friction_never_exceeds_the_static_limit() -> None:
    """Sliding inside the standstill window, the friction is the static limit
    (Coulomb), not the regularisation's m v / tau: 0.0343 m/s^2, not 0.5."""
    t, y = rk4_fixed(_standstill(v0=np.full(1, 0.05), davis=(0.0, 0.0, 0.0)).batch(),
                     1.0, 0.001, every=10)
    moving = y[:, 1] > 0.01
    assert np.abs(y[moving, 1] - (0.05 - _R_START * t[moving])).max() < 1e-9
