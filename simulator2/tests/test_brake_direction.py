"""Brake force must oppose the direction of travel.

Regression guard for a bug that reached the thesis: brake force was applied as
an unconditional negative-x term, so a vehicle braked to a standstill was
accelerated in reverse without bound. The Chapter 4 Stage 7 figure was showing
the lead vehicle reaching -99 km/h under braking as a result.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator2.constants import DEFAULT_V_BRAKE_EPS
from simulator2.params import CouplerParameters, VehicleParameters
from simulator2.rhs import train_rhs_extended, train_rhs_tensorized
from simulator2.route import RouteProfile

N = 5


def _setup():
    vehicles = [
        VehicleParameters(
            mass_kg=130_000.0 if i == 0 else 100_000.0,
            davis_A=800.0, davis_B=15.0, davis_C=0.8,
            can_traction=(i == 0),
            F_trac_max_N=600_000.0 if i == 0 else 0.0,
            F_brk_max_N=250_000.0,
        )
        for i in range(N)
    ]
    couplers = [
        CouplerParameters(L0_m=20.0, slack_half_m=0.02, k_draft=9.0e6,
                          c_draft=5.0e5, k_buff=12.0e6, c_buff=7.0e5)
        for _ in range(N - 1)
    ]
    s = np.linspace(0.0, 20_000.0, 2001)
    route = RouteProfile(s, np.zeros_like(s), np.zeros_like(s), np.full_like(s, 25.0))
    return vehicles, couplers, route


def _state(v_scalar: float) -> np.ndarray:
    x = np.array([1000.0 - 20.0 * i for i in range(N)])
    v = np.full(N, v_scalar)
    z_brk = np.full(N, 60_000.0)
    z_trac = np.zeros(N)
    return np.concatenate([x, v, z_brk, z_trac])


def _rhs(y, *, flag, tensorized=True):
    vehicles, couplers, route = _setup()
    fn = train_rhs_tensorized if tensorized else train_rhs_extended
    return fn(
        0.0, y, vehicles, couplers, route,
        lambda t, i: 0.0, lambda t, i: 60_000.0,
        0.0, 3.0, 5.0, 3.5e6,
        brake_opposes_motion=flag,
    )


@pytest.mark.parametrize("tensorized", [True, False])
def test_brake_does_not_accelerate_a_stopped_train(tensorized: bool) -> None:
    dv = _rhs(_state(0.0), flag=True, tensorized=tensorized)[N : 2 * N]
    assert np.all(dv >= -1e-9), f"brake drives a stopped consist backwards: {dv}"


@pytest.mark.parametrize("tensorized", [True, False])
def test_the_bug_is_reproducible_when_disabled(tensorized: bool) -> None:
    """The flag must actually change something, or the guard is vacuous."""
    dv = _rhs(_state(0.0), flag=False, tensorized=tensorized)[N : 2 * N]
    assert np.all(dv < 0.0)


@pytest.mark.parametrize("tensorized", [True, False])
def test_brake_decelerates_when_moving_forward(tensorized: bool) -> None:
    dv = _rhs(_state(12.0), flag=True, tensorized=tensorized)[N : 2 * N]
    assert np.all(dv < 0.0)


@pytest.mark.parametrize("tensorized", [True, False])
def test_brake_decelerates_when_moving_backward(tensorized: bool) -> None:
    """Rolling backwards down a grade, the brake must push forward."""
    dv = _rhs(_state(-12.0), flag=True, tensorized=tensorized)[N : 2 * N]
    assert np.all(dv > 0.0)


@pytest.mark.parametrize("tensorized", [True, False])
def test_no_effect_at_operating_speed(tensorized: bool) -> None:
    """tanh(v / 0.1) is 1.0 to machine precision well before track speed, so
    only runs approaching standstill differ from the historical behaviour."""
    y = _state(12.0)
    on = _rhs(y, flag=True, tensorized=tensorized)
    off = _rhs(y, flag=False, tensorized=tensorized)
    assert np.abs(on - off).max() == 0.0


def test_both_rhs_variants_agree() -> None:
    """The tensorized and classical paths must stay equivalent under the flag,
    in both directions of travel and at rest."""
    for v in (0.0, 5.0, -5.0, 20.0):
        for flag in (True, False):
            y = _state(v)
            a = _rhs(y, flag=flag, tensorized=False)
            b = _rhs(y, flag=flag, tensorized=True)
            assert np.abs(a - b).max() < 1e-9, (v, flag)


def test_blend_is_smooth_through_zero() -> None:
    """A hard sign(v) would make the RHS discontinuous at v = 0, which hurts
    the solver and would break autograd after the torch port."""
    eps = DEFAULT_V_BRAKE_EPS
    dvs = np.array([
        _rhs(_state(v), flag=True)[N : 2 * N][2]
        for v in np.linspace(-4 * eps, 4 * eps, 81)
    ])
    swing = float(dvs.max() - dvs.min())
    largest_step = float(np.abs(np.diff(dvs)).max())
    # With a hard sign(v) the entire swing happens in a single sample step, so
    # largest_step / swing would be ~1. A tanh blend spreads it out.
    assert swing > 0.5, "brake should reverse sign across the window"
    assert largest_step / swing < 0.25, largest_step / swing
