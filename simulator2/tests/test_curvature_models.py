"""Tests for the curvature-resistance models.

The ``roeckl`` and ``linear`` laws are a prototype, added because the original
``proxy_v2`` is both uncalibrated *and* the wrong shape: real curve resistance is
speed-independent, while the proxy scales with ``v^2``, so no single
``k_curv_scale`` is correct across the speed range.

They are strictly opt-in. ``proxy_v2`` stays the default because Chapter 4, the
reference fixtures, and every result recorded before this flag existed depend on
it -- ``test_default_curvature_model_is_the_original_proxy`` is what keeps that
true.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simulator2.consist import Train, build_consist
from simulator2.dataset import materialize_commands_fast
from simulator2.driving_regimes import Regime, generate_regime_profile
from simulator2.randomize import RandomizationConfig, make_randomized_scenario
from simulator2.route import (
    MIN_CURVE_RADIUS_M,
    RouteProfile,
    curvature_force_longitudinal,
    curvature_specific_resistance,
)
from simulator2.train_generator import ensure_extended_car_library

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROUTE_DIR = _REPO_ROOT / "route_generator" / "route_profiles"
_MANIFEST = _REPO_ROOT / "saved_trains" / "generated" / "manifest.json"
_TRAIN_DIR = _REPO_ROOT / "saved_trains"


def _constant_curvature_route(kappa: float) -> RouteProfile:
    s = np.linspace(0.0, 1000.0, 101)
    return RouteProfile(
        s_nodes_m=s,
        sin_theta_nodes=np.zeros_like(s),
        kappa_nodes=np.full_like(s, kappa),
    )


# ---------------------------------------------------------------------------
# the default must not move
# ---------------------------------------------------------------------------

def test_default_curvature_model_is_the_original_proxy() -> None:
    from simulator2.io_types import ExtendedTrainScenario

    assert RandomizationConfig().curvature_model == "proxy_v2"
    field = ExtendedTrainScenario.__dataclass_fields__["curvature_model"]
    assert field.default == "proxy_v2"


# ---------------------------------------------------------------------------
# the published formulae
# ---------------------------------------------------------------------------

def test_roeckl_matches_the_published_formula() -> None:
    """w_c = 650/(R-55) N/kN above 300 m, 500/(R-30) below."""
    for r in (2000.0, 1000.0, 500.0, 301.0):
        assert float(curvature_specific_resistance(1.0 / r, "roeckl")) == pytest.approx(
            1e-3 * 650.0 / (r - 55.0)
        )
    for r in (299.0, 150.0, 80.0):
        assert float(curvature_specific_resistance(1.0 / r, "roeckl")) == pytest.approx(
            1e-3 * 500.0 / (r - 30.0)
        )


def test_roeckl_is_discontinuous_at_300m() -> None:
    """A property of the published formula, not of this implementation.

    Recorded so it cannot be "fixed" by accident, and so the gradient-quality
    argument for preferring the linear form stays visible.
    """
    above = float(curvature_specific_resistance(1.0 / 300.5, "roeckl"))
    below = float(curvature_specific_resistance(1.0 / 299.5, "roeckl"))
    assert above > below
    assert (above - below) / above > 0.25  # about a 30 % jump


def test_linear_form_is_smooth_through_the_roeckl_boundary() -> None:
    k = np.linspace(1.0 / 400.0, 1.0 / 200.0, 401)
    w = curvature_specific_resistance(k, "linear")
    steps = np.abs(np.diff(w))
    assert steps.max() < 5.0 * float(np.median(steps))


def test_calibrated_models_agree_with_standard_magnitudes() -> None:
    """Both should land near the ~6.8 * m * kappa newtons the literature gives."""
    route = _constant_curvature_route(1.0e-3)
    m = 100_000.0
    f_r = curvature_force_longitudinal(m, 15.0, 500.0, route, 1.0, "roeckl")
    f_l = curvature_force_longitudinal(m, 15.0, 500.0, route, 1.0, "linear")
    assert abs(f_r - f_l) / f_r < 0.05
    assert f_r == pytest.approx(6.8 * m * 1.0e-3, rel=0.05)


# ---------------------------------------------------------------------------
# shape of the law
# ---------------------------------------------------------------------------

def test_speed_independent_models_do_not_scale_with_speed() -> None:
    """The substantive point of the change."""
    route = _constant_curvature_route(1.0e-3)
    for model in ("roeckl", "linear"):
        f5 = curvature_force_longitudinal(100_000.0, 5.0, 500.0, route, 1.0, model)
        f25 = curvature_force_longitudinal(100_000.0, 25.0, 500.0, route, 1.0, model)
        assert f5 == pytest.approx(f25)

    # while the original proxy changes by 25x over the same range
    p5 = curvature_force_longitudinal(100_000.0, 5.0, 500.0, route, 1.0, "proxy_v2")
    p25 = curvature_force_longitudinal(100_000.0, 25.0, 500.0, route, 1.0, "proxy_v2")
    assert p25 / p5 == pytest.approx(25.0)


def test_force_opposes_motion_and_is_off_at_zero_scale() -> None:
    route = _constant_curvature_route(2.0e-3)
    for model in ("proxy_v2", "roeckl", "linear"):
        assert curvature_force_longitudinal(1e5, 15.0, 500.0, route, 0.0, model) == 0.0
        fwd = curvature_force_longitudinal(1e5, 15.0, 500.0, route, 1.0, model)
        rev = curvature_force_longitudinal(1e5, -15.0, 500.0, route, 1.0, model)
        assert fwd > 0 and rev < 0
        assert fwd == pytest.approx(-rev)


def test_tight_curvature_is_clamped_not_infinite() -> None:
    """Roeckl's denominators vanish at R = 55 m and 30 m. A noisy curvature
    field must not be able to put an infinity into the RHS."""
    for model in ("roeckl", "linear"):
        vals = curvature_specific_resistance(
            np.array([1.0 / 55.0, 1.0 / 30.0, 1.0, 1.0e6]), model
        )
        assert np.isfinite(vals).all()
        ceiling = float(curvature_specific_resistance(1.0 / MIN_CURVE_RADIUS_M, model))
        assert (vals <= ceiling + 1e-12).all()


def test_unknown_model_is_rejected() -> None:
    route = _constant_curvature_route(1.0e-3)
    with pytest.raises(ValueError):
        curvature_force_longitudinal(1e5, 15.0, 500.0, route, 1.0, "not_a_model")


# ---------------------------------------------------------------------------
# torch / numpy parity
# ---------------------------------------------------------------------------

def _scenario(model: str, k: float, seed: int = 4):
    library = ensure_extended_car_library()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    train = Train.load(_TRAIN_DIR / manifest[0]["filename"])
    vehicles, couplers = build_consist(train, library)
    vehicles, couplers = vehicles[:8], couplers[:7]
    # line2 is the twistiest corridor, so curvature actually bites
    route = RouteProfile.from_tensor_npz(_ROUTE_DIR / "route_line2_33km.npz")
    profile = generate_regime_profile(Regime.CRUISE, seed=seed)
    scenario, _ = make_randomized_scenario(
        route, vehicles, couplers, profile, train=None, seed=seed,
        config=RandomizationConfig(duration_s_range=(60.0, 60.0), dt_s=0.25,
                                   k_curv_scale=k, curvature_model=model),
    )
    return scenario, profile


@pytest.mark.skipif(not _MANIFEST.exists(), reason="train manifest not present")
@pytest.mark.parametrize("model,k", [("proxy_v2", 0.5), ("roeckl", 1.0), ("linear", 1.0)])
def test_torch_curvature_matches_numpy(model: str, k: float) -> None:
    """The two implementations of each law must not drift apart."""
    torch = pytest.importorskip("torch")
    from simulator2.rhs import train_rhs_tensorized
    from simulator2.torch_rhs import TorchScenarioBatch, torch_rhs

    scenario, profile = _scenario(model, k)
    cmds = materialize_commands_fast(profile, scenario.vehicles,
                                     scenario.t_eval, train=None)
    batch = TorchScenarioBatch.from_scenarios(
        [scenario], device="cpu", dtype=torch.float64, commands=[cmds]
    )
    assert batch.curvature_model == model

    n = len(scenario.vehicles)
    y = scenario.y0.copy()
    y[n:2 * n] = 15.0  # move it, so the curvature term is actually active
    got = torch_rhs(3.0, torch.as_tensor(y[None, :], dtype=torch.float64),
                    batch).numpy()[0]
    ref = train_rhs_tensorized(
        3.0, y, scenario.vehicles, scenario.couplers, scenario.route,
        scenario.u_trac_cmd, scenario.u_brk_cmd, scenario.k_curv_scale,
        scenario.tau_brk_s, scenario.tau_trac_s, scenario.p_max_w,
        brake_opposes_motion=scenario.brake_opposes_motion,
        curvature_model=model,
    )
    rel = np.abs(got - ref).max() / max(1.0, np.abs(ref).max())
    assert rel < 1e-12, f"{model}: relative difference {rel:.2e}"


@pytest.mark.skipif(not _MANIFEST.exists(), reason="train manifest not present")
def test_mixed_curvature_models_in_one_batch_are_rejected() -> None:
    """Silently picking one model for a mixed batch would corrupt half of it."""
    torch = pytest.importorskip("torch")
    from simulator2.torch_rhs import TorchScenarioBatch

    scenarios = [_scenario(m, 1.0, seed=2)[0] for m in ("proxy_v2", "roeckl")]
    with pytest.raises(ValueError, match="curvature model"):
        TorchScenarioBatch.from_scenarios(scenarios, device="cpu", dtype=torch.float64)


@pytest.mark.skipif(not _MANIFEST.exists(), reason="train manifest not present")
def test_curvature_stays_differentiable() -> None:
    """The port exists to be differentiable; a new force term must not break it."""
    torch = pytest.importorskip("torch")
    from simulator2.torch_rhs import TorchScenarioBatch, torch_rhs

    for model in ("roeckl", "linear"):
        scenario, profile = _scenario(model, 1.0)
        cmds = materialize_commands_fast(profile, scenario.vehicles,
                                         scenario.t_eval, train=None)
        batch = TorchScenarioBatch.from_scenarios(
            [scenario], device="cpu", dtype=torch.float64, commands=[cmds]
        )
        n = len(scenario.vehicles)
        y0 = scenario.y0.copy()
        y0[n:2 * n] = 15.0
        y = torch.tensor(y0[None, :], dtype=torch.float64, requires_grad=True)
        torch_rhs(3.0, y, batch).sum().backward()
        assert y.grad is not None and torch.isfinite(y.grad).all()
        assert y.grad.abs().sum() > 0
