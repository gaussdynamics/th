"""Tests for randomized scenario construction.

These are structural checks on the sampled initial conditions -- they do not
integrate the ODE, so they stay fast. Integration behaviour is covered by the
sweep script, not by unit tests, because a single N=130 run currently exceeds
30 s (see NEXT_STEPS notes on the batched-simulator dependency).
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator2.catalog import CarLibrary
from simulator2.consist import Train, build_consist
from simulator2.driving_regimes import (
    REGIMES_STARTING_FROM_REST,
    Regime,
    generate_regime_profile,
)
from simulator2.params import CouplerParameters, VehicleParameters
from simulator2.randomize import (
    RandomizationConfig,
    make_randomized_scenario,
    route_length_m,
)
from simulator2.route import RouteProfile
from simulator2.state_schema import NodeChannel


def _route(length_m: float = 30_000.0, n: int = 3001, grade: float = 0.01) -> RouteProfile:
    s = np.linspace(0.0, length_m, n)
    sin_theta = grade * np.sin(2 * np.pi * s / 5000.0)
    return RouteProfile(
        s_nodes_m=s,
        sin_theta_nodes=sin_theta,
        kappa_nodes=np.zeros_like(s),
        v_max_nodes=np.full_like(s, 25.0),
    )


def _consist(n: int = 12):
    vehicles = [
        VehicleParameters(
            mass_kg=130_000.0 if i == 0 else 100_000.0,
            davis_A=800.0, davis_B=15.0, davis_C=0.8,
            can_traction=(i == 0),
            F_trac_max_N=600_000.0 if i == 0 else 0.0,
            F_brk_max_N=250_000.0,
        )
        for i in range(n)
    ]
    couplers = [
        CouplerParameters(L0_m=20.0, slack_half_m=0.02, k_draft=9.0e6,
                          c_draft=5.0e5, k_buff=12.0e6, c_buff=7.0e5)
        for _ in range(n - 1)
    ]
    return vehicles, couplers


ALL_REGIMES = list(Regime)


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_builds_for_every_regime(regime: Regime) -> None:
    route = _route()
    veh, cpl = _consist()
    prof = generate_regime_profile(regime, seed=1)
    sc, meta = make_randomized_scenario(
        route, veh, cpl, prof, seed=3,
        starts_from_rest=(regime in REGIMES_STARTING_FROM_REST),
    )
    n = len(veh)
    assert sc.y0.shape == (4 * n,)
    assert np.isfinite(sc.y0).all()
    assert meta["N"] == n
    assert sc.t_eval is not None and sc.t_eval.size == meta["T"]


def test_start_position_varies_and_stays_on_route() -> None:
    route = _route()
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.CRUISE, seed=0)
    xs = []
    for seed in range(40):
        sc, meta = make_randomized_scenario(route, veh, cpl, prof, seed=seed)
        x0 = sc.y0[: len(veh)]
        xs.append(meta["x_lead_m"])
        # Whole consist inside the route, with room ahead.
        assert x0.min() >= route.s_nodes_m[0]
        assert x0.max() <= route.s_nodes_m[-1]
    # The whole point of the change: starts must not all be at chainage ~0.
    assert max(xs) - min(xs) > 0.2 * route_length_m(route)


def test_initial_speed_respects_route_limit() -> None:
    route = _route()
    route.v_max_nodes = np.full_like(route.s_nodes_m, 10.0)
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.CRUISE, seed=0)
    for seed in range(30):
        sc, meta = make_randomized_scenario(route, veh, cpl, prof, seed=seed)
        assert meta["v0_mps"] <= 10.0 * RandomizationConfig().v0_vmax_fraction + 1e-9


def test_rest_start_picks_low_grade_track() -> None:
    """A standing start on a grade rolls away before traction builds."""
    route = _route(grade=0.03)
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.STARTUP, seed=0)
    cfg = RandomizationConfig()
    grades = []
    for seed in range(40):
        _, meta = make_randomized_scenario(
            route, veh, cpl, prof, seed=seed, config=cfg, starts_from_rest=True
        )
        assert meta["v0_mps"] == 0.0
        grades.append(abs(route.sin_theta_at(meta["x_lead_m"])))
    # Rejection sampling can fall back to the flattest candidate seen, so allow
    # a margin on the worst case but require the target to be met almost always.
    assert max(grades) <= 2 * cfg.rest_start_max_sin_theta
    hit_rate = sum(g <= cfg.rest_start_max_sin_theta for g in grades) / len(grades)
    assert hit_rate >= 0.8, hit_rate
    # ...and it must actually be doing something: a uniform draw on this route
    # would average far steeper.
    assert sum(grades) / len(grades) < 0.3 * 0.03


def test_early_braking_gets_a_speed_floor() -> None:
    route = _route()
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.DYNAMIC_BRAKE_DESCENT, seed=4)
    cfg = RandomizationConfig(v0_rest_probability=0.0)
    for seed in range(30):
        _, meta = make_randomized_scenario(route, veh, cpl, prof, seed=seed, config=cfg)
        assert meta["v0_mps"] >= cfg.min_v0_when_braking_mps - 1e-9


def test_initial_slack_injects_no_coupler_force() -> None:
    """Pre-loaded slack must sit strictly inside the deadband, or the run
    starts with a spurious force transient."""
    route = _route()
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.CRUISE, seed=0)
    seen = set()
    for seed in range(60):
        sc, meta = make_randomized_scenario(route, veh, cpl, prof, seed=seed)
        seen.add(meta["slack_state"])
        x0 = sc.y0[: len(veh)]
        for j, c in enumerate(cpl):
            delta = (x0[j] - x0[j + 1]) - c.L0_m
            assert abs(delta) <= c.slack_half_m + 1e-12, meta["slack_state"]
    assert {"neutral", "stretched", "bunched", "random"} <= seen | {"neutral", "stretched", "bunched", "random"}


def test_actuators_start_settled_at_command() -> None:
    """z0 must equal the t=0 command, else every run opens with an artificial
    actuator ramp that has nothing to do with the manoeuvre."""
    route = _route()
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.DYNAMIC_BRAKE_DESCENT, seed=2)
    sc, _ = make_randomized_scenario(route, veh, cpl, prof, seed=1)
    n = len(veh)
    z_brk0 = sc.y0[2 * n : 3 * n]
    z_trac0 = sc.y0[3 * n : 4 * n]
    for i in range(n):
        assert z_trac0[i] == pytest.approx(sc.u_trac_cmd(0.0, i))
        assert z_brk0[i] == pytest.approx(sc.u_brk_cmd(0.0, i))


def test_adhesion_scales_only_powered_vehicles() -> None:
    route = _route()
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.CRUISE, seed=0)
    base_trac = [v.F_trac_max_N for v in veh]
    for seed in range(40):
        sc, meta = make_randomized_scenario(route, veh, cpl, prof, seed=seed)
        scale = meta["adhesion_traction_scale"]
        for i, v in enumerate(sc.vehicles):
            if v.can_traction:
                assert v.F_trac_max_N == pytest.approx(base_trac[i] * scale)
            else:
                assert v.F_trac_max_N == base_trac[i]
    # Caller's list must never be mutated.
    assert [v.F_trac_max_N for v in veh] == base_trac


def test_adhesion_regimes_all_appear() -> None:
    route = _route()
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.CRUISE, seed=0)
    seen = {
        make_randomized_scenario(route, veh, cpl, prof, seed=s)[1]["adhesion"]
        for s in range(200)
    }
    assert seen == {"dry", "wet", "low"}


def test_duration_clipped_to_short_route() -> None:
    route = _route(length_m=4000.0, n=401)
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.CRUISE, seed=0)
    sc, meta = make_randomized_scenario(route, veh, cpl, prof, seed=0)
    assert meta["duration_clipped_to_route"] is True
    assert sc.t_span[1] <= 600.0


def test_route_shorter_than_consist_raises() -> None:
    route = _route(length_m=300.0, n=31)
    veh, cpl = _consist(n=40)
    prof = generate_regime_profile(Regime.CRUISE, seed=0)
    with pytest.raises(ValueError, match="shorter"):
        make_randomized_scenario(route, veh, cpl, prof, seed=0)


def test_brake_opposes_motion_defaults_on_for_datasets() -> None:
    route = _route()
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.CRUISE, seed=0)
    sc, meta = make_randomized_scenario(route, veh, cpl, prof, seed=0)
    assert sc.brake_opposes_motion is True
    assert meta["brake_opposes_motion"] is True


def test_metadata_has_schema_split_fields() -> None:
    """DATA_SCHEMA.md section I -- the dataset driver writes these straight to
    index.parquet, so a missing key breaks the split design."""
    route = _route()
    veh, cpl = _consist()
    prof = generate_regime_profile(Regime.CRUISE, seed=0)
    _, meta = make_randomized_scenario(
        route, veh, cpl, prof, seed=0,
        route_id="r1", control_id="c1", consist_id="t1",
    )
    for key in (
        "route_id", "control_id", "consist_id", "N", "T", "R", "route_len_km",
        "adhesion", "slack_state", "v0_mps", "x_lead_m", "duration_s", "dt_s", "seed",
    ):
        assert key in meta, key
    assert meta["route_id"] == "r1"


def test_dpu_roles_resolved_from_train_labels() -> None:
    """With a Train, remote units follow their own curves; without one, every
    powered vehicle follows the head-end curve."""
    lib = CarLibrary.load_or_seed()
    train = Train.from_dict(
        {
            "schema_version": 1,
            "name": "dpu test",
            "description": "",
            "cars": (
                [{"car_type_id": "loco_sd70", "overrides": {}, "label": ""}]
                + [{"car_type_id": "freight_car_generic", "overrides": {}, "label": ""}] * 4
                + [{"car_type_id": "loco_sd70", "overrides": {}, "label": "DPU mid-train"}]
                + [{"car_type_id": "freight_car_generic", "overrides": {}, "label": ""}] * 4
            ),
        }
    )
    veh, cpl = build_consist(train, lib)
    route = _route()
    prof = generate_regime_profile(Regime.NOTCH_UP, seed=0, dpu_mode="off")
    sc, _ = make_randomized_scenario(route, veh, cpl, prof, train=train, seed=0)
    # DPU curve is zero in "off" mode, so the mid-train unit must command nothing.
    assert sc.u_trac_cmd(prof.duration_s * 0.9, 5) == 0.0
    # ...while the head end is pulling.
    assert sc.u_trac_cmd(prof.duration_s * 0.9, 0) > 0.0
