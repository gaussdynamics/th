"""Tests for the dataset serializer (`DATA_SCHEMA.md`).

The equivalence test at the bottom is the important one: the driver replaces
``simulate.materialize_commands`` with a vectorized version for speed, and this
is what stops the two drifting apart.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from simulator2.consist import Train, build_consist
from simulator2.dataset import (
    STATE_CHANNELS,
    arrival_time,
    build_scenario_arrays,
    compute_norm_stats,
    coupler_dynamics,
    materialize_commands_fast,
    route_grade_stats,
    sample_command_curve,
    scenario_constants,
    write_route_npz,
    write_scenario_arrays,
)
from simulator2.driving_regimes import Regime, generate_regime_profile, sample_regime
from simulator2.randomize import RandomizationConfig, make_randomized_scenario
from simulator2.route import RouteProfile
from simulator2.simulate import materialize_commands
from simulator2.train_generator import ensure_extended_car_library

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROUTE = _REPO_ROOT / "route_generator" / "route_profiles" / "route_line3_29km.npz"
_MANIFEST = _REPO_ROOT / "saved_trains" / "generated" / "manifest.json"
_TRAIN_DIR = _REPO_ROOT / "saved_trains"

pytestmark = pytest.mark.skipif(
    not (_ROUTE.exists() and _MANIFEST.exists()),
    reason="route profiles / train manifest not present",
)


def _scenarios(n: int = 6, seed: int = 0):
    """A handful of real randomized scenarios, spanning regimes and consists."""
    library = ensure_extended_car_library()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    route = RouteProfile.from_tensor_npz(_ROUTE)
    cfg = RandomizationConfig(duration_s_range=(60.0, 120.0), dt_s=0.25)
    rng = random.Random(seed)

    out = []
    for k in range(n):
        entry = manifest[rng.randrange(len(manifest))]
        train = Train.load(_TRAIN_DIR / entry["filename"])
        vehicles, couplers = build_consist(train, library)
        profile = generate_regime_profile(sample_regime(rng), seed=seed * 100 + k)
        scenario, meta = make_randomized_scenario(
            route, vehicles, couplers, profile, train=train,
            seed=seed * 100 + k, config=cfg, route_id="route_line3_29km",
        )
        out.append((scenario, profile, train, meta, route))
    return out


# ---------------------------------------------------------------------------
# the equivalence that the driver's speed depends on
# ---------------------------------------------------------------------------

def test_fast_commands_match_the_callable_path_exactly() -> None:
    """``materialize_commands_fast`` must be bit-identical to the oracle.

    The driver uses the fast path because the callable one costs T*N Python
    calls (measured 166x slower). Any divergence would silently corrupt the
    control inputs of every scenario in the dataset, so this asserts equality
    rather than closeness.

    Note the commands must be built from ``scenario.vehicles``, not the
    pre-randomization list: ``make_randomized_scenario`` rescales
    ``F_trac_max_N`` by the sampled adhesion regime.
    """
    for scenario, profile, train, _, _ in _scenarios(n=8):
        t = np.asarray(scenario.t_eval)
        ref_trac, ref_brk = materialize_commands(scenario, t)
        got_trac, got_brk = materialize_commands_fast(
            profile, scenario.vehicles, t, train=train
        )
        assert np.array_equal(got_trac, ref_trac)
        assert np.array_equal(got_brk, ref_brk)


def test_fast_commands_match_without_a_train() -> None:
    """The ``train=None`` fallback drives every powered vehicle from the
    head-end curve; it has to match the same fallback in ``randomize``."""
    library = ensure_extended_car_library()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    route = RouteProfile.from_tensor_npz(_ROUTE)
    entry = manifest[0]
    train = Train.load(_TRAIN_DIR / entry["filename"])
    vehicles, couplers = build_consist(train, library)
    profile = generate_regime_profile(Regime.NOTCH_UP, seed=3)
    scenario, _ = make_randomized_scenario(
        route, vehicles, couplers, profile, train=None, seed=3,
        config=RandomizationConfig(duration_s_range=(60.0, 60.0), dt_s=0.25),
    )
    t = np.asarray(scenario.t_eval)
    ref = materialize_commands(scenario, t)
    got = materialize_commands_fast(profile, scenario.vehicles, t, train=None)
    assert np.array_equal(got[0], ref[0])
    assert np.array_equal(got[1], ref[1])


def test_sample_command_curve_matches_value_at() -> None:
    """Including outside the curve's own range, where value_at holds the ends."""
    profile = generate_regime_profile(Regime.THROTTLE_MODULATION, seed=11)
    curve = profile.traction_head_end
    t = np.linspace(-20.0, profile.duration_s + 20.0, 997)
    got = sample_command_curve(curve, t)
    ref = np.array([curve.value_at(float(tt)) for tt in t])
    assert np.allclose(got, ref, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------

def test_coupler_dynamics_deadband() -> None:
    """Inside the deadband the coupler carries no force; outside it is linear."""
    edge_static = np.array([[20.0, 0.02, 9.0e6, 5.0e5, 12.0e6, 7.0e5]])
    def state(gap: float, dv: float = 0.0) -> np.ndarray:
        s = np.zeros((1, 2, 4))
        s[0, 0, 0], s[0, 1, 0] = 20.0 + gap, 0.0
        s[0, 0, 1], s[0, 1, 1] = dv, 0.0
        return s

    assert coupler_dynamics(state(0.0), edge_static)[0, 0, 2] == 0.0
    assert coupler_dynamics(state(0.019), edge_static)[0, 0, 2] == 0.0
    assert coupler_dynamics(state(-0.019), edge_static)[0, 0, 2] == 0.0
    # draft: (0.05 - 0.02) * 9e6
    assert coupler_dynamics(state(0.05), edge_static)[0, 0, 2] == pytest.approx(270_000.0)
    # buff is stiffer and pushes the other way
    assert coupler_dynamics(state(-0.05), edge_static)[0, 0, 2] == pytest.approx(-360_000.0)


def test_arrival_time_interpolates_and_reports_unreached() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 10.0, 20.0, 30.0])
    assert arrival_time(t, x, 15.0) == pytest.approx(1.5)
    assert arrival_time(t, x, 0.0) == pytest.approx(0.0)
    assert np.isnan(arrival_time(t, x, 100.0))


def test_route_grade_stats() -> None:
    route = RouteProfile.from_tensor_npz(_ROUTE)
    stats = route_grade_stats(route)
    assert 0.0 < stats["grade_rms_pct"] <= stats["grade_max_pct"] < 100.0


# ---------------------------------------------------------------------------
# serialization round-trip
# ---------------------------------------------------------------------------

def _arrays_for(scenario, profile, train):
    n = len(scenario.vehicles)
    t = np.asarray(scenario.t_eval)[:40]
    node_static = np.array([
        [v.mass_kg, v.davis_A, v.davis_B, v.davis_C,
         float(v.can_traction), v.F_trac_max_N, v.F_brk_max_N]
        for v in scenario.vehicles
    ])
    edge_static = np.array([
        [c.L0_m, c.slack_half_m, c.k_draft, c.c_draft, c.k_buff, c.c_buff]
        for c in scenario.couplers
    ]).reshape(n - 1, 6)
    # A stand-in trajectory: the writer does not care how y was produced.
    y = np.tile(scenario.y0, (t.size, 1))
    y[:, :n] += np.linspace(0.0, 100.0, t.size)[:, None]
    u_trac, u_brk = materialize_commands_fast(profile, scenario.vehicles, t, train=train)
    return build_scenario_arrays(
        t=t, y=y, node_static=node_static, edge_static=edge_static,
        u_trac=u_trac, u_brk=u_brk, constants=scenario_constants(scenario),
    )


def test_scenario_npz_round_trip(tmp_path: Path) -> None:
    scenario, profile, train, meta, route = _scenarios(n=1)[0]
    arrays = _arrays_for(scenario, profile, train)
    meta = {**meta, "scenario_id": "t_run0", "route_id": "route_line3_29km",
            "split": "train"}
    path = write_scenario_arrays(arrays, tmp_path / "t_run0.npz", meta)

    with np.load(path) as d:
        n, t_len = len(scenario.vehicles), arrays.t.size
        assert d["state"].shape == (t_len, n, len(STATE_CHANNELS))
        assert d["node_static"].shape == (n, 7)
        assert d["edge_static"].shape == (n - 1, 6)
        assert d["edge_dynamic"].shape == (t_len, n - 1, 3)
        assert d["edge_index"].shape == (2, n - 1)
        assert d["u_trac"].shape == (t_len, n)
        assert d["y0"].shape == (4 * n,)
        # The route is referenced, not duplicated (DATA_SCHEMA.md §File format).
        assert "route_s" not in d
        assert str(d["route_id"]) == "route_line3_29km"
        assert bool(d["edge_dynamic_is_derived"])
        assert json.loads(str(d["metadata_json"]))["scenario_id"] == "t_run0"
        # §H constants must all be present, or the RHS cannot be recomputed.
        for key in ("tau_brk_s", "tau_trac_s", "p_max_w", "k_curv_scale",
                    "v_eps", "v_brake_eps", "brake_opposes_motion"):
            assert key in d


def test_edge_index_is_the_chain() -> None:
    scenario, profile, train, _, _ = _scenarios(n=1)[0]
    arrays = _arrays_for(scenario, profile, train)
    n = len(scenario.vehicles)
    assert np.array_equal(arrays.edge_index[0], np.arange(n - 1))
    assert np.array_equal(arrays.edge_index[1], np.arange(1, n))


def test_route_npz_carries_all_four_fields(tmp_path: Path) -> None:
    route = RouteProfile.from_tensor_npz(_ROUTE)
    path = write_route_npz(route, "route_line3_29km", tmp_path)
    with np.load(path) as d:
        r = d["route_s"].size
        for key in ("route_s", "route_sin_theta", "route_kappa", "route_vmax"):
            assert d[key].shape == (r,)


def test_norm_stats_are_per_channel_and_train_only(tmp_path: Path) -> None:
    """Consists differ in N, so the accumulator must key on channels, not vehicles."""
    paths = []
    for k, (scenario, profile, train, meta, _) in enumerate(_scenarios(n=3, seed=5)):
        arrays = _arrays_for(scenario, profile, train)
        meta = {**meta, "scenario_id": f"s{k}", "route_id": "r", "split": "train"}
        paths.append(write_scenario_arrays(arrays, tmp_path / f"s{k}.npz", meta))
    route_path = write_route_npz(RouteProfile.from_tensor_npz(_ROUTE), "r", tmp_path)

    stats = compute_norm_stats(paths, [route_path])
    assert stats["_computed_over"]["split"] == "train"
    assert stats["_computed_over"]["n_scenarios"] == 3
    assert len(stats["state"]["mean"]) == 4
    assert stats["state"]["channels"] == list(STATE_CHANNELS)
    assert len(stats["node_static"]["mean"]) == 7
    assert len(stats["edge_static"]["mean"]) == 6
    assert len(stats["edge_dynamic"]["mean"]) == 3
    # u_trac is [T, N]: the vehicle axis is not a channel axis.
    assert len(stats["u_trac"]["mean"]) == 1
    # A constant channel must not produce a zero divisor.
    assert all(s > 0 for s in stats["node_static"]["std"])
