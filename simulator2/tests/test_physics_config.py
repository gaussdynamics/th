"""The physics configuration switch (:mod:`simulator2.physics_config`) and the
speed planner (:mod:`simulator2.speed_planner`)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from simulator2 import physics_config as pc
from simulator2.consist import Train, build_consist
from simulator2.control_profile import CommandCurve, CommandKnot, ControlProfile
from simulator2.dataset import materialize_commands_fast
from simulator2.route import RouteProfile
from simulator2.speed_planner import PlannerParams, allowed_speed_field, plan_commands
from simulator2.torch_rhs import TorchScenarioBatch
from simulator2.train_generator import ensure_extended_car_library

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "saved_trains" / "generated_unit_ballast_0000.json"


@pytest.fixture(scope="module")
def train_and_library():
    return Train.load(TRAIN), ensure_extended_car_library()


def _curve(points):
    return CommandCurve(knots=[CommandKnot(t_s=t, fraction=f) for t, f in points])


def _profile(trac, brake=((0.0, 0.0),)):
    return ControlProfile(name="test", duration_s=120.0, brake=_curve(brake),
                          traction_head_end=_curve(trac), traction_dpu_mid=_curve(trac),
                          traction_dpu_rear=_curve(trac))


# -- the presets --------------------------------------------------------------------

def test_presets_round_trip_and_validate() -> None:
    assert pc.LEGACY.is_legacy and not pc.V3.is_legacy
    assert pc.from_json(pc.V3.to_json()) == pc.V3
    assert pc.physics_of({}) == pc.LEGACY
    with pytest.raises(ValueError, match="notch"):
        pc.PhysicsConfig(traction_model="notch")              # needs sourced locos
    with pytest.raises(ValueError, match="grade_model"):
        pc.PhysicsConfig(grade_model="sine")
    assert pc.V3.default_dt_s == 0.01 and pc.LEGACY.default_dt_s == 0.02


# -- legacy is unchanged ------------------------------------------------------------

def test_legacy_consist_and_commands_are_the_old_path(train_and_library) -> None:
    train, lib = train_and_library
    c = pc.configure_consist(train, lib, pc.LEGACY)
    assert (c.vehicles, c.couplers) == build_consist(train, lib)
    assert c.loco_table is None and c.inertia_factor is None
    t = np.arange(0.0, 60.0, 0.25)
    prof = _profile([(0.0, 0.2), (30.0, 0.8)], [(0.0, 0.0), (40.0, 0.3)])
    m = pc.materialize_commands(prof, c, t, train, pc.LEGACY)
    ref = materialize_commands_fast(prof, c.vehicles, t, train=train)
    assert np.array_equal(m.u_trac, ref[0]) and np.array_equal(m.u_brk, ref[1])


def test_legacy_batch_is_returned_untouched() -> None:
    b = TorchScenarioBatch.synthetic(batch_size=2, n_vehicles=5)
    assert pc.configure_batch(b, [], [], pc.LEGACY) is b


# -- v3 commands and batch ----------------------------------------------------------

def test_v3_commands_notches_dynamic_brake_and_air_brake(train_and_library) -> None:
    train, lib = train_and_library
    physics = replace(pc.V3, speed_planner=False)
    c = pc.configure_consist(train, lib, physics)
    n = len(c.vehicles)
    x0 = -np.cumsum([0.0] + [cp.L0_m for cp in c.couplers])
    t = np.arange(0.0, 60.0, 0.25)
    prof = _profile([(0.0, 0.5)], [(0.0, 0.0), (20.0, 0.4), (60.0, 0.4)])
    m = pc.materialize_commands(prof, c, t, train, physics, regime="cruise", x0=x0)
    powered = c.is_loco
    assert set(np.unique(m.u_trac[:, powered])) <= set(range(9))       # integer notches
    assert np.all(m.u_trac[:, ~powered] == 0)
    assert m.u_brk[-1].max() > 0 and m.tau_apply.shape == (n,)
    # the dynamic-brake regime moves the brake curve onto the locomotives
    d = pc.materialize_commands(prof, c, t, train, physics,
                                regime="dynamic_brake_descent", x0=x0)
    assert d.u_brk.max() == 0.0
    assert d.u_trac[-1, powered].min() == -np.rint(8 * 0.4)


def test_v3_batch_carries_the_models_and_gear_states(train_and_library) -> None:
    train, lib = train_and_library
    physics = replace(pc.V3, speed_planner=False)
    c = pc.configure_consist(train, lib, physics)
    n = len(c.vehicles)
    t = np.arange(0.0, 10.0, 0.25)
    x0 = -np.cumsum([0.0] + [cp.L0_m for cp in c.couplers])
    m = pc.materialize_commands(_profile([(0.0, 0.5)]), c, t, train, physics,
                                regime="cruise", x0=x0)
    b = TorchScenarioBatch.synthetic(batch_size=1, n_vehicles=n)
    fields = {f: getattr(b, f) for f in b.__dataclass_fields__}
    fields["y0"] = torch.cat([torch.as_tensor(x0), torch.zeros(3 * n)])[None].to(b.y0.dtype)
    b = type(b)(**fields)
    out = pc.configure_batch(b, [c], [m], physics)
    assert out.grade_model == "exact" and out.traction_model == "notch"
    assert out.coupler_model == "draft_gear" and out.standstill_model == "breakaway"
    assert out.y0.shape[1] == 5 * n - 1
    assert torch.equal(out.y0[0, 3 * n:4 * n], torch.as_tensor(m.u_trac[0], dtype=out.y0.dtype))
    assert torch.allclose(out.inertia_factor[0], torch.as_tensor(c.inertia_factor, dtype=out.y0.dtype))


# -- speed planner ------------------------------------------------------------------

def _route(limit_drop_at: float = 3000.0) -> RouteProfile:
    s = np.arange(0.0, 8000.0, 10.0)
    vmax = np.where((s >= limit_drop_at) & (s < limit_drop_at + 1000.0), 10.0, 20.0)
    return RouteProfile(s_nodes_m=s, sin_theta_nodes=np.full_like(s, -0.005), v_max_nodes=vmax)


def test_allowed_speed_field_window_and_braking_curve() -> None:
    route = _route()
    s, va = allowed_speed_field(route, train_length_m=500.0, decel_mps2=0.15)
    assert va[(s >= 3000.0) & (s < 4500.0)].max() == pytest.approx(10.0)   # tail clears at 4500
    assert va[s >= 4600.0].min() == pytest.approx(20.0)
    ahead = s < 3000.0
    expect = np.minimum(20.0, np.sqrt(10.0 ** 2 + 2 * 0.15 * (3000.0 - s[ahead])))
    assert np.allclose(va[ahead], expect, atol=0.05)


def test_planner_keeps_full_power_under_the_limit(train_and_library) -> None:
    """Full power on a downgrade into a 10 m/s zone: ungoverned, the point mass
    overruns; governed, it stays within the air brake's switch-on band."""
    train, lib = train_and_library
    physics = replace(pc.V3, speed_planner=True)
    c = pc.configure_consist(train, lib, physics)
    x0 = 1500.0 - np.cumsum([0.0] + [cp.L0_m for cp in c.couplers])
    t = np.arange(0.0, 600.0, 0.25)
    n = len(c.vehicles)
    trac = np.where(c.is_loco[None, :], 1.0, 0.0) * np.ones((t.size, n))
    zero = np.zeros(t.size)
    route = _route()
    loose = plan_commands(t, trac, zero, zero, c, x0, 15.0, route, physics,
                          PlannerParams(traction_margin_mps=-1e3, db_margin_mps=-1e3,
                                        air_on_mps=1e3))
    tight = plan_commands(t, trac, zero, zero, c, x0, 15.0, route, physics)
    assert (loose["v_plan"] - loose["v_allow"]).max() > 3.0
    assert (tight["v_plan"] - tight["v_allow"]).max() < PlannerParams().air_on_mps + 0.5
    assert tight["db_frac"].max() > 0.0


def test_equilibrium_start_has_no_start_up_snatch(train_and_library) -> None:
    """A powered v3 train started at speed with its gears in equilibrium: the
    start is loaded and the coupler forces stay at their initial tensions.
    (Started unloaded, a 126-car train snatched to 9 MN within 9 s; see
    SIMULATOR_VALIDATION_NOTE.md, 2026-10-07.)"""
    from simulator2.draft_gear import draft_gear_stateful_torch
    from simulator2.torch_rhs import rollout_rk4

    train, lib = train_and_library
    physics = replace(pc.V3, speed_planner=False, air_brake=False)
    c = pc.configure_consist(train, lib, physics)
    n = len(c.vehicles)
    t = np.arange(0.0, 10.0 + 1e-9, 0.25)
    m = pc.materialize_commands(_profile([(0.0, 0.75)]), c, t, train, physics, regime="cruise")
    x0 = -np.cumsum([0.0] + [cp.L0_m for cp in c.couplers])
    route = RouteProfile(s_nodes_m=np.array([-1e5, 1e5]), sin_theta_nodes=np.zeros(2))

    class _Sc:  # the attributes from_scenarios reads
        vehicles, couplers, t_eval = c.vehicles, c.couplers, t
        y0 = np.concatenate([x0, np.full(n, 3.0), m.u_brk[0], m.u_trac[0]])
        tau_brk_s, tau_trac_s, p_max_w, k_curv_scale = 3.0, 5.0, 1e12, 0.0
        brake_opposes_motion, curvature_model = True, "proxy_v2"
    _Sc.route = route

    def peak(batch):
        _, y = rollout_rk4(batch, batch.y0, torch.as_tensor(t, dtype=torch.float64), dt=0.01)
        y = y[0]
        xs, vs, z = y[:, :n], y[:, n:2 * n], y[:, 4 * n:5 * n - 1]
        es = batch.edge_static[0, : n - 1]
        f, _ = draft_gear_stateful_torch(xs[:, :-1] - xs[:, 1:] - es[:, 0], vs[:, :-1] - vs[:, 1:],
                                         z, es[:, 1], batch.draft_gear_table)
        return float(f.abs().max()), float(z[0].abs().max())

    base = TorchScenarioBatch.from_scenarios([_Sc], dtype=torch.float64,
                                             commands=[(m.u_trac, m.u_brk)])
    eq = pc.configure_batch(base, [c], [m], physics)
    f_eq, z0 = peak(eq)
    assert z0 > 1e5                                   # the start is loaded
    assert f_eq < 1.1 * z0


def test_notch_rate_limit_steps_up_slowly_and_cuts_at_once() -> None:
    t = np.arange(0.0, 60.0, 1.0)
    want = np.where((t > 0.0) & (t < 40.0), 8.0, 0.0)[:, None] * np.ones((1, 2))
    want[:, 1] *= -1                                       # a dynamic brake column
    out = pc.rate_limit_notches(want, t, interval_s=8.0)
    # starts from the t = 0 command (0), then one notch per 8 s: 1 at t=1 ... 5 at 33
    assert out[0, 0] == 0 and out[t == 1.0, 0][0] == 1
    assert out[t == 33.0, 0][0] == 5 and out[t == 39.0, 0][0] == 5
    assert np.all(np.diff(out[t < 40.0, 0]) <= 1)
    assert np.array_equal(out[:, 1], -out[:, 0])          # same limit for the brake
    assert np.all(out[t >= 40.0] == 0)                     # power cut at once
