"""Tests for simulator2/control_profile.py (randomized traction/brake commands)."""

from __future__ import annotations

import pytest

from simulator2.catalog import CarLibrary, default_seed_car_types
from simulator2.consist import CarInstance, Train, build_consist
from simulator2.control_profile import (
    CommandCurve,
    CommandKnot,
    ControlProfile,
    RandomProfileConfig,
    build_command_callables,
    generate_random_profile,
    list_saved_control_profiles,
    role_for_car,
)


@pytest.fixture()
def library() -> CarLibrary:
    return CarLibrary(car_types={ct.id: ct for ct in default_seed_car_types()})


def test_command_curve_holds_before_first_and_after_last_knot():
    curve = CommandCurve(knots=[CommandKnot(10.0, 0.2), CommandKnot(20.0, 0.8)])
    assert curve.value_at(0.0) == 0.2
    assert curve.value_at(10.0) == 0.2
    assert curve.value_at(100.0) == 0.8


def test_command_curve_interpolates_monotonically_between_knots():
    curve = CommandCurve(knots=[CommandKnot(0.0, 0.0), CommandKnot(10.0, 1.0)])
    values = [curve.value_at(t) for t in (0.0, 2.5, 5.0, 7.5, 10.0)]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert 0.0 < values[2] < 1.0  # midpoint strictly between endpoints


def test_role_for_car_follows_label_and_can_traction():
    assert role_for_car("", can_traction=True) == "head_end"
    assert role_for_car("DPU mid-train", can_traction=True) == "dpu_mid"
    assert role_for_car("DPU rear helper", can_traction=True) == "dpu_rear"
    assert role_for_car("DPU mid-train", can_traction=False) is None


def test_control_profile_json_roundtrip(tmp_path):
    profile = ControlProfile(
        name="test profile",
        duration_s=90.0,
        brake=CommandCurve(knots=[CommandKnot(0.0, 0.0), CommandKnot(90.0, 0.4)]),
        traction_head_end=CommandCurve(knots=[CommandKnot(0.0, 0.0), CommandKnot(30.0, 1.0)]),
        seed=7,
    )
    path = profile.save(tmp_path, filename="test_profile")
    loaded = ControlProfile.load(path)
    assert loaded.name == profile.name
    assert loaded.duration_s == profile.duration_s
    assert loaded.seed == 7
    assert [ (k.t_s, k.fraction) for k in loaded.brake.knots ] == [(0.0, 0.0), (90.0, 0.4)]

    assert list_saved_control_profiles(tmp_path) == [path]


def test_generate_random_profile_is_deterministic_under_seed():
    config = RandomProfileConfig(seed=42, duration_s=200.0)
    a = generate_random_profile(config)
    b = generate_random_profile(config)
    assert a.to_dict() == b.to_dict()


def test_generate_random_profile_respects_fraction_ranges():
    config = RandomProfileConfig(
        seed=1, duration_s=150.0, traction_fraction_range=(0.2, 0.6), brake_fraction_range=(0.0, 0.3)
    )
    profile = generate_random_profile(config)
    for k in profile.traction_head_end.knots:
        assert 0.2 <= k.fraction <= 0.6
    for k in profile.brake.knots:
        assert 0.0 <= k.fraction <= 0.3


def test_generate_random_profile_synced_dpu_lags_head_end():
    config = RandomProfileConfig(seed=3, duration_s=120.0, dpu_mode="synced", dpu_lag_s=5.0)
    profile = generate_random_profile(config)
    # Synced DPU should reproduce the head-end curve shifted forward in time.
    t = 40.0
    assert profile.traction_dpu_mid.value_at(t) == pytest.approx(
        profile.traction_head_end.value_at(t - 5.0), abs=1e-6
    )


def test_generate_random_profile_dpu_off_is_zero():
    config = RandomProfileConfig(seed=5, duration_s=100.0, dpu_mode="off")
    profile = generate_random_profile(config)
    assert all(k.fraction == 0.0 for k in profile.traction_dpu_mid.knots)
    assert all(k.fraction == 0.0 for k in profile.traction_dpu_rear.knots)


def test_build_command_callables_respects_roles_and_ceilings(library: CarLibrary):
    train = Train(
        name="dp train",
        cars=[
            CarInstance("loco_sd70"),  # head-end
            CarInstance("coal_hopper_loaded"),  # no traction
            CarInstance("loco_sd70", label="DPU mid-train"),
        ],
    )
    vehicles, _ = build_consist(train, library)

    profile = ControlProfile(
        name="p",
        duration_s=60.0,
        brake=CommandCurve(knots=[CommandKnot(0.0, 0.5), CommandKnot(60.0, 0.5)]),
        traction_head_end=CommandCurve(knots=[CommandKnot(0.0, 1.0), CommandKnot(60.0, 1.0)]),
        traction_dpu_mid=CommandCurve(knots=[CommandKnot(0.0, 0.25), CommandKnot(60.0, 0.25)]),
    )
    u_trac_cmd, u_brk_cmd = build_command_callables(profile, train, vehicles)

    assert u_trac_cmd(30.0, 0) == pytest.approx(vehicles[0].F_trac_max_N)  # head-end, full
    assert u_trac_cmd(30.0, 1) == 0.0  # hopper: never powered
    assert u_trac_cmd(30.0, 2) == pytest.approx(0.25 * vehicles[2].F_trac_max_N)  # DPU mid

    # Brake applies train-wide, including the non-powered hopper.
    assert u_brk_cmd(30.0, 0) == pytest.approx(0.5 * vehicles[0].F_brk_max_N)
    assert u_brk_cmd(30.0, 1) == pytest.approx(0.5 * vehicles[1].F_brk_max_N)
