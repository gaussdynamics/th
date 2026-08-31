"""Catalog + consist engine (§9 of APP_STACK_SKETCH.md): pure backend, no UI."""

from __future__ import annotations

import numpy as np
import pytest

from simulator2.catalog import CarLibrary, CarType, CarCategory, default_seed_car_types
from simulator2.consist import CarInstance, Train, build_consist, from_blocks
from simulator2.route import RouteProfile
from simulator2.scenarios import make_scenario_from_consist
from simulator2.simulate import simulate_train_tensorized


@pytest.fixture()
def library() -> CarLibrary:
    return CarLibrary(car_types={ct.id: ct for ct in default_seed_car_types()})


def test_seed_library_has_expected_ids(library: CarLibrary):
    ids = {ct.id for ct in library.list_car_types()}
    assert ids == {"loco_sd70", "freight_car_generic", "coal_hopper_loaded", "tank_car_water"}


def test_car_type_mass_is_tare_plus_lading(library: CarLibrary):
    tank = library.get("tank_car_water")
    assert tank.mass_kg == tank.mass_tare_kg + tank.mass_lading_kg


def test_with_overrides_does_not_mutate_original(library: CarLibrary):
    base = library.get("coal_hopper_loaded")
    heavier = base.with_overrides({"mass_lading_kg": 95_000})
    assert heavier.mass_lading_kg == 95_000
    assert base.mass_lading_kg == 70_000  # untouched
    assert heavier is not base


def test_with_overrides_rejects_unknown_field(library: CarLibrary):
    base = library.get("loco_sd70")
    with pytest.raises(ValueError):
        base.with_overrides({"not_a_real_field": 1})


def test_from_blocks_orders_cars_head_to_tail():
    train = from_blocks("test", [("loco_sd70", 1), ("coal_hopper_loaded", 2)])
    assert [c.car_type_id for c in train.cars] == [
        "loco_sd70",
        "coal_hopper_loaded",
        "coal_hopper_loaded",
    ]


def test_build_consist_shapes(library: CarLibrary):
    train = from_blocks("drag", [("loco_sd70", 1), ("coal_hopper_loaded", 4), ("tank_car_water", 2)])
    vehicles, couplers = build_consist(train, library)
    assert len(vehicles) == 7
    assert len(couplers) == 6
    assert vehicles[0].can_traction is True
    assert all(not v.can_traction for v in vehicles[1:])


def test_build_consist_applies_per_car_override(library: CarLibrary):
    train = Train(
        name="one heavy hopper",
        cars=[
            CarInstance("loco_sd70"),
            CarInstance("coal_hopper_loaded", overrides={"mass_lading_kg": 95_000}),
        ],
    )
    vehicles, _ = build_consist(train, library)
    heavy = library.get("coal_hopper_loaded").with_overrides({"mass_lading_kg": 95_000})
    assert vehicles[1].mass_kg == pytest.approx(heavy.mass_kg)


def test_coupler_length_derives_from_adjacent_car_lengths(library: CarLibrary):
    train = from_blocks("lengths", [("loco_sd70", 1), ("tank_car_water", 1)])
    _, couplers = build_consist(train, library)
    loco = library.get("loco_sd70")
    tank = library.get("tank_car_water")
    assert couplers[0].L0_m == pytest.approx(0.5 * (loco.length_m + tank.length_m))


def test_train_json_roundtrip(tmp_path, library: CarLibrary):
    train = from_blocks("roundtrip", [("loco_sd70", 1), ("coal_hopper_loaded", 2)])
    train.cars[1].overrides = {"mass_lading_kg": 12345.0}
    path = train.save(tmp_path, filename="roundtrip")
    loaded = Train.load(path)
    assert loaded.name == train.name
    assert [c.car_type_id for c in loaded.cars] == [c.car_type_id for c in train.cars]
    assert loaded.cars[1].overrides == {"mass_lading_kg": 12345.0}


def test_car_type_json_roundtrip(tmp_path, library: CarLibrary):
    lib = CarLibrary(car_types={ct.id: ct for ct in default_seed_car_types()})
    lib.save(tmp_path)
    reloaded = CarLibrary.load(tmp_path)
    for cid in ("loco_sd70", "freight_car_generic", "coal_hopper_loaded", "tank_car_water"):
        assert reloaded.get(cid) == lib.get(cid)


def test_multi_loco_traction_is_not_hardcoded_to_index_zero(library: CarLibrary):
    """Regression for the §9 'open question': traction must follow can_traction,
    not vehicle position, so a mid-train or rear helper loco also ramps up."""
    train = Train(
        name="two locos",
        cars=[
            CarInstance("coal_hopper_loaded"),
            CarInstance("loco_sd70"),  # helper NOT at index 0
        ],
    )
    vehicles, couplers = build_consist(train, library)
    route = RouteProfile(s_nodes_m=np.array([0.0, 50_000.0]), sin_theta_nodes=np.array([0.0, 0.0]))
    scenario = make_scenario_from_consist(
        route, vehicles, couplers, t_span=(0.0, 30.0), n_time_samples=61
    )
    assert scenario.u_trac_cmd(20.0, 0) == 0.0  # hopper: no traction
    assert scenario.u_trac_cmd(20.0, 1) > 0.0  # helper loco at index 1: ramped up


def test_make_scenario_from_consist_runs_end_to_end(library: CarLibrary):
    train = from_blocks("smoke", [("loco_sd70", 1), ("coal_hopper_loaded", 3), ("tank_car_water", 1)])
    vehicles, couplers = build_consist(train, library)
    route = RouteProfile(s_nodes_m=np.array([0.0, 50_000.0]), sin_theta_nodes=np.array([0.0, 0.0]))
    scenario = make_scenario_from_consist(
        route, vehicles, couplers, t_span=(0.0, 30.0), n_time_samples=61
    )
    result = simulate_train_tensorized(scenario)
    n = len(vehicles)
    assert result.H_hist.shape == (61, n, result.metadata["d_node"])
    assert result.E_hist.shape == (61, n - 1, result.metadata["d_edge"])
