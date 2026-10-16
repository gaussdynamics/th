"""Per-route locomotive power assignment (:mod:`simulator2.power_assignment`)."""

from __future__ import annotations

import numpy as np
import pytest

from simulator2 import physics_config as pc
from simulator2.catalog import CarCategory
from simulator2.consist import CarInstance, Train
from simulator2.constants import GRAVITY_MPS2
from simulator2.power_assignment import (
    DPU_MID_LABEL, DPU_REAR_LABEL, MAX_LOCOS, locos_required, ruling_sin_theta,
    with_loco_count,
)
from simulator2.route import RouteProfile
from simulator2.train_generator import ensure_extended_car_library
from simulator2.vehicle_physics import ES44DC, SD70ACE, modified_davis, RESISTANCE_BY_CATEGORY

WAGON = "coal_hopper_loaded"   # 100 t, 12 m


@pytest.fixture(scope="module")
def library():
    return ensure_extended_car_library()


def _route(s, sin_theta):
    return RouteProfile(s_nodes_m=np.asarray(s, float), sin_theta_nodes=np.asarray(sin_theta, float))


def _train(head, wagons, mid=0, rear=0, loco="loco_sd70"):
    w = [CarInstance(WAGON, label=f"w{i}") for i in range(wagons)]
    cars = ([CarInstance(loco) for _ in range(head)] + w[:wagons // 2]
            + [CarInstance(loco, label=DPU_MID_LABEL) for _ in range(mid)] + w[wagons // 2:]
            + [CarInstance(loco, label=DPU_REAR_LABEL) for _ in range(rear)])
    return Train(name="t", cars=cars)


# -- ruling grade -------------------------------------------------------------------

def test_ruling_grade_is_the_steepest_train_length_average() -> None:
    assert ruling_sin_theta(_route([0, 5000], [0.01, 0.01]), 800.0) == pytest.approx(0.01)
    # A 200 m climb at 2 % on the flat: a 1000 m train feels a fifth of it.
    s = [0, 2000, 2000.001, 2200, 2200.001, 5000]
    g = [0, 0, 0.02, 0.02, 0, 0]
    assert ruling_sin_theta(_route(s, g), 100.0) == pytest.approx(0.02, rel=1e-3)
    assert ruling_sin_theta(_route(s, g), 1000.0) == pytest.approx(0.004, rel=0.02)
    # Downhill only: no climb to size for.
    assert ruling_sin_theta(_route([0, 5000], [-0.01, -0.01]), 500.0) == 0.0


# -- the sizing rule ----------------------------------------------------------------

def _wagon_types(library, n):
    return [library.get(WAGON)] * n


def _load(library, n_wagons, n_locos, spec, sin_r):
    """The sizing rule's load, written out independently."""
    v = spec.v_cont_mps
    ct = library.get(WAGON)
    a, b, c = modified_davis(ct.mass_kg, 4, RESISTANCE_BY_CATEGORY[CarCategory.HOPPER])
    load = n_wagons * (a + b * v + c * v * v + ct.mass_kg * GRAVITY_MPS2 * sin_r)
    for i in range(n_locos):
        a, b, c = modified_davis(spec.mass_kg, 6, RESISTANCE_BY_CATEGORY[CarCategory.LOCOMOTIVE],
                                 3.2 if i == 0 else 1.0)
        load += a + b * v + c * v * v + spec.mass_kg * GRAVITY_MPS2 * sin_r
    return load


@pytest.mark.parametrize("spec", [SD70ACE, ES44DC])
@pytest.mark.parametrize("sin_r", [0.0, 0.008, 0.015, 0.025])
def test_locos_required_is_the_fewest_that_hold_the_train(library, spec, sin_r) -> None:
    wagons = 100
    n = locos_required(_wagon_types(library, wagons), spec, sin_r)
    assert 1 <= n <= MAX_LOCOS
    assert n * spec.te_cont_n >= _load(library, wagons, n, spec, sin_r)
    if n > 1:
        assert (n - 1) * spec.te_cont_n < _load(library, wagons, n - 1, spec, sin_r)


def test_more_grade_more_power_and_es44dc_needs_more(library) -> None:
    wagons = _wagon_types(library, 100)                     # 10,000 t
    counts = [locos_required(wagons, SD70ACE, g) for g in (0.0, 0.01, 0.02, 0.03)]
    assert counts == sorted(counts) and counts[0] < counts[-1]
    assert locos_required(wagons, ES44DC, 0.02) >= locos_required(wagons, SD70ACE, 0.02)
    # 10,000 t on 1 %: about 1 MN of grade force plus resistance, so 2 SD70ACe.
    assert locos_required(wagons, SD70ACE, 0.01) == 2


def test_unpowerable_train_is_reported(library) -> None:
    assert locos_required(_wagon_types(library, 400), SD70ACE, 0.04) == MAX_LOCOS + 1


def test_continuous_ratings_are_the_tabulated_values() -> None:
    assert SD70ACE.te_cont_n == 698e3 and SD70ACE.v_cont_mps * 3.6 == pytest.approx(14.16)
    assert ES44DC.te_cont_n == 470.08e3 and ES44DC.v_cont_mps * 3.6 == pytest.approx(21.0)


# -- resizing the consist -----------------------------------------------------------

def _layout(train, library):
    return ["L" + (c.label[4] if c.label.startswith("DPU") else "")
            if library.get(c.car_type_id).category == CarCategory.LOCOMOTIVE else "w"
            for c in train.cars]


def test_head_end_train_stays_head_end(library) -> None:
    t = with_loco_count(_train(head=3, wagons=6), library, 2)
    assert _layout(t, library) == ["L", "L"] + ["w"] * 6
    assert [c.label for c in t.cars[2:]] == [f"w{i}" for i in range(6)]   # order kept


def test_distributed_power_is_split_over_the_positions_it_had(library) -> None:
    t = with_loco_count(_train(head=1, wagons=6, mid=1, rear=1), library, 5)
    assert _layout(t, library) == ["L"] * 3 + ["w"] * 3 + ["Lm"] + ["w"] * 3 + ["Lr"]
    t = with_loco_count(_train(head=2, wagons=6, rear=2), library, 3)
    assert _layout(t, library) == ["L", "L"] + ["w"] * 6 + ["Lr"]
    # One unit: head end only.
    t = with_loco_count(_train(head=1, wagons=6, mid=1), library, 1)
    assert _layout(t, library) == ["L"] + ["w"] * 6


def test_resizing_keeps_the_locomotive_type(library) -> None:
    t = with_loco_count(_train(head=1, wagons=4, loco="loco_gp38"), library, 3)
    assert {c.car_type_id for c in t.cars[:3]} == {"loco_gp38"}


# -- the configuration --------------------------------------------------------------

def test_ruling_grade_sizing_needs_sourced_locomotives() -> None:
    assert pc.V3.power_sizing == "ruling_grade" and pc.LEGACY.power_sizing == "manifest"
    with pytest.raises(ValueError, match="power_sizing"):
        pc.PhysicsConfig(power_sizing="ruling_grade")
    # Files written before the field existed read back as manifest sizing.
    old = pc.V3.to_json().replace(', "power_sizing": "ruling_grade"', "")
    assert pc.from_json(old).power_sizing == "manifest"
