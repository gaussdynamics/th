"""Per-vehicle v3 physics (:mod:`simulator2.vehicle_physics`)."""

from __future__ import annotations

import numpy as np
import pytest

from simulator2 import ltd_benchmark as lb
from simulator2.catalog import CarCategory, CarType
from simulator2.params import VehicleParameters
from simulator2.vehicle_physics import (
    LEAD_LOCO_FACTOR,
    RESISTANCE_BY_CATEGORY,
    ResistanceClass,
    apply_modified_davis,
    modified_davis,
)


def _per_tonne(abc: tuple[float, float, float], mass_kg: float, v_kmh: float) -> float:
    a, b, c = abc
    v = v_kmh / 3.6
    return (a + b * v + c * v * v) / (mass_kg / 1000.0)


def test_modified_davis_matches_handbook_equation() -> None:
    """Per-vehicle A/B/C reproduce Table 13.3's N/tonne form at any speed."""
    rc = ResistanceClass(ka=1.05, kad=0.07)
    m, n = 120_000.0, 4
    abc = modified_davis(m, n, rc)
    m_a = m / 1000.0 / n
    for v_kmh in (0.0, 30.0, 80.0):
        handbook = rc.ka * (2.943 + 89 / m_a + 0.0305 * v_kmh
                            + 1.718 * rc.kad * v_kmh ** 2 / (m_a * n))
        assert _per_tonne(abc, m, v_kmh) == pytest.approx(handbook, rel=1e-12)


def test_modified_davis_matches_the_benchmark_law() -> None:
    """Ka = 1, k_ad = 0.07 is the LTD benchmark's Eq. 1, as already implemented
    in ltd_benchmark.davis_coefficients (which rounds to 89.2, 0.0306, 0.122)."""
    consist = lb.build_consist(1)
    ref = lb.davis_coefficients(consist)
    rc = ResistanceClass(ka=1.0, kad=0.07)
    for i, vt in enumerate(consist.types[:4]):
        n = round(vt.mass_t / vt.axle_load_t)
        ours = modified_davis(vt.mass_t * 1e3, n, rc, LEAD_LOCO_FACTOR if i == 0 else 1.0)
        assert np.allclose(ours, ref[i], rtol=0.015)


def test_lead_locomotive_factor_and_categories() -> None:
    loco = CarType(id="loco_sd70", name="", category=CarCategory.LOCOMOTIVE,
                   length_m=22.0, mass_tare_kg=180_000.0)
    car = CarType(id="coal_hopper_loaded", name="", category=CarCategory.HOPPER,
                  length_m=16.0, mass_tare_kg=120_000.0)
    vehicles = [VehicleParameters(mass_kg=ct.mass_kg, davis_A=0, davis_B=0, davis_C=0)
                for ct in (loco, loco, car)]
    out = apply_modified_davis(vehicles, [loco, loco, car])
    assert out[0].davis_A == pytest.approx(LEAD_LOCO_FACTOR * out[1].davis_A)
    assert out[0].davis_C == pytest.approx(LEAD_LOCO_FACTOR * out[1].davis_C)
    rc = RESISTANCE_BY_CATEGORY[CarCategory.HOPPER]
    assert (out[2].davis_A, out[2].davis_B, out[2].davis_C) == pytest.approx(
        modified_davis(120_000.0, 4, rc))
    # F4: the catalog's A/B for a loaded 4-axle car were 2-4x low
    assert out[2].davis_A > 700.0 and out[2].davis_B > 13.0
    assert vehicles[0].davis_A == 0                          # inputs untouched


def test_inertia_factor_slows_acceleration_but_not_grade_weight() -> None:
    """F9: a vehicle with inertia factor 1 + J_f on a grade decelerates at
    g sin(theta) / (1 + J_f): the grade force uses m, the inertia m (1 + J_f)."""
    import math

    from simulator2.constants import GRAVITY_MPS2
    from simulator2.verification import Case, rk4_fixed
    from simulator2.vehicle_physics import ROTATING_MASS_FRACTION, inertia_factors

    st, f = 0.02, 1.125
    case = Case(mass=np.full(1, 1e5), x0=np.zeros(1), v0=np.full(1, 20.0),
                sin_theta=st, grade_model="exact", inertia_factor=np.full(1, f))
    t, y = rk4_fixed(case.batch(), 20.0, 0.02, every=100)
    assert np.abs(y[:, 1] - (20.0 - GRAVITY_MPS2 * st / f * t)).max() < 1e-9
    loco = CarType(id="loco_sd70", name="", category=CarCategory.LOCOMOTIVE,
                   length_m=22.0, mass_tare_kg=1.0)
    car = CarType(id="x", name="", category=CarCategory.TANK, length_m=15.0,
                  mass_tare_kg=1.0)
    assert inertia_factors([loco, car]) == [1 + ROTATING_MASS_FRACTION["locomotive"],
                                           1 + ROTATING_MASS_FRACTION["wagon"]]
    assert math.isclose(ROTATING_MASS_FRACTION["wagon"], 0.015)
