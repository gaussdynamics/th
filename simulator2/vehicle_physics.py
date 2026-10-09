"""Per-vehicle physics for the v3 configuration (:mod:`simulator2.physics_config`).

Replaces catalog values that ``SIMULATOR_VALIDATION_NOTE.md`` found
unsourced, vehicle by vehicle, when a consist is built:

* **F4, propulsion resistance.** The modified Davis equation (USA) as given
  in the Handbook of Railway Vehicle Dynamics, 2nd ed., Table 13.3 (Eq. 13.8):

      R = Ka [2.943 + 89/m_a + 0.0305 V + 1.718 k_ad V^2 / (m_a n)]  N/tonne,

  with m_a the axle mass [t], n the number of axles and V [km/h]. Multiplied
  by the vehicle mass m = m_a n [t] it is Davis-form per vehicle:

      F = Ka (2.943 m + 89 n) + Ka 0.0305 m V + 1.718 Ka k_ad V^2  [N].

  This is the law of the LTD benchmark (Spiryagin, Wu & Cole 2017, Eq. 1,
  coefficients rounded there to 89.2 and 0.0306), which the simulator was
  validated against, including its factor ``Q = 3.2`` on the lead locomotive.
  The benchmark applies ``Q`` to the whole equation, so the same is done here.
* **F9, rotating mass.** Handbook Eq. 13.2 writes each vehicle's inertia as
  ``m (1 + J_f)``, J_f being the wheelsets', gearboxes' and motors' rotating
  inertia as translational mass: "typically 1%-2% of loaded wagon mass and
  10%-15% of locomotive mass" (p. 459). The midpoints are used, as fractions
  of the vehicle's own mass (the library's cars are loaded; for an empty car
  this underestimates J_f, which does not scale with lading).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .catalog import CarCategory, CarType
from .params import VehicleParameters

__all__ = [
    "HANDBOOK_T133", "BENCHMARK_EQ1", "ResistanceClass", "RESISTANCE_BY_CATEGORY",
    "LEAD_LOCO_FACTOR", "axles_of", "modified_davis", "apply_modified_davis",
    "ROTATING_MASS_FRACTION", "inertia_factors",
]

HANDBOOK_T133 = ("Iwnicki (ed.), Handbook of Railway Vehicle Dynamics, 2nd ed. (2019), "
                 "Table 13.3, modified Davis equation (USA)")
BENCHMARK_EQ1 = ("Spiryagin, Wu & Cole (2017), International benchmarking of longitudinal "
                 "train dynamics simulators: benchmarking questions, VSD 55(4), Eq. 1")

KMH_PER_MPS = 3.6


@dataclass(frozen=True)
class ResistanceClass:
    """Modified Davis factors for one kind of vehicle (Handbook Table 13.3)."""

    #: Equipment factor Ka: 0.85 post-1950 equipment, 1.05 hopper cars, 0.95
    #: container on flatcar.
    ka: float
    #: Aerodynamic factor k_ad: 0.07 conventional equipment, 0.0935 containers.
    kad: float


#: Ka and k_ad by car category. Locomotives take Ka = 1.0 with the benchmark's
#: lead factor instead (the Table 13.3 Ka list has no locomotive entry).
RESISTANCE_BY_CATEGORY: dict[CarCategory, ResistanceClass] = {
    CarCategory.LOCOMOTIVE: ResistanceClass(ka=1.0, kad=0.07),
    CarCategory.HOPPER: ResistanceClass(ka=1.05, kad=0.07),
    CarCategory.TANK: ResistanceClass(ka=0.85, kad=0.07),
    CarCategory.BOXCAR: ResistanceClass(ka=0.85, kad=0.07),
    CarCategory.GONDOLA: ResistanceClass(ka=0.85, kad=0.07),
    CarCategory.FLATCAR: ResistanceClass(ka=0.95, kad=0.0935),   # intermodal well car
}

#: Benchmark Eq. 1: ``Q = 3.2`` for the leading locomotive, 1 otherwise.
LEAD_LOCO_FACTOR = 3.2

#: Axles per car type. Every freight car type in the library rides on two
#: 2-axle trucks; the SD70 family is C-C (6 axles), the GP38 B-B (4).
_AXLES_BY_TYPE = {"loco_sd70": 6, "loco_gp38": 4}


def axles_of(car_type: CarType) -> int:
    return _AXLES_BY_TYPE.get(car_type.id, 4)


def modified_davis(mass_kg: float, n_axles: int, rc: ResistanceClass,
                   q: float = 1.0) -> tuple[float, float, float]:
    """Davis ``(A [N], B [N s/m], C [N s^2/m^2])`` of one vehicle, SI speed."""
    m_t = mass_kg / 1000.0
    a = q * rc.ka * (2.943 * m_t + 89.0 * n_axles)
    b = q * rc.ka * 0.0305 * m_t * KMH_PER_MPS
    c = q * 1.718 * rc.ka * rc.kad * KMH_PER_MPS ** 2
    return a, b, c


def apply_modified_davis(vehicles: Sequence[VehicleParameters],
                         car_types: Sequence[CarType],
                         n_axles: Sequence[int] | None = None) -> list[VehicleParameters]:
    """Vehicles with Davis A/B/C from the modified Davis equation (F4).

    The lead factor applies to the first vehicle if it is a locomotive.
    ``n_axles`` defaults to :func:`axles_of` per car type.
    """
    axles = [axles_of(ct) for ct in car_types] if n_axles is None else list(n_axles)
    out = []
    for i, (v, ct, n) in enumerate(zip(vehicles, car_types, axles, strict=True)):
        rc = RESISTANCE_BY_CATEGORY[ct.category]
        lead = i == 0 and ct.category == CarCategory.LOCOMOTIVE
        a, b, c = modified_davis(v.mass_kg, n, rc, LEAD_LOCO_FACTOR if lead else 1.0)
        out.append(replace(v, davis_A=a, davis_B=b, davis_C=c))
    return out


#: J_f as a fraction of vehicle mass (Handbook p. 459, midpoints).
ROTATING_MASS_FRACTION = {"wagon": 0.015, "locomotive": 0.125}


def inertia_factors(car_types: Sequence[CarType]) -> list[float]:
    """``1 + J_f`` per vehicle (F9)."""
    return [1.0 + ROTATING_MASS_FRACTION[
        "locomotive" if ct.category == CarCategory.LOCOMOTIVE else "wagon"]
        for ct in car_types]


# -- F5: sourced locomotives ------------------------------------------------------

SPIRYAGIN2017_CH2 = ("Spiryagin, Wolfs, Cole, Spiryagin, Sun & McSweeney (2017), Design and "
                     "Simulation of Heavy Haul Locomotives and Trains, CRC Press, Ch. 2")


@dataclass(frozen=True)
class LocoSpec:
    """A sourced locomotive: mass, length and its notch table.

    ``table`` is the ``loco_table`` row of ``torch_rhs`` (``LocoParam``):
    ``(Te_max N, P_max W, DB_max N, P_db_max W, v_db_min m/s, v_db_max m/s)``.
    """

    name: str
    mass_kg: float
    length_m: float
    n_axles: int
    table: tuple[float, float, float, float, float, float]
    source: str


#: SD70ACe: Spiryagin et al. Table 2.9 (195 t, 22.63 m, C-C, 850 kN starting,
#: 472 kN dynamic brake). Table 2.9 gives no traction power; the LTD
#: benchmark's locomotive type 2, the same 195 t / 850 kN / 472 kN unit
#: (Spiryagin, Wu & Cole 2017, Tables 3 and 6), supplies P_max 3207 kW and the
#: dynamic brake curve (5500 kW, 5-43 km/h), as in ``ltd_benchmark``.
SD70ACE = LocoSpec(
    name="SD70ACe", mass_kg=195_000.0, length_m=22.63, n_axles=6,
    table=(850e3, 3207e3, 472e3, 5500e3, 1.39, 11.94),
    source=f"{SPIRYAGIN2017_CH2}, Table 2.9; {BENCHMARK_EQ1.split(', Eq.')[0]}, Tables 3 and 6",
)

#: ES44DC: Spiryagin et al. Table 2.3 (189.6 t, 22.3 m, C-C, traction power
#: 4390 hp = 3274 kW, 631.64 kN starting, 332.17 kN dynamic brake from 42 down
#: to 14.5 km/h). The dynamic brake power is not tabulated; P_db_max is set to
#: DB_max x 42 km/h so the curve is continuous at that corner (derived).
ES44DC = LocoSpec(
    name="ES44DC", mass_kg=189_600.0, length_m=22.30, n_axles=6,
    table=(631.64e3, 4390 * 745.7, 332.17e3, 332.17e3 * 42.0 / 3.6, 14.5 / 3.6, 42.0 / 3.6),
    source=f"{SPIRYAGIN2017_CH2}, Table 2.3 (P_db_max derived)",
)

#: Which sourced locomotive replaces each illustrative catalog entry under v3.
LOCO_SUBSTITUTES = {"loco_sd70": SD70ACE, "loco_gp38": ES44DC}


def sourced_car_type(ct: CarType) -> tuple[CarType, LocoSpec | None]:
    """``ct`` with a locomotive replaced by its sourced substitute (F5).

    Cars pass through unchanged. ``F_trac_max_N`` becomes the starting
    tractive effort, which the notch model uses as Te_max.
    """
    spec = LOCO_SUBSTITUTES.get(ct.id)
    if spec is None:
        if ct.category == CarCategory.LOCOMOTIVE:
            raise ValueError(f"no sourced locomotive for car type {ct.id!r}")
        return ct, None
    return replace(ct, name=spec.name, length_m=spec.length_m, mass_tare_kg=spec.mass_kg,
                   mass_lading_kg=0.0, F_trac_max_N=spec.table[0]), spec


def axles_of_spec(ct: CarType, spec: LocoSpec | None) -> int:
    return spec.n_axles if spec is not None else axles_of(ct)
