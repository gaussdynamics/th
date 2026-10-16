"""Locomotive power assignment per route (v3, ``power_sizing = "ruling_grade"``).

The generated consists carry locomotive counts sized for the illustrative
catalog locomotives (300 / 180 kN starting, ``train_generator``). The sourced
SD70ACe and ES44DC are two to three times stronger, so under v3 the count is
re-derived for each scenario from the route it runs on. That is how a railroad
assigns power: enough units to haul the train up the ruling grade of the
subdivision.

Sizing rule. The fewest locomotives ``n`` whose combined maximum continuous
tractive effort (Spiryagin et al. 2017, Tables 2.3 and 2.9) holds the whole
train, locomotives included, at their minimum continuous speed on the ruling
grade:

    n TE_cont >= sum_i [ R_davis,i(v_cont) + m_i g sin(theta_ruling) ]

with ``R_davis`` the modified Davis resistance of the v3 physics (lead factor
on the first locomotive). The ruling grade is the steepest climb averaged over
the train's trailing length (the train's weight is spread along it), taken
over the whole corridor in the direction of travel, so a train sized for a
corridor can run any segment of it. Curvature resistance is not included.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from .catalog import CarCategory, CarType
from .constants import GRAVITY_MPS2
from .consist import CarInstance, Train
from .vehicle_physics import (
    LEAD_LOCO_FACTOR, RESISTANCE_BY_CATEGORY, LocoSpec, axles_of, modified_davis,
)

__all__ = ["ruling_sin_theta", "locos_required", "with_loco_count",
           "DPU_MID_LABEL", "DPU_REAR_LABEL", "MAX_LOCOS"]

#: Labels ``train_generator`` gives distributed-power units.
DPU_MID_LABEL = "DPU mid-train"
DPU_REAR_LABEL = "DPU rear helper"

#: Resampling step for the train-length average of the grade [m].
_GRADE_STEP_M = 10.0

#: A count above this means the train cannot be powered for the route; the
#: caller skips the scenario rather than build an implausible consist.
MAX_LOCOS = 12


def ruling_sin_theta(route: Any, train_length_m: float) -> float:
    """Steepest ``sin(theta)`` averaged over ``train_length_m``, uphill in the
    direction of increasing chainage; 0 for a corridor with no climb."""
    s = np.asarray(route.s_nodes_m, dtype=float)
    sin_theta = np.asarray(route.sin_theta_nodes, dtype=float)
    grid = np.arange(s[0], s[-1] + 0.5 * _GRADE_STEP_M, _GRADE_STEP_M)
    g = np.interp(grid, s, sin_theta)
    w = max(1, int(round(train_length_m / _GRADE_STEP_M)))
    if w >= g.size:
        return max(0.0, float(g.mean()))
    c = np.concatenate([[0.0], np.cumsum(g)])
    return max(0.0, float(((c[w:] - c[:-w]) / w).max()))


def _resistance_n(mass_kg: float, n_axles: int, category: CarCategory,
                  v_mps: float, lead: bool) -> float:
    a, b, c = modified_davis(mass_kg, n_axles, RESISTANCE_BY_CATEGORY[category],
                             LEAD_LOCO_FACTOR if lead else 1.0)
    return a + b * v_mps + c * v_mps * v_mps


def locos_required(wagons: Sequence[CarType], spec: LocoSpec, sin_ruling: float,
                   max_locos: int = MAX_LOCOS) -> int:
    """Fewest ``spec`` locomotives that hold ``wagons`` at the continuous rating
    on ``sin_ruling``; ``max_locos + 1`` if even ``max_locos`` cannot."""
    v = spec.v_cont_mps
    sin_ruling = max(0.0, sin_ruling)
    wagon_load = sum(
        _resistance_n(ct.mass_kg, axles_of(ct), ct.category, v, lead=False)
        + ct.mass_kg * GRAVITY_MPS2 * sin_ruling
        for ct in wagons)
    per_loco = spec.mass_kg * GRAVITY_MPS2 * sin_ruling
    load = wagon_load
    for n in range(1, max_locos + 1):
        load += per_loco + _resistance_n(spec.mass_kg, spec.n_axles,
                                         CarCategory.LOCOMOTIVE, v, lead=(n == 1))
        if n * spec.te_cont_n >= load:
            return n
    return max_locos + 1


def with_loco_count(train: Train, library: Any, n_locos: int) -> Train:
    """``train`` with its power resized to ``n_locos`` units, keeping its layout.

    A head-end-only train stays head-end only. A distributed-power train keeps
    at least one unit at the head and splits the rest over the remote positions
    it had (mid-train and/or rear) as ``train_generator`` does; with one unit
    it becomes head-end only. Wagons keep their order.
    """
    if n_locos < 1:
        raise ValueError("a train needs at least one locomotive")
    is_loco = [library.get(c.car_type_id).category == CarCategory.LOCOMOTIVE
               for c in train.cars]
    if not is_loco[0]:
        raise ValueError(f"train {train.name!r} is not led by a locomotive")
    loco_id = train.cars[0].car_type_id
    wagons = [c for c, lo in zip(train.cars, is_loco) if not lo]
    labels = {c.label for c, lo in zip(train.cars, is_loco) if lo}
    has_mid, has_rear = DPU_MID_LABEL in labels, DPU_REAR_LABEL in labels

    head_n, mid_n, rear_n = n_locos, 0, 0
    if (has_mid or has_rear) and n_locos >= 2:
        head_n = max(1, n_locos - n_locos // 2)
        rest = n_locos - head_n
        if has_mid and has_rear:
            mid_n, rear_n = rest - rest // 2, rest // 2
        elif has_mid:
            mid_n = rest
        else:
            rear_n = rest

    mid_idx = len(wagons) // 2
    cars = ([CarInstance(loco_id) for _ in range(head_n)]
            + wagons[:mid_idx]
            + [CarInstance(loco_id, label=DPU_MID_LABEL) for _ in range(mid_n)]
            + wagons[mid_idx:]
            + [CarInstance(loco_id, label=DPU_REAR_LABEL) for _ in range(rear_n)])
    return replace(train, cars=cars)
