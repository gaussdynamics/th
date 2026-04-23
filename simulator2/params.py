"""Dataclasses for vehicle, coupler, and simulation configuration (notebook-aligned)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class VehicleParameters:
    mass_kg: float
    davis_A: float
    davis_B: float
    davis_C: float
    can_traction: bool = False
    F_trac_max_N: float = 0.0
    F_brk_max_N: float = 250_000.0


@dataclass
class CouplerParameters:
    L0_m: float
    slack_half_m: float
    k_draft: float
    c_draft: float
    k_buff: float
    c_buff: float


@dataclass
class TrainSimulationConfig:
    """Configuration for time integration (matches notebook ``TrainSimulationConfig``)."""

    vehicles: List[VehicleParameters]
    couplers: List[CouplerParameters]
    route: "RouteProfile"
    t_span: Tuple[float, float]
    t_eval: Optional[np.ndarray] = None
    k_curv_scale: float = 0.0
    rtol: float = 1e-6
    atol: float = 1e-8
    method: str = "RK45"
