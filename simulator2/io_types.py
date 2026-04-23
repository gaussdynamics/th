"""Result and scenario types for tensorized simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .params import CouplerParameters, VehicleParameters
from .route import RouteProfile


@dataclass
class TensorSimulationResult:
    """Output of ``simulate_train_tensorized``."""

    t: np.ndarray
    y_flat: np.ndarray
    H_hist: np.ndarray
    E_hist: np.ndarray
    sol: Any
    metadata: Dict[str, Any]


@dataclass
class ExtendedTrainScenario:
    """Everything needed to run the extended (4N-state) tensorized simulator."""

    vehicles: List[VehicleParameters]
    couplers: List[CouplerParameters]
    route: RouteProfile
    y0: np.ndarray
    u_trac_cmd: Callable[[float, int], float]
    u_brk_cmd: Callable[[float, int], float]
    k_curv_scale: float = 0.0
    tau_brk_s: float = 3.0
    tau_trac_s: float = 5.0
    p_max_w: float = 3.5e6
    t_span: Tuple[float, float] = (0.0, 120.0)
    t_eval: Optional[np.ndarray] = None
    rtol: float = 1e-6
    atol: float = 1e-8
    method: str = "RK45"
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
