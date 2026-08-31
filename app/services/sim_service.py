"""Wraps make_scenario_from_consist + simulate_train_tensorized for the Simulate screen.

Pure functions over the backend. No Qt in here -- easy to unit test, and the
only file (besides the worker that calls it) that imports simulator2 for the
simulation path. Takes plain vehicle/coupler lists rather than a Train, so it
has no dependency on the catalog/consist services -- ConsistView.build_consist()
resolves those, SimView just passes the result through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from simulator2.io_types import ExtendedTrainScenario, TensorSimulationResult
from simulator2.params import CouplerParameters, VehicleParameters
from simulator2.route import load_route_profile
from simulator2.scenarios import make_scenario_from_consist
from simulator2.simulate import simulate_train_tensorized


class SimService:
    def build_scenario(
        self,
        route_path: Path | str,
        vehicles: List[VehicleParameters],
        couplers: List[CouplerParameters],
        *,
        t_span: Tuple[float, float] = (0.0, 120.0),
        f_trac_max_N: Optional[float] = None,
        k_curv_scale: float = 0.0,
        u_trac_cmd: Optional[Callable[[float, int], float]] = None,
        u_brk_cmd: Optional[Callable[[float, int], float]] = None,
    ) -> ExtendedTrainScenario:
        route = load_route_profile(route_path)
        return make_scenario_from_consist(
            route,
            vehicles,
            couplers,
            t_span=t_span,
            F_trac_max_N=f_trac_max_N,
            k_curv_scale=k_curv_scale,
            u_trac_cmd=u_trac_cmd,
            u_brk_cmd=u_brk_cmd,
        )

    def run(self, scenario: ExtendedTrainScenario) -> TensorSimulationResult:
        return simulate_train_tensorized(scenario)  # the blocking solve_ivp
