"""Tensor-state longitudinal train dynamics (ported from the LTD buildup notebook)."""

from . import constants, forces, rhs, route, simulate, state_schema, tensor_state
from .io_types import ExtendedTrainScenario, TensorSimulationResult
from .params import CouplerParameters, TrainSimulationConfig, VehicleParameters
from .route import RouteProfile
from .scenarios import make_simple_train, make_simple_train_scenario
from .simulate import simulate_train, simulate_train_tensorized

__all__ = [
    "constants",
    "forces",
    "rhs",
    "route",
    "simulate",
    "state_schema",
    "tensor_state",
    "CouplerParameters",
    "VehicleParameters",
    "TrainSimulationConfig",
    "RouteProfile",
    "ExtendedTrainScenario",
    "TensorSimulationResult",
    "make_simple_train",
    "make_simple_train_scenario",
    "simulate_train",
    "simulate_train_tensorized",
]
