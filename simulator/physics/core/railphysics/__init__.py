"""
Rail longitudinal dynamics physics core.
Single locomotive, Torch-based, batched; extensible to multi-vehicle + couplers.
"""
from .units import G, V_EPS_DEFAULT, ensure_tensor
from .typing import RouteProfile, State
from .params import LocomotiveParams, default_loco_params
from .route import RouteProfileTorch, level_route
from .forces import (
    davis_resistance,
    grade_force,
    curve_force,
    traction_force,
    brake_force,
    effective_throttle_brake,
)
from .diagnostics import ForceBreakdown
from .dynamics import compute_forces, acceleration
from .integrators import semi_implicit_euler, step, step_state
from .rollout import rollout, rollout_constant_action, breakdowns_to_tensors

__all__ = [
    "G",
    "V_EPS_DEFAULT",
    "ensure_tensor",
    "RouteProfile",
    "State",
    "LocomotiveParams",
    "default_loco_params",
    "RouteProfileTorch",
    "level_route",
    "davis_resistance",
    "grade_force",
    "curve_force",
    "traction_force",
    "brake_force",
    "effective_throttle_brake",
    "ForceBreakdown",
    "compute_forces",
    "acceleration",
    "semi_implicit_euler",
    "step",
    "step_state",
    "rollout",
    "rollout_constant_action",
    "breakdowns_to_tensors",
]
