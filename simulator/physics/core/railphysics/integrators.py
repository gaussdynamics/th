"""
Time integration for longitudinal dynamics.
Semi-implicit Euler: v_next = clamp(v + dt*a, min=0), x_next = x + dt*v_next.
"""
import torch

from .dynamics import compute_forces, acceleration
from .params import LocomotiveParams
from .diagnostics import ForceBreakdown
from .typing import RouteProfile, State


def semi_implicit_euler(
    x: torch.Tensor,
    v: torch.Tensor,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    route: RouteProfile,
    params: LocomotiveParams,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, ForceBreakdown]:
    """
    One step: v_next = clamp(v + dt*a, 0), x_next = x + dt*v_next.
    Returns (x_next, v_next, breakdown).
    """
    a = acceleration(x, v, throttle, brake, route, params)
    v_next = torch.clamp(v + dt * a, min=0.0)
    x_next = x + dt * v_next
    breakdown = compute_forces(x, v, throttle, brake, route, params)
    return x_next, v_next, breakdown


def step(
    x: torch.Tensor,
    v: torch.Tensor,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    route: RouteProfile,
    params: LocomotiveParams,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, ForceBreakdown]:
    """
    Single integration step. State (x, v), action (throttle, brake), dt in seconds.
    Returns (x_next, v_next, ForceBreakdown).
    """
    return semi_implicit_euler(x, v, throttle, brake, route, params, dt)


def step_state(
    state: State,
    action: dict[str, torch.Tensor],
    route: RouteProfile,
    params: LocomotiveParams,
    dt: float,
) -> tuple[State, ForceBreakdown]:
    """
    Single step with State and action dict. action has 'throttle' and 'brake' [B].
    Returns (next_state, ForceBreakdown).
    """
    x_next, v_next, bd = step(
        state.x, state.v,
        action["throttle"], action["brake"],
        route, params, dt,
    )
    return State(x=x_next, v=v_next), bd
