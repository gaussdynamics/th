"""
Batched rollout: simulate T steps from initial state with given actions.
Returns states [T+1, B] and per-step force breakdowns.
"""
import torch

from .typing import RouteProfile, State
from .params import LocomotiveParams
from .integrators import step


def rollout(
    initial_state: State,
    actions: dict[str, torch.Tensor],
    route: RouteProfile,
    params: LocomotiveParams,
    dt: float,
) -> tuple[dict[str, torch.Tensor], list]:
    """
    Roll out dynamics for T steps.
    initial_state: State with .x, .v shape [B]
    actions: dict with 'throttle' and 'brake', each shape [T, B]
    dt: time step (s)
    Returns:
      states: dict with 'x', 'v' tensors shape [T+1, B]
      breakdowns: list of T ForceBreakdowns (each with [B] tensors)
    """
    throttle = actions["throttle"]
    brake = actions["brake"]
    T, B = throttle.shape[0], throttle.shape[1]
    assert brake.shape == (T, B), f"brake shape {brake.shape} vs (T,B)=({T},{B})"

    x = initial_state.x
    v = initial_state.v
    assert x.shape[0] == B and v.shape[0] == B

    xs = [x]
    vs = [v]
    breakdowns = []

    for t in range(T):
        x_next, v_next, bd = step(
            x, v,
            throttle[t], brake[t],
            route, params, dt,
        )
        xs.append(x_next)
        vs.append(v_next)
        breakdowns.append(bd)
        x, v = x_next, v_next

    states = {
        "x": torch.stack(xs, dim=0),
        "v": torch.stack(vs, dim=0),
    }
    return states, breakdowns


def rollout_constant_action(
    initial_state: State,
    throttle: float,
    brake: float,
    route: RouteProfile,
    params: LocomotiveParams,
    dt: float,
    num_steps: int,
    device=None,
) -> tuple[dict[str, torch.Tensor], list]:
    """
    Roll out with constant throttle and brake. Convenience for demos/tests.
    initial_state.x, .v shape [B]. throttle, brake are scalars in [0,1].
    """
    B = initial_state.x.shape[0]
    if device is None:
        device = initial_state.x.device
    th = torch.full((num_steps, B), throttle, device=device, dtype=initial_state.x.dtype)
    br = torch.full((num_steps, B), brake, device=device, dtype=initial_state.x.dtype)
    actions = {"throttle": th, "brake": br}
    return rollout(initial_state, actions, route, params, dt)


def breakdowns_to_tensors(breakdowns: list) -> dict[str, torch.Tensor]:
    """Stack list of ForceBreakdown into tensors [T, B]."""
    if not breakdowns:
        return {}
    return {
        "F_trac": torch.stack([b.F_trac for b in breakdowns], dim=0),
        "F_brake": torch.stack([b.F_brake for b in breakdowns], dim=0),
        "F_resist": torch.stack([b.F_resist for b in breakdowns], dim=0),
        "F_grade": torch.stack([b.F_grade for b in breakdowns], dim=0),
        "F_curve": torch.stack([b.F_curve for b in breakdowns], dim=0),
        "F_net": torch.stack([b.F_net for b in breakdowns], dim=0),
    }
