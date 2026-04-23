"""
Net acceleration and step function for single-loco longitudinal dynamics.
State: {x, v}; action: throttle, brake in [0,1]. Mutual exclusion applied.
"""
import torch

from .params import LocomotiveParams
from .typing import RouteProfile
from .forces import (
    davis_resistance,
    grade_force,
    curve_force,
    traction_force,
    brake_force,
    effective_throttle_brake,
)
from .diagnostics import ForceBreakdown


def compute_forces(
    x: torch.Tensor,
    v: torch.Tensor,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    route: RouteProfile,
    params: LocomotiveParams,
) -> ForceBreakdown:
    """
    Compute all force components. x, v, throttle, brake shape [B].
    Returns ForceBreakdown with F_trac, F_brake, F_resist, F_grade, F_curve, F_net [B].
    """
    B = x.shape[0]
    p = params.for_batch(B, device=x.device)

    eff_throttle, eff_brake = effective_throttle_brake(throttle, brake)
    grade = route.grade(x)
    curvature = route.curvature(x)

    F_resist = davis_resistance(v, p.A, p.B, p.C)
    F_grade = grade_force(p.mass, grade)
    F_curve = curve_force(p.mass, curvature, p.c_kappa)
    F_trac = traction_force(
        v, eff_throttle, p.F_max, p.P_max, p.eta, p.mu, p.mass, p.v_eps
    )
    F_brake = brake_force(eff_brake, p.F_brake_max, p.mu_brake, p.mass)

    F_net = F_trac - F_brake - F_resist - F_grade - F_curve
    return ForceBreakdown(
        F_trac=F_trac,
        F_brake=F_brake,
        F_resist=F_resist,
        F_grade=F_grade,
        F_curve=F_curve,
        F_net=F_net,
    )


def acceleration(
    x: torch.Tensor,
    v: torch.Tensor,
    throttle: torch.Tensor,
    brake: torch.Tensor,
    route: RouteProfile,
    params: LocomotiveParams,
) -> torch.Tensor:
    """a = F_net / m. Shape [B]."""
    breakdown = compute_forces(x, v, throttle, brake, route, params)
    B = x.shape[0]
    p = params.for_batch(B, device=x.device)
    return breakdown.F_net / p.mass
