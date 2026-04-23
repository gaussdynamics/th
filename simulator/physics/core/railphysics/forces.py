"""
Force components for rail longitudinal dynamics.
All functions vectorized over batch B; inputs/outputs tensors shape [B] unless noted.
Units: N, m/s, m, kg; grade decimal; curvature 1/m.
"""
import torch

from .units import G


def davis_resistance(v: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """F_resist = A + B*v + C*v^2. v, A, B, C same shape [B]. Returns [B] in N."""
    return A + B * v + C * (v * v)


def grade_force(mass: torch.Tensor, grade: torch.Tensor) -> torch.Tensor:
    """F_grade = m * g * grade. grade is decimal slope. Returns [B] in N (positive uphill = resisting)."""
    return mass * G * grade


def curve_force(mass: torch.Tensor, curvature: torch.Tensor, c_kappa: torch.Tensor) -> torch.Tensor:
    """F_curve = m * g * c_kappa * |curvature|. Returns [B] in N."""
    return mass * G * c_kappa * curvature.abs()


def traction_force(
    v: torch.Tensor,
    throttle: torch.Tensor,
    F_max: torch.Tensor,
    P_max: torch.Tensor,
    eta: torch.Tensor,
    mu: torch.Tensor,
    mass: torch.Tensor,
    v_eps: torch.Tensor,
) -> torch.Tensor:
    """
    Effective traction from throttle, limited by command, power, and adhesion.
    F_cmd = throttle * F_max
    F_power = (eta * P_max) / max(v, v_eps)
    F_adh = mu * m * g
    F_trac = min(F_cmd, F_power, F_adh)
    """
    F_cmd = throttle * F_max
    v_safe = torch.clamp(v, min=v_eps)
    F_power = (eta * P_max) / v_safe
    F_adh = mu * mass * G
    F_trac = torch.minimum(torch.minimum(F_cmd, F_power), F_adh)
    return F_trac.clamp(min=0.0)


def brake_force(
    brake: torch.Tensor,
    F_brake_max: torch.Tensor,
    mu_brake: torch.Tensor,
    mass: torch.Tensor,
) -> torch.Tensor:
    """
    F_brake_raw = brake * F_brake_max; F_brake = min(F_brake_raw, mu_brake * m * g).
    """
    F_raw = brake * F_brake_max
    F_adh_brake = mu_brake * mass * G
    return torch.minimum(F_raw, F_adh_brake).clamp(min=0.0)


def effective_throttle_brake(throttle: torch.Tensor, brake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mutual exclusion: effective_throttle = throttle * (1 - brake),
    effective_brake = brake * (1 - throttle).
    """
    eff_throttle = throttle * (1.0 - brake)
    eff_brake = brake * (1.0 - throttle)
    return eff_throttle, eff_brake
