"""
Parameter dataclasses for rail physics. All fields are tensors or float (convertible to tensor).
Units: see UNITS.md (m, m/s, s, kg, N, W; grade decimal; curvature 1/m).
"""
from dataclasses import dataclass
from typing import Union
import torch

from .units import ensure_tensor


@dataclass
class LocomotiveParams:
    """Single-locomotive parameters. All in SI: kg, N, W, etc."""

    mass: Union[float, torch.Tensor]  # kg
    # Davis: F_resist = A + B*v + C*v^2  (N, N/(m/s), N/(m/s)^2)
    A: Union[float, torch.Tensor]
    B: Union[float, torch.Tensor]
    C: Union[float, torch.Tensor]
    # Traction: F_max (N), P_max (W), eta (0-1), mu (0-1), v_eps (m/s)
    F_max: Union[float, torch.Tensor]
    P_max: Union[float, torch.Tensor]
    eta: Union[float, torch.Tensor]
    mu: Union[float, torch.Tensor]
    v_eps: Union[float, torch.Tensor] = 0.1
    # Brake: F_brake_max (N), mu_brake (0-1)
    F_brake_max: Union[float, torch.Tensor] = 0.0
    mu_brake: Union[float, torch.Tensor] = 0.0
    # Curve: F_curve = m*g*c_kappa*|curvature(x)|; c_kappa dimensionless
    c_kappa: Union[float, torch.Tensor] = 0.0

    def to(self, device: torch.device) -> "LocomotiveParams":
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            else:
                out[k] = v
        return LocomotiveParams(**out)

    def float(self) -> "LocomotiveParams":
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.float()
            else:
                out[k] = v
        return LocomotiveParams(**out)

    def to_tensors(self, device=None, dtype=torch.float32, batch_size: int = 1):
        """Return a new LocomotiveParams with all scalars as tensors of shape [batch_size] or 0-dim."""
        out = {}
        for k, v in self.__dict__.items():
            t = ensure_tensor(v, dtype=dtype, device=device)
            if t.dim() == 0 and batch_size > 1:
                t = t.expand(batch_size)
            out[k] = t
        return LocomotiveParams(**out)

    def for_batch(self, B: int, device=None, dtype=torch.float32) -> "LocomotiveParams":
        """Return params with all fields as tensors of shape [B] for batched computation."""
        return self.to_tensors(device=device, dtype=dtype, batch_size=B)


def default_loco_params(device=None, batch_size: int = 1) -> LocomotiveParams:
    """Sensible default single-loco params for testing (approx. heavy freight loco)."""
    return LocomotiveParams(
        mass=180_000.0,
        A=4000.0,
        B=80.0,
        C=2.5,
        F_max=600_000.0,
        P_max=4.5e6,
        eta=0.85,
        mu=0.35,
        v_eps=0.1,
        F_brake_max=400_000.0,
        mu_brake=0.25,
        c_kappa=0.0,
    ).to_tensors(device=device, batch_size=batch_size)
