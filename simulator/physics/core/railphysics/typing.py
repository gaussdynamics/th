"""
Types and protocols for rail physics (minimal typing).
"""
from dataclasses import dataclass
from typing import Protocol
import torch


@dataclass
class State:
    """Single-loco state: position x (m), speed v (m/s). Tensors shape [B]."""

    x: torch.Tensor
    v: torch.Tensor


class RouteProfile(Protocol):
    """Protocol for route profile: grade(x) and curvature(x) for batched x."""

    def grade(self, x: torch.Tensor) -> torch.Tensor:
        """Return grade (decimal slope) at positions x. Shape: same as x."""
        ...

    def curvature(self, x: torch.Tensor) -> torch.Tensor:
        """Return curvature (1/m) at positions x. Shape: same as x."""
        ...
