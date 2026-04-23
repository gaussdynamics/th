"""
Force breakdown container for per-step diagnostics.
All tensors shape [B].
"""
from dataclasses import dataclass
import torch


@dataclass
class ForceBreakdown:
    """Per-step force components (N). All tensors shape [B]."""

    F_trac: torch.Tensor
    F_brake: torch.Tensor
    F_resist: torch.Tensor
    F_grade: torch.Tensor
    F_curve: torch.Tensor
    F_net: torch.Tensor

    def to(self, device: torch.device) -> "ForceBreakdown":
        return ForceBreakdown(
            F_trac=self.F_trac.to(device),
            F_brake=self.F_brake.to(device),
            F_resist=self.F_resist.to(device),
            F_grade=self.F_grade.to(device),
            F_curve=self.F_curve.to(device),
            F_net=self.F_net.to(device),
        )
