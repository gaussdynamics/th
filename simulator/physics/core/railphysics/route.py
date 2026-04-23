"""
Route profile: piecewise-linear grade and curvature as functions of position.
Torch-only, batched x; clamp at segment ends.
"""
import torch


class RouteProfileTorch:
    """
    Piecewise-linear route. x_km and segment values are 1D tensors.
    grade(x) and curvature(x) support batched x of shape [B]; output shape [B].
    Clamping: x < x_km[0] uses first segment; x > x_km[-1] uses last segment.
    """

    def __init__(
        self,
        x_m: torch.Tensor,
        grade: torch.Tensor,
        curvature: torch.Tensor,
    ):
        """
        x_m: segment start positions in meters, shape [S], strictly increasing
        grade: decimal slope per segment, shape [S]
        curvature: 1/m per segment, shape [S]
        """
        assert x_m.dim() == 1 and grade.dim() == 1 and curvature.dim() == 1
        assert x_m.shape[0] == grade.shape[0] == curvature.shape[0]
        self.x_m = x_m
        self._grade_vals = grade
        self._curvature_vals = curvature

    def grade_at(self, x: torch.Tensor) -> torch.Tensor:
        """Grade (decimal) at positions x. Shape: same as x."""
        return _piecewise_linear(x, self.x_m, self._grade_vals)

    def curvature_at(self, x: torch.Tensor) -> torch.Tensor:
        """Curvature (1/m) at positions x. Shape: same as x."""
        return _piecewise_linear(x, self.x_m, self._curvature_vals)

    def grade(self, x: torch.Tensor) -> torch.Tensor:
        return self.grade_at(x)

    def curvature(self, x: torch.Tensor) -> torch.Tensor:
        return self.curvature_at(x)


def _piecewise_linear(x: torch.Tensor, x_knots: torch.Tensor, y_knots: torch.Tensor) -> torch.Tensor:
    """
    Piecewise-linear interpolation. x shape [B], x_knots [S], y_knots [S].
    Clamp x to [x_knots[0], x_knots[-1]] for interpolation; outside use first/last value.
    Returns shape [B].
    """
    device = x.device
    x_flat = x.reshape(-1)
    x_k = x_knots.to(device)
    y_k = y_knots.to(device)
    x_clamped = torch.clamp(x_flat, x_k[0].item(), x_k[-1].item())
    # Find segment index: right-bound index such that x_k[i] <= x < x_k[i+1], or last index
    # searchsorted right gives index where x would be inserted to keep order (so segment index)
    idx = torch.searchsorted(x_k, x_clamped, right=True)
    idx = torch.clamp(idx - 1, 0, x_k.shape[0] - 2)
    x_lo = x_k[idx]
    x_hi = x_k[idx + 1]
    t = (x_clamped - x_lo) / (x_hi - x_lo + 1e-12)
    y_lo = y_k[idx]
    y_hi = y_k[idx + 1]
    out = y_lo + t * (y_hi - y_lo)
    return out.reshape(x.shape)


def level_route(length_m: float = 100_000.0, device=None) -> RouteProfileTorch:
    """Level route (grade=0, curvature=0) for tests."""
    if device is None:
        device = torch.device("cpu")
    x_m = torch.tensor([0.0, length_m], device=device)
    grade = torch.tensor([0.0, 0.0], device=device)
    curvature = torch.tensor([0.0, 0.0], device=device)
    return RouteProfileTorch(x_m, grade, curvature)
