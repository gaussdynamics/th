"""
Constants and small helpers for unit-consistent rail physics.
All units: m, m/s, s, kg, N, W; grade = decimal; curvature = 1/m.
"""
import torch

# Standard gravity (m/s^2)
G = 9.81

# Small velocity (m/s) for power-limit denominator to avoid divide-by-zero
V_EPS_DEFAULT = 0.1


def ensure_tensor(x, dtype=None, device=None):
    """Convert float or list to 0-dim or 1-d tensor; leave existing tensors as-is (optionally cast)."""
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x, dtype=dtype or torch.float32)
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype)
    return t
