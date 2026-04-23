"""
Demo: climb steep grade with low throttle. Prints whether speed stalls (drops to near zero).
Run from repo root: python simulator/physics/scripts/demo_grade_climb.py
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "simulator" / "physics" / "core"))

import torch
from railphysics import (
    State,
    default_loco_params,
    RouteProfileTorch,
    rollout_constant_action,
)

def main():
    torch.manual_seed(42)
    B = 1
    # Steep grade 2%
    route = RouteProfileTorch(
        x_m=torch.tensor([0.0, 5000.0]),
        grade=torch.tensor([0.02, 0.02]),
        curvature=torch.tensor([0.0, 0.0]),
    )
    state = State(x=torch.zeros(B), v=torch.full((B,), 8.0))
    params = default_loco_params(batch_size=B)
    dt = 0.5
    num_steps = 300
    throttle = 0.15
    states, _ = rollout_constant_action(
        state, throttle=throttle, brake=0.0,
        route=route, params=params,
        dt=dt, num_steps=num_steps,
    )
    v_final = states["v"][-1].item()
    v_min = states["v"].min().item()
    stalled = v_final < 0.5
    print(f"Grade climb: 2% grade, throttle={throttle}, initial v=8 m/s")
    print(f"  Final speed: {v_final:.4f} m/s")
    print(f"  Min speed: {v_min:.4f} m/s")
    print(f"  Stalled (v < 0.5 m/s): {stalled}")


if __name__ == "__main__":
    main()
