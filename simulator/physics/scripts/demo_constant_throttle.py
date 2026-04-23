"""
Demo: constant throttle on level track until near equilibrium. Prints equilibrium speed estimate.
Run from repo root: python simulator/physics/scripts/demo_constant_throttle.py
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "simulator" / "physics" / "core"))

import torch
from railphysics import (
    State,
    default_loco_params,
    level_route,
    rollout_constant_action,
)

def main():
    torch.manual_seed(42)
    B = 1
    state = State(x=torch.zeros(B), v=torch.zeros(B))
    params = default_loco_params(batch_size=B)
    route = level_route(length_m=100_000.0)
    dt = 0.5
    num_steps = 500
    throttle = 0.25
    states, _ = rollout_constant_action(
        state, throttle=throttle, brake=0.0,
        route=route, params=params,
        dt=dt, num_steps=num_steps,
    )
    v_last = states["v"][-20:].mean().item()
    v_max = states["v"].max().item()
    print(f"Constant throttle: level track, throttle={throttle}")
    print(f"  Equilibrium speed (mean of last 20 steps): {v_last:.2f} m/s")
    print(f"  Max speed reached: {v_max:.2f} m/s")


if __name__ == "__main__":
    main()
