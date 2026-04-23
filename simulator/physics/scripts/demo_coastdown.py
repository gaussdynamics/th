"""
Demo: coastdown from 20 m/s on level track. Prints final speed; optionally saves plot.
Run from repo root: python simulator/physics/scripts/demo_coastdown.py
"""
import sys
from pathlib import Path

# Allow importing railphysics when run from repo root (thesis/)
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
    state = State(x=torch.zeros(B), v=torch.full((B,), 20.0))
    params = default_loco_params(batch_size=B)
    route = level_route(length_m=100_000.0)
    dt = 0.5
    num_steps = 200
    states, _ = rollout_constant_action(
        state, throttle=0.0, brake=0.0,
        route=route, params=params,
        dt=dt, num_steps=num_steps,
    )
    final_v = states["v"][-1].item()
    final_x = states["x"][-1].item()
    print(f"Coastdown: initial v=20 m/s, {num_steps} steps dt={dt}s")
    print(f"  Final speed: {final_v:.4f} m/s")
    print(f"  Final position: {final_x:.2f} m")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = torch.arange(num_steps + 1, dtype=torch.float32) * dt
        plt.figure(figsize=(6, 4))
        plt.plot(t.numpy(), states["v"].squeeze().numpy(), label="v (m/s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.title("Coastdown on level track")
        plt.legend()
        out = root / "simulator" / "physics" / "coastdown_plot.png"
        plt.savefig(out, dpi=100)
        print(f"  Plot saved: {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
