#!/usr/bin/env python3
"""Minimal demo: simple consist + tensorized rollout.

Run from repository root::

    python simulator2/examples/demo_tensorized_sim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from simulator2.scenarios import make_simple_train_scenario
from simulator2.simulate import simulate_train_tensorized
from simulator2.state_schema import NodeChannel


def main() -> None:
    scenario = make_simple_train_scenario(n_cars=8, n_time_samples=601, t_span=(0.0, 60.0))
    res = simulate_train_tensorized(scenario)

    print("H_hist shape:", res.H_hist.shape)
    print("E_hist shape:", res.E_hist.shape)
    print("metadata N, d_node, d_edge:", res.metadata["N"], res.metadata["d_node"], res.metadata["d_edge"])

    n = res.metadata["N"]
    v_lead = res.H_hist[:, 0, int(NodeChannel.V)]
    v_rear = res.H_hist[:, n - 1, int(NodeChannel.V)]
    print("Lead speed final (m/s):", float(v_lead[-1]))
    print("Rear speed final (m/s):", float(v_rear[-1]))
    print("Max |v_lead - v_rear| (m/s):", float(np.max(np.abs(v_lead - v_rear))))

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax[0].plot(res.t, v_lead * 3.6, label="lead")
        ax[0].plot(res.t, v_rear * 3.6, label="rear")
        ax[0].set_ylabel("km/h")
        ax[0].legend()
        ax[0].set_title("Tensorized demo: lead vs rear speed")
        if res.E_hist.shape[1] > 0:
            ax[1].plot(res.t, res.E_hist[:, 0, 2] / 1e3, label="F coupler 0 [kN]")
        ax[1].set_xlabel("time [s]")
        ax[1].legend()
        plt.tight_layout()
        out_png = Path(__file__).resolve().parent / "demo_tensorized_preview.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print("Wrote figure:", out_png)
        try:
            plt.show()
        except Exception:
            pass
    except ImportError:
        print("(matplotlib not installed; skipping plots)")


if __name__ == "__main__":
    main()
