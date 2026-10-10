#!/usr/bin/env python3
"""Rollout error against horizon for GNS-1, for fig:train_rollout (Chapter 9).

Reads results/gns1_ch9/summary.json (scripts/summarize_gns1.py) and draws the
seed-mean velocity and stretch error of the model, solid, against holding the
start state, dashed, for four splits. The shaded band is the min-max over the
three seeds.

    python paper/scripts/plot_gns1_rollout.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "gns1_ch9" / "summary.json"
OUT = REPO / "paper" / "figures" / "gns1_rollout.pdf"

SPLITS = [
    ("val", "val", "#1f4e79"),
    ("ood_corridor@80", r"ood\_corridor, $N \leq 80$".replace("\\_", "_"), "#2d7552"),
    ("ood_size", "ood_size", "#b23a2c"),
    ("ood_grade@80", r"ood_grade, $N \leq 80$", "#b0701a"),
]
H = 0.25


def main() -> None:
    d = json.loads(SRC.read_text())["splits"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    for ax, (chan, scale, ylabel) in zip(
            axes, [("v", 1.0, "Velocity error (m/s)"), ("d", 1.0, "Stretch error (mm)")]):
        for key, label, col in SPLITS:
            r = d[key]["rollout"]
            hz = sorted(int(k) for k in r)
            t = [k * H for k in hz]
            m = [r[str(k)][f"{chan}_model_med"]["mean"] * scale for k in hz]
            lo = [r[str(k)][f"{chan}_model_med"]["min"] * scale for k in hz]
            hi = [r[str(k)][f"{chan}_model_med"]["max"] * scale for k in hz]
            hold = [r[str(k)][f"{chan}_hold_med"]["mean"] * scale for k in hz]
            ax.plot(t, m, color=col, lw=1.8, marker="o", ms=3.5, label=label)
            ax.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
            ax.plot(t, hold, color=col, lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rollout horizon (s)")
        ax.set_ylabel(ylabel)
        ax.grid(which="major", color="0.9")
    axes[0].plot([], [], color="0.4", lw=1.0, ls="--", label="hold (same split)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[0].set_title("(a) Velocity", loc="left", fontsize=10)
    axes[1].set_title("(b) Coupler stretch", loc="left", fontsize=10)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
