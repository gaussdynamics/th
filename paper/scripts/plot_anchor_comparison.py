#!/usr/bin/env python3
"""Plot the step-anchor comparison, for fig:arch_anchor in Chapter 8.

Each bar is the size of the learning target a step model would face if its
prediction were anchored on the given physics estimate: the worst-vehicle
change over one 0.25 s transition that the network would still have to supply.
A smaller target is an easier problem.

Values are measured on data/v2 (480 transitions from 40 train scenarios, seed 0)
by scripts/measure_formulation.py on branch surrogate-gns, and recorded in
SURROGATE_FORMULATION_NOTE.md, sections "Measured: the anchor makes the problem
harder" and "But the anchor is valuable for position". They are copied here
rather than recomputed because this machine does not hold the corpus; rerun
that script and update the tables below after a corpus rebuild.

    python scripts/plot_anchor_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures/arch_anchor_comparison.pdf"

# max |dv| over the consist per transition, m/s: (median, p90, p99)
VELOCITY = {
    "No anchor\n(predict the change)": (0.0176, 0.0908, 0.2231),
    "Explicit Euler\nstep, $h = 0.25$ s": (0.1108, 0.2192, 0.4379),
    "RK4 step,\ncoupler force frozen": (0.1016, 0.1968, 0.3911),
    "RK4 step,\n$h = 0.25$ s": (1.1869, 7.5268, 27.3700),
}
# max |dx| over the consist per transition, m: (median, p90, p99)
POSITION = {
    "No anchor\n(predict the change)": (1.5818, 6.7955, 11.7254),
    "RK4 step,\ncoupler force frozen": (0.0140, 0.0367, 0.0550),
}

PCT = ("median", "90th percentile", "99th percentile")
SHADES = ("#1f4e79", "#5b8bb5", "#a9c4dd")
BEST = "#C8553D"


def panel(ax, data: dict, unit: str, title: str, best: str) -> None:
    names = list(data)
    x = np.arange(len(names))
    w = 0.26
    for i, (lab, col) in enumerate(zip(PCT, SHADES)):
        vals = [data[n][i] for n in names]
        ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel(f"Remaining target ({unit})")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="y", which="major", color="0.88", zorder=0)
    for t in ax.get_xticklabels():
        if t.get_text() == best:
            t.set_color(BEST)
            t.set_fontweight("bold")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.1),
                               gridspec_kw={"width_ratios": [2, 1]},
                               constrained_layout=True)
    panel(a, VELOCITY, "m/s", "(a) Velocity: every anchor makes the target larger",
          "No anchor\n(predict the change)")
    panel(b, POSITION, "m", "(b) Position: the anchor helps",
          "RK4 step,\ncoupler force frozen")
    a.legend(frameon=False, fontsize=9, loc="upper left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
