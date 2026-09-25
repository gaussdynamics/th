#!/usr/bin/env python3
"""Plot the three route fields the simulator consumes, for fig:route_grade_curvature.

Reads one exported route profile (.npz with route_s, route_sin_theta,
route_kappa, route_vmax) and draws grade, curvature and speed limit against
chainage, in the layout of the route application's profile view.

The thesis figure is built from the corrected corridor set on branch
surrogate-gns (800 m smoothing window, 2.5% grade clip):

    git show origin/surrogate-gns:route_generator/route_profiles_v3/route_line0_49km.npz \
        > /tmp/route_line0_49km.npz
    python scripts/plot_route_fields.py /tmp/route_line0_49km.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "thesis_imgs_7_15/route_generator/route_grade_curvature.png"
LINE = "#1f4e79"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", type=Path, help="exported route profile")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    d = np.load(args.npz)
    s_km = d["route_s"] / 1000.0
    grade_pct = 100.0 * d["route_sin_theta"]
    kappa = d["route_kappa"]
    vmax = d["route_vmax"]

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(10, 5.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(s_km, grade_pct, color=LINE, lw=1.0)
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_ylabel(r"Grade $\sin\theta(s)$ (%)")
    ax.set_xlabel("Chainage (km)")
    ax.set_xlim(s_km[0], s_km[-1])
    ax.grid(axis="y", color="0.9")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(s_km, kappa, color=LINE, lw=1.0)
    ax.set_ylabel(r"Curvature $\kappa(s)$ (1/m)")
    ax.set_xlabel("Chainage (km)")
    ax.set_xlim(s_km[0], s_km[-1])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.grid(axis="y", color="0.9")

    ax = fig.add_subplot(gs[1, 1])
    ax.step(s_km, vmax, where="post", color=LINE, lw=1.4)
    ax.set_ylabel(r"Speed Limit $v_{\max}(s)$ (m/s)")
    ax.set_xlabel("Chainage (km)")
    ax.set_xlim(s_km[0], s_km[-1])
    ax.set_ylim(0, 1.15 * float(vmax.max()))
    ax.grid(axis="y", color="0.9")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240)
    print(f"wrote {args.out}")
    print(f"grade rms {np.sqrt(np.mean(grade_pct**2)):.2f}%  "
          f"max |grade| {np.abs(grade_pct).max():.2f}%  "
          f"vmax {sorted(set(np.round(vmax, 2)))}")


if __name__ == "__main__":
    main()
