#!/usr/bin/env python3
"""Figure for Chapter 12: the air brake model against measured cylinder pressures.

Writes ``paper/figures/air_brake_calibration.pdf``: the four panels of
Spiryagin et al. (2017), Figs 5.67-5.68 (brake cylinder pressure at three
positions along a ~2000 m brake pipe, minimum and full-service applications,
exhausted at the head and at both ends), digitized by
``scripts/digitize_spiryagin_brake.py``, against
:func:`simulator2.air_brake.cylinder_level` with the default parameters. Prints
the rms error per curve and overall, as Chapter 12 quotes them.

    python paper/scripts/plot_air_brake.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from calibrate_air_brake import CASES, P_FULL, PIPE_M, SPIRYAGIN, wei_fig7  # noqa: E402
from simulator2.air_brake import AirBrakeParams, cylinder_level  # noqa: E402

FIG = REPO / "paper" / "figures"
REF = "#8a9096"
SHADES = ("#0b2e4f", "#2f6fa8", "#6fa3cf")
TITLES = {
    "5_67a": "Minimum application, head end",
    "5_67b": "Full service, head end",
    "5_68a": "Minimum application, both ends",
    "5_68b": "Full service, both ends",
}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def main() -> None:
    p = AirBrakeParams()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=True)
    all_err = []
    for ax, (name, level, t_end, curves, vent_tail) in zip(axes.flat, CASES):
        data = pd.read_csv(SPIRYAGIN / f"fig{name}.csv")
        t = np.arange(0.0, t_end + p.dt_s / 2, p.dt_s)
        cyl = cylinder_level(t, np.where(t > 0.0, level, 0.0), list(curves), p,
                             length_m=PIPE_M, vent_tail=vent_tail) * P_FULL
        for i, ((pos, label), shade) in enumerate(zip(curves.items(), SHADES)):
            c = data[data.curve == label]
            err = np.interp(c.t_s, t, cyl[:, i]) - c.p_kPa.to_numpy()
            all_err.append(err)
            print(f"  {name} {label:>10}: rms {np.sqrt(np.mean(err ** 2)):5.1f} kPa")
            where = label.replace("BC", "").replace("+", " and ")
            ax.plot(c.t_s, c.p_kPa, color=REF, lw=2.4, alpha=0.6)
            ax.plot(t, cyl[:, i], color=shade, lw=1.2, label=f"{where} m")
        ax.plot([], [], color=REF, lw=2.4, alpha=0.6, label="measured")
        ax.set_title(TITLES[name], fontsize=9)
        ax.set_xlim(0.0, t_end)
        ax.set_xlabel("Time after the brake command (s)")
        ax.legend(frameon=False, fontsize=8, loc="upper right" if level < 1.0 else "lower right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cylinder pressure (kPa)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "air_brake_calibration.pdf")
    print(f"overall rms {np.sqrt(np.mean(np.concatenate(all_err) ** 2)):.1f} kPa")
    print("Wei et al. (2017) Fig. 7, out of sample:",
          {k: round(v, 1) for k, v in wei_fig7(p).items()})
    print(f"wrote {FIG / 'air_brake_calibration.pdf'}")


if __name__ == "__main__":
    main()
