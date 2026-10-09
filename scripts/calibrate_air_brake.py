#!/usr/bin/env python3
"""Fit the air brake's pipe and valve parameters to Spiryagin et al. (2017).

Fits ``pipe_diffusivity_m2ps``, ``quick_service_frac``, ``inshot_frac``,
``inshot_rate_per_s`` and ``service_rate_per_s`` of
:class:`simulator2.air_brake.AirBrakeParams` to the brake cylinder curves of
Figs 5.67 and 5.68 (digitized by ``scripts/digitize_spiryagin_brake.py``):
a 100-wagon train, pipe ~2000 m, minimum and full-service applications, the
pipe exhausted at the head (5.67) or at both ends (5.68). The model is
:func:`simulator2.air_brake.cylinder_level` itself; the front speed is held
at the measured ``propagation_mps``.

Cylinder pressure is taken as proportional to the pipe reduction, with the
full service at 350 kPa and the minimum application at 125 kPa (the plateaus
of the head cylinders), so the minimum application is level 125/350.

Then predicts, without refitting, the measured large service reduction of
Wei et al. (2017) Fig. 7 (``simulator2/data/wei2017_air_brake/fig7_bc.csv``):
a different train (HXD1 + 116 C80, pipe ~1430 m with the unsourced 12 m car
length) and valve (Chinese 120), a 50 kPa reduction then the full 140 kPa at
55.3 s, full service 401.9 kPa. Its own front speed is used. Run::

    python scripts/calibrate_air_brake.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator2.air_brake import (  # noqa: E402
    C80_PITCH_M, WEI2017_APPLY_DELAY_S, AirBrakeParams, cylinder_level,
)

SPIRYAGIN = ROOT / "simulator2/data/spiryagin2017_air_brake"
WEI = ROOT / "simulator2/data/wei2017_air_brake/fig7_bc.csv"
FITTED = ("pipe_diffusivity_m2ps", "quick_service_frac", "inshot_frac",
          "inshot_rate_per_s", "service_rate_per_s")
P_FULL, P_MIN, PIPE_M = 350.0, 125.0, 2000.0

# panel, level, duration [s], {position [m]: curve label}, vented at the tail
CASES = [
    ("5_67a", P_MIN / P_FULL, 30.0, {16: "BC16", 996: "BC996", 1996: "BC1996"}, False),
    ("5_67b", 1.0, 90.0, {16: "BC16", 996: "BC996", 1996: "BC1996"}, False),
    ("5_68a", P_MIN / P_FULL, 30.0, {32: "BC32", 968: "BC968", 1943: "BC1943"}, True),
    ("5_68b", 1.0, 90.0, {32: "BC32+1943", 968: "BC968"}, True),
]


def _with(p: AirBrakeParams, theta) -> AirBrakeParams:
    return replace(p, **dict(zip(FITTED, map(float, theta))))


def residuals(p: AirBrakeParams, data: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Model minus measured cylinder pressure [kPa], per panel and curve."""
    out = {}
    for name, level, t_end, curves, vent_tail in CASES:
        t = np.arange(0.0, t_end + p.dt_s / 2, p.dt_s)
        cyl = cylinder_level(t, np.where(t > 0.0, level, 0.0), list(curves), p,
                             length_m=PIPE_M, vent_tail=vent_tail) * P_FULL
        d = data[name]
        for i, label in enumerate(curves.values()):
            c = d[d.curve == label]
            out[f"{name} {label}"] = np.interp(c.t_s, t, cyl[:, i]) - c.p_kPa.to_numpy()
    return out


def wei_fig7(p: AirBrakeParams) -> dict[str, float]:
    """rms model - measured [kPa] for the 4th and 116th car of Wei Fig. 7."""
    d = pd.read_csv(WEI)
    first, last = 3.5 * C80_PITCH_M + 35.0, 115.5 * C80_PITCH_M + 35.0  # 2-unit HXD1 ahead
    c = (last - first) / WEI2017_APPLY_DELAY_S[1]
    pw = replace(p, propagation_mps=c, release_propagation_mps=min(c, p.release_propagation_mps))
    t = np.arange(0.0, 200.0, p.dt_s)
    start = 3.33 - first / c                    # 4th car onset measured at 3.33 s
    level = np.where(t < start, 0.0, np.where(t < 55.3, 50.0 / 140.0, 1.0))
    cyl = cylinder_level(t, level, [first, last], pw, length_m=last + 6.0) * 401.9
    out = {}
    for i, col in enumerate(("bc_4th_kPa", "bc_116th_kPa")):
        m = d[col].notna()
        e = np.interp(d.t_s[m], t, cyl[:, i]) - d[col][m]
        out[col] = float(np.sqrt(np.mean(e ** 2)))
    return out


def main() -> None:
    data = {name: pd.read_csv(SPIRYAGIN / f"fig{name}.csv") for name, *_ in CASES}
    base = AirBrakeParams()

    def cost(theta) -> float:
        if np.any(np.asarray(theta) <= 0.0):
            return 1e9
        return float(np.mean(np.concatenate(list(residuals(_with(base, theta), data).values())) ** 2))

    x0 = [getattr(base, f) for f in FITTED]
    fit = minimize(cost, x0, method="Nelder-Mead",
                   options=dict(maxiter=2000, xatol=1e-5, fatol=1e-3))
    p = _with(base, fit.x)
    print(f"Fit to Spiryagin et al. (2017) Figs 5.67-5.68, front {base.propagation_mps:.0f} m/s:")
    for f, v0, v in zip(FITTED, x0, fit.x):
        print(f"  {f:>22}: {v:10.4g}   (current default {v0:.4g})")
    tau1 = 4 * PIPE_M ** 2 / (np.pi ** 2 * p.pipe_diffusivity_m2ps)
    print(f"  slowest pipe mode at {PIPE_M:.0f} m: {tau1:.1f} s; "
          f"inshot {p.inshot_frac * P_FULL:.0f} kPa, service rate "
          f"{p.service_rate_per_s * P_FULL:.1f} kPa/s on {P_FULL:.0f} kPa")
    res = residuals(p, data)
    print(f"  overall rms {np.sqrt(fit.fun):.1f} kPa")
    for k, e in res.items():
        print(f"    {k:>18}: rms {np.sqrt(np.mean(e ** 2)):5.1f} kPa, end error {e[-1]:+6.1f} kPa")
    print("Out of sample, Wei et al. (2017) Fig. 7 (no refit):")
    for k, v in wei_fig7(p).items():
        print(f"    {k:>18}: rms {v:5.1f} kPa")


if __name__ == "__main__":
    main()
