#!/usr/bin/env python3
"""Digitize the brake cylinder curves of Wei et al. (2017) and fit the air brake.

Source: Wei, Hu, Wu, Zhao, Zhang & Zhang, "An air brake model for
longitudinal train dynamics studies", VSD 55(4):517-533. The test train is a
Shuo-Huang 10,000 t train, HXD1 + 116 C80 cars with 120 distribution valves;
pressures were measured at the 4th and the 116th car. Figures 5, 7 and 9 are
the *measured* results (6, 8 and 10 are the authors' simulation):

* Fig. 5: small service reduction, 50 kPa then 70 kPa, release at 243.7 s.
* Fig. 7: large service reduction, 50 kPa then 140 kPa (the line's maximum).
* Fig. 9: emergency on a 120-vehicle test rig, 1st and 120th vehicle.

The figures are vector graphics, so the curves are read from the PDF's line
segments, not from pixels. Axes are calibrated from two tick marks each,
whose page positions are checked to exist. Fig. 5 is in colour (4th BC red,
116th BC green); Figs 7 and 9 are monochrome, so there the head and tail
cylinders are separated as the upper and lower envelope in the window where
only brake cylinder curves fall, with the legend masked out.

Writes ``simulator2/data/wei2017_air_brake/fig{5,7,9}_bc.csv`` and prints the
quantities ``simulator2/air_brake.py`` is calibrated from. Run::

    python scripts/digitize_air_brake.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper/refs/papers/air_brake_model_ltd.pdf"
OUT = ROOT / "simulator2/data/wei2017_air_brake"

DARK = (0.137, 0.122, 0.125)
RED = (0.929, 0.133, 0.141)
GREEN = (0.059, 0.506, 0.251)

# page index, (x tick pos, s) x2, (y tick pos, kPa) x2, plot box (x0, x1, y0, y1)
FIGS = {
    5: dict(page=11, xt=((184.15, 0.0), (335.95, 450.0)),
            yt=((264.44, 0.0), (209.85, 200.0)), box=(181.0, 336.0, 207.5, 267.0)),
    7: dict(page=13, xt=((179.32, 0.0), (337.75, 800.0)),
            yt=((180.56, 0.0), (55.46, 600.0)), box=(179.5, 337.5, 55.5, 180.4)),
    9: dict(page=14, xt=((179.53, 0.0), (343.67, 25.0)),
            yt=((541.76, 0.0), (418.29, 500.0)), box=(179.7, 343.5, 418.3, 541.6)),
}


def _close(c, ref) -> bool:
    return c is not None and all(abs(a - b) < 2e-3 for a, b in zip(c, ref))


def segments(page) -> list[tuple[tuple, float, float, float, float]]:
    return [(dr.get("color"), it[1].x, it[1].y, it[2].x, it[2].y)
            for dr in page.get_drawings() for it in dr["items"] if it[0] == "l"]


def _check_tick(segs, pos: float, horizontal_axis: bool) -> None:
    """A tick mark must sit at ``pos`` (guards against a wrong page or figure)."""
    for _, x0, y0, x1, y1 in segs:
        if horizontal_axis and abs(x0 - x1) < .01 and abs(x0 - pos) < .02 and 0.3 < abs(y0 - y1) < 6:
            return
        if not horizontal_axis and abs(y0 - y1) < .01 and abs(y0 - pos) < .02 and 0.3 < abs(x0 - x1) < 6:
            return
    raise RuntimeError(f"no tick at {pos}: has the PDF changed?")


def curve_points(fig: int, colour) -> np.ndarray:
    """``(t [s], p [kPa])`` of every segment end point of ``colour`` in the plot."""
    import pymupdf
    cfg = FIGS[fig]
    segs = segments(pymupdf.open(PDF)[cfg["page"]])
    for pos, _ in cfg["xt"]:
        _check_tick(segs, pos, True)
    for pos, _ in cfg["yt"]:
        _check_tick(segs, pos, False)
    (xa, ta), (xb, tb) = cfg["xt"]
    (ya, pa), (yb, pb) = cfg["yt"]
    x0, x1, y0, y1 = cfg["box"]
    pts = [(x, y) for c, *xy in segs if _close(c, colour)
           for x, y in (xy[:2], xy[2:]) if x0 < x < x1 and y0 < y < y1]
    pts = np.unique(np.array(pts), axis=0)
    t = ta + (pts[:, 0] - xa) * (tb - ta) / (xb - xa)
    p = pa + (pts[:, 1] - ya) * (pb - pa) / (yb - ya)
    o = np.argsort(t)
    return np.c_[t[o], p[o]]


def envelope(pts: np.ndarray, grid: np.ndarray, lo: float, hi: float,
             upper: bool, mask=None) -> np.ndarray:
    """Max (upper) or min of the points with ``lo < p < hi`` in each grid bin."""
    t, p = pts[:, 0], pts[:, 1]
    keep = (p > lo) & (p < hi)
    if mask is not None:
        keep &= ~mask(t, p)
    dt = grid[1] - grid[0]
    out = np.full(grid.shape, np.nan)
    for k, g in enumerate(grid):
        m = keep & (t >= g) & (t < g + dt)
        if m.any():
            out[k] = p[m].max() if upper else p[m].min()
    return out


def onset(t: np.ndarray, p: np.ndarray, after: float, level: float = 12.0,
          below: float = 100.0) -> float:
    """First time after ``after`` that a curve sits in ``(level, below)``."""
    m = (t > after) & (p > level) & (p < below)
    return float(t[m].min())


def fit_tau(t: np.ndarray, p: np.ndarray, t0: float, p0: float, p1: float) -> tuple[float, float]:
    """First-order lag from ``p0`` towards ``p1`` starting at ``t0``: (tau, rms)."""
    k = ~np.isnan(p)
    t, p = t[k], p[k]

    def model(tau):
        return np.where(t > t0, p1 - (p1 - p0) * np.exp(-(t - t0) / tau), p0)

    r = minimize_scalar(lambda tau: np.mean((p - model(tau)) ** 2),
                        bounds=(0.3, 200.0), method="bounded")
    return float(r.x), float(np.sqrt(r.fun))


def fit_release(t: np.ndarray, p: np.ndarray, t_on: float) -> float:
    """Exhaust time constant from release onset down to piston retraction (~40 kPa)."""
    m = (t > t_on) & (t < t_on + 12.0) & (p > 40.0) & (p < 175.0)
    slope, _ = np.polyfit(t[m] - t_on, np.log(p[m]), 1)
    return -1.0 / slope


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Wei et al. (2017), digitized from {PDF.name}\n")

    # -- Fig. 5: small service reduction (colour) ----------------------------
    head, tail = curve_points(5, RED), curve_points(5, GREEN)
    np.savetxt(OUT / "fig5_bc.csv", np.r_[np.c_[head, np.full(len(head), 4)],
                                           np.c_[tail, np.full(len(tail), 116)]],
               fmt="%.3f", delimiter=",", header="t_s,bc_kPa,car", comments="")
    on5 = onset(*head.T, -1.0), onset(*tail.T, -1.0)
    rel5 = (onset(head[:, 0], -head[:, 1], 240.0, -168.0, 0.0),  # first drop below plateau
            onset(tail[:, 0], -tail[:, 1], 240.0, -134.0, 0.0))
    tau_rel = fit_release(*head.T, rel5[0]), fit_release(*tail.T, rel5[1])
    print("Fig. 5 (service, small reduction)")
    print(f"  BC onset   4th {on5[0]:6.2f} s  116th {on5[1]:6.2f} s  -> {on5[1] - on5[0]:.2f} s")
    print(f"  release    4th {rel5[0]:6.2f} s  116th {rel5[1]:6.2f} s  -> {rel5[1] - rel5[0]:.2f} s"
          "  (handle to release at 243.7 s)")
    print(f"  exhaust tau 4th {tau_rel[0]:.2f} s  116th {tau_rel[1]:.2f} s")

    # -- Fig. 7: large service reduction (monochrome) ------------------------
    pts = curve_points(7, DARK)
    grid = np.arange(0.0, 200.0, 1.0)
    legend7 = lambda t, p: t > 200.0                                   # noqa: E731
    up = envelope(pts, grid, 60.0, 420.0, True, legend7)
    lo = envelope(pts, grid, 60.0, 420.0, False, legend7)
    up[grid < 40], lo[grid < 40] = np.nan, np.nan                      # first step: below 60
    np.savetxt(OUT / "fig7_bc.csv", np.c_[grid, up, lo], fmt="%.3f", delimiter=",",
               header="t_s,bc_4th_kPa,bc_116th_kPa", comments="")
    t, p = pts.T
    on7 = onset(t, p, -1.0), onset(t, p, 8.0, below=50.0)
    step7 = float(t[(t > 40) & (t < 60) & (p > 130) & (p < 420)].min())
    tau7 = (fit_tau(grid, up, step7, 110.0, 401.9),                     # 401.9 kPa: text
            fit_tau(grid, lo, step7 + on7[1] - on7[0], 87.0, 392.0))
    print("Fig. 7 (service, large reduction)")
    print(f"  BC onset   4th {on7[0]:6.2f} s  116th {on7[1]:6.2f} s  -> {on7[1] - on7[0]:.2f} s")
    print(f"  2nd step at {step7:.1f} s; build-up tau 4th {tau7[0][0]:.1f} s "
          f"(rms {tau7[0][1]:.0f} kPa), 116th {tau7[1][0]:.1f} s (rms {tau7[1][1]:.0f} kPa)")

    # -- Fig. 9: emergency, 120-vehicle rig (monochrome) ---------------------
    pts = curve_points(9, DARK)
    grid = np.arange(0.0, 25.0, 0.25)
    legend9 = lambda t, p: (t > 14.3) & (t < 17.3) & (p < 240.0)       # noqa: E731
    first = envelope(pts, grid, 60.0, 368.0, True, legend9)
    first[grid < 1.5] = np.nan                                         # pipe crosses here
    last = envelope(pts, grid, 60.0, 368.0, False, legend9)
    last[grid < 10.0] = np.nan                                         # 1st BC / 120th pipe
    np.savetxt(OUT / "fig9_bc.csv", np.c_[grid, first, last], fmt="%.3f", delimiter=",",
               header="t_s,bc_1st_kPa,bc_120th_kPa", comments="")
    t, p = pts.T
    on9 = onset(t, p, 0.0, 5.0, 50.0), onset(t, p, 5.0, 5.0, 50.0)
    tau9 = fit_tau(grid, first, on9[0], 0.0, 365.0), fit_tau(grid, last, on9[1], 0.0, 365.0)
    print("Fig. 9 (emergency, rig)")
    print(f"  BC onset   1st {on9[0]:6.2f} s  120th {on9[1]:6.2f} s  -> {on9[1] - on9[0]:.2f} s"
          "  (text: 7.7 s)")
    print(f"  build-up tau 1st {tau9[0][0]:.1f} s (rms {tau9[0][1]:.0f} kPa), "
          f"120th {tau9[1][0]:.1f} s (rms {tau9[1][1]:.0f} kPa)")


if __name__ == "__main__":
    main()
