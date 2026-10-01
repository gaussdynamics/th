#!/usr/bin/env python3
"""Digitize the benchmark draft gear cross plot (Spiryagin et al. 2017, Fig. 3).

The figure is a raster (905 x 670) embedded in the PDF, so it is digitized by
colour: blue is the loading curve, red the unloading curve. The axes are
calibrated from the plot frame -- the longest dark row and column runs --
against the printed limits, 0-90 mm and 0-2500 kN.

Each curve is reduced to a centreline. Loading and the flat part of unloading
are single-valued in deflection, so they are traced column by column. The
unloading curve ends in a near-vertical drop at full travel, which is
single-valued in *force* instead, so above ``_RED_ROW_KN`` it is traced row
by row.

Output is written to ``simulator2/data/draft_gear_benchmark.csv``, columns
``deflection_mm, loading_kN, unloading_kN`` on a common grid. The pixel
traces are kept alongside for the thesis figure.

Run::

    python scripts/digitize_draft_gear.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper/refs/papers/internatl_benchmarking_ltd_simulators.pdf"
OUT = ROOT / "simulator2/data/draft_gear_benchmark.csv"
TRACE = ROOT / "simulator2/data/draft_gear_benchmark_trace.npz"

PAGE = 5                  # p. 454 of the journal
X_MAX_MM, Y_MAX_KN = 90.0, 2500.0
_RED_ROW_KN = 400.0       # above this, the unloading curve is the vertical drop


def load_figure() -> np.ndarray:
    import pymupdf
    doc = pymupdf.open(PDF)
    xref = doc[PAGE].get_images(full=True)[0][0]
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).astype(int)


def frame(img: np.ndarray) -> tuple[int, int, int, int]:
    """Pixel columns of x = 0 and x = 90, rows of y = 0 and y = 2500: the
    frame is the only set of long, fully dark lines in the figure."""
    dark = img.sum(2) < 200
    cols = np.where(dark.mean(0) > 0.6)[0]
    rows = np.where(dark.mean(1) > 0.6)[0]
    return cols.min(), cols.max(), rows.max(), rows.min()


def masks(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    blue = (b > 120) & (r < 110) & (g < 110) & (b - r > 60)
    red = (r > 170) & (g < 90) & (b < 90)
    return blue, red


def main() -> int:
    img = load_figure()
    c0, c1, r0, r1 = frame(img)
    to_mm = lambda c: (c - c0) / (c1 - c0) * X_MAX_MM          # noqa: E731
    to_kn = lambda r: (r0 - r) / (r0 - r1) * Y_MAX_KN          # noqa: E731
    from_kn = lambda f: r0 - f / Y_MAX_KN * (r0 - r1)          # noqa: E731
    print(f"frame: columns {c0}-{c1}, rows {r1}-{r0} "
          f"({(c1 - c0) / X_MAX_MM:.2f} px/mm, {(r0 - r1) / Y_MAX_KN * 1000:.2f} px/MN)")

    blue, red = masks(img)
    inside = np.zeros_like(blue)
    inside[r1 + 2:r0 - 1, c0 + 2:c1 - 1] = True
    blue &= inside
    red &= inside

    def by_column(m):
        xs, ys = [], []
        for c in range(m.shape[1]):
            rr = np.where(m[:, c])[0]
            if len(rr):
                xs.append(to_mm(c))
                ys.append(to_kn(np.median(rr)))
        return np.array(xs), np.array(ys)

    lx, ly = by_column(blue)

    # Unloading: columns up to where the drop starts, rows above it.
    red_flat = red.copy()
    red_flat[: int(from_kn(_RED_ROW_KN)), :] = False
    ux, uy = by_column(red_flat)
    # Keep the lower branch in columns the drop passes through.
    keep = uy < _RED_ROW_KN
    ux, uy = ux[keep], uy[keep]
    vx, vy = [], []
    for r in range(r1, int(from_kn(_RED_ROW_KN))):
        cc = np.where(red[r])[0]
        if len(cc):
            vx.append(to_mm(np.median(cc)))
            vy.append(to_kn(r))
    vx, vy = np.array(vx)[::-1], np.array(vy)[::-1]

    # Travel limit: where the curves meet. The red drop is drawn over the tip
    # of the blue loading curve, so the top of the drop is the true peak.
    top = int(np.argmax(vy))
    x_end, f_end = float(vx[top]), float(vy[top])
    if ly.max() < f_end:
        lx, ly = np.append(lx, x_end), np.append(ly, f_end)
    print(f"loading: {len(lx)} columns, peak {f_end:.0f} kN at {x_end:.2f} mm")
    print(f"unloading: {len(ux)} columns + {len(vx)} rows of vertical drop, "
          f"drop at {vx.min():.2f}-{vx.max():.2f} mm")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TRACE, load_mm=lx, load_kn=ly, unload_mm=ux, unload_kn=uy,
             drop_mm=vx, drop_kn=vy, x_end_mm=x_end, f_end_kn=f_end)

    # Common grid: 0.5 mm to the start of the drop, then the drop itself.
    x_drop = float(ux.max())
    grid = np.arange(0.0, x_drop, 0.5)
    grid = np.append(grid, [x_drop, x_end])
    order = np.argsort(lx)
    load = np.interp(grid, lx[order], ly[order])
    uo = np.argsort(ux)
    unload = np.interp(grid, ux[uo], uy[uo])
    # Both curves start at the origin; the pixel trace at 0 mm is the line
    # width, not a force.
    load[0] = unload[0] = 0.0
    unload[-1] = f_end
    unload = np.minimum(unload, load)

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["deflection_mm", "loading_kN", "unloading_kN"])
        for row in zip(grid, load, unload):
            w.writerow([f"{v:.3f}" for v in row])
    print(f"wrote {OUT.relative_to(ROOT)} ({len(grid)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
