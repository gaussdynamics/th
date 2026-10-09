#!/usr/bin/env python3
"""Digitize the brake pipe and cylinder curves of Spiryagin et al. (2017).

Source: Spiryagin, Wolfs, Cole, Spiryagin, Sun & McSweeney, *Design and
Simulation of Heavy Haul Locomotives and Trains*, CRC Press, 2017, §5.6,
pp. 215-216. Captioned "operational data", 100 wagons, pipe length ~2000 m:

* Fig. 5.67: head-end train, one valve per wagon. BC and BP at 16, 996 and
  1996 m along the pipe. (a) minimum application, (b) full service.
* Fig. 5.68: head-tail train (pipe exhausted at both ends), one valve per
  tandem wagon pair. BC and BP at 32, 968 and 1943 m. (a) minimum, (b) full
  service.

The book does not name the valve type or the railway (Australian heavy haul
context). Unlike Wei et al. (2017) this is a second, non-Chinese check on
propagation speed and cylinder build-up.

Fig. 5.67 (a, b) and 5.68 (a) are vector graphics whose curves are *filled
outlines* of the stroked lines, so stroke width cannot be read from them.
Each curve is drawn as a left-to-right run of x chunks, in legend order
(first, middle, last position); fill separates BC (dark) from BP (grey). The
centre line is the midpoint of the outline in each x bin. The labelling was
checked against overlays of the rendered page.
Fig. 5.68 (b) is a raster image placed over vector axes; there the dark
pixels give the BC curves and BP is not read. The pipe is exhausted at both
ends, so the 32 m and 1943 m cylinders fill almost together and overlap (top
dark run); the 968 m cylinder at mid-train is the bottom run.

Writes ``simulator2/data/spiryagin2017_air_brake/fig5_6{7,8}{a,b}.csv`` and
prints onset times, propagation speeds and first-order build-up time
constants. ``scripts/calibrate_air_brake.py`` fits the air brake to the data. Run::

    python scripts/digitize_spiryagin_brake.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / ("paper/refs/papers/Maksym Spiryagin, Peter Wolfs, et al., Design and "
              "simulation of heavy haul locomotives and trains (2017) (1).pdf")
OUT = ROOT / "simulator2/data/spiryagin2017_air_brake"

DARK, GREY = 0.137, 0.506               # fill grey levels: BC curves, BP curves

# PDF page index, x axis (pos at 0 s, pos at t_max, t_max), y axis (pos at 0, pos at
# 600 kPa), plot box (x0, x1, y0, y1), positions along the pipe [m]
PANELS = {
    "5_67a": dict(page=234, x=(83.9, 335.05, 30.0), y=(218.6, 67.3),
                  box=(83.0, 336.0, 60.0, 220.0), pos=(16, 996, 1996)),
    "5_67b": dict(page=234, x=(83.9, 335.1, 90.0), y=(411.25, 260.0),
                  box=(83.0, 336.0, 255.0, 413.0), pos=(16, 996, 1996)),
    "5_68a": dict(page=235, x=(85.7, 326.7, 30.0), y=(212.35, 67.25),
                  box=(84.5, 328.0, 60.0, 214.0), pos=(32, 968, 1943)),
    "5_68b": dict(page=235, x=(85.2, 335.4, 90.0), y=(401.1, 250.35),
                  box=(85.2, 335.4, 250.0, 402.0), pos=(32, 968, 1943)),
}


def _ticks_present(page, cfg) -> None:
    """Axis ends must be where the config says (guards against a changed PDF)."""
    rects = [d["rect"] for d in page.get_drawings()]
    (x0, x1, _), (y0, y1) = cfg["x"], cfg["y"]
    for want, horiz in ((x0, True), (x1, True), (y0, False), (y1, False)):
        edges = [(r.x0, r.x1, (r.x0 + r.x1) / 2) if horiz else (r.y0, r.y1, (r.y0 + r.y1) / 2)
                 for r in rects]
        if not any(abs(e - want) < 0.3 for es in edges for e in es):
            raise RuntimeError(f"no axis mark at {want}: has the PDF changed?")


def _to_data(cfg, x, y):
    (xa, xb, tmax), (ya, yb) = cfg["x"], cfg["y"]
    return (np.asarray(x) - xa) * tmax / (xb - xa), (np.asarray(y) - ya) * 600.0 / (yb - ya)


def _centre_line(drawing, x0: float, x1: float, binw: float = 0.25) -> np.ndarray:
    """``(x, y)`` midpoint of a filled stroke outline per x bin, clipped to [x0, x1]."""
    pts = []
    for it in drawing["items"]:
        if it[0] == "re":
            pts += [(it[1].x0, it[1].y0), (it[1].x1, it[1].y1)]
        else:
            pts += [(q.x, q.y) for q in it[1:] if hasattr(q, "x")]
    pts = np.array(pts)
    pts = pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1)]
    b = np.round(pts[:, 0] / binw)
    return np.array([(pts[b == k, 0].mean(), (pts[b == k, 1].min() + pts[b == k, 1].max()) / 2)
                     for k in np.unique(b)])


def vector_panel(name: str) -> dict[str, np.ndarray]:
    """``{label: (t, p)}`` for the six curves of a vector panel.

    Each curve is drawn as a left-to-right run of x chunks, curves in legend
    order (first, middle, last position; BC then BP). A chunk drawn out of
    sequence (seen in Fig. 5.67a) is an orphan: it is attached to the run that
    starts where it ends, matched on y at the chunk boundary.
    """
    import itertools

    import pymupdf
    cfg = PANELS[name]
    page = pymupdf.open(PDF)[cfg["page"]]
    _ticks_present(page, cfg)
    x0, x1, y0, y1 = cfg["box"]
    out = {}
    for kind, grey in (("BC", DARK), ("BP", GREY)):
        chunks = []
        for d in page.get_drawings():
            f, r = d.get("fill"), d["rect"]
            if (f is None or abs(f[0] - grey) > 5e-3 or not x0 <= r.x0 < x1
                    or r.y0 < y0 or r.y1 > y1 or r.width > 0.98 * (x1 - x0)):
                continue                                 # other colour, legend, axes
            if r.width < 1.0:
                continue                                 # vertical axis line, ticks
            chunks.append((r.x0, _centre_line(d, x0, x1)))
        runs: list[list] = []
        for xs, c in chunks:
            if runs and xs > runs[-1][-1][0] + 1.0:
                runs[-1].append((xs, c))
            else:
                runs.append([(xs, c)])
        left = min(xs for xs, _ in chunks)
        full = [r for r in runs if r[0][0] < left + 2.0 and len(r) > 1]
        partial = [r for r in runs if r[0][0] >= left + 2.0]
        orphans = [r for r in runs if r[0][0] < left + 2.0 and len(r) == 1]
        if len(full) + len(partial) != 3 or len(orphans) != len(partial):
            raise RuntimeError(f"{name} {kind}: {len(full)} full, {len(partial)} partial runs, "
                               f"{len(orphans)} orphans; expected 3 curves")
        if partial:
            cost = lambda o, r: abs(o[0][1][-1, 1] - r[0][1][0, 1])   # noqa: E731
            best = min(itertools.permutations(orphans),
                       key=lambda perm: sum(cost(o, r) for o, r in zip(perm, partial)))
            for o, r in zip(best, partial):
                r.insert(0, o[0])
        curves = sorted(full + partial, key=lambda r: next(i for i, q in enumerate(runs) if q is r))
        for pos, run in zip(cfg["pos"], curves):
            pts = np.vstack([c for _, c in run])
            pts = pts[np.argsort(pts[:, 0], kind="stable")]
            t, p = _to_data(cfg, *pts.T)
            out[f"{kind}{pos}"] = np.c_[t, p][t >= 0.0]   # stroke width pokes left of 0 s
    return out


def raster_panel(name: str) -> dict[str, np.ndarray]:
    """Fig. 5.68 (b): BC at both ends (top dark run) and mid-train (bottom run)."""
    import pymupdf
    cfg = PANELS[name]
    page = pymupdf.open(PDF)[cfg["page"]]
    _ticks_present(page, cfg)
    (xref, *_), = page.get_images(full=True)
    bbox = page.get_image_bbox(page.get_images(full=True)[0])
    pix = pymupdf.Pixmap(page.parent, xref)
    if pix.n > 1:
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    sx, sy = bbox.width / pix.width, bbox.height / pix.height
    x0, x1, *_ = cfg["box"]
    dark = img < 90                                      # BC black; BP and grid are grey
    ends, mid = [], []
    for c in range(pix.width):
        x = bbox.x0 + (c + 0.5) * sx
        if not x0 + 0.6 < x < x1 - 0.3:
            continue
        rows = np.flatnonzero(dark[:, c])
        if rows.size == 0:
            continue
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        y = [bbox.y0 + (r.mean() + 0.5) * sy for r in (runs[0], runs[-1])]
        ends.append((x, y[0]))
        mid.append((x, y[1]))
    pos = cfg["pos"]
    out = {}
    for label, pts in ((f"BC{pos[0]}+{pos[2]}", ends), (f"BC{pos[1]}", mid)):
        a = np.array(pts)
        t, p = _to_data(cfg, a[:, 0], a[:, 1])
        out[label] = np.c_[t, p]
    return out


def onset(c: np.ndarray, level: float = 8.0, bp: bool = False) -> float:
    """Start of the BC rise: last time at or below ``level`` kPa before it first
    passes ``4 * level`` (BP: the same for the drop below 500 kPa)."""
    t, p = c.T
    rise = (500.0 - p) if bp else p
    up = np.flatnonzero(rise > 4 * level)
    if up.size == 0:
        return np.nan
    low = np.flatnonzero((rise <= level) & (t < t[up[0]]))
    return float(t[low[-1]]) if low.size else float(t[0])


def time_to(c: np.ndarray, frac: float, p_final: float, t_on: float) -> float:
    """Time from onset until BC first reaches ``frac`` of ``p_final``."""
    t, p = c.T
    m = (t > t_on) & (p >= frac * p_final)
    return float(t[m].min() - t_on) if m.any() else np.nan


def fit_tau(c: np.ndarray, t_on: float, p_final: float) -> tuple[float, float]:
    """First-order lag from 0 at ``t_on`` towards ``p_final``: (tau, rms kPa)."""
    t, p = c.T
    def sse(tau):
        return np.mean((p - np.where(t > t_on, p_final * (1 - np.exp(-(t - t_on) / tau)), 0.0)) ** 2)
    r = minimize_scalar(sse, bounds=(0.3, 500.0), method="bounded")
    return float(r.x), float(np.sqrt(r.fun))


def _pos(label: str) -> int:
    return int(label[2:].split("+")[0])


def save(name: str, curves: dict[str, np.ndarray]) -> None:
    lines = ["t_s,p_kPa,curve"] + [f"{t:.3f},{p:.3f},{label}"
                                   for label, c in curves.items() for t, p in c]
    (OUT / f"fig{name}.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Spiryagin et al. (2017) Figs 5.67-5.68, digitized from {PDF.name}\n")
    speeds = []
    for name in PANELS:
        curves = raster_panel(name) if name == "5_68b" else vector_panel(name)
        curves = dict(sorted(curves.items(), key=lambda kv: (kv[0][:2], _pos(kv[0]))))
        save(name, curves)
        full = name.endswith("b")
        p_final = 350.0 if full else 125.0
        print(f"Fig. {name.replace('_', '.')}  ({'full service' if full else 'minimum'}, "
              f"{'head-end' if name.startswith('5_67') else 'head-tail'})")
        ons = {}
        for label, c in curves.items():
            bp = label.startswith("BP")
            ons[label] = on = onset(c, bp=bp)
            line = f"  {label:>10}: onset {on:6.2f} s"
            if not bp:
                line += (f"  t50 {time_to(c, .5, p_final, on):5.1f} s"
                         f"  t90 {time_to(c, .9, p_final, on):5.1f} s"
                         f"  p(end) {c[-1, 1]:4.0f} kPa")
                if full:
                    tau, rms = fit_tau(c, on, p_final)
                    line += f"  tau {tau:5.1f} s (rms {rms:2.0f})"
            print(line)
        if name.startswith("5_67"):                      # head-end: wave from 16 m
            for kind in ("BC", "BP"):
                d = np.array([16, 996, 1996])
                t = np.array([ons[f"{kind}{x}"] for x in d])
                v = np.polyfit(t, d, 1)[0]
                speeds.append(v)
                print(f"  {kind} onset wave 16 -> 1996 m: {v:.0f} m/s")
        print()
    print(f"Head-end propagation {min(speeds):.0f}-{max(speeds):.0f} m/s, "
          f"mean {np.mean(speeds):.0f} m/s")


if __name__ == "__main__":
    main()
