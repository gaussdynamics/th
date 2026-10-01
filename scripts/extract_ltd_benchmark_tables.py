#!/usr/bin/env python3
"""Transcribe the LTD benchmark's appendix tables from the questions paper.

Spiryagin, Wu & Cole (2017), *VSD* 55(4):450-463. Appendix 1 (p. 462) is the
track gradient table and Appendix 2 (p. 463) the four trains' driving
controls. Both are set as text in the PDF, so they are read from word
positions rather than typed in by hand: each word is assigned to the column
whose header (``Location (m)``/``Gradient``, ``t(s)``/``N``) it sits under,
and rows are formed by line.

The paper prints minus signs as U+2212 and thousands separators as commas;
both are normalized. The tables are checked for shape before they are
written (strictly increasing locations and times, integer notches in
[-8, 8]), so a mis-assigned column fails here rather than in a run.

Output, consumed by :mod:`simulator2.ltd_benchmark`::

    simulator2/data/ltd_benchmark/gradient.csv          location_m, gradient_pct
    simulator2/data/ltd_benchmark/controls_train{k}.csv  t_s, notch

Run::

    python scripts/extract_ltd_benchmark_tables.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper/refs/papers/internatl_benchmarking_ltd_simulators.pdf"
OUT = ROOT / "simulator2/data/ltd_benchmark"

PAGE_GRADIENT = 13   # p. 462 of the journal
PAGE_CONTROLS = 14   # p. 463

#: Appendix 2 column pairs, left to right, and the train each belongs to.
#: Train 1 runs over four pairs, Train 2 over three, Trains 3 and 4 one each.
CONTROL_PAIRS = [1, 1, 1, 1, 2, 2, 2, 3, 4]

#: Words whose y-centres are within this many points share a printed line.
_LINE_TOL = 3.0


def _num(s: str) -> float:
    return float(s.replace("−", "-").replace(",", ""))


def _is_num(s: str) -> bool:
    try:
        _num(s)
        return True
    except ValueError:
        return False


def _columns(words, header: str) -> list[float]:
    """x-centres of every header word equal to ``header``, left to right."""
    return sorted(0.5 * (w[0] + w[2]) for w in words if w[4] == header)


def _rows(words, centres: list[float], y_min: float) -> list[list[float | None]]:
    """Numeric words below ``y_min``, one row per text line, each placed in
    the column whose centre is nearest."""
    cells = sorted(
        (0.5 * (y0 + y1), 0.5 * (x0 + x1), _num(text))
        for x0, y0, x1, y1, text, *_ in words
        if y0 > y_min and _is_num(text)
    )
    # Words on one printed line differ in y by a fraction of a point, so a
    # line is a run of words within _LINE_TOL of the run's first word.
    lines: list[list[tuple[float, float]]] = []
    y_line = None
    for yc, xc, val in cells:
        if y_line is None or yc - y_line > _LINE_TOL:
            lines.append([])
            y_line = yc
        lines[-1].append((xc, val))
    out = []
    for line in lines:
        row: list[float | None] = [None] * len(centres)
        for xc, val in line:
            k = min(range(len(centres)), key=lambda i: abs(centres[i] - xc))
            if row[k] is not None:
                raise ValueError(f"two values in column {k} on one line: {line}")
            row[k] = val
        out.append(row)
    return out


def _pairs(rows, n_pairs: int) -> list[list[tuple[float, float]]]:
    """Read each (key, value) column pair top to bottom, stopping at blanks."""
    cols = []
    for p in range(n_pairs):
        seq = []
        for r in rows:
            a, b = r[2 * p], r[2 * p + 1]
            if a is None and b is None:
                continue
            if a is None or b is None:
                raise ValueError(f"half-empty cell in pair {p}: {r}")
            seq.append((a, b))
        cols.append(seq)
    return cols


def gradient_table(doc) -> list[tuple[float, float]]:
    words = doc[PAGE_GRADIENT].get_text("words")
    loc = _columns(words, "Location")
    grad = _columns(words, "Gradient")
    if len(loc) != 4 or len(grad) != 4:
        raise ValueError(f"expected 4 column pairs, got {len(loc)}/{len(grad)}")
    centres = [c for pair in zip(loc, grad) for c in pair]
    y_hdr = max(w[3] for w in words if w[4] == "Gradient")
    table = [row for col in _pairs(_rows(words, centres, y_hdr), 4) for row in col]
    s = [r[0] for r in table]
    if any(b <= a for a, b in zip(s, s[1:])):
        raise ValueError("gradient locations are not strictly increasing")
    return table


def control_tables(doc) -> dict[int, list[tuple[float, int]]]:
    words = doc[PAGE_CONTROLS].get_text("words")
    ts = _columns(words, "t(s)")
    ns = _columns(words, "N")
    if len(ts) != len(CONTROL_PAIRS) or len(ns) != len(CONTROL_PAIRS):
        raise ValueError(f"expected {len(CONTROL_PAIRS)} column pairs, got {len(ts)}/{len(ns)}")
    centres = [c for pair in zip(ts, ns) for c in pair]
    y_hdr = max(w[3] for w in words if w[4] == "t(s)")
    cols = _pairs(_rows(words, centres, y_hdr), len(CONTROL_PAIRS))
    out: dict[int, list[tuple[float, int]]] = {}
    for train, col in zip(CONTROL_PAIRS, cols):
        out.setdefault(train, []).extend((t, int(n)) for t, n in col)
    for train, seq in out.items():
        t = [r[0] for r in seq]
        if any(b <= a for a, b in zip(t, t[1:])):
            raise ValueError(f"train {train}: times are not strictly increasing")
        if any(abs(n) > 8 or n != int(n) for _, n in seq):
            raise ValueError(f"train {train}: notch outside [-8, 8]")
    return out


def main() -> None:
    import pymupdf
    doc = pymupdf.open(PDF)
    OUT.mkdir(parents=True, exist_ok=True)

    grad = gradient_table(doc)
    with open(OUT / "gradient.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["location_m", "gradient_pct"])
        w.writerows((f"{s:.3f}", f"{g:g}") for s, g in grad)
    print(f"gradient: {len(grad)} rows, {grad[0][0]:g}-{grad[-1][0]:g} m")

    for train, seq in sorted(control_tables(doc).items()):
        with open(OUT / f"controls_train{train}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "notch"])
            w.writerows((f"{t:g}", n) for t, n in seq)
        print(f"train {train}: {len(seq)} rows, t = {seq[0][0]:g}-{seq[-1][0]:g} s")


if __name__ == "__main__":
    main()
