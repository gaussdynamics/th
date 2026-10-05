#!/usr/bin/env python3
"""Compare a benchmark run with the nine simulators of Wu et al. (2018).

Reads the ``.npz`` files written by ``scripts/run_ltd_benchmark.py`` and the
results paper's Tables 3-6 (``simulator2/data/ltd_benchmark/wu2018_tables.csv``),
computes the same quantities, and prints where ours falls against the
participants. ``--json`` also writes the comparison and down-sampled traces.

Quantities, as the paper defines them (section 3):

* max and average lead-locomotive speed over the 3-50 km section;
* largest tensile / compressive force over all connections, and its position;
* mean over all connections of each connection's maximum force;
* maximum deflection at one selected position (10, 61, 147, 146).

The paper does not define its deflection. Several participants report more
than one gear's 82.5 mm travel (TABLDSS 146 mm at Train 3), so it is read as
the whole connection: both gears plus the slack. One gear's travel is also
reported.

The "core" group is the six simulators the paper finds in general agreement
on in-train forces (section 8): TABLDSS, UM, CRE-LTS, TDEAS, TsDyn, CARS.
PoliTo and BODYSIM had draft-gear instabilities and VOCO ran Train 1 only.

Run::

    python scripts/compare_ltd_benchmark.py --tag _s3k
    python scripts/compare_ltd_benchmark.py --tag _s3k --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator2 import ltd_benchmark as lb  # noqa: E402

RUNS = ROOT / "data" / "ltd_benchmark"
REF = lb.DATA / "wu2018_tables.csv"
FIG2 = lb.DATA / "wu2018_fig2_speed.csv"

CORE = ("TABLDSS", "UM", "CRE-LTS", "TDEAS", "TsDyn", "CARS")
SELECTED = {1: 10, 2: 61, 3: 147, 4: 146}
SECTION_M = (3000.0, 50000.0)

#: (key, label, unit); forces and deflections are magnitudes.
METRICS = [
    ("max_speed_kmh", "Max speed, 3-50 km", "km/h"),
    ("avg_speed_kmh", "Average speed, 3-50 km", "km/h"),
    ("largest_tension_kn", "Largest tension", "kN"),
    ("largest_comp_kn", "Largest compression", "kN"),
    ("mean_max_tension_kn", "Mean of max tension", "kN"),
    ("mean_max_comp_kn", "Mean of max compression", "kN"),
    ("max_defl_tension_mm", "Max deflection, tension", "mm"),
    ("max_defl_comp_mm", "Max deflection, compression", "mm"),
]


def load_reference() -> dict[int, dict[str, dict[str, float]]]:
    """``ref[train][simulator][metric]``, forces and deflections as magnitudes."""
    ref: dict[int, dict[str, dict[str, float]]] = {}
    with open(REF, encoding="utf-8") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for r in rows:
            d = {k: abs(float(v)) for k, v in r.items() if k not in ("train", "simulator")}
            ref.setdefault(int(r["train"]), {})[r["simulator"]] = d
    return ref


def ours(train: int, o: dict) -> dict[str, float]:
    s, v = o["lead_front_m"], o["lead_speed_mps"] * 3.6
    sec = (s >= SECTION_M[0]) & (s <= SECTION_M[1])
    c = lb.build_consist(train)
    pos = SELECTED[train]
    col = list(o["positions"]).index(pos)
    slack = 0.0 if c.drawbar[pos - 1] else 2 * lb.COUPLER_SLACK_HALF_M
    gear = o["gear_defl_m"][:, col].astype(float)          # one gear, draft positive
    whole = 2 * np.abs(gear) + np.where(gear != 0, slack, 0.0)
    tmax, cmax = o["max_tension_n"], -o["max_compression_n"]
    return {
        "max_speed_kmh": float(v[sec].max()),
        "avg_speed_kmh": float(v[sec].mean()),
        "largest_tension_kn": float(tmax.max() / 1e3),
        "largest_tension_pos": int(np.argmax(tmax)) + 1,
        "largest_comp_kn": float(cmax.max() / 1e3),
        "largest_comp_pos": int(np.argmax(cmax)) + 1,
        "mean_max_tension_kn": float(np.clip(tmax, 0, None).mean() / 1e3),
        "mean_max_comp_kn": float(np.clip(cmax, 0, None).mean() / 1e3),
        "max_defl_tension_mm": float(whole[gear > 0].max(initial=0.0) * 1e3),
        "max_defl_comp_mm": float(whole[gear < 0].max(initial=0.0) * 1e3),
        "max_gear_tension_mm": float(gear.max() * 1e3),
        "max_gear_comp_mm": float(-gear.min() * 1e3),
        "reach_50km_s": float(o["t_s"][np.argmax(s >= SECTION_M[1])]) if s[-1] >= SECTION_M[1] else None,
    }


def verdict(x: float, vals: list[float]) -> tuple[str, float]:
    """Where ``x`` sits against ``vals``: inside the range, or the excess as a
    fraction of the range's nearer end."""
    lo, hi = min(vals), max(vals)
    if lo <= x <= hi:
        return "inside", 0.0
    edge = lo if x < lo else hi
    return ("below" if x < lo else "above"), (x - edge) / edge


def traces(o: dict, train: int) -> dict:
    s, v = o["lead_front_m"], o["lead_speed_mps"] * 3.6
    grid = np.round(np.arange(3.0, 50.01, 0.1), 2)
    keep = np.r_[True, np.diff(s) > 0]                      # monotone for interp
    speed = np.interp(grid * 1e3, s[keep], v[keep], left=np.nan, right=np.nan)
    every = 100                                             # 1 s
    pos = SELECTED[train]
    col = list(o["positions"]).index(pos)
    return {
        "location_km": grid.tolist(),
        "speed_kmh": np.round(speed, 2).tolist(),
        "t_s": o["t_s"][::every].tolist(),
        "lead_km": np.round(s[::every] / 1e3, 3).tolist(),
        "selected_force_kn": np.round(o["force_n"][::every, col] / 1e3, 1).tolist(),
        "max_tension_kn": np.round(o["max_tension_n"] / 1e3, 1).tolist(),
        "max_comp_kn": np.round(-o["max_compression_n"] / 1e3, 1).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="")
    ap.add_argument("--trains", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--json", type=Path, help="write the comparison and traces here")
    args = ap.parse_args()

    ref = load_reference()
    report: dict = {"tag": args.tag, "trains": {}}
    for train in args.trains:
        path = RUNS / f"train{train}{args.tag}.npz"
        o = dict(np.load(path))
        m = ours(train, o)
        rows = []
        print(f"\nTrain {train}  ({path.name}, start {float(o.get('start_m', 0.0)):g} m)")
        print(f"  {'':30s} {'ours':>9s}  {'all sims':>17s}  {'core 6':>17s}  vs core")
        for key, label, unit in METRICS:
            allv = [d[key] for d in ref[train].values()]
            core = [ref[train][k][key] for k in CORE]
            where, by = verdict(m[key], core)
            tag = "inside" if where == "inside" else f"{where} by {100 * abs(by):.0f} %"
            print(f"  {label + ' [' + unit + ']':30s} {m[key]:9.1f}  "
                  f"{min(allv):7.1f}-{max(allv):7.1f}  {min(core):7.1f}-{max(core):7.1f}  {tag}")
            rows.append({"key": key, "label": label, "unit": unit, "ours": m[key],
                         "all": {k: d[key] for k, d in ref[train].items()},
                         "where": where, "by": by})
        pos_t = [int(d["largest_tension_pos"]) for d in ref[train].values()]
        pos_c = [int(d["largest_comp_pos"]) for d in ref[train].values()]
        print(f"  largest tension at position {m['largest_tension_pos']} (paper: {sorted(set(pos_t))}); "
              f"compression at {m['largest_comp_pos']} (paper: {sorted(set(pos_c))})")
        print(f"  one-gear travel at position {SELECTED[train]}: {m['max_gear_tension_mm']:.1f} mm draft, "
              f"{m['max_gear_comp_mm']:.1f} mm buff")
        report["trains"][train] = {"ours": m, "rows": rows, "traces": traces(o, train),
                                   "selected_position": SELECTED[train],
                                   "ref_positions": {"tension": pos_t, "comp": pos_c}}

    if args.json:
        lines = [ln for ln in FIG2.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
        names = lines[0].split(",")
        cols = np.genfromtxt(lines[1:], delimiter=",").T
        report["fig2"] = {n: [None if np.isnan(x) else float(x) for x in c] for n, c in zip(names, cols)}
        args.json.write_text(json.dumps(report), encoding="utf-8")
        print(f"\nwritten to {args.json}")


if __name__ == "__main__":
    main()
