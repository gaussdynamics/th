#!/usr/bin/env python3
"""Assign elevated corridors to splits, from measured grade.

`select_split_corridors.py` chose what to measure; this chooses what each
corridor is for. Nothing here reads a corridor's name, owner or state -- a
corridor is steep because its grade field says so.

The split design, and what each one has to be able to claim:

``ood_grade``     grade beyond what training saw. Taken as the steepest
                  corridors by rms grade, and the assignment is only honest if
                  a real gap opens between them and the training pool, so the
                  gap is reported and a thin one is called out.
``ood_corridor``  unseen corridors, otherwise ordinary. Drawn to match the
                  training pool's terrain mix rather than taken from one end
                  of it -- an ood_corridor split made of the flattest leftovers
                  would test nothing. Drawn by evenly spaced rank in rms grade.
``control_eval``  reserved for the MPC chapter and never trained on. Same
                  representative draw as ood_corridor.
the rest          train / val / test_id, partitioned per *scenario* by
                  build_dataset.py, not here.

Precedence in build_dataset.assign_split checks corridor reservations before
the consist-level ood_size flag, so a corridor listed here cannot leak into
training through a size-generalization scenario.

Run::

    python scripts/assign_split_corridors.py --routes data/narn_routes_splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rms_grade_pct(npz_path: Path) -> float:
    d = np.load(npz_path)
    key = "route_sin_theta" if "route_sin_theta" in d else "sin_theta"
    # The stored field is sin(arctan(grade)), so inverting it is tan(arcsin(.)),
    # not arctan(.). The two agree to O(g^3) and would have been invisible here,
    # which is exactly why it is worth writing correctly.
    st = np.clip(np.asarray(d[key], dtype=float), -1.0, 1.0)
    g = np.abs(np.tan(np.arcsin(st))) * 100.0
    return float(np.sqrt((g ** 2).mean()))


def evenly_spaced(items: list, n: int) -> list:
    if n >= len(items):
        return list(items)
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routes", type=Path, default=Path("data/narn_routes_splits"))
    ap.add_argument("--out", type=Path, default=Path("data/split_corridors.json"))
    ap.add_argument("--n-ood-grade", type=int, default=20)
    ap.add_argument("--n-ood-corridor", type=int, default=30)
    ap.add_argument("--n-control-eval", type=int, default=20)
    ap.add_argument("--gap-pp", type=float, default=0.25,
                    help="rms-grade gap (percentage points) required between "
                         "ood_grade and the training pool")
    args = ap.parse_args()

    files = sorted((args.routes / "routes").glob("*.npz"))
    if not files:
        raise SystemExit(f"no built routes under {args.routes}/routes")
    grades = {f.stem: rms_grade_pct(f) for f in files}
    ranked = sorted(grades, key=lambda c: grades[c], reverse=True)
    print(f"{len(ranked)} corridors built")
    print(f"  rms grade  min {grades[ranked[-1]]:.2f}%  median "
          f"{np.median(list(grades.values())):.2f}%  max {grades[ranked[0]]:.2f}%")

    ood_grade = ranked[:args.n_ood_grade]

    # Enforce the gap rather than reporting it. Taking the top N of a continuous
    # distribution makes a cut point, not a held-out grade regime: the first run
    # gave ood_grade 0.88-1.88% against a training pool topping out at 0.86%, a
    # +0.02 pp "separation" that would not support any claim made about it.
    # Corridors within `gap_pp` below the ood_grade floor are excluded from every
    # split. They could go to ood_corridor or control_eval instead, but those
    # would then be systematically steeper than training, which confounds
    # "unseen corridor" with "steeper corridor". Spent elevation is cheap; a
    # confounded split is not.
    floor = min(grades[c] for c in ood_grade)
    rest = ranked[args.n_ood_grade:]
    buffer_band = [c for c in rest if grades[c] > floor - args.gap_pp]
    remaining = [c for c in rest if grades[c] <= floor - args.gap_pp]

    # ood_corridor and control_eval are drawn across the remaining grade range,
    # interleaved so neither gets a systematically steeper half.
    picks = evenly_spaced(remaining, args.n_ood_corridor + args.n_control_eval)
    ood_corridor = picks[0::2][:args.n_ood_corridor]
    control_eval = picks[1::2][:args.n_control_eval]
    # The buffer band is excluded here too -- leaving it in the training pool
    # would put the gap back exactly where it was meant to be opened.
    reserved = set(ood_grade) | set(ood_corridor) | set(control_eval) | set(buffer_band)
    train_pool = [c for c in ranked if c not in reserved]

    g_train = [grades[c] for c in train_pool]
    g_ood = [grades[c] for c in ood_grade]
    gap = min(g_ood) - max(g_train)
    print(f"\n  train pool     n={len(train_pool):3d}  rms "
          f"{min(g_train):.2f}-{max(g_train):.2f}%  median {np.median(g_train):.2f}%")
    print(f"  ood_grade      n={len(ood_grade):3d}  rms "
          f"{min(g_ood):.2f}-{max(g_ood):.2f}%  median {np.median(g_ood):.2f}%")
    print(f"  ood_corridor   n={len(ood_corridor):3d}  rms median "
          f"{np.median([grades[c] for c in ood_corridor]):.2f}%")
    print(f"  control_eval   n={len(control_eval):3d}  rms median "
          f"{np.median([grades[c] for c in control_eval]):.2f}%")

    print()
    print(f"  buffer band    n={len(buffer_band):3d}  excluded to hold the gap open")
    print(f"  ood_grade separation from the training pool: {gap:+.2f} pp "
          f"(required {args.gap_pp:.2f})")
    if gap < args.gap_pp:
        print("  WARNING: the gap could not be met even after excluding the")
        print("  buffer band -- the pool does not contain enough steep corridors.")
        print("  Widen the mountain stratum rather than lowering --gap-pp.")

    out = {
        "source_routes": str(args.routes),
        "n_corridors": len(ranked),
        "rms_grade_pct": {c: round(grades[c], 4) for c in ranked},
        "ood_grade_corridors": sorted(ood_grade),
        "ood_corridors": sorted(ood_corridor),
        "control_eval_corridors": sorted(control_eval),
        "train_pool": sorted(train_pool),
        "buffer_band_excluded": sorted(buffer_band),
        "gap_pp_required": args.gap_pp,
        "ood_grade_separation_pp": round(gap, 4),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
