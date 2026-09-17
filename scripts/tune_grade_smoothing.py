#!/usr/bin/env python3
"""Phase A: choose the elevation smoothing window against a stated criterion.

The grade field is the dominant force on a freight train, and it is derived by
fitting a Savitzky-Golay curve through DEM samples and taking its slope. The
window length of that fit is a real judgement call with two opposite failure
modes:

too narrow
    DEM noise survives into the grade. 3DEP is accurate to ~1-2 m vertically
    and samples sit 10 m apart, so a 1 m error between neighbours is a 10%
    grade by itself. This is the current bug: ``route_line2`` sits at the 4%
    clip for 22.7% of its length with an rms grade of 2.46%, against ~0.9% for
    every other corridor.

too wide
    Real terrain is erased. At 1600 m every corridor flattens below 0.82%,
    which would be a corpus with no meaningful grades in it.

So the window is chosen by criteria, not by eye:

1. **The clip must stop binding.** More than ``CLIP_TOL`` of a corridor at the
   ceiling means the field is still noise-dominated. Cheap, automatic, and it
   catches the current bug by construction.
2. **The surviving grade must be physically plausible.** Freight ruling grades
   are an engineering constraint, not a free parameter: peaks belong in roughly
   the 1-2.5% band, and anything above that should be justified rather than
   silently clipped.

Run::

    python scripts/tune_grade_smoothing.py
    python scripts/tune_grade_smoothing.py --windows 400 600 800 1200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROUTEGEN = _REPO_ROOT / "route_generator"
for p in (_REPO_ROOT, _ROUTEGEN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from routegen.profile import derive_profile  # noqa: E402

#: Share of a corridor allowed to sit at the clip before the field is treated
#: as noise-dominated rather than steep.
CLIP_TOL = 0.01

#: Ruling grades on North American Class I freight lines are generally held at
#: or below this. A corridor whose p95 exceeds it is either genuinely
#: exceptional track or still noisy, and either way wants a human look.
PLAUSIBLE_PEAK_PCT = 2.5

#: Share of 10 m samples allowed to change elevation by more than a metre.
#: Real track cannot exceed this by much: a metre over ten is a 10% grade.
#: Healthy corridors sit at 0.7-1.5%; a corridor whose centreline has drifted
#: off the alignment, or whose DEM does not resolve its cut or fill, samples
#: the ground *beside* the track and lands far above it. No smoothing window
#: repairs that -- it only averages wrong numbers -- so this is a QA gate
#: applied before the window is chosen, not a parameter to tune around.
RAW_JUMP_TOL = 0.05

ELEVATED = _ROUTEGEN / "elevated_routes"


def elevation_qa(feature: dict) -> dict:
    """Is the raw elevation usable at all, before any smoothing is applied?"""
    z = np.array([(c[2] if len(c) > 2 and c[2] is not None else np.nan)
                  for c in feature["geometry"]["coordinates"]], dtype=float)
    z = z[np.isfinite(z)]
    dz = np.abs(np.diff(z))
    frac = float((dz > 1.0).mean())
    return {
        "median_step_m": float(np.median(dz)),
        "frac_over_1m": frac,
        "roughness_m": float(np.median(np.abs(np.diff(z, 2)))),
        "pass": frac <= RAW_JUMP_TOL,
    }


def summarize(prof: dict, clip: float) -> dict:
    g = np.abs(np.asarray(prof["grade"], dtype=float)) * 100.0
    at_clip = float((g >= clip * 100.0 - 1e-9).mean())
    return {
        "rms_pct": float(np.sqrt((g ** 2).mean())),
        "p95_pct": float(np.percentile(g, 95)),
        "max_pct": float(g.max()),
        "at_clip": at_clip,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=float, nargs="+",
                    default=[200, 400, 600, 800, 1200, 1600])
    ap.add_argument("--clip", type=float, default=0.04,
                    help="grade_clip in use while measuring (default: today's 4%%)")
    ap.add_argument("--routes", type=Path, default=ELEVATED)
    args = ap.parse_args()

    files = sorted(args.routes.glob("*.geojson"))
    if not files:
        raise SystemExit(f"no elevated routes under {args.routes}")

    print(f"{len(files)} corridors, clip {args.clip*100:.1f}%, "
          f"criteria: <{CLIP_TOL:.0%} at clip and p95 <= {PLAUSIBLE_PEAK_PCT}%\n")

    # -- gate 0: is the raw elevation usable at all? ----------------------
    print("  raw elevation QA (before any smoothing)")
    print(f"    {'corridor':22s} {'median step':>12s} {'>1 m':>8s} "
          f"{'roughness':>10s}  verdict")
    feats, usable = {}, []
    for f in files:
        feats[f.stem] = json.loads(f.read_text())
        q = elevation_qa(feats[f.stem])
        print(f"    {f.stem:22s} {q['median_step_m']:11.3f}m {q['frac_over_1m']:8.1%} "
              f"{q['roughness_m']:9.3f}m  {'ok' if q['pass'] else 'REJECT'}")
        if q["pass"]:
            usable.append(f.stem)
    rejected = [f.stem for f in files if f.stem not in usable]
    if rejected:
        print("")
        print(f"    {len(rejected)} rejected: {', '.join(rejected)}")
        print("    The window below is chosen over the usable corridors only:")
        print("    smoothing a rejected corridor into range hides bad data.")
    print("")

    table: dict[float, dict[str, dict]] = {}
    for w in args.windows:
        table[w] = {}
        for name in usable:
            prof = derive_profile(feats[name], smooth_window_m=w,
                                  grade_clip=args.clip)
            table[w][name] = summarize(prof, args.clip)

    names = usable
    for metric, label, fmt in (("rms_pct", "rms grade [%]", "{:8.2f}"),
                               ("p95_pct", "p95 grade [%]", "{:8.2f}"),
                               ("at_clip", "at the clip", "{:7.1%} ")):
        print(f"  {label}")
        print("    " + "window".ljust(9) + "".join(n.replace("route_", "")[:9].rjust(10)
                                                   for n in names))
        for w in args.windows:
            row = "".join(fmt.format(table[w][n][metric]).rjust(10) for n in names)
            print(f"    {w:<9.0f}{row}")
        print()

    print("  verdict per window (both criteria must hold for every corridor)")
    best = None
    for w in args.windows:
        bad_clip = [n for n in names if table[w][n]["at_clip"] > CLIP_TOL]
        bad_peak = [n for n in names if table[w][n]["p95_pct"] > PLAUSIBLE_PEAK_PCT]
        ok = not bad_clip and not bad_peak
        why = ""
        if bad_clip:
            why += f" clip binds on {len(bad_clip)} ({bad_clip[0].replace('route_','')}…)"
        if bad_peak:
            why += f" p95 too high on {len(bad_peak)}"
        print(f"    {w:<9.0f}{'PASS' if ok else 'fail'}{why}")
        if ok and best is None:
            best = w

    if best is None:
        print("\n  no window satisfies both criteria -- widen the sweep or "
              "revisit the criteria")
    else:
        print(f"\n  narrowest window satisfying both: {best:.0f} m")
        print("  (narrowest is the right pick: it is the least smoothing that "
              "clears the noise, so it preserves the most real terrain)")


if __name__ == "__main__":
    main()
