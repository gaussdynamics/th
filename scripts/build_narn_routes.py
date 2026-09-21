#!/usr/bin/env python3
"""Carry NARN corridors through resample -> elevate -> QA -> profile -> export.

`build_narn_corridors.py` turns the NARN graph into continuous alignments. This
script runs them through the four existing `routegen` stages, unchanged, and
writes the four-channel ``.npz`` the simulator consumes.

Only the elevation stage costs anything. `ROUTE_PIPELINE_NOTE.md` records
16,427 pts/s from 2026-09-17; that does **not** reproduce. Measured 2026-09-20
on the same code path (``method="batch"``, zero EPQS fallback): throughput rises
with batch size and plateaus at **~1,700 pts/s** -- 204 pts/s at 1,000 points,
1,367 at 20,000, 1,680 at 60,000. Raising ``batch_workers`` from 16 to 64 buys
only 1.5x, so the service is rate-limiting rather than latency-bound now. Budget
~4.5 h for the whole continent at 10 m spacing, not 0.46 h. A few hundred
corridors, which is all the splits need, is 10-20 minutes.

Because throughput depends on batch size, sampling per corridor (~5,000 points)
is about 2x worse than sampling many corridors at once. Batching elevation
across corridors is the obvious next optimization; it is not done here.

The Phase A raw-elevation QA gate is applied **between** elevation and profile,
not before: it reads sampled elevation, so it cannot gate ahead of the spend.
A corridor whose raw elevation fails (more than 5% of 10 m steps jumping over
1 m, the test that rejected `route_line2`) is recorded and skipped rather than
smoothed into range -- smoothing bad data hides it.

Output, under ``--out``:

``routes/<corridor_id>.npz``     the DATA_SCHEMA section F route tensor
``profiles/<corridor_id>.geojson``  the full profile, for inspection and figures
``build_report.json``            per-corridor QA verdicts, grade stats, timings

Run::

    python scripts/build_narn_routes.py --pilot          # 12 corridors, all cases
    python scripts/build_narn_routes.py --limit 200
    python scripts/build_narn_routes.py                  # everything
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "route_generator"))

from routegen.elevation import elevate_feature, sample_elevations  # noqa: E402
from routegen.profile import derive_profile, profile_to_geojson, save_profile_npz  # noqa: E402
from routegen.resample import resample_feature  # noqa: E402

# Phase A settings, selected against stated criteria by tune_grade_smoothing.py
# and recorded in ROUTE_PIPELINE_NOTE.md. Not defaults -- the routegen defaults
# are still the pre-Phase-A 200 m / 4%.
SMOOTH_WINDOW_M = 800.0
GRADE_CLIP = 0.025
# Raw-elevation gate: the fraction of 10 m steps allowed to jump over 1 m.
# route_line2 failed at 19.3%; every other corridor measured 0.7-1.5%.
RAW_JUMP_TOL = 0.05

# A pilot spanning the three topology cases, both extremes of terrain, and the
# steep corridors the ood_grade split needs. Ids come from the continental index
# and are checked against it at run time, so a re-traversal that renumbers pieces
# reports the miss rather than silently shrinking the pilot.
PILOT_IDS = [
    "rrrr_tennessee_pass_p0_39km",   # steep, tunnelled: Tennessee Pass
    "bnsf_raton_p0_56km",            # steep: Raton Pass
    "bnsf_raton_p2_47km",
    "up_moffat_tunnel_p0_56km",      # steep, heavily tunnelled
    "up_moffat_tunnel_p2_56km",
    "bnsf_cajon_p0_52km",            # steep: Cajon Pass, multi-track
    "up_greeley_p2_49km",            # plains, cut from a long subdivision
    "up_green_river_p0_51km",        # desert plateau
    "cxrg_alamosa_p0_59km",          # short-line, single track
    "bnsf_boise_city_p0_46km",       # plains, mixed track class
    "bnsf_front_range_p2_47km",      # multi-track, passenger
    "nkcr_wallace_p1_55km",          # sparse native geometry
]


def elevation_qa(z: np.ndarray) -> dict:
    """The Phase A gate, on the raw sampled elevation before any smoothing."""
    dz = np.abs(np.diff(z))
    frac = float((dz > 1.0).mean()) if dz.size else 0.0
    return {
        "median_step_m": float(np.median(dz)) if dz.size else 0.0,
        "frac_over_1m": frac,
        "roughness_m": float(np.median(np.abs(np.diff(z, 2)))) if dz.size > 1 else 0.0,
        "pass": bool(frac <= RAW_JUMP_TOL),
    }


def grade_stats(prof: dict, clip: float) -> dict:
    g = np.abs(np.asarray(prof["grade"], dtype=float)) * 100.0
    return {
        "rms_pct": float(np.sqrt((g ** 2).mean())),
        "p95_pct": float(np.percentile(g, 95)),
        "max_pct": float(g.max()),
        "frac_at_clip": float((g >= clip * 100.0 - 1e-9).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corridors", default="data/narn_corridors")
    ap.add_argument("--out", default="data/narn_routes")
    ap.add_argument("--ds-m", type=float, default=10.0)
    ap.add_argument("--smooth-window-m", type=float, default=SMOOTH_WINDOW_M)
    ap.add_argument("--grade-clip", type=float, default=GRADE_CLIP)
    ap.add_argument("--pilot", action="store_true", help="run the 12-corridor pilot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="evenly spaced sample of the index (deterministic)")
    ap.add_argument("--only", nargs="*", default=None, help="explicit corridor ids")
    args = ap.parse_args()

    cdir = Path(args.corridors)
    index = {r["corridor_id"]: r for r in csv.DictReader(open(cdir / "index.csv"))}
    if args.only:
        ids = list(args.only)
    elif args.pilot:
        ids = [i for i in PILOT_IDS if i in index]
        missing = [i for i in PILOT_IDS if i not in index]
        if missing:
            print("pilot ids absent from the index (skipped): %s" % ", ".join(missing))
    elif args.sample:
        # Evenly spaced through the sorted index: spreads the sample over owners
        # and regions without an RNG, so the same N is always the same N.
        allids = sorted(index)
        step = max(1, len(allids) // args.sample)
        ids = allids[::step][:args.sample]
    else:
        ids = sorted(index)
        if args.limit:
            ids = ids[:args.limit]

    out = Path(args.out)
    (out / "routes").mkdir(parents=True, exist_ok=True)
    (out / "profiles").mkdir(parents=True, exist_ok=True)

    records, t_all = [], time.time()
    n_pts_total = 0
    for k, cid in enumerate(ids, 1):
        t0 = time.time()
        feat = json.loads((cdir / "corridors" / ("%s.geojson" % cid)).read_text())

        rs = resample_feature(feat, ds_m=args.ds_m)
        lonlats = [(c[0], c[1]) for c in rs["geometry"]["coordinates"]]
        n_pts = len(lonlats)
        t_res = time.time() - t0

        t1 = time.time()
        z, est = sample_elevations(lonlats)
        t_elev = time.time() - t1
        n_pts_total += n_pts

        qa = elevation_qa(np.asarray(z, float))
        rec = {"corridor_id": cid, "n_points": n_pts,
               "length_km": float(index[cid]["length_m"]) / 1000.0 if cid in index else None,
               "elevation_qa": qa, "resample_s": round(t_res, 2),
               "elevation_s": round(t_elev, 2)}

        if not qa["pass"]:
            # Rejected, not smoothed. See the module docstring.
            rec["status"] = "rejected_raw_elevation"
            records.append(rec)
            print("[%3d/%d] %-34s REJECT  %.1f%% of steps over 1 m"
                  % (k, len(ids), cid, qa["frac_over_1m"] * 100))
            continue

        t2 = time.time()
        elev = elevate_feature(rs, np.asarray(z, float), est)
        prof = derive_profile(elev, smooth_window_m=args.smooth_window_m,
                              grade_clip=args.grade_clip)
        save_profile_npz(prof, str(out / "routes" / ("%s.npz" % cid)))
        with open(out / "profiles" / ("%s.geojson" % cid), "w") as fh:
            json.dump(profile_to_geojson(elev, prof), fh)
        rec["profile_s"] = round(time.time() - t2, 2)
        rec["grade"] = grade_stats(prof, args.grade_clip)
        rec["status"] = "ok"
        records.append(rec)
        print("[%3d/%d] %-34s ok  %6d pts  rms %.2f%%  p95 %.2f%%  max %.2f%%  "
              "at-clip %.1f%%  (%.1fs elev)"
              % (k, len(ids), cid, n_pts, rec["grade"]["rms_pct"],
                 rec["grade"]["p95_pct"], rec["grade"]["max_pct"],
                 rec["grade"]["frac_at_clip"] * 100, t_elev))

    elapsed = time.time() - t_all
    ok = [r for r in records if r["status"] == "ok"]
    rejected = [r for r in records if r["status"] != "ok"]
    report = {
        "n_corridors": len(records), "n_ok": len(ok), "n_rejected": len(rejected),
        "rejected_ids": [r["corridor_id"] for r in rejected],
        "ds_m": args.ds_m, "smooth_window_m": args.smooth_window_m,
        "grade_clip": args.grade_clip, "raw_jump_tol": RAW_JUMP_TOL,
        "n_points_total": n_pts_total, "elapsed_s": round(elapsed, 1),
        "elevation_pts_per_s": round(
            n_pts_total / sum(r["elevation_s"] for r in records), 0)
        if records else 0,
        "corridors": records,
    }
    with open(out / "build_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n%d ok, %d rejected, %d points in %.0f s (%.0f pts/s on elevation)"
          % (len(ok), len(rejected), n_pts_total, elapsed,
             report["elevation_pts_per_s"]))
    if ok:
        clip = sorted(r["grade"]["frac_at_clip"] for r in ok)
        rms = sorted(r["grade"]["rms_pct"] for r in ok)
        print("  at-clip fraction  median %.2f%%  worst %.2f%%"
              % (clip[len(clip) // 2] * 100, clip[-1] * 100))
        print("  rms grade         median %.2f%%  worst %.2f%%"
              % (rms[len(rms) // 2], rms[-1]))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
