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

The Phase A elevation QA gate is applied **between** elevation and profile, not
before: it reads sampled elevation, so it cannot gate ahead of the spend. A
corridor that fails (more than 5% of 10 m steps jumping over 1 m, the test that
rejected `route_line2`) is recorded and skipped rather than smoothed into range
-- smoothing bad data hides it.

The gate measures **structure-corrected** elevation, not the raw samples.
3DEP returns the mountain surface above a tunnel bore, which is a cliff by the
gate's measure, so gating raw rejected exactly the steep tunnelled corridors
the ood_grade split wants. NARN carries no structure tags (see narn.py), so the
tunnel flag `derive_profile` interpolates across is recovered from the
elevation signature by `routegen.profile.detect_tunnel_runs` and attached to
the feature here. The raw verdict is kept alongside as `elevation_qa_raw`, so
a corridor that only passes because of the correction is visible in the report.

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
from routegen.profile import (  # noqa: E402
    _fill_invalid,
    derive_profile,
    detect_tunnel_runs,
    profile_to_geojson,
    save_profile_npz,
)
from routegen.resample import resample_feature  # noqa: E402

# Phase A settings, selected against stated criteria by tune_grade_smoothing.py
# and recorded in ROUTE_PIPELINE_NOTE.md. Not defaults -- the routegen defaults
# are still the pre-Phase-A 200 m / 4%.
SMOOTH_WINDOW_M = 800.0
# Re-tuned 2026-09-24 on a 71-corridor mountain-inclusive sample. The 2.5% clip
# was doing two jobs -- bounding grade to physical values, and suppressing DEM
# artifacts -- and it was bad at the second, because artifacts and real mountain
# grade differ in LENGTH, not magnitude. Split: the clip is now a physical
# tripwire, and a persistence filter does the artifact work.
GRADE_CLIP = 0.040
EXCURSION_THRESHOLD = 0.025
EXCURSION_MIN_LEN_M = 500.0
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
    """The Phase A gate, on sampled elevation before any smoothing.

    Applied to the *structure-corrected* elevation, not the raw samples. The
    gate asks whether the elevation is usable as a track profile; a tunnel is
    unusable as sampled but recoverable, so correcting first and gating after
    is what separates "this corridor is broken" from "this corridor has a
    tunnel in it". See detect_tunnel_runs in routegen.profile.
    """
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
    ap.add_argument("--excursion-threshold", type=float, default=EXCURSION_THRESHOLD)
    ap.add_argument("--excursion-min-len-m", type=float, default=EXCURSION_MIN_LEN_M,
                    help="0 disables the persistence filter")
    ap.add_argument("--pilot", action="store_true", help="run the 12-corridor pilot")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="evenly spaced sample of the index (deterministic)")
    ap.add_argument("--only", nargs="*", default=None, help="explicit corridor ids")
    ap.add_argument("--only-file", type=Path, default=None,
                    help="JSON list of corridor ids, or {'all': [...]}; too many "
                         "ids for a command line")
    ap.add_argument("--resume", action="store_true",
                    help="skip corridors whose .npz already exists under --out")
    args = ap.parse_args()

    cdir = Path(args.corridors)
    index = {r["corridor_id"]: r for r in csv.DictReader(open(cdir / "index.csv"))}
    if args.only_file:
        blob = json.loads(args.only_file.read_text())
        args.only = blob["all"] if isinstance(blob, dict) else list(blob)
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
        if args.resume and (out / "routes" / ("%s.npz" % cid)).exists():
            print("[%3d/%d] %-34s skip (already built)" % (k, len(ids), cid))
            continue
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

        z = np.asarray(z, float)
        s_m = np.asarray(rs["properties"]["s_m"], float)
        # NARN carries no structure tags, so recover them from the elevation
        # itself before gating -- otherwise every tunnelled mountain corridor
        # is rejected for having a tunnel in it.
        tunnel = detect_tunnel_runs(z, s_m)
        z_corr = _fill_invalid(z, s_m, tunnel | ~np.isfinite(z))

        qa = elevation_qa(z_corr)
        qa_raw = elevation_qa(z)
        rec = {"corridor_id": cid, "n_points": n_pts,
               "length_km": float(index[cid]["length_m"]) / 1000.0 if cid in index else None,
               "elevation_qa": qa, "elevation_qa_raw": qa_raw,
               "tunnel": {"n_samples": int(tunnel.sum()),
                          "length_m": float(tunnel.sum()) * args.ds_m},
               "resample_s": round(t_res, 2),
               "elevation_s": round(t_elev, 2)}

        if not qa["pass"]:
            # Rejected, not smoothed. See the module docstring.
            rec["status"] = "rejected_raw_elevation"
            records.append(rec)
            print("[%3d/%d] %-34s REJECT  %.1f%% of steps over 1 m"
                  "  (raw %.1f%%, %.1f km flagged tunnel)"
                  % (k, len(ids), cid, qa["frac_over_1m"] * 100,
                     qa_raw["frac_over_1m"] * 100, tunnel.sum() * args.ds_m / 1000.0))
            continue

        t2 = time.time()
        elev = elevate_feature(rs, z, est)
        # Store the raw samples and the flag, not the corrected elevation, so
        # the artifact stays honest and derive_profile applies the one fill.
        elev["properties"]["is_tunnel"] = tunnel.tolist()
        prof = derive_profile(elev, smooth_window_m=args.smooth_window_m,
                              grade_clip=args.grade_clip,
                              excursion_threshold=args.excursion_threshold,
                              excursion_min_len_m=args.excursion_min_len_m)
        save_profile_npz(prof, str(out / "routes" / ("%s.npz" % cid)))
        with open(out / "profiles" / ("%s.geojson" % cid), "w") as fh:
            json.dump(profile_to_geojson(elev, prof), fh)
        rec["profile_s"] = round(time.time() - t2, 2)
        rec["grade"] = grade_stats(prof, args.grade_clip)
        rec["status"] = "ok"
        records.append(rec)
        print("[%3d/%d] %-34s ok  %6d pts  rms %.2f%%  p95 %.2f%%  max %.2f%%  "
              "at-clip %.1f%%  tun %.1fkm  (%.1fs elev)"
              % (k, len(ids), cid, n_pts, rec["grade"]["rms_pct"],
                 rec["grade"]["p95_pct"], rec["grade"]["max_pct"],
                 rec["grade"]["frac_at_clip"] * 100,
                 tunnel.sum() * args.ds_m / 1000.0, t_elev))

    elapsed = time.time() - t_all
    ok = [r for r in records if r["status"] == "ok"]
    rejected = [r for r in records if r["status"] != "ok"]
    report = {
        "resumed": bool(args.resume),
        "n_corridors": len(records), "n_ok": len(ok), "n_rejected": len(rejected),
        "rejected_ids": [r["corridor_id"] for r in rejected],
        "ds_m": args.ds_m, "smooth_window_m": args.smooth_window_m,
        "grade_clip": args.grade_clip, "raw_jump_tol": RAW_JUMP_TOL,
        "excursion_threshold": args.excursion_threshold,
        "excursion_min_len_m": args.excursion_min_len_m,
        "n_tunnel_corridors": sum(1 for r in records if r["tunnel"]["n_samples"] > 0),
        "tunnel_detection": "routegen.profile.detect_tunnel_runs",
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
