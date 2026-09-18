#!/usr/bin/env python3
"""What does mapping the whole network into simulator-ready form actually cost?

The pipeline is four stages, and only one of them is expensive:

1. **resample** (`routegen.resample`) -- alignment to a uniform chainage grid at
   ``ds`` metres. Pure CPU.
2. **elevate** (`routegen.elevation`) -- 3DEP ``getSamples``, batched. Network
   bound, and the only stage whose cost is not obviously negligible.
3. **profile** (`routegen.profile`) -- Savitzky-Golay smoothing for grade,
   planform derivatives for curvature, ``v_max``. Pure CPU.
4. **export** (`routegen.export`) -- the four-channel tensor ``.npz`` the
   simulator consumes. Pure CPU.

Stage 2 is issued **sequentially** in the current code: ``sample_elevations``
loops over 100-point chunks, and only the EPQS *fallback* uses a thread pool.
So this benchmark measures chunk size and concurrency separately, because those
are the two knobs that decide whether a continental build is hours or weeks.

Run::

    python scripts/bench_route_pipeline.py
    python scripts/bench_route_pipeline.py --points 2000 --ds 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROUTEGEN = _REPO_ROOT / "route_generator"
for p in (_REPO_ROOT, _ROUTEGEN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import requests  # noqa: E402

from routegen.elevation import IMAGESERVER_URL, parse_getsamples_json  # noqa: E402
from routegen.profile import derive_profile  # noqa: E402
from routegen.resample import resample_feature  # noqa: E402

UA = {"User-Agent": "thesis-routegen/0.1 (pipeline benchmark; research)"}


def sample_once(pts: list[tuple[float, float]], session, timeout=90) -> float:
    """One getSamples request. Returns wall-clock seconds, raises on failure."""
    geom = {"points": [[lon, lat] for lon, lat in pts],
            "spatialReference": {"wkid": 4326}}
    params = {
        "geometry": json.dumps(geom),
        "geometryType": "esriGeometryMultipoint",
        "returnFirstValueOnly": "true",
        "f": "json",
    }
    t0 = time.time()
    r = session.post(IMAGESERVER_URL, data=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    vals = parse_getsamples_json(payload, len(pts))
    dt = time.time() - t0
    got = sum(v is not None for v in vals)
    if got < len(pts) * 0.9:
        raise RuntimeError(f"only {got}/{len(pts)} resolved")
    return dt


def load_track(n_points: int) -> list[tuple[float, float]]:
    """Real alignment vertices to sample against, not synthetic points."""
    p = _ROUTEGEN / "elevated_routes" / "route_line5_22km.geojson"
    coords = json.loads(p.read_text())["geometry"]["coordinates"]
    pts = [(c[0], c[1]) for c in coords]
    while len(pts) < n_points:
        pts = pts + pts
    return pts[:n_points]


def bench_elevation(args) -> dict:
    sess = requests.Session()
    print("== stage 2: elevation (3DEP getSamples) ==\n")
    print("  chunk size sweep, sequential")
    print(f"    {'points/req':>11s} {'s/req':>8s} {'points/s':>10s}  note")
    best_chunk, best_rate = None, 0.0
    for n in args.chunks:
        pts = load_track(n)
        try:
            dts = [sample_once(pts, sess) for _ in range(args.repeat)]
            dt = float(np.median(dts))
            rate = n / dt
            print(f"    {n:11d} {dt:8.2f} {rate:10.0f}")
            if rate > best_rate:
                best_chunk, best_rate = n, rate
        except Exception as e:
            print(f"    {n:11d} {'--':>8s} {'--':>10s}  FAILED: {str(e)[:60]}")
            break

    print(f"\n  concurrency sweep at {best_chunk} points/request")
    print(f"    {'workers':>8s} {'s/batch':>9s} {'points/s':>10s}")
    conc_best = (1, best_rate)
    for w in args.workers:
        pts_sets = [load_track(best_chunk) for _ in range(w * 2)]
        t0 = time.time()
        ok = 0
        with ThreadPoolExecutor(max_workers=w) as ex:
            for r in ex.map(lambda p: _safe(sample_once, p, sess), pts_sets):
                ok += int(r)
        dt = time.time() - t0
        rate = ok * best_chunk / dt if dt > 0 else 0.0
        print(f"    {w:8d} {dt:9.2f} {rate:10.0f}" + ("" if ok == len(pts_sets)
                                                      else f"   ({ok}/{len(pts_sets)} ok)"))
        if rate > conc_best[1]:
            conc_best = (w, rate)
    print(f"\n  best: {best_chunk} points/request x {conc_best[0]} workers "
          f"= {conc_best[1]:,.0f} points/s")
    return {"chunk": best_chunk, "workers": conc_best[0], "rate": conc_best[1]}


def _safe(fn, *a):
    try:
        fn(*a)
        return True
    except Exception:
        return False


def bench_cpu(args) -> dict:
    print("\n== stages 1, 3: resample and profile (CPU) ==\n")
    p = _ROUTEGEN / "elevated_routes" / "route_line0_49km.geojson"
    feat = json.loads(p.read_text())
    km = float(feat["properties"]["length_m"]) / 1000.0

    t0 = time.time()
    rs = resample_feature(feat, ds_m=args.ds)
    t_res = time.time() - t0

    t0 = time.time()
    derive_profile(rs, smooth_window_m=800.0, grade_clip=0.025)
    t_prof = time.time() - t0

    n = len(rs["geometry"]["coordinates"])
    print(f"  corridor {p.stem}: {km:.1f} km -> {n:,} vertices at {args.ds:.0f} m")
    print(f"    resample        {t_res:7.2f}s   {km/max(t_res,1e-9):8.0f} km/s")
    print(f"    derive_profile  {t_prof:7.2f}s   {km/max(t_prof,1e-9):8.0f} km/s")
    return {"km": km, "t_res": t_res, "t_prof": t_prof}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=int, nargs="+",
                    default=[100, 250, 500, 1000, 2000])
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--ds", type=float, default=10.0)
    ap.add_argument("--network-km", type=float, default=None,
                    help="total km to project onto (default: read data/narn)")
    args = ap.parse_args()

    total_km = args.network_km
    if total_km is None:
        seg = _REPO_ROOT / "data" / "narn" / "segments.parquet"
        if seg.exists():
            import pandas as pd
            total_km = float(pd.read_parquet(seg, columns=["KM"]).KM.fillna(0).sum())
    if total_km is None:
        total_km = 323_000.0
        print(f"(no data/narn yet; projecting onto an estimated {total_km:,.0f} km)\n")

    elev = bench_elevation(args)
    cpu = bench_cpu(args)

    n_pts = total_km * 1000.0 / args.ds
    print(f"\n== projection: {total_km:,.0f} km of main line at {args.ds:.0f} m "
          f"= {n_pts/1e6:.1f}M points ==\n")
    print(f"    {'stage':16s} {'hours':>8s}  basis")
    h_res = total_km / (cpu['km'] / cpu['t_res']) / 3600
    h_prof = total_km / (cpu['km'] / cpu['t_prof']) / 3600
    h_elev = n_pts / max(elev['rate'], 1e-9) / 3600
    print(f"    {'resample':16s} {h_res:8.2f}  {cpu['km']/cpu['t_res']:,.0f} km/s")
    print(f"    {'elevation':16s} {h_elev:8.2f}  {elev['rate']:,.0f} pts/s "
          f"({elev['chunk']} x {elev['workers']})")
    print(f"    {'profile':16s} {h_prof:8.2f}  {cpu['km']/cpu['t_prof']:,.0f} km/s")
    print(f"    {'TOTAL':16s} {h_res+h_elev+h_prof:8.2f}")
    seq = n_pts / (elev['rate'] / max(elev['workers'], 1)) / 3600
    print(f"\n  elevation at today's sequential 100-point path: {seq:,.1f} h")


if __name__ == "__main__":
    main()
