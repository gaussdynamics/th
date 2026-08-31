"""Sweep driving regimes x consist size, with a hard per-run wall-clock budget.

Reports physical sanity (does the run stay finite, does it reverse, what peak
coupler force) alongside the wall-clock cost that sizes the dataset build.

The ``SIGALRM`` budget is not incidental: the dataset driver needs the same
guard, or one pathological consist/regime pairing hangs a 10,000-scenario build
with no diagnostic.

Run from anywhere:  python3 scripts/bench_scenarios.py
"""
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import numpy as np

from simulator2.catalog import CarLibrary
from simulator2.consist import Train, build_consist
from simulator2.driving_regimes import (
    REGIMES_STARTING_FROM_REST,
    Regime,
    generate_regime_profile,
)
from simulator2.randomize import RandomizationConfig, make_randomized_scenario
from simulator2.route import load_route_profile
from simulator2.simulate import simulate_train_tensorized
from simulator2.state_schema import EdgeChannel, NodeChannel


class Timeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(Timeout()))

BUDGET_S = int(os.environ.get("BENCH_BUDGET_S", "30"))
lib = CarLibrary.load("car_library")
ROUTES = {
    n: load_route_profile(f"route_generator/route_profiles/{n}.npz")
    for n in ["route_line0_49km", "route_line5_22km"]
}
TRAINS = [
    "generated_unit_coal_0010",      # N=11
    "generated_unit_water_0002",     # N=60, DPU
    "generated_unit_ballast_0008",   # N=130, DPU
]

cfg = RandomizationConfig(duration_s_range=(120.0, 120.0))
rows = []
for tn in TRAINS:
    tr = Train.load(f"saved_trains/{tn}.json")
    veh, cpl = build_consist(tr, lib)
    for rn, route in ROUTES.items():
        for k, r in enumerate(list(Regime)):
            prof = generate_regime_profile(r, seed=100 + k)
            sc, meta = make_randomized_scenario(
                route, veh, cpl, prof, train=tr, seed=k * 13 + len(rn),
                config=cfg, starts_from_rest=(r in REGIMES_STARTING_FROM_REST),
                route_id=rn, control_id=prof.name, consist_id=tn,
            )
            t0 = time.time()
            signal.alarm(BUDGET_S)
            try:
                res = simulate_train_tensorized(sc)
                signal.alarm(0)
                w = time.time() - t0
                H, E = res.H_hist, res.E_hist
                v = H[:, :, NodeChannel.V]
                x = H[:, :, NodeChannel.X]
                F = E[:, :, EdgeChannel.F_CPL]
                rows.append(dict(
                    status="ok", train=tn, N=meta["N"], route=rn, regime=r.value,
                    v0=meta["v0_mps"], adh=meta["adhesion"], slack=meta["slack_state"],
                    vmin=float(v.min()), vmax=float(v.max()),
                    net_km=float((x[-1, 0] - x[0, 0]) / 1000.0),
                    Fmax_MN=float(np.abs(F).max() / 1e6),
                    finite=bool(np.isfinite(H).all() and np.isfinite(E).all()),
                    wall_s=round(w, 2), nfev=int(res.sol.nfev),
                ))
            except Timeout:
                signal.alarm(0)
                rows.append(dict(status="timeout", train=tn, N=meta["N"], route=rn,
                                 regime=r.value, wall_s=BUDGET_S))
            except Exception as e:
                signal.alarm(0)
                rows.append(dict(status="error", train=tn, N=meta["N"], route=rn,
                                 regime=r.value, err=f"{type(e).__name__}: {e}"[:100]))
            print(".", end="", flush=True)
print()
json.dump(rows, open(_REPO_ROOT / "sweep_results.json", "w"), indent=1)

ok = [r for r in rows if r["status"] == "ok"]
print(f"runs={len(rows)} ok={len(ok)} timeout={sum(1 for r in rows if r['status']=='timeout')} "
      f"error={sum(1 for r in rows if r['status']=='error')}")
for r in rows:
    if r["status"] == "error":
        print("  ERR", r["regime"], r["N"], r["err"])
    if r["status"] == "timeout":
        print("  TMO", r["regime"], r["N"], r["route"])
if ok:
    print("all finite:", all(r["finite"] for r in ok))
    rev = [r for r in ok if r["vmin"] < -0.5]
    print(f"reversing runs (vmin < -0.5 m/s): {len(rev)}")
    for r in rev[:8]:
        print(f"   {r['regime']:22s} N={r['N']:4d} v0={r['v0']:5.1f} vmin={r['vmin']:6.2f} net_km={r['net_km']:6.2f}")
    print()
    print(f"{'N':>5} {'runs':>5} {'med_wall_s':>11} {'max_wall_s':>11} {'med_nfev':>10}")
    for N in sorted({r["N"] for r in ok}):
        g = [r for r in ok if r["N"] == N]
        print(f"{N:5d} {len(g):5d} {statistics.median(r['wall_s'] for r in g):11.2f} "
              f"{max(r['wall_s'] for r in g):11.2f} {statistics.median(r['nfev'] for r in g):10.0f}")
    print()
    print(f"{'regime':24s}{'med_wall_s':>11}{'medFmaxMN':>11}{'med_net_km':>11}")
    for reg in sorted({r["regime"] for r in ok}):
        g = [r for r in ok if r["regime"] == reg]
        print(f"{reg:24s}{statistics.median(r['wall_s'] for r in g):11.2f}"
              f"{statistics.median(r['Fmax_MN'] for r in g):11.3f}"
              f"{statistics.median(r['net_km'] for r in g):11.2f}")
