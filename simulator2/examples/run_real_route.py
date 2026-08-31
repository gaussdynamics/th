"""Run the tensorized LTD simulator on a **real route** tensor.

This is the Stage-1 integration entry point: it loads a route tensor produced by
the route-generation pipeline (``route_profiles/<name>.npz`` from the Route-
profile tab, or ``routes/<name>.npz`` from the export bundle), builds a default
consist + lead-loco traction ramp over that route, integrates, and prints a
physical sanity summary.

Usage (from the repo root, i.e. the folder that contains ``simulator2/``):

    python -m simulator2.examples.run_real_route path/to/route_line0_49km.npz
    python -m simulator2.examples.run_real_route route.npz --cars 80 --seconds 180
    python -m simulator2.examples.run_real_route route.npz --curv 1e-4 --out rollout.npz

``--curv`` engages the curvature resistance proxy using the route's kappa field;
``--out`` saves t / H_hist / E_hist for inspection. Requires SciPy (the simulator
integrates with ``scipy.integrate.solve_ivp``).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Make ``simulator2`` importable when run as a plain script, not just -m.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from simulator2 import (  # noqa: E402
    load_route_profile,
    make_scenario_on_route,
    simulate_train_tensorized,
)
from simulator2.state_schema import EdgeChannel, NodeChannel  # noqa: E402

_MS_TO_KMH = 3.6


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the LTD simulator on a real route tensor.")
    ap.add_argument("npz", help="Route tensor .npz (route_s/route_sin_theta/route_kappa/route_vmax)")
    ap.add_argument("--cars", type=int, default=20, help="Number of freight cars (default 20)")
    ap.add_argument("--seconds", type=float, default=120.0, help="Sim horizon (default 120 s)")
    ap.add_argument("--trac", type=float, default=200_000.0, help="Lead traction ramp peak (N)")
    ap.add_argument("--curv", type=float, default=0.0, help="k_curv_scale (>0 enables curvature drag)")
    ap.add_argument("--out", default=None, help="Optional .npz to save t / H_hist / E_hist")
    args = ap.parse_args()

    route = load_route_profile(args.npz)
    route_len_km = float(route.s_nodes_m[-1]) / 1000.0
    grade_pct = np.abs(route.sin_theta_nodes) * 100.0
    print(f"Route: {os.path.basename(args.npz)} — {route_len_km:.1f} km, "
          f"{route.s_nodes_m.size} nodes, grade max {grade_pct.max():.2f} % "
          f"(RMS {np.sqrt(np.mean(grade_pct**2)):.2f} %)")
    if route.v_max_nodes is not None:
        print(f"       v_max {route.v_max_nodes.min()*_MS_TO_KMH:.0f}–"
              f"{route.v_max_nodes.max()*_MS_TO_KMH:.0f} km/h")

    scenario = make_scenario_on_route(
        route, n_cars=args.cars, t_span=(0.0, args.seconds),
        F_trac_max_N=args.trac, k_curv_scale=args.curv)
    print(f"Consist: {len(scenario.vehicles)} vehicles (1 loco + {args.cars} cars). Integrating…")

    result = simulate_train_tensorized(scenario)

    t = result.t
    X = result.H_hist[:, :, NodeChannel.X]
    V = result.H_hist[:, :, NodeChannel.V]
    lead_travel = float(X[-1, 0] - X[0, 0])
    v_final = float(V[-1, 0])
    v_max = float(V.max())
    grade_seen = np.array([route.sin_theta_at(float(x)) for x in X[:, 0]]) * 100.0

    print("\n── rollout summary ─────────────────────────────")
    print(f"  steps integrated     {t.size}  ({t[0]:.0f}–{t[-1]:.0f} s)")
    print(f"  lead distance        {lead_travel:7.1f} m")
    print(f"  lead speed final/max {v_final*_MS_TO_KMH:5.1f} / {v_max*_MS_TO_KMH:5.1f} km/h")
    print(f"  grade under lead     {grade_seen.min():+.2f} … {grade_seen.max():+.2f} %")
    if result.E_hist.size:
        f_cpl = result.E_hist[:, :, EdgeChannel.F_CPL]
        kmax, jmax = np.unravel_index(np.argmax(np.abs(f_cpl)), f_cpl.shape)
        print(f"  peak |coupler force| {np.abs(f_cpl).max()/1e3:7.1f} kN "
              f"(coupler {jmax} @ t={t[kmax]:.1f}s)")
    print("────────────────────────────────────────────────")
    print("Sane if: distance > 0, speeds bounded (~< track v_max), coupler force < ~"
          "coupler limits (a few hundred kN). If grade is flat the train just accelerates "
          "under traction; on a grade you should see speed respond to the profile.")

    if args.out:
        np.savez_compressed(args.out, t=t, H_hist=result.H_hist, E_hist=result.E_hist)
        print(f"\nSaved t / H_hist / E_hist → {args.out}")


if __name__ == "__main__":
    main()
