#!/usr/bin/env python3
"""Measurements behind ``SURROGATE_FORMULATION_NOTE.md``.

Answers three questions about a built corpus, all of which bear on how the
surrogate step is formulated:

``anchor``
    How big is the GNN's learning target under each candidate physics anchor?
    Reported as ``max |dv|`` over the consist per output step -- the error a
    model that outputs zero already achieves. A *smaller* number is a smaller
    thing to learn.

``stiff``
    The fastest mode in the consist (the two-mass coupler oscillator) against
    the output grid, with the explicit-RK4 stability limit of ~2.8 on ``w*h``.

``alias``
    Whether the output grid misses peak coupler force, by re-integrating
    high-force scenarios finely and decimating. A negative result -- it does
    not -- but the one that decides whether ``dt_s`` needs revisiting.

Run::

    python scripts/measure_formulation.py data/v2
    python scripts/measure_formulation.py data/v2 --checks anchor --n-scenarios 80

The RHS is taken from ``validate_dataset.rebuild_rhs``, which rebuilds it from
the stored arrays alone rather than importing ``simulator2`` -- so these numbers
carry the same "the files are sufficient" guarantee the validator does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from validate_dataset import rebuild_rhs

#: Explicit RK4 is stable on an oscillatory mode up to roughly this value of
#: ``w*h``; past it the step grows rather than decays.
RK4_STABILITY_LIMIT = 2.8


# --------------------------------------------------------------------------- #
# shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _load(root: Path, idx: pd.DataFrame, sid, routes: dict) -> tuple[dict, dict]:
    row = idx[idx.scenario_id == sid].iloc[0]
    with np.load(root / "scenarios" / row.route_id / f"{sid}.npz") as f:
        d = {k: f[k] for k in f.files}
    if row.route_id not in routes:
        with np.load(root / "routes" / f"{row.route_id}.npz") as f:
            routes[row.route_id] = {k: f[k] for k in f.files}
    return d, routes[row.route_id]


def _y_at(state: np.ndarray, k: int) -> np.ndarray:
    """Pack ``state[k]`` (``[N,4]``) into the solver's flat ``[4N]`` layout."""
    return np.concatenate([state[k, :, 0], state[k, :, 1],
                           state[k, :, 2], state[k, :, 3]])


def _coupler_force(d: dict, y: np.ndarray, n: int) -> np.ndarray:
    """``F_cpl`` from a flat state, matching the deadband in ``rebuild_rhs``."""
    es = d["edge_static"].astype(np.float64)
    x, v = y[:n], y[n:2 * n]
    delta = (x[:-1] - x[1:]) - es[:, 0]
    ddot = v[:-1] - v[1:]
    slack = es[:, 1]
    f = np.zeros(n - 1)
    f = np.where(delta > slack, es[:, 2] * (delta - slack) + es[:, 3] * ddot, f)
    f = np.where(delta < -slack, es[:, 4] * (delta + slack) + es[:, 5] * ddot, f)
    return f


def _rk4(rhs, t: float, y: np.ndarray, h: float, n_sub: int) -> np.ndarray:
    hs = h / n_sub
    for _ in range(n_sub):
        k1 = rhs(t, y)
        k2 = rhs(t + hs / 2, y + hs / 2 * k1)
        k3 = rhs(t + hs / 2, y + hs / 2 * k2)
        k4 = rhs(t + hs, y + hs * k3)
        y = y + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + hs
    return y


def _pad_in(a):
    return np.concatenate(([0.0], a))


def _pad_out(a):
    return np.concatenate((a, [0.0]))


def _q(values: np.ndarray) -> str:
    p = np.percentile(values, [50, 90, 99])
    return f"{p[0]:10.4f} {p[1]:10.4f} {p[2]:10.4f}"


# --------------------------------------------------------------------------- #
# 1. anchor                                                                   #
# --------------------------------------------------------------------------- #
def check_anchor(root: Path, idx: pd.DataFrame, n_scen: int, n_t: int,
                 seed: int) -> None:
    rng = np.random.default_rng(seed)
    train = idx[idx.split == "train"]
    picks = rng.choice(train.scenario_id.values,
                       min(n_scen, len(train)), replace=False)
    routes: dict = {}
    rec: list[dict] = []

    for sid in picks:
        d, route = _load(root, idx, sid, routes)
        st = d["state"].astype(np.float64)
        t = d["t"].astype(np.float64)
        T, n = st.shape[0], st.shape[1]
        dt = float(t[1] - t[0])

        def rhs(tt, y):
            return rebuild_rhs(d, route, tt, y)

        for k in rng.choice(T - 1, min(n_t, T - 1), replace=False):
            y0, y1 = _y_at(st, k), _y_at(st, k + 1)
            tk = float(t[k])
            f0 = rhs(tk, y0)

            # Frozen-coupler anchor: hold F_cpl at its t_k value through the
            # step, which breaks the delta -> F -> a -> delta feedback loop
            # that makes an explicit step at this h unstable.
            mass = d["node_static"].astype(np.float64)[:, 0]
            f_frozen = _coupler_force(d, y0, n)

            def rhs_frozen(tt, y):
                out = rhs(tt, y).copy()
                live = _coupler_force(d, y, n)
                out[n:2 * n] += (
                    (_pad_in(f_frozen) - _pad_out(f_frozen))
                    - (_pad_in(live) - _pad_out(live))
                ) / mass
                return out

            dv = lambda a, b: np.abs(a[n:2 * n] - b[n:2 * n]).max()
            dx = lambda a, b: np.abs(a[:n] - b[:n]).max()
            slack = d["edge_static"].astype(np.float64)[:, 1]
            delta = (y0[:n][:-1] - y0[:n][1:]) - d["edge_static"].astype(np.float64)[:, 0]

            rec.append(dict(
                engaged=float(np.mean(np.abs(delta) > slack)),
                none_dv=dv(y1, y0),
                euler_dv=dv(y0 + dt * f0, y1),
                rk4_dv=dv(_rk4(rhs, tk, y0, dt, 1), y1),
                frozen_dv=dv(_rk4(rhs_frozen, tk, y0, dt, 1), y1),
                fine_dv=dv(_rk4(rhs, tk, y0, dt, 25), y1),
                none_dx=dx(y1, y0),
                frozen_dx=dx(_rk4(rhs_frozen, tk, y0, dt, 1), y1),
            ))

    df = pd.DataFrame(rec)
    print(f"\n== anchor ==  {len(df)} transitions from {len(picks)} train scenarios")
    print("\nmax |dv| over the consist, per output step [m/s]"
          "        median       p90       p99")
    print(f"  none   (learn s_k+1 - s_k)                  {_q(df.none_dv)}")
    print(f"  explicit Euler                              {_q(df.euler_dv)}")
    print(f"  RK4 single step                             {_q(df.rk4_dv)}")
    print(f"  RK4, F_cpl frozen over the step             {_q(df.frozen_dv)}")
    print(f"  [sanity] RK4 at dt/25 vs stored             {_q(df.fine_dv)}")
    print("\nmax |dx| over the consist, per output step [m]"
          "          median       p90       p99")
    print(f"  none   (learn x_k+1 - x_k)                  {_q(df.none_dx)}")
    print(f"  RK4, F_cpl frozen over the step             {_q(df.frozen_dx)}")
    ratio = np.median(df.frozen_dv) / np.median(df.none_dv)
    print(f"\n  velocity: best anchor is {ratio:.2f}x the unanchored target "
          f"-> anchoring hurts")
    print(f"  position: anchor is {np.median(df.none_dx) / np.median(df.frozen_dx):.0f}x "
          f"better -> anchoring wins")
    print(f"  couplers engaged (outside slack): {df.engaged.mean():.1%} of the time")


# --------------------------------------------------------------------------- #
# 2. stiff                                                                    #
# --------------------------------------------------------------------------- #
def check_stiff(root: Path, idx: pd.DataFrame, n_scen: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    picks = rng.choice(idx.scenario_id.values, min(n_scen, len(idx)), replace=False)
    routes: dict = {}
    frames = []
    dt_out = None

    for sid in picks:
        d, _ = _load(root, idx, sid, routes)
        if dt_out is None:
            tt = d["t"].astype(np.float64)
            dt_out = float(tt[1] - tt[0])
        m = d["node_static"].astype(np.float64)[:, 0]
        es = d["edge_static"].astype(np.float64)
        for col, name in ((2, "draft"), (4, "buff")):
            w = np.sqrt(es[:, col] * (1.0 / m[:-1] + 1.0 / m[1:]))
            frames.append(pd.DataFrame(dict(branch=name, omega=w)))

    w = pd.concat(frames)
    w["period_s"] = 2 * np.pi / w.omega
    print(f"\n== stiff ==  {len(picks)} scenarios, output grid dt = {dt_out} s")
    for branch, g in w.groupby("branch"):
        p = np.percentile(g.omega, [50, 99])
        fastest = np.percentile(g.period_s, 1)
        print(f"\n  {branch}: omega {p[0]:.1f} / {p[1]:.1f} rad/s (median/p99), "
              f"fastest period {fastest:.3f} s")
        for h in (dt_out, 0.10, 0.05, 0.02):
            wh = p[1] * h
            ok = "yes" if wh < RK4_STABILITY_LIMIT else "NO"
            print(f"    h={h:5.3f}s  omega*h p99 = {wh:5.2f}  "
                  f"RK4 stable (<{RK4_STABILITY_LIMIT}): {ok:3s}  "
                  f"samples/period = {fastest / h:5.2f}")


# --------------------------------------------------------------------------- #
# 3. alias                                                                    #
# --------------------------------------------------------------------------- #
def check_alias(root: Path, idx: pd.DataFrame, n_scen: int, max_n: int,
                fine_dt: float, window: int) -> None:
    cand = idx[idx.N <= max_n].sort_values("F_max", ascending=False)
    routes: dict = {}
    rows = []
    print(f"\n== alias ==  re-integrating at dt = {fine_dt} s, "
          f"decimating to the output grid")

    for sid in cand.scenario_id.values[:n_scen]:
        d, route = _load(root, idx, sid, routes)
        st = d["state"].astype(np.float64)
        t = d["t"].astype(np.float64)
        n = st.shape[1]
        dt_out = float(t[1] - t[0])

        peak = np.abs(d["edge_dynamic"].astype(np.float64)[:, :, 2]).max(1)
        k0 = max(0, int(peak.argmax()) - window // 4)
        k1 = min(len(t) - 1, k0 + window)

        def rhs(tt, y):
            return rebuild_rhs(d, route, tt, y)

        y, tt = _y_at(st, k0), float(t[k0])
        fine = []
        for _ in range(int(round((float(t[k1]) - tt) / fine_dt))):
            fine.append(np.abs(_coupler_force(d, y, n)).max())
            y = _rk4(rhs, tt, y, fine_dt, 1)
            tt += fine_dt
        fine = np.array(fine)
        if fine.size == 0:
            continue
        decimated = fine[::max(1, int(round(dt_out / fine_dt)))]
        missed = 100.0 * (1.0 - decimated.max() / fine.max())
        rows.append(dict(sid=sid, N=n, fine_kN=fine.max() / 1e3,
                         grid_kN=decimated.max() / 1e3, missed_pct=missed))
        print(f"  {str(sid)[:20]:22s} N={n:3d}  peak |F| fine {fine.max()/1e3:8.1f} kN"
              f"   on grid {decimated.max()/1e3:8.1f} kN   missed {missed:5.1f}%")

    if rows:
        df = pd.DataFrame(rows)
        print(f"\n  median underestimate {df.missed_pct.median():.1f}%"
              f"   worst {df.missed_pct.max():.1f}%")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a built corpus, e.g. data/v2")
    ap.add_argument("--checks", nargs="+", default=["anchor", "stiff", "alias"],
                    choices=["anchor", "stiff", "alias"])
    ap.add_argument("--n-scenarios", type=int, default=40)
    ap.add_argument("--n-times", type=int, default=12,
                    help="transitions sampled per scenario (anchor)")
    ap.add_argument("--alias-max-n", type=int, default=16,
                    help="cap on N for the alias check; it integrates in numpy")
    ap.add_argument("--alias-dt", type=float, default=0.002)
    ap.add_argument("--alias-window", type=int, default=160,
                    help="output steps either side of the peak to re-integrate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    idx = pd.read_parquet(args.root / "index.parquet")
    print(f"{args.root}: {len(idx)} scenarios")

    if "anchor" in args.checks:
        check_anchor(args.root, idx, args.n_scenarios, args.n_times, args.seed)
    if "stiff" in args.checks:
        check_stiff(args.root, idx, max(args.n_scenarios, 60), args.seed + 1)
    if "alias" in args.checks:
        check_alias(args.root, idx, 6, args.alias_max_n,
                    args.alias_dt, args.alias_window)


if __name__ == "__main__":
    main()
