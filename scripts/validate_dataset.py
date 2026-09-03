"""Integrity check for a built dataset.

Answers the question the build itself cannot: is what landed on disk internally
consistent, and does it actually contain the physics it claims to?

    python scripts/validate_dataset.py data/v1

Three classes of check:

1. **Manifest consistency** -- ``index.parquet`` and ``splits.json`` describe
   the same set of scenarios, splits are disjoint, and every referenced file
   exists.
2. **Physics self-consistency** -- the RHS is rebuilt from *only* what the
   scenario ``.npz`` and its route file contain, and compared against the
   derivative implied by the stored trajectory. This is the check that catches
   a dropped or mislabelled field: if ``node_static`` columns were permuted, or
   a constant from §H were missing, the residual explodes.

   Note the comparison is against a central difference on the *output* grid,
   which carries its own O(dt_out^2) truncation error. At the default 0.25 s
   spacing that is several m/s^2 on scenarios with active couplers, and it is
   not a defect in the data -- ``--fd-tolerance`` is set accordingly. The
   scaling was verified separately: the residual falls from 2.2e-2 to 3.9e-5
   m/s^2 as dt_out goes from 0.25 to 0.01 on a fixed scenario.
3. **Distribution sanity** -- speed-limit violations, reversals, and consist
   sizes, so a build whose inputs went wrong is visible without opening files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def rebuild_rhs(d: dict, route: dict, t: float, y: np.ndarray) -> np.ndarray:
    """The extended RHS, from the stored arrays alone.

    Deliberately a standalone reimplementation -- importing ``simulator2.rhs``
    or ``torch_rhs`` would let a shared bug hide, and the point of this check is
    that the files are sufficient on their own.
    """
    n = d["node_static"].shape[0]
    x, v = y[:n], y[n:2 * n]
    z_brk, z_trac = y[2 * n:3 * n], y[3 * n:4 * n]
    ns = d["node_static"].astype(np.float64)
    es = d["edge_static"].astype(np.float64)
    mass, d_a, d_b, d_c = ns[:, 0], ns[:, 1], ns[:, 2], ns[:, 3]
    can_trac, f_brk_max = ns[:, 4], ns[:, 6]

    delta = (x[:-1] - x[1:]) - es[:, 0]
    ddot = v[:-1] - v[1:]
    slack, k_d, c_d, k_b, c_b = es[:, 1], es[:, 2], es[:, 3], es[:, 4], es[:, 5]
    f_c = np.zeros(n - 1)
    f_c = np.where(delta > slack, k_d * (delta - slack) + c_d * ddot, f_c)
    f_c = np.where(delta < -slack, k_b * (delta + slack) + c_b * ddot, f_c)

    tt = d["t"].astype(np.float64)
    u_t = np.array([np.interp(t, tt, d["u_trac"][:, i].astype(np.float64))
                    for i in range(n)])
    u_b = np.array([np.interp(t, tt, d["u_brk"][:, i].astype(np.float64))
                    for i in range(n)])
    u_t = np.where(can_trac > 0, u_t, 0.0)

    sgn = np.where(np.abs(v) < 1e-9, 0.0, np.sign(v))
    r = (d_a + d_b * np.abs(v) + d_c * v * v) * sgn
    g = mass * 9.81 * np.sin(
        np.interp(x, route["route_s"], route["route_sin_theta"])
    )
    kap = np.interp(x, route["route_s"], route["route_kappa"])
    kcs = float(d["k_curv_scale"])
    c_cur = np.where(np.abs(v) < 1e-9, 0.0, kcs * mass * v * v * np.abs(kap) * sgn)

    zt, zb = np.maximum(z_trac, 0.0), np.maximum(z_brk, 0.0)
    f_tr = np.where(
        can_trac > 0,
        np.minimum(zt, float(d["p_max_w"]) / np.maximum(np.abs(v), float(d["v_eps"]))),
        0.0,
    )
    f_br = np.minimum(zb, f_brk_max)
    if bool(d["brake_opposes_motion"]):
        f_br = f_br * np.tanh(v / float(d["v_brake_eps"]))

    f_in = np.concatenate([[0.0], f_c])
    f_out = np.concatenate([f_c, [0.0]])
    dv = (f_tr - f_br - r - g - c_cur + f_in - f_out) / mass
    return np.concatenate([
        v, dv,
        (u_b - z_brk) / float(d["tau_brk_s"]),
        (u_t - z_trac) / float(d["tau_trac_s"]),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--n-physics", type=int, default=24,
                    help="scenarios to spot-check the physics on")
    ap.add_argument("--fd-tolerance", type=float, default=25.0,
                    help="max |dv/dt| residual (m/s^2) against a central difference "
                         "on the output grid; loose because that difference has its "
                         "own O(dt_out^2) error, not because the data is loose")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    root = args.root
    idx = pd.read_parquet(root / "index.parquet")
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    norm = json.loads((root / "norm_stats.json").read_text(encoding="utf-8"))
    failures: list[str] = []

    print(f"index.parquet : {len(idx)} rows x {len(idx.columns)} columns")
    print(f"splits.json   : {sum(len(v) for v in splits.values())} ids "
          f"across {len(splits)} splits")

    # -- 1. manifests ------------------------------------------------------
    ids_split = {i for v in splits.values() for i in v}
    if set(idx.scenario_id) != ids_split:
        failures.append("index.parquet and splits.json disagree on scenario ids")
    seen: set = set()
    for name, members in splits.items():
        overlap = seen & set(members)
        if overlap:
            failures.append(f"split {name} overlaps another ({len(overlap)} ids)")
        seen |= set(members)

    missing = [
        sid for sid, rid in zip(idx.scenario_id, idx.route_id)
        if not (root / "scenarios" / rid / f"{sid}.npz").exists()
    ]
    if missing:
        failures.append(f"{len(missing)} scenario files missing, e.g. {missing[:3]}")
    for rid in sorted(set(idx.route_id)):
        if not (root / "routes" / f"{rid}.npz").exists():
            failures.append(f"route file missing: {rid}")

    if norm.get("_computed_over", {}).get("split") != "train":
        failures.append("norm_stats was not computed over the train split")
    n_train = int((idx.split == "train").sum())
    if norm.get("_computed_over", {}).get("n_scenarios") != n_train:
        failures.append(
            f"norm_stats covers {norm['_computed_over']['n_scenarios']} scenarios "
            f"but the train split has {n_train}"
        )
    for field, entry in norm.items():
        if field.startswith("_"):
            continue
        if any(s <= 0 for s in entry["std"]):
            failures.append(f"norm_stats[{field}] has a non-positive std")

    print("manifests     : " + ("OK" if not failures else f"{len(failures)} PROBLEM(S)"))

    # -- 2. physics --------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(idx.scenario_id.values,
                       min(args.n_physics, len(idx)), replace=False)
    worst_x = worst_v = 0.0
    worst_id = ""
    routes: dict = {}
    for sid in picks:
        row = idx[idx.scenario_id == sid].iloc[0]
        with np.load(root / "scenarios" / row.route_id / f"{sid}.npz") as f:
            d = {k: f[k] for k in f.files}
        if row.route_id not in routes:
            with np.load(root / "routes" / f"{row.route_id}.npz") as f:
                routes[row.route_id] = {k: f[k] for k in f.files}
        route = routes[row.route_id]

        st = d["state"].astype(np.float64)
        t = d["t"].astype(np.float64)
        n = st.shape[1]
        y = np.concatenate([st[:, :, 0], st[:, :, 1], st[:, :, 2], st[:, :, 3]], axis=1)
        for k in (len(t) // 4, len(t) // 2, 3 * len(t) // 4):
            dy = rebuild_rhs(d, route, float(t[k]), y[k])
            fd = (y[k + 1] - y[k - 1]) / (t[k + 1] - t[k - 1])
            worst_x = max(worst_x, float(np.abs(dy[:n] - fd[:n]).max()))
            ev = float(np.abs(dy[n:2 * n] - fd[n:2 * n]).max())
            if ev > worst_v:
                worst_v, worst_id = ev, str(sid)
        if not np.isfinite(st).all():
            failures.append(f"{sid}: non-finite values in state")

    print(f"physics       : {len(picks)} scenarios x 3 points")
    print(f"  max |dx/dt residual| = {worst_x:.3e} m/s")
    print(f"  max |dv/dt residual| = {worst_v:.3e} m/s^2   ({worst_id})")
    if worst_v > args.fd_tolerance:
        failures.append(
            f"dv/dt residual {worst_v:.2f} exceeds {args.fd_tolerance} m/s^2"
        )

    # -- 3. distribution ---------------------------------------------------
    print("\nsplits:")
    for name in ("train", "val", "test_id", "ood_size",
                 "ood_grade", "ood_corridor", "control_eval"):
        n = int((idx.split == name).sum())
        print(f"  {name:14s} {n:7d}{'' if n else '   <-- EMPTY'}")

    over = idx.v_over_limit_max_mps if "v_over_limit_max_mps" in idx else None
    print("\ndistribution:")
    print(f"  N                       {idx.N.min()} .. {idx.N.max()}")
    print(f"  duration_s              {idx.duration_s.min():.0f} .. {idx.duration_s.max():.0f}")
    print(f"  F_max (N)               {idx.F_max.min():.3e} .. {idx.F_max.max():.3e}")
    print(f"  v_max_mps               {idx.v_max_mps.max():.2f}")
    print(f"  v_min_mps               {idx.v_min_mps.min():.2f}")
    print(f"  reversing (< -1 m/s)    {int((idx.v_min_mps < -1).sum())} "
          f"({(idx.v_min_mps < -1).mean() * 100:.0f}%)")
    if over is not None:
        print(f"  never over speed limit  {int((over <= 0).sum())} "
              f"({(over <= 0).mean() * 100:.0f}%)")
        print(f"  over limit by > 5 m/s   {int((over > 5).sum())} "
              f"({(over > 5).mean() * 100:.0f}%)")
        print(f"  over limit by > 20 m/s  {int((over > 20).sum())} "
              f"({(over > 20).mean() * 100:.0f}%)")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f_ in failures:
            print(f"  - {f_}")
        raise SystemExit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
