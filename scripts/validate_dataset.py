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

   Under the friction draft gear, a gear driven into its end stop gives an
   impact (100+ m/s^2 within one output step) that no difference on the
   output grid can resolve, so check points are moved off samples where any
   gear is beyond its travel (``draft_gear_end_m``).
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


def _notch_traction(n, speed, lp, v_eps):
    """LTD benchmark Table 4 at integer notch n: (n/8) Te_max, power-capped."""
    return np.minimum(n / 8.0 * lp[:, 0], (n / 8.0) ** 2 * lp[:, 1] / np.maximum(speed, v_eps))


def _notch_db(n, speed, lp, v_eps):
    """Benchmark Table 5 as read against its Figs 4b/5b (see torch_rhs)."""
    db, p_db = lp[:, 2], lp[:, 3]
    v_min, v_max = np.maximum(lp[:, 4], 1e-9), lp[:, 5]
    ramp = speed / v_min * db
    hyper = p_db / np.maximum(speed, v_eps)
    level = np.where(n <= 3, n / 8.7, n / 7.5) * db - 10.0 * speed
    partial = np.minimum(np.minimum(ramp, level), hyper)
    full = np.where(speed <= v_min, ramp, np.where(speed <= v_max, db - 10.0 * speed, hyper))
    f = np.where(n >= 7, full, partial)
    return np.where(n > 0, np.maximum(f, 0.0), 0.0)


def _notch_force(z, v, lp, v_eps):
    """Signed locomotive force at lagged notch z, blended between notches."""
    speed = np.abs(v)
    mag = np.minimum(np.abs(z), 8.0)
    lo = np.floor(mag)
    hi = np.minimum(lo + 1.0, 8.0)
    w = mag - lo
    trac = (_notch_traction(lo, speed, lp, v_eps) * (1 - w)
            + _notch_traction(hi, speed, lp, v_eps) * w)
    brake = _notch_db(lo, speed, lp, v_eps) * (1 - w) + _notch_db(hi, speed, lp, v_eps) * w
    sgn = np.where(speed < 1e-9, 0.0, np.sign(v))
    return np.where(z >= 0, trac, -brake * sgn)


def rebuild_rhs(d: dict, route: dict, t: float, y: np.ndarray,
                k: int | None = None) -> np.ndarray:
    """The extended RHS, from the stored arrays alone.

    Deliberately a standalone reimplementation -- importing ``simulator2.rhs``
    or ``torch_rhs`` would let a shared bug hide, and the point of this check is
    that the files are sufficient on their own.

    Files from a non-legacy physics configuration carry ``physics_json`` and
    the per-vehicle arrays it needs. Under the friction draft gear the
    coupler force depends on the gears' history, so the stored ``F_cpl`` at
    output sample ``k`` is used, and the check is that it is consistent with
    the node accelerations.
    """
    phys = json.loads(str(d["physics_json"])) if "physics_json" in d else {}
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
    if phys.get("coupler_model") == "draft_gear":
        f_c = d["edge_dynamic"][k, :, 2].astype(np.float64)
    else:
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
    sin_th = np.interp(x, route["route_s"], route["route_sin_theta"])
    # the legacy code applied sin to a field that already holds sin(theta)
    g = mass * 9.81 * (sin_th if phys.get("grade_model") == "exact" else np.sin(sin_th))
    kap = np.interp(x, route["route_s"], route["route_kappa"])
    kcs = float(d["k_curv_scale"])
    # The curvature law is per-build, so it has to be read from the file rather
    # than assumed -- applying proxy_v2's k*m*v^2*|kappa| to a build made with a
    # speed-independent law is wrong by a factor of ~33 at 15 m/s, which the
    # loose finite-difference tolerance would happily swallow.
    model = str(d.get("curvature_model", "proxy_v2"))
    if model == "proxy_v2":
        mag = kcs * mass * v * v * np.abs(kap)
    else:
        k_abs = np.minimum(np.abs(kap), 1.0 / 60.0)  # route.MIN_CURVE_RADIUS_M
        if model == "linear":
            w_c = 0.0004 * 1746.4 * k_abs
        elif model == "roeckl":
            w_c = np.where(
                k_abs <= 1.0 / 300.0,
                1e-3 * 650.0 * k_abs / (1.0 - 55.0 * k_abs),
                1e-3 * 500.0 * k_abs / (1.0 - 30.0 * k_abs),
            )
        else:
            raise ValueError(f"unknown curvature_model {model!r} in the dataset")
        mag = kcs * mass * 9.81 * w_c
    c_cur = np.where(np.abs(v) < 1e-9, 0.0, mag * sgn)

    zt, zb = np.maximum(z_trac, 0.0), np.maximum(z_brk, 0.0)
    if phys.get("traction_model") == "notch":
        lp = d["loco_table"].astype(np.float64)
        f_tr = np.where(can_trac > 0, _notch_force(z_trac, v, lp, float(d["v_eps"])), 0.0)
    else:
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
    inertia = mass * (d["inertia_factor"].astype(np.float64) if "inertia_factor" in d else 1.0)
    net = f_tr - f_br - r - g - c_cur + f_in - f_out
    if phys.get("standstill_model") == "breakaway":
        drive = f_tr - g + f_in - f_out
        f_static = np.minimum(zb, f_brk_max) + float(d["standstill_resistance_npkg"]) * mass
        friction = np.clip(drive + inertia * v / float(d["standstill_tau_s"]),
                           -f_static, f_static)
        net = np.where(np.abs(v) < float(d["v_brake_eps"]), drive - friction, net)
    if "tau_brk_node" in d:
        tau_b = np.where(u_b < z_brk, d["tau_brk_release_node"], d["tau_brk_node"])
    else:
        tau_b = float(d["tau_brk_s"])
    return np.concatenate([
        v, net / inertia,
        (u_b - z_brk) / tau_b,
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
        locked = np.zeros(len(t), bool)
        if "draft_gear_end_m" in d:
            e = np.abs(d["edge_dynamic"][:, :, 0]) - d["edge_static"][None, :, 1]
            locked = (e >= float(d["draft_gear_end_m"])).any(axis=1)
        clear = ~(locked | np.roll(locked, 1) | np.roll(locked, -1))
        clear[[0, -1]] = False
        for k0 in (len(t) // 4, len(t) // 2, 3 * len(t) // 4):
            ok = np.flatnonzero(clear)
            if ok.size == 0:
                continue
            k = int(ok[np.abs(ok - k0).argmin()])
            dy = rebuild_rhs(d, route, float(t[k]), y[k], k=k)
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
    if "coupler_over_ultimate" in idx:
        print(f"  F_max > knuckle yield   {int(idx.coupler_over_yield.sum())} "
              f"({idx.coupler_over_yield.mean() * 100:.0f}%)   (1.8 MN)")
        print(f"  F_max > knuckle ult.    {int(idx.coupler_over_ultimate.sum())} "
              f"({idx.coupler_over_ultimate.mean() * 100:.0f}%)   (2.9 MN: would part)")
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
