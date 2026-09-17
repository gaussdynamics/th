#!/usr/bin/env python3
"""Why does coupler shape stop extrapolating over long rollouts?

One-step shape error barely degrades on ood_size (RMSE 0.47 -> 0.55 mm), yet at
200 steps it goes 5.8 -> 41.0 mm. So this is compounding, not capacity, and the
question is what *kind* of compounding:

bias
    A systematic offset in the predicted correction integrates linearly in the
    rollout: error ~ n_steps.
random walk
    Independent per-step errors integrate as sqrt(n_steps).
chain accumulation
    Error grows with position along the consist, i.e. the model cannot
    coordinate the two ends of a long train.

Each points at a different fix, so it is worth separating them before changing
the model.

Run::

    python scripts/diagnose_shape.py data/v2 checkpoints/gns_push3.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from surrogate.data import N_EDGE_FEATURES, N_NODE_FEATURES, ScenarioSet, build_features, interp_route
from surrogate.model import ChainGNN
from surrogate.step import advance, coupler_features


@torch.no_grad()
def signed_rollout(model, ds, blk, si, k0, horizon, tgt_std):
    """Signed delta error per coupler per step: [horizon, B, N-1]."""
    es, ns = blk.edge_static[si], blk.node_static[si]
    l0 = es[..., 0]
    rs, rf = ds.route_s[blk.corridor[si]], ds.route_f[blk.corridor[si]]
    st = blk.state[si, k0]
    delta = blk.edge_dyn[si, k0][..., 0]
    out = []
    for j in range(horizon):
        kj = k0 + j
        u0, u1 = blk.u[si, kj], blk.u[si, kj + 1]
        ed = coupler_features(delta, st[..., 1], es)
        route = interp_route(rs, rf, st[..., 0])
        node, edge = build_features(st, ed, ns, es, route, u0, u1, ds.norm)
        a, d = model(node, edge)
        st, delta = advance(st, delta, a * tgt_std["abar"], d * tgt_std["dcorr"],
                            u0, u1, l0, tau_brk=ds.tau_brk,
                            tau_trac=ds.tau_trac, h=ds.h)
        out.append(delta - blk.edge_dyn[si, kj + 1][..., 0])
    return torch.stack(out)


def report(name, err, horizon):
    """err: [horizon, B, N-1] signed, metres."""
    e = err.double()
    print(f"\n== {name} ==   couplers={e.shape[-1]}")
    print(f"  {'step':>5s} {'mean (bias)':>13s} {'std':>10s} {'max|.|':>10s}  "
          f"{'bias/total':>10s}")
    for j in (0, 9, 49, 99, horizon - 1):
        if j >= e.shape[0]:
            continue
        v = e[j]
        m, sd = float(v.mean()), float(v.std())
        tot = (m ** 2 + sd ** 2) ** 0.5
        print(f"  {j+1:5d} {m*1e3:13.4f} {sd*1e3:10.4f} "
              f"{float(v.abs().max())*1e3:10.3f}  {abs(m)/max(tot,1e-12):10.1%}")

    # growth law: fit log|rms| vs log step
    rms = e.pow(2).mean(dim=(1, 2)).sqrt().cpu().numpy()
    n = np.arange(1, len(rms) + 1)
    lo = max(4, len(rms) // 10)
    p = np.polyfit(np.log(n[lo:]), np.log(rms[lo:]), 1)[0]
    print(f"  growth exponent: rms ~ n^{p:.2f}   "
          f"(0.5 = random walk, 1.0 = systematic bias)")

    # along-chain structure, averaged over the last steps
    tail = e[-1].abs().mean(dim=0).cpu().numpy()
    q = len(tail) // 4
    if q:
        quarters = [tail[i * q:(i + 1) * q].mean() * 1e3 for i in range(4)]
        print("  mean |err| by quarter along the consist [mm]: "
              + "  ".join(f"{v:.2f}" for v in quarters))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--batch", type=int, default=96)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = ChainGNN(N_NODE_FEATURES, N_EDGE_FEATURES,
                     hidden=ck["hidden"], rounds=ck["rounds"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    tgt_std = {k: torch.tensor(v, device=dev) for k, v in ck["tgt_std"].items()}

    for split in ("val", "ood_size"):
        ds = ScenarioSet(args.root, split, budget_gib=1.5, device=dev, seed=0)
        gen = torch.Generator(device=dev).manual_seed(0)
        got = ds.sample_starts(args.batch, gen, args.horizon + 1)
        if got is None:
            print(f"{split}: no scenario long enough for {args.horizon} steps")
            continue
        blk, si, k0 = got
        err = signed_rollout(model, ds, blk, si, k0, args.horizon, tgt_std)
        report(f"{split}  N={blk.n}", err, args.horizon)
        coupling_report(model, ds, blk, si, k0, args.horizon, tgt_std,
                        f"{split}  N={blk.n}")




# ---------------------------------------------------------------------------
# coupling: is shape error caused by velocity error, or its own?
# ---------------------------------------------------------------------------
@torch.no_grad()
def coupled_rollout(model, ds, blk, si, k0, horizon, tgt_std, *, oracle):
    """Roll out with one channel forced to the truth at every step.

    ``oracle='none'``  both channels run on the model (the real rollout).
    ``oracle='v'``     node state is replaced by the truth each step, and the
                       stretch trapezoid uses the true closing rates, so the
                       only model error left in ``delta`` is its own correction.
    ``oracle='delta'`` the reverse: stretch is replaced by the truth, isolating
                       how much of the velocity error comes from bad geometry.

    ``delta`` integrates ``v_j - v_{j+1}``, so velocity error feeds shape
    directly. Forcing one side answers which way the causation runs, which
    correlating errors across checkpoints cannot.
    """
    es, ns = blk.edge_static[si], blk.node_static[si]
    l0 = es[..., 0]
    rs, rf = ds.route_s[blk.corridor[si]], ds.route_f[blk.corridor[si]]
    st = blk.state[si, k0]
    delta = blk.edge_dyn[si, k0][..., 0]
    v_err, d_err = [], []

    for j in range(horizon):
        kj = k0 + j
        u0, u1 = blk.u[si, kj], blk.u[si, kj + 1]
        ed = coupler_features(delta, st[..., 1], es)
        route = interp_route(rs, rf, st[..., 0])
        node, edge = build_features(st, ed, ns, es, route, u0, u1, ds.norm)
        a, d = model(node, edge)

        st_true = blk.state[si, kj + 1]
        d_true = blk.edge_dyn[si, kj + 1][..., 0]

        if oracle == "v":
            v_k = blk.state[si, kj][..., 1]
            ddot = v_k[..., :-1] - v_k[..., 1:]
            ddot1 = st_true[..., 1][..., :-1] - st_true[..., 1][..., 1:]
            delta = delta + ds.h * 0.5 * (ddot + ddot1) + d * tgt_std["dcorr"]
            st = st_true
        else:
            st, delta = advance(st, delta, a * tgt_std["abar"],
                                d * tgt_std["dcorr"], u0, u1, l0,
                                tau_brk=ds.tau_brk, tau_trac=ds.tau_trac, h=ds.h)
            if oracle == "delta":
                delta = d_true

        v_err.append((st[..., 1] - st_true[..., 1]).abs().amax(-1))
        d_err.append((delta - d_true).abs().amax(-1))
    return torch.stack(v_err), torch.stack(d_err)


def coupling_report(model, ds, blk, si, k0, horizon, tgt_std, label):
    print(f"\n== coupling: {label} ==")
    print(f"  {'oracle':>10s} {'max|dv| @200 [m/s]':>20s} {'max|ddelta| @200 [mm]':>23s}")
    out = {}
    for oracle in ("none", "v", "delta"):
        v, d = coupled_rollout(model, ds, blk, si, k0, horizon, tgt_std,
                               oracle=oracle)
        out[oracle] = (float(v[-1].median()), float(d[-1].median()) * 1e3)
        vs = "-- forced --" if oracle == "v" else f"{out[oracle][0]:20.5f}"
        dsx = "-- forced --" if oracle == "delta" else f"{out[oracle][1]:23.3f}"
        print(f"  {oracle:>10s} {vs:>20s} {dsx:>23s}")
    base_d = out["none"][1]
    orac_d = out["v"][1]
    print(f"\n  shape error with a perfect velocity field: {orac_d:.2f} mm "
          f"vs {base_d:.2f} mm  ->  {1 - orac_d/max(base_d,1e-9):.0%} of it is "
          f"caused by velocity error")
    return out

if __name__ == "__main__":
    main()
