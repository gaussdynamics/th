#!/usr/bin/env python3
"""Evaluate trained first-generation surrogates (GNS-1) for thesis Chapter 9.

Scores one or more checkpoints from ``scripts/train_surrogate.py`` on every
held-out split of ``data/v2``, one-step and rolled out, against three
baselines that need no training:

* **zero-output**: predict no change (mean acceleration 0, no shape correction).
  One-step only; rolled out it is the same as *hold*.
* **hold**: freeze the start state for the whole rollout.
* **constant acceleration**: repeat the previous step's true mean acceleration
  and stretch correction. It uses one more stored state than the model is
  given (the state at ``k-1``), so it is a slightly generous baseline.

Starts are drawn with ``k0 >= 1`` (the constant-acceleration baseline needs
the previous step) and enough room for the longest horizon, so every number at
every horizon comes from the same starts.

Run::

    python scripts/eval_gns1.py data/v2 checkpoints/gns1_seed*.pt \\
        --out results/gns1_ch9.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from surrogate.data import N_EDGE_FEATURES, N_NODE_FEATURES, ScenarioSet, build_features, interp_route
from surrogate.model import ChainGNN
from surrogate.step import advance, coupler_features, edge_targets, node_targets

# "name@80" restricts a split to consists of at most 80 vehicles, the trained
# range. ood_corridor and ood_grade hold every consist size, so without the
# restriction they mix a corridor shift with a length shift.
SPLITS = ("val", "test_id", "ood_size", "ood_corridor@80", "ood_corridor",
          "ood_grade@80", "ood_grade")


def load_model(path: Path, device: str):
    ck = torch.load(path, map_location=device)
    model = ChainGNN(N_NODE_FEATURES, N_EDGE_FEATURES,
                     hidden=ck["hidden"], rounds=ck["rounds"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tgt_std = {k: torch.tensor(v, device=device) for k, v in ck["tgt_std"].items()}
    return model, tgt_std


def sample_starts(ds, horizon, n_per_block, gen):
    """(block, si, k0) triples with 1 <= k0 and k0 + horizon + 1 <= length."""
    out = []
    for key in ds.block_keys:
        blk = ds.blocks[key]
        room = blk.lengths - horizon - 2          # k0 in [1, length - horizon - 1)
        ok = torch.nonzero(room > 0).squeeze(-1)
        if ok.numel() == 0:
            continue
        m = max(1, round(n_per_block * float(ds.block_p[ds.block_keys.index(key)]) * len(ds.block_keys)))
        si = ok[torch.randint(ok.numel(), (m,), generator=gen, device=ds.device)]
        k0 = 1 + (torch.rand(m, generator=gen, device=ds.device) * room[si].float()).long()
        out.append((blk, si, k0))
    return out


@torch.no_grad()
def score(model, tgt_std, ds, horizons, n_starts, seed):
    gen = torch.Generator(device=ds.device).manual_seed(seed)
    hmax = max(horizons)
    h = ds.h
    n_blocks = len(ds.block_keys)
    starts = sample_starts(ds, hmax, max(1, n_starts // n_blocks), gen)

    one = {k: [] for k in ("a_model", "a_ca", "a_true", "c_model", "c_ca", "c_true")}
    roll = {k: [] for k in ("v_model", "v_hold", "v_ca", "d_model", "d_hold", "d_ca")}

    for blk, si, k0 in starts:
        for chunk in torch.split(torch.arange(si.numel(), device=ds.device), 256):
            s, k = si[chunk], k0[chunk]
            es, ns = blk.edge_static[s], blk.node_static[s]
            l0 = es[..., 0]
            rs, rf = ds.route_s[blk.corridor[s]], ds.route_f[blk.corridor[s]]

            st_prev, st0 = blk.state[s, k - 1], blk.state[s, k]
            d_prev, d0 = blk.edge_dyn[s, k - 1][..., 0], blk.edge_dyn[s, k][..., 0]
            a_ca = node_targets(st_prev, st0, h)
            c_ca = edge_targets(d_prev, d0, st_prev, st0, h)

            st_m, d_m = st0, d0
            st_c, d_c = st0, d0
            for j in range(hmax):
                kj = k + j
                u0, u1 = blk.u[s, kj], blk.u[s, kj + 1]
                ed = coupler_features(d_m, st_m[..., 1], es)
                route = interp_route(rs, rf, st_m[..., 0])
                node, edge = build_features(st_m, ed, ns, es, route, u0, u1, ds.norm)
                a_hat, c_hat = model(node, edge)
                a_hat, c_hat = a_hat * tgt_std["abar"], c_hat * tgt_std["dcorr"]

                st_t, d_t = blk.state[s, kj + 1], blk.edge_dyn[s, kj + 1][..., 0]
                if j == 0:
                    one["a_model"].append((a_hat - node_targets(st0, st_t, h)).flatten())
                    one["a_ca"].append((a_ca - node_targets(st0, st_t, h)).flatten())
                    one["a_true"].append(node_targets(st0, st_t, h).flatten())
                    c_true = edge_targets(d0, d_t, st0, st_t, h)
                    one["c_model"].append((c_hat - c_true).flatten())
                    one["c_ca"].append((c_ca - c_true).flatten())
                    one["c_true"].append(c_true.flatten())

                st_m, d_m = advance(st_m, d_m, a_hat, c_hat, u0, u1, l0,
                                    tau_brk=ds.tau_brk, tau_trac=ds.tau_trac, h=h)
                st_c, d_c = advance(st_c, d_c, a_ca, c_ca, u0, u1, l0,
                                    tau_brk=ds.tau_brk, tau_trac=ds.tau_trac, h=h)
                if (j + 1) in horizons:
                    roll["v_model"].append((j + 1, (st_m[..., 1] - st_t[..., 1]).abs().amax(-1)))
                    roll["v_hold"].append((j + 1, (st0[..., 1] - st_t[..., 1]).abs().amax(-1)))
                    roll["v_ca"].append((j + 1, (st_c[..., 1] - st_t[..., 1]).abs().amax(-1)))
                    roll["d_model"].append((j + 1, (d_m - d_t).abs().amax(-1)))
                    roll["d_hold"].append((j + 1, (d0 - d_t).abs().amax(-1)))
                    roll["d_ca"].append((j + 1, (d_c - d_t).abs().amax(-1)))

    rms = lambda xs: float(torch.cat(xs).pow(2).mean().sqrt())
    res = {
        "one_step": {
            "rmse_abar_model": rms(one["a_model"]),
            "rmse_abar_zero": rms(one["a_true"]),
            "rmse_abar_constacc": rms(one["a_ca"]),
            "rmse_corr_mm_model": 1e3 * rms(one["c_model"]),
            "rmse_corr_mm_zero": 1e3 * rms(one["c_true"]),
            "rmse_corr_mm_constacc": 1e3 * rms(one["c_ca"]),
        },
        "rollout": {},
    }
    for hz in horizons:
        r = {}
        for key, vals in roll.items():
            x = torch.cat([v for (hh, v) in vals if hh == hz])
            scale = 1e3 if key.startswith("d_") else 1.0
            r[key + "_med"] = scale * float(x.median())
        res["rollout"][hz] = r
    res["n_starts"] = int(sum(si.numel() for _, si, _ in starts))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 50, 100, 200])
    ap.add_argument("--n-starts", type=int, default=1024)
    ap.add_argument("--budget-gib", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = {"splits": {}, "checkpoints": [str(p) for p in args.checkpoints]}
    import pandas as pd
    sizes = sorted(pd.read_parquet(args.root / "index.parquet").N.unique().tolist())
    for split in args.splits:
        name, _, cap = split.partition("@")
        n_values = [n for n in sizes if n <= int(cap)] if cap else None
        ds = ScenarioSet(args.root, name, n_values=n_values, budget_gib=args.budget_gib,
                         device=args.device, seed=args.seed)
        lo, hi = ds.n_range
        corridors = sorted(ds.rows.route_id.unique().tolist())
        info = {"n_scenarios": len(ds.rows), "N_range": [lo, hi],
                "corridors": corridors, "per_checkpoint": []}
        print(f"\n{split}: {len(ds.rows)} scenarios, N {lo}-{hi}, {corridors}")
        for ck in args.checkpoints:
            model, tgt_std = load_model(ck, args.device)
            r = score(model, tgt_std, ds, set(args.horizons), args.n_starts, args.seed)
            info["per_checkpoint"].append(r)
            o = r["one_step"]
            print(f"  {ck.name}: abar {o['rmse_abar_model']:.4f} (zero {o['rmse_abar_zero']:.4f},"
                  f" ca {o['rmse_abar_constacc']:.4f})  corr {o['rmse_corr_mm_model']:.3f} mm"
                  f" (zero {o['rmse_corr_mm_zero']:.3f}, ca {o['rmse_corr_mm_constacc']:.3f})")
            for hz in sorted(args.horizons):
                q = r["rollout"][hz]
                print(f"    {hz:4d}  v {q['v_model_med']:.4f} hold {q['v_hold_med']:.4f}"
                      f" ca {q['v_ca_med']:.4f}   d {q['d_model_med']:.2f} hold"
                      f" {q['d_hold_med']:.2f} ca {q['d_ca_med']:.2f} mm")
        out["splits"][split] = info
        del ds
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
