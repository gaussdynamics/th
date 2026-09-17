#!/usr/bin/env python3
"""Smoke training run: one consist size, one-step prediction.

The smallest thing that exercises the whole path -- load, featurize, message
pass, score -- so the unknowns surface while they are cheap. It is not the
Chapter 6 experiment and does not try to be: fixed ``N``, a few hundred
scenarios, no rollout, no OOD splits.

Every metric is reported beside what predicting **nothing** would score. That
bar is not trivially low: over 0.25 s a freight consist barely changes state,
so a step predictor has to beat a very strong "no change" default before it has
learned anything at all. On the full train split the zero-output baseline is
0.0176 m/s median of ``max |dv|`` over the consist
(``SURROGATE_FORMULATION_NOTE.md``).

Run::

    python scripts/train_surrogate.py data/v2 --n-vehicles 42 --steps 3000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from surrogate.data import N_EDGE_FEATURES, N_NODE_FEATURES, ScenarioSet
from surrogate.model import ChainGNN


@torch.no_grad()
def evaluate(model, ds, tgt_std, batch_size: int) -> dict[str, float]:
    model.eval()
    se_a = sz_a = n_a = 0.0
    se_d = sz_d = n_d = 0.0
    dv_model, dv_zero, dd_model, dd_zero = [], [], [], []

    for b in ds.iter_all(batch_size, tgt_std):
        a_hat, d_hat = model(b.node, b.edge)
        a_hat = a_hat * tgt_std["abar"]
        d_hat = d_hat * tgt_std["dcorr"]
        a_true = b.abar * tgt_std["abar"]
        d_true = b.dcorr * tgt_std["dcorr"]

        se_a += float(((a_hat - a_true) ** 2).sum())
        sz_a += float((a_true ** 2).sum())
        n_a += a_true.numel()
        se_d += float(((d_hat - d_true) ** 2).sum())
        sz_d += float((d_true ** 2).sum())
        n_d += d_true.numel()

        # max over the consist -- the statistic the formulation note uses
        dv_model.append((a_hat - a_true).abs().max(dim=1).values * ds.h)
        dv_zero.append(a_true.abs().max(dim=1).values * ds.h)
        dd_model.append((d_hat - d_true).abs().max(dim=1).values)
        dd_zero.append(d_true.abs().max(dim=1).values)

    model.train()
    dv_model, dv_zero = torch.cat(dv_model), torch.cat(dv_zero)
    dd_model, dd_zero = torch.cat(dd_model), torch.cat(dd_zero)
    return {
        "rmse_abar": (se_a / n_a) ** 0.5,
        "rmse_abar_zero": (sz_a / n_a) ** 0.5,
        "rmse_dcorr": (se_d / n_d) ** 0.5,
        "rmse_dcorr_zero": (sz_d / n_d) ** 0.5,
        "dv_med": float(dv_model.median()),
        "dv_p99": float(torch.quantile(dv_model.float(), 0.99)),
        "dv_med_zero": float(dv_zero.median()),
        "dv_p99_zero": float(torch.quantile(dv_zero.float(), 0.99)),
        # The shape correction is sparse: 64% of transitions have every
        # coupler under 50 um, and the signal lives entirely in the tail
        # (p90 4.2 mm, p99 23.5 mm). A median statistic here measures the
        # quiet majority, where predicting exactly zero is unbeatable and
        # beating it is not the point.
        "dd_p90": float(torch.quantile(dd_model.float(), 0.90)),
        "dd_p99": float(torch.quantile(dd_model.float(), 0.99)),
        "dd_p90_zero": float(torch.quantile(dd_zero.float(), 0.90)),
        "dd_p99_zero": float(torch.quantile(dd_zero.float(), 0.99)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--n-vehicles", type=int, nargs="+", default=None,
                    help="restrict to these consist sizes (default: all in the split)")
    ap.add_argument("--train-scenarios", type=int, default=None)
    ap.add_argument("--budget-gib", type=float, default=6.0,
                    help="device memory for the loaded train split")
    ap.add_argument("--val-budget-gib", type=float, default=2.0)
    ap.add_argument("--ood", action="store_true",
                    help="also score the ood_size split (N 120-150, disjoint from train)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=2048)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t_load = time.time()
    train = ScenarioSet(args.root, "train", n_values=args.n_vehicles,
                        max_scenarios=args.train_scenarios,
                        budget_gib=args.budget_gib, device=args.device,
                        seed=args.seed)
    val = ScenarioSet(args.root, "val", n_values=args.n_vehicles,
                      budget_gib=args.val_budget_gib, device=args.device,
                      seed=args.seed)
    for name, ds in (("train", train), ("val", val)):
        lo, hi = ds.n_range
        print(f"  {name:9s} {len(ds.rows):5d} scenarios  {len(ds):>10,} transitions  "
              f"N {lo}-{hi} in {len(ds.blocks)} buckets  {ds.gib:.2f} GiB")
    print(f"  loaded in {time.time()-t_load:.1f}s   dt={train.h}s  "
          f"tau_brk={train.tau_brk}s  tau_trac={train.tau_trac}s")

    tgt_std = train.target_std()
    print(f"target std  abar {float(tgt_std['abar']):.4f} m/s^2   "
          f"ddelta_corr {float(tgt_std['dcorr'])*1e3:.3f} mm")

    model = ChainGNN(N_NODE_FEATURES, N_EDGE_FEATURES,
                     hidden=args.hidden, rounds=args.rounds).to(args.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"ChainGNN  hidden={args.hidden} rounds={args.rounds}  "
          f"{n_par:,} parameters\n")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    run_loss, t0 = 0.0, time.time()
    for step in range(1, args.steps + 1):
        b = train.sample(args.batch_size, gen, tgt_std)
        a_hat, d_hat = model(b.node, b.edge)
        # Both targets are unit-variance here, so a plain sum weights them
        # equally in normalized space rather than by physical magnitude.
        loss = ((a_hat - b.abar) ** 2).mean() + ((d_hat - b.dcorr) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += float(loss.detach())

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, val, tgt_std, args.eval_batch_size)
            print(f"step {step:5d}  loss {run_loss/args.eval_every:.4f}   "
                  f"a {m['rmse_abar']:.4f}/{m['rmse_abar_zero']:.4f} m/s^2   "
                  f"ddelta {m['rmse_dcorr']*1e3:6.3f}/{m['rmse_dcorr_zero']*1e3:.3f} mm"
                  f"   (model/zero)")
            run_loss = 0.0

    print(f"\ndone in {time.time()-t0:.0f}s")

    sets = [("val", val)]
    if args.ood:
        # Consist sizes held out entirely: train covers N 10-80, ood_size is
        # N 120-150 with zero overlap. Message passing is what is supposed to
        # make that work, so this is the universality claim in miniature.
        ood = ScenarioSet(args.root, "ood_size", budget_gib=args.val_budget_gib,
                          device=args.device, seed=args.seed)
        lo, hi = ood.n_range
        print(f"  ood_size  {len(ood.rows)} scenarios  {len(ood):,} transitions  "
              f"N {lo}-{hi}  {ood.gib:.2f} GiB")
        sets.append(("ood_size", ood))

    for name, ds in sets:
        m = evaluate(model, ds, tgt_std, args.eval_batch_size)
        lo, hi = ds.n_range
        print(f"\n  {name}  (N {lo}-{hi})")
        print(f"  {'metric':34s} {'model':>11s} {'zero-output':>12s} {'gain':>7s}")
        for label, got, zero in (
            ("velocity  RMSE [m/s^2]", m["rmse_abar"], m["rmse_abar_zero"]),
            ("velocity  max|dv| median [m/s]", m["dv_med"], m["dv_med_zero"]),
            ("velocity  max|dv| p99 [m/s]", m["dv_p99"], m["dv_p99_zero"]),
            ("shape     RMSE [mm]", m["rmse_dcorr"] * 1e3, m["rmse_dcorr_zero"] * 1e3),
            ("shape     max|ddelta| p90 [mm]", m["dd_p90"] * 1e3, m["dd_p90_zero"] * 1e3),
            ("shape     max|ddelta| p99 [mm]", m["dd_p99"] * 1e3, m["dd_p99_zero"] * 1e3),
        ):
            print(f"  {label:34s} {got:11.5f} {zero:12.5f} {zero/max(got,1e-12):6.1f}x")


if __name__ == "__main__":
    main()
