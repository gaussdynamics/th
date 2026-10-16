#!/usr/bin/env python3
"""Wall-clock cost of a GNS-1 rollout, for comparison with the GPU integrator.

Times the real rollout code (``surrogate.rollout.rollout``: feature building,
route lookup, network, state update, and its error bookkeeping) advancing ``B``
scenarios of ``N`` vehicles for 480 output steps, which is 120 s of simulated
time at 0.25 s: the workload of ``tab:gpu_throughput`` in Chapter 7. Scenarios
come from ``ood_size`` at the requested ``N`` and are repeated to fill ``B``.
Plain eager PyTorch with no CUDA graph, so the integrator it is compared with
has an advantage the surrogate does not.

Run::

    python scripts/time_gns1.py data/v2 checkpoints/gns1_seed0.pt --b 1 64 256 --n 130
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

from surrogate.data import ScenarioSet
from surrogate.rollout import rollout
from scripts.eval_gns1 import load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--b", type=int, nargs="+", default=[1, 64, 256])
    ap.add_argument("--n", type=int, default=130)
    ap.add_argument("--steps", type=int, default=480)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = ScenarioSet(args.root, "ood_size", n_values=[args.n], budget_gib=2.0,
                     device=args.device)
    blk = ds.blocks[args.n]
    model, tgt_std = load_model(args.checkpoint, args.device)
    ok = torch.nonzero(blk.lengths > args.steps + 1).squeeze(-1)
    kw = dict(route_s=ds.route_s, route_f=ds.route_f, tgt_std=tgt_std)
    for b in args.b:
        si = ok[torch.arange(b, device=args.device) % ok.numel()]
        k0 = torch.zeros(b, dtype=torch.long, device=args.device)
        rollout(model, blk, si, k0, 10, **kw)          # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rollout(model, blk, si, k0, args.steps, **kw)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        print(f"B={b:4d} N={args.n}  {t:.2f} s for {args.steps} steps"
              f"  ({1e3 * t / args.steps:.2f} ms/step, {1e3 * t / b:.1f} ms/scenario)")


if __name__ == "__main__":
    main()
