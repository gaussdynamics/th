#!/usr/bin/env python3
"""Run the four LTD benchmark trains and write the section 3.6 outputs.

Builds the batch from :mod:`simulator2.ltd_benchmark`, integrates 5000 s at
``dt = 0.01 s`` with 10 ms output, and writes one ``.npz`` per train to
``data/ltd_benchmark/`` (gitignored; regenerate with this script).

Until the participants' results arrive (Wu et al. 2018), the summary checks
the one thing the questions paper does give: the controls were generated so
the train stays within 10 % of the Fig. 7 speed board. A run far outside that
means traction, resistance or the track is wrong.

Run::

    python scripts/run_ltd_benchmark.py                 # all four, GPU if present
    python scripts/run_ltd_benchmark.py --trains 1 --t-end 600
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator2 import ltd_benchmark as lb  # noqa: E402

OUT = ROOT / "data" / "ltd_benchmark"

#: Fig. 7 speed board: (from m, to m, km/h).
SPEED_BOARD = [(0.0, 5020.0, 60.0), (5020.0, 11420.0, 80.0),
               (11420.0, 12360.0, 60.0), (12360.0, 50000.0, 80.0)]


def speed_limit_kmh(s: np.ndarray) -> np.ndarray:
    out = np.full_like(s, SPEED_BOARD[-1][2])
    for a, b, lim in SPEED_BOARD:
        out = np.where((s >= a) & (s < b), lim, out)
    return out


def summarize(train: int, o: dict) -> None:
    v_kmh = o["lead_speed_mps"] * 3.6
    s = o["lead_front_m"]
    on_board = s <= SPEED_BOARD[-1][1]
    ratio = v_kmh[on_board] / speed_limit_kmh(s[on_board])
    k = int(np.argmax(ratio))
    running = o["t_s"] > min(300.0, 0.5 * o["t_s"][-1])
    print(f"  train {train}: {s[-1] / 1000:7.2f} km at {o['t_s'][-1]:.0f} s, "
          f"top speed {v_kmh.max():5.1f} km/h, min once running {v_kmh[running].min():5.1f} km/h")
    print(f"    worst against the board: {100 * ratio[k]:.0f} % of limit at "
          f"{s[on_board][k] / 1000:.2f} km ({'within' if ratio[k] <= 1.1 else 'OUTSIDE'} the 110 % the controls allow)")
    print(f"    max tension {o['max_tension_n'].max() / 1e3:7.0f} kN at position "
          f"{int(np.argmax(o['max_tension_n'])) + 1}, max compression "
          f"{-o['max_compression_n'].min() / 1e3:7.0f} kN at position "
          f"{int(np.argmin(o['max_compression_n'])) + 1}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trains", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--t-end", type=float, default=lb.T_END_S)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--tau-notch", type=float, default=5.0,
                    help="first-order lag on the notch [s] (the benchmark leaves it open)")
    ap.add_argument("--out-dt", type=float, default=lb.OUTPUT_DT_S,
                    help="output step [s]; the integrator never steps further than this")
    ap.add_argument("--k-transition", type=float, default=lb.K_TRANSITION_N_PER_M,
                    help="draft gear transitional stiffness [N/m]")
    ap.add_argument("--start-m", type=float, default=lb.START_M,
                    help="lead front at t = 0 [m] (the results paper starts at 3 km)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()

    batch, consists = lb.build_batch(tuple(args.trains), device=args.device,
                                     tau_notch_s=args.tau_notch, t_end=args.t_end,
                                     k_transition=args.k_transition,
                                     start_m=args.start_m)
    print(f"trains {args.trains} on {args.device}, dt = {args.dt} s, out_dt = {args.out_dt} s, "
          f"k_t = {args.k_transition:g} N/m, "
          f"tau_notch = {args.tau_notch} s, start {args.start_m:g} m, {args.t_end:g} s")
    t0 = time.time()

    def progress(t):
        if t % 500 < 50 - 1e-9 or t >= args.t_end:
            print(f"  {t:6.0f} s simulated, {time.time() - t0:6.0f} s wall", flush=True)

    outs = lb.run(batch, consists, t_end=args.t_end, dt=args.dt, out_dt=args.out_dt,
                  progress=progress)
    wall = time.time() - t0

    OUT.mkdir(parents=True, exist_ok=True)
    for train, o in zip(args.trains, outs):
        path = OUT / f"train{train}{args.tag}.npz"
        np.savez_compressed(path, dt=args.dt, out_dt=args.out_dt, k_transition=args.k_transition,
                            tau_notch=args.tau_notch, start_m=args.start_m,
                            wall_s=wall, **o)
    print(f"done in {wall:.0f} s; written to {OUT.relative_to(ROOT)}")
    for train, o in zip(args.trains, outs):
        summarize(train, o)


if __name__ == "__main__":
    main()
