#!/usr/bin/env python3
"""How much the benchmark results move with the integration step.

Compares runs of ``scripts/run_ltd_benchmark.py`` that differ only in ``dt``
(and the draft gear's transitional stiffness ``k_t``) against a fine-step
reference with the same ``k_t``:

* speed against location, and position at the end;
* the paper's table quantities (``compare_ltd_benchmark.ours``);
* the per-connection maximum force envelope;
* the force time series at the Table 7 positions.

A coarse run writes its output every ``dt`` (the integrator never steps
further than the output grid), so its extremes are taken over coarser
samples. The script measures that sampling effect on the reference's own
time series and reports it beside the step error.

Run, after the four runs listed in ``SIMULATOR_VALIDATION_NOTE.md``::

    python scripts/ltd_benchmark_dt_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_ltd_benchmark import METRICS, ours  # noqa: E402

RUNS = ROOT / "data" / "ltd_benchmark"

#: (tag, label, reference tag)
CASES = [
    ("_dt020_k7", "dt 0.02 s, k_t 7e8", "_dt005_k7"),
    ("_s3k", "dt 0.01 s, k_t 7e8", "_dt005_k7"),
    ("_dt020_k1", "dt 0.02 s, k_t 1e8", "_dt005_k1"),
]


def load(train: int, tag: str) -> dict | None:
    p = RUNS / f"train{train}{tag}.npz"
    return dict(np.load(p)) if p.exists() else None


def on_grid(o: dict, t: np.ndarray, key: str) -> np.ndarray:
    """Sample ``o[key]`` at times ``t`` (all runs share a 10 ms lattice)."""
    idx = np.rint(t / 0.01).astype(int)
    stride = int(round(float(o["t_s"][1] - o["t_s"][0]) / 0.01))
    return o[key][idx // stride]


def compare(train: int, tag: str, label: str, ref_tag: str) -> None:
    o, r = load(train, tag), load(train, ref_tag)
    if o is None or r is None:
        print(f"  {label}: missing ({tag} or {ref_tag})")
        return
    t = o["t_s"]
    s_o, s_r = o["lead_front_m"], on_grid(r, t, "lead_front_m")
    v_o, v_r = o["lead_speed_mps"] * 3.6, on_grid(r, t, "lead_speed_mps") * 3.6
    # Speed against location: both on the reference's location grid.
    grid = np.arange(3000.0, 50000.0, 10.0)
    keep_o = np.r_[True, np.diff(s_o) > 0]
    keep_r = np.r_[True, np.diff(r["lead_front_m"]) > 0]
    vs_o = np.interp(grid, s_o[keep_o], v_o[keep_o])
    vs_r = np.interp(grid, r["lead_front_m"][keep_r], r["lead_speed_mps"][keep_r] * 3.6)
    dv = vs_o - vs_r

    f_o = o["force_n"] / 1e3
    f_r = on_grid(r, t, "force_n") / 1e3
    df = np.abs(f_o - f_r)

    env = []
    for k in ("max_tension_n", "max_compression_n"):
        a, b = np.abs(o[k]) / 1e3, np.abs(r[k]) / 1e3
        env.append((np.abs(a - b).max(), int(np.argmax(np.abs(a - b))) + 1, np.median(np.abs(a - b))))

    # Sampling alone: the reference's force series at the coarse run's times.
    f_ref_full = r["force_n"] / 1e3
    samp = (np.abs(f_ref_full).max(axis=0) - np.abs(f_r).max(axis=0)).max()

    print(f"  {label}  vs  {ref_tag}  (wall {float(o['wall_s']):.0f} s vs {float(r['wall_s']):.0f} s)")
    print(f"    position at 5000 s: {s_o[-1] / 1e3:.3f} vs {s_r[-1] / 1e3:.3f} km  (diff {s_o[-1] - s_r[-1]:+.1f} m)")
    print(f"    speed vs location 3-50 km: max |dv| {np.abs(dv).max():.2f} km/h, RMS {np.sqrt((dv ** 2).mean()):.3f}")
    print(f"    force at the Table 7 positions: median |dF| {np.median(df):.1f} kN, p99 {np.percentile(df, 99):.1f}, max {df.max():.0f}")
    print(f"    max-tension envelope: max |d| {env[0][0]:.0f} kN at pos {env[0][1]}, median {env[0][2]:.1f}")
    print(f"    max-compression envelope: max |d| {env[1][0]:.0f} kN at pos {env[1][1]}, median {env[1][2]:.1f}")
    print(f"    (sampling the reference at this run's output step lowers its peaks by up to {samp:.0f} kN)")
    mo, mr = ours(train, o), ours(train, r)
    for key, lab, unit in METRICS:
        d = mo[key] - mr[key]
        print(f"      {lab + ' [' + unit + ']':34s} {mo[key]:8.1f}  ref {mr[key]:8.1f}  ({100 * d / mr[key]:+.1f} %)")


def main() -> None:
    for train in (1, 3):
        print(f"\nTrain {train}")
        for tag, label, ref in CASES:
            compare(train, tag, label, ref)
        # The stiffness change on its own, at the fine step.
        a, b = load(train, "_dt005_k1"), load(train, "_dt005_k7")
        if a is not None and b is not None:
            print("  k_t 1e8 vs 7e8, both at dt 0.005 s (model change, not step error):")
            ma, mb = ours(train, a), ours(train, b)
            for key, lab, unit in METRICS:
                print(f"      {lab + ' [' + unit + ']':34s} {ma[key]:8.1f}  vs {mb[key]:8.1f}  "
                      f"({100 * (ma[key] - mb[key]) / mb[key]:+.1f} %)")


if __name__ == "__main__":
    main()
