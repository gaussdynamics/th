#!/usr/bin/env python3
"""What does a coarser output grid lose, on a given corpus?

Reads a corpus built at a fine output spacing and decimates each trace to
candidate spacings. It is the same trajectory sampled two ways, so every
difference is sampling, not physics. Build the input with the sample cap
lifted, or every run is cut to its first 40 s::

    python scripts/build_dataset.py --physics v3 --allow-unsourced-brake \\
        --n-scenarios 64 --batch-size 8 --seed 0 --dt-out 0.01 \\
        --max-time-samples 1000000 --out data/smoke64_dt001
    python scripts/check_output_rate.py data/smoke64_dt001

Reports, per candidate spacing:

* **peak |F_cpl| missed** -- the train's peak, as the safety constraint and the
  knuckle rule use it, and each coupler's own peak where that exceeds 500 kN;
* **power above Nyquist** -- the share of the fluctuating coupler force and of
  the in-train velocity (vehicle speed minus the train mean) that the spacing
  cannot represent;
* **end-stop contact** -- episodes with a gear beyond its travel, and how many
  are shorter than the spacing and can fall between samples;
* **disk** -- compressed bytes per simulated second, re-saved at each spacing.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd

KNUCKLE_YIELD_N = 1.8e6
HIGH_FORCE_N = 500e3


def scenario_record(d, candidates) -> dict | None:
    dt0 = float(d["dt"])
    F = d["edge_dynamic"][..., 2].astype(np.float64)          # [T, E]
    if F.shape[1] == 0 or not np.abs(F).max() > 0:
        return None
    delta = d["edge_dynamic"][..., 0].astype(np.float64)
    v = d["state"][..., 1].astype(np.float64)                # [T, N]
    absF = np.abs(F)
    peak = absF.max()
    rec = {"peak_kN": peak / 1e3}

    vrel = v - v.mean(axis=1, keepdims=True)
    freq = np.fft.rfftfreq(F.shape[0], dt0)
    pF = np.abs(np.fft.rfft(F - F.mean(axis=0), axis=0)) ** 2
    pv = np.abs(np.fft.rfft(vrel, axis=0)) ** 2

    end = float(d["draft_gear_end_m"]) if "draft_gear_end_m" in d.files else np.inf
    beyond = np.abs(delta) > end
    episodes = []
    for e in range(beyond.shape[1]):
        edges = np.diff(np.concatenate([[0], beyond[:, e].astype(np.int8), [0]]))
        episodes.extend((np.where(edges == -1)[0] - np.where(edges == 1)[0]) * dt0)
    episodes = np.asarray(episodes)
    rec["contacts"] = episodes.size

    pc_full = absF.max(axis=0)
    hi = pc_full > HIGH_FORCE_N
    for dt in candidates:
        k = int(round(dt / dt0))
        dec = absF[::k]
        rec[f"miss_{dt}"] = 100.0 * (peak - dec.max()) / peak
        rec[f"yield_flip_{dt}"] = bool(peak > KNUCKLE_YIELD_N and dec.max() <= KNUCKLE_YIELD_N)
        pc_dec = dec.max(axis=0)
        rec[f"cpl_miss_{dt}"] = (float((100 * (pc_full - pc_dec) / pc_full)[hi].max())
                                 if hi.any() else np.nan)
        rec[f"cpl_miss_kN_{dt}"] = (float((pc_full - pc_dec)[hi].max()) / 1e3
                                    if hi.any() else np.nan)
        fn = 0.5 / dt
        rec[f"F_above_{dt}"] = 100.0 * pF[freq > fn].sum() / pF[1:].sum()
        rec[f"v_above_{dt}"] = (100.0 * pv[freq > fn].sum() / pv[1:].sum()
                                if pv[1:].sum() > 0 else 0.0)
        rec[f"contacts_shorter_{dt}"] = int((episodes < dt).sum())
    return rec


def bytes_per_second(d, dt) -> float:
    t = d["t"]
    k = int(round(dt / float(d["dt"])))
    arrays = {n: (d[n][::k] if d[n].ndim and d[n].shape[0] == t.size else d[n])
              for n in d.files}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.tell() / float(t[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--candidates", type=float, nargs="+", default=(0.05, 0.10, 0.25))
    ap.add_argument("--disk-sample", type=int, default=8,
                    help="scenarios re-saved for the disk estimate")
    args = ap.parse_args()
    cands = tuple(args.candidates)

    idx = pd.read_parquet(args.corpus / "index.parquet")
    rows, disk = [], {dt: [] for dt in cands}
    for _, r in idx.iterrows():
        d = np.load(args.corpus / "scenarios" / r.route_id / f"{r.scenario_id}.npz",
                    allow_pickle=True)
        for dt in cands:
            ratio = dt / float(d["dt"])          # stored dt is float32
            if abs(ratio - round(ratio)) > 1e-3:
                raise SystemExit(f"{dt} s is not a multiple of the corpus spacing {float(d['dt'])} s")
        rec = scenario_record(d, cands)
        if rec is None:
            continue
        rec.update(id=r.scenario_id, N=int(r.N), locos=int(r.loco_count),
                   control=r.control_label, duration_s=float(r.duration_s))
        rows.append(rec)
        if len(rows) <= args.disk_sample:
            for dt in cands:
                disk[dt].append(bytes_per_second(d, dt))
    df = pd.DataFrame(rows)
    hi = df[df.peak_kN > HIGH_FORCE_N / 1e3]

    print(f"{len(df)} scenarios with coupler force, N {df.N.min()}-{df.N.max()}, "
          f"peak |F| max {df.peak_kN.max():.0f} kN, {len(hi)} above "
          f"{HIGH_FORCE_N / 1e3:.0f} kN\n")
    print("Train peak |F| missed")
    for dt in cands:
        m, mh = df[f"miss_{dt}"], hi[f"miss_{dt}"]
        print(f"  {dt:>5} s  all: median {m.median():5.2f}%  worst {m.max():5.2f}%   "
              f"> {HIGH_FORCE_N / 1e3:.0f} kN: worst {mh.max():5.2f}% "
              f"({(mh * hi.peak_kN / 100).max():5.1f} kN)   yield flips "
              f"{int(df[f'yield_flip_{dt}'].sum())}")
    print(f"Own peak missed, couplers above {HIGH_FORCE_N / 1e3:.0f} kN (worst)")
    for dt in cands:
        print(f"  {dt:>5} s  {df[f'cpl_miss_{dt}'].max():5.2f}%  "
              f"({df[f'cpl_miss_kN_{dt}'].max():5.1f} kN)")
    print("Power above Nyquist, coupler force | in-train velocity")
    for dt in cands:
        print(f"  {dt:>5} s ({0.5 / dt:4.1f} Hz)  median {df[f'F_above_{dt}'].median():6.3f}%  "
              f"worst {df[f'F_above_{dt}'].max():6.3f}%  |  median "
              f"{df[f'v_above_{dt}'].median():6.3f}%  worst {df[f'v_above_{dt}'].max():6.3f}%")
    print(f"End-stop contact episodes: {int(df.contacts.sum())} in "
          f"{int((df.contacts > 0).sum())} scenarios")
    for dt in cands:
        print(f"  shorter than {dt} s: {int(df[f'contacts_shorter_{dt}'].sum())}")
    med = df.duration_s.median()
    print(f"Disk, compressed (median run {med:.0f} s)")
    for dt in cands:
        bps = float(np.mean(disk[dt]))
        print(f"  {dt:>5} s  {bps / 1e3:6.1f} kB/s  {bps * med / 1e6:5.2f} MB/scenario  "
              f"{bps * med * 1e5 / 1e9:6.0f} GB per 100k")
    df.to_csv(args.corpus / "output_rate_check.csv", index=False)


if __name__ == "__main__":
    main()
