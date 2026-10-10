#!/usr/bin/env python3
"""Aggregate per-seed ``eval_gns1.py`` results into the Chapter 9 tables.

Reads ``results/gns1_ch9/eval_seed*.json`` and prints, per split, the mean
over seeds of each quantity with its min-max range, and the gain of the model
over each baseline (ratio of seed means). Writes the merged summary to
``results/gns1_ch9/summary.json``.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "gns1_ch9"


def main() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(ROOT.glob("eval_seed*.json"))]
    if not runs:
        sys.exit("no eval_seed*.json files")
    out = {"n_seeds": len(runs), "splits": {}}
    for split in runs[0]["splits"]:
        per = [r["splits"][split]["per_checkpoint"][0] for r in runs]
        info = runs[0]["splits"][split]
        agg = {"n_scenarios": [r["splits"][split]["n_scenarios"] for r in runs],
               "N_range": info["N_range"], "corridors": info["corridors"],
               "one_step": {}, "rollout": {}}
        for k in per[0]["one_step"]:
            xs = [p["one_step"][k] for p in per]
            agg["one_step"][k] = {"mean": st.mean(xs), "min": min(xs), "max": max(xs)}
        for hz in per[0]["rollout"]:
            agg["rollout"][hz] = {}
            for k in per[0]["rollout"][hz]:
                xs = [p["rollout"][hz][k] for p in per]
                agg["rollout"][hz][k] = {"mean": st.mean(xs), "min": min(xs), "max": max(xs)}
        out["splits"][split] = agg

        o = agg["one_step"]
        m = lambda k: o[k]["mean"]
        print(f"\n== {split}  N {agg['N_range']}  scenarios {agg['n_scenarios']}")
        print(f"  abar RMSE  model {m('rmse_abar_model'):.4f} [{o['rmse_abar_model']['min']:.4f}-{o['rmse_abar_model']['max']:.4f}]"
              f"  zero {m('rmse_abar_zero'):.4f} ({m('rmse_abar_zero')/m('rmse_abar_model'):.1f}x)"
              f"  ca {m('rmse_abar_constacc'):.4f} ({m('rmse_abar_constacc')/m('rmse_abar_model'):.1f}x)")
        print(f"  corr RMSE  model {m('rmse_corr_mm_model'):.3f} [{o['rmse_corr_mm_model']['min']:.3f}-{o['rmse_corr_mm_model']['max']:.3f}] mm"
              f"  zero {m('rmse_corr_mm_zero'):.3f} ({m('rmse_corr_mm_zero')/m('rmse_corr_mm_model'):.1f}x)"
              f"  ca {m('rmse_corr_mm_constacc'):.3f} ({m('rmse_corr_mm_constacc')/m('rmse_corr_mm_model'):.1f}x)")
        for hz, r in agg["rollout"].items():
            g = lambda k: r[k]["mean"]
            print(f"  {int(hz):4d}  v {g('v_model_med'):.4f} [{r['v_model_med']['min']:.4f}-{r['v_model_med']['max']:.4f}]"
                  f" hold {g('v_hold_med'):.4f} ({g('v_hold_med')/g('v_model_med'):.1f}x)"
                  f" ca {g('v_ca_med'):.4f} ({g('v_ca_med')/g('v_model_med'):.2f}x)"
                  f" | d {g('d_model_med'):.2f} [{r['d_model_med']['min']:.2f}-{r['d_model_med']['max']:.2f}]"
                  f" hold {g('d_hold_med'):.2f} ({g('d_hold_med')/g('d_model_med'):.1f}x)"
                  f" ca {g('d_ca_med'):.1f}")
    (ROOT / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
