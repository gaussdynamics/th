"""Plot one scenario from a built dataset.

The dataset is arrays on disk; this is the eyeball on top of it. Pick a
scenario by id, or let a filter over ``index.parquet`` pick one for you --
"the worst coupler force in the test split" is a query, not a file name.

    python scripts/view_scenario.py --root data/v2 \
        --scenario-id route_line0_49km_run000001

    python scripts/view_scenario.py --query "split=='test_id'" \
        --sort-by F_max --out media/worst_coupler.png

    python scripts/view_scenario.py --query "adhesion=='low'" --list 20

Five panels, sharing an x axis of time (or of lead chainage under
``--vs-distance``): speed against the route speed limit, coupler force,
commands and their actuator states, the route the run is on (grade and
curvature, sampled at the lead vehicle), and coupler displacement.

Everything is read from the scenario ``.npz`` plus its corridor file under
``routes/``; nothing is recomputed, so what you see is what a dataloader
would hand the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Columns worth printing next to the picture. Ordered, not alphabetical.
_SUMMARY_FIELDS = [
    "scenario_id", "split", "route_id", "consist_label", "control_label",
    "N", "T", "total_mass_kg", "consist_len_m", "loco_count",
    "distributed_power", "adhesion", "slack_state", "curvature_model",
    "k_curv_scale", "v0_mps", "x_lead_m", "duration_s", "dt_s",
    "distance_travelled_m", "v_mean_mps", "v_max_mps", "v_min_mps",
    "E_trip", "F_max", "v_over_limit_max_mps", "frac_time_over_limit",
    "grade_max_pct", "grade_rms_pct", "seed",
]

_LIST_COLUMNS = [
    "scenario_id", "split", "route_id", "consist_label", "control_label",
    "adhesion", "F_max", "v_max_mps", "v_over_limit_max_mps", "E_trip",
]


def select_row(idx, args):
    """The one row of ``index.parquet`` to plot, and the pool it came from."""
    pool = idx
    if args.scenario_id:
        pool = pool[pool.scenario_id == args.scenario_id]
        if pool.empty:
            raise SystemExit(f"no scenario with id {args.scenario_id!r}")
    if args.split:
        pool = pool[pool.split == args.split]
    if args.route_id:
        pool = pool[pool.route_id == args.route_id]
    if args.query:
        pool = pool.query(args.query)
    if pool.empty:
        raise SystemExit("no scenario matches that filter")

    if args.sort_by:
        if args.sort_by not in pool.columns:
            raise SystemExit(f"--sort-by {args.sort_by!r} is not a column; columns "
                             f"are: {', '.join(pool.columns)}")
        pool = pool.sort_values(args.sort_by, ascending=args.ascending)
    elif args.random:
        pool = pool.sample(frac=1.0, random_state=args.seed)

    if args.rank >= len(pool):
        raise SystemExit(f"--rank {args.rank} but only {len(pool)} scenarios match")
    return pool.iloc[args.rank], pool


def load_scenario(root: Path, row):
    """Scenario arrays plus the corridor file they reference."""
    path = root / "scenarios" / row.route_id / f"{row.scenario_id}.npz"
    if not path.exists():
        raise SystemExit(f"missing scenario file: {path}")
    d = dict(np.load(path, allow_pickle=True))
    route = dict(np.load(root / "routes" / f"{row.route_id}.npz", allow_pickle=True))
    return d, route, path


def at_lead(route: dict, field: str, x_lead: np.ndarray) -> np.ndarray:
    """A route field sampled where the lead vehicle actually is.

    The field is stored over arc length, not time -- this is the same
    interpolation the RHS does, at the positions the run actually reached.
    """
    return np.interp(x_lead, route["route_s"], route[field])


def plot_scenario(d: dict, route: dict, row, vs_distance: bool):
    import matplotlib.pyplot as plt

    t = d["t"].astype(np.float64)
    state = d["state"].astype(np.float64)      # [T, N, 4] = x, v, z_brk, z_trac
    x, v = state[:, :, 0], state[:, :, 1]
    z_brk, z_trac = state[:, :, 2], state[:, :, 3]
    f_cpl = d["edge_dynamic"][:, :, 2] / 1e3   # kN, + draft / - buff
    delta = d["edge_dynamic"][:, :, 0]         # m
    can_trac = d["node_static"][:, 4] > 0.5
    n = state.shape[1]

    x_lead = x[:, 0]
    axis = x_lead / 1e3 if vs_distance else t
    axis_label = "lead chainage (km)" if vs_distance else "time (s)"

    fig, axes = plt.subplots(5, 1, figsize=(11.5, 12.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.2, 1.0, 1.0, 0.9]})

    # -- 1. speed against the limit ----------------------------------------
    ax = axes[0]
    ax.plot(axis, v, color="0.75", lw=0.5, alpha=0.7)
    ax.plot(axis, v[:, 0], color="C0", lw=1.6, label="lead")
    ax.plot(axis, v[:, -1], color="C1", lw=1.0, label="rear")
    ax.plot(axis, at_lead(route, "route_vmax", x_lead), color="C3", lw=1.2, ls="--",
            label="speed limit (at lead)")
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_ylabel("speed (m/s)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    # -- 2. coupler force --------------------------------------------------
    ax = axes[1]
    ax.plot(axis, f_cpl, color="0.8", lw=0.4, alpha=0.8)
    ax.plot(axis, f_cpl.max(axis=1), color="C3", lw=1.3, label="max (draft)")
    ax.plot(axis, f_cpl.min(axis=1), color="C0", lw=1.3, label="min (buff)")
    ax.axhline(0.0, color="0.4", lw=0.6)
    peak = float(np.abs(f_cpl).max())
    k_peak = int(np.unravel_index(int(np.abs(f_cpl).argmax()), f_cpl.shape)[1])
    ax.set_ylabel("coupler force (kN)")
    ax.legend(loc="upper right", fontsize=8, ncol=2, title_fontsize=8,
              title=f"peak |F| = {peak:,.0f} kN at coupler {k_peak}")

    # -- 3. commands and the actuator states they drive --------------------
    # The commands are absolute forces (N), not notch fractions, so plot train
    # totals in kN -- directly comparable to the coupler forces above -- and put
    # the fraction of installed capability in the legend title. Solid is the
    # command, dashed the first-order actuator state that actually acts.
    ax = axes[2]
    u_trac_kn = d["u_trac"].sum(axis=1) / 1e3
    u_brk_kn = d["u_brk"].sum(axis=1) / 1e3
    ax.plot(axis, u_trac_kn, color="C2", lw=1.2, label="u_trac")
    ax.plot(axis, z_trac.sum(axis=1) / 1e3, color="C2", lw=1.0, ls="--", label="z_trac")
    ax.plot(axis, u_brk_kn, color="C3", lw=1.2, label="u_brk")
    ax.plot(axis, z_brk.sum(axis=1) / 1e3, color="C3", lw=1.0, ls="--", label="z_brk")
    ax.set_ylabel("train total (kN)")
    trac_cap = float(d["node_static"][can_trac, 5].sum()) or np.nan
    brk_cap = float(d["node_static"][:, 6].sum()) or np.nan
    ax.legend(loc="upper right", fontsize=8, ncol=2, title_fontsize=8,
              title=f"peak {u_trac_kn.max() * 1e3 / trac_cap:.0%} of traction, "
                    f"{u_brk_kn.max() * 1e3 / brk_cap:.0%} of brake capability")

    # -- 4. the route under the train --------------------------------------
    ax = axes[3]
    sin_theta = at_lead(route, "route_sin_theta", x_lead)
    grade_pct = 100.0 * sin_theta / np.sqrt(np.clip(1.0 - sin_theta ** 2, 1e-12, None))
    ax.fill_between(axis, 0.0, grade_pct, color="C4", alpha=0.3, lw=0)
    handles = ax.plot(axis, grade_pct, color="C4", lw=1.1, label="grade (%)")
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_ylabel("grade (%)", color="C4")
    ax_k = ax.twinx()
    handles += ax_k.plot(axis, at_lead(route, "route_kappa", x_lead) * 1e3,
                         color="C5", lw=0.9, label="curvature (1/km)")
    ax_k.set_ylabel("curvature (1/km)", color="C5")
    ax_k.grid(False)
    ax.legend(handles, [h.get_label() for h in handles], loc="upper right",
              fontsize=8, ncol=2)

    # -- 5. coupler displacement -------------------------------------------
    ax = axes[4]
    ax.plot(axis, delta, color="0.8", lw=0.4, alpha=0.8)
    envelope = ax.plot(axis, delta.max(axis=1), color="C3", lw=1.2, label="max")
    envelope += ax.plot(axis, delta.min(axis=1), color="C0", lw=1.2, label="min")
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_ylabel("coupler travel (m)")
    ax.set_xlabel(axis_label)
    # The train-length change is the sum of N-1 couplers, so it lives an order
    # of magnitude above the per-coupler traces -- its own axis or neither reads.
    ax_len = ax.twinx()
    stretch = (x[:, 0] - x[:, -1]) - (x[0, 0] - x[0, -1])
    ax_len.plot(axis, stretch, color="C6", lw=1.0, label="train length change")
    ax_len.set_ylabel("length change (m)", color="C6")
    ax_len.grid(False)
    handles = envelope + ax_len.get_lines()[:1]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper right",
              fontsize=8, ncol=3)

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.margins(x=0.01)

    subtitle = _wrap_facts([
        f"{row.consist_label} / {row.control_label} / {row.adhesion} adhesion / "
        f"{row.slack_state} slack",
        f"N={n}, {row.total_mass_kg / 1e6:.2f} kt, {row.consist_len_m:.0f} m",
        f"curvature {row.curvature_model} k={row.k_curv_scale:g}",
        f"peak |F_cpl| {peak:,.0f} kN, "
        f"over limit {row.v_over_limit_max_mps:.2f} m/s",
    ])
    fig.suptitle(f"{row.scenario_id}   [{row.split}]\n{subtitle}",
                 fontsize=9.5, y=0.995)
    # Every wrapped line of the subtitle costs the axes a little headroom.
    fig.tight_layout(rect=(0, 0, 1, 0.975 - 0.013 * subtitle.count("\n")))
    return fig


# The subtitle carries whatever the consist and control labels happen to be,
# and for the longer ones one line runs off an 11.5 in canvas -- matplotlib
# clips it at both ends rather than complaining. Pack the facts onto as many
# lines as they need instead.
_SUBTITLE_COLS = 112


def _wrap_facts(facts, cols: int = _SUBTITLE_COLS) -> str:
    """Join ``facts`` with " | ", breaking the line before it overruns."""
    lines, cur = [], ""
    for fact in facts:
        candidate = f"{cur}   |   {fact}" if cur else fact
        if cur and len(candidate) > cols:
            lines.append(cur)
            cur = fact
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def print_summary(row) -> None:
    width = max(len(f) for f in _SUMMARY_FIELDS)
    for field in _SUMMARY_FIELDS:
        if field in row.index:
            value = row[field]
            if isinstance(value, (float, np.floating)):
                value = f"{value:,.4g}"
            print(f"  {field:<{width}}  {value}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/v2"),
                    help="dataset root (default: data/v2)")

    sel = ap.add_argument_group(
        "selection", "filters compose, then --sort-by / --random order them, "
                     "then --rank picks one")
    sel.add_argument("--scenario-id", help="exact scenario id")
    sel.add_argument("--query", help="pandas query over index.parquet, e.g. "
                                     "\"split=='test_id' and adhesion=='low'\"")
    sel.add_argument("--split", help="shorthand for --query \"split=='...'\"")
    sel.add_argument("--route-id", help="restrict to one corridor")
    sel.add_argument("--sort-by", help="index column to order by, descending "
                                       "(F_max, v_over_limit_max_mps, E_trip, ...)")
    sel.add_argument("--ascending", action="store_true",
                     help="with --sort-by, take the smallest instead of the largest")
    sel.add_argument("--random", action="store_true", help="pick at random")
    sel.add_argument("--rank", type=int, default=0,
                     help="which of the ordered matches to plot (default: the first)")
    sel.add_argument("--seed", type=int, default=0, help="seed for --random")

    out = ap.add_argument_group("output")
    out.add_argument("--list", type=int, metavar="N", default=0,
                     help="print the first N matches and exit without plotting")
    out.add_argument("--out", type=Path,
                     help="PNG to write (default: media/scenarios/<scenario_id>.png)")
    out.add_argument("--show", action="store_true",
                     help="open a window instead of writing a file")
    out.add_argument("--dpi", type=int, default=140)
    out.add_argument("--vs-distance", action="store_true",
                     help="x axis is lead chainage rather than time")
    args = ap.parse_args()

    import pandas as pd

    if not (args.root / "index.parquet").exists():
        raise SystemExit(f"no index.parquet under {args.root}")
    idx = pd.read_parquet(args.root / "index.parquet")
    row, pool = select_row(idx, args)

    if args.list:
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(pool[_LIST_COLUMNS].head(args.list).to_string(index=False))
        print(f"\n{len(pool)} scenarios match.")
        return

    d, route, path = load_scenario(args.root, row)
    print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"{len(pool)} scenarios matched; plotting rank {args.rank}.\n")
    print_summary(row)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plot_scenario(d, route, row, args.vs_distance)
    if args.show:
        plt.show()
        return
    dest = args.out or (_REPO_ROOT / "media" / "scenarios" / f"{row.scenario_id}.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=args.dpi)
    plt.close(fig)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
