#!/usr/bin/env python3
"""Poster-oriented demo: 10-vehicle train (1 locomotive + 9 cars), track grade, controls, plots.

Run from repository root::

    python simulator2/examples/poster_train_demo.py

Writes high-resolution PNGs next to this script (``poster_train_*.png``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from simulator2 import constants as C
from simulator2.forces import traction_ramp
from simulator2.io_types import ExtendedTrainScenario
from simulator2.route import RouteProfile
from simulator2.scenarios import make_simple_train, _initial_positions
from simulator2.simulate import simulate_train_tensorized
from simulator2.state_schema import EdgeChannel, NodeChannel


def _smooth_brake_pulse(t: float, t_center: float, half_width: float, peak_N: float) -> float:
    """Hann-like bump symmetric around ``t_center`` (zero outside ``half_width``)."""
    if half_width <= 0.0:
        return 0.0
    u = (t - t_center) / half_width
    if abs(u) >= 1.0:
        return 0.0
    return float(peak_N * 0.5 * (1.0 + np.cos(np.pi * u)))


def make_poster_scenario(
    n_cars: int = 9,
    t_span: Tuple[float, float] = (0.0, 100.0),
    n_time_samples: int = 2001,
    x_lead_m: float = 1200.0,
    flat_track: bool = False,
) -> ExtendedTrainScenario:
    """One locomotive + ``n_cars`` cars, optional mild grade, traction + fleet brake pulse.

    Default track is a nonnegative ``sin(theta)`` bump (no downhill tail), which keeps
    braking episodes from re-accelerating the consist backward. Set ``flat_track=True``
    for a perfectly level tangent.
    """
    vehicles, couplers = make_simple_train(n_cars, l0_m=20.0)
    n = len(vehicles)

    if flat_track:
        s_nodes_m = np.array([0.0, 50_000.0], dtype=float)
        sin_theta_nodes = np.array([0.0, 0.0], dtype=float)
    else:
        # Mild "rise then return to level" along chainage; sin(theta) >= 0 everywhere here.
        s_nodes_m = np.array([0.0, 1_800.0, 5_500.0, 18_000.0], dtype=float)
        sin_theta_nodes = np.array([0.0, 0.0, 0.0024, 0.0], dtype=float)
    route = RouteProfile(s_nodes_m=s_nodes_m, sin_theta_nodes=sin_theta_nodes, kappa_nodes=None)

    x0 = _initial_positions(vehicles, couplers, x_lead_m)
    v0 = np.zeros(n, dtype=float)
    z0 = np.zeros(n, dtype=float)
    y0 = np.concatenate([x0, v0, z0, z0])

    F_trac_hold = 210_000.0
    t_hold_end = 42.0
    t_trac_release_end = 58.0

    def u_trac_cmd(t: float, i: int) -> float:
        if i != 0 or not vehicles[0].can_traction:
            return 0.0
        if t < 8.0:
            return 0.0
        if t < 28.0:
            return traction_ramp(t, t0=8.0, t1=28.0, F_max=F_trac_hold)
        if t < t_hold_end:
            return F_trac_hold
        if t < t_trac_release_end:
            s = (t - t_hold_end) / max(t_trac_release_end - t_hold_end, 1e-9)
            return F_trac_hold * (1.0 - s)
        return 0.0

    brake_center = 66.0
    brake_half_w = 11.5
    brake_peak = 50_000.0

    def u_brk_cmd(t: float, i: int) -> float:
        return _smooth_brake_pulse(t, brake_center, brake_half_w, brake_peak)

    t_eval = np.linspace(t_span[0], t_span[1], n_time_samples)

    return ExtendedTrainScenario(
        vehicles=vehicles,
        couplers=couplers,
        route=route,
        y0=y0,
        u_trac_cmd=u_trac_cmd,
        u_brk_cmd=u_brk_cmd,
        k_curv_scale=0.0,
        tau_brk_s=2.8,
        tau_trac_s=4.5,
        p_max_w=C.DEFAULT_P_MAX_W,
        t_span=t_span,
        t_eval=t_eval,
        rtol=C.DEFAULT_RTOL,
        atol=C.DEFAULT_ATOL,
        method=C.DEFAULT_METHOD,
        extra_metadata={
            "scenario": "poster_train_demo",
            "n_vehicles": n,
            "grade_s_nodes_m": s_nodes_m.tolist(),
            "sin_theta_nodes": sin_theta_nodes.tolist(),
        },
    )


def _sample_commands(
    t: np.ndarray,
    n: int,
    u_trac_cmd: Callable[[float, int], float],
    u_brk_cmd: Callable[[float, int], float],
) -> Tuple[np.ndarray, np.ndarray]:
    ut = np.zeros((t.size, n), dtype=float)
    ub = np.zeros((t.size, n), dtype=float)
    for k, tk in enumerate(t):
        for i in range(n):
            ut[k, i] = u_trac_cmd(float(tk), i)
            ub[k, i] = u_brk_cmd(float(tk), i)
    return ut, ub


def _vehicle_labels(n: int) -> list[str]:
    labels = ["Loco"]
    for j in range(1, n):
        labels.append(f"Car {j}")
    return labels


def plot_poster_figures(
    scenario: ExtendedTrainScenario,
    res,
    out_dir: Path,
    dpi: int = 220,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.35,
        }
    )

    t = res.t
    n = int(res.metadata["N"])
    labels = _vehicle_labels(n)

    ut_hist, ub_hist = _sample_commands(t, n, scenario.u_trac_cmd, scenario.u_brk_cmd)

    z_trac = res.H_hist[:, :, int(NodeChannel.Z_TRAC)]
    z_brk = res.H_hist[:, :, int(NodeChannel.Z_BRK)]
    v_ms = res.H_hist[:, :, int(NodeChannel.V)]
    v_kmh = v_ms * 3.6

    cmap = plt.cm.cividis(np.linspace(0.15, 0.95, n))

    # --- Figure 1: control vs realized actuation (locomotive traction, mid-train brake) ---
    fig1 = plt.figure(figsize=(11.0, 5.2), constrained_layout=True)
    gs1 = gridspec.GridSpec(2, 1, figure=fig1, height_ratios=[1.0, 1.0], hspace=0.22)
    ax_cmd = fig1.add_subplot(gs1[0, 0])
    ax_act = fig1.add_subplot(gs1[1, 0], sharex=ax_cmd)

    ax_cmd.plot(t, ut_hist[:, 0] / 1e3, color="#1b5e20", lw=2.4, label="Commanded traction (lead)")
    ax_cmd.plot(t, ub_hist[:, 0] / 1e3, color="#b71c1c", lw=2.0, ls="--", label="Commanded brake (lead)")
    ax_cmd.set_ylabel("Force command (kN)")
    ax_cmd.set_title("Control inputs (same brake command on every vehicle)")
    ax_cmd.legend(loc="upper right", frameon=True)
    ax_cmd.grid(True, alpha=0.35)

    ax_act.plot(t, z_trac[:, 0] / 1e3, color="#2e7d32", lw=2.2, label="Realized traction state $z_T$ (lead)")
    mid = n // 2
    ax_act.plot(t, z_brk[:, mid] / 1e3, color="#c62828", lw=2.0, label=f"Realized brake state $z_B$ ({labels[mid]})")
    ax_act.set_xlabel("Time (s)")
    ax_act.set_ylabel("Internal actuator state (kN)")
    ax_act.legend(loc="upper right", frameon=True)
    ax_act.grid(True, alpha=0.35)

    p1 = out_dir / "poster_train_controls.png"
    fig1.savefig(p1, dpi=dpi)
    plt.close(fig1)

    # --- Figure 2: speed of every vehicle ---
    fig2, ax2 = plt.subplots(figsize=(11.0, 5.0), constrained_layout=True)
    for i in range(n):
        ax2.plot(t, v_kmh[:, i], color=cmap[i], lw=1.9, label=labels[i], alpha=0.95)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (km/h)")
    ax2.set_title("Longitudinal speed — locomotive to tail (dark → light colormap)")
    ax2.legend(ncol=2, fontsize=8, frameon=True, loc="lower right")
    ax2.grid(True, alpha=0.35)
    p2 = out_dir / "poster_train_speeds.png"
    fig2.savefig(p2, dpi=dpi)
    plt.close(fig2)

    # --- Figure 3: coupler forces (draft positive convention from simulator) ---
    if res.E_hist.shape[1] > 0:
        fig3, ax3 = plt.subplots(figsize=(11.0, 5.0), constrained_layout=True)
        m = res.E_hist.shape[1]
        cmap_e = plt.cm.turbo(np.linspace(0.08, 0.92, m))
        f_kn = res.E_hist[:, :, int(EdgeChannel.F_CPL)] / 1e3
        for j in range(m):
            ax3.plot(t, f_kn[:, j], color=cmap_e[j], lw=1.7, label=f"Coupler {j}→{j+1}")
        ax3.axhline(0.0, color="0.35", lw=0.8, ls=":")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Coupler force (kN)")
        ax3.set_title("Inter-car coupling forces (slack–asymmetric draft/buff model)")
        ax3.legend(ncol=3, fontsize=8, frameon=True, loc="upper right")
        ax3.grid(True, alpha=0.35)
        p3 = out_dir / "poster_train_couplers.png"
        fig3.savefig(p3, dpi=dpi)
        plt.close(fig3)

    # --- Figure 4: compact grid for a single poster strip ---
    fig4 = plt.figure(figsize=(12.5, 10.0), constrained_layout=True)
    gs4 = gridspec.GridSpec(3, 1, figure=fig4, height_ratios=[1.05, 1.15, 1.15], hspace=0.28)
    a0 = fig4.add_subplot(gs4[0, 0])
    a1 = fig4.add_subplot(gs4[1, 0])
    a2 = fig4.add_subplot(gs4[2, 0])

    a0.plot(t, ut_hist[:, 0] / 1e3, color="#1b5e20", lw=2.2, label="$u_T$ lead")
    a0.plot(t, ub_hist[:, 0] / 1e3, color="#b71c1c", lw=1.8, ls="--", label="$u_B$ (all)")
    a0.plot(t, z_trac[:, 0] / 1e3, color="#66bb6a", lw=1.6, label="$z_T$ lead")
    a0.plot(t, z_brk[:, -1] / 1e3, color="#ef5350", lw=1.4, label="$z_B$ tail")
    a0.set_ylabel("kN")
    a0.set_title("Commands vs realized traction/brake states")
    a0.legend(ncol=4, fontsize=9, loc="upper right")
    a0.grid(True, alpha=0.35)

    for i in range(n):
        a1.plot(t, v_kmh[:, i], color=cmap[i], lw=1.7, label=labels[i])
    a1.set_ylabel("km/h")
    a1.set_title("Speed per vehicle")
    a1.legend(loc="lower right", ncol=2, fontsize=8)
    a1.grid(True, alpha=0.35)

    if res.E_hist.shape[1] > 0:
        m = res.E_hist.shape[1]
        cmap_e = plt.cm.turbo(np.linspace(0.08, 0.92, m))
        for j in range(m):
            a2.plot(t, f_kn[:, j], color=cmap_e[j], lw=1.5, label=f"{j}|{j+1}")
    a2.set_xlabel("Time (s)")
    a2.set_ylabel("kN")
    a2.set_title("Coupler forces")
    a2.legend(ncol=5, fontsize=8, loc="upper right", title="interface")
    a2.grid(True, alpha=0.35)

    p4 = out_dir / "poster_train_summary_strip.png"
    fig4.savefig(p4, dpi=dpi)
    plt.close(fig4)

    print("Wrote:", p1)
    print("Wrote:", p2)
    if res.E_hist.shape[1] > 0:
        print("Wrote:", p3)
    print("Wrote:", p4)


def main() -> None:
    scenario = make_poster_scenario(n_cars=9, t_span=(0.0, 100.0), n_time_samples=2201)
    res = simulate_train_tensorized(scenario)
    n = int(res.metadata["N"])
    print("Vehicles:", n, "(expected 10 with n_cars=9)")

    v_kmh = res.H_hist[:, :, int(NodeChannel.V)] * 3.6
    print("Final speed lead / tail (km/h):", float(v_kmh[-1, 0]), float(v_kmh[-1, -1]))

    out_dir = Path(__file__).resolve().parent
    try:
        plot_poster_figures(scenario, res, out_dir=out_dir, dpi=240)
    except ImportError:
        print("(matplotlib not installed; skipping figures)")


if __name__ == "__main__":
    main()
