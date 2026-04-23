#!/usr/bin/env python3
"""Schematic: physics-based train consist (node states + coupler edges).

Run from repository root::

    python simulator2/examples/draw_consist_diagram.py

Writes ``consist_physics_diagram.png`` and ``consist_physics_diagram.svg`` next to this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def _zigzag_coupler(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    n_zigs: int = 6,
    amp: float = 0.055,
    lw: float = 1.35,
    color: str = "#37474f",
) -> float:
    """Draw a horizontal zigzag between ``x0`` and ``x1`` at height ``y``. Returns midpoint x."""
    xs = np.linspace(x0, x1, 2 * n_zigs + 1)
    zs = y + np.zeros_like(xs)
    zs[1::2] = y + amp
    zs[2::2] = y - amp
    ax.plot(xs, zs, color=color, lw=lw, solid_capstyle="round", zorder=1)
    return float(0.5 * (x0 + x1))


def draw_consist_diagram(
    out_path: Path,
    dpi: int = 200,
    also_svg: bool = True,
    tail_label: str = "Car N",
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.9), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.rcParams.update(
        {
            "font.size": 10,
            "mathtext.fontset": "stixsans",
        }
    )

    y_mid = 0.55
    box_w, box_h = 1.08, 0.78
    gap = 0.42

    labels = ["Locomotive", "Car 1", "Car 2", tail_label]
    n = len(labels)
    total_w = n * box_w + (n - 1) * gap
    x_start = -total_w / 2 + box_w / 2
    centers = [x_start + i * (box_w + gap) for i in range(n)]

    node_lines = (
        r"$x_i,\; v_i$",
        r"$z_{B,i},\; z_{T,i}$",
        r"$m_i,\;\mathrm{Davis}_i$",
    )
    edge_lines = (
        r"$\delta_j,\; \dot\delta_j$",
        r"$F_{\mathrm{cpl},j}$",
        r"slack / draft–buff",
    )

    face = "#eceff1"
    edge_c = "#263238"

    for i, (cx, lab) in enumerate(zip(centers, labels)):
        w, h = box_w, box_h
        bx = cx - w / 2
        by = y_mid - h / 2
        fancy = mpatches.FancyBboxPatch(
            (bx, by),
            w,
            h,
            boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.06),
            linewidth=1.6,
            edgecolor=edge_c,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(fancy)

        ax.text(
            cx,
            by + h - 0.11,
            lab,
            ha="center",
            va="top",
            fontsize=11 if i == 0 else 10,
            fontweight="bold",
            color=edge_c,
        )

        if i == 0:
            idx = r"$i=0$"
        elif i == n - 1:
            idx = r"$i=N$"
        else:
            idx = rf"$i={i}$"
        detail = "node " + idx + "\n" + "\n".join(node_lines)
        ax.text(
            cx,
            by + 0.12,
            detail,
            ha="center",
            va="bottom",
            fontsize=8.6,
            color="#37474f",
            linespacing=1.35,
        )

    # Couplers
    mids: list[float] = []
    for i in range(n - 1):
        x_right = centers[i] + box_w / 2
        x_left = centers[i + 1] - box_w / 2
        xm = _zigzag_coupler(ax, x_right, x_left, y_mid)
        mids.append(xm)
        ax.text(
            xm,
            y_mid + box_h * 0.58,
            "coupler " + rf"$j={i+1}$",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#546e7a",
        )

    # One richer edge callout (middle coupler)
    j_mid = 2
    if len(mids) >= j_mid:
        xm = mids[j_mid - 1]
        ax.text(
            xm,
            y_mid - box_h * 0.62,
            "edge " + r"$j$" + " (coupler)\n" + "\n".join(edge_lines),
            ha="center",
            va="top",
            fontsize=8.6,
            color="#455a64",
            bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#90a4ae", alpha=0.96),
        )

    ax.text(
        0.0,
        y_mid - box_h * 1.32,
        r"State: node tensor $\mathbf{H}\in\mathbb{R}^{N\times d_{\mathrm{node}}}$"
        + "   ·   "
        + r"Couplers: edge tensor $\mathbf{E}\in\mathbb{R}^{(N-1)\times d_{\mathrm{edge}}}$",
        ha="center",
        va="top",
        fontsize=9.4,
        color="#37474f",
        style="italic",
    )

    ax.set_xlim(-total_w / 2 - 0.4, total_w / 2 + 0.4)
    ax.set_ylim(y_mid - box_h * 1.48, y_mid + box_h * 1.05)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if also_svg:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    png = out_dir / "consist_physics_diagram.png"
    draw_consist_diagram(png, dpi=220, also_svg=True, tail_label="Car N")
    print("Wrote:", png)
    print("Wrote:", png.with_suffix(".svg"))


if __name__ == "__main__":
    main()
