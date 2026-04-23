#!/usr/bin/env python3
"""Minimal figure: definitions of node tensor H and edge tensor E.

Output PNG matches the pixel size of ``consist_physics_diagram.png`` (by default).

Run::

    python simulator2/examples/draw_HE_tensors_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _reference_png_size() -> tuple[int, int]:
    ref = Path(__file__).resolve().parent / "consist_physics_diagram.png"
    if ref.exists():
        try:
            from PIL import Image

            return Image.open(ref).size
        except ImportError:
            pass
    return 1834, 597


def draw_he_figure(out_path: Path, dpi: int = 220) -> None:
    w_px, h_px = _reference_png_size()
    fig_w_in = w_px / dpi
    fig_h_in = h_px / dpi

    plt.rcParams.update(
        {
            "font.size": 10,
            "mathtext.fontset": "stixsans",
        }
    )

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    face = "#eceff1"
    edge_c = "#263238"
    muted = "#37474f"
    # Titles sit above panel tops (panels end near y ≈ 0.84 in transAxes).
    title_y = 0.945

    # Light panels (match consist diagram tone)
    pad_x, pad_y = 0.02, 0.06
    panel_kw = dict(
        boxstyle=mpatches.BoxStyle("Round", pad=0.01, rounding_size=0.012),
        linewidth=1.35,
        edgecolor="#90a4ae",
        facecolor=face,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(
        mpatches.FancyBboxPatch((pad_x, 0.08), 0.46, 0.84 - 0.08, **panel_kw)
    )
    ax.add_patch(
        mpatches.FancyBboxPatch((0.52, 0.08), 0.46, 0.84 - 0.08, **panel_kw)
    )

    ax.axvline(0.5, color="#cfd8dc", lw=1.0, linestyle=(0, (2, 4)), zorder=1)

    left_body = (
        r"$\mathbf{H}_{i,:}$ — one row per vehicle $i=0,\ldots,N-1$." + "\n\n"
        r"$\mathbf{H} \in \mathbb{R}^{N \times d_{\mathrm{node}}}$, "
        r"$d_{\mathrm{node}} = 11$." + "\n\n"
        r"Dynamic: $x_i,\; v_i,\; z_{B,i},\; z_{T,i}$." + "\n"
        r"Static: $m_i$, Davis $(A_i,B_i,C_i)$, traction flag, "
        r"$F_{T,i}^{\max}$, $F_{B,i}^{\max}$."
    )
    ax.text(
        0.25,
        0.5,
        left_body,
        ha="center",
        va="center",
        fontsize=10.2,
        color=muted,
        linespacing=1.45,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.text(
        0.25,
        title_y,
        r"Node tensor $\mathbf{H}$",
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=edge_c,
        transform=ax.transAxes,
        zorder=2,
    )

    right_body = (
        r"$\mathbf{E}_{j,:}$ — one row per coupler $j=0,\ldots,N-2$" + "\n"
        r"(between vehicles $j$ and $j+1$)." + "\n\n"
        r"$\mathbf{E} \in \mathbb{R}^{(N-1) \times d_{\mathrm{edge}}}$, "
        r"$d_{\mathrm{edge}} = 9$." + "\n\n"
        r"$\delta_j,\; \dot\delta_j,\; F_{\mathrm{cpl},j}$; geometry / stiffness: "
        r"$L_{0,j}$, slack; $(k^d,c^d)$, $(k^b,c^b)$."
    )
    ax.text(
        0.75,
        0.5,
        right_body,
        ha="center",
        va="center",
        fontsize=10.2,
        color=muted,
        linespacing=1.45,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.text(
        0.75,
        title_y,
        r"Edge tensor $\mathbf{E}$",
        ha="center",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=edge_c,
        transform=ax.transAxes,
        zorder=2,
    )

    fig.savefig(
        out_path,
        dpi=dpi,
        facecolor="white",
        pad_inches=0,
        bbox_inches=None,
    )
    plt.close(fig)


def main() -> None:
    out = Path(__file__).resolve().parent / "HE_tensors_diagram.png"
    draw_he_figure(out, dpi=220)
    print("Wrote:", out)
    try:
        from PIL import Image

        print("Size (px):", Image.open(out).size)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
