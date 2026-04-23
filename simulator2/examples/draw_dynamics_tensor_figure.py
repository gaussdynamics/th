#!/usr/bin/env python3
"""Two-panel figure: extended train dynamics (left) vs tensorized formulation (right).

Run from repository root::

    python simulator2/examples/draw_dynamics_tensor_figure.py

Writes ``train_dynamics_tensor_formulation.png`` (and optional SVG) under this directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _left_panel_text() -> str:
    return r"""
\bf Extended state
\rm
\[
\mathbf{y} =
\big[\mathbf{x}^\top,\ \mathbf{v}^\top,\ \mathbf{z}_B^\top,\ \mathbf{z}_T^\top\big]^\top
\in\mathbb{R}^{4N}
\]

\bf Coupler kinematics\ \rm (edge $j=1,\ldots,N-1$)
\[
\delta_j = (x_j - x_{j+1}) - L_{0,j},\qquad
\dot\delta_j = v_j - v_{j+1}
\]
\[
F_{\mathrm{cpl},j} = f_{\mathrm{slack}}(\delta_j,\dot\delta_j;\ s_j,\ k^d_j,c^d_j,\ k^b_j,c^b_j)
\]

\bf Longitudinal motion\ \rm (vehicle $i$)
\[
\dot x_i = v_i
\]
\[
m_i\,\dot v_i
= F_{T,i} - F_{B,i} - R_i(v_i) - G_i(x_i) - F_{\kappa,i}(x_i,v_i)
+ F_{\mathrm{cpl},i-1} - F_{\mathrm{cpl},i}
\]
with $F_{\mathrm{cpl},0}=F_{\mathrm{cpl},N}=0$, and
\[
R_i = A_i + B_i|v_i| + C_i v_i^2\ \ (\mathrm{Davis}),\quad
G_i = m_i g\,\sin\theta(x_i)
\]

\bf Actuators\ \rm (first--order lags; $u$ are commands)
\[
\dot z_{B,i} = \frac{u_{B,i}(t) - z_{B,i}}{\tau_B},\qquad
\dot z_{T,i} = \frac{u_{T,i}(t) - z_{T,i}}{\tau_T}
\]
\[
F_{B,i} = \min\bigl(\max(z_{B,i},0),\,F_{B,i}^{\max}\bigr)
\]
\[
F_{T,i} =
\begin{cases}
0, & \text{no traction motor}\\
\displaystyle
\min\!\Bigl(\max(z_{T,i},0),\,\frac{P_{\max}}{\max(|v_i|,\varepsilon)}\Bigr), & \text{lead/loco}
\end{cases}
\]
"""


def _right_panel_text() -> str:
    return r"""
\bf Node tensor\ \rm (per vehicle $i$; static params appended as columns)
\[
\mathbf{H}\in\mathbb{R}^{N\times d_{\mathrm{node}}},\qquad
\text{dynamic columns: }(x_i,\ v_i,\ z_{B,i},\ z_{T,i})
\]
\[
+\ (m_i,\ A_i,B_i,C_i,\ \mathbb{1}_{\mathrm{trac}},\ F_T^{\max},\,F_B^{\max})
\]
\[
d_{\mathrm{node}}=11,\quad d_{\mathrm{edge}}=9
\]

\bf Edge tensor\ \rm (per coupler $j$; rebuilt from $\mathbf{H}$ each RHS eval.)
\[
\mathbf{E}\in\mathbb{R}^{(N-1)\times d_{\mathrm{edge}}},\qquad
\text{columns include }(\delta_j,\ \dot\delta_j,\ F_{\mathrm{cpl},j},\ L_{0,j},\ s_j,\ k^d,c^d,k^b,c^b)
\]

\bf Pack / unpack\ \rm (same $\mathbf{y}$ as left)
\[
\mathbf{y} \ \mapsto\  \mathbf{H}=\mathcal{P}_H(\mathbf{y}),\quad
\mathbf{E}=\mathcal{E}(\mathbf{H})
\]
\[
\dot{\mathbf{y}} = \mathcal{F}_{\mathrm{tensor}}(t,\mathbf{H},\mathbf{E};\ \mathbf{u},\,\mathrm{route})
\]

\bf RHS algorithmically identical\ \rm to the vector extended model:
read $(x_i,v_i,z_{B,i},z_{T,i})$ and coupler forces $F_{\mathrm{cpl},j}$ from
$(\mathbf{H},\mathbf{E})$, then form $(\dot x_i,\dot v_i,\dot z_{B,i},\dot z_{T,i})$
as on the left; stack to $\dot{\mathbf{y}}\in\mathbb{R}^{4N}$.
"""


def draw_figure(out_path: Path, dpi: int = 200, also_svg: bool = True) -> None:
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.size": 11,
        }
    )

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(13.5, 8.6),
        dpi=dpi,
        constrained_layout=True,
    )

    for ax, title, body in (
        (ax_l, "Train dynamics (extended ODE)", _left_panel_text()),
        (ax_r, "Tensorized formulation ($\\mathbf{H}$, $\\mathbf{E}$)", _right_panel_text()),
    ):
        ax.set_title(title, fontsize=13, pad=10, fontweight="semibold")
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            body.strip(),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            linespacing=1.55,
        )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if also_svg:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    out = Path(__file__).resolve().parent / "train_dynamics_tensor_formulation.png"
    draw_figure(out, dpi=220, also_svg=True)
    print("Wrote:", out)
    print("Wrote:", out.with_suffix(".svg"))


if __name__ == "__main__":
    main()
