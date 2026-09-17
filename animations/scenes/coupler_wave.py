"""Coupler-force propagation style scene."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import DOWN, RIGHT, UP, Text, VGroup
from manim import Scene

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from animations.utils.color_mapping import ColorMapping
from animations.utils.data_loader import load_animation_data
from animations.utils.tensor_visualizer import TensorSliceView
from animations.utils.train_shapes import build_train_group


class CouplerWave(Scene):
    def construct(self) -> None:
        data = load_animation_data(_REPO_ROOT / "animations" / "data" / "demo_tensor_run.npz")
        title = Text("Coupler Disturbance Wave", font_size=34).to_edge(UP)
        self.add(title)

        if data.coupler_forces_n.shape[1] == 0:
            msg = Text(
                "Coupler-force channel not found in simulator output.\nExport E_hist[:,:,F_CPL] to enable this view.",
                font_size=28,
            )
            self.add(msg)
            self.wait(2)
            return

        n = data.n_vehicles
        train_group, _, couplers = build_train_group(n)
        train_group.scale(0.95).shift(1.3 * UP)
        self.add(train_group)

        matrix = TensorSliceView(
            n_rows=1,
            n_cols=data.coupler_forces_n.shape[1],
            row_labels=["F_cpl"],
            col_labels=[f"c{j}" for j in range(data.coupler_forces_n.shape[1])],
            title="E[t, coupler, F_cpl]",
            cell_size=0.55,
        ).scale(0.85).shift(1.8 * DOWN)
        self.add(matrix)

        cmap = ColorMapping(data.tensor_slice, data.coupler_forces_n, data.velocities_mps)

        bars = VGroup()
        for _ in range(data.coupler_forces_n.shape[1]):
            bars.add(Text("|", font_size=28, color="#b0bec5"))
        bars.arrange(RIGHT, buff=0.55).next_to(title, DOWN, buff=0.4)
        self.add(bars)

        for k in np.linspace(0, data.n_time - 1, 180).astype(int):
            f = data.coupler_forces_n[k]
            for j, val in enumerate(f):
                col = cmap.diverging_color(float(val), norm=cmap.force_norm)
                matrix.set_cell_color(0, j, col)
                if j < len(couplers):
                    couplers[j].set_color(col)
                bars[j].set_color(col)
            self.wait(0.03)
