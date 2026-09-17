"""Focused tensor time-index explanation scene."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import DOWN, RIGHT, UP, DecimalNumber, Rectangle, Text
from manim import Scene

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from animations.utils.color_mapping import ColorMapping
from animations.utils.data_loader import load_animation_data
from animations.utils.tensor_visualizer import TensorSliceView


class TensorTimeEvolution(Scene):
    def construct(self) -> None:
        data = load_animation_data(_REPO_ROOT / "animations" / "data" / "demo_tensor_run.npz")
        n = data.n_vehicles
        feature_names = ["x", "v", "zB", "zT", "Fleft"]
        color_map = ColorMapping(data.tensor_slice, data.coupler_forces_n, data.velocities_mps)

        title = Text("Tensor Time Evolution: X[t, vehicle, feature]", font_size=34).to_edge(UP)
        self.add(title)

        view = TensorSliceView(
            n_rows=n,
            n_cols=len(feature_names),
            row_labels=[f"v{i}" for i in range(n)],
            col_labels=feature_names,
            title="Current slice X[t,:,:]",
        ).scale(0.9)
        view.shift(0.7 * DOWN)
        self.add(view)

        marker = Rectangle(width=0.16, height=0.65, color="#ffd54f").next_to(view, RIGHT, buff=0.25)
        marker_label = Text("sweep over t", font_size=18).next_to(marker, RIGHT, buff=0.15)
        t_num = DecimalNumber(0.0, num_decimal_places=2, font_size=24).next_to(title, DOWN, buff=0.2).shift(4.8 * RIGHT)
        self.add(marker, marker_label, t_num)

        for k in np.linspace(0, data.n_time - 1, 180).astype(int):
            t_num.set_value(float(data.time_s[k]))
            sl = data.tensor_slice[k]
            for r in range(n):
                for c in range(len(feature_names)):
                    if c in (1, 4):
                        view.set_cell_color(r, c, color_map.diverging_color(float(sl[r, c])))
                    else:
                        view.set_cell_color(r, c, color_map.sequential_color(float(sl[r, c])))
            self.wait(0.025)
