"""Main side-by-side train and tensor evolution scene."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Arrow, DecimalNumber, Text

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from animations.utils.color_mapping import ColorMapping
from animations.utils.data_loader import load_animation_data
from animations.utils.tensor_visualizer import TensorSliceView
from animations.utils.train_shapes import build_train_group, update_spring_damper
from manim import Scene


class TrainTensorSideBySide(Scene):
    def construct(self) -> None:
        data = load_animation_data(_REPO_ROOT / "animations" / "data" / "demo_tensor_run.npz")
        n = data.n_vehicles
        t = data.time_s
        n_steps = min(420, data.n_time)
        idxs = np.linspace(0, data.n_time - 1, n_steps).astype(int)
        color_map = ColorMapping(data.tensor_slice, data.coupler_forces_n, data.velocities_mps)

        title = Text("Tensorized Train Dynamics Evolution", font_size=36).to_edge(UP)
        self.add(title)

        train_group, cars, couplers = build_train_group(n)
        train_anchor = 2.2 * UP + 0.2 * RIGHT
        train_group.scale(0.82).move_to(train_anchor)
        self.add(train_group)

        feature_names = ["Veh0"] + [f"V{i}" for i in range(1, n)]
        tensor_view = TensorSliceView(
            n_rows=5,
            n_cols=n,
            row_labels=["x", "v", "zB", "zT", "Fleft"],
            col_labels=feature_names,
            title="X[t, vehicle, feature]",
            cell_size=0.46,
        ).scale(1.0)
        tensor_view.move_to(1.85 * DOWN + 0.2 * RIGHT)
        self.add(tensor_view)

        time_label = Text("t [s]", font_size=20).next_to(title, DOWN, buff=0.2).shift(4.8 * RIGHT)
        time_value = DecimalNumber(0.0, num_decimal_places=2, font_size=24).next_to(time_label, RIGHT, buff=0.2)
        self.add(time_label, time_value)

        vel_arrow = Arrow(start=LEFT, end=RIGHT, buff=0.0, stroke_width=7, max_tip_length_to_length_ratio=0.2)
        vel_arrow.scale(0.7).next_to(cars[0], UP, buff=0.35)
        self.add(vel_arrow)

        coupler_legend = Text("Spring-damper coupler color = tensor Fleft row", font_size=16).next_to(train_group, DOWN, buff=0.2)
        self.add(coupler_legend)

        pos0 = data.positions_m[0]
        pos_span = max(float(np.max(pos0) - np.min(pos0)), 1.0)
        x_scale = 2.4 / pos_span

        for k in idxs:
            time_value.set_value(float(t[k]))

            xk = data.positions_m[k]
            x_shift = (xk - pos0) * x_scale
            for i, car in enumerate(cars):
                # Keep locomotive at front (right side) so positive velocity reads as rightward motion.
                car.move_to(train_anchor + (3.4 - i * 0.95 + float(x_shift[i])) * RIGHT)

            if data.coupler_forces_n.shape[1] > 0:
                for j, cpl in enumerate(couplers):
                    cpl_col = color_map.diverging_color(float(data.coupler_forces_n[k, j]), norm=color_map.force_norm)
                    update_spring_damper(cpl, cars[j].get_left(), cars[j + 1].get_right(), cpl_col)

            v0 = float(data.velocities_mps[k, 0])
            vel_arrow.set_color(color_map.diverging_color(v0, norm=color_map.velocity_norm))
            arrow_len = 0.2 + 0.6 * np.tanh(abs(v0))
            vel_arrow.put_start_and_end_on(
                cars[0].get_top() + 0.15 * RIGHT,
                cars[0].get_top() + arrow_len * LEFT,
            )

            tensor_k = data.tensor_slice[k]
            for veh in range(n):
                for feat in range(5):
                    val = float(tensor_k[veh, feat])
                    if feat in (1, 4):
                        tensor_view.set_cell_color(
                            feat,
                            veh,
                            color_map.diverging_color(
                                val,
                                norm=color_map.velocity_norm if feat == 1 else color_map.force_norm,
                            ),
                        )
                    else:
                        tensor_view.set_cell_color(feat, veh, color_map.sequential_color(val))
                    tensor_view.set_cell_value(feat, veh, val)
            self.wait(0.06)
