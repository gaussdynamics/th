"""Conceptual scene: sequential vs tensorized updates."""

from __future__ import annotations

from manim import DOWN, LEFT, RIGHT, UP, Arrow, Rectangle, Text, VGroup
from manim import Scene


class TensorizationAdvantage(Scene):
    def construct(self) -> None:
        title = Text("Why Tensorization Helps", font_size=36).to_edge(UP)
        self.add(title)

        left_title = Text("Sequential loop", font_size=28).shift(3.3 * LEFT + 2.0 * UP)
        right_title = Text("Tensorized update", font_size=28).shift(3.3 * RIGHT + 2.0 * UP)
        self.add(left_title, right_title)

        seq_boxes = VGroup()
        for i in range(6):
            box = Rectangle(width=1.0, height=0.55, color="#90a4ae", fill_opacity=0.75)
            lbl = Text(f"v{i+1}", font_size=20).move_to(box.get_center())
            grp = VGroup(box, lbl)
            seq_boxes.add(grp)
        seq_boxes.arrange(DOWN, buff=0.24).shift(3.3 * LEFT + 0.2 * DOWN)

        seq_arrows = VGroup()
        for i in range(5):
            seq_arrows.add(Arrow(seq_boxes[i].get_bottom(), seq_boxes[i + 1].get_top(), buff=0.05, stroke_width=5))
        self.add(seq_boxes, seq_arrows)

        tensor_box = Rectangle(width=3.8, height=3.2, color="#4dd0e1", fill_opacity=0.3).shift(3.3 * RIGHT + 0.2 * DOWN)
        tensor_lbl = Text("X[:, vehicles, features]", font_size=24).move_to(tensor_box.get_center() + 1.0 * UP)
        op_lbl = Text("single batched op", font_size=24).move_to(tensor_box.get_center() + 0.2 * DOWN)
        arrow = Arrow(
            tensor_box.get_left() + 0.9 * UP,
            tensor_box.get_right() + 0.9 * UP,
            buff=0.2,
            stroke_width=7,
            color="#80cbc4",
        )
        self.add(tensor_box, tensor_lbl, op_lbl, arrow)

        foot = Text("Same physics, cleaner parallel formulation", font_size=24).to_edge(DOWN)
        self.add(foot)
        self.wait(2)
