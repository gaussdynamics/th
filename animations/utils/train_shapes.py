"""Reusable train drawing primitives for Manim scenes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Circle,
    Line,
    Rectangle,
    RoundedRectangle,
    Text,
    VGroup,
    VMobject,
)


@dataclass(frozen=True)
class TrainStyle:
    car_width: float = 1.0
    car_height: float = 0.45
    wheel_r: float = 0.09
    gap: float = 0.2


def make_car(label: str, color) -> VGroup:
    body = RoundedRectangle(corner_radius=0.08, width=1.0, height=0.45, color=color, fill_opacity=0.8)
    w1 = Circle(radius=0.09, color=color, fill_opacity=0.8).move_to(body.get_bottom() + 0.25 * LEFT + 0.07 * DOWN)
    w2 = Circle(radius=0.09, color=color, fill_opacity=0.8).move_to(body.get_bottom() + 0.25 * RIGHT + 0.07 * DOWN)
    txt = Text(label, font_size=18).move_to(body.get_center())
    return VGroup(body, w1, w2, txt)


def _spring_poly(start, end, turns: int = 5, amplitude: float = 0.05) -> VMobject:
    direction = end - start
    length = np.linalg.norm(direction)
    if length <= 1e-6:
        return Line(start, end, color="#b0bec5")
    tangent = direction / length
    normal = np.array([-tangent[1], tangent[0], 0.0])
    points = [start]
    for i in range(1, 2 * turns):
        s = i / (2 * turns)
        base = start + s * direction
        offset = amplitude * normal if i % 2 else -amplitude * normal
        points.append(base + offset)
    points.append(end)
    spring = VMobject(color="#b0bec5")
    spring.set_points_as_corners(points)
    return spring


def make_spring_damper(start, end) -> VGroup:
    spring = _spring_poly(start, end)
    mid = 0.5 * (start + end)
    damper_body = Rectangle(width=0.12, height=0.05, color="#cfd8dc", fill_opacity=0.9).move_to(mid)
    damper_rod_left = Line(start, mid - 0.07 * RIGHT, color="#cfd8dc")
    damper_rod_right = Line(mid + 0.07 * RIGHT, end, color="#cfd8dc")
    return VGroup(spring, damper_body, damper_rod_left, damper_rod_right)


def update_spring_damper(coupler: VGroup, start, end, color) -> None:
    spring, damper_body, rod_left, rod_right = coupler
    spring.become(_spring_poly(start, end))
    spring.set_color(color)
    mid = 0.5 * (start + end)
    damper_body.move_to(mid).set_color(color)
    rod_left.become(Line(start, mid - 0.07 * RIGHT, color=color))
    rod_right.become(Line(mid + 0.07 * RIGHT, end, color=color))


def build_train_group(n_vehicles: int) -> tuple[VGroup, list[VGroup], list[VGroup]]:
    cars: list[VGroup] = []
    couplers: list[VGroup] = []
    group = VGroup()
    for i in range(n_vehicles):
        car = make_car("L" if i == 0 else f"C{i}", color="#4fc3f7" if i == 0 else "#90a4ae")
        if i == 0:
            car.move_to(0.0 * RIGHT)
        else:
            car.next_to(cars[-1], RIGHT, buff=0.2)
            coupler = make_spring_damper(cars[-1].get_right(), car.get_left())
            couplers.append(coupler)
            group.add(coupler)
        cars.append(car)
        group.add(car)
    return group, cars, couplers
