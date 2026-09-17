"""Layout helpers for consistent scene composition."""

from __future__ import annotations

from manim import LEFT, RIGHT


def left_panel_anchor():
    return 3.6 * LEFT


def right_panel_anchor():
    return 3.2 * RIGHT
