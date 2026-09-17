"""A labelled matrix with per-column units, used by the three-vehicle scene.

``TensorSliceView`` renders one grid of same-kind numbers. The schema matrices
are not like that: a row of ``node_static`` is a mass in kg next to a Davis
coefficient next to a force in newtons, and showing all of them raw makes the
grid unreadable. Each column here carries its own display scale and precision,
printed in the header, so what you read is 130 t and 300 kN rather than 1.3e5
and 3.0e5.

Panels also know whether they are dynamic or static. A static panel -- one whose
contents are constant in time -- is drawn muted and never heat-mapped, which is
the whole point the scene is making about the schema's split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    DecimalNumber,
    Rectangle,
    Text,
    VGroup,
)

from animations.utils.text_style import label

#: Palette. Dark ground, so cells are dark and text is light.
CELL_DYNAMIC = "#161c22"
CELL_STATIC = "#14181c"
EDGE_DYNAMIC = "#42545f"
EDGE_STATIC = "#2c363d"
TEXT_MAIN = "#e6edf3"
TEXT_DIM = "#8b9aa7"


@dataclass(frozen=True)
class Column:
    """One matrix column: what it is called, and how to print it."""

    label: str
    scale: float = 1.0  # displayed = raw * scale
    decimals: int = 2
    #: Thousands separators cost a character the cell may not have; chainage in
    #: metres reads fine without one.
    commas: bool = True

    def render(self, raw: float) -> float:
        return float(raw) * self.scale


class MatrixPanel(VGroup):
    def __init__(
        self,
        title: str,
        columns: list[Column],
        row_labels: list[str],
        row_colors: list[str] | None = None,
        static: bool = False,
        cell_w: float = 0.62,
        cell_h: float = 0.34,
        font_size: float = 13,
        header_font_size: float = 12,
        title_font_size: float = 17,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.static = static
        self.n_rows = len(row_labels)
        self.n_cols = len(columns)
        self.cell_w = cell_w
        self.cell_h = cell_h

        fill = CELL_STATIC if static else CELL_DYNAMIC
        stroke = EDGE_STATIC if static else EDGE_DYNAMIC

        self.cells: list[list[Rectangle]] = []
        self.numbers: list[list[DecimalNumber]] = []
        grid = VGroup()
        for r in range(self.n_rows):
            cells_r, nums_r = [], []
            for c in range(self.n_cols):
                cell = Rectangle(
                    width=cell_w, height=cell_h,
                    stroke_width=1.0, stroke_color=stroke,
                    fill_color=fill, fill_opacity=1.0,
                )
                cell.shift(
                    (c - (self.n_cols - 1) / 2) * cell_w * RIGHT
                    + ((self.n_rows - 1) / 2 - r) * cell_h * UP
                )
                num = DecimalNumber(
                    0.0,
                    num_decimal_places=columns[c].decimals,
                    group_with_commas=columns[c].commas,
                    mob_class=Text,          # Pango, not LaTeX
                    font_size=font_size,
                    color=TEXT_MAIN if not static else TEXT_DIM,
                ).move_to(cell.get_center())
                cells_r.append(cell)
                nums_r.append(num)
                grid.add(cell, num)
            self.cells.append(cells_r)
            self.numbers.append(nums_r)
        self.grid = grid

        self.row_tags = VGroup()
        colors = row_colors or [TEXT_DIM] * self.n_rows
        for r, lbl in enumerate(row_labels):
            tag = label(lbl, font_size=header_font_size + 1, color=colors[r])
            tag.next_to(self.cells[r][0], LEFT, buff=0.11)
            self.row_tags.add(tag)

        headers = VGroup()
        for c, col in enumerate(columns):
            h = label(col.label, font_size=header_font_size, color=TEXT_DIM)
            # Headers sit one cell pitch apart, so anything wider than the cell
            # collides with its neighbour and the two read as one word. Leave a
            # visible gutter rather than merely avoiding overlap.
            if h.width > cell_w * 0.86:
                h.scale(cell_w * 0.86 / h.width)
            h.next_to(self.cells[0][c], UP, buff=0.08)
            headers.add(h)
        self.headers = headers

        shape = f"[{self.n_rows}x{self.n_cols}]"
        title_txt = label(f"{title}  {shape}", font_size=title_font_size, color=TEXT_MAIN)
        title_txt.next_to(headers, UP, buff=0.10)
        title_txt.align_to(self.cells[0][0], LEFT)
        self.title_txt = title_txt

        note = label(
            "constant in t" if static else "evolves with t",
            font_size=header_font_size - 1,
            color=TEXT_DIM,
        )
        note.next_to(self.grid, DOWN, buff=0.09)
        note.align_to(self.cells[0][0], LEFT)
        self.note = note

        self.add(self.grid, self.row_tags, self.headers, self.title_txt, self.note)

    # -- values -----------------------------------------------------------
    def set_values(self, raw: np.ndarray) -> None:
        """Write a whole ``[n_rows, n_cols]`` block of raw SI values."""
        for r in range(self.n_rows):
            for c, col in enumerate(self.columns):
                self.numbers[r][c].set_value(col.render(raw[r, c]))

    def set_heat(self, r: int, c: int, color) -> None:
        self.cells[r][c].set_fill(color, opacity=1.0)
