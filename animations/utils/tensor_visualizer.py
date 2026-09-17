"""Matrix/tensor-slice view component used by multiple scenes."""

from __future__ import annotations

from manim import DOWN, LEFT, RIGHT, UP, DecimalNumber, Rectangle, Text, VGroup


class TensorSliceView(VGroup):
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        row_labels: list[str],
        col_labels: list[str],
        title: str,
        cell_size: float = 0.42,
    ) -> None:
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.cells: list[list[Rectangle]] = []
        self.values: list[list[DecimalNumber]] = []
        self.cell_size = cell_size

        grid = VGroup()
        for r in range(n_rows):
            row_cells: list[Rectangle] = []
            row_vals: list[DecimalNumber] = []
            for c in range(n_cols):
                cell = Rectangle(width=cell_size, height=cell_size, stroke_width=1.2, color="#37474f", fill_opacity=1.0)
                cell.shift((c - (n_cols - 1) / 2) * cell_size * RIGHT + ((n_rows - 1) / 2 - r) * cell_size * UP)
                val = DecimalNumber(0.0, num_decimal_places=1, font_size=13).move_to(cell.get_center())
                row_cells.append(cell)
                row_vals.append(val)
                grid.add(cell)
                grid.add(val)
            self.cells.append(row_cells)
            self.values.append(row_vals)

        row_txt = VGroup(*[Text(lbl, font_size=16).next_to(self.cells[r][0], LEFT, buff=0.15) for r, lbl in enumerate(row_labels)])
        col_txt = VGroup(*[Text(lbl, font_size=16).next_to(self.cells[0][c], UP, buff=0.12) for c, lbl in enumerate(col_labels)])
        title_txt = Text(title, font_size=24).next_to(grid, UP, buff=0.5)
        axes_txt = Text("rows=vehicle, cols=feature", font_size=15).next_to(grid, DOWN, buff=0.25)
        self.add(grid, row_txt, col_txt, title_txt, axes_txt)

    def set_cell_color(self, row: int, col: int, color) -> None:
        self.cells[row][col].set_fill(color, opacity=1.0)

    def set_cell_value(self, row: int, col: int, value: float) -> None:
        self.values[row][col].set_value(value)
