"""A small painted status LED — idle / running / error / off."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

_STATES = {
    "idle": QColor("#e8c170"),     # amber
    "running": QColor("#9ef0a0"),  # green
    "error": QColor("#ff6b6b"),    # red
    "off": QColor("#3a352d"),      # dark
}


class Led(QWidget):
    def __init__(self, state: str = "idle", diameter: int = 12, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._d = diameter
        self.setFixedSize(diameter + 8, diameter + 6)

    def set_state(self, state: str) -> None:
        self._state = state
        self.update()

    def sizeHint(self) -> QSize:  # noqa: D102
        return QSize(self._d + 8, self._d + 6)

    def paintEvent(self, _event) -> None:  # noqa: D102
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = _STATES.get(self._state, _STATES["off"])
        painter.setBrush(color)
        painter.setPen(QColor("#0d0c0a"))
        x = 4
        y = (self.height() - self._d) // 2
        painter.drawEllipse(x, y, self._d, self._d)
        painter.end()
