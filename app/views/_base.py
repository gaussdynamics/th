"""Shared placeholder screen used by the skeleton.

Each real screen will replace its body; the title/subtitle scaffolding and
object names stay so the theme keeps applying.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderView(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("screenTitle")
        layout.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("screenSubtitle")
            subtitle_label.setWordWrap(True)
            layout.addWidget(subtitle_label)

        layout.addStretch(1)
