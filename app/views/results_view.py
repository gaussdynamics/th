"""RESULTS screen -- pyqtgraph plots of a finished simulation.

show_result() is the sink end of the SimView -> SimWorker -> here pipeline
(wired by MainWindow). pyqtgraph is imported defensively so the app still
launches without it, matching route_view's fallback.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from simulator2.state_schema import EdgeChannel, NodeChannel

try:
    import pyqtgraph as pg

    _HAS_PG = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_PG = False

_PLOT_BG = "#b0b0b0"
_PLOT_FG = "#2b2b2b"
_SPEED_PEN = "#3a5e8c"
_FORCE_PEN = "#b5651d"


class ResultsView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(8)

        title = QLabel("RESULTS")
        title.setObjectName("screenTitle")
        outer.addWidget(title)

        self.subtitle = QLabel(
            "Car speeds, coupler forces will plot here after a run completes."
        )
        self.subtitle.setObjectName("screenSubtitle")
        self.subtitle.setWordWrap(True)
        outer.addWidget(self.subtitle)

        outer.addWidget(self._build_plot_area(), 1)

    def _build_plot_area(self) -> QWidget:
        if not _HAS_PG:
            hint = QLabel("Result plots need pyqtgraph.\n\n    pip install pyqtgraph")
            hint.setObjectName("screenSubtitle")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return hint

        pg.setConfigOption("background", _PLOT_BG)
        pg.setConfigOption("foreground", _PLOT_FG)
        pg.setConfigOptions(antialias=True)

        self.glw = pg.GraphicsLayoutWidget()
        self.p_speed = self.glw.addPlot(row=0, col=0)
        self.p_speed.setLabel("left", "Speed", units="m/s")
        self.p_speed.showGrid(x=True, y=True, alpha=0.25)

        self.p_force = self.glw.addPlot(row=1, col=0)
        self.p_force.setLabel("left", "Coupler force", units="N")
        self.p_force.setLabel("bottom", "Time", units="s")
        self.p_force.showGrid(x=True, y=True, alpha=0.25)
        self.p_force.setXLink(self.p_speed)
        return self.glw

    def show_result(self, result) -> None:
        n = result.metadata.get("N", result.H_hist.shape[1])
        self.subtitle.setText(
            f"{result.t.size} samples over {result.t[-1]:.0f}s, {n} vehicles."
        )
        if not _HAS_PG:
            return

        t = result.t
        self.p_speed.clear()
        for i in range(result.H_hist.shape[1]):
            v = result.H_hist[:, i, NodeChannel.V]
            pen = pg.mkPen(_SPEED_PEN, width=1.5 if i == 0 else 1)
            self.p_speed.plot(t, v, pen=pen)

        self.p_force.clear()
        for j in range(result.E_hist.shape[1]):
            f = result.E_hist[:, j, EdgeChannel.F_CPL]
            self.p_force.plot(t, f, pen=pg.mkPen(_FORCE_PEN, width=1))
