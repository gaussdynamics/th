"""ROUTE screen — browse routes, preview the profile curve, read the stats.

Layout: a route list on the left; on the right an info panel over a stack of
linked profile plots (elevation / grade / curvature vs chainage).

pyqtgraph is imported defensively so the app still launches without it (the
plot area falls back to a hint label).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.services.route_service import RouteData, RouteService

try:
    import pyqtgraph as pg

    _HAS_PG = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_PG = False

# plot pens (theme-friendly)
_ELEV_PEN = "#3b6e3b"
_GRADE_PEN = "#b5651d"
_KAPPA_PEN = "#3a5e8c"
_PLOT_BG = "#b0b0b0"
_PLOT_FG = "#2b2b2b"

# (field key, label) shown in the info panel, in order
_INFO_FIELDS = [
    ("length", "Length"),
    ("vertices", "Vertices"),
    ("elev", "Elevation"),
    ("climb", "Total climb"),
    ("descent", "Total descent"),
    ("grade", "Grade range"),
    ("kappa", "Max curvature"),
    ("radius", "Min radius"),
    ("vmax", "Speed limit"),
    ("coverage", "Maxspeed cov."),
]


class RouteView(QWidget):
    def __init__(self, service: RouteService | None = None) -> None:
        super().__init__()
        self.service = service or RouteService()
        self._entries: list = []
        self._value_labels: dict[str, QLabel] = {}
        self._build_ui()
        self._populate()

    # ---------- construction ----------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(8)

        title = QLabel("ROUTE")
        title.setObjectName("screenTitle")
        outer.addWidget(title)
        subtitle = QLabel("Select a route to preview its profile and stats.")
        subtitle.setObjectName("screenSubtitle")
        outer.addWidget(subtitle)

        split = QSplitter(Qt.Orientation.Horizontal)

        self.list = QListWidget()
        self.list.setObjectName("routeList")
        self.list.currentRowChanged.connect(self._on_select)
        split.addWidget(self.list)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)
        right_lay.addWidget(self._build_info_panel())
        right_lay.addWidget(self._build_plot_area(), 1)
        split.addWidget(right)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([220, 700])
        outer.addWidget(split, 1)

    def _build_info_panel(self) -> QGroupBox:
        box = QGroupBox("ROUTE INFO")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        # two (label: value) pairs per row -> 4 columns
        per_row = 2
        for i, (key, text) in enumerate(_INFO_FIELDS):
            r, c = divmod(i, per_row)
            name = QLabel(f"{text}:")
            name.setObjectName("infoKey")
            value = QLabel("—")
            value.setObjectName("readout")
            self._value_labels[key] = value
            grid.addWidget(name, r, c * 2)
            grid.addWidget(value, r, c * 2 + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return box

    def _build_plot_area(self) -> QWidget:
        if not _HAS_PG:
            hint = QLabel(
                "Profile plots need pyqtgraph.\n\n    pip install pyqtgraph"
            )
            hint.setObjectName("screenSubtitle")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return hint

        pg.setConfigOption("background", _PLOT_BG)
        pg.setConfigOption("foreground", _PLOT_FG)
        pg.setConfigOptions(antialias=True)

        self.glw = pg.GraphicsLayoutWidget()
        self.p_elev = self.glw.addPlot(row=0, col=0)
        self.p_elev.setLabel("left", "Elev", units="m")
        self.p_grade = self.glw.addPlot(row=1, col=0)
        self.p_grade.setLabel("left", "Grade", units="%")
        self.p_kappa = self.glw.addPlot(row=2, col=0)
        self.p_kappa.setLabel("left", "Curv", units="1/m")
        self.p_kappa.setLabel("bottom", "Chainage", units="km")

        for p in (self.p_elev, self.p_grade, self.p_kappa):
            p.showGrid(x=True, y=True, alpha=0.25)
        self.p_grade.setXLink(self.p_elev)
        self.p_kappa.setXLink(self.p_elev)
        return self.glw

    # ---------- data flow ----------
    def _populate(self) -> None:
        self._entries = self.service.list_routes()
        self.list.clear()
        for e in self._entries:
            tag = "" if e.has_npz else "   (no .npz)"
            self.list.addItem(f"{e.name}{tag}")
        if self._entries:
            self.list.setCurrentRow(0)

    def _on_select(self, row: int) -> None:
        if row < 0 or row >= len(self._entries):
            return
        try:
            data = self.service.load(self._entries[row].path)
        except Exception as exc:  # show the error in the panel rather than crash
            for v in self._value_labels.values():
                v.setText("—")
            self._value_labels["length"].setText(f"load error: {exc}")
            return
        self._update_info(data)
        self._update_plots(data)

    def _update_info(self, d: RouteData) -> None:
        def _radius() -> str:
            return f"{d.min_radius_m:.0f} m" if d.min_radius_m != float("inf") else "—"

        self._value_labels["length"].setText(f"{d.length_km:.2f} km")
        self._value_labels["vertices"].setText(f"{d.num_vertices}")
        self._value_labels["elev"].setText(f"{d.elev_min_m:.0f}–{d.elev_max_m:.0f} m")
        self._value_labels["climb"].setText(f"{d.elev_gain_m:.0f} m")
        self._value_labels["descent"].setText(f"{d.elev_loss_m:.0f} m")
        self._value_labels["grade"].setText(
            f"{d.min_grade_pct:+.1f}% … {d.max_grade_pct:+.1f}%"
        )
        self._value_labels["kappa"].setText(f"{d.max_abs_kappa_1pm:.4f} 1/m")
        self._value_labels["radius"].setText(_radius())
        self._value_labels["vmax"].setText(
            f"{d.v_max_min_kph:.0f}–{d.v_max_max_kph:.0f} km/h"
        )
        self._value_labels["coverage"].setText(f"{d.maxspeed_coverage * 100:.0f}%")

    def _update_plots(self, d: RouteData) -> None:
        if not _HAS_PG:
            return
        x = d.s_km
        self.p_elev.clear()
        self.p_elev.plot(x, d.elevation_m, pen=pg.mkPen(_ELEV_PEN, width=2),
                         connect="finite")
        self.p_grade.clear()
        self.p_grade.plot(x, d.grade_pct, pen=pg.mkPen(_GRADE_PEN, width=2),
                          connect="finite")
        self.p_kappa.clear()
        self.p_kappa.plot(x, d.kappa_1pm, pen=pg.mkPen(_KAPPA_PEN, width=2),
                          connect="finite")
        self.p_elev.setTitle(d.name)
