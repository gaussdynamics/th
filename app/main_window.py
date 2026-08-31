"""Application shell: a left nav rail, a stack of screens, and a status bar.

This is the skeleton. Each screen is currently a placeholder; they get filled
in over the build order in APP_STACK_SKETCH.md (§8).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QWidget,
)

from app import __version__
from app.services.consist_service import ConsistService
from app.services.control_service import ControlService
from app.widgets.led import Led
from app.views.route_view import RouteView
from app.views.consist_view import ConsistView
from app.views.control_view import ControlView
from app.views.sim_view import SimView
from app.views.results_view import ResultsView

# (label, view class) in nav order. Mirrors the four-stop workflow in the sketch.
NAV = [
    ("ROUTE", RouteView),
    ("CONSIST", ConsistView),
    ("CONTROLS", ControlView),
    ("SIMULATE", SimView),
    ("RESULTS", ResultsView),
]
_LAND_ON = 3  # open on the Simulate screen


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RailLab — Research Console")
        self.resize(1100, 720)
        self.setMinimumSize(880, 560)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- left nav rail ---
        self.rail = QListWidget()
        self.rail.setObjectName("navRail")
        self.rail.setFixedWidth(168)

        # --- stacked screens ---
        # CONSIST and SIMULATE share one ConsistService (and its CatalogService)
        # so a car type / train edited on one screen is immediately visible on
        # the other, without either view reaching into the other's state.
        # CONTROLS and SIMULATE likewise share one ControlService, so a saved
        # control profile shows up in Simulate's dropdown right away.
        consist_service = ConsistService()
        control_service = ControlService()

        self.stack = QStackedWidget()
        self.stack.setObjectName("screenStack")
        self.views: dict[str, QWidget] = {}
        for label, view_cls in NAV:
            QListWidgetItem(label, self.rail)
            if label == "CONSIST":
                view = view_cls(consist=consist_service)
            elif label == "CONTROLS":
                view = view_cls(control_service=control_service)
            elif label == "SIMULATE":
                view = view_cls(consist_service=consist_service, control_service=control_service)
            else:
                view = view_cls()
            self.views[label] = view
            self.stack.addWidget(view)

        self.rail.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.rail.setCurrentRow(_LAND_ON)

        root.addWidget(self.rail)
        root.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self._build_status_bar()
        self._wire_sim_to_results()

    def _wire_sim_to_results(self) -> None:
        """SimView emits run_started/run_finished/run_failed; MainWindow is the
        wiring hub that fans them out to the Results screen and the status bar,
        so neither view needs a reference to the other."""
        sim_view = self.views["SIMULATE"]
        results_view = self.views["RESULTS"]
        results_index = [label for label, _ in NAV].index("RESULTS")

        sim_view.run_started.connect(lambda: self.led.set_state("running"))
        sim_view.run_started.connect(lambda: self.status_label.setText("RUNNING"))

        sim_view.run_finished.connect(results_view.show_result)
        sim_view.run_finished.connect(lambda _r: self.led.set_state("idle"))
        sim_view.run_finished.connect(lambda _r: self.status_label.setText("READY"))
        sim_view.run_finished.connect(lambda _r: self.rail.setCurrentRow(results_index))

        sim_view.run_failed.connect(lambda _m: self.led.set_state("error"))
        sim_view.run_failed.connect(lambda m: self.status_label.setText(f"ERROR: {m}"))

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        bar.setObjectName("statusBar")
        self.led = Led(state="idle")
        bar.addWidget(self.led)
        self.status_label = QLabel("READY")
        self.status_label.setObjectName("statusText")
        bar.addWidget(self.status_label)
        bar.addPermanentWidget(QLabel(f"RailLab v{__version__} · skeleton"))
        self.setStatusBar(bar)
