"""SIMULATE screen -- the buttons-and-dials control panel.

Pick a saved Train and a Route, set duration/traction-override/curvature with
plain numeric entry, then GO. GO wires through a real QThread + SimWorker so
the blocking solve_ivp never touches the UI thread. STOP stays disabled until
a stepwise integrator exists (see APP_STACK_SKETCH.md §4) -- solve_ivp
integrates the whole t_span in one call, so there is no mid-run hook to stop
at yet.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from app.services.consist_service import ConsistService
from app.services.control_service import ControlService
from app.services.route_service import RouteService
from app.services.sim_service import SimService
from app.views._base import PlaceholderView
from app.workers.sim_worker import SimWorker
from simulator2.control_profile import build_command_callables

_DEFAULT_DURATION_S = 120.0
_DEFAULT_F_TRAC_MAX_N = 200_000.0
_DEFAULT_K_CURV_SCALE = 0.0
_NO_PROFILE = "(default ramp)"


class SimView(PlaceholderView):
    run_started = Signal()
    run_finished = Signal(object)  # TensorSimulationResult
    run_failed = Signal(str)

    def __init__(
        self,
        sim_service: SimService | None = None,
        consist_service: ConsistService | None = None,
        route_service: RouteService | None = None,
        control_service: ControlService | None = None,
    ) -> None:
        super().__init__(
            "SIMULATE",
            "Pick a train and a route, set the run, then GO.",
        )
        self.svc = sim_service or SimService()
        self.consist_svc = consist_service or ConsistService()
        self.route_svc = route_service or RouteService()
        self.control_svc = control_service or ControlService()
        self._thread: QThread | None = None
        self._worker: SimWorker | None = None

        panel = QGroupBox("CONTROL PANEL")
        col = QVBoxLayout(panel)

        form = QFormLayout()

        self.train_box = QComboBox()
        form.addRow("Train:", self.train_box)

        self.route_box = QComboBox()
        form.addRow("Route:", self.route_box)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 3600.0)
        self.duration_spin.setDecimals(0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setValue(_DEFAULT_DURATION_S)
        form.addRow("Duration:", self.duration_spin)

        trac_row = QHBoxLayout()
        self.f_trac_check = QCheckBox("override")
        self.f_trac_spin = QDoubleSpinBox()
        self.f_trac_spin.setRange(0.0, 1_000_000.0)
        self.f_trac_spin.setDecimals(0)
        self.f_trac_spin.setSingleStep(10_000.0)
        self.f_trac_spin.setSuffix(" N")
        self.f_trac_spin.setValue(_DEFAULT_F_TRAC_MAX_N)
        self.f_trac_spin.setEnabled(False)
        self.f_trac_check.toggled.connect(self.f_trac_spin.setEnabled)
        trac_row.addWidget(self.f_trac_spin, 1)
        trac_row.addWidget(self.f_trac_check)
        form.addRow("Max traction:", trac_row)

        self.k_curv_spin = QDoubleSpinBox()
        self.k_curv_spin.setRange(0.0, 1.0)
        self.k_curv_spin.setDecimals(3)
        self.k_curv_spin.setSingleStep(0.01)
        self.k_curv_spin.setValue(_DEFAULT_K_CURV_SCALE)
        form.addRow("Curvature scale:", self.k_curv_spin)

        self.control_box = QComboBox()
        form.addRow("Control profile:", self.control_box)

        col.addLayout(form)

        row = QHBoxLayout()
        row.setSpacing(16)
        self.go_btn = QPushButton("GO")
        self.go_btn.setObjectName("runButton")
        self.go_btn.clicked.connect(self._on_go)
        row.addWidget(self.go_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip(
            "Not available yet: one solve_ivp call integrates the whole run "
            "(see APP_STACK_SKETCH.md §4)."
        )
        row.addWidget(self.stop_btn)
        row.addStretch(1)
        col.addLayout(row)

        self.readout = QLabel("—")
        self.readout.setObjectName("readout")
        col.addWidget(self.readout)

        self.layout().insertWidget(self.layout().count() - 1, panel)

        self._populate_routes()
        self._populate_trains()
        self._populate_control_profiles()

    # ---------- population ----------
    def _populate_control_profiles(self) -> None:
        current = self.control_box.currentData()
        self.control_box.blockSignals(True)
        self.control_box.clear()
        self.control_box.addItem(_NO_PROFILE, None)
        for path in self.control_svc.list_saved():
            self.control_box.addItem(path.stem, path)
        idx = self.control_box.findData(current) if current is not None else 0
        self.control_box.setCurrentIndex(idx if idx >= 0 else 0)
        self.control_box.blockSignals(False)

    def _populate_routes(self) -> None:
        self.route_box.clear()
        for entry in self.route_svc.list_routes():
            if not entry.has_npz:
                continue
            npz_path = Path(entry.path).with_suffix(".npz")
            self.route_box.addItem(entry.name, npz_path)

    def _populate_trains(self) -> None:
        current = self.train_box.currentData()
        self.train_box.blockSignals(True)
        self.train_box.clear()
        for path in self.consist_svc.list_saved_trains():
            self.train_box.addItem(path.stem, path)
        if current is not None:
            idx = self.train_box.findData(current)
            if idx >= 0:
                self.train_box.setCurrentIndex(idx)
        self.train_box.blockSignals(False)

    def showEvent(self, event) -> None:  # noqa: D102
        super().showEvent(event)
        self._populate_trains()  # pick up trains saved on the Consist screen
        self._populate_control_profiles()  # pick up profiles saved on the Controls screen

    # ---------- run ----------
    def _on_go(self) -> None:
        route_path = self.route_box.currentData()
        train_path = self.train_box.currentData()
        if route_path is None:
            self.readout.setText("no route with a .npz profile found")
            self.run_failed.emit("no route available")
            return
        if train_path is None:
            self.readout.setText("no saved train found — build one on the Consist screen")
            self.run_failed.emit("no train available")
            return

        try:
            train = self.consist_svc.load_train(train_path)
            vehicles, couplers = self.consist_svc.build_consist(train)

            u_trac_cmd = u_brk_cmd = None
            control_path = self.control_box.currentData()
            if control_path is not None:
                profile = self.control_svc.load(control_path)
                u_trac_cmd, u_brk_cmd = build_command_callables(profile, train, vehicles)

            scenario = self.svc.build_scenario(
                route_path,
                vehicles,
                couplers,
                t_span=(0.0, self.duration_spin.value()),
                f_trac_max_N=self.f_trac_spin.value() if self.f_trac_check.isChecked() else None,
                k_curv_scale=self.k_curv_spin.value(),
                u_trac_cmd=u_trac_cmd,
                u_brk_cmd=u_brk_cmd,
            )
        except Exception as exc:
            self.readout.setText(f"build error: {exc}")
            self.run_failed.emit(str(exc))
            return

        self.go_btn.setEnabled(False)
        self.readout.setText("running…")
        self.run_started.emit()

        self._thread = QThread()
        self._worker = SimWorker(self.svc, scenario)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_finished(self, result) -> None:
        self.go_btn.setEnabled(True)
        n = result.metadata.get("N", result.H_hist.shape[1])
        self.readout.setText(f"OK: {result.t.size} samples, {n} vehicles")
        self.run_finished.emit(result)

    def _on_failed(self, message: str) -> None:
        self.go_btn.setEnabled(True)
        self.readout.setText(f"FAILED: {message}")
        self.run_failed.emit(message)
