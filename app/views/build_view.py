"""BUILD screen -- start, watch, pause and resume dataset builds on this machine.

Everything here goes through BuildService and the build's own files
(``status.json``, ``build.log``), polled on a timer. The builder is a separate
process, so this screen can be closed and reopened mid-build: it picks the
build back up from its files.

A pause takes effect at the next batch boundary, a minute or two at the v3
batch sizes; until then the build shows PAUSING.
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.services.build_service import ACTIVE, DATA_DIR, BuildService, NewBuild
from app.widgets.led import Led

try:
    import pyqtgraph as pg

    _HAS_PG = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_PG = False

_POLL_MS = 2000
_PLOT_BG = "#b0b0b0"
_PLOT_FG = "#2b2b2b"
_RATE_PEN = "#3a5e8c"
_DEFAULT_ROUTES = DATA_DIR / "narn_routes_splits" / "routes"

_LED = {"sampling": "running", "running": "running", "finalizing": "running",
        "pausing": "idle", "paused": "idle", "done": "off",
        "stopped": "error", "failed": "error"}


def _duration(seconds) -> str:
    if seconds is None:
        return "--"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{d}d {h}h {m:02d}m"
    return f"{h}h {m:02d}m" if h else f"{m}m {seconds % 60:02d}s"


class NewBuildDialog(QDialog):
    """The arguments for a new build. Anything not on the form goes in
    'Extra args' verbatim (e.g. the corridor reservations)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New dataset build")
        self.setMinimumWidth(560)
        form = QFormLayout(self)

        self.out_edit = QLineEdit("v3")
        form.addRow("Output (under data/):", self.out_edit)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 10_000_000)
        self.n_spin.setSingleStep(1000)
        self.n_spin.setValue(100_000)
        form.addRow("Scenarios:", self.n_spin)
        self.physics_box = QComboBox()
        self.physics_box.addItems(["v3", "legacy"])
        form.addRow("Physics:", self.physics_box)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        form.addRow("Seed:", self.seed_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 4096)
        self.batch_spin.setValue(128)
        form.addRow("Batch size:", self.batch_spin)
        self.routes_edit = QLineEdit(str(_DEFAULT_ROUTES))
        form.addRow("Route dir:", self.routes_edit)
        self.extra_edit = QLineEdit()
        self.extra_edit.setPlaceholderText("e.g. --ood-corridors a b --control-eval-corridors c")
        form.addRow("Extra args:", self.extra_edit)

        self.preview = QLabel()
        self.preview.setObjectName("screenSubtitle")
        self.preview.setWordWrap(True)
        self.preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.preview)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("START BUILD")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        for w in (self.out_edit, self.routes_edit, self.extra_edit):
            w.textChanged.connect(self._update_preview)
        for w in (self.n_spin, self.seed_spin, self.batch_spin):
            w.valueChanged.connect(self._update_preview)
        self.physics_box.currentTextChanged.connect(self._update_preview)
        self._update_preview()

    def build(self) -> NewBuild:
        routes = self.routes_edit.text().strip()
        return NewBuild(
            out=DATA_DIR / self.out_edit.text().strip(),
            n_scenarios=self.n_spin.value(),
            physics=self.physics_box.currentText(),
            seed=self.seed_spin.value(),
            batch_size=self.batch_spin.value(),
            route_dir=Path(routes) if routes else None,
            extra_args=self.extra_edit.text(),
        )

    def _update_preview(self) -> None:
        try:
            argv = self.build().argv()
        except ValueError as exc:  # unbalanced quotes in Extra args
            self.preview.setText(f"Extra args: {exc}")
            return
        self.preview.setText("python scripts/build_dataset.py " + " ".join(argv))


class BuildView(QWidget):
    def __init__(self, build_service: BuildService | None = None) -> None:
        super().__init__()
        self.svc = build_service or BuildService()
        self._shown_out: Path | None = None
        self._build_ui()
        self.refresh_builds()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.poll)
        self._timer.start(_POLL_MS)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(8)

        title = QLabel("BUILD")
        title.setObjectName("screenTitle")
        outer.addWidget(title)
        sub = QLabel("Dataset builds on this machine. The builder runs on its own: "
                     "closing the console does not stop it. PAUSE stops after the "
                     "batch in flight; RESUME continues from there.")
        sub.setObjectName("screenSubtitle")
        sub.setWordWrap(True)
        outer.addWidget(sub)

        pick = QHBoxLayout()
        pick.addWidget(QLabel("Build:"))
        self.build_box = QComboBox()
        self.build_box.setMinimumWidth(260)
        self.build_box.currentIndexChanged.connect(lambda _i: self.poll())
        pick.addWidget(self.build_box, 1)
        refresh = QPushButton("REFRESH")
        refresh.clicked.connect(self.refresh_builds)
        pick.addWidget(refresh)
        new = QPushButton("NEW BUILD…")
        new.clicked.connect(self.new_build)
        pick.addWidget(new)
        outer.addLayout(pick)

        panel = QGroupBox("STATUS")
        grid = QGridLayout(panel)
        self.led = Led(state="off", diameter=16)
        self.state_label = QLabel("NO BUILD")
        self.state_label.setObjectName("readout")
        head = QHBoxLayout()
        head.addWidget(self.led)
        head.addWidget(self.state_label, 1)
        grid.addLayout(head, 0, 0, 1, 4)

        self.progress = QProgressBar()
        self.progress.setFormat("batch %v / %m")
        grid.addWidget(self.progress, 1, 0, 1, 4)

        self.fields: dict[str, QLabel] = {}
        rows = [("scenarios", "Scenarios written"), ("rate", "Per scenario"),
                ("eta", "Time left"), ("refused", "Refused"),
                ("config", "Build"), ("updated", "Last heartbeat")]
        for k, (key, label) in enumerate(rows):
            r, c = 2 + k // 2, (k % 2) * 2
            name = QLabel(label + ":")
            name.setObjectName("infoKey")
            value = QLabel("--")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(name, r, c)
            grid.addWidget(value, r, c + 1)
            self.fields[key] = value
        self.note = QLabel("")
        self.note.setObjectName("screenSubtitle")
        self.note.setWordWrap(True)
        grid.addWidget(self.note, 5, 0, 1, 4)

        buttons = QHBoxLayout()
        self.pause_btn = QPushButton("PAUSE")
        self.pause_btn.clicked.connect(self.pause)
        self.resume_btn = QPushButton("RESUME")
        self.resume_btn.setObjectName("runButton")
        self.resume_btn.clicked.connect(self.resume)
        buttons.addStretch(1)
        buttons.addWidget(self.pause_btn)
        buttons.addWidget(self.resume_btn)
        grid.addLayout(buttons, 6, 0, 1, 4)
        outer.addWidget(panel)

        lower = QHBoxLayout()
        lower.addWidget(self._build_plot(), 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lower.addWidget(self.log, 1)
        outer.addLayout(lower, 1)

    def _build_plot(self) -> QWidget:
        if not _HAS_PG:
            hint = QLabel("The throughput plot needs pyqtgraph.\n\n    pip install pyqtgraph")
            hint.setObjectName("screenSubtitle")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return hint
        pg.setConfigOption("background", _PLOT_BG)
        pg.setConfigOption("foreground", _PLOT_FG)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Time per scenario", units="s")
        self.plot.setLabel("bottom", "Batch (shortest scenarios first)")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.curve = self.plot.plot([], [], pen=pg.mkPen(_RATE_PEN, width=1.5))
        return self.plot

    # -------------------------------------------------------------- actions
    def current_out(self) -> Path | None:
        data = self.build_box.currentData()
        return Path(data) if data else None

    def refresh_builds(self, select: Path | None = None) -> None:
        keep = select or self.current_out()
        self.build_box.blockSignals(True)
        self.build_box.clear()
        for p in self.svc.list_builds():
            self.build_box.addItem(p.name, str(p))
        if keep is not None:
            i = self.build_box.findData(str(keep))
            if i >= 0:
                self.build_box.setCurrentIndex(i)
            elif select is not None:
                # Just started: no journal yet, so it is not in the list.
                self.build_box.insertItem(0, Path(select).name, str(select))
                self.build_box.setCurrentIndex(0)
        self.build_box.blockSignals(False)
        self.poll()

    def new_build(self) -> None:
        dlg = NewBuildDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            build = dlg.build()
            self.svc.start(build)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot start build", str(exc))
            return
        self.refresh_builds(select=build.out)

    def pause(self) -> None:
        out = self.current_out()
        if out is not None:
            self.svc.pause(out)
            self.poll()

    def resume(self) -> None:
        out = self.current_out()
        if out is None:
            return
        try:
            self.svc.resume(out)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot resume", str(exc))
        self.poll()

    # ----------------------------------------------------------------- poll
    def poll(self) -> None:
        out = self.current_out()
        status = self.svc.status(out) if out else None
        self._show_status(out, status)
        if out is not None:
            text = self.svc.log_tail(out)
            if out != self._shown_out or text != self.log.toPlainText():
                self.log.setPlainText(text)
                self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        else:
            self.log.clear()
        self._shown_out = out

    def _show_status(self, out: Path | None, s: dict | None) -> None:
        state = (s or {}).get("state")
        if out is None:
            label = "NO BUILD -- start one with NEW BUILD…"
        elif s is None:
            label = "STARTING…"
        else:
            label = state.upper()
            if state == "paused" and s.get("paused_because"):
                label += f"  ({s['paused_because']})"
        self.state_label.setText(label)
        self.led.set_state(_LED.get(state, "off"))

        s = s or {}
        self.progress.setMaximum(max(int(s.get("n_batches") or 0), 1))
        self.progress.setValue(int(s.get("batch_done") or 0))
        total = s.get("scenarios_total")
        self.fields["scenarios"].setText(
            f"{s.get('scenarios_written', 0):,} of {total:,} sampled" if total else "--")
        rate = s.get("ms_per_scenario")
        self.fields["rate"].setText(f"{rate:.0f} ms (last 20 batches)" if rate else "--")
        self.fields["eta"].setText(_duration(s.get("eta_s")) if state in ACTIVE else "--")
        ref = s.get("refused") or {}
        self.fields["refused"].setText(
            f"{ref.get('diverged', 0)} diverged, {ref.get('over_knuckle_ultimate', 0)} "
            f"over ultimate, {ref.get('reversed', 0)} reversed" if ref else "--")
        self.fields["config"].setText(
            f"{s.get('physics', '?')}, seed {s.get('seed', '?')}, batch "
            f"{s.get('batch_size', '?')}, {s.get('device', '?')}" if s else "--")
        upd = s.get("updated")
        self.fields["updated"].setText(f"{time.time() - upd:.0f} s ago" if upd else "--")
        self.note.setText(s.get("error", "") if state in ("stopped", "failed") else "")

        self.pause_btn.setEnabled(state in ("sampling", "running"))
        self.resume_btn.setEnabled(out is not None and state in ("paused", "stopped", "failed"))

        if _HAS_PG:
            times = s.get("batch_times") or []
            self.curve.setData([b for b, _, _ in times],
                               [t / max(n, 1) for _, n, t in times])
