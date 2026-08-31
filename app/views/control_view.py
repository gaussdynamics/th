"""CONTROLS screen -- generate and store randomized traction/brake command
profiles (§ control_profile.py). A console like the others: a generator panel
on the left (randomization ranges + Randomize/Save), saved profiles below it,
and a live preview plot on the right.

pyqtgraph is imported defensively so the app still launches without it,
matching route_view/results_view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.services.control_service import ControlService
from simulator2.control_profile import (
    ROLE_DPU_MID,
    ROLE_DPU_REAR,
    ROLE_HEAD_END,
    ControlProfile,
    RandomProfileConfig,
)

try:
    import pyqtgraph as pg

    _HAS_PG = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_PG = False

_PLOT_BG = "#b0b0b0"
_PLOT_FG = "#2b2b2b"
_BRAKE_PEN = "#b5651d"
_ROLE_PENS = {
    ROLE_HEAD_END: "#3a5e8c",
    ROLE_DPU_MID: "#3b6e3b",
    ROLE_DPU_REAR: "#8c3a5e",
}
_N_PREVIEW_SAMPLES = 401


class ControlView(QWidget):
    def __init__(self, control_service: Optional[ControlService] = None) -> None:
        super().__init__()
        self.svc = control_service or ControlService()
        self._profile = ControlProfile(name="Untitled control profile")

        self._build_ui()
        self._populate_saved()
        self._update_preview()

    # ---------- construction ----------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(8)

        title = QLabel("CONTROLS")
        title.setObjectName("screenTitle")
        outer.addWidget(title)
        subtitle = QLabel(
            "Generate randomized traction/brake command profiles and store them "
            "for the Simulate screen."
        )
        subtitle.setObjectName("screenSubtitle")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        row = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self._build_generator_pane())
        left.addWidget(self._build_saved_pane(), 1)
        row.addLayout(left, 1)
        row.addWidget(self._build_plot_area(), 2)
        outer.addLayout(row, 1)

    def _build_generator_pane(self) -> QGroupBox:
        box = QGroupBox("GENERATOR")
        col = QVBoxLayout(box)
        form = QFormLayout()

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 3600.0)
        self.duration_spin.setDecimals(0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setValue(120.0)
        form.addRow("Duration:", self.duration_spin)

        knots_row = QHBoxLayout()
        self.knots_min_spin = QSpinBox()
        self.knots_min_spin.setRange(2, 30)
        self.knots_min_spin.setValue(3)
        self.knots_max_spin = QSpinBox()
        self.knots_max_spin.setRange(2, 30)
        self.knots_max_spin.setValue(6)
        knots_row.addWidget(self.knots_min_spin)
        knots_row.addWidget(QLabel("to"))
        knots_row.addWidget(self.knots_max_spin)
        form.addRow("Knots per curve:", knots_row)

        trac_row = QHBoxLayout()
        self.trac_min_spin = QDoubleSpinBox()
        self.trac_min_spin.setRange(0.0, 1.0)
        self.trac_min_spin.setDecimals(2)
        self.trac_min_spin.setSingleStep(0.05)
        self.trac_min_spin.setValue(0.0)
        self.trac_max_spin = QDoubleSpinBox()
        self.trac_max_spin.setRange(0.0, 1.0)
        self.trac_max_spin.setDecimals(2)
        self.trac_max_spin.setSingleStep(0.05)
        self.trac_max_spin.setValue(1.0)
        trac_row.addWidget(self.trac_min_spin)
        trac_row.addWidget(QLabel("to"))
        trac_row.addWidget(self.trac_max_spin)
        form.addRow("Traction fraction:", trac_row)

        brk_row = QHBoxLayout()
        self.brk_min_spin = QDoubleSpinBox()
        self.brk_min_spin.setRange(0.0, 1.0)
        self.brk_min_spin.setDecimals(2)
        self.brk_min_spin.setSingleStep(0.05)
        self.brk_min_spin.setValue(0.0)
        self.brk_max_spin = QDoubleSpinBox()
        self.brk_max_spin.setRange(0.0, 1.0)
        self.brk_max_spin.setDecimals(2)
        self.brk_max_spin.setSingleStep(0.05)
        self.brk_max_spin.setValue(0.6)
        brk_row.addWidget(self.brk_min_spin)
        brk_row.addWidget(QLabel("to"))
        brk_row.addWidget(self.brk_max_spin)
        form.addRow("Brake fraction:", brk_row)

        self.dpu_mode_box = QComboBox()
        self.dpu_mode_box.addItems(["synced", "independent", "off"])
        form.addRow("Distributed power:", self.dpu_mode_box)

        self.dpu_lag_spin = QDoubleSpinBox()
        self.dpu_lag_spin.setRange(0.0, 60.0)
        self.dpu_lag_spin.setSuffix(" s")
        self.dpu_lag_spin.setValue(3.0)
        form.addRow("DPU lag:", self.dpu_lag_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setValue(0)
        form.addRow("Seed:", self.seed_spin)

        col.addLayout(form)

        randomize_btn = QPushButton("RANDOMIZE")
        randomize_btn.setObjectName("runButton")
        randomize_btn.clicked.connect(self._on_randomize)
        col.addWidget(randomize_btn)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(self._profile.name)
        name_row.addWidget(self.name_edit, 1)
        col.addLayout(name_row)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save)
        col.addWidget(save_btn)

        self.readout = QLabel("—")
        self.readout.setObjectName("readout")
        self.readout.setWordWrap(True)
        col.addWidget(self.readout)

        return box

    def _build_saved_pane(self) -> QGroupBox:
        box = QGroupBox("SAVED PROFILES")
        lay = QVBoxLayout(box)
        self.saved_list = QListWidget()
        lay.addWidget(self.saved_list, 1)

        btn_row1 = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load)
        dup_btn = QPushButton("Duplicate")
        dup_btn.clicked.connect(self._on_duplicate)
        btn_row1.addWidget(load_btn)
        btn_row1.addWidget(dup_btn)
        lay.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._on_rename)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._on_delete)
        btn_row2.addWidget(rename_btn)
        btn_row2.addWidget(delete_btn)
        lay.addLayout(btn_row2)
        return box

    def _build_plot_area(self) -> QWidget:
        if not _HAS_PG:
            hint = QLabel("Preview plots need pyqtgraph.\n\n    pip install pyqtgraph")
            hint.setObjectName("screenSubtitle")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return hint

        pg.setConfigOption("background", _PLOT_BG)
        pg.setConfigOption("foreground", _PLOT_FG)
        pg.setConfigOptions(antialias=True)

        self.glw = pg.GraphicsLayoutWidget()
        self.p_trac = self.glw.addPlot(row=0, col=0)
        self.p_trac.setLabel("left", "Traction", units="frac")
        self.p_trac.addLegend(offset=(10, 10))
        self.p_trac.showGrid(x=True, y=True, alpha=0.25)

        self.p_brake = self.glw.addPlot(row=1, col=0)
        self.p_brake.setLabel("left", "Brake", units="frac")
        self.p_brake.setLabel("bottom", "Time", units="s")
        self.p_brake.showGrid(x=True, y=True, alpha=0.25)
        self.p_brake.setXLink(self.p_trac)
        return self.glw

    # ---------- generation ----------
    def _on_randomize(self) -> None:
        config = RandomProfileConfig(
            seed=self.seed_spin.value(),
            duration_s=self.duration_spin.value(),
            n_knots_range=(self.knots_min_spin.value(), self.knots_max_spin.value()),
            traction_fraction_range=(self.trac_min_spin.value(), self.trac_max_spin.value()),
            brake_fraction_range=(self.brk_min_spin.value(), self.brk_max_spin.value()),
            dpu_mode=self.dpu_mode_box.currentText(),
            dpu_lag_s=self.dpu_lag_spin.value(),
        )
        self._profile = self.svc.generate_random(config)
        self.name_edit.setText(self._profile.name)
        self.seed_spin.setValue(config.seed + 1)  # next click explores a new profile
        self._update_preview()

    def _update_preview(self) -> None:
        n_head = len(self._profile.traction_head_end.knots)
        n_brake = len(self._profile.brake.knots)
        self.readout.setText(
            f"{self._profile.description or self._profile.name}\n"
            f"{n_head} head-end knots · {n_brake} brake knots · {self._profile.duration_s:.0f}s"
        )
        if not _HAS_PG:
            return
        duration = max(1.0, self._profile.duration_s)
        ts = [i * duration / (_N_PREVIEW_SAMPLES - 1) for i in range(_N_PREVIEW_SAMPLES)]

        self.p_trac.clear()
        for role, curve in (
            (ROLE_HEAD_END, self._profile.traction_head_end),
            (ROLE_DPU_MID, self._profile.traction_dpu_mid),
            (ROLE_DPU_REAR, self._profile.traction_dpu_rear),
        ):
            self.p_trac.plot(ts, curve.sample(ts), pen=pg.mkPen(_ROLE_PENS[role], width=2), name=role)

        self.p_brake.clear()
        self.p_brake.plot(ts, self._profile.brake.sample(ts), pen=pg.mkPen(_BRAKE_PEN, width=2))

    # ---------- saved profiles ----------
    def _populate_saved(self) -> None:
        self.saved_list.clear()
        for path in self.svc.list_saved():
            item = QListWidgetItem(path.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.saved_list.addItem(item)

    def _selected_saved_path(self) -> Optional[Path]:
        item = self.saved_list.currentItem()
        return Path(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _on_save(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing name", "Give the profile a name before saving.")
            return
        self._profile.name = name
        self.svc.save(self._profile)
        self._populate_saved()

    def _on_load(self) -> None:
        path = self._selected_saved_path()
        if path is None:
            return
        try:
            self._profile = self.svc.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self.name_edit.setText(self._profile.name)
        self.duration_spin.setValue(self._profile.duration_s)
        if self._profile.seed is not None:
            self.seed_spin.setValue(self._profile.seed)
        self._update_preview()

    def _on_duplicate(self) -> None:
        path = self._selected_saved_path()
        if path is None:
            return
        profile = self.svc.load(path)
        new_name, ok = QInputDialog.getText(self, "Duplicate profile", "New name:", text=f"{profile.name} copy")
        if not ok or not new_name.strip():
            return
        self.svc.duplicate(path, new_name.strip())
        self._populate_saved()

    def _on_rename(self) -> None:
        path = self._selected_saved_path()
        if path is None:
            return
        profile = self.svc.load(path)
        new_name, ok = QInputDialog.getText(self, "Rename profile", "New name:", text=profile.name)
        if not ok or not new_name.strip():
            return
        self.svc.rename(path, new_name.strip())
        self._populate_saved()

    def _on_delete(self) -> None:
        path = self._selected_saved_path()
        if path is None:
            return
        if QMessageBox.question(self, "Delete profile", f"Delete {path.stem}?") != QMessageBox.StandardButton.Yes:
            return
        self.svc.delete(path)
        self._populate_saved()
