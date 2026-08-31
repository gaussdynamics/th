"""CONSIST screen -- car-type library + train builder + saved trains (§9).

Three panes, left to right: the Car Library (saved CarTypes), the Train
Builder (an ordered, editable consist), and Saved Trains. Mirrors the
build-then-save workflow described in APP_STACK_SKETCH.md §9.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.services.catalog_service import CatalogService
from app.services.consist_service import ConsistService
from app.views.car_editor import CarEditorDialog
from simulator2.catalog import CarType
from simulator2.consist import CarInstance, Train

_COL_NUM, _COL_TYPE, _COL_LABEL, _COL_OVERRIDES, _COL_MASS = range(5)
_COLUMNS = ["#", "Car type", "Label", "Overrides", "Mass (kg)"]

_OVERRIDE_FIELDS = [
    ("mass_tare_kg", "Tare mass", " kg", 0, 1_000_000.0),
    ("mass_lading_kg", "Lading mass", " kg", 0, 1_000_000.0),
    ("davis_A", "Davis A", " N", 1, 100_000.0),
    ("davis_B", "Davis B", " N*s/m", 2, 10_000.0),
    ("davis_C", "Davis C", " N*s^2/m^2", 3, 1_000.0),
    ("F_trac_max_N", "Max traction", " N", 0, 1_000_000.0),
    ("F_brk_max_N", "Max brake", " N", 0, 1_000_000.0),
]


class CarOverridesDialog(QDialog):
    """Per-car tuning: check a field to override it, otherwise it follows the
    shared CarType. The escape hatch from §9 ("fatten one hopper without
    touching the type")."""

    def __init__(self, base_ct: CarType, current_overrides: Dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Overrides — {base_ct.name}")
        self._checks: Dict[str, QCheckBox] = {}
        self._spins: Dict[str, QDoubleSpinBox] = {}
        self._result: Optional[Dict] = None

        form = QFormLayout()
        for field, label, suffix, decimals, maximum in _OVERRIDE_FIELDS:
            chk = QCheckBox()
            chk.setChecked(field in current_overrides)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, maximum)
            spin.setDecimals(decimals)
            spin.setSuffix(suffix)
            spin.setValue(current_overrides.get(field, getattr(base_ct, field)))
            spin.setEnabled(chk.isChecked())
            chk.toggled.connect(spin.setEnabled)

            row = QWidget()
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.addWidget(chk)
            row_lay.addWidget(spin, 1)
            form.addRow(f"{label}:", row)

            self._checks[field] = chk
            self._spins[field] = spin

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self._result = {
            field: self._spins[field].value()
            for field, *_ in _OVERRIDE_FIELDS
            if self._checks[field].isChecked()
        }
        self.accept()

    @classmethod
    def edit(cls, parent, base_ct: CarType, current_overrides: Dict) -> Optional[Dict]:
        dlg = cls(base_ct, current_overrides, parent=parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg._result
        return None


class ConsistView(QWidget):
    def __init__(
        self,
        catalog: Optional[CatalogService] = None,
        consist: Optional[ConsistService] = None,
    ) -> None:
        super().__init__()
        self.consist = consist or ConsistService(catalog)
        self.catalog = catalog or self.consist.catalog
        self._train = Train(name="New Train", cars=[])

        self._build_ui()
        self._populate_library()
        self._populate_saved_trains()
        self._refresh_builder()

    # ---------- construction ----------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(8)

        title = QLabel("CONSIST")
        title.setObjectName("screenTitle")
        outer.addWidget(title)
        subtitle = QLabel(
            "Build trains from saved car types, tune each car, and save whole trains."
        )
        subtitle.setObjectName("screenSubtitle")
        outer.addWidget(subtitle)

        row = QHBoxLayout()
        row.addWidget(self._build_library_pane(), 1)
        row.addWidget(self._build_builder_pane(), 2)
        row.addWidget(self._build_saved_trains_pane(), 1)
        outer.addLayout(row, 1)

    def _build_library_pane(self) -> QGroupBox:
        box = QGroupBox("CAR LIBRARY")
        lay = QVBoxLayout(box)
        self.library_list = QListWidget()
        lay.addWidget(self.library_list, 1)

        btn_row = QHBoxLayout()
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._on_new_car_type)
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._on_edit_car_type)
        dup_btn = QPushButton("Duplicate")
        dup_btn.clicked.connect(self._on_duplicate_car_type)
        for b in (new_btn, edit_btn, dup_btn):
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

        add_btn = QPushButton("Add to train ▶")
        add_btn.setObjectName("runButton")
        add_btn.clicked.connect(self._on_add_car)
        lay.addWidget(add_btn)
        return box

    def _build_builder_pane(self) -> QGroupBox:
        box = QGroupBox("TRAIN BUILDER")
        lay = QVBoxLayout(box)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.train_name_edit = QLineEdit(self._train.name)
        self.train_name_edit.textChanged.connect(self._on_train_name_changed)
        name_row.addWidget(self.train_name_edit, 1)
        lay.addLayout(name_row)

        self.builder_table = QTableWidget(0, len(_COLUMNS))
        self.builder_table.setHorizontalHeaderLabels(_COLUMNS)
        self.builder_table.horizontalHeader().setSectionResizeMode(_COL_TYPE, QHeaderView.ResizeMode.Stretch)
        self.builder_table.verticalHeader().setVisible(False)
        self.builder_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.builder_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.builder_table.itemChanged.connect(self._on_table_item_changed)
        lay.addWidget(self.builder_table, 1)

        btn_row = QHBoxLayout()
        for text, handler in (
            ("Remove", self._on_remove_car),
            ("▲ Up", self._on_move_up),
            ("▼ Down", self._on_move_down),
            ("Overrides…", self._on_edit_overrides),
        ):
            b = QPushButton(text)
            b.clicked.connect(handler)
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

        self.totals_readout = QLabel("—")
        self.totals_readout.setObjectName("readout")
        lay.addWidget(self.totals_readout)

        new_btn = QPushButton("New train")
        new_btn.clicked.connect(self._on_new_train)
        lay.addWidget(new_btn)
        return box

    def _build_saved_trains_pane(self) -> QGroupBox:
        box = QGroupBox("SAVED TRAINS")
        lay = QVBoxLayout(box)
        self.saved_list = QListWidget()
        lay.addWidget(self.saved_list, 1)

        btn_row1 = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_load_train)
        save_btn = QPushButton("Save")
        save_btn.setObjectName("runButton")
        save_btn.clicked.connect(self._on_save_train)
        btn_row1.addWidget(load_btn)
        btn_row1.addWidget(save_btn)
        lay.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        dup_btn = QPushButton("Duplicate")
        dup_btn.clicked.connect(self._on_duplicate_train)
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._on_rename_train)
        btn_row2.addWidget(dup_btn)
        btn_row2.addWidget(rename_btn)
        lay.addLayout(btn_row2)
        return box

    # ---------- car library ----------
    def _populate_library(self) -> None:
        self.library_list.clear()
        for ct in self.catalog.list_car_types():
            item = QListWidgetItem(f"[{ct.category.value}] {ct.name}")
            item.setData(Qt.ItemDataRole.UserRole, ct.id)
            self.library_list.addItem(item)

    def _selected_library_id(self) -> Optional[str]:
        item = self.library_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _on_new_car_type(self) -> None:
        existing_ids = {ct.id for ct in self.catalog.list_car_types()}
        new_ct = CarEditorDialog.edit(self, existing_ids, self.catalog.list_coupler_ids())
        if new_ct is None:
            return
        self.catalog.save(new_ct)
        self._populate_library()

    def _on_edit_car_type(self) -> None:
        car_id = self._selected_library_id()
        if car_id is None:
            return
        ct = self.catalog.get(car_id)
        existing_ids = {c.id for c in self.catalog.list_car_types()} - {car_id}
        edited = CarEditorDialog.edit(
            self, existing_ids, self.catalog.list_coupler_ids(), car_type=ct, id_editable=False
        )
        if edited is None:
            return
        self.catalog.save(edited)
        self._populate_library()
        self._refresh_builder()

    def _on_duplicate_car_type(self) -> None:
        car_id = self._selected_library_id()
        if car_id is None:
            return
        ct = self.catalog.get(car_id)
        existing_ids = {c.id for c in self.catalog.list_car_types()}
        candidate = replace(ct, id=f"{ct.id}_copy", name=f"{ct.name} (copy)")
        n = 2
        while candidate.id in existing_ids:
            candidate = replace(candidate, id=f"{ct.id}_copy{n}")
            n += 1
        edited = CarEditorDialog.edit(
            self, existing_ids, self.catalog.list_coupler_ids(), car_type=candidate, id_editable=True
        )
        if edited is None:
            return
        self.catalog.save(edited)
        self._populate_library()

    # ---------- train builder ----------
    def _on_add_car(self) -> None:
        car_id = self._selected_library_id()
        if car_id is None:
            return
        self._train.cars.append(CarInstance(car_type_id=car_id))
        self._refresh_builder()

    def _on_remove_car(self) -> None:
        row = self.builder_table.currentRow()
        if row < 0 or row >= len(self._train.cars):
            return
        del self._train.cars[row]
        self._refresh_builder()

    def _on_move_up(self) -> None:
        row = self.builder_table.currentRow()
        if row <= 0:
            return
        cars = self._train.cars
        cars[row - 1], cars[row] = cars[row], cars[row - 1]
        self._refresh_builder(select_row=row - 1)

    def _on_move_down(self) -> None:
        row = self.builder_table.currentRow()
        if row < 0 or row >= len(self._train.cars) - 1:
            return
        cars = self._train.cars
        cars[row + 1], cars[row] = cars[row], cars[row + 1]
        self._refresh_builder(select_row=row + 1)

    def _on_edit_overrides(self) -> None:
        row = self.builder_table.currentRow()
        if row < 0 or row >= len(self._train.cars):
            return
        inst = self._train.cars[row]
        try:
            base_ct = self.catalog.get(inst.car_type_id)
        except KeyError as exc:
            QMessageBox.warning(self, "Unknown car type", str(exc))
            return
        new_overrides = CarOverridesDialog.edit(self, base_ct, inst.overrides)
        if new_overrides is None:
            return
        inst.overrides = new_overrides
        self._refresh_builder(select_row=row)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != _COL_LABEL:
            return
        row = item.row()
        if 0 <= row < len(self._train.cars):
            self._train.cars[row].label = item.text()

    def _on_train_name_changed(self, text: str) -> None:
        self._train.name = text

    def _on_new_train(self) -> None:
        name, ok = QInputDialog.getText(self, "New train", "Name:")
        if not ok or not name.strip():
            return
        self._train = Train(name=name.strip(), cars=[])
        self.train_name_edit.setText(self._train.name)
        self._refresh_builder()

    def _refresh_builder(self, select_row: int = -1) -> None:
        self.builder_table.blockSignals(True)
        self.builder_table.setRowCount(len(self._train.cars))
        total_mass = 0.0
        total_length = 0.0
        powered = 0
        for row, inst in enumerate(self._train.cars):
            try:
                ct = self.catalog.get(inst.car_type_id).with_overrides(inst.overrides)
                type_text = ct.name
                mass_text = f"{ct.mass_kg:,.0f}"
                total_mass += ct.mass_kg
                total_length += ct.length_m
                powered += 1 if ct.can_traction else 0
            except KeyError:
                type_text = f"?? {inst.car_type_id}"
                mass_text = "—"

            num_item = QTableWidgetItem(str(row + 1))
            num_item.setFlags(num_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.builder_table.setItem(row, _COL_NUM, num_item)

            type_item = QTableWidgetItem(type_text)
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.builder_table.setItem(row, _COL_TYPE, type_item)

            self.builder_table.setItem(row, _COL_LABEL, QTableWidgetItem(inst.label))

            overrides_text = ", ".join(f"{k}={v:g}" for k, v in inst.overrides.items()) or "—"
            overrides_item = QTableWidgetItem(overrides_text)
            overrides_item.setFlags(overrides_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.builder_table.setItem(row, _COL_OVERRIDES, overrides_item)

            mass_item = QTableWidgetItem(mass_text)
            mass_item.setFlags(mass_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.builder_table.setItem(row, _COL_MASS, mass_item)
        self.builder_table.blockSignals(False)

        if 0 <= select_row < self.builder_table.rowCount():
            self.builder_table.selectRow(select_row)

        self.totals_readout.setText(
            f"{len(self._train.cars)} cars · {total_mass:,.0f} kg · "
            f"{total_length:,.0f} m · {powered} powered"
        )

    # ---------- saved trains ----------
    def _populate_saved_trains(self) -> None:
        self.saved_list.clear()
        for path in self.consist.list_saved_trains():
            item = QListWidgetItem(path.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.saved_list.addItem(item)

    def _selected_saved_path(self) -> Optional[Path]:
        item = self.saved_list.currentItem()
        return Path(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _on_load_train(self) -> None:
        path = self._selected_saved_path()
        if path is None:
            return
        try:
            self._train = self.consist.load_train(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self.train_name_edit.setText(self._train.name)
        self._refresh_builder()

    def _on_save_train(self) -> None:
        if not self._train.name.strip():
            QMessageBox.warning(self, "Missing name", "Give the train a name before saving.")
            return
        self.consist.save_train(self._train)
        self._populate_saved_trains()

    def _on_duplicate_train(self) -> None:
        path = self._selected_saved_path()
        if path is None:
            return
        train = self.consist.load_train(path)
        new_name, ok = QInputDialog.getText(self, "Duplicate train", "New name:", text=f"{train.name} copy")
        if not ok or not new_name.strip():
            return
        self.consist.duplicate_train(path, new_name.strip())
        self._populate_saved_trains()

    def _on_rename_train(self) -> None:
        path = self._selected_saved_path()
        if path is None:
            return
        train = self.consist.load_train(path)
        new_name, ok = QInputDialog.getText(self, "Rename train", "New name:", text=train.name)
        if not ok or not new_name.strip():
            return
        self.consist.rename_train(path, new_name.strip())
        self._populate_saved_trains()
