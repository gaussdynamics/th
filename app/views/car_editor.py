"""CAR EDITOR -- a plain form dialog over the CarType fields.

Used by the Consist screen's New / Edit / Duplicate buttons. No Qt model,
just a form that returns a CarType (or None if cancelled).
"""

from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from simulator2.catalog import CarCategory, CarType


def _big_spin(maximum: float = 10_000_000.0, decimals: int = 0, suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(0.0, maximum)
    box.setDecimals(decimals)
    if suffix:
        box.setSuffix(suffix)
    return box


class CarEditorDialog(QDialog):
    def __init__(
        self,
        existing_ids: Iterable[str],
        coupler_ids: Iterable[str],
        car_type: Optional[CarType] = None,
        id_editable: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._existing_ids = set(existing_ids)
        self._editing_id = car_type.id if car_type else None
        self.setWindowTitle("Edit car type" if car_type else "New car type")

        self.id_edit = QLineEdit(car_type.id if car_type else "")
        self.id_edit.setEnabled(id_editable)
        self.name_edit = QLineEdit(car_type.name if car_type else "")

        self.category_box = QComboBox()
        for cat in CarCategory:
            self.category_box.addItem(cat.value, cat)
        if car_type:
            self.category_box.setCurrentIndex(self.category_box.findData(car_type.category))

        self.length_spin = _big_spin(500.0, 1, " m")
        self.length_spin.setValue(car_type.length_m if car_type else 15.0)

        self.mass_tare_spin = _big_spin(1_000_000.0, 0, " kg")
        self.mass_tare_spin.setValue(car_type.mass_tare_kg if car_type else 30_000.0)

        self.lading_edit = QLineEdit(car_type.lading if car_type else "")

        self.mass_lading_spin = _big_spin(1_000_000.0, 0, " kg")
        self.mass_lading_spin.setValue(car_type.mass_lading_kg if car_type else 0.0)

        self.davis_a_spin = _big_spin(100_000.0, 1, " N")
        self.davis_a_spin.setValue(car_type.davis_A if car_type else 600.0)
        self.davis_b_spin = _big_spin(10_000.0, 2, " N*s/m")
        self.davis_b_spin.setValue(car_type.davis_B if car_type else 12.0)
        self.davis_c_spin = _big_spin(1_000.0, 3, " N*s^2/m^2")
        self.davis_c_spin.setValue(car_type.davis_C if car_type else 0.7)

        self.can_traction_check = QCheckBox("Can apply traction")
        self.can_traction_check.setChecked(car_type.can_traction if car_type else False)

        self.f_trac_spin = _big_spin(1_000_000.0, 0, " N")
        self.f_trac_spin.setValue(car_type.F_trac_max_N if car_type else 0.0)

        self.f_brk_spin = _big_spin(1_000_000.0, 0, " N")
        self.f_brk_spin.setValue(car_type.F_brk_max_N if car_type else 250_000.0)

        self.coupler_box = QComboBox()
        self.coupler_box.addItems(list(coupler_ids))
        if car_type and car_type.coupler_type in coupler_ids:
            self.coupler_box.setCurrentText(car_type.coupler_type)

        self.notes_edit = QLineEdit(car_type.notes if car_type else "")

        form = QFormLayout()
        form.addRow("ID:", self.id_edit)
        form.addRow("Name:", self.name_edit)
        form.addRow("Category:", self.category_box)
        form.addRow("Length:", self.length_spin)
        form.addRow("Tare mass:", self.mass_tare_spin)
        form.addRow("Lading:", self.lading_edit)
        form.addRow("Lading mass:", self.mass_lading_spin)
        form.addRow("Davis A:", self.davis_a_spin)
        form.addRow("Davis B:", self.davis_b_spin)
        form.addRow("Davis C:", self.davis_c_spin)
        form.addRow(self.can_traction_check)
        form.addRow("Max traction:", self.f_trac_spin)
        form.addRow("Max brake:", self.f_brk_spin)
        form.addRow("Coupler type:", self.coupler_box)
        form.addRow("Notes:", self.notes_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self._result: Optional[CarType] = None

    def _on_accept(self) -> None:
        new_id = self.id_edit.text().strip()
        if not new_id:
            QMessageBox.warning(self, "Missing ID", "Car type ID cannot be empty.")
            return
        if new_id != self._editing_id and new_id in self._existing_ids:
            QMessageBox.warning(self, "Duplicate ID", f"Car type ID {new_id!r} already exists.")
            return
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Missing name", "Car type name cannot be empty.")
            return

        self._result = CarType(
            id=new_id,
            name=self.name_edit.text().strip(),
            category=self.category_box.currentData(),
            length_m=self.length_spin.value(),
            mass_tare_kg=self.mass_tare_spin.value(),
            lading=self.lading_edit.text().strip(),
            mass_lading_kg=self.mass_lading_spin.value(),
            davis_A=self.davis_a_spin.value(),
            davis_B=self.davis_b_spin.value(),
            davis_C=self.davis_c_spin.value(),
            can_traction=self.can_traction_check.isChecked(),
            F_trac_max_N=self.f_trac_spin.value(),
            F_brk_max_N=self.f_brk_spin.value(),
            coupler_type=self.coupler_box.currentText(),
            notes=self.notes_edit.text().strip(),
        )
        self.accept()

    @classmethod
    def edit(
        cls,
        parent,
        existing_ids: Iterable[str],
        coupler_ids: Iterable[str],
        car_type: Optional[CarType] = None,
        id_editable: bool = True,
    ) -> Optional[CarType]:
        dlg = cls(existing_ids, coupler_ids, car_type=car_type, id_editable=id_editable, parent=parent)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg._result
        return None
