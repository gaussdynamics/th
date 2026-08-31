"""Train build/load/save adapter for the Consist screen.

Thin wrapper over ``simulator2.consist`` -- no Qt in here, easy to unit test.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from app.services.catalog_service import CatalogService
from simulator2.consist import (
    CarInstance,
    DEFAULT_SAVED_TRAINS_DIR,
    Train,
    build_consist,
    list_saved_trains,
    slugify,
)
from simulator2.params import CouplerParameters, VehicleParameters


class ConsistService:
    def __init__(
        self,
        catalog: Optional[CatalogService] = None,
        trains_dir: Path | str = DEFAULT_SAVED_TRAINS_DIR,
    ) -> None:
        self.catalog = catalog or CatalogService()
        self.trains_dir = Path(trains_dir)

    def list_saved_trains(self) -> List[Path]:
        return list_saved_trains(self.trains_dir)

    def load_train(self, path: Path | str) -> Train:
        return Train.load(path)

    def save_train(self, train: Train, filename: Optional[str] = None) -> Path:
        return train.save(self.trains_dir, filename)

    def delete_train(self, path: Path | str) -> None:
        Path(path).unlink(missing_ok=True)

    def duplicate_train(self, path: Path | str, new_name: str) -> Path:
        train = self.load_train(path)
        dup = Train(
            name=new_name,
            description=train.description,
            cars=[CarInstance(c.car_type_id, dict(c.overrides), c.label) for c in train.cars],
        )
        return self.save_train(dup)

    def rename_train(self, path: Path | str, new_name: str) -> Path:
        """Save the train under its new name/filename, then remove the old file."""
        old_path = Path(path)
        train = self.load_train(old_path)
        train.name = new_name
        new_path = self.save_train(train)
        if new_path != old_path:
            self.delete_train(old_path)
        return new_path

    def build_consist(self, train: Train) -> Tuple[List[VehicleParameters], List[CouplerParameters]]:
        return build_consist(train, self.catalog.library)
