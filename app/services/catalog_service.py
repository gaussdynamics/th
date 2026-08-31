"""Car-type library adapter for the Consist screen.

Thin wrapper over ``simulator2.catalog.CarLibrary`` -- no Qt in here, easy to
unit test. Seeds ``car_library/`` with the default car types on first use.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List

from simulator2.catalog import CarLibrary, CarType, DEFAULT_CAR_LIBRARY_DIR


class CatalogService:
    def __init__(self, directory: Path | str = DEFAULT_CAR_LIBRARY_DIR) -> None:
        self.directory = Path(directory)
        self.library = CarLibrary.load_or_seed(self.directory)

    def list_car_types(self) -> List[CarType]:
        return self.library.list_car_types()

    def list_coupler_ids(self) -> List[str]:
        return self.library.list_coupler_ids()

    def get(self, car_type_id: str) -> CarType:
        return self.library.get(car_type_id)

    def save(self, car_type: CarType) -> None:
        self.library.add(car_type)
        self.library.save(self.directory, car_type.id)

    def duplicate(self, car_type_id: str, new_id: str, new_name: str) -> CarType:
        if self.library.has(new_id):
            raise ValueError(f"Car type id already exists: {new_id!r}")
        new_ct = replace(self.library.get(car_type_id), id=new_id, name=new_name)
        self.save(new_ct)
        return new_ct
