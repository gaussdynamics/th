"""Trains: ordered car instances built from the catalog (§9 of APP_STACK_SKETCH.md).

``build_consist()`` is the single bridge back to the existing engine: everything
downstream (``make_scenario_from_consist``, ``simulate_train_tensorized``) only
ever sees plain ``VehicleParameters`` / ``CouplerParameters`` lists, unchanged.

Pure stdlib (dataclasses + json) — no numpy, no Qt.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .catalog import CarLibrary, CarType
from .params import CouplerParameters, VehicleParameters

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAVED_TRAINS_DIR = _REPO_ROOT / "saved_trains"


@dataclass
class CarInstance:
    car_type_id: str  # ref into the car library
    overrides: Dict = field(default_factory=dict)  # per-car tweaks, e.g. {"mass_lading_kg": 80_000}
    label: str = ""  # optional, e.g. "head-end helper"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CarInstance":
        return cls(
            car_type_id=d["car_type_id"],
            overrides=dict(d.get("overrides") or {}),
            label=d.get("label", ""),
        )


@dataclass
class Train:
    name: str
    cars: List[CarInstance]  # ORDERED, head -> tail
    description: str = ""
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "cars": [c.to_dict() for c in self.cars],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Train":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            cars=[CarInstance.from_dict(c) for c in d.get("cars", [])],
            schema_version=int(d.get("schema_version", 1)),
        )

    def save(self, directory: Path | str = DEFAULT_SAVED_TRAINS_DIR, filename: Optional[str] = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        stem = filename or slugify(self.name)
        path = d / f"{stem}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Train":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def slugify(name: str) -> str:
    return "_".join(name.lower().split())


def list_saved_trains(directory: Path | str = DEFAULT_SAVED_TRAINS_DIR) -> List[Path]:
    d = Path(directory)
    return sorted(d.glob("*.json")) if d.exists() else []


def from_blocks(name: str, blocks: List[Tuple[str, int]], description: str = "") -> Train:
    """Build a train ergonomically: ``blocks = [("loco_sd70", 1), ("coal_hopper_loaded", 4)]``."""
    cars = [CarInstance(car_type_id=tid) for tid, n in blocks for _ in range(n)]
    return Train(name=name, cars=cars, description=description)


def _resolve_car_types(train: Train, library: CarLibrary) -> List[CarType]:
    resolved = []
    for inst in train.cars:
        ct = library.get(inst.car_type_id)
        resolved.append(ct.with_overrides(inst.overrides))
    return resolved


def build_consist(
    train: Train, library: CarLibrary
) -> Tuple[List[VehicleParameters], List[CouplerParameters]]:
    """The single bridge back to the existing engine.

    One coupler sits between each adjacent pair of cars, tuned by the *lead*
    car's ``coupler_type``; its rest length defaults to the mean of the two
    adjacent cars' ``length_m`` unless the coupler type pins its own ``L0_m``.
    """
    if len(train.cars) < 1:
        raise ValueError("Train must have at least one car")

    resolved = _resolve_car_types(train, library)
    vehicles = [ct.to_vehicle() for ct in resolved]

    couplers: List[CouplerParameters] = []
    for lead_ct, trail_ct in zip(resolved, resolved[1:]):
        coupler_type = library.get_coupler(lead_ct.coupler_type)
        l0_m = 0.5 * (lead_ct.length_m + trail_ct.length_m)
        couplers.append(coupler_type.to_coupler(l0_m))

    return vehicles, couplers
