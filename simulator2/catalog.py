"""Car-type and coupler-type library (§9 of APP_STACK_SKETCH.md).

``CarType`` and ``CouplerType`` are named, reusable templates that know how to
emit the existing ``VehicleParameters`` / ``CouplerParameters``. ``CarLibrary``
loads/saves them as plain JSON, one file per item, mirroring
``route_generator/saved_routes/``.

Pure stdlib (dataclasses + json) — no numpy, no Qt. ``consist.py`` is the only
module that reaches into here from the app side.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from . import constants as C
from .params import CouplerParameters, VehicleParameters

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAR_LIBRARY_DIR = _REPO_ROOT / "car_library"


class CarCategory(str, Enum):
    LOCOMOTIVE = "locomotive"
    HOPPER = "hopper"  # coal, ballast
    TANK = "tank"  # liquids
    BOXCAR = "boxcar"
    FLATCAR = "flatcar"
    GONDOLA = "gondola"


@dataclass
class CarType:
    """A reusable car template. Mass is split tare + lading so 'tank car filled
    with water' is just a lading value on the same shell."""

    id: str  # "tank_car_water"  (filename stem)
    name: str  # "Tank car — water (full)"
    category: CarCategory
    length_m: float  # physical length -> default coupler spacing (see build_consist)

    mass_tare_kg: float  # empty car
    lading: str = "empty"  # "water", "coal", "" ...
    mass_lading_kg: float = 0.0

    davis_A: float = 0.0  # rolling/mech resistance (N)
    davis_B: float = 0.0  # (N*s/m)
    davis_C: float = 0.0  # aero (N*s^2/m^2)

    can_traction: bool = False
    F_trac_max_N: float = 0.0
    F_brk_max_N: float = 250_000.0

    coupler_type: str = "standard_knuckle"  # ref into COUPLER_TYPES
    notes: str = ""
    schema_version: int = 1

    @property
    def mass_kg(self) -> float:
        return self.mass_tare_kg + self.mass_lading_kg

    def to_vehicle(self) -> VehicleParameters:
        return VehicleParameters(
            mass_kg=self.mass_kg,
            davis_A=self.davis_A,
            davis_B=self.davis_B,
            davis_C=self.davis_C,
            can_traction=self.can_traction,
            F_trac_max_N=self.F_trac_max_N,
            F_brk_max_N=self.F_brk_max_N,
        )

    def with_overrides(self, overrides: Optional[Dict]) -> "CarType":
        """Return a copy with per-instance field overrides applied (e.g. a
        fatter lading on one hopper without touching the shared type)."""
        if not overrides:
            return self
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown CarType override field(s): {sorted(unknown)}")
        return replace(self, **overrides)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CarType":
        d = dict(d)
        d.pop("schema_version", None)
        d["category"] = CarCategory(d["category"])
        return cls(schema_version=1, **d)


@dataclass
class CouplerType:
    """A reusable, named coupler tuning. ``L0_m`` is the coupler's own rest
    length if it has one; leave ``None`` to derive the pairwise rest length
    from the two adjacent cars' ``length_m`` instead (the common case)."""

    id: str
    name: str
    L0_m: Optional[float] = None
    slack_half_m: float = C.DEFAULT_SLACK_HALF_M
    k_draft: float = C.DEFAULT_K_DRAFT
    c_draft: float = C.DEFAULT_C_DRAFT
    k_buff: float = C.DEFAULT_K_BUFF
    c_buff: float = C.DEFAULT_C_BUFF

    def to_coupler(self, l0_m: float) -> CouplerParameters:
        """``l0_m`` is used when this type doesn't pin its own ``L0_m``."""
        return CouplerParameters(
            L0_m=self.L0_m if self.L0_m is not None else l0_m,
            slack_half_m=self.slack_half_m,
            k_draft=self.k_draft,
            c_draft=self.c_draft,
            k_buff=self.k_buff,
            c_buff=self.c_buff,
        )


# Coupler tunings are few and rarely edited, so they ship as a small built-in
# registry rather than a JSON directory (unlike car types, which are the thing
# users actually build out).
DEFAULT_COUPLER_TYPES: Dict[str, CouplerType] = {
    "standard_knuckle": CouplerType(id="standard_knuckle", name="Standard knuckle coupler"),
}


def default_seed_car_types() -> List[CarType]:
    """Built from the existing constants, so the app isn't empty on first launch."""
    return [
        CarType(
            id="loco_sd70",
            name="Locomotive — SD70",
            category=CarCategory.LOCOMOTIVE,
            length_m=22.0,
            mass_tare_kg=C.DEFAULT_M_LOCO_KG,
            lading="",
            mass_lading_kg=0.0,
            davis_A=C.DEFAULT_DAVIS_A_LOCO,
            davis_B=C.DEFAULT_DAVIS_B_LOCO,
            davis_C=C.DEFAULT_DAVIS_C_LOCO,
            can_traction=True,
            F_trac_max_N=300_000.0,
            notes="Default seed locomotive.",
        ),
        CarType(
            id="freight_car_generic",
            name="Freight car — generic",
            category=CarCategory.BOXCAR,
            length_m=15.0,
            mass_tare_kg=C.DEFAULT_M_CAR_KG,
            lading="",
            mass_lading_kg=0.0,
            davis_A=C.DEFAULT_DAVIS_A_CAR,
            davis_B=C.DEFAULT_DAVIS_B_CAR,
            davis_C=C.DEFAULT_DAVIS_C_CAR,
            notes="Default seed freight car (no tare/lading split).",
        ),
        CarType(
            id="coal_hopper_loaded",
            name="Hopper — coal (loaded)",
            category=CarCategory.HOPPER,
            length_m=12.0,
            mass_tare_kg=30_000.0,
            lading="coal",
            mass_lading_kg=70_000.0,
            davis_A=C.DEFAULT_DAVIS_A_CAR,
            davis_B=C.DEFAULT_DAVIS_B_CAR,
            davis_C=C.DEFAULT_DAVIS_C_CAR,
            notes="Worked example of the tare+lading split.",
        ),
        CarType(
            id="tank_car_water",
            name="Tank car — water (full)",
            category=CarCategory.TANK,
            length_m=18.0,
            mass_tare_kg=30_000.0,
            lading="water",
            mass_lading_kg=80_000.0,
            davis_A=600.0,
            davis_B=12.0,
            davis_C=0.7,
            notes="Worked example of the tare+lading split.",
        ),
    ]


class CarLibrary:
    """In-memory car-type catalog, backed by one JSON file per type."""

    def __init__(self, car_types: Optional[Dict[str, CarType]] = None,
                 coupler_types: Optional[Dict[str, CouplerType]] = None) -> None:
        self._cars: Dict[str, CarType] = dict(car_types or {})
        self._couplers: Dict[str, CouplerType] = dict(coupler_types or DEFAULT_COUPLER_TYPES)

    # ---------- lookup ----------
    def get(self, car_type_id: str) -> CarType:
        try:
            return self._cars[car_type_id]
        except KeyError:
            raise KeyError(f"Unknown car type id: {car_type_id!r}") from None

    def get_coupler(self, coupler_type_id: str) -> CouplerType:
        try:
            return self._couplers[coupler_type_id]
        except KeyError:
            raise KeyError(f"Unknown coupler type id: {coupler_type_id!r}") from None

    def list_car_types(self) -> List[CarType]:
        return sorted(self._cars.values(), key=lambda ct: (ct.category.value, ct.id))

    def list_coupler_ids(self) -> List[str]:
        return sorted(self._couplers)

    def has(self, car_type_id: str) -> bool:
        return car_type_id in self._cars

    # ---------- mutation ----------
    def add(self, car_type: CarType) -> None:
        self._cars[car_type.id] = car_type

    def register_coupler(self, coupler_type: CouplerType) -> None:
        self._couplers[coupler_type.id] = coupler_type

    # ---------- persistence ----------
    def save(self, directory: Path | str = DEFAULT_CAR_LIBRARY_DIR, car_type_id: Optional[str] = None) -> None:
        """Write one file, or (if ``car_type_id`` is None) the whole library."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        ids = [car_type_id] if car_type_id else list(self._cars)
        for cid in ids:
            path = d / f"{cid}.json"
            path.write_text(json.dumps(self._cars[cid].to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str = DEFAULT_CAR_LIBRARY_DIR) -> "CarLibrary":
        d = Path(directory)
        cars: Dict[str, CarType] = {}
        if d.exists():
            for path in sorted(d.glob("*.json")):
                ct = CarType.from_dict(json.loads(path.read_text(encoding="utf-8")))
                cars[ct.id] = ct
        return cls(car_types=cars)

    @classmethod
    def load_or_seed(cls, directory: Path | str = DEFAULT_CAR_LIBRARY_DIR) -> "CarLibrary":
        """Load the library, seeding ``directory`` with the default car types
        first if it doesn't exist yet (first-launch convenience)."""
        d = Path(directory)
        if not d.exists() or not any(d.glob("*.json")):
            lib = cls(car_types={ct.id: ct for ct in default_seed_car_types()})
            lib.save(d)
            return lib
        return cls.load(d)
