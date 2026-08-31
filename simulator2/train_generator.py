"""Procedural freight-consist generator: the "many trains" side of the
"many trains, few routes" world-model experiment (see PROJECT_DIRECTION.md,
DATA_SCHEMA.md §I). Produces reproducible ``Train`` objects and saves them
through the existing ``Train.save`` / ``CarLibrary.save`` path, so
``list_saved_trains`` and ``build_consist`` need no changes downstream.

Pure stdlib + ``random`` — no numpy, no Qt (matches ``consist.py``/``catalog.py``).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .catalog import CarLibrary, CarType, CarCategory, DEFAULT_CAR_LIBRARY_DIR
from .consist import CarInstance, DEFAULT_SAVED_TRAINS_DIR, Train, slugify

# Manifest lives one level under the trains directory so its filename never
# collides with a saved train and never gets picked up by
# ``list_saved_trains``'s flat, non-recursive ``glob("*.json")``.
MANIFEST_SUBDIR = "generated"
MANIFEST_FILENAME = "manifest.json"

# Rough power-to-tonnage sanity check: required tractive force per kg of
# trailing mass to handle a sustained ruling grade + rolling resistance
# (illustrative, not calibrated to real stock -- see PROJECT_DIRECTION.md).
DEFAULT_SPECIFIC_TRAC_FORCE_N_PER_KG = 0.12

# Categories that commonly run as dedicated unit trains get a higher weight
# when picking the dominant car type for a unit train.
_UNIT_TRAIN_CATEGORY_WEIGHT = {
    CarCategory.HOPPER: 2.0,
    CarCategory.TANK: 2.0,
    CarCategory.GONDOLA: 2.0,
    CarCategory.BOXCAR: 1.0,
    CarCategory.FLATCAR: 1.0,
}


def extra_seed_car_types() -> List[CarType]:
    """A handful of additional realistic ``CarType`` templates, on top of
    ``catalog.default_seed_car_types()``, so generated consists have real
    variety: more bulk-commodity cars, a boxcar/intermodal type, and a
    second locomotive for power-mix diversity."""
    return [
        CarType(
            id="loco_gp38",
            name="Locomotive — GP38 (road switcher)",
            category=CarCategory.LOCOMOTIVE,
            length_m=18.6,
            mass_tare_kg=115_000.0,
            lading="",
            mass_lading_kg=0.0,
            davis_A=750.0,
            davis_B=14.0,
            davis_C=0.75,
            can_traction=True,
            F_trac_max_N=180_000.0,
            F_brk_max_N=220_000.0,
            notes="Second, lower-power road unit for shorter/manifest trains.",
        ),
        CarType(
            id="grain_hopper_loaded",
            name="Covered hopper — grain (loaded)",
            category=CarCategory.HOPPER,
            length_m=17.0,
            mass_tare_kg=29_000.0,
            lading="grain",
            mass_lading_kg=91_000.0,
            davis_A=620.0,
            davis_B=12.5,
            davis_C=0.72,
            notes="Covered hopper, grain unit-train car.",
        ),
        CarType(
            id="ballast_hopper_loaded",
            name="Open hopper — ballast (loaded)",
            category=CarCategory.HOPPER,
            length_m=11.0,
            mass_tare_kg=26_000.0,
            lading="ballast",
            mass_lading_kg=100_000.0,
            davis_A=600.0,
            davis_B=12.0,
            davis_C=0.65,
            notes="Open triple hopper, aggregate/ballast unit-train car.",
        ),
        CarType(
            id="boxcar_manifest_loaded",
            name="Boxcar — mixed manifest freight (loaded)",
            category=CarCategory.BOXCAR,
            length_m=17.0,
            mass_tare_kg=32_000.0,
            lading="mixed freight",
            mass_lading_kg=55_000.0,
            davis_A=650.0,
            davis_B=13.0,
            davis_C=0.85,
            notes="General manifest boxcar.",
        ),
        CarType(
            id="intermodal_well_car_loaded",
            name="Well car — double-stack intermodal (loaded)",
            category=CarCategory.FLATCAR,
            length_m=27.0,
            mass_tare_kg=36_000.0,
            lading="containers",
            mass_lading_kg=48_000.0,
            davis_A=550.0,
            davis_B=10.0,
            davis_C=0.9,
            notes="Double-stack well car; loading varies with container/TEU count.",
        ),
        CarType(
            id="gondola_scrap_loaded",
            name="Gondola — scrap steel (loaded)",
            category=CarCategory.GONDOLA,
            length_m=16.0,
            mass_tare_kg=27_000.0,
            lading="scrap steel",
            mass_lading_kg=93_000.0,
            davis_A=610.0,
            davis_B=12.0,
            davis_C=0.6,
            notes="Open gondola, scrap/aggregate unit-train car.",
        ),
    ]


def ensure_extended_car_library(directory: Path | str = DEFAULT_CAR_LIBRARY_DIR) -> CarLibrary:
    """Load (seeding on first use) the car library, then add any of
    ``extra_seed_car_types()`` that aren't already present, persisting
    additions through the normal ``CarLibrary.save`` path."""
    library = CarLibrary.load_or_seed(directory)
    added = False
    for ct in extra_seed_car_types():
        if not library.has(ct.id):
            library.add(ct)
            added = True
    if added:
        library.save(directory)
    return library


@dataclass
class GeneratorConfig:
    count: int = 100
    seed: int = 0

    train_n_range: Tuple[int, int] = (10, 80)
    ood_n_range: Tuple[int, int] = (120, 150)
    ood_fraction: float = 0.15  # fraction of trains drawn from the held-out size band

    unit_train_fraction: float = 0.6  # fraction of trains that are single-commodity blocks
    distributed_power_fraction: float = 0.25  # fraction of trains using mid/rear DPUs

    loco_count_range: Tuple[int, int] = (1, 4)
    specific_trac_force_n_per_kg: float = DEFAULT_SPECIFIC_TRAC_FORCE_N_PER_KG

    load_state_weights: Dict[str, float] = field(
        default_factory=lambda: {"loaded": 0.5, "partial": 0.25, "empty": 0.25}
    )

    car_library_dir: Path = DEFAULT_CAR_LIBRARY_DIR
    out_dir: Path = DEFAULT_SAVED_TRAINS_DIR
    name_prefix: str = "Generated"


def _load_fraction(rng: random.Random, car_type: CarType, weights: Dict[str, float]) -> Optional[float]:
    """Pick a per-car ``mass_lading_kg`` override, or ``None`` for no override
    (car types with no tare/lading split, e.g. locomotives, are left alone)."""
    if car_type.mass_lading_kg <= 0:
        return None
    states = list(weights.keys())
    probs = list(weights.values())
    state = rng.choices(states, weights=probs, k=1)[0]
    if state == "loaded":
        frac = rng.uniform(0.85, 1.0)
    elif state == "partial":
        frac = rng.uniform(0.3, 0.7)
    else:  # empty
        frac = 0.0
    return round(car_type.mass_lading_kg * frac, 1)


def _pick_dominant_unit_car_type(rng: random.Random, candidates: List[CarType]) -> CarType:
    weights = [_UNIT_TRAIN_CATEGORY_WEIGHT.get(ct.category, 1.0) for ct in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def _split_counts(rng: random.Random, n: int, n_parts: int) -> List[int]:
    """Split ``n`` into ``n_parts`` positive integers that sum exactly to ``n``."""
    n_parts = min(n_parts, n)
    raw = [rng.random() for _ in range(n_parts)]
    total = sum(raw)
    counts = [max(1, round(n * r / total)) for r in raw]
    # Reconcile rounding drift against the exact target.
    diff = n - sum(counts)
    i = 0
    while diff != 0:
        idx = i % n_parts
        if diff > 0:
            counts[idx] += 1
            diff -= 1
        elif counts[idx] > 1:
            counts[idx] -= 1
            diff += 1
        i += 1
    return counts


def _build_trailing_blocks(
    rng: random.Random, n: int, unit_train: bool, non_loco_types: List[CarType]
) -> List[CarType]:
    """Return ``n`` car types in head->tail order, grouped into like-type
    blocks (unit train: one block; manifest: several shuffled blocks)."""
    if unit_train or n < 3:
        dominant = _pick_dominant_unit_car_type(rng, non_loco_types)
        return [dominant] * n

    n_block_types = min(rng.randint(3, min(6, len(non_loco_types))), n)
    block_types = rng.sample(non_loco_types, n_block_types)
    counts = _split_counts(rng, n, n_block_types)
    blocks = list(zip(block_types, counts))
    rng.shuffle(blocks)

    trailing: List[CarType] = []
    for ct, count in blocks:
        trailing.extend([ct] * count)
    return trailing


def _pick_loco_count(
    rng: random.Random,
    trailing_mass_kg: float,
    loco_type: CarType,
    loco_count_range: Tuple[int, int],
    specific_trac_force_n_per_kg: float,
) -> int:
    lo, hi = loco_count_range
    target_force_n = trailing_mass_kg * specific_trac_force_n_per_kg
    count = lo
    while count < hi and count * loco_type.F_trac_max_N < target_force_n:
        count += 1
    return count


def _distribute_power(rng: random.Random, loco_count: int) -> Tuple[int, int, int]:
    """Split ``loco_count`` locomotives into (head, mid, rear) blocks."""
    if loco_count < 2:
        return loco_count, 0, 0
    head_n = max(1, loco_count - loco_count // 2)
    remaining = loco_count - head_n
    choice = rng.random()
    if choice < 0.4 or remaining < 2:
        return head_n, remaining, 0  # all remaining mid-train
    elif choice < 0.8:
        return head_n, 0, remaining  # all remaining rear helper
    else:
        mid_n = remaining // 2
        return head_n, mid_n, remaining - mid_n


def _consist_label(unit_train: bool, dominant: CarType) -> str:
    if unit_train:
        tag = dominant.lading.replace(" ", "_") if dominant.lading else dominant.category.value
        return f"unit_{tag}"
    return "mixed_manifest"


def generate_trains(
    config: GeneratorConfig, library: Optional[CarLibrary] = None
) -> List[Tuple[Train, dict]]:
    """Pure generation: no disk writes. Returns ``(Train, manifest_row)`` pairs,
    deterministic for a given ``config.seed``."""
    library = library or ensure_extended_car_library(config.car_library_dir)
    rng = random.Random(config.seed)

    all_types = library.list_car_types()
    loco_types = [ct for ct in all_types if ct.category == CarCategory.LOCOMOTIVE]
    non_loco_types = [ct for ct in all_types if ct.category != CarCategory.LOCOMOTIVE]
    if not loco_types or not non_loco_types:
        raise ValueError("Car library needs at least one locomotive and one non-locomotive type")

    results: List[Tuple[Train, dict]] = []
    for idx in range(config.count):
        is_ood = rng.random() < config.ood_fraction
        lo, hi = config.ood_n_range if is_ood else config.train_n_range
        n_total = rng.randint(lo, hi)
        split_band = "ood_size" if is_ood else "train"

        unit_train = rng.random() < config.unit_train_fraction
        loco_type = rng.choice(loco_types)

        # Reserve at least one car for a locomotive; solve for the trailing
        # length once we know how many locomotives the tonnage calls for by
        # iterating: guess trailing = n_total - 1, refine after loco_count.
        n_trailing_guess = max(1, n_total - 1)
        trailing_types = _build_trailing_blocks(rng, n_trailing_guess, unit_train, non_loco_types)

        trailing_overrides = [_load_fraction(rng, ct, config.load_state_weights) for ct in trailing_types]
        trailing_mass_kg = sum(
            ct.with_overrides({"mass_lading_kg": ov} if ov is not None else None).mass_kg
            for ct, ov in zip(trailing_types, trailing_overrides)
        )

        loco_count = _pick_loco_count(
            rng, trailing_mass_kg, loco_type, config.loco_count_range, config.specific_trac_force_n_per_kg
        )
        # n_total is the whole consist (locos + trailing); keep it on target
        # by trimming/padding the trailing block count to match exactly.
        n_trailing = max(1, n_total - loco_count)
        if n_trailing != len(trailing_types):
            if n_trailing < len(trailing_types):
                trailing_types = trailing_types[:n_trailing]
                trailing_overrides = trailing_overrides[:n_trailing]
            else:
                pad = n_trailing - len(trailing_types)
                dominant = trailing_types[-1] if trailing_types else non_loco_types[0]
                trailing_types.extend([dominant] * pad)
                trailing_overrides.extend([_load_fraction(rng, dominant, config.load_state_weights)] * pad)
            trailing_mass_kg = sum(
                ct.with_overrides({"mass_lading_kg": ov} if ov is not None else None).mass_kg
                for ct, ov in zip(trailing_types, trailing_overrides)
            )

        distributed_power = rng.random() < config.distributed_power_fraction
        head_n, mid_n, rear_n = _distribute_power(rng, loco_count) if distributed_power else (loco_count, 0, 0)
        distributed_power = mid_n > 0 or rear_n > 0

        head_cars = [CarInstance(loco_type.id) for _ in range(head_n)]
        trailing_cars = []
        for ct, ov in zip(trailing_types, trailing_overrides):
            overrides = {"mass_lading_kg": ov} if ov is not None else {}
            trailing_cars.append(CarInstance(ct.id, overrides=overrides))

        if mid_n > 0:
            mid_idx = len(trailing_cars) // 2
            mid_cars = [CarInstance(loco_type.id, label="DPU mid-train") for _ in range(mid_n)]
            trailing_cars = trailing_cars[:mid_idx] + mid_cars + trailing_cars[mid_idx:]
        if rear_n > 0:
            trailing_cars.extend(CarInstance(loco_type.id, label="DPU rear helper") for _ in range(rear_n))

        cars = head_cars + trailing_cars
        total_mass_kg = loco_count * loco_type.mass_kg + trailing_mass_kg

        dominant_type = trailing_types[0] if unit_train else max(
            set(ct.id for ct in trailing_types),
            key=lambda cid: sum(1 for ct in trailing_types if ct.id == cid),
        )
        dominant_ct = trailing_types[0] if unit_train else next(
            ct for ct in trailing_types if ct.id == dominant_type
        )
        consist_label = _consist_label(unit_train, dominant_ct)

        name = f"{config.name_prefix} {consist_label} {idx:04d}"
        description = (
            f"Procedurally generated ({split_band}): {len(cars)} cars, "
            f"{loco_count} loco(s){' (distributed power)' if distributed_power else ''}, "
            f"~{total_mass_kg / 1000:.0f} t total."
        )
        train = Train(name=name, cars=cars, description=description)

        row = {
            "filename": f"{slugify(name)}.json",
            "N": len(cars),
            "loco_count": loco_count,
            "distributed_power": distributed_power,
            "consist_label": consist_label,
            "total_mass_kg": round(total_mass_kg, 1),
            "split": split_band,
            "seed": config.seed,
            "index": idx,
        }
        results.append((train, row))

    return results


def run(config: GeneratorConfig) -> dict:
    """Generate + save trains and a manifest. Returns a summary dict."""
    library = ensure_extended_car_library(config.car_library_dir)
    pairs = generate_trains(config, library=library)

    out_dir = Path(config.out_dir)
    manifest_dir = out_dir / MANIFEST_SUBDIR
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for train, row in pairs:
        train.save(out_dir, filename=Path(row["filename"]).stem)
        manifest_rows.append(row)

    manifest_path = manifest_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest_rows, indent=2) + "\n", encoding="utf-8")

    n_ood = sum(1 for r in manifest_rows if r["split"] == "ood_size")
    n_dp = sum(1 for r in manifest_rows if r["distributed_power"])
    mean_n = sum(r["N"] for r in manifest_rows) / len(manifest_rows)
    mean_loco = sum(r["loco_count"] for r in manifest_rows) / len(manifest_rows)
    mean_mass_t = sum(r["total_mass_kg"] for r in manifest_rows) / len(manifest_rows) / 1000.0

    return {
        "count": len(manifest_rows),
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "n_train_band": len(manifest_rows) - n_ood,
        "n_ood_size_band": n_ood,
        "n_distributed_power": n_dp,
        "mean_N": mean_n,
        "mean_loco_count": mean_loco,
        "mean_total_mass_t": mean_mass_t,
    }


def _parse_range(s: str) -> Tuple[int, int]:
    lo, hi = s.split(",")
    return int(lo), int(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="Procedural freight-consist generator")
    ap.add_argument("--count", type=int, default=100, help="Number of trains to generate")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    ap.add_argument("--out-dir", default=str(DEFAULT_SAVED_TRAINS_DIR),
                    help="Directory to save Train JSON into (default: saved_trains/)")
    ap.add_argument("--car-library-dir", default=str(DEFAULT_CAR_LIBRARY_DIR),
                    help="Car type library directory (extended in place if needed)")
    ap.add_argument("--train-n-range", type=_parse_range, default="10,80",
                    help="min,max consist size for the training band")
    ap.add_argument("--ood-n-range", type=_parse_range, default="120,150",
                    help="min,max consist size for the held-out large band")
    ap.add_argument("--ood-fraction", type=float, default=0.15,
                    help="Fraction of trains drawn from the OOD size band")
    ap.add_argument("--unit-train-fraction", type=float, default=0.6,
                    help="Fraction of trains that are single-commodity unit trains")
    ap.add_argument("--distributed-power-fraction", type=float, default=0.25,
                    help="Fraction of trains using mid-train/rear distributed power")
    ap.add_argument("--loco-count-range", type=_parse_range, default="1,4",
                    help="min,max locomotives per train")
    args = ap.parse_args()

    config = GeneratorConfig(
        count=args.count,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        car_library_dir=Path(args.car_library_dir),
        train_n_range=args.train_n_range,
        ood_n_range=args.ood_n_range,
        ood_fraction=args.ood_fraction,
        unit_train_fraction=args.unit_train_fraction,
        distributed_power_fraction=args.distributed_power_fraction,
        loco_count_range=args.loco_count_range,
    )
    summary = run(config)

    print(f"Generated {summary['count']} trains -> {summary['out_dir']}")
    print(f"  train band:    {summary['n_train_band']}")
    print(f"  ood_size band: {summary['n_ood_size_band']}")
    print(f"  distributed power: {summary['n_distributed_power']}")
    print(f"  mean N: {summary['mean_N']:.1f}, mean loco count: {summary['mean_loco_count']:.2f}, "
          f"mean total mass: {summary['mean_total_mass_t']:.0f} t")
    print(f"  manifest: {summary['manifest_path']}")


if __name__ == "__main__":
    main()
