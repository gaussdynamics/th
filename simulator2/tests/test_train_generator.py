"""Tests for the procedural consist generator (simulator2/train_generator.py)."""

from __future__ import annotations

import json

from simulator2.catalog import CarLibrary
from simulator2.consist import Train, build_consist
from simulator2.train_generator import (
    GeneratorConfig,
    ensure_extended_car_library,
    generate_trains,
    run,
)


def _small_config(tmp_path, **overrides) -> GeneratorConfig:
    defaults = dict(
        count=20,
        seed=0,
        train_n_range=(10, 20),
        ood_n_range=(30, 40),
        ood_fraction=0.3,
        loco_count_range=(1, 4),
        out_dir=tmp_path / "trains",
        car_library_dir=tmp_path / "car_library",
    )
    defaults.update(overrides)
    return GeneratorConfig(**defaults)


def test_extended_library_has_new_types(tmp_path):
    library = ensure_extended_car_library(tmp_path / "car_library")
    ids = {ct.id for ct in library.list_car_types()}
    assert {
        "loco_gp38",
        "grain_hopper_loaded",
        "ballast_hopper_loaded",
        "boxcar_manifest_loaded",
        "intermodal_well_car_loaded",
        "gondola_scrap_loaded",
    } <= ids


def test_generated_trains_respect_n_and_loco_ranges(tmp_path):
    config = _small_config(tmp_path)
    pairs = generate_trains(config)
    assert len(pairs) == config.count
    for train, row in pairs:
        assert len(train.cars) == row["N"]
        if row["split"] == "train":
            assert config.train_n_range[0] <= row["N"] <= config.train_n_range[1]
        else:
            assert config.ood_n_range[0] <= row["N"] <= config.ood_n_range[1]
        assert config.loco_count_range[0] <= row["loco_count"] <= config.loco_count_range[1]


def test_generation_is_deterministic_under_fixed_seed(tmp_path):
    config_a = _small_config(tmp_path / "a")
    config_b = _small_config(tmp_path / "b")
    pairs_a = generate_trains(config_a)
    pairs_b = generate_trains(config_b)
    for (train_a, row_a), (train_b, row_b) in zip(pairs_a, pairs_b):
        assert train_a.to_dict() == train_b.to_dict()
        assert row_a == row_b


def test_run_writes_trains_and_manifest_discoverable_by_list_saved_trains(tmp_path):
    from simulator2.consist import list_saved_trains

    config = _small_config(tmp_path)
    summary = run(config)
    assert summary["count"] == config.count

    saved = list_saved_trains(config.out_dir)
    assert len(saved) == config.count

    manifest_path = config.out_dir / "generated" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == config.count
    assert {"filename", "N", "loco_count", "distributed_power", "consist_label",
            "total_mass_kg", "split"} <= set(manifest[0])


def test_generated_train_round_trips_through_build_consist(tmp_path):
    config = _small_config(tmp_path, count=5)
    summary = run(config)
    library = CarLibrary.load(config.car_library_dir)

    from simulator2.consist import list_saved_trains

    for path in list_saved_trains(config.out_dir):
        loaded = Train.load(path)
        vehicles, couplers = build_consist(loaded, library)
        assert len(vehicles) == len(loaded.cars)
        assert len(couplers) == len(loaded.cars) - 1
        assert any(v.can_traction for v in vehicles)  # every generated train has power
