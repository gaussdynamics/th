"""Tests for the driving-regime control profile library."""

from __future__ import annotations

import random

from pathlib import Path

import pytest

from simulator2.control_profile import ControlProfile
from simulator2.driving_regimes import (
    DEFAULT_MAX_OVERLAP_FRACTION,
    MAX_OVERLAP_FRACTION,
    REGIMES_STARTING_FROM_REST,
    Regime,
    RegimeLibraryConfig,
    generate_profile_library,
    generate_regime_profile,
    overlap_fraction,
    sample_regime,
)

SEEDS = range(60)


@pytest.mark.parametrize("regime", list(Regime))
def test_every_regime_builds(regime: Regime) -> None:
    p = generate_regime_profile(regime, seed=0)
    assert isinstance(p, ControlProfile)
    assert p.duration_s > 0
    for curve in (p.brake, p.traction_head_end, p.traction_dpu_mid, p.traction_dpu_rear):
        assert len(curve.knots) >= 2
        assert all(0.0 <= k.fraction <= 1.0 for k in curve.knots)
        assert all(0.0 <= k.t_s <= p.duration_s + 1e-6 for k in curve.knots)


@pytest.mark.parametrize("regime", list(Regime))
def test_knots_are_time_ordered(regime: Regime) -> None:
    """Guards the DPU jitter: unbounded time jitter can reorder knots and
    invert a manoeuvre (e.g. turn an emergency power cut into a power
    restoration under full braking)."""
    for s in SEEDS:
        p = generate_regime_profile(regime, seed=s)
        for curve in (p.brake, p.traction_head_end, p.traction_dpu_mid, p.traction_dpu_rear):
            ts = [k.t_s for k in curve.knots]
            assert ts == sorted(ts)


@pytest.mark.parametrize("regime", list(Regime))
def test_traction_brake_exclusivity(regime: Regime) -> None:
    """Only STRETCH_BRAKE may hold both on; EMERGENCY gets a short window."""
    limit = MAX_OVERLAP_FRACTION.get(regime, DEFAULT_MAX_OVERLAP_FRACTION)
    for s in SEEDS:
        p = generate_regime_profile(regime, seed=s)
        ov = overlap_fraction(p)
        assert ov <= limit, f"{regime.value} seed={s} overlap={ov:.3f} > {limit}"


def test_stretch_brake_actually_overlaps() -> None:
    """The exception must genuinely exercise the coupled-command case, or the
    surrogate never sees traction fighting brake."""
    ovs = [overlap_fraction(generate_regime_profile(Regime.STRETCH_BRAKE, seed=s)) for s in SEEDS]
    assert min(ovs) > 0.5


@pytest.mark.parametrize("regime", list(Regime))
def test_deterministic_for_seed(regime: Regime) -> None:
    assert (
        generate_regime_profile(regime, seed=11).to_dict()
        == generate_regime_profile(regime, seed=11).to_dict()
    )


@pytest.mark.parametrize("regime", list(Regime))
def test_distinct_seeds_give_distinct_profiles(regime: Regime) -> None:
    dicts = [generate_regime_profile(regime, seed=s).to_dict() for s in range(12)]
    unique = {repr(d) for d in dicts}
    assert len(unique) > 1


def test_braking_regimes_actually_brake() -> None:
    for regime in (Regime.DYNAMIC_BRAKE_DESCENT, Regime.COAST_TO_BRAKE, Regime.EMERGENCY):
        for s in SEEDS:
            p = generate_regime_profile(regime, seed=s)
            assert max(k.fraction for k in p.brake.knots) > 0.15, regime


def test_traction_regimes_actually_pull() -> None:
    for regime in (Regime.STARTUP, Regime.NOTCH_UP, Regime.CRUISE, Regime.THROTTLE_MODULATION):
        for s in SEEDS:
            p = generate_regime_profile(regime, seed=s)
            assert max(k.fraction for k in p.traction_head_end.knots) > 0.10, regime
            assert max(k.fraction for k in p.brake.knots) == 0.0, regime


def test_coast_is_idle() -> None:
    for s in SEEDS:
        p = generate_regime_profile(Regime.COAST, seed=s)
        assert max(k.fraction for k in p.traction_head_end.knots) <= 0.05
        assert max(k.fraction for k in p.brake.knots) == 0.0


def test_dpu_off_mode_silences_remotes() -> None:
    p = generate_regime_profile(Regime.CRUISE, seed=3, dpu_mode="off")
    assert max(k.fraction for k in p.traction_dpu_mid.knots) == 0.0
    assert max(k.fraction for k in p.traction_dpu_rear.knots) == 0.0


def test_startup_is_the_only_rest_regime() -> None:
    assert REGIMES_STARTING_FROM_REST == frozenset({Regime.STARTUP})


def test_sample_regime_covers_all_regimes() -> None:
    rng = random.Random(0)
    seen = {sample_regime(rng) for _ in range(3000)}
    assert seen == set(Regime)


def test_library_manifest(tmp_path) -> None:
    profiles, records = generate_profile_library(40, seed=5, out_dir=tmp_path, write=True)
    assert len(profiles) == len(records) == 40
    assert (tmp_path / "manifest.json").exists()
    for rec in records:
        assert (tmp_path / rec["filename"]).exists()
        assert rec["regime"] in {r.value for r in Regime}
        assert rec["dpu_mode"] in {"synced", "independent", "off"}
        limit = MAX_OVERLAP_FRACTION.get(Regime(rec["regime"]), DEFAULT_MAX_OVERLAP_FRACTION)
        assert rec["overlap_fraction"] <= limit


def test_library_roundtrips_through_disk(tmp_path) -> None:
    profiles, records = generate_profile_library(6, seed=2, out_dir=tmp_path, write=True)
    for prof, rec in zip(profiles, records):
        loaded = ControlProfile.load(tmp_path / rec["filename"])
        assert loaded.to_dict() == prof.to_dict()


def test_custom_weights_respected() -> None:
    cfg = RegimeLibraryConfig(weights={Regime.COAST: 1.0})
    rng = random.Random(1)
    assert {sample_regime(rng, cfg) for _ in range(50)} == {Regime.COAST}


def test_deterministic_across_processes() -> None:
    """The within-process determinism test above cannot catch a seed derived
    from ``hash()`` of a string: CPython randomizes string hashes per process,
    so such a function is stable inside one run and different in the next.

    That is exactly the bug this guards -- ``generate_regime_profile`` used
    ``hash(regime.value)``, which made every dataset build unreproducible from
    its recorded seed while every same-process test passed. Subprocesses are
    launched with distinct PYTHONHASHSEED values to force the issue.
    """
    import json as _json
    import os
    import subprocess
    import sys

    snippet = (
        "from simulator2.driving_regimes import Regime, generate_regime_profile;"
        "import json;"
        "p = generate_regime_profile(Regime.THROTTLE_MODULATION, seed=42);"
        "print(json.dumps(p.to_dict()))"
    )
    repo_root = Path(__file__).resolve().parents[2]
    outputs = []
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        res = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=str(repo_root), env=env, capture_output=True, text=True, check=True,
        )
        outputs.append(_json.loads(res.stdout))

    assert outputs[0] == outputs[1] == outputs[2], (
        "generate_regime_profile is not reproducible across processes; "
        "something in its seeding depends on PYTHONHASHSEED"
    )
