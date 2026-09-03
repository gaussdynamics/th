"""Driving-regime control profile library.

``control_profile.generate_random_profile`` draws every knot uniformly at
random. That is fine for exercising the UI, but it is poor *training* data for
a surrogate: it spends most of its probability mass on command combinations no
train handler would ever produce (notably full traction and heavy braking at
the same instant), and it never reproduces the structured manoeuvres the
downstream MPC controller will actually generate.

This module samples **regimes** instead: each regime is a recognisable
manoeuvre shape (notch up, cruise, coast, brake application, dynamic-brake
descent, ...) whose *parameters* are randomised, rather than the raw knots.
The result is a profile library that covers the command space the surrogate has
to model while staying inside the region an operator would actually visit.

Everything here emits plain ``ControlProfile`` objects, so the existing
``build_command_callables`` -> ``make_scenario_from_consist`` path is unchanged.

Traction/brake exclusivity
--------------------------
With one deliberate exception, regimes never command traction and brake
simultaneously. The exception is ``STRETCH_BRAKE``: holding light air brake
against light traction to keep the slack stretched is a real handling
technique, and it is exactly the kind of coupled command the surrogate needs to
see. :func:`overlap_fraction` measures how much of a profile violates
exclusivity so tests can assert on it.

Pure stdlib (dataclasses + random) -- no numpy, no Qt.
"""

from __future__ import annotations

import json
import random
import zlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .control_profile import (
    DEFAULT_CONTROL_PROFILES_DIR,
    CommandCurve,
    CommandKnot,
    ControlProfile,
)

__all__ = [
    "Regime",
    "RegimeLibraryConfig",
    "generate_regime_profile",
    "generate_profile_library",
    "overlap_fraction",
    "DEFAULT_REGIME_WEIGHTS",
    "REGIMES_STARTING_FROM_REST",
    "MAX_OVERLAP_FRACTION",
    "DEFAULT_MAX_OVERLAP_FRACTION",
]


class Regime(str, Enum):
    """Recognisable manoeuvre shapes."""

    STARTUP = "startup"                        # from rest, ramp to a working notch
    NOTCH_UP = "notch_up"                      # stepwise increase in traction
    CRUISE = "cruise"                          # near-constant traction
    THROTTLE_MODULATION = "throttle_modulation"  # wandering traction -> slack action
    POWER_TO_COAST = "power_to_coast"          # hold, then release to zero
    COAST = "coast"                            # no traction, no brake
    COAST_TO_BRAKE = "coast_to_brake"          # release power, then apply brake
    DYNAMIC_BRAKE_DESCENT = "dynamic_brake_descent"  # sustained braking, no power
    BRAKE_RELEASE = "brake_release"            # brake applied -> released -> power on
    STRETCH_BRAKE = "stretch_brake"            # light brake held against light power
    EMERGENCY = "emergency"                    # power cut, brake to maximum


#: Regimes that only make sense when the consist starts at (or near) rest.
REGIMES_STARTING_FROM_REST = frozenset({Regime.STARTUP})

#: Maximum fraction of a run each regime may spend commanding traction and
#: brake simultaneously. ``STRETCH_BRAKE`` does it throughout by design.
#: ``EMERGENCY`` does it briefly: the brake pipe drops while tractive effort is
#: still decaying, which is what actually happens and is worth learning.
#: Everything else must be clean.
MAX_OVERLAP_FRACTION: Dict[Regime, float] = {
    Regime.STRETCH_BRAKE: 1.0,
    Regime.EMERGENCY: 0.12,
}
DEFAULT_MAX_OVERLAP_FRACTION = 0.02

#: Sampling weights. Steady-state and modulation regimes dominate because they
#: are what the consist spends most of its time doing; emergency is rare but
#: must be represented because it produces the extreme coupler forces the
#: safety constraint cares about.
DEFAULT_REGIME_WEIGHTS: Dict[Regime, float] = {
    Regime.STARTUP: 0.08,
    Regime.NOTCH_UP: 0.12,
    Regime.CRUISE: 0.15,
    Regime.THROTTLE_MODULATION: 0.15,
    Regime.POWER_TO_COAST: 0.10,
    Regime.COAST: 0.08,
    Regime.COAST_TO_BRAKE: 0.10,
    Regime.DYNAMIC_BRAKE_DESCENT: 0.10,
    Regime.BRAKE_RELEASE: 0.06,
    Regime.STRETCH_BRAKE: 0.04,
    Regime.EMERGENCY: 0.02,
}


@dataclass
class RegimeLibraryConfig:
    """Sampling ranges shared across regimes."""

    duration_s_range: Tuple[float, float] = (180.0, 600.0)
    dpu_mode_weights: Dict[str, float] = field(
        default_factory=lambda: {"synced": 0.6, "independent": 0.25, "off": 0.15}
    )
    dpu_lag_s_range: Tuple[float, float] = (1.0, 6.0)
    #: Below this fraction a command counts as "off" for exclusivity checks.
    off_threshold: float = 0.05
    weights: Dict[Regime, float] = field(
        default_factory=lambda: dict(DEFAULT_REGIME_WEIGHTS)
    )


# ----------------------------------------------------------------------------
# curve construction helpers
# ----------------------------------------------------------------------------

def _curve(points: Sequence[Tuple[float, float]]) -> CommandCurve:
    """Build a CommandCurve from (t_s, fraction) pairs, sorted and clamped."""
    knots = [
        CommandKnot(t_s=round(max(0.0, float(t)), 3), fraction=round(min(1.0, max(0.0, float(f))), 4))
        for t, f in points
    ]
    knots.sort(key=lambda k: k.t_s)
    # Collapse exact duplicate times, keeping the last value written.
    dedup: List[CommandKnot] = []
    for k in knots:
        if dedup and abs(dedup[-1].t_s - k.t_s) < 1e-9:
            dedup[-1] = k
        else:
            dedup.append(k)
    return CommandCurve(knots=dedup)


def _flat(value: float, duration_s: float) -> CommandCurve:
    return _curve([(0.0, value), (duration_s, value)])


def _zero(duration_s: float) -> CommandCurve:
    return _flat(0.0, duration_s)


def _jittered_hold(
    rng: random.Random, level: float, duration_s: float, n_seg: int, jitter: float
) -> CommandCurve:
    """A hold at ``level`` with small random deviations -- 'steady' driving."""
    ts = [duration_s * i / n_seg for i in range(n_seg + 1)]
    return _curve([(t, level + rng.uniform(-jitter, jitter)) for t in ts])


def _time_shift(curve: CommandCurve, lag_s: float, duration_s: float) -> CommandCurve:
    return _curve(
        [(min(duration_s, k.t_s + lag_s), k.fraction) for k in curve.knots]
    )


def _perturb(
    curve: CommandCurve, rng: random.Random, duration_s: float,
    t_jitter_s: float, f_jitter: float,
) -> CommandCurve:
    """Jitter a curve's knot times and levels.

    Used for ``dpu_mode="independent"``: the remote units execute the *same*
    manoeuvre as the head end, just not in lockstep. Resampling a fresh shape
    instead would let a DPU command traction while the head end is braking,
    which is both unrealistic and a training-data trap.

    Two properties are enforced, and both matter:

    * The level jitter is **multiplicative**, so a commanded zero stays exactly
      zero. An additive jitter would lift the zero segments of traction-free
      regimes (coast, dynamic brake, post-event emergency) just above the off
      threshold and silently reintroduce traction/brake overlap on every
      remote unit.
    * The time jitter is **bounded by the local knot spacing**, so knots can
      never reorder. An unbounded jitter can swap a "cut power" knot past the
      "hold power" knot before it, which inverts the manoeuvre -- e.g. turning
      an emergency power cut into a power *restoration* under full braking.
    """
    ks = sorted(curve.knots, key=lambda k: k.t_s)
    n = len(ks)
    out: List[Tuple[float, float]] = []
    for i, k in enumerate(ks):
        gap_prev = k.t_s - ks[i - 1].t_s if i > 0 else float("inf")
        gap_next = ks[i + 1].t_s - k.t_s if i < n - 1 else float("inf")
        # 0.4 of the smaller neighbouring gap keeps strict ordering even if
        # both this knot and its neighbour jitter toward each other.
        bound = min(t_jitter_s, 0.4 * min(gap_prev, gap_next))
        dt = rng.uniform(-bound, bound) if bound > 0 else 0.0
        out.append(
            (
                min(duration_s, max(0.0, k.t_s + dt)),
                k.fraction * (1.0 + rng.uniform(-f_jitter, f_jitter)),
            )
        )
    return _curve(out)


def _gate_traction(
    trac: CommandCurve,
    brake: CommandCurve,
    duration_s: float,
    off_threshold: float,
    n_samples: int = 400,
) -> CommandCurve:
    """Force traction to zero wherever brake is commanded on.

    A hard guarantee rather than a hope: the per-regime shapes are built to be
    exclusive, but DPU time-shifting and jitter can nudge a traction ramp into
    a braking window. This zeroes any knot inside a brake-on interval and pins
    an explicit zero at each interval boundary, so the smoothstep ramp starts
    from zero exactly when the brake lets go.
    """
    dur = max(duration_s, 1e-9)
    ts = [dur * i / (n_samples - 1) for i in range(n_samples)]
    on = [brake.value_at(t) > off_threshold for t in ts]
    if not any(on):
        return trac

    def is_on(t: float) -> bool:
        i = int(round(t / dur * (n_samples - 1)))
        return on[min(n_samples - 1, max(0, i))]

    pts: List[Tuple[float, float]] = [
        (k.t_s, 0.0 if is_on(k.t_s) else k.fraction) for k in trac.knots
    ]
    for i in range(1, n_samples):
        if on[i] != on[i - 1]:
            pts.append((ts[i], 0.0))
    return _curve(pts)


# ----------------------------------------------------------------------------
# per-regime traction/brake shapes
# ----------------------------------------------------------------------------

def _shape_for_regime(
    regime: Regime, rng: random.Random, dur: float
) -> Tuple[CommandCurve, CommandCurve]:
    """Return ``(traction_head_end, brake)`` for one regime."""

    if regime is Regime.STARTUP:
        hold = rng.uniform(0.55, 1.0)
        t_start = rng.uniform(0.0, 0.10 * dur)
        t_full = t_start + rng.uniform(20.0, 60.0)
        return _curve([(0.0, 0.0), (t_start, 0.0), (min(t_full, dur), hold), (dur, hold)]), _zero(dur)

    if regime is Regime.NOTCH_UP:
        lo = rng.uniform(0.10, 0.40)
        hi = rng.uniform(max(lo + 0.15, 0.55), 1.0)
        n_steps = rng.randint(2, 4)
        pts = [(0.0, lo)]
        for i in range(1, n_steps + 1):
            lvl = lo + (hi - lo) * i / n_steps
            t = dur * (0.15 + 0.7 * i / (n_steps + 1)) + rng.uniform(-0.03, 0.03) * dur
            pts.append((t, lvl))
        pts.append((dur, hi))
        return _curve(pts), _zero(dur)

    if regime is Regime.CRUISE:
        return _jittered_hold(rng, rng.uniform(0.30, 0.75), dur, rng.randint(3, 6), 0.04), _zero(dur)

    if regime is Regime.THROTTLE_MODULATION:
        n = rng.randint(4, 7)
        pts = [(dur * i / (n - 1), rng.uniform(0.15, 0.90)) for i in range(n)]
        return _curve(pts), _zero(dur)

    if regime is Regime.POWER_TO_COAST:
        hold = rng.uniform(0.35, 0.85)
        t_rel = dur * rng.uniform(0.35, 0.70)
        t_off = t_rel + rng.uniform(15.0, 45.0)
        return _curve([(0.0, hold), (t_rel, hold), (min(t_off, dur), 0.0), (dur, 0.0)]), _zero(dur)

    if regime is Regime.COAST:
        return _flat(rng.uniform(0.0, 0.03), dur), _zero(dur)

    if regime is Regime.COAST_TO_BRAKE:
        t_pow_off = dur * rng.uniform(0.05, 0.25)
        t_brk_on = t_pow_off + rng.uniform(10.0, 40.0)
        lvl = rng.uniform(0.20, 0.55)
        trac = _curve([(0.0, rng.uniform(0.0, 0.25)), (t_pow_off, 0.0), (dur, 0.0)])
        brk = _curve([(0.0, 0.0), (min(t_brk_on, dur), 0.0),
                      (min(t_brk_on + rng.uniform(10.0, 30.0), dur), lvl), (dur, lvl)])
        return trac, brk

    if regime is Regime.DYNAMIC_BRAKE_DESCENT:
        lvl = rng.uniform(0.25, 0.60)
        return _zero(dur), _jittered_hold(rng, lvl, dur, rng.randint(3, 5), 0.06)

    if regime is Regime.BRAKE_RELEASE:
        lvl = rng.uniform(0.30, 0.65)
        t_rel = dur * rng.uniform(0.20, 0.45)
        t_off = t_rel + rng.uniform(10.0, 30.0)
        t_pow = min(t_off + rng.uniform(5.0, 25.0), dur)
        hold = rng.uniform(0.30, 0.75)
        brk = _curve([(0.0, lvl), (t_rel, lvl), (min(t_off, dur), 0.0), (dur, 0.0)])
        trac = _curve([(0.0, 0.0), (min(t_off, dur), 0.0), (t_pow, hold), (dur, hold)])
        return trac, brk

    if regime is Regime.STRETCH_BRAKE:
        # The one regime that deliberately overlaps traction and brake.
        # The brake floor is kept above the 0.05 off-threshold (0.09 - 0.02
        # jitter = 0.07) so the hold never dips into "brake released" -- a
        # stretch-brake profile that stops holding brake is just a cruise.
        return (
            _jittered_hold(rng, rng.uniform(0.20, 0.50), dur, rng.randint(2, 4), 0.03),
            _jittered_hold(rng, rng.uniform(0.09, 0.20), dur, rng.randint(2, 4), 0.02),
        )

    if regime is Regime.EMERGENCY:
        t_ev = dur * rng.uniform(0.15, 0.60)
        pre = rng.uniform(0.20, 0.85)
        trac = _curve([(0.0, pre), (t_ev, pre), (min(t_ev + 2.0, dur), 0.0), (dur, 0.0)])
        brk = _curve([(0.0, 0.0), (t_ev, 0.0),
                      (min(t_ev + rng.uniform(2.0, 6.0), dur), rng.uniform(0.85, 1.0)),
                      (dur, 1.0)])
        return trac, brk

    raise ValueError(f"Unhandled regime: {regime!r}")


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------

def generate_regime_profile(
    regime: Regime,
    seed: int,
    config: Optional[RegimeLibraryConfig] = None,
    duration_s: Optional[float] = None,
    dpu_mode: Optional[str] = None,
) -> ControlProfile:
    """Deterministic for a given ``(regime, seed)``, across processes.

    The regime is folded into the seed with ``zlib.crc32`` rather than the
    builtin ``hash``. ``Regime`` is a ``str`` enum and CPython randomizes string
    hashes per process unless ``PYTHONHASHSEED`` is set, so the previous
    ``hash(regime.value)`` made this function reproducible *within* a run and
    different on every new one -- which silently made dataset builds
    unreproducible from their recorded seed. ``crc32`` is stable across
    processes, platforms and versions.
    """
    cfg = config or RegimeLibraryConfig()
    rng = random.Random((zlib.crc32(regime.value.encode()) & 0xFFFF) * 1_000_003 + seed)

    # Rounded to the same precision _curve() rounds knot times to, so a knot
    # placed at `dur` round-trips exactly instead of landing a few hundred
    # microseconds past the profile's own duration.
    dur = round(float(duration_s) if duration_s is not None else rng.uniform(*cfg.duration_s_range), 3)
    if dpu_mode is None:
        modes, wts = zip(*cfg.dpu_mode_weights.items())
        dpu_mode = rng.choices(list(modes), weights=list(wts), k=1)[0]

    head, brake = _shape_for_regime(regime, rng, dur)
    gate = MAX_OVERLAP_FRACTION.get(regime, DEFAULT_MAX_OVERLAP_FRACTION) < 1.0 and \
        regime not in (Regime.EMERGENCY,)

    if dpu_mode == "off":
        mid = _zero(dur)
        rear = _zero(dur)
    elif dpu_mode == "independent":
        lag = rng.uniform(*cfg.dpu_lag_s_range)
        mid = _perturb(head, rng, dur, t_jitter_s=lag, f_jitter=0.06)
        rear = _perturb(head, rng, dur, t_jitter_s=2 * lag, f_jitter=0.10)
    else:  # "synced" -- DPUs follow the head end with a handling lag
        lag = rng.uniform(*cfg.dpu_lag_s_range)
        mid = _time_shift(head, lag, dur)
        rear = _time_shift(head, 2 * lag, dur)

    if gate:
        head = _gate_traction(head, brake, dur, cfg.off_threshold)
        mid = _gate_traction(mid, brake, dur, cfg.off_threshold)
        rear = _gate_traction(rear, brake, dur, cfg.off_threshold)

    return ControlProfile(
        name=f"{regime.value} {seed:05d}",
        description=(
            f"Driving-regime profile: regime={regime.value!r}, dpu_mode={dpu_mode!r}, "
            f"duration={dur:.1f}s, seed={seed}"
        ),
        duration_s=dur,
        brake=brake,
        traction_head_end=head,
        traction_dpu_mid=mid,
        traction_dpu_rear=rear,
        seed=seed,
    )


def sample_regime(rng: random.Random, config: Optional[RegimeLibraryConfig] = None) -> Regime:
    cfg = config or RegimeLibraryConfig()
    regimes = list(cfg.weights.keys())
    weights = [cfg.weights[r] for r in regimes]
    return rng.choices(regimes, weights=weights, k=1)[0]


def overlap_fraction(profile: ControlProfile, off_threshold: float = 0.05, n_samples: int = 400) -> float:
    """Fraction of the profile's duration where traction *and* brake are both on.

    Used as a guard in tests: every regime except ``STRETCH_BRAKE`` should score
    at or near zero. Traction is taken as the max across the three DPU roles,
    since any powered vehicle fighting the brake counts.
    """
    dur = max(profile.duration_s, 1e-9)
    hits = 0
    for i in range(n_samples):
        t = dur * i / max(n_samples - 1, 1)
        trac = max(
            profile.traction_head_end.value_at(t),
            profile.traction_dpu_mid.value_at(t),
            profile.traction_dpu_rear.value_at(t),
        )
        if trac > off_threshold and profile.brake.value_at(t) > off_threshold:
            hits += 1
    return hits / n_samples


def generate_profile_library(
    n_profiles: int,
    seed: int = 0,
    config: Optional[RegimeLibraryConfig] = None,
    out_dir: Optional[Path | str] = None,
    write: bool = True,
) -> Tuple[List[ControlProfile], List[dict]]:
    """Sample ``n_profiles`` regime profiles and (optionally) write them out.

    Returns ``(profiles, manifest_records)``. The manifest mirrors the style of
    ``saved_trains/generated/manifest.json``: one flat record per profile with
    the fields a dataset driver needs for stratification and split metadata.
    """
    cfg = config or RegimeLibraryConfig()
    rng = random.Random(seed)
    directory = Path(out_dir) if out_dir is not None else DEFAULT_CONTROL_PROFILES_DIR

    profiles: List[ControlProfile] = []
    records: List[dict] = []

    for i in range(n_profiles):
        regime = sample_regime(rng, cfg)
        prof = generate_regime_profile(regime, seed=rng.randrange(2**31), config=cfg)
        # Rename to a stable, sortable id so filenames don't collide.
        stem = f"ctrl_{regime.value}_{i:05d}"
        prof.name = stem
        profiles.append(prof)

        records.append(
            {
                "filename": f"{stem}.json",
                "control_id": stem,
                "regime": regime.value,
                "duration_s": round(prof.duration_s, 3),
                "dpu_mode": _infer_dpu_mode(prof),
                "starts_from_rest": regime in REGIMES_STARTING_FROM_REST,
                "overlap_fraction": round(overlap_fraction(prof, cfg.off_threshold), 4),
                "peak_traction": round(_peak(prof.traction_head_end), 4),
                "peak_brake": round(_peak(prof.brake), 4),
                "seed": prof.seed,
                "index": i,
            }
        )

        if write:
            prof.save(directory, filename=stem)

    if write:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )

    return profiles, records


def _peak(curve: CommandCurve) -> float:
    return max((k.fraction for k in curve.knots), default=0.0)


def _infer_dpu_mode(profile: ControlProfile) -> str:
    mid_peak = _peak(profile.traction_dpu_mid)
    rear_peak = _peak(profile.traction_dpu_rear)
    if mid_peak <= 1e-9 and rear_peak <= 1e-9:
        return "off"
    head = [(k.t_s, k.fraction) for k in profile.traction_head_end.knots]
    mid = [(k.t_s, k.fraction) for k in profile.traction_dpu_mid.knots]
    if len(head) == len(mid) and all(
        abs(h[1] - m[1]) < 1e-9 for h, m in zip(head, mid)
    ):
        return "synced"
    return "independent"
