"""Randomized traction/brake command profiles for the Controls screen.

A ``ControlProfile`` is a reusable, train-agnostic recipe: piecewise
``CommandCurve``s expressed as a *fraction* of each vehicle's own
``F_trac_max_N`` / ``F_brk_max_N`` ceiling, so the same profile can drive any
consist. Roles (``head_end`` / ``dpu_mid`` / ``dpu_rear``) are resolved from
each car's ``CarInstance.label`` (the same convention ``train_generator.py``
already writes), so a profile generated here plugs straight into
``make_scenario_from_consist`` without either side knowing about the other.

Pure stdlib (dataclasses + json + random) -- no numpy, no Qt.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .consist import Train, slugify
from .params import VehicleParameters

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL_PROFILES_DIR = _REPO_ROOT / "control_profiles"

ROLE_HEAD_END = "head_end"
ROLE_DPU_MID = "dpu_mid"
ROLE_DPU_REAR = "dpu_rear"

_DPU_MID_LABEL = "DPU mid-train"
_DPU_REAR_LABEL = "DPU rear helper"


def _smoothstep(u: float) -> float:
    u = 0.0 if u < 0.0 else 1.0 if u > 1.0 else u
    return 3 * u * u - 2 * u * u * u


@dataclass
class CommandKnot:
    t_s: float
    fraction: float  # 0..1 of the vehicle's own force ceiling

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CommandKnot":
        return cls(t_s=float(d["t_s"]), fraction=float(d["fraction"]))


@dataclass
class CommandCurve:
    """A piecewise command curve: holds the first/last knot's value outside
    its own time range, smoothstep-interpolates between interior knots (same
    S-curve shape as ``forces.traction_ramp``, generalized to many knots)."""

    knots: List[CommandKnot] = field(default_factory=list)

    def value_at(self, t: float) -> float:
        if not self.knots:
            return 0.0
        ks = sorted(self.knots, key=lambda k: k.t_s)
        if t <= ks[0].t_s:
            return ks[0].fraction
        if t >= ks[-1].t_s:
            return ks[-1].fraction
        for k0, k1 in zip(ks, ks[1:]):
            if k0.t_s <= t <= k1.t_s:
                span = k1.t_s - k0.t_s
                u = (t - k0.t_s) / span if span > 0 else 1.0
                return k0.fraction + (k1.fraction - k0.fraction) * _smoothstep(u)
        return ks[-1].fraction  # unreachable given the bracketing above

    def sample(self, ts: Sequence[float]) -> List[float]:
        return [self.value_at(t) for t in ts]

    def to_dict(self) -> dict:
        return {"knots": [k.to_dict() for k in self.knots]}

    @classmethod
    def from_dict(cls, d: dict) -> "CommandCurve":
        return cls(knots=[CommandKnot.from_dict(k) for k in d.get("knots", [])])


@dataclass
class ControlProfile:
    name: str
    description: str = ""
    duration_s: float = 120.0
    brake: CommandCurve = field(default_factory=CommandCurve)
    traction_head_end: CommandCurve = field(default_factory=CommandCurve)
    traction_dpu_mid: CommandCurve = field(default_factory=CommandCurve)
    traction_dpu_rear: CommandCurve = field(default_factory=CommandCurve)
    seed: Optional[int] = None
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "duration_s": self.duration_s,
            "brake": self.brake.to_dict(),
            "traction_head_end": self.traction_head_end.to_dict(),
            "traction_dpu_mid": self.traction_dpu_mid.to_dict(),
            "traction_dpu_rear": self.traction_dpu_rear.to_dict(),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ControlProfile":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            duration_s=float(d.get("duration_s", 120.0)),
            brake=CommandCurve.from_dict(d.get("brake", {})),
            traction_head_end=CommandCurve.from_dict(d.get("traction_head_end", {})),
            traction_dpu_mid=CommandCurve.from_dict(d.get("traction_dpu_mid", {})),
            traction_dpu_rear=CommandCurve.from_dict(d.get("traction_dpu_rear", {})),
            seed=d.get("seed"),
            schema_version=int(d.get("schema_version", 1)),
        )

    def save(self, directory: Path | str = DEFAULT_CONTROL_PROFILES_DIR, filename: Optional[str] = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        stem = filename or slugify(self.name)
        path = d / f"{stem}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ControlProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def list_saved_control_profiles(directory: Path | str = DEFAULT_CONTROL_PROFILES_DIR) -> List[Path]:
    d = Path(directory)
    return sorted(d.glob("*.json")) if d.exists() else []


def role_for_car(label: str, can_traction: bool) -> Optional[str]:
    """Resolve a car's control role from its label (see ``train_generator.py``'s
    DPU labeling convention) and whether it can produce traction at all."""
    if not can_traction:
        return None
    if label == _DPU_MID_LABEL:
        return ROLE_DPU_MID
    if label == _DPU_REAR_LABEL:
        return ROLE_DPU_REAR
    return ROLE_HEAD_END


def build_command_callables(profile: ControlProfile, train: Train, vehicles: List[VehicleParameters]):
    """Turn a stored ``ControlProfile`` into the ``u_trac_cmd(t, i)`` /
    ``u_brk_cmd(t, i)`` callables ``ExtendedTrainScenario`` expects. Brake
    applies train-wide (real air brakes act on every car); traction only on
    powered vehicles, scaled by each one's own ``F_trac_max_N``."""
    if len(train.cars) != len(vehicles):
        raise ValueError(f"Train has {len(train.cars)} cars but got {len(vehicles)} vehicles")
    roles = [role_for_car(c.label, v.can_traction) for c, v in zip(train.cars, vehicles)]
    curve_by_role = {
        ROLE_HEAD_END: profile.traction_head_end,
        ROLE_DPU_MID: profile.traction_dpu_mid,
        ROLE_DPU_REAR: profile.traction_dpu_rear,
    }

    def u_trac_cmd(t: float, i: int) -> float:
        role = roles[i]
        if role is None:
            return 0.0
        frac = max(0.0, curve_by_role[role].value_at(t))
        return frac * vehicles[i].F_trac_max_N

    def u_brk_cmd(t: float, i: int) -> float:
        frac = max(0.0, profile.brake.value_at(t))
        return frac * vehicles[i].F_brk_max_N

    return u_trac_cmd, u_brk_cmd


@dataclass
class RandomProfileConfig:
    seed: int = 0
    duration_s: float = 120.0
    n_knots_range: Tuple[int, int] = (3, 6)
    traction_fraction_range: Tuple[float, float] = (0.0, 1.0)
    brake_fraction_range: Tuple[float, float] = (0.0, 0.6)
    min_knot_spacing_s: float = 10.0
    dpu_mode: str = "synced"  # "synced" | "independent" | "off"
    dpu_lag_s: float = 3.0
    name_prefix: str = "Random control"


def _random_curve(
    rng: random.Random,
    duration_s: float,
    n_knots_range: Tuple[int, int],
    value_range: Tuple[float, float],
    min_spacing_s: float,
) -> CommandCurve:
    n_knots = max(2, rng.randint(*n_knots_range))
    step = duration_s / (n_knots - 1)
    times = [min(duration_s, max(0.0, i * step + rng.uniform(-0.25 * step, 0.25 * step))) for i in range(n_knots)]
    times[0], times[-1] = 0.0, duration_s
    times.sort()
    for i in range(1, len(times)):
        if times[i] - times[i - 1] < min_spacing_s:
            times[i] = min(duration_s, times[i - 1] + min_spacing_s)
    times = sorted(set(round(t, 3) for t in times))
    if len(times) < 2:
        times = [0.0, duration_s]

    lo, hi = value_range
    knots = [CommandKnot(t_s=t, fraction=round(rng.uniform(lo, hi), 3)) for t in times]
    return CommandCurve(knots=knots)


def _time_shift_curve(curve: CommandCurve, lag_s: float, duration_s: float) -> CommandCurve:
    knots = sorted(
        (CommandKnot(t_s=min(duration_s, max(0.0, k.t_s + lag_s)), fraction=k.fraction) for k in curve.knots),
        key=lambda k: k.t_s,
    )
    return CommandCurve(knots=knots)


def generate_random_profile(config: RandomProfileConfig) -> ControlProfile:
    """Deterministic for a given ``config.seed``."""
    rng = random.Random(config.seed)
    brake = _random_curve(
        rng, config.duration_s, config.n_knots_range, config.brake_fraction_range, config.min_knot_spacing_s
    )
    head = _random_curve(
        rng, config.duration_s, config.n_knots_range, config.traction_fraction_range, config.min_knot_spacing_s
    )

    if config.dpu_mode == "off":
        zero = CommandCurve(knots=[CommandKnot(0.0, 0.0), CommandKnot(config.duration_s, 0.0)])
        mid, rear = zero, CommandCurve(knots=list(zero.knots))
    elif config.dpu_mode == "synced":
        mid = _time_shift_curve(head, config.dpu_lag_s, config.duration_s)
        rear = _time_shift_curve(head, 2 * config.dpu_lag_s, config.duration_s)
    else:  # "independent"
        mid = _random_curve(
            rng, config.duration_s, config.n_knots_range, config.traction_fraction_range, config.min_knot_spacing_s
        )
        rear = _random_curve(
            rng, config.duration_s, config.n_knots_range, config.traction_fraction_range, config.min_knot_spacing_s
        )

    return ControlProfile(
        name=f"{config.name_prefix} {config.seed}",
        description=f"Randomized (seed={config.seed}, dpu_mode={config.dpu_mode!r})",
        duration_s=config.duration_s,
        brake=brake,
        traction_head_end=head,
        traction_dpu_mid=mid,
        traction_dpu_rear=rear,
        seed=config.seed,
    )
