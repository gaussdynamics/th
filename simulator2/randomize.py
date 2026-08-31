"""Randomized scenario construction for dataset generation.

``make_scenario_from_consist`` builds one canonical scenario: the consist starts
**at rest** with its rear car at chainage ~0, runs for 120 s, and every actuator
starts at zero. That is the right default for a reproducible demo, but as a
dataset it is degenerate -- at freight speeds a 120 s run covers 1-2 km of a
30 km route, so nearly every sample is the same cold-start transient over the
same few hundred metres of terrain, and the surrogate never sees cruise, brake,
or slack-loaded initial conditions.

:func:`make_randomized_scenario` samples the things that were previously fixed:

* **start chainage** -- anywhere along the route with room to run,
* **initial speed** -- respecting the route speed limit at the start point,
* **initial slack state** -- stretched / bunched / neutral / random, within the
  coupler deadband so no artificial force is injected at ``t = 0``,
* **initial actuator states** -- settled at the control profile's ``t = 0``
  command, so the run starts mid-manoeuvre rather than with a spurious ramp,
* **adhesion regime** -- as a multiplier on every powered vehicle's tractive
  effort ceiling,
* **run duration** -- clipped so the consist cannot run off the end of the route.

It returns the scenario together with the flat metadata record
``DATA_SCHEMA.md`` §I asks for, so the dataset driver can write
``index.parquet`` without recomputing anything.

Adhesion note: the reference model has no wheel-rail creep law, so "adhesion"
here is a scale factor on ``F_trac_max_N``, not a friction model. That is a
domain-randomization knob, and it should be described as such -- it changes how
much tractive effort is available, not how the wheel behaves at its limit.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import constants as C
from .control_profile import ControlProfile, build_command_callables
from .consist import Train
from .io_types import ExtendedTrainScenario
from .params import CouplerParameters, VehicleParameters
from .route import RouteProfile
from .scenarios import _initial_positions

__all__ = [
    "RandomizationConfig",
    "SlackState",
    "make_randomized_scenario",
    "route_length_m",
]

SlackState = str  # "neutral" | "stretched" | "bunched" | "random"

_SLACK_STATES: Tuple[SlackState, ...] = ("neutral", "stretched", "bunched", "random")


@dataclass
class RandomizationConfig:
    """Sampling ranges for :func:`make_randomized_scenario`."""

    # --- kinematics -------------------------------------------------------
    v0_mps_range: Tuple[float, float] = (0.0, 24.0)
    #: Never start above this fraction of the route speed limit at the start point.
    v0_vmax_fraction: float = 0.9
    #: Probability the run starts from a standstill even when a rolling start
    #: was allowed (keeps genuine cold starts represented).
    v0_rest_probability: float = 0.10

    # --- geometry ---------------------------------------------------------
    #: Keep this much clear route behind the rear car and ahead of the stopping point.
    start_margin_m: float = 250.0
    #: Multiplier on the distance the consist could plausibly cover, used to
    #: reserve room ahead so a run cannot leave the route.
    travel_headroom_factor: float = 1.25
    #: Speed used to bound travel when the sampled v0 is low (m/s).
    travel_speed_floor_mps: float = 12.0
    #: Standing starts are placed on track no steeper than this |sin theta|
    #: (0.005 = 0.5 %), so the consist does not roll away before traction
    #: builds. Rejection-sampled with this many attempts.
    rest_start_max_sin_theta: float = 0.005
    rest_start_grade_attempts: int = 64
    #: If the profile commands brake within ``brake_lookahead_s`` of t = 0,
    #: start at least this fast. Braking from a crawl is a degenerate run: the
    #: consist stops almost immediately and the rest of the window is a
    #: near-stationary rollback that teaches the surrogate nothing.
    min_v0_when_braking_mps: float = 6.0
    brake_lookahead_s: float = 20.0

    slack_state_weights: Dict[str, float] = field(
        default_factory=lambda: {"neutral": 0.25, "stretched": 0.30, "bunched": 0.30, "random": 0.15}
    )
    #: Fraction of the coupler half-slack used when pre-loading slack. Kept
    #: below 1.0 so the initial state sits strictly inside the deadband and
    #: no coupler force exists at t = 0.
    slack_fill_fraction: float = 0.85

    # --- timing -----------------------------------------------------------
    duration_s_range: Tuple[float, float] = (180.0, 600.0)
    #: Output sample spacing (s). T = duration / dt + 1.
    dt_s: float = 0.25
    #: Hard cap on stored timesteps, so a long run cannot blow up memory.
    max_time_samples: int = 4001

    # --- physics randomization -------------------------------------------
    adhesion_weights: Dict[str, float] = field(
        default_factory=lambda: {"dry": 0.60, "wet": 0.30, "low": 0.10}
    )
    adhesion_traction_scale: Dict[str, float] = field(
        default_factory=lambda: {"dry": 1.00, "wet": 0.80, "low": 0.60}
    )
    #: Curvature resistance proxy scale. Left at 0.0 by default because the
    #: proxy's magnitude is not calibrated; set deliberately to engage the
    #: route's kappa field.
    k_curv_scale: float = 0.0
    p_max_w: float = C.DEFAULT_P_MAX_W
    tau_trac_s: float = C.DEFAULT_TAU_TRAC_S
    tau_brk_s: float = C.DEFAULT_TAU_BRK_S

    # --- integration ------------------------------------------------------
    rtol: float = C.DEFAULT_RTOL
    atol: float = C.DEFAULT_ATOL
    method: str = C.DEFAULT_METHOD
    #: See rhs.train_rhs_tensorized. True is correct for any run that may brake
    #: to a stop, which randomized starts make common.
    brake_opposes_motion: bool = True


def route_length_m(route: RouteProfile) -> float:
    return float(route.s_nodes_m[-1] - route.s_nodes_m[0])


def _weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    keys = list(weights.keys())
    return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


def _route_vmax_stats(route: RouteProfile) -> Tuple[float, float]:
    """Return ``(median, max)`` finite speed limit over the route (m/s)."""
    if route.v_max_nodes is None:
        return math.inf, math.inf
    v = np.asarray(route.v_max_nodes, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.inf, math.inf
    return float(np.median(v)), float(v.max())


def _slack_offsets(
    rng: random.Random,
    couplers: Sequence[CouplerParameters],
    state: SlackState,
    fill: float,
) -> np.ndarray:
    """Per-coupler initial ``delta`` offsets, strictly inside the deadband."""
    n_edges = len(couplers)
    out = np.zeros(n_edges, dtype=float)
    for j, c in enumerate(couplers):
        s = float(c.slack_half_m) * fill
        if state == "stretched":
            out[j] = +s
        elif state == "bunched":
            out[j] = -s
        elif state == "random":
            out[j] = rng.uniform(-s, s)
        else:  # neutral
            out[j] = 0.0
    return out


def _apply_adhesion(
    vehicles: Sequence[VehicleParameters], scale: float
) -> List[VehicleParameters]:
    """Scale powered vehicles' tractive ceiling; never mutate the input list."""
    out: List[VehicleParameters] = []
    for v in vehicles:
        if v.can_traction and scale != 1.0:
            out.append(replace(v, F_trac_max_N=float(v.F_trac_max_N) * float(scale)))
        else:
            out.append(replace(v))
    return out


def make_randomized_scenario(
    route: RouteProfile,
    vehicles: Sequence[VehicleParameters],
    couplers: Sequence[CouplerParameters],
    profile: ControlProfile,
    train: Optional[Train] = None,
    seed: int = 0,
    config: Optional[RandomizationConfig] = None,
    starts_from_rest: bool = False,
    route_id: str = "",
    control_id: str = "",
    consist_id: str = "",
) -> Tuple[ExtendedTrainScenario, Dict]:
    """Build one randomized scenario plus its metadata record.

    ``train`` is only needed to resolve DPU roles from car labels via
    :func:`control_profile.build_command_callables`; pass ``None`` to drive
    every powered vehicle from the head-end curve.

    ``starts_from_rest`` forces ``v0 = 0`` -- set it for control profiles whose
    regime only makes sense from a standstill (see
    ``driving_regimes.REGIMES_STARTING_FROM_REST``).
    """
    cfg = config or RandomizationConfig()
    rng = random.Random(seed)
    n = len(vehicles)
    if len(couplers) != n - 1:
        raise ValueError(f"Expected {n - 1} couplers for {n} vehicles, got {len(couplers)}")

    # ---- adhesion --------------------------------------------------------
    adhesion = _weighted_choice(rng, cfg.adhesion_weights)
    trac_scale = float(cfg.adhesion_traction_scale.get(adhesion, 1.0))
    vehicles = _apply_adhesion(vehicles, trac_scale)

    # ---- geometry & duration --------------------------------------------
    consist_len_m = float(sum(c.L0_m for c in couplers))
    route_len_m = route_length_m(route)
    s0 = float(route.s_nodes_m[0])

    duration_s = rng.uniform(*cfg.duration_s_range)
    duration_s = min(duration_s, profile.duration_s)

    v_med, _ = _route_vmax_stats(route)
    v_ref = cfg.travel_speed_floor_mps if not math.isfinite(v_med) else max(v_med, cfg.travel_speed_floor_mps)

    usable = route_len_m - consist_len_m - 2.0 * cfg.start_margin_m
    if usable <= 0.0:
        raise ValueError(
            f"Route is shorter ({route_len_m:.0f} m) than the consist plus margins "
            f"({consist_len_m + 2 * cfg.start_margin_m:.0f} m)"
        )

    # Clip duration so the reserved travel headroom fits inside the route.
    max_duration_s = usable / (v_ref * cfg.travel_headroom_factor)
    duration_clipped = duration_s > max_duration_s
    duration_s = max(30.0, min(duration_s, max_duration_s))

    headroom_m = v_ref * duration_s * cfg.travel_headroom_factor
    lo = s0 + consist_len_m + cfg.start_margin_m
    hi = s0 + route_len_m - headroom_m - cfg.start_margin_m

    # Decide up front whether this is a standing start, because it constrains
    # where the consist may be placed (see below).
    from_rest = bool(starts_from_rest or rng.random() < cfg.v0_rest_probability)

    if hi <= lo:
        x_lead_m = lo
    elif from_rest:
        # A consist standing at v = 0 with no brake applied will roll away down
        # any appreciable grade before traction builds through tau_trac -- the
        # run then travels *backwards* and is useless as training data. Real
        # standing trains hold with the air brake; rather than model that, put
        # standing starts on near-level track. Rejection-sample a low-grade
        # spot and fall back to the flattest candidate seen.
        best_x, best_grade = None, math.inf
        for _ in range(cfg.rest_start_grade_attempts):
            cand = rng.uniform(lo, hi)
            g = abs(route.sin_theta_at(cand))
            if g < best_grade:
                best_x, best_grade = cand, g
            if g <= cfg.rest_start_max_sin_theta:
                break
        x_lead_m = float(best_x)
    else:
        x_lead_m = rng.uniform(lo, hi)

    # ---- initial speed ---------------------------------------------------
    v_limit_here = route.v_max_at(x_lead_m)
    v_hi = cfg.v0_mps_range[1]
    if math.isfinite(v_limit_here):
        v_hi = min(v_hi, cfg.v0_vmax_fraction * v_limit_here)
    v_hi = max(v_hi, 0.0)

    if from_rest:
        v0_scalar = 0.0
    else:
        v_lo = min(cfg.v0_mps_range[0], v_hi)
        # Braking from a crawl produces a run that stops in the first few
        # seconds and then sits still; raise the floor when the profile brakes
        # early so the window contains actual deceleration dynamics.
        brakes_early = any(
            profile.brake.value_at(tt) > 0.05
            for tt in np.linspace(0.0, min(cfg.brake_lookahead_s, duration_s), 9)
        )
        if brakes_early:
            v_lo = min(max(v_lo, cfg.min_v0_when_braking_mps), v_hi)
        v0_scalar = rng.uniform(v_lo, v_hi)

    # ---- initial slack ---------------------------------------------------
    slack_state = _weighted_choice(rng, cfg.slack_state_weights)
    offsets = _slack_offsets(rng, couplers, slack_state, cfg.slack_fill_fraction)

    x0 = _initial_positions(list(vehicles), list(couplers), x_lead_m)
    # delta_j = (x_j - x_{j+1}) - L0_j; push the trailing block back by the
    # cumulative offset so each coupler sits at its target delta.
    if n > 1:
        x0[1:] = x0[1:] - np.cumsum(offsets)

    v0 = np.full(n, v0_scalar, dtype=float)

    # ---- commands & settled actuators ------------------------------------
    if train is not None:
        u_trac_cmd, u_brk_cmd = build_command_callables(profile, train, list(vehicles))
    else:
        def u_trac_cmd(t: float, i: int) -> float:
            v = vehicles[i]
            if not v.can_traction:
                return 0.0
            return max(0.0, profile.traction_head_end.value_at(t)) * v.F_trac_max_N

        def u_brk_cmd(t: float, i: int) -> float:
            return max(0.0, profile.brake.value_at(t)) * vehicles[i].F_brk_max_N

    z_trac0 = np.array([u_trac_cmd(0.0, i) for i in range(n)], dtype=float)
    z_brk0 = np.array([u_brk_cmd(0.0, i) for i in range(n)], dtype=float)

    y0 = np.concatenate([x0, v0, z_brk0, z_trac0])

    # ---- time grid -------------------------------------------------------
    n_samples = int(round(duration_s / cfg.dt_s)) + 1
    n_samples = max(2, min(n_samples, cfg.max_time_samples))
    t_span = (0.0, float(duration_s))
    t_eval = np.linspace(t_span[0], t_span[1], n_samples)

    meta = {
        "route_id": route_id,
        "control_id": control_id or profile.name,
        "consist_id": consist_id,
        "N": n,
        "T": int(n_samples),
        "R": int(np.asarray(route.s_nodes_m).size),
        "route_len_km": round(route_len_m / 1000.0, 4),
        "x_lead_m": round(float(x_lead_m), 3),
        "start_chainage_frac": round(float((x_lead_m - s0) / route_len_m), 5),
        "v0_mps": round(float(v0_scalar), 4),
        "slack_state": slack_state,
        "adhesion": adhesion,
        "adhesion_traction_scale": trac_scale,
        "duration_s": round(float(duration_s), 3),
        "dt_s": round(float(duration_s) / (n_samples - 1), 6),
        "duration_clipped_to_route": bool(duration_clipped),
        "consist_len_m": round(consist_len_m, 3),
        "k_curv_scale": cfg.k_curv_scale,
        "brake_opposes_motion": cfg.brake_opposes_motion,
        "seed": int(seed),
    }

    scenario = ExtendedTrainScenario(
        vehicles=list(vehicles),
        couplers=list(couplers),
        route=route,
        y0=y0,
        u_trac_cmd=u_trac_cmd,
        u_brk_cmd=u_brk_cmd,
        k_curv_scale=cfg.k_curv_scale,
        tau_brk_s=cfg.tau_brk_s,
        tau_trac_s=cfg.tau_trac_s,
        p_max_w=cfg.p_max_w,
        brake_opposes_motion=cfg.brake_opposes_motion,
        t_span=t_span,
        t_eval=t_eval,
        rtol=cfg.rtol,
        atol=cfg.atol,
        method=cfg.method,
        extra_metadata={"scenario": "make_randomized_scenario", **meta},
    )
    return scenario, meta
