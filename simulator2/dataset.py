"""Dataset serialization for the surrogate training corpus.

Implements the per-scenario and dataset-level artifacts of `DATA_SCHEMA.md`:
arrays A-H into one compressed ``.npz`` per scenario, route fields stored once
per corridor, and the four manifests (`index.parquet`, `norm_stats.json`,
`splits.json`, `build_config.json`).

Two schema decisions are worth restating because they shape the code:

* **Static and dynamic are separated.** ``H_hist`` is ``[T, N, 11]`` but seven
  of those channels are constant in time, so this module never materializes it
  -- it writes ``state [T, N, 4]`` plus ``node_static [N, 7]`` (§A.1).
* **Route fields are written once per corridor**, into ``routes/<route_id>.npz``,
  and referenced by ``route_id``. A corridor is reused across hundreds of
  consist/control variations, so duplicating ``R``-length arrays per scenario
  would dominate the dataset size for no information gain (§File format).

Edge dynamics are algebraically derived from node state in this simulator.
They are stored anyway, as supervision targets -- ``F_cpl`` in particular is the
safety-critical quantity -- and flagged derived via ``edge_dynamic_is_derived``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from .constants import DEFAULT_V_BRAKE_EPS, DEFAULT_V_EPS

__all__ = [
    "SCHEMA_VERSION",
    "ScenarioArrays",
    "build_scenario_arrays",
    "write_scenario_npz",
    "write_scenario_arrays",
    "write_route_npz",
    "compute_norm_stats",
    "write_manifests",
]

SCHEMA_VERSION = 1

#: Channel names, in the order they appear on disk. Consumers should index by
#: these rather than by position.
STATE_CHANNELS = ("x", "v", "z_brk", "z_trac")
NODE_STATIC_CHANNELS = (
    "mass_kg", "davis_A", "davis_B", "davis_C",
    "can_traction", "F_trac_max_N", "F_brk_max_N",
)
EDGE_DYNAMIC_CHANNELS = ("delta", "delta_dot", "F_cpl")
EDGE_STATIC_CHANNELS = (
    "L0_m", "slack_half_m", "k_draft", "c_draft", "k_buff", "c_buff",
)


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------

def coupler_dynamics(
    state: np.ndarray, edge_static: np.ndarray
) -> np.ndarray:
    """``edge_dynamic [T, N-1, 3]`` from ``state [T, N, 4]``.

    Reproduces ``forces.coupler_force_slack_asymmetric`` in vectorized form:
    a dead zone of half-width ``slack_half_m`` with asymmetric draft/buff
    stiffness and damping on either side.
    """
    x, v = state[:, :, 0], state[:, :, 1]
    l0 = edge_static[:, 0]
    slack = edge_static[:, 1]
    k_d, c_d = edge_static[:, 2], edge_static[:, 3]
    k_b, c_b = edge_static[:, 4], edge_static[:, 5]

    delta = (x[:, :-1] - x[:, 1:]) - l0[None, :]
    delta_dot = v[:, :-1] - v[:, 1:]

    f = np.zeros_like(delta)
    f = np.where(delta > slack, k_d * (delta - slack) + c_d * delta_dot, f)
    f = np.where(delta < -slack, k_b * (delta + slack) + c_b * delta_dot, f)
    return np.stack([delta, delta_dot, f], axis=-1)


def traction_power(
    state: np.ndarray,
    node_static: np.ndarray,
    p_max_w: float,
    v_eps: float = DEFAULT_V_EPS,
) -> np.ndarray:
    """Instantaneous tractive power ``[T]``, summed over vehicles.

    Applies the same power cap the RHS does, so the integral matches the work
    the simulated consist actually did rather than what was commanded.
    """
    v = state[:, :, 1]
    z_trac = np.maximum(state[:, :, 3], 0.0)
    can_traction = node_static[None, :, 4] > 0
    cap = p_max_w / np.maximum(np.abs(v), v_eps)
    f_trac = np.where(can_traction, np.minimum(z_trac, cap), 0.0)
    return (f_trac * v).sum(axis=1)


def sample_command_curve(curve: Any, t: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of ``CommandCurve.value_at`` over a whole grid.

    ``value_at`` re-sorts the knot list on every call, so materializing a
    ``[T, N]`` command block through it costs ``T*N`` sorts -- measured at 67 ms
    per scenario, which would dominate the build once the GPU rollout is
    batched properly. This sorts once and evaluates with ``searchsorted``.

    Semantics are matched exactly, including holding the end knots' values
    outside the curve's own range and the ``3u^2 - 2u^3`` smoothstep between
    interior knots. ``tests/test_dataset.py`` asserts equality against
    ``simulate.materialize_commands`` rather than trusting this note.
    """
    ts = np.asarray(t, dtype=np.float64)
    knots = sorted(curve.knots, key=lambda k: k.t_s)
    if not knots:
        return np.zeros_like(ts)
    kt = np.array([k.t_s for k in knots], dtype=np.float64)
    kf = np.array([k.fraction for k in knots], dtype=np.float64)
    if kt.size == 1:
        return np.full_like(ts, kf[0])

    idx = np.clip(np.searchsorted(kt, ts, side="right"), 1, kt.size - 1)
    t0, t1 = kt[idx - 1], kt[idx]
    f0, f1 = kf[idx - 1], kf[idx]
    span = t1 - t0
    u = np.where(span > 0, (ts - t0) / np.where(span > 0, span, 1.0), 1.0)
    u = np.clip(u, 0.0, 1.0)
    out = f0 + (f1 - f0) * (3.0 * u * u - 2.0 * u * u * u)
    out = np.where(ts <= kt[0], kf[0], out)
    return np.where(ts >= kt[-1], kf[-1], out)


def materialize_commands_fast(
    profile: Any, vehicles: Sequence[Any], t: np.ndarray, train: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """``u_trac``, ``u_brk`` as ``[T, N]``, mirroring ``build_command_callables``.

    Brake applies train-wide; traction only to powered vehicles, from the curve
    for their DPU role, each scaled by its own force ceiling. There are at most
    three distinct traction curves however long the consist is, so each is
    sampled once and broadcast.
    """
    from .control_profile import (
        ROLE_DPU_MID, ROLE_DPU_REAR, ROLE_HEAD_END, role_for_car,
    )

    ts = np.asarray(t, dtype=np.float64)
    n = len(vehicles)

    brake_frac = np.maximum(0.0, sample_command_curve(profile.brake, ts))
    f_brk_max = np.array([v.F_brk_max_N for v in vehicles], dtype=np.float64)
    u_brk = brake_frac[:, None] * f_brk_max[None, :]

    f_trac_max = np.array([v.F_trac_max_N for v in vehicles], dtype=np.float64)
    u_trac = np.zeros((ts.size, n), dtype=np.float64)

    if train is not None:
        curve_by_role = {
            ROLE_HEAD_END: profile.traction_head_end,
            ROLE_DPU_MID: profile.traction_dpu_mid,
            ROLE_DPU_REAR: profile.traction_dpu_rear,
        }
        roles = [role_for_car(c.label, v.can_traction)
                 for c, v in zip(train.cars, vehicles)]
        cache: Dict[str, np.ndarray] = {}
        for i, role in enumerate(roles):
            if role is None:
                continue
            if role not in cache:
                cache[role] = np.maximum(
                    0.0, sample_command_curve(curve_by_role[role], ts)
                )
            u_trac[:, i] = cache[role] * f_trac_max[i]
    else:
        frac = np.maximum(0.0, sample_command_curve(profile.traction_head_end, ts))
        for i, v in enumerate(vehicles):
            if v.can_traction:
                u_trac[:, i] = frac * f_trac_max[i]

    return u_trac, u_brk


def speed_limit_stats(
    state: np.ndarray, route_s: np.ndarray, route_vmax: np.ndarray
) -> Dict[str, float]:
    """How far a run exceeded the route speed limit, and for how long.

    The open-loop RHS deliberately does not enforce ``route_vmax`` -- the field
    is exogenous information for the controller (see ``route.RouteProfile``) --
    so nothing stops a coasting consist accelerating indefinitely down a steep
    grade. On corridors whose grade field is pinned at the pipeline's 4 % clip,
    that produces runs at 50+ m/s: consistent with the model, but not with any
    real operation.

    These stats go into ``index.parquet`` so such runs can be filtered or
    reweighted per split without rebuilding the dataset. They are recorded
    rather than acted on, because whether an overspeed run is bad *data* or a
    useful hard case is a modelling decision, not a serialization one.
    """
    x, v = state[:, :, 0], state[:, :, 1]
    limit = np.interp(x, route_s, route_vmax)
    over = v - limit
    any_over = (over > 0).any(axis=1)
    return {
        "v_over_limit_max_mps": float(max(over.max(), 0.0)),
        "frac_time_over_limit": float(any_over.mean()),
        "v_limit_min_mps": float(limit.min()),
    }


def arrival_time(t: np.ndarray, x_lead: np.ndarray, target_chainage_m: float) -> float:
    """First time the lead vehicle reaches ``target_chainage_m``.

    Linearly interpolated between samples; ``nan`` if the target is never
    reached within the run.
    """
    reached = np.nonzero(x_lead >= target_chainage_m)[0]
    if reached.size == 0:
        return float("nan")
    k = int(reached[0])
    if k == 0:
        return float(t[0])
    x0, x1 = float(x_lead[k - 1]), float(x_lead[k])
    if x1 == x0:
        return float(t[k])
    w = (target_chainage_m - x0) / (x1 - x0)
    return float(t[k - 1] + w * (t[k] - t[k - 1]))


# ---------------------------------------------------------------------------
# per-scenario arrays
# ---------------------------------------------------------------------------

@dataclass
class ScenarioArrays:
    """Arrays A-H of `DATA_SCHEMA.md` for one scenario, ready to serialize."""

    t: np.ndarray               # [T]
    state: np.ndarray           # [T, N, 4]
    node_static: np.ndarray     # [N, 7]
    edge_dynamic: np.ndarray    # [T, N-1, 3]
    edge_static: np.ndarray     # [N-1, 6]
    edge_index: np.ndarray      # [2, N-1]
    u_trac: np.ndarray          # [T, N]
    u_brk: np.ndarray           # [T, N]
    y0: np.ndarray              # [4N]
    constants: Dict[str, float]
    summaries: Dict[str, float]


def build_scenario_arrays(
    t: np.ndarray,
    y: np.ndarray,
    node_static: np.ndarray,
    edge_static: np.ndarray,
    u_trac: np.ndarray,
    u_brk: np.ndarray,
    constants: Dict[str, float],
    target_chainage_m: float | None = None,
) -> ScenarioArrays:
    """Assemble one scenario's arrays from a flat state trajectory.

    ``y`` is ``[T, 4N]`` in the simulator's flat layout
    ``[x(N), v(N), z_brk(N), z_trac(N)]`` -- the same ordering the torch
    rollout and ``solve_ivp`` both produce, so this works for either path.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = node_static.shape[0]
    if y.shape[1] != 4 * n:
        raise ValueError(f"expected y of width {4 * n}, got {y.shape[1]}")

    state = np.stack(
        [y[:, 0:n], y[:, n:2 * n], y[:, 2 * n:3 * n], y[:, 3 * n:4 * n]], axis=-1
    )
    edge_dynamic = coupler_dynamics(state, edge_static)
    edge_index = np.stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int32)

    power = traction_power(state, node_static, float(constants["p_max_w"]))
    x_lead = state[:, 0, 0]
    summaries = {
        "E_trip": float(np.trapezoid(power, t)),
        "F_max": float(np.abs(edge_dynamic[:, :, 2]).max()) if n > 1 else 0.0,
        "distance_travelled_m": float(x_lead[-1] - x_lead[0]),
        "v_mean_mps": float(state[:, :, 1].mean()),
        "v_max_mps": float(state[:, :, 1].max()),
        "v_min_mps": float(state[:, :, 1].min()),
        # DATA_SCHEMA.md asks for T_arr, "arrival time at target chainage".
        # An open-loop scenario has no target, so this is nan unless one is
        # supplied; the field exists for the control-eval runs, which do.
        "T_arr": (float("nan") if target_chainage_m is None
                  else arrival_time(t, x_lead, target_chainage_m)),
        "target_chainage_m": (float("nan") if target_chainage_m is None
                              else float(target_chainage_m)),
    }

    return ScenarioArrays(
        t=t, state=state, node_static=node_static,
        edge_dynamic=edge_dynamic, edge_static=edge_static, edge_index=edge_index,
        u_trac=np.asarray(u_trac, dtype=np.float64),
        u_brk=np.asarray(u_brk, dtype=np.float64),
        y0=y[0].copy(), constants=dict(constants), summaries=summaries,
    )


def write_scenario_arrays(
    arrays: ScenarioArrays, path: Path | str, metadata: Dict[str, Any],
    compress: bool = True,
) -> Path:
    """Write one scenario ``.npz``. Arrays are stored float32 per the schema.

    ``compress=False`` swaps zlib for a raw ``.npz``. Compression costs about
    62 ms per scenario -- roughly a quarter of the build once the rollout is
    batched -- and buys about 2x on disk, so it is worth turning off when the
    target volume is cheap.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    save = np.savez_compressed if compress else np.savez

    save(
        path,
        schema_version=SCHEMA_VERSION,
        # A
        t=f32(arrays.t),
        dt=np.float32(np.diff(arrays.t).mean() if arrays.t.size > 1 else 0.0),
        # B
        state=f32(arrays.state),
        # C
        node_static=f32(arrays.node_static),
        # D
        edge_dynamic=f32(arrays.edge_dynamic),
        edge_static=f32(arrays.edge_static),
        edge_index=arrays.edge_index.astype(np.int32),
        edge_dynamic_is_derived=True,
        # E
        u_trac=f32(arrays.u_trac),
        u_brk=f32(arrays.u_brk),
        # F is referenced, not duplicated: see routes/<route_id>.npz
        route_id=str(metadata["route_id"]),
        # G
        **{k: np.float32(v) for k, v in arrays.summaries.items()},
        # H
        y0=f32(arrays.y0),
        **{k: np.float32(v) for k, v in arrays.constants.items()},
        # I
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return path


def write_scenario_npz(
    result: Any, scenario: Any, metadata: Dict[str, Any], path: Path | str
) -> Path:
    """Serializer with the signature `DATA_SCHEMA.md` names, for the NumPy path.

    ``result`` is a ``TensorSimulationResult``. The torch driver does not go
    through here -- it already holds the flat trajectory and calls
    :func:`build_scenario_arrays` directly -- but this keeps the NumPy path able
    to emit the same files, which matters for spot-checking the GPU output
    against the oracle.
    """
    from .simulate import materialize_commands

    n = len(scenario.vehicles)
    node_static = np.array([
        [v.mass_kg, v.davis_A, v.davis_B, v.davis_C,
         float(v.can_traction), v.F_trac_max_N, v.F_brk_max_N]
        for v in scenario.vehicles
    ], dtype=np.float64)
    edge_static = np.array([
        [c.L0_m, c.slack_half_m, c.k_draft, c.c_draft, c.k_buff, c.c_buff]
        for c in scenario.couplers
    ], dtype=np.float64).reshape(max(n - 1, 0), len(EDGE_STATIC_CHANNELS))

    u_trac, u_brk = materialize_commands(scenario, result.t)
    arrays = build_scenario_arrays(
        t=result.t, y=result.sol.y.T,
        node_static=node_static, edge_static=edge_static,
        u_trac=u_trac, u_brk=u_brk,
        constants=scenario_constants(scenario),
        target_chainage_m=metadata.get("target_chainage_m"),
    )
    return write_scenario_arrays(arrays, path, metadata)


def scenario_constants(scenario: Any) -> Dict[str, float]:
    """Integrator / physics constants (§H) needed to recompute the RHS."""
    return {
        "tau_brk_s": float(scenario.tau_brk_s),
        "tau_trac_s": float(scenario.tau_trac_s),
        "p_max_w": float(scenario.p_max_w),
        "k_curv_scale": float(scenario.k_curv_scale),
        "v_eps": float(DEFAULT_V_EPS),
        "v_brake_eps": float(DEFAULT_V_BRAKE_EPS),
        "brake_opposes_motion": float(bool(scenario.brake_opposes_motion)),
    }


def write_route_npz(route: Any, route_id: str, out_dir: Path | str) -> Path:
    """Write a corridor's route field once (§F)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{route_id}.npz"
    s = np.asarray(route.s_nodes_m, dtype=np.float32)
    zeros = np.zeros_like(s)
    np.savez_compressed(
        path,
        route_s=s,
        route_sin_theta=np.asarray(route.sin_theta_nodes, dtype=np.float32),
        route_kappa=(zeros if route.kappa_nodes is None
                     else np.asarray(route.kappa_nodes, dtype=np.float32)),
        route_vmax=(np.full_like(s, np.inf) if route.v_max_nodes is None
                    else np.asarray(route.v_max_nodes, dtype=np.float32)),
        route_id=route_id,
    )
    return path


def route_grade_stats(route: Any) -> Dict[str, float]:
    """Route severity stats for the §I metadata record."""
    sin_theta = np.asarray(route.sin_theta_nodes, dtype=np.float64)
    return {
        "grade_max_pct": round(float(np.abs(sin_theta).max() * 100.0), 4),
        "grade_rms_pct": round(float(np.sqrt((sin_theta ** 2).mean()) * 100.0), 4),
    }


# ---------------------------------------------------------------------------
# dataset-level manifests
# ---------------------------------------------------------------------------

# Fields normalized by norm_stats.json, each with its channel names. Route
# fields are included because the model samples them as inputs at every RHS
# evaluation.
#
# The channel count is what the accumulator reshapes to, so it must be the
# genuinely per-channel axis. For u_trac/u_brk that is a *single* channel: the
# array is [T, N] and N is the vehicle axis, which varies between scenarios and
# is not a feature dimension.
_NORM_FIELDS = {
    "state": STATE_CHANNELS,
    "node_static": NODE_STATIC_CHANNELS,
    "edge_dynamic": EDGE_DYNAMIC_CHANNELS,
    "edge_static": EDGE_STATIC_CHANNELS,
    "u_trac": ("u_trac",),
    "u_brk": ("u_brk",),
}
_ROUTE_FIELDS = ("route_sin_theta", "route_kappa", "route_vmax")


def compute_norm_stats(
    scenario_paths: Sequence[Path | str],
    route_paths: Sequence[Path | str] = (),
) -> Dict[str, Any]:
    """Per-channel mean/std, streamed so the whole dataset never has to be resident.

    **Computed over the training split only** (§Design principle 4) -- the
    caller is responsible for passing train-split paths, and the returned dict
    records how many scenarios went into it so that cannot be lost.
    """
    acc: Dict[str, Dict[str, Any]] = {}

    def accumulate(name: str, a: np.ndarray, channels) -> None:
        a = np.asarray(a, dtype=np.float64)
        n_ch = len(channels)
        if a.shape[-1] != n_ch and n_ch != 1:
            raise ValueError(
                f"{name}: trailing axis {a.shape[-1]} does not match "
                f"{n_ch} declared channels"
            )
        flat = a.reshape(-1, n_ch)
        e = acc.setdefault(name, {
            "n": 0, "sum": np.zeros(n_ch), "sumsq": np.zeros(n_ch),
            "channels": list(channels),
            "min": np.full(n_ch, np.inf),
            "max": np.full(n_ch, -np.inf),
        })
        finite = flat[np.isfinite(flat).all(axis=1)] if flat.size else flat
        e["n"] += finite.shape[0]
        e["sum"] += finite.sum(axis=0)
        e["sumsq"] += (finite ** 2).sum(axis=0)
        if finite.size:
            e["min"] = np.minimum(e["min"], finite.min(axis=0))
            e["max"] = np.maximum(e["max"], finite.max(axis=0))

    for p in scenario_paths:
        with np.load(p) as d:
            for name, channels in _NORM_FIELDS.items():
                if name in d:
                    accumulate(name, d[name], channels)

    for p in route_paths:
        with np.load(p) as d:
            for name in _ROUTE_FIELDS:
                if name in d:
                    accumulate(name, d[name], (name,))

    out: Dict[str, Any] = {
        "_computed_over": {
            "n_scenarios": len(scenario_paths),
            "n_routes": len(route_paths),
            "split": "train",
        }
    }
    for name, e in acc.items():
        n = max(e["n"], 1)
        mean = e["sum"] / n
        var = np.maximum(e["sumsq"] / n - mean ** 2, 0.0)
        std = np.sqrt(var)
        # A channel that never varies (can_traction on an all-powered consist,
        # kappa when curvature is disabled) would divide by zero downstream.
        std_safe = np.where(std > 1e-12, std, 1.0)
        entry = {
            "mean": mean.tolist(),
            "std": std_safe.tolist(),
            "std_raw": std.tolist(),
            "min": e["min"].tolist(),
            "max": e["max"].tolist(),
            "count": int(e["n"]),
        }
        if e["channels"]:
            entry["channels"] = e["channels"]
        out[name] = entry
    return out


def git_commit() -> str:
    """Current commit, for `build_config.json` reproducibility."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001 - not a git checkout, or no git
        return "unknown"


def write_manifests(
    out_dir: Path | str,
    records: Sequence[Dict[str, Any]],
    norm_stats: Dict[str, Any],
    build_config: Dict[str, Any],
) -> Dict[str, Path]:
    """Write ``index.parquet``, ``splits.json``, ``norm_stats.json``, ``build_config.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    import pandas as pd

    df = pd.DataFrame.from_records(list(records))
    index_path = out_dir / "index.parquet"
    df.to_parquet(index_path, index=False)
    written["index"] = index_path

    splits: Dict[str, list] = {}
    for r in records:
        splits.setdefault(str(r["split"]), []).append(str(r["scenario_id"]))
    for k in splits:
        splits[k].sort()
    splits_path = out_dir / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=1) + "\n", encoding="utf-8")
    written["splits"] = splits_path

    norm_path = out_dir / "norm_stats.json"
    norm_path.write_text(json.dumps(norm_stats, indent=1) + "\n", encoding="utf-8")
    written["norm_stats"] = norm_path

    cfg = {"schema_version": SCHEMA_VERSION, "git_commit": git_commit(), **build_config}
    cfg_path = out_dir / "build_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1, default=str) + "\n", encoding="utf-8")
    written["build_config"] = cfg_path

    return written
