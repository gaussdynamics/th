"""Dataset driver: sample scenarios, roll them out on the GPU, write the corpus.

Implements step 4 of `NEXT_STEPS.md` against `DATA_SCHEMA.md`. Samples
``(route, consist, control regime)`` triples, integrates them in batches through
``simulator2.torch_rhs``, and emits the per-scenario ``.npz`` files plus the four
dataset-level manifests.

Usage::

    python scripts/build_dataset.py --n-scenarios 500 --out data/pilot
    python scripts/build_dataset.py --n-scenarios 10000 --out data/v1 --batch-size 128

Two things about this driver are worth knowing before reading it.

**Batching needs a shared time grid.** ``rollout_rk4`` integrates one batch
against one output grid, but sampled durations vary continuously (180-600 s).
The driver therefore snaps every duration to an exact multiple of ``dt_s``, so
all grids lie on one lattice and a shorter scenario's grid is an exact prefix of
a longer one. Batches are then rolled out on the longest grid in the batch and
sliced per scenario. Scenarios are sorted by ``(T, N)`` so batches are
homogeneous and little of that is wasted.

**No per-scenario wall-clock budget is needed any more.** `NEXT_STEPS.md` called
one mandatory, because adaptive stepping's cost varied 61x across driving
regimes and a pathological pairing could hang the build. Fixed-step RK4 removes
the failure mode at the source: cost is now ``ceil(duration/dt)`` steps
regardless of regime. (Which is just as well -- the ``SIGALRM`` mechanism
``scripts/bench_scenarios.py`` used is Unix-only and this workstation is native
Windows.)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from simulator2.air_brake import AirBrakeParams, require_sourced, unsourced
from simulator2.consist import Train
from simulator2.dataset import (
    build_scenario_arrays,
    compute_norm_stats,
    route_grade_stats,
    scenario_constants,
    speed_limit_stats,
    write_manifests,
    write_route_npz,
    write_scenario_arrays,
)
from simulator2.driving_regimes import (
    REGIMES_STARTING_FROM_REST,
    Regime,
    generate_regime_profile,
    sample_regime,
)
from simulator2.physics_config import (
    KNUCKLE_ULTIMATE_N,
    KNUCKLE_YIELD_N,
    LEGACY,
    PRESETS,
    PhysicsConfig,
    configure_batch,
    configure_consist,
    materialize_commands,
    scenario_outputs,
)
from simulator2.power_assignment import (
    DPU_MID_LABEL,
    DPU_REAR_LABEL,
    MAX_LOCOS,
    locos_required,
    ruling_sin_theta,
    with_loco_count,
)
from simulator2.randomize import RandomizationConfig, make_randomized_scenario
from simulator2.route import RouteProfile
from simulator2.torch_rhs import TorchScenarioBatch, rollout_rk4
from simulator2.train_generator import ensure_extended_car_library
from simulator2.vehicle_physics import LOCO_SUBSTITUTES

DEFAULT_ROUTE_DIR = _REPO_ROOT / "route_generator" / "route_profiles"
DEFAULT_TRAIN_MANIFEST = _REPO_ROOT / "saved_trains" / "generated" / "manifest.json"
DEFAULT_TRAIN_DIR = _REPO_ROOT / "saved_trains"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_routes(route_dir: Path) -> Dict[str, RouteProfile]:
    """Load every corridor in ``route_dir``, keyed by a stable ``route_id``."""
    routes = {}
    for p in sorted(Path(route_dir).glob("*.npz")):
        routes[p.stem] = RouteProfile.from_tensor_npz(p)
    if not routes:
        raise SystemExit(f"no route profiles found in {route_dir}")
    return routes


def load_trains(manifest_path: Path, train_dir: Path, limit: Optional[int] = None,
                physics: PhysicsConfig = LEGACY):
    """Load the generated consists and resolve each to vehicles + couplers
    under ``physics`` (legacy: :func:`build_consist`)."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if limit:
        manifest = manifest[:limit]
    library = ensure_extended_car_library()
    out = []
    for entry in manifest:
        train = Train.load(Path(train_dir) / entry["filename"])
        consist = configure_consist(train, library, physics)
        out.append({"entry": entry, "train": train, "consist": consist,
                    "vehicles": consist.vehicles, "couplers": consist.couplers})
    return out


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def assign_split(
    route_id: str,
    consist_split: str,
    rng: random.Random,
    policy: Dict[str, object],
) -> str:
    """Decide which split a scenario belongs to.

    Precedence matters. A corridor reserved for ``control_eval`` must never leak
    into surrogate training, so corridor reservations are checked before the
    consist-level ``ood_size`` flag, and both before the random train/val/test
    partition.
    """
    if route_id in policy["control_eval_corridors"]:
        return "control_eval"
    if route_id in policy["ood_corridors"]:
        return "ood_corridor"
    if route_id in policy["ood_grade_corridors"]:
        return "ood_grade"
    if consist_split == "ood_size":
        return "ood_size"
    r = rng.random()
    if r < policy["val_fraction"]:
        return "val"
    if r < policy["val_fraction"] + policy["test_id_fraction"]:
        return "test_id"
    return "train"


def build_split_policy(
    routes: Dict[str, RouteProfile],
    ood_corridors: Sequence[str],
    control_eval_corridors: Sequence[str],
    ood_grade_corridors: Sequence[str],
    auto_ood_grade: bool,
    val_fraction: float,
    test_id_fraction: float,
) -> Dict[str, object]:
    """Resolve the split policy, picking the steepest corridor for ``ood_grade``
    when asked to choose automatically."""
    ood_grade = set(ood_grade_corridors)
    if auto_ood_grade and not ood_grade:
        reserved = set(ood_corridors) | set(control_eval_corridors)
        candidates = {k: route_grade_stats(v)["grade_rms_pct"]
                      for k, v in routes.items() if k not in reserved}
        if candidates:
            ood_grade = {max(candidates, key=candidates.get)}
    return {
        "ood_corridors": set(ood_corridors),
        "control_eval_corridors": set(control_eval_corridors),
        "ood_grade_corridors": ood_grade,
        "val_fraction": val_fraction,
        "test_id_fraction": test_id_fraction,
    }


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def snap_to_lattice(scenario, dt_s: float):
    """Force a scenario's output grid onto the global ``dt_s`` lattice.

    ``make_randomized_scenario`` builds ``t_eval = linspace(0, duration,
    round(duration/dt)+1)``, so the spacing is only approximately ``dt_s`` and
    every scenario gets a slightly different grid. Batched rollout needs one
    shared grid, so the duration is snapped down to an exact multiple of
    ``dt_s``. The change is at most one sample and it makes short scenarios
    exact prefixes of long ones.
    """
    n_samples = int(np.asarray(scenario.t_eval).size)
    duration = (n_samples - 1) * dt_s
    scenario.t_eval = np.arange(n_samples, dtype=float) * dt_s
    scenario.t_span = (0.0, duration)
    return scenario, n_samples, duration


class PowerSizer:
    """Per-scenario locomotive counts under ``power_sizing = "ruling_grade"``.

    Consists are cached by (train, count) and ruling grades by (route, train),
    so a corpus builds each distinct consist once.
    """

    def __init__(self, physics: PhysicsConfig):
        self.physics = physics
        self.library = ensure_extended_car_library()
        self._consists: Dict[Tuple[str, int], dict] = {}
        self._ruling: Dict[Tuple[str, str], float] = {}

    def power(self, tr: dict, route_id: str, route: RouteProfile) -> dict:
        """``tr`` re-powered for ``route``: a dict like :func:`load_trains`'s
        entries plus ``loco_count`` and ``ruling_grade_pct``. Raises
        ``ValueError`` if the route needs more than ``MAX_LOCOS`` units."""
        key = tr["entry"]["filename"]
        wagons = [ct for ct, lo in zip(tr["consist"].car_types, tr["consist"].is_loco)
                  if not lo]
        if (route_id, key) not in self._ruling:
            self._ruling[(route_id, key)] = ruling_sin_theta(
                route, sum(ct.length_m for ct in wagons))
        sin_r = self._ruling[(route_id, key)]
        spec = LOCO_SUBSTITUTES[tr["train"].cars[0].car_type_id]
        n = locos_required(wagons, spec, sin_r)
        if n > MAX_LOCOS:
            raise ValueError(f"needs more than {MAX_LOCOS} locomotives for the ruling grade")
        if (key, n) not in self._consists:
            train = with_loco_count(tr["train"], self.library, n)
            consist = configure_consist(train, self.library, self.physics)
            self._consists[(key, n)] = {
                "train": train, "consist": consist,
                "vehicles": consist.vehicles, "couplers": consist.couplers,
                "total_mass_kg": float(sum(v.mass_kg for v in consist.vehicles)),
                "distributed_power": any(c.label in (DPU_MID_LABEL, DPU_REAR_LABEL)
                                         for c in train.cars),
            }
        return {**tr, **self._consists[(key, n)], "loco_count": n,
                "ruling_grade_pct": 100.0 * sin_r}


def sample_scenarios(
    n_scenarios: int,
    routes: Dict[str, RouteProfile],
    trains: List[dict],
    cfg: RandomizationConfig,
    policy: Dict[str, object],
    seed: int,
    physics: PhysicsConfig = LEGACY,
) -> Tuple[List[dict], Counter]:
    """Build the scenario specs. Cheap relative to integration, so done up front."""
    sizer = PowerSizer(physics) if physics.power_sizing == "ruling_grade" else None
    rng = random.Random(seed)
    split_rng = random.Random(seed + 1)
    route_ids = sorted(routes)
    grade_stats = {rid: route_grade_stats(r) for rid, r in routes.items()}

    specs: List[dict] = []
    failures: Counter = Counter()

    for i in range(n_scenarios):
        route_id = rng.choice(route_ids)
        route = routes[route_id]
        tr = rng.choice(trains)
        regime = sample_regime(rng)
        scenario_seed = seed * 1_000_003 + i

        profile = generate_regime_profile(regime, seed=scenario_seed)
        power = {
            "loco_count": int(tr["entry"].get("loco_count", 0)),
            "distributed_power": bool(tr["entry"].get("distributed_power", False)),
            "total_mass_kg": float(tr["entry"].get("total_mass_kg", 0.0)),
        }
        try:
            if sizer is not None:
                tr = sizer.power(tr, route_id, route)
                power = {k: tr[k] for k in ("loco_count", "distributed_power", "total_mass_kg")}
                power["loco_count_manifest"] = int(tr["entry"].get("loco_count", 0))
                power["ruling_grade_pct"] = round(tr["ruling_grade_pct"], 4)
            scenario, meta = make_randomized_scenario(
                route, tr["vehicles"], tr["couplers"], profile,
                train=tr["train"], seed=scenario_seed, config=cfg,
                starts_from_rest=(regime in REGIMES_STARTING_FROM_REST),
                route_id=route_id,
                control_id=str(regime.value),
                consist_id=tr["entry"]["filename"],
            )
        except ValueError as exc:
            # Most often: the route is shorter than the consist plus margins.
            failures[str(exc).split("(")[0].strip()[:60]] += 1
            continue

        scenario, n_samples, duration = snap_to_lattice(scenario, cfg.dt_s)
        split = assign_split(route_id, tr["entry"].get("split", "train"),
                             split_rng, policy)
        scenario_id = f"{route_id}_run{i:06d}"

        meta.update({
            "scenario_id": scenario_id,
            "corridor_name": route_id,
            "region": "unknown",  # the route pipeline does not yet emit one
            "consist_label": tr["entry"].get("consist_label", ""),
            "control_label": str(regime.value),
            **power,
            "T": int(n_samples),
            "duration_s": round(float(duration), 6),
            "dt_s": round(float(cfg.dt_s), 6),
            "split": split,
            **grade_stats[route_id],
        })
        specs.append({"scenario": scenario, "meta": meta, "route_id": route_id,
                      "profile": profile, "train": tr["train"],
                      "consist": tr["consist"], "regime": str(regime.value)})

    return specs, failures


# ---------------------------------------------------------------------------
# integration + writing
# ---------------------------------------------------------------------------

def run_batches(
    specs: List[dict],
    routes: Dict[str, RouteProfile],
    out_dir: Path,
    batch_size: int,
    dt: float,
    device: str,
    dtype: torch.dtype,
    compress: bool = True,
    progress_every: int = 5,
    physics: PhysicsConfig = LEGACY,
    diverged: Optional[List[str]] = None,
    parted: Optional[List[str]] = None,
) -> List[dict]:
    """Integrate the specs in batches and write one ``.npz`` per scenario.

    Scenarios whose state goes non-finite are not written; their ids are
    appended to ``diverged``. If ``parted`` is a list, scenarios whose peak
    coupler force exceeds the knuckle's ultimate strength are not written
    either, and their ids are appended to it: the real train would have parted
    there and the model has no coupler failure. Runs between yield and
    ultimate are kept and flagged.
    """
    # Sort by (T, N): T sets the step count and therefore the real cost, while
    # throughput is flat in N, so grouping by T first keeps batches homogeneous
    # where it matters.
    order = sorted(range(len(specs)),
                   key=lambda k: (specs[k]["meta"]["T"], specs[k]["meta"]["N"]))
    records: List[dict] = []
    diverged = [] if diverged is None else diverged
    n_batches = math.ceil(len(order) / batch_size)
    t_start = time.perf_counter()

    for bi in range(n_batches):
        chunk = [specs[k] for k in order[bi * batch_size:(bi + 1) * batch_size]]
        scenarios = [c["scenario"] for c in chunk]

        cmds = [
            materialize_commands(
                c["profile"], c["consist"], c["scenario"].t_eval, c["train"], physics,
                regime=c["regime"], route=routes[c["route_id"]],
                x0=np.asarray(c["scenario"].y0)[:len(c["scenario"].vehicles)],
                v0=float(np.asarray(c["scenario"].y0)[len(c["scenario"].vehicles)]))
            for c in chunk
        ]
        batch = TorchScenarioBatch.from_scenarios(
            scenarios, device=device, dtype=dtype,
            commands=[(m.u_trac, m.u_brk) for m in cmds])
        batch = configure_batch(batch, [c["consist"] for c in chunk], cmds, physics)
        t_max = max((np.asarray(s.t_eval) for s in scenarios), key=lambda a: a.size)
        t_grid = torch.as_tensor(t_max, dtype=dtype, device=device)
        _, y_out = rollout_rk4(batch, batch.y0, t_grid, dt=dt)
        y_np = y_out.detach().cpu().numpy()

        for i, c in enumerate(chunk):
            sc, meta = c["scenario"], c["meta"]
            n, n_t = len(sc.vehicles), int(meta["T"])
            # Slice off this scenario's own prefix of the shared grid, and the
            # padded vehicle columns of the shared width.
            width = batch.n_vehicles
            y_flat = y_np[i, :n_t, :]
            y_scn = np.concatenate(
                [y_flat[:, blk * width: blk * width + n] for blk in range(4)], axis=1
            )
            # Under the draft gear, one gear state per connection follows the
            # four node blocks.
            gear = y_flat[:, 4 * width: 4 * width + n - 1]
            if not (np.isfinite(y_scn).all() and np.isfinite(gear).all()):
                # A diverged run is not data. Count it rather than write it.
                print(f"  NOT WRITTEN (non-finite state): {meta['scenario_id']}",
                      flush=True)
                diverged.append(meta["scenario_id"])
                continue
            t_scn = np.asarray(sc.t_eval)[:n_t]
            # Commands were already materialized when the batch was built; the
            # per-vehicle callables are expensive (T*N Python calls), so read
            # them back off the batch rather than paying for them twice.
            u_trac = batch.u_trac[i, :n_t, :n].double().cpu().numpy()
            u_brk = batch.u_brk[i, :n_t, :n].double().cpu().numpy()

            node_static = batch.node_static[i, :n].double().cpu().numpy()
            edge_static = batch.edge_static[i, :max(n - 1, 0)].double().cpu().numpy()

            state = np.stack([y_scn[:, k * n:(k + 1) * n] for k in range(4)], axis=-1)
            edge_dyn, power, extras = scenario_outputs(
                physics, state, gear, edge_static, c["consist"], cmds[i], batch)
            arrays = build_scenario_arrays(
                t=t_scn, y=y_scn, node_static=node_static, edge_static=edge_static,
                u_trac=u_trac, u_brk=u_brk, constants=scenario_constants(sc),
                edge_dynamic=edge_dyn, power=power, extras=extras,
            )
            meta.update({k: float(v) for k, v in arrays.summaries.items()})
            meta["coupler_over_yield"] = bool(meta["F_max"] > KNUCKLE_YIELD_N)
            meta["coupler_over_ultimate"] = bool(meta["F_max"] > KNUCKLE_ULTIMATE_N)
            if parted is not None and meta["coupler_over_ultimate"]:
                print(f"  NOT WRITTEN (coupler force {meta['F_max'] / 1e6:.2f} MN above "
                      f"the knuckle ultimate): {meta['scenario_id']}", flush=True)
                parted.append(meta["scenario_id"])
                continue
            route = routes[meta["route_id"]]
            meta.update(speed_limit_stats(
                arrays.state,
                np.asarray(route.s_nodes_m, dtype=np.float64),
                np.asarray(route.v_max_nodes, dtype=np.float64),
            ))
            path = out_dir / "scenarios" / meta["route_id"] / f"{meta['scenario_id']}.npz"
            write_scenario_arrays(arrays, path, meta, compress=compress)
            records.append(meta)

        if progress_every and (bi + 1) % progress_every == 0 or bi == n_batches - 1:
            done, el = len(records), time.perf_counter() - t_start
            print(f"  batch {bi + 1}/{n_batches}  {done}/{len(specs)} scenarios  "
                  f"{el:.1f}s  ({el / max(done, 1) * 1000:.0f} ms/scenario)", flush=True)

    return records


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scenarios", type=int, default=100)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "data" / "pilot")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="scenarios per GPU batch; graph replays are per-step, "
                         "so per-scenario cost falls roughly as 1/batch-size")
    ap.add_argument("--no-compress", action="store_true",
                    help="write raw .npz instead of zlib (~25%% faster, ~2x disk)")
    ap.add_argument("--dt", type=float, default=None,
                    help="integration step (s); default per --physics "
                         "(0.02 legacy, 0.01 with the draft gear)")
    ap.add_argument("--dt-out", type=float, default=0.25, help="output sample spacing (s)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"),
                    help="integration precision; float64 costs ~20 %% on this "
                         "launch-bound workload and keeps the error budget for physics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-trains", type=int, default=None)
    ap.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    ap.add_argument("--duration-range", type=float, nargs=2, default=(180.0, 600.0))
    ap.add_argument("--k-curv-scale", type=float, default=0.0,
                    help="0 disables curvature. With --curvature-model roeckl "
                         "or linear, 1.0 is the standard formula")
    ap.add_argument("--curvature-model", default="proxy_v2",
                    choices=("proxy_v2", "roeckl", "linear"),
                    help="proxy_v2 is the original k*m*v^2*|kappa| law and the "
                         "default; roeckl and linear are speed-independent and "
                         "calibrated (see TORCH_PORT_REPORT.md)")
    ap.add_argument("--ood-corridors", nargs="*", default=[])
    ap.add_argument("--control-eval-corridors", nargs="*", default=[])
    ap.add_argument("--ood-grade-corridors", nargs="*", default=[])
    ap.add_argument("--auto-ood-grade", action="store_true",
                    help="reserve the steepest unreserved corridor for ood_grade")
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--test-id-fraction", type=float, default=0.10)
    ap.add_argument("--physics", default="legacy", choices=sorted(PRESETS),
                    help="legacy: the Chapter 4 physics of data/v1-v2. v3: the "
                         "benchmark-validated configuration plus the F1-F9 fixes "
                         "(simulator2/physics_config.py)")
    ap.add_argument("--allow-unsourced-brake", action="store_true",
                    help="DRY RUN ONLY: build v3 although the air brake's force "
                         "parameters are placeholders (recorded in the output)")
    args = ap.parse_args()
    physics = PRESETS[args.physics]
    if args.dt is None:
        args.dt = physics.default_dt_s
    unsourced_brake: list = []
    if physics.air_brake:
        unsourced_brake = unsourced(AirBrakeParams())
        if unsourced_brake and not args.allow_unsourced_brake:
            require_sourced(AirBrakeParams())

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading routes from {args.route_dir}")
    routes = load_routes(args.route_dir)
    print(f"  {len(routes)} corridors: {', '.join(sorted(routes))}")

    print("loading consists")
    trains = load_trains(DEFAULT_TRAIN_MANIFEST, DEFAULT_TRAIN_DIR, args.max_trains,
                         physics=physics)
    sizes = [t["entry"]["N"] for t in trains]
    print(f"  {len(trains)} consists, N in [{min(sizes)}, {max(sizes)}]")

    policy = build_split_policy(
        routes, args.ood_corridors, args.control_eval_corridors,
        args.ood_grade_corridors, args.auto_ood_grade,
        args.val_fraction, args.test_id_fraction,
    )

    cfg = RandomizationConfig(
        duration_s_range=tuple(args.duration_range),
        dt_s=args.dt_out,
        k_curv_scale=args.k_curv_scale,
        curvature_model=args.curvature_model,
        v0_limit_over_train=physics.speed_planner,
    )

    print(f"sampling {args.n_scenarios} scenarios")
    t0 = time.perf_counter()
    specs, failures = sample_scenarios(args.n_scenarios, routes, trains, cfg,
                                       policy, args.seed, physics=physics)
    for spec in specs:
        spec["meta"]["physics"] = physics.name
        if unsourced_brake:
            spec["meta"]["unsourced_air_brake_fields"] = unsourced_brake
    print(f"  {len(specs)} built in {time.perf_counter() - t0:.1f}s")
    for reason, count in failures.most_common():
        print(f"  skipped {count}: {reason}")
    if not specs:
        raise SystemExit("no scenarios could be built")

    print(f"integrating on {args.device} in {args.dtype}, "
          f"batch={args.batch_size}, dt={args.dt}")
    t0 = time.perf_counter()
    diverged: List[str] = []
    # Legacy corpora keep their original contents, so the exclusion is v3 only.
    parted: Optional[List[str]] = None if physics.is_legacy else []
    records = run_batches(specs, routes, out_dir, args.batch_size, args.dt,
                          args.device, dtype, compress=not args.no_compress,
                          physics=physics, diverged=diverged, parted=parted)
    if diverged:
        print(f"  {len(diverged)} scenario(s) diverged and were not written")
    if parted:
        print(f"  {len(parted)} scenario(s) exceeded the knuckle ultimate and were not written")
    integrate_s = time.perf_counter() - t0

    print("writing route fields")
    used = sorted({r["route_id"] for r in records})
    route_paths = [write_route_npz(routes[rid], rid, out_dir / "routes") for rid in used]

    split_counts = Counter(r["split"] for r in records)
    print("splits:")
    for name in ("train", "val", "test_id", "ood_size",
                 "ood_grade", "ood_corridor", "control_eval"):
        n = split_counts.get(name, 0)
        flag = "" if n else "   <-- EMPTY"
        print(f"  {name:14s} {n:6d}{flag}")

    print("computing norm_stats over the train split only")
    train_paths = [
        out_dir / "scenarios" / r["route_id"] / f"{r['scenario_id']}.npz"
        for r in records if r["split"] == "train"
    ]
    if not train_paths:
        raise SystemExit("train split is empty; cannot compute norm_stats")
    train_routes = sorted({r["route_id"] for r in records if r["split"] == "train"})
    norm_stats = compute_norm_stats(
        train_paths, [out_dir / "routes" / f"{rid}.npz" for rid in train_routes]
    )

    build_config = {
        "n_scenarios_requested": args.n_scenarios,
        "n_scenarios_written": len(records),
        "seed": args.seed,
        "integration": {"dt": args.dt, "dtype": args.dtype, "device": args.device,
                        "batch_size": args.batch_size,
                        "compressed": not args.no_compress,
                        "method": "fixed-step RK4 (simulator2.torch_rhs)"},
        "randomization": asdict(cfg),
        "physics": asdict(physics),
        "unsourced_air_brake_fields": unsourced_brake,
        "split_policy": {k: (sorted(v) if isinstance(v, set) else v)
                         for k, v in policy.items()},
        "routes": {rid: route_grade_stats(routes[rid]) for rid in used},
        "integrate_seconds": round(integrate_s, 2),
        "sampling_failures": dict(failures),
        "diverged_scenarios": diverged,
        "over_knuckle_ultimate_scenarios": parted,
    }
    written = write_manifests(out_dir, records, norm_stats, build_config)

    total_mb = sum(p.stat().st_size for p in out_dir.rglob("*.npz")) / 1e6
    print(f"\nwrote {len(records)} scenarios + {len(route_paths)} routes "
          f"to {out_dir}  ({total_mb:.1f} MB)")
    for k, p in written.items():
        print(f"  {k:14s} {p.relative_to(out_dir)}")
    print(f"integration: {integrate_s:.1f}s "
          f"({integrate_s / len(records) * 1000:.0f} ms/scenario)")


if __name__ == "__main__":
    main()
