# Dataset Build — Results

_Written 2026-09-03 on the RTX 4090 workstation. Covers steps 4–6 of
`NEXT_STEPS.md`. Companion to `DATA_SCHEMA.md` (what the files must contain) and
`TORCH_PORT_REPORT.md` (the integrator underneath). Source material for the data
chapter._

---

## What exists

Two corpora, built from the same seed and the same code, differing only in the
curvature law:

| | `data/v1` | `data/v2` |
|---|---|---|
| curvature model | `proxy_v2`, `k = 0.0` (off) | `linear`, `k = 1.0` |
| scenarios | 10,000 | 10,000 |
| on disk | 12.4 GB | 12.4 GB |
| integration | 1336 s (134 ms/scenario) | 1389 s (139 ms/scenario) |
| git commit | `fd14fcd` | `fd14fcd` |

Both were produced by:

```bash
python scripts/build_dataset.py --n-scenarios 10000 --batch-size 256 \
    --dtype float64 --auto-ood-grade --ood-corridors route_line5_22km \
    --out data/v1                       # v2 adds: --curvature-model linear --k-curv-scale 1.0
```

`data/v2` is the one to train on. `data/v1` exists so the curvature term can be
measured on the real corpus rather than argued about; see
`CURVATURE_MODEL_NOTE.md`.

---

## Code

| file | role |
|---|---|
| `simulator2/dataset.py` | serializer: arrays A–H, route files, `norm_stats`, manifests |
| `scripts/build_dataset.py` | driver: sample → batch → integrate on GPU → write |
| `scripts/validate_dataset.py` | integrity checker, run against a finished build |
| `simulator2/tests/test_dataset.py` | 10 tests |
| `simulator2/tests/test_curvature_models.py` | 14 tests |

Suite: **220 passing** (135 pre-existing + 60 torch acceptance + 10 dataset + 14
curvature, plus 1 added to the driving-regime tests).

---

## Contents of a build

Per `DATA_SCHEMA.md`. One compressed `.npz` per scenario under
`scenarios/<route_id>/`, route fields written **once per corridor** under
`routes/`, four manifests at the top level.

```
data/v2/
  build_config.json     sampling ranges, git commit, seed, integrator settings
  index.parquet         10,000 rows x 43 columns  (the §I metadata record)
  norm_stats.json       per-channel mean/std, TRAIN SPLIT ONLY
  splits.json           explicit scenario_id lists per split
  routes/               5 corridors x (route_s, route_sin_theta, route_kappa, route_vmax)
  scenarios/<route_id>/<scenario_id>.npz
```

Each scenario file holds `state [T,N,4]`, `node_static [N,7]`,
`edge_dynamic [T,N-1,3]`, `edge_static [N-1,6]`, `edge_index [2,N-1]`,
`u_trac`/`u_brk [T,N]`, `y0 [4N]`, the §H integrator constants, the summary
scalars, and `metadata_json`.

Two schema decisions are load-bearing and worth restating:

- **Static and dynamic are separated.** `H_hist` is `[T,N,11]` but seven of
  those channels never change, so the writer never materializes it.
- **Routes are referenced, not duplicated.** A corridor is reused across ~2,000
  scenarios; duplicating `R`-length arrays would dominate the size for no
  information gain. `edge_dynamic` *is* stored despite being derivable, as a
  supervision target, and is flagged `edge_dynamic_is_derived`.

### Scale

- **12,796,900 timesteps**, **752,720,421 vehicle-steps** per corpus
- `N` 10–150, `T` 721–2,397, duration 180–599 s (mean 320 s)
- 500 distinct consists, 11 regimes, 5 corridors

### Composition

| split | n | source |
|---|---|---|
| train | 4,043 | corridors line0 / line3 / line4 |
| val | 517 | random 10% of eligible |
| test_id | 496 | random 10% of eligible |
| ood_size | 955 | consists flagged `ood_size` in the train manifest |
| ood_grade | 1,999 | `route_line2` (steepest, rms 2.46%) |
| ood_corridor | 1,990 | `route_line5` |
| **control_eval** | **0** | **not buildable — see caveats** |

Regime mix (as sampled by `DEFAULT_REGIME_WEIGHTS`): `throttle_modulation` 1514,
`cruise` 1473, `notch_up` 1230, `power_to_coast` 1016,
`dynamic_brake_descent` 1001, `coast_to_brake` 995, `coast` 808, `startup` 782,
`brake_release` 598, `stretch_brake` 397, `emergency` 186.

Adhesion: dry 6079, wet 2940, low 981.

---

## Throughput

The driver batches scenarios onto the GPU through `simulator2.torch_rhs`.
Measured while building:

| | |
|---|---|
| per scenario | 134 ms (v1), 139 ms (v2) |
| full 10,000-scenario build | ~22 min integration, ~25 min wall clock |
| NumPy baseline, single core | 89.2 s per *120 s* scenario at `N=130` |

These runs average 320 s, so the like-for-like NumPy figure is well over 200 s
per scenario. `NEXT_STEPS.md` had estimated ~170 CPU-hours for a corpus of this
size and noted large consists were effectively excluded; the actual build is
under half an hour, and `N=150` costs no more per scenario than `N=11`.

Two knobs matter, and both were measured rather than guessed:

- **Batch size.** Graph replays are per *step*, not per scenario, so
  per-scenario cost falls roughly as `1/batch`. At `B=8` the pilot ran 461
  ms/scenario; at `B=256`, 84 ms on the same scenarios. Default is 128; 256 was
  used for these builds.
- **Compression.** zlib costs ~62 ms/scenario, about a quarter of the build, and
  buys ~2× on disk. `--no-compress` is available. Both corpora are compressed.

---

## Validation

`scripts/validate_dataset.py` runs three classes of check. Both corpora **PASS**.

**1. Manifest consistency.** `index.parquet` and `splits.json` describe the same
scenarios, splits are disjoint, every referenced scenario and route file exists,
`norm_stats` was computed over the train split and contains no zero divisors.

**2. Physics self-consistency.** The RHS is rebuilt from *only* what the
scenario `.npz` and its route file contain — a standalone reimplementation, so a
shared bug with `simulator2` cannot hide — and compared against the derivative
implied by the stored trajectory. This is what catches a dropped §H constant or
a permuted `node_static` column.

The comparison is against a central difference on the 0.25 s **output** grid,
which carries its own `O(Δt²)` error. That error is not a defect in the data, and
it was verified rather than assumed: re-sampling one fixed scenario
(`N = 31`, cruise, `route_line2`, `linear` curvature) at progressively finer
output spacing gives

| `dt_out` (s) | max \|dx/dt\| residual (m/s) | max \|dv/dt\| residual (m/s²) |
|---|---|---|
| 0.25 | 2.105e-02 | 1.935e-01 |
| 0.10 | 4.474e-03 | 5.571e-02 |
| 0.05 | 1.131e-03 | 1.632e-02 |
| 0.02 | 1.863e-04 | 2.471e-03 |
| 0.01 | 4.588e-05 | 6.126e-04 |

Each 2× refinement cuts the residual ~4×: clean second-order convergence, i.e.
the residual is the finite difference, not the trajectory. (The same check on
`data/v1` without curvature gives 2.230e-02 → 3.882e-05 over the same range.) **Consequence for training: do not derive
`d(state)/dt|_true` by finite-differencing `state`,** as §Consumption of
`DATA_SCHEMA.md` suggests. At 0.25 s the error is the same order as the residual
the GNN is meant to learn. Recompute the RHS analytically from the stored
arrays; §H guarantees everything needed is present.

**3. Distribution sanity.** Speed-limit violations, reversals, consist sizes and
force ranges, so a build whose *inputs* went wrong is visible without opening
files.

---

## Known data-quality caveats

### Overspeed — the corridor set, not the simulator

The open-loop RHS deliberately does not enforce `route_vmax`: the field is
exogenous information for the controller. Nothing therefore stops a coasting
consist accelerating down a steep grade for 600 s.

| | `data/v1` | `data/v2` |
|---|---|---|
| never over the limit | 72.1% | 72.9% |
| over by > 5 m/s | 18.6% | 17.8% |
| over by > 20 m/s | 2.7% | 2.4% |
| worst | 71.3 m/s | 67.8 m/s |

This tracks the grade field exactly. Every corridor is pinned at the route
pipeline's `grade_clip = 0.04` ceiling, and in the pilot 39% of `route_line2`
scenarios exceeded the limit by >5 m/s against 2% of `route_line5`. It is
faithful physics on a flawed input — see `NEXT_STEPS.md` item 1.

Each scenario records `v_over_limit_max_mps`, `frac_time_over_limit` and
`v_limit_min_mps` in `index.parquet`, so these are filterable without
rebuilding: `idx[idx.v_over_limit_max_mps <= 5]` retains ~81%. They are
**recorded rather than dropped**, because whether an overspeed run is bad data or
a useful hard case is a modelling decision, not a serialization one.

Related: **22% of scenarios reverse below −1 m/s.** Same root cause — the
randomizer rejection-samples standing starts onto track flatter than 0.5%, and
on a corridor set pinned at 4% it often cannot find any.

### `control_eval` is empty

The driver supports all seven splits and checks corridor reservations *before*
the consist-level `ood_size` flag, so a `control_eval` corridor cannot leak into
training. The obstruction is the input: with 5 corridors every holdout costs
~20% of the corpus, and reserving a third would leave only two corridors for
training. Populating it needs the wider corridor set from `NEXT_STEPS.md` item 1.

### `T_arr` is `NaN`

`DATA_SCHEMA.md` §G asks for "arrival time at target chainage". An open-loop
scenario has no target, so `T_arr` and `target_chainage_m` are `NaN` throughout;
`arrival_time()` exists for the control-eval runs, which will have one.
`distance_travelled_m` is stored as a meaningful substitute.

---

## Two defects found and fixed during this work

Both are recorded because they have implications beyond the code.

### Builds were not reproducible from their own seed

`generate_regime_profile` seeded its RNG with `hash(regime.value)`. `Regime` is a
`str` enum and CPython randomizes string hashes per process, so a function
documented as *"Deterministic for a given `(regime, seed)`"* was stable within a
run and different in the next. Two identical invocations of the driver sampled
different durations, start chainages, initial speeds and slack states — only
quantities drawn *before* the profile (adhesion, regime) matched.

It surfaced only because two builds that should have shared scenarios did not.
Fixed with `zlib.crc32`, which is stable across processes and platforms.

The lesson worth carrying: **`test_deterministic_for_seed` could not catch this**,
because it calls the function twice in one process where `hash()` is stable. The
replacement, `test_deterministic_across_processes`, forks under three
`PYTHONHASHSEED` values, and was verified to fail against the old seeding before
being kept. Any same-process determinism test has this blind spot.

**Any dataset built before commit `fd14fcd` is unreproducible from its
`build_config.json`.** Both corpora here were rebuilt afterwards.

### The validator silently ignored the curvature law

`validate_dataset.py` hardcoded the `proxy_v2` law when recomputing the RHS.
Against a build using a speed-independent law that is wrong by a factor of ~33 at
15 m/s — and the deliberately loose finite-difference tolerance absorbed it, so
the first `data/v2` passed validation *without its curvature term ever being
checked*. It now reads `curvature_model` from the file. A generous tolerance and
a hardcoded assumption combine badly.

---

## Reproducing

```bash
python scripts/build_dataset.py --n-scenarios 10000 --batch-size 256 \
    --dtype float64 --curvature-model linear --k-curv-scale 1.0 \
    --auto-ood-grade --ood-corridors route_line5_22km --out data/v2
python scripts/validate_dataset.py data/v2
```

`build_config.json` records the git commit, seed, sampling ranges and integrator
settings for each build. At commit `fd14fcd` or later the seed genuinely
reproduces the corpus.

**Precision.** Both corpora were integrated in **float64** and stored float32.
`TORCH_PORT_REPORT.md` measures float64 at 1.20× the cost of float32 — not the
64× the port spec assumed, because the rollout is latency-bound rather than
FLOP-bound — while float32 rounding alone consumes 60% of the 0.05 m/s velocity
tolerance on the worst fixture over just 40 s. The module default is still
float32 to match the spec's API; `--dtype` is explicit in the commands above.

---

## What the data chapter still needs

1. **A wider corridor set** — the binding constraint. It drives the overspeed
   population, the empty `control_eval` split, and the fact that training sees
   only three corridors from one region. Needs the route pipeline with OSM + DEM
   network access.
2. **A decision on the curvature law** — `CURVATURE_MODEL_NOTE.md`.
3. **A decision on whether to filter overspeed scenarios**, and if so at what
   threshold. The fields are in `index.parquet`; nothing has been dropped.
