# Dataset Schema — LTD Surrogate Training Data

_Companion to `PROJECT_DIRECTION.md`. Defines what the route/data-generation step must emit so
the GNN/Neural-ODE surrogate (and the control demo) can consume it directly._

Grounded in `simulator2`: `state_schema.NodeChannel` / `EdgeChannel`,
`io_types.TensorSimulationResult` (`t`, `H_hist`, `E_hist`), and `route.RouteProfile`.
All quantities SI (see `UNITS.md`).

---

## Design principles

1. **Separate static from dynamic.** `H_hist` is `[T, N, 11]`, but 7 of those 11 channels are
   constant in time (mass, Davis, limits). Storing them at every timestep is wasteful and
   error-prone. Split: dynamic state `[T, N, 4]` + static node params `[N, 7]` stored once.
2. **Store the raw route field, not pre-sampled grade.** During free rollout the model samples
   the route at its *own predicted* positions, so a per-timestep grade array baked at the true
   positions is only valid under teacher forcing. The canonical route object is the arc-length
   field; sampling happens inside the model's RHS.
3. **Store ragged (true `N`, true `R`); pad/batch in the dataloader.** A message-passing GNN
   handles variable `N` natively (disjoint-union graph batching, PyG-style) with no padding. Keep
   the on-disk data at true sizes; collate at load time.
4. **Normalization lives in a dataset-level manifest, not per scenario.** Compute per-channel
   mean/std over the *training split only* and store once; every consumer reads the same stats.
5. **Everything needed to recompute the RHS must be present.** A Neural-ODE/residual model
   re-evaluates the physics RHS, which needs static params + route + control + integrator
   constants. If it's an input to `train_rhs_extended`, it's in the schema.

---

## Per-scenario contents

One scenario = one simulator run. Arrays below, with `T` = timesteps, `N` = vehicles,
`R` = route arc-length nodes (all vary across scenarios).

### A. Time

| Field | Shape | Dtype | Role |
|---|---|---|---|
| `t` | `[T]` | f32 | integration time grid (s); from `result.t` |
| `dt` | scalar | f32 | nominal step (s); convenience, = mean diff of `t` |

### B. Dynamic node state — the ODE state (model predicts/integrates this)

Columns are `NodeChannel.X, V, Z_BRK, Z_TRAC` (indices 0–3 of `H_hist`).

| Field | Shape | Dtype | Role |
|---|---|---|---|
| `state` | `[T, N, 4]` | f32 | `[x, v, z_brk, z_trac]` per vehicle over time. **Rollout target / ground truth.** Slice `H_hist[:, :, 0:4]`. |

> The surrogate's learned RHS produces `d(state)/dt`; the Neural-ODE solver integrates it.
> Under physics-residual, the target for the *learned part* is the residual between true
> `d(state)/dt` and the physics RHS (see §Consumption).

### C. Static node params — per-vehicle, constant in time

Columns `NodeChannel.MASS_KG … F_BRK_MAX_N` (indices 4–10 of `H`).

| Field | Shape | Dtype | Role |
|---|---|---|---|
| `node_static` | `[N, 7]` | f32 | `[mass_kg, davis_A, davis_B, davis_C, can_traction, F_trac_max_N, F_brk_max_N]` |

### D. Edge (coupler) data

Dynamic edge channels `EdgeChannel.DELTA, DELTA_DOT, F_CPL` (0–2) are **algebraically derived**
from node state in this simulator — store them as supervision targets, but flag them derived.
Static edge channels (3–8) stored once.

| Field | Shape | Dtype | Role |
|---|---|---|---|
| `edge_dynamic` | `[T, N-1, 3]` | f32 | `[delta, delta_dot, F_cpl]`; supervision target (esp. `F_cpl` for safety). Slice `E_hist[:, :, 0:3]`. |
| `edge_static` | `[N-1, 6]` | f32 | `[L0_m, slack_half_m, k_draft, c_draft, k_buff, c_buff]` |
| `edge_index` | `[2, N-1]` | i32 | chain connectivity `(j, j+1)`; explicit for PyG/graph-batch compatibility |

### E. Control inputs — exogenous, time-varying

`simulator2` stores commands as callables `u_trac_cmd(t, i)` / `u_brk_cmd(t, i)`. **Materialize
them onto the time grid** for storage and learning.

| Field | Shape | Dtype | Role |
|---|---|---|---|
| `u_trac` | `[T, N]` | f32 | commanded traction per vehicle over time (N) |
| `u_brk` | `[T, N]` | f32 | commanded brake per vehicle over time (N) |

### F. Route field — the environment (store raw; model samples at positions)

Canonical = `RouteProfile` arc-length field. `R` is route-specific and independent of `N`/`T`.

| Field | Shape | Dtype | Role |
|---|---|---|---|
| `route_s` | `[R]` | f32 | arc-length nodes (m); `RouteProfile.s_nodes_m` |
| `route_sin_theta` | `[R]` | f32 | `sin θ(s)` grade field; `sin_theta_nodes` |
| `route_kappa` | `[R]` | f32 | curvature proxy `κ(s)` (0 if disabled) |
| `route_vmax` | `[R]` | f32 | speed limit `v_max(s)` (m/s); needed for control |

Optional, **derivable** convenience arrays (don't treat as ground truth):

| Field | Shape | Dtype | Role |
|---|---|---|---|
| `route_grid_grade` | `[G]` | f32 | grade resampled to uniform `Δs` (e.g. 50 m), `G` fixed dataset-wide — feeds the optional 1-D route **preview encoder** (the place attention earns its keep) |
| `grade_at_pos` | `[T, N]` | f32 | `sin θ` sampled at true vehicle positions; **teacher-forcing convenience only** |

### G. Scalar targets / summaries (derived; for summary-loss and eval)

| Field | Shape | Role |
|---|---|---|
| `E_trip` | scalar | trip energy `∫ P_trac dt` — the control objective |
| `T_arr` | scalar | arrival time at target chainage |
| `F_max` | scalar | peak `|F_cpl|` over `(t, j)` — safety metric |

### H. Integrator / physics constants (needed to recompute the physics RHS)

From `ExtendedTrainScenario`: `tau_brk_s`, `tau_trac_s`, `p_max_w`, `k_curv_scale`, and the
`y0` initial condition. Store as a small dict per scenario.

### I. Split / OOD metadata (the reason real terrain matters)

A flat record per scenario enabling clean held-out slices. **No experiment design works without
this.**

```
{
  "scenario_id":   "corridor07_run0143",
  "route_id":      "corridor07",        # hold out whole corridors for OOD
  "corridor_name": "Donner Pass W-bound",
  "region":        "...",
  "N":             64,                   # consist size  -> size-generalization splits
  "T":             2400,
  "R":             1810,
  "grade_max_pct": 2.2,                  # route severity stats
  "grade_rms_pct": 0.7,
  "route_len_km":  118.4,
  "adhesion":      "dry",                # dry / wet / low  -> regime splits
  "control_label": "coast_optimized",   # control strategy used
  "consist_label": "mixed_loaded",
  "seed":          143,
  "split":         "train"               # train / val / test_id / ood_size / ood_grade / ood_corridor / control_eval
}
```

---

## Dataset-level manifest (one per dataset build)

Separate from per-scenario files:

- `index.parquet` — one row per scenario = the metadata record in §I (fast filtering for splits).
- `norm_stats.json` — per-channel mean/std for `state`, `node_static`, `edge_static`,
  `edge_dynamic`, `u_*`, route fields. **Computed over the train split only.**
- `splits.json` — explicit `scenario_id` lists per split, including the reserved
  `control_eval` real routes that must never appear in surrogate training.
- `build_config.json` — sampling ranges, simulator git commit, DEM source/version, seeds.
  (Reproducibility + the calibration caveat live here.)

---

## File format & layout

Pragmatic for thesis scale (tens of thousands of scenarios):

```
data/
  build_config.json
  index.parquet
  norm_stats.json
  splits.json
  scenarios/
    corridor07/
      corridor07_run0143.npz      # one compressed npz per scenario (arrays A–H)
      ...
  routes/
    corridor07.npz                # route field stored ONCE per corridor (F), referenced by id
```

Store each **route once per corridor** and reference it by `route_id` from many scenarios — a
corridor is reused across hundreds of consist/control variations, so don't duplicate `R`-length
arrays. Use `npz` now; migrate to `zarr`/HDF5 only if scale demands streaming.

---

## How the GNN/Neural-ODE consumes it

1. **Graph build (per scenario):** nodes ← `node_static` (+ current dynamic `state`),
   edges ← `edge_static` with `edge_index`. Variable `N` handled by graph batching, no padding.
2. **Route sampling inside the RHS:** at integration time `t` with current positions `x_i(t)`
   (from `state`), interpolate `route_sin_theta`/`route_kappa`/`route_vmax` at each `x_i` →
   per-node forcing features. This is the exogenous-environment input; it is recomputed every
   solver evaluation, at the model's *own* positions during free rollout.
3. **Control gather:** interpolate `u_trac`/`u_brk` at `t` → per-node command features.
4. **Targets:**
   - *Residual training (recommended):* learned target = `d(state)/dt|_true − F_phys(state, …)`,
     so the GNN learns only the correction. `d(state)/dt|_true` from finite-difference of `state`
     or stored RHS.
   - *Rollout supervision:* integrate the surrogate and match `state` trajectories; add
     `edge_dynamic` (esp. `F_cpl`) and scalar (`E_trip`, `F_max`) losses.
5. **Teacher forcing vs. free rollout:** short-horizon transition training uses true positions
   (may use `grade_at_pos`); evaluation uses free rollout with on-the-fly route sampling at
   predicted positions. Report both — the gap is the honest measure of compounding error.
6. **Normalization:** apply `norm_stats` (train-only) to all inputs/targets; keep `F_cpl` and
   `v_max` in physical units for the safety constraints in the control chapter.

---

## What `simulator2` needs to emit this (small additions)

- A `materialize_commands(scenario, t)` helper → `u_trac [T,N]`, `u_brk [T,N]` from the callables.
- A `route_vmax` field on `RouteProfile` (currently only `sin_theta`/`kappa`).
- A `write_scenario_npz(result, scenario, metadata, path)` serializer producing arrays A–H.
- A dataset driver that loops sampled `(route_id, consist, control)`, runs
  `simulate_train_tensorized`, computes summaries (`E_trip`, `T_arr`, `F_max`), and appends the
  §I record to `index.parquet`.
- Batched/parallel execution (the current path is single-train NumPy + `solve_ivp`).
