# simulator2 — tensor-state longitudinal train dynamics

This package ports the **extended** longitudinal train dynamics (LTD) model from the guided lab notebook `notebooks/ltd_buildup_learning_notebook.ipynb` into a normal Python layout. The physics and naming follow that notebook: `VehicleParameters`, `CouplerParameters`, `RouteProfile`, Davis/grade/curvature forces, slack/asymmetric couplers, and the **Stage 7** extended state with first-order traction and brake buildup plus a power-limited traction force.

## What was ported

- Constants: gravity, default actuator time constants (`tau_brk`, `tau_trac`), power cap `P_MAX_W`, `V_EPS`, default masses/Davis/coupler numbers matching the notebook cells.
- Dataclasses: `VehicleParameters`, `CouplerParameters`, `TrainSimulationConfig`, `RouteProfile`.
- Forces: `davis_resistance_longitudinal`, `grade_force`, `curvature_force_longitudinal`, `coupler_force_slack_asymmetric`, and the shared `traction_ramp` helper used in demos.
- RHS: `train_rhs_n` (2N state), `train_rhs_extended` (4N state), and `train_rhs_tensorized` (same equations as extended, expressed via tensors).
- Integration helpers: `simulate_train` (notebook-style dict output), `simulate_train_tensorized` (tensor histories).
- Scenarios: `make_simple_train`, `make_simple_train_scenario`.

## What “tensorized” means here

- **Node tensor `H`**: shape `[N, d_node]` with named channels (`NodeChannel`): positions, speeds, actuator states `z_brk`, `z_trac`, plus static per-vehicle parameters (mass, Davis coefficients, traction flags, force limits).
- **Edge tensor `E`**: shape `[N-1, d_edge]` with `delta`, `delta_dot`, algebraic `F_cpl`, and static coupler parameters (`EdgeChannel`).
- **`solve_ivp` still integrates a flat vector** `y` of length `4N` (`x`, `v`, `z_brk`, `z_trac`). Static parameters live in `H`/`E` but are **not** part of `y`.
- Each call to the tensorized RHS **unpacks** `y` → `H`, **recomputes** `E` (including coupler forces), evaluates the same dynamics as `train_rhs_extended`, and **packs** the time derivatives back into a flat vector.

## Classical vs tensorized state

| Representation | Content |
|----------------|---------|
| Flat ODE `y` | Only dynamic variables: `x`, `v`, `z_brk`, `z_trac` (length `4N`). |
| `TensorTrainState` | Full `H` and `E`: dynamics plus embedded static parameters; `delta`, `F_cpl` etc. are reconstructed each RHS evaluation. |

## Current limitations

- Still uses `scipy.integrate.solve_ivp` (no custom integrator).
- Edge kinematics and `F_cpl` are **algebraic**, not separate integrated states.
- NumPy only — **not** PyTorch and **not** differentiable.
- No batching over multiple trains yet; see `make_node_mask` / `make_edge_mask` in `tensor_state.py` for padding hooks.

## Likely next step

Batch or dataset generation over many scenarios (possibly with fixed `N_max` and masks) for surrogate / transformer-style models, then optional export to torch tensors.

## Quick usage

```python
from simulator2.scenarios import make_simple_train_scenario
from simulator2.simulate import simulate_train_tensorized

scenario = make_simple_train_scenario(n_cars=20)
res = simulate_train_tensorized(scenario)
print(res.H_hist.shape)  # (T, N, d_node)
print(res.E_hist.shape)  # (T, N-1, d_edge)
```

Demo (from repository root):

```bash
python simulator2/examples/demo_tensorized_sim.py
```

Tests (from repository root):

```bash
python -m pytest simulator2/tests -v
```

## Assumptions when porting from the notebook

- **Self-contained defaults**: Global notebook names like `M_LOCO_KG`, `SLACK_HALF`, etc. live in `constants.py` and `scenarios.py` with the same numeric values as in the notebook cells we mirrored.
- **`TrainSimulationConfig.route`**: Typed as `RouteProfile`; import `RouteProfile` from `simulator2.route` when constructing configs.
- **`make_simple_train_scenario`**: Uses the Stage 7-style traction ramp on the lead only (`F_max=200_000`, `t0=5`, `t1=25`) and **zero** brake command by default, so it is a stable demo; the notebook’s Stage 7 cell also used a brake step at `T_BRK` — change `u_brk_cmd` on the returned `ExtendedTrainScenario` if you need that behavior.
- **Single-track API**: `simulate_train_tensorized` expects an `ExtendedTrainScenario` (see `io_types.py`) bundling vehicles, couplers, route, `y0`, commands, and integrator settings.
