# Next Steps — Data Generation → GNN Surrogate

_Companion to `PROJECT_DIRECTION.md` and `DATA_SCHEMA.md`. Written 2026-08-17 after
building the control-profile library and scenario randomizer, and benchmarking the
reference simulator on realistic consists._

---

## The headline finding: the NumPy simulator cannot build this dataset

A sweep of 66 runs (3 consists × 2 routes × 11 driving regimes, **120 s of simulated time
each**, 30 s wall-clock budget per run) gave:

| Consist size `N` | Runs completed | Median wall-clock | Notes |
|---|---|---|---|
| 11  | 16 / 22 | **4.0 s** | 3 regimes exceeded 30 s |
| 60  | 4 / 22  | **28.1 s** | 18 of 22 exceeded 30 s |
| 130 | 0 / 22  | **> 30 s** | **every single run timed out** |

**At `N = 130` — the range the `ood_size` split and the entire universality claim depend
on — not one 120-second run finished inside 30 seconds.** And 120 s is short: it covers
1–2 km of a 30 km route. `DATA_SCHEMA.md` wants runs long enough to traverse varied
terrain, which is 300–600 s, i.e. 3–5× this cost again.

Rough arithmetic for a 10,000-scenario dataset at 300 s per run, optimistically averaging
60 s of wall-clock each: **~170 CPU-hours**, and that is single-threaded with the large
consists effectively excluded. It is not a "run it overnight" job in its current form.

Two things drive the cost, and both are fixable in the same piece of work:

1. `train_rhs_tensorized` loops over vehicles **in Python**, so each RHS evaluation is
   O(N) interpreted operations — at N=130 that is 130 Python iterations per solver step,
   and there are tens of thousands of steps.
2. The coupler slack deadband is a derivative discontinuity. `solve_ivp` shrinks its step
   whenever a coupler crosses in or out of slack, and on a long consist *some* coupler is
   crossing almost always. Per-regime medians make this visible: the cheap regimes are
   the smooth ones (`cruise` 4.1 s, `notch_up` 4.1 s, `power_to_coast` 3.9 s) and the
   expensive ones are exactly the slack-action regimes — `dynamic_brake_descent` **27.3 s**
   and `throttle_modulation` **15.8 s**. Those are the regimes that generate the coupler
   forces the safety constraint cares about, so they cannot simply be down-weighted.

### Revised ordering

I previously suggested the torch port as step 5, after a pilot dataset. **The measurement
says it is the blocker, not a later optimization.** Revised critical path:

1. ~~Control-profile regime library~~ ✅ done (this pass)
2. ~~Scenario randomization~~ ✅ done (this pass)
3. **Batched torch N-vehicle RHS** ← now the gate on everything downstream.
   Specified in `TORCH_PORT_SPEC.md`, with the acceptance contract already
   committed and red at `simulator2/tests/test_torch_rhs.py` and reference
   fixtures under `simulator2/tests/fixtures/torch_reference/`. Handed to
   Claude Code CLI on the 4090 workstation.
4. `write_scenario_npz` + dataset driver + `index.parquet` / `norm_stats.json` / `splits.json`
5. Pilot dataset (~500 scenarios) to debug graph batching and normalization
6. Full dataset
7. GNN / Neural-ODE surrogate

The port was always required anyway — the physics-residual design (`ŷ = y_phys + Δy_GNN`)
has to evaluate `F_phys` inside autograd on every training step, so a differentiable
batched RHS is a hard dependency of the *model*, not just of data generation. The
benchmark just means it cannot be deferred.

Note `simulator/physics/core/railphysics/` is already torch and batched, but it is
**single-locomotive** — no couplers, no consist. It is a useful reference for the Davis /
grade / integrator patterns and for the device-parity test style, but the N-vehicle chain
with the slack deadband is new work.

When porting, the deadband is worth attention beyond speed: a `tanh`-softened or
explicitly event-detected coupler transition would cut solver chatter *and* give the
Neural-ODE a better-conditioned gradient. That is the same change, done once.

---

## What was built this pass

### `simulator2/driving_regimes.py` — regime-based control profiles

`control_profile.generate_random_profile` draws every knot uniformly, which spends most of
its probability mass on command combinations no operator would produce, and never
reproduces the structured manoeuvres the MPC controller will generate. This module samples
**11 manoeuvre archetypes** and randomizes their parameters instead:

`startup`, `notch_up`, `cruise`, `throttle_modulation`, `power_to_coast`, `coast`,
`coast_to_brake`, `dynamic_brake_descent`, `brake_release`, `stretch_brake`, `emergency`

Weighted so steady-state and modulation dominate; `emergency` is rare but present because
it produces the extreme coupler forces the safety constraint is about.

**Traction/brake exclusivity is enforced, with two deliberate exceptions.** `stretch_brake`
holds light brake against light traction throughout — a real handling technique, and the
coupled-command case the surrogate needs to see. `emergency` overlaps for a few seconds
because the brake pipe drops while tractive effort is still decaying. Everything else is
clean, guaranteed by a post-hoc gate (`_gate_traction`) rather than by construction alone.
Verified: **0 violations across 5,500 sampled profiles.**

Three bugs were found and fixed while building this, all of which would have quietly
poisoned the dataset:

- **Additive DPU jitter lifted commanded zeros above the off-threshold**, silently
  reintroducing traction/brake overlap on every remote unit in the traction-free regimes
  (coast, dynamic brake, post-event emergency). Now multiplicative, so zero stays zero.
- **Unbounded time jitter reordered knots**, which inverts a manoeuvre — one emergency
  profile became a power *restoration* under full braking. Jitter is now bounded by 0.4×
  the local knot spacing, so ordering cannot change.
- **`dpu_mode="independent"` resampled a fresh shape per remote unit**, letting a DPU pull
  while the head end braked. It now perturbs the head-end curve instead: remote units run
  the same manoeuvre, just not in lockstep.

### `simulator2/randomize.py` — scenario randomization

`make_scenario_from_consist` fixes everything that should vary: rest start, chainage ~0,
120 s, zeroed actuators. `make_randomized_scenario` samples:

- **start chainage** anywhere with room to run (was: always ~0)
- **initial speed**, capped at 90% of the route limit at that point (was: always 0)
- **initial slack state** — neutral / stretched / bunched / random, strictly inside the
  coupler deadband so no force is injected at `t = 0`
- **initial actuator states**, settled at the profile's `t = 0` command, so a run opens
  mid-manoeuvre instead of with a spurious ramp
- **adhesion regime** — dry/wet/low as a multiplier on powered vehicles' `F_trac_max_N`
- **duration**, clipped so the consist cannot run off the end of the route

It returns the flat `DATA_SCHEMA.md` §I metadata record alongside the scenario, so the
dataset driver can write `index.parquet` without recomputing anything.

Two degenerate cases surfaced in testing and are now handled:

- **Standing starts rolled backwards.** A consist at `v = 0` with no brake applied rolls
  away down any real grade before traction builds through `tau_trac`; one test run
  finished 130 m *behind* where it started. Real standing trains hold with the air brake.
  Rather than model that, standing starts are now rejection-sampled onto track flatter
  than 0.5%.
- **Braking from a crawl** stopped the train in the first few seconds and left the rest of
  the window near-stationary. Profiles that brake within the first 20 s now get a minimum
  starting speed.

### `simulator2/rhs.py` — brake direction bug (opt-in fix)

Brake force was applied as a pure `-x` force regardless of velocity:

```python
dv[i] = (f_tr - f_br - r - g - ccur + fi_in - fi_out) / m
```

At `v = 0` with brake applied this gives `dv < 0`, so **the consist accelerates backwards
without bound.** Measured on a 9-car test consist: `dv = -0.146 m/s²` at standstill under
brake. This never surfaced because every existing scenario started from rest and only ever
applied traction — but it makes any run that brakes to a stop unusable, and randomized
starts make those common.

Fixed behind `brake_opposes_motion`, blended with `tanh(v / v_brake_eps)` rather than a
hard `sign(v)` so the RHS stays smooth for the solver and for autograd later. **The flag
now defaults to `True`** in both `train_rhs_extended` and `train_rhs_tensorized`, in the
notebook, and in `ExtendedTrainScenario` — code, notebook, and thesis all describe the same
model. Set it `False` only to reproduce results recorded before 2026-08-17.

At `v = 12 m/s` the flag is a no-op to machine precision (`tanh(120) = 1`), so only runs
approaching standstill differ at all.

### Chapter 4 figures — re-run and resolved

All nine stage figures were regenerated from the notebook and compared pixel-by-pixel.

1. **The notebook reproduces every committed figure bit-for-bit** (`maxdiff = 0` on all 9),
   confirming it is the true source and the re-run environment is faithful.
2. **The fix changes exactly one figure: Stage 7.** Stages 1–6 and the Stage 5 heatmap are
   traction-only, so the brake term never enters. Stage 4b does apply a brake pulse, but at
   speed — `max |Δstate| = 0.0`.
3. **Stage 7 was showing the bug in print.** Lead speed: brake commanded at t = 70 s, the
   trace passes straight through zero and continues to **−99 km/h** by t = 120 s — a
   consist reversing at 99 km/h under full braking. Corrected, it decelerates to zero at
   ≈88 s and holds. The lower panel (actuator lag states `z_brk`, `z_trac`) is unchanged,
   since those are command-lag states independent of velocity.

Regenerated `notebooks/notebook_images/ltd_stage07_...png`; the other eight are untouched
and remain byte-identical. The notebook itself is patched at all three brake sites (Stage
4b, Stage 7, and the Stage 9 refactor cell) and re-executed so its stored outputs match.

`paper/chapters/04_reference_simulator.tex` §Stage 7 now states the convention explicitly
as `eq:ltd_brake_sign`, with a note on why `tanh` is preferred to `sgn`.

`simulator2/tests/test_brake_direction.py` locks the behaviour in: brake never accelerates
a stopped consist, pushes forward when rolling backwards, is a no-op at operating speed,
the two RHS variants agree under both flag settings, and the blend is smooth through zero
(a hard `sign(v)` would fail the last one).

### Tests — `test_driving_regimes.py`, `test_randomize.py`, `test_brake_direction.py`

59 new tests, **135 passing total** (including the pre-existing suite, so
`test_tensorized_matches_classical` still holds with the fix on by default). They cover exclusivity limits per regime, knot ordering,
determinism per seed, that braking regimes brake and traction regimes pull, that slack
pre-load injects no force, that actuators start settled, that adhesion scales only powered
vehicles and never mutates the caller's list, that standing starts land on flat track, and
that the §I metadata keys the split design needs are all present.

### `scripts/bench_scenarios.py`

The sweep that produced the table above. Each run is bounded by `SIGALRM`, which is not
incidental: **the dataset driver must impose a per-scenario wall-clock budget**, or one
pathological consist/regime pairing will hang a 10,000-scenario build with no diagnostic.

---

## Still open

1. **Route corridor set.** 5 corridors, one region, 22–49 km, all pinned at the
   `grade_clip=0.04` ceiling — `route_line2` sits *at* the clamp for 22.7% of its length
   (rms grade 2.46% vs ~0.9% for the others), which is DEM noise surviving into the grade
   field rather than real terrain. 4% is also very steep for freight (ruling grades are
   usually ≤2.2%). As it stands `ood_corridor` and `ood_grade` are not buildable, and the
   train manifest confirms the gap: only `train` and `ood_size` exist — no `val`, no
   `test_id`, no `control_eval`. Agreed plan: fix the smoothing so the clamp stops binding,
   then pull 15–25 corridors across varied terrain.
   *Note: the route pipeline needs OSM + DEM network access, which the device bridge does
   not have. Run it locally, or stage the `routegen` package into the cloud container where
   network is available.*
2. **Splits.** `splits.json` needs `val`, `test_id`, `ood_grade`, `ood_corridor`, and the
   reserved `control_eval` routes that must never touch surrogate training.
3. **`k_curv_scale` is 0.0 by default** in the randomizer because the curvature proxy's
   magnitude is not calibrated — the route `kappa` field is generated but unused. Worth a
   deliberate decision before the dataset is built, not after.
4. ~~Chapter 4 validation figures~~ ✅ resolved — see above. Only Stage 7 needed
   regenerating; it has been.
