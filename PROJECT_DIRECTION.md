# Project Direction — Freight LTD Surrogate for Fuel-Optimal Control

_Realignment reference. Written 2026-06. Supersedes the implicit scope of the current paper draft._

## The spine (one sentence)

> A fast, **train-agnostic** surrogate of longitudinal train dynamics that serves as the
> differentiable simulation backbone for **fuel-optimal control** of freight consists.

Everything in the project earns its place against this sentence. If a chapter or component
doesn't serve either *the surrogate* or *the control payoff it enables*, it is scope creep and
gets demoted to a section or to future work.

## Where the project actually stands

**Done and solid.** The reference physics simulator (`simulator2`, `simulator/physics`): the
stage 1–8 LTD model — nonlinear slack couplers with asymmetric buff/draft, per-vehicle
grade/curvature from a route profile, actuator lag, power-limited traction — plus a clean
**tensorized node/edge representation** (`H ∈ R^{N×d_node}`, `E ∈ R^{(N-1)×d_edge}`).
Paper Chapter 3 (Problem Formulation) is mature and well-written.

**Scaffolding only.** Chapters 4–8 are TODO skeletons (22 TODOs in the evaluation chapter
alone). There is no trained surrogate, no generated dataset, and no results. The simulator is
the only finished artifact.

**Diagnosis.** This is not "behind" — the data generator is finished and the modeling work
hasn't started. The disorientation came from **three theses layered on top of each other**:
(1) model-based RL for fuel-optimal control, (2) a physics-informed *transformer* surrogate for
forecasting + UQ, (3) a universal model for the whole family of freight trains. The draft commits
to (2); the stated goal is (1); the recurring instinct is (3). They are now unified under the spine
above.

## Locked decisions

| Decision | Choice | Consequence |
|---|---|---|
| **Scope / where to land** | Surrogate **+ control demo** | Surrogate is the core contribution; a working fuel-optimal control proof-of-concept (MPC or model-based RL) on a few real routes is the payoff chapter. |
| **Surrogate architecture** | **GNN / Neural-ODE** (not transformer) | Message passing over the consist chain graph; transformer demoted to an ablation/route-encoder. |
| **Route source** | **Real-world terrain data** | Grade/curvature profiles derived from real elevation + alignment data; defensible realism, adds a data-engineering step. |
| **Uncertainty quantification** | Demoted | Survives only as coupler-force **safety constraints** feeding the controller, not as a standalone forecasting contribution. |

## Surrogate design

The tensorized formulation already wrote the target equation (Chapter 3):

    d/dt vec(H_dyn) = F_n(H, E, U, R; t)

**Design: make a message-passing GNN the learned part of `F_n`, on top of the known physics RHS,
and integrate it as a Neural ODE (or a learned Δt step, "graph network simulator" style).**

- Nodes = vehicles, edges = couplers — exactly the existing `H`/`E` chain graph.
- **Physics-residual:** GNN learns the correction to the known force balance
  (`ŷ = y_phys + Δy_GNN`). Strong inductive bias, less to learn, better extrapolation.

Why this beats the transformer for this problem:

1. **Train-agnostic by construction.** Message passing handles any `N` with no padding/masking
   hacks → the *universality* claim becomes structural, not aspirational. This is the "build any
   train" property.
2. **Differentiable + batched.** Exactly what the control demo needs — MPC with analytic
   gradients, or a differentiable world-model for model-based RL. The surrogate stops being a
   forecasting toy and becomes the *enabler* of control.
3. **Right inductive bias.** A consist is a chain graph with local force propagation; message
   passing matches it. Attention (transformer) is still reasonable for the long, variable-length
   **route** field — keep it there as an encoder/ablation, not the headline.

Baselines to include (so the architecture claim is earned, not asserted): physics-only,
physics-residual MLP, and the demoted transformer.

## Revised chapter plan

1. **Introduction & motivation** — lead with fuel/control, not surrogate-for-its-own-sake.
2. **Background & related work** — LTD, surrogate/operator learning, GNN/Neural-ODE simulators,
   learned control (MPC + model-based RL). _(Existing Ch2 mostly reusable.)_
3. **Problem formulation + tensorized representation** — _existing Ch3, keep._
4. **Reference simulator** — _existing Ch3 stage 1–8, reframed as the data generator._
5. **Surrogate model** — representation, GNN-on-physics-RHS Neural ODE, baselines, training.
   _(Rewrite of Ch4 around the GNN; transformer → ablation.)_
6. **Surrogate evaluation** — accuracy, inference speedup, and the **generalization / universality
   experiments** (variable `N`, unseen routes). This chapter must carry real results.
7. **Fuel-optimal control using the surrogate** — the payoff: reduce fuel on a route subject to
   coupler-force safety limits (where the remaining UQ lives).
8. **Discussion, limitations, future work** — incl. the calibration caveat below.

## Route + data generation plan (real terrain)

This is the immediate next step. The two locked choices give it sharp requirements:

**Real-terrain sourcing pipeline.**
1. Select a set of real rail corridors (alignment geometry).
2. Pull elevation along the alignment (DEM — e.g. SRTM / public elevation services) and rail
   geometry/speed restrictions where available.
3. Resample to an **arc-length grade profile** `sin θ(s)`, curvature proxy `κ(s)`, and speed limit
   `v_max(s)` — the exact fields `RouteProfile` already consumes.
4. Validate profiles (smoothing, grade clamping, monotonic chainage) before use.

**Data volume strategy (faithful to the real-terrain choice).** Use a curated set of real routes
as the *backbone*; generate volume by perturbing real baselines (grade noise, adhesion, mass,
control plans) — i.e. domain randomization *on real profiles*, not invented synthetic routes.

**What to store.** Dense transitions, not just trip summaries — the Neural-ODE/GNS surrogate
learns dynamics. Per scenario: integration `dt`, per-vehicle route fields sampled along the run,
control commands `U`, static `H`/`E` parameter channels, and full `H_hist`/`E_hist`.

**Design the splits into the generator.** Universality only counts if shown. Tag every scenario
with `N`, route ID/type, grade severity, adhesion regime → enables clean OOD slices (train 10–80
cars, test 150; hold out steeper-grade routes; hold out specific corridors).

**Reserve control-evaluation routes.** Fix a handful of real held-out routes with a fuel/energy
objective for the control demo. These must never leak into surrogate training.

**Batch the simulator.** `simulator2` is NumPy + single-train. Dataset generation needs either
parallelized runs or a batched/torch RHS path — which you want anyway for the differentiable
surrogate. Plan this port.

## Control demo plan (payoff chapter)

- **Objective:** minimize trip energy/fuel `E_trip = ∫ P_trac dt` on a held-out real route.
- **Constraints:** speed limits `v_max(s)`; **coupler-force safety** `|F_j| ≤ F_limit` (or a
  risk-aware `P(|F_j| > F_limit) ≤ ε` — the surviving role of UQ).
- **Method options:** (a) MPC over the differentiable surrogate with analytic gradients
  (lower-risk, cleaner to present); (b) model-based RL using the surrogate as world-model
  (higher-risk, more "RL" framing). Recommend starting with MPC; it directly exploits
  differentiability and de-risks the results chapter.
- **Baseline to beat:** a constant-speed or simple driver-model control plan, on energy at equal
  trip time + zero safety violations.

## Critical path & risks

**Critical path:** real-terrain route pipeline → batched simulator + dataset → GNN/Neural-ODE
surrogate → surrogate results (Ch6) → MPC control demo (Ch7).

**Top risks.**
1. **No results = no thesis.** Everything past Ch3 is unbuilt. Protect the surrogate-results
   chapter first; control is the bonus that the scope choice commits to but can shrink to a
   proof-of-concept if time runs short.
2. **Calibration caveat (state it explicitly).** The simulator parameters are *illustrative, not
   calibrated to real stock*. So claims are speedup + fidelity-**to-the-simulator** and
   structural generalization — **not** fidelity-to-real-trains. Naming this boundary first
   pre-empts the obvious examiner attack.
3. **Real-terrain data engineering can balloon.** Time-box the sourcing pipeline; a small,
   well-documented corridor set beats an open-ended GIS project.
4. **Differentiable-rollout stability.** Neural-ODE training over long rollouts can be unstable;
   plan for short-horizon transition training + scheduled rollout length, and keep the
   physics-residual to anchor it.

## Open questions to resolve next

- Which specific real rail corridors / region(s)? (availability of DEM + alignment data)
- MPC vs. model-based RL for the control chapter (recommendation: MPC first).
- Target consist-size and route ranges for the train/OOD split.
- Port simulator to torch now, or parallelize NumPy for data-gen and port later?
