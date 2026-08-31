# Spec: Batched, Differentiable Torch RHS for `simulator2`

_Handoff document for Claude Code. Written 2026-08-18. Companion to
`PROJECT_DIRECTION.md`, `DATA_SCHEMA.md`, and `NEXT_STEPS.md`._

---

## 1. The task in one sentence

Port the `N`-vehicle longitudinal train dynamics right-hand side from NumPy to
batched, GPU-resident, differentiable PyTorch, with a fixed-step RK4 rollout, so
that it reproduces the existing NumPy simulator to tolerance and runs orders of
magnitude faster.

**In scope:** `simulator2/torch_rhs.py` (batched RHS + fixed-step rollout), the
tests in `simulator2/tests/test_torch_rhs.py` going green, and a benchmark
report on the target GPU.

**Out of scope — do not build these:** the dataset writer, the sampling driver,
`index.parquet` / `norm_stats.json` / `splits.json`, the GNN, any training loop.
Those come after review. Building them here would outrun their acceptance
criteria.

---

## 2. Why: the measurements that justify this

All measured on the current NumPy path. Do not re-derive these; they are the
reason the design below is the way it is.

**Per-evaluation cost is a Python loop, linear in `N`:**

| `N` | µs per RHS evaluation | µs per vehicle |
|---|---|---|
| 11 | 446 | 40.6 |
| 60 | 2,384 | 39.7 |
| 130 | 5,142 | 39.6 |

Flat ~40 µs per vehicle per evaluation. That is interpreter overhead, not
arithmetic — the arithmetic per vehicle is a few dozen flops.

**Evaluation count is set by the coupler deadband, and it explodes.** For 120 s
of simulated time at `N = 11` with adaptive RK45 at `rtol=1e-6`:

| Regime | RHS evaluations | Effective step |
|---|---|---|
| cruise | 10,658 | 0.068 s |
| coast | 13,880 | 0.052 s |
| dynamic_brake_descent | 36,734 | 0.020 s |
| **throttle_modulation** | **650,336** | **0.0011 s** |

A 61× spread across regimes, driven entirely by step rejection at slack
transitions. Worse, the expensive regimes are the ones that matter most: slack
action is what generates the coupler forces the safety constraint is about.

**Standstill under brake is pathological for adaptive stepping.** Generating the
test fixtures (only `N = 6`, only 40 s of simulated time) needed 656,714
evaluations and 196 s of wall-clock for the brake-to-stop case, against ~2,700
for cruise. At `v = 0` the `tanh(v / v_brake_eps)` sign blend has slope
`1/v_brake_eps`, so error control chases vanishing oscillation about the
equilibrium. LSODA did not help — this is a discontinuity, not stiffness. A
held brake at rest is a completely ordinary operating state and the dataset will
be full of it.

**Consequence at scale.** In the 66-run sweep (`scripts/bench_scenarios.py`),
every one of the 22 runs at `N = 130` exceeded a 30 s budget for 120 s of
simulated time. The `ood_size` split — and with it the universality claim the
thesis rests on — is currently ungeneratable.

**The system is not actually stiff.** Numerical Jacobian spectral radius, from
the existing RHS:

| `N` | ‖λ‖max (rad/s) | RK4 stability limit |
|---|---|---|
| 11 | 17.6 | 0.159 s |
| 60 | 18.3 | 0.153 s |
| 130 | 20.8 | 0.135 s |

Barely grows with `N`. So a **fixed step of 0.02 s sits 7× inside the stability
limit** while costing a constant 24,000 evaluations per 120 s run — 27× less
work than adaptive stepping's worst case, and immune to the regime spread.
That, plus vectorising away the 40 µs/vehicle interpreter cost, is where the
speedup comes from.

---

## 3. Environment (WSL2 + RTX 4090)

Target: Windows 11 workstation, RTX 4090 (24 GB, Ada, sm_89), via WSL2.

- Install the NVIDIA driver **on Windows only**. Do not install a driver inside
  WSL; CUDA is passed through.
- Keep the repo on the **WSL filesystem** (e.g. `~/projects/thesis/th`), not
  under `/mnt/c`. Cross-filesystem I/O is slow enough to distort benchmarks.
- Install a CUDA 12.x PyTorch wheel from the selector at
  <https://pytorch.org/get-started/locally/>.
- Verify before starting:

  ```bash
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  # expect: ... True NVIDIA GeForce RTX 4090
  ```

- Existing suite must stay green throughout: `python -m pytest simulator2/tests/ -q`
  (135 tests before this work starts).

---

## 4. Locked design decisions

Do not revisit these without saying so explicitly and giving a reason.

1. **Fixed-step RK4**, step configurable, default `dt = 0.02 s`. Justified by
   §2. No adaptive stepping — bounded, predictable, regime-independent cost is
   the point.
2. **The NumPy path is the oracle and does not change.** `simulator2/rhs.py`,
   `simulator2/simulate.py`, `simulator2/randomize.py` and everything under
   `simulator2/tests/` except the new torch tests must be left alone. If you
   believe one of them is wrong, report it — do not edit it.
3. **New module: `simulator2/torch_rhs.py`.** Self-contained. It may import
   `constants.py` and `state_schema.py` for shared constants and channel
   indices; it must not import `rhs.py` or `simulate.py`.
4. **State layout mirrors NumPy exactly:** `y` is `(B, 4N)`, flat, ordered
   `[x(N), v(N), z_brk(N), z_trac(N)]`. Same order as the fixtures.
5. **Ragged `N` via padding plus a boolean `node_mask`**, not disjoint-union
   graph batching. (Disjoint union is for the GNN's dataloader later; for the
   integrator, mask-and-pad is simpler and the batch can be bucketed by `N`.)
   Padded vehicles must contribute exactly zero force and zero acceleration.
6. **Differentiable end to end.** No in-place mutation of anything on the
   autograd graph, no `.item()`/`.numpy()` inside the RHS, no data-dependent
   Python control flow over batch elements. `torch.where`, not `if`.
7. **`brake_opposes_motion` semantics are preserved exactly**, including the
   `tanh(v / v_brake_eps)` blend and the ability to disable it. See
   `simulator2/rhs.py` and `test_brake_direction.py`.
8. **The coupler deadband keeps its exact piecewise form.** Softening it would
   help both solver behaviour and gradient conditioning — and it would change
   the physics documented in Chapter 4. That is a thesis decision, not a
   refactor decision. If the fixed-step error at deadband crossings turns out to
   be unacceptable, **report it and stop**; do not soften the law to pass a test.

---

## 5. Decisions to make from data — measure, report, do not guess

For each of these, run the experiment, put the numbers in the final report, and
pick the option the data supports.

- **Default dtype.** A 4090 runs float64 at roughly 1/64 of its float32
  throughput, so float32 is strongly preferred. It is only acceptable if
  accuracy holds: `test_float32_error_is_acceptable` measures the divergence.
  If float32 fails, report the number and default to float64 rather than
  loosening the tolerance.
- **Step size.** Sweep `dt ∈ {0.05, 0.02, 0.01, 0.005}` against the fixtures and
  report accuracy vs cost. `0.02` is the default; change it only with evidence.
- **Kernel launch overhead.** This workload is elementwise and gather-bound, not
  FLOP-bound: ~24,000 steps × 4 RK4 stages × O(10) kernels is on the order of a
  million launches, at roughly 5 µs each. **Expect launch overhead to dominate
  and plan for it** — `torch.compile(mode="reduce-overhead")` or CUDA graphs is
  likely necessary rather than optional. Report timings with and without, and
  confirm compilation does not change results.
- **Batch size.** Memory is not the constraint: `(B=64, N=130)` state is ~133 KB
  per step, and storing 481 output samples is ~64 MB. Push `B` until throughput
  stops improving and report the curve.

---

## 6. API contract

`simulator2/tests/test_torch_rhs.py` imports exactly this. Match it.

```python
from simulator2.torch_rhs import TorchScenarioBatch, torch_rhs, rollout_rk4

batch = TorchScenarioBatch.from_fixtures(paths, device="cuda", dtype=torch.float32)
batch = TorchScenarioBatch.synthetic(batch_size=64, n_vehicles=130,
                                     device="cuda", dtype=torch.float32)

dy = torch_rhs(t: float, y: Tensor, batch, *, brake_opposes_motion: bool | None = None) -> Tensor
t_out, y_out = rollout_rk4(batch, y0: Tensor, t_grid: Tensor, dt: float)
```

- `y`: `(B, 4N)` → `dy`: `(B, 4N)`.
- `y_out`: `(B, T, 4N)`, sampled at `t_grid` (which is generally coarser than
  `dt`; interpolate or subsample from the integration steps).
- `batch` must expose at least `y0`, `node_mask` `(B, N)`, `node_static`
  `(B, N, 7)`, `edge_static` `(B, N-1, 6)`, and the route and command fields.
- `brake_opposes_motion=None` means "use the per-scenario value carried in the
  batch"; an explicit bool overrides it.

Commands are **materialized in time**, per `DATA_SCHEMA.md` §E: the batch
carries `u_trac (B, T, N)`, `u_brk (B, T, N)` on a time grid, and the RHS
interpolates at `t`. Do not pass Python callables into the RHS — that would
serialise the batch.

Route sampling happens **inside** the RHS at each vehicle's current position, per
`DATA_SCHEMA.md` §2 — batched piecewise-linear interpolation
(`torch.searchsorted` + gather), never a Python loop.

---

## 7. Acceptance

`simulator2/tests/test_torch_rhs.py` is the contract. It is committed and
currently red. **Make it green without editing it.**

If a tolerance in that file is genuinely wrong, say so in your report and change
it in a separate, clearly-labelled commit with the reasoning. Silently loosening
a tolerance to pass is the specific failure mode this handoff is designed to
prevent.

What it checks, and why each one is there:

| Test | Guards against |
|---|---|
| `test_rhs_matches_numpy_reference` | Any physics divergence, checked at 40 points along each reference trajectory in float64 against an **independently reimplemented** RHS (so a shared bug can't make both sides agree) |
| `test_batched_rhs_equals_individual_calls` | Batching coupling scenarios together |
| `test_padding_does_not_leak` | Padded vehicles contributing force |
| `test_rollout_matches_reference_trajectory` | Fixed-step drift (0.05 m/s velocity, 1.0 m position) |
| `test_peak_coupler_force_preserved` | The safety-critical quantity, 5% — checked separately so it can't hide inside a trajectory norm |
| `test_deadband_crossings_tracked` | Missing slack transitions (≥98% engagement agreement on the adversarial slack-cycling fixture) |
| `test_standstill_is_held_not_reversed` | Regression of the brake-direction bug |
| `test_brake_flag_disabled_reproduces_historical_behaviour` | Losing the ability to reproduce pre-fix results |
| `test_rhs_gradcheck`, `test_gradients_flow_through_rollout`, `test_no_inplace_breaks_autograd` | Silently non-differentiable code — the reason this port exists |
| `test_cuda_matches_cpu`, `test_determinism` | Device-dependent or nondeterministic results |
| `test_float32_error_is_acceptable` | Precision loss from the fp32 default |
| `test_throughput_gate_on_gpu` | The whole point: 64 × `N=130` × 120 s in under 20 s |

The throughput gate is deliberately conservative. The NumPy baseline could not
finish **one** such run in 30 s; the gate asks for 64 in 20 s (~0.3 s each,
about 100×). If the hardware does much better, say so in the report.

Reference fixtures: `simulator2/tests/fixtures/torch_reference/*.npz`, 15 cases
covering all 11 driving regimes plus slack-cycling, brake-to-standstill,
curvature-enabled, and a disabled-flag case. Each is self-describing;
regenerate with `python simulator2/tests/fixtures/generate_torch_fixtures.py`
(slow — ~4 minutes — and unnecessary unless the physics changes).

---

## 8. Deliverables

1. `simulator2/torch_rhs.py`.
2. `simulator2/tests/test_torch_rhs.py` green on the 4090 (and the pre-existing
   135 tests still green).
3. A short `TORCH_PORT_REPORT.md` with:
   - throughput vs `B` and vs `N`, on the 4090, against the NumPy baseline;
   - the `dt` sweep: accuracy vs cost;
   - float32 vs float64 divergence, and which you defaulted to and why;
   - timings with and without `torch.compile` / CUDA graphs;
   - anything you found that contradicts this spec.

Work on a branch. Do not modify anything listed under "out of scope" in §1 or
"locked" in §4.

---

## 9. Known traps

- `scripts/bench_scenarios.py` uses `signal.SIGALRM`, which is Unix-only. Fine
  under WSL2; it would need a different timeout mechanism on native Windows.
- The fixtures were generated at `N = 6`, 40 s, because generating larger
  references with the NumPy path is impractical (§2). They validate
  *correctness*; the throughput gate uses `TorchScenarioBatch.synthetic` for
  *performance* at realistic size. Do not conflate the two.
- Five fixtures (`regime_emergency`, `adversarial_brake_to_stop`,
  `brake_flag_disabled`, `regime_brake_release`, `regime_startup`) have
  references generated at the simulator's default tolerance rather than a
  tightened one, because their references are the pathological standstill cases.
  The test file already scales their tolerances by 4×; that is intentional.
- `v_eps = 1.0 m/s` in the power cap and `v_brake_eps = 0.1 m/s` in the brake
  blend are different constants with different jobs. Don't merge them.
- Davis resistance uses a hard `sign(v)` with a `1e-9` threshold. That is a
  genuine discontinuity at `v = 0` in the existing model — reproduce it exactly
  rather than smoothing it, and note in your report that it is a candidate for
  the same `tanh` treatment as the brake if gradient quality suffers.
