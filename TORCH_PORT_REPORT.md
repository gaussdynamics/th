# Torch Port — Results

_Deliverable 3 of `TORCH_PORT_SPEC.md`. Written 2026-09-03 on the RTX 4090
workstation. Companion to `NEXT_STEPS.md`. What was built on top of this port
is in `DATASET_BUILD_REPORT.md`._

---

## Status

`simulator2/torch_rhs.py` is built and the acceptance contract is green.

```
python -m pytest simulator2/tests/ -q
195 passed in 231.00s
```

That is the 135 pre-existing tests, untouched and still passing, plus all 60
tests in `simulator2/tests/test_torch_rhs.py`. The throughput gate passes at
**4.18 s against its 20 s budget**.

Nothing in `simulator2/rhs.py`, `simulate.py`, `randomize.py` or any
pre-existing test was modified. Two defects in `test_torch_rhs.py` itself were
fixed in a separate labelled commit — see [Test-contract changes](#test-contract-changes).

---

## Environment — the spec's WSL2 assumption did not hold

Spec §3 assumes WSL2. **WSL is not installed on this machine** and the port runs
on native Windows 11 against Python 3.11.6. Consequences:

- `torch` was **CPU-only** (`2.11.0+cpu`) on arrival. Replaced with
  `2.11.0+cu130` — same version, CUDA runtime added, so the green 135-test
  baseline had no version drift underneath it.
- `torch.compile` needed two fixes, both per-shell and both automated in
  `scripts/gpu_env.ps1`: MSVC's `cl.exe` is not on the global PATH, and
  triton's `driver.c` calls `alloca()` while compiling with `/std:c11`, which
  makes MSVC suppress the `alloca → _alloca` macro in `<malloc.h>` and fail at
  link with `LNK2019`. Worked around with `CL=/Dalloca=_alloca`.
- **`scripts/bench_scenarios.py` uses `signal.SIGALRM`, which is Unix-only.**
  Spec §9 anticipated this. *(Follow-up: no Windows replacement was needed. The
  per-scenario wall-clock budget `NEXT_STEPS.md` called mandatory existed
  because adaptive stepping's cost varied 61× across regimes and a pathological
  pairing could hang a build. Fixed-step RK4 removes the failure mode at the
  source — cost is `ceil(duration/dt)` steps regardless of regime — so the
  dataset driver imposes no timeout and needs none. See
  `DATASET_BUILD_REPORT.md`.)*

The final port does **not** depend on the compiler toolchain at run time (see
[CUDA graphs](#torchcompile-vs-cuda-graphs)), so `gpu_env.ps1` is only needed if
you want to experiment with `torch.compile` again.

---

## Throughput

All on the RTX 4090 (sm_89, 24 GB, 128 SMs), 120 s of simulated time,
`dt = 0.02`, CUDA-graph path.

### vs batch size (N = 130, float32)

| `B` | wall clock | ms / scenario | scenarios / s |
|---|---|---|---|
| 1 | 3.28 s | 3281.9 | 0.30 |
| 4 | 3.63 s | 907.1 | 1.10 |
| 16 | 3.65 s | 227.9 | 4.39 |
| 32 | 3.66 s | 114.5 | 8.73 |
| 64 | 3.71 s | 58.0 | 17.23 |
| 128 | 3.79 s | 29.6 | 33.77 |
| 256 | **4.01 s** | **15.7** | **63.77** |

### vs consist size (B = 64, float32)

| `N` | wall clock | ms / scenario |
|---|---|---|
| 11 | 3.72 s | 58.1 |
| 30 | 3.80 s | 59.4 |
| 60 | 3.77 s | 58.9 |
| 130 | 3.71 s | 58.0 |
| 200 | 3.76 s | 58.7 |

**Wall clock is flat in both `B` and `N`.** Going from 1 to 256 scenarios costs
22 % more time; going from 11 to 200 vehicles costs nothing measurable. The
rollout is bound by the number of graph replays — 6,240 of them, one per
integration step — and not by arithmetic. The GPU is nowhere near saturated even
at `B=256, N=200`.

Two things follow. Batch throughput is nearly free, so **push `B` as high as
memory allows** for dataset generation; 256 was the largest measured and had not
yet turned over. And `ood_size` at `N = 130` costs the same per scenario as
`N = 11`, which removes the specific obstacle that made the split ungeneratable.

### against the NumPy baseline

Measured on this machine, `train_rhs_tensorized`, single-threaded:

| `N` | µs / RHS eval | projected s per 120 s scenario |
|---|---|---|
| 11 | 296.5 | 7.1 |
| 60 | 1,999.5 | 48.0 |
| 130 | 3,715.3 | **89.2** |

(Projection is 6,000 steps × 4 RK4 stages. The adaptive solver the NumPy path
actually uses needs *more* evaluations than that in the slack-action regimes —
up to 650,336 per 120 s at `N=11` per spec §2 — so 89.2 s is a floor, not an
estimate.)

Against 89.2 s per scenario:

- **1,538× at `B = 64`** (58.0 ms/scenario)
- **5,682× at `B = 256`** (15.7 ms/scenario)

The gate asked for ~100×. Reporting the excess as spec §7 requests.

**What this does to the dataset.** `NEXT_STEPS.md` estimated ~170 CPU-hours for
10,000 scenarios and noted large consists were effectively excluded. At
`B = 256` in float64 (21.4 ms/scenario at 120 s, so ~53 ms at 300 s), 10,000
scenarios of 300 s is **about 9 minutes of GPU time**. Data generation is no
longer the bottleneck; it is now a rounding error against the work of writing
and validating the dataset driver.

---

## `torch.compile` vs CUDA graphs

Spec §5 predicted launch overhead would dominate. It does, emphatically.
Profiling the eager path at `B=64, N=130`:

```
Self CPU time total: 581.856 ms
Self CUDA time total:  49.792 ms      <- GPU idle ~92 % of the time
cudaLaunchKernel: 33,050 calls @ 8.5 us   (661 launches per RK4 step)
```

| configuration | `B=64, N=130`, 120 s |
|---|---|
| eager, before kernel sharing | 61.5 s ❌ |
| eager, after kernel sharing | 38.2 s ❌ |
| **CUDA graph** | **3.79 s** ✅ |

CUDA graph capture is worth **10.1×** on top of a 1.6× win from removing
redundant kernels (sharing one `searchsorted` between fields on a common grid,
and skipping the curvature interpolation when `k_curv_scale == 0`, which is what
the NumPy path already does).

**`torch.compile(mode="reduce-overhead")` — the route the spec suggests — does
not work here.** It compiles and produces correct results on small cases, but in
a rollout it fails inside its own cudagraph-trees bookkeeping:

```
File "torch/_inductor/cudagraph_trees.py", line 2608, in ...
    assert len(node.tensor_weakrefs) == len(node.stack_traces)
AssertionError
```

The trigger is the caller retaining one output tensor per sampled step, which a
rollout does by construction — 481 live tensors from the graph's memory pool.

Capturing the RK4 step into a `torch.cuda.CUDAGraph` directly avoids the issue
and is better in two further ways: state and time live in static buffers the
captured graph updates itself, so a replay crosses no host boundary at all; and
there is no dependency on a working inductor toolchain at run time, which on this
native-Windows machine is a real liability.

The graph path is used only for **inference rollouts on CUDA**. Autograd and CPU
rollouts take the plain eager path — capture would freeze the backward graph into
static buffers, and the differentiable rollout is not the throughput-critical
one. `FAST_ROLLOUT = "never"` forces eager.

---

## Step size

`dt` swept against all 15 fixtures in float64. The fixtures split into two
groups and they behave differently, so they are reported separately — pooling
them hides the signal.

**Ten fixtures with tight references (`rtol=1e-8`):**

| `dt` | max velocity error | worst fixture |
|---|---|---|
| 0.05 | 0.012040 m/s | `regime_throttle_modulation` |
| **0.02** | **0.006263 m/s** | `regime_stretch_brake` |
| 0.01 | 0.004012 m/s | `regime_cruise` |
| 0.005 | 0.002489 m/s | `regime_stretch_brake` |

**Five fixtures whose references were generated at default tolerance** (the
pathological standstill cases, per spec §9):

| `dt` | max velocity error |
|---|---|
| 0.05 | 0.018886 m/s |
| 0.02 | 0.013152 m/s |
| 0.01 | 0.008716 m/s |
| 0.005 | **0.013141 m/s** ← non-monotonic |

The error on the tight group converges cleanly as `dt` falls. On the loose group
it bottoms out around 0.009 m/s and then gets *worse* at `dt = 0.005` — the
residual there is the reference's own noise, not our truncation error. That is
consistent with why those five were flagged in the first place, and it means
they cannot be used to justify a smaller step.

Convergence on the tight group is roughly first-order, not fourth. That is
expected: local error is dominated by the coupler deadband and the Davis sign
discontinuity, not by RK4 truncation. Refining `dt` therefore buys much less
than the method order suggests, which strengthens the case for the default.

**Keeping `dt = 0.02`.** It sits 8× inside the tolerance on trustworthy
references, and `dt = 0.01` costs 1.8× more for a 1.6× error reduction that the
0.05 m/s budget does not need.

---

## Precision — recommending float64, against the spec's expectation

Spec §5 states: *"A 4090 runs float64 at roughly 1/64 of its float32 throughput,
so float32 is strongly preferred."* **The measurement contradicts this for this
workload.**

| `B` | `N` | float32 | float64 | ratio |
|---|---|---|---|---|
| 64 | 130 | 4.21 s | 5.03 s | 1.20× |
| 128 | 130 | 4.27 s | 5.14 s | 1.20× |
| 256 | 130 | 4.54 s | 5.49 s | 1.21× |
| 256 | 200 | 4.58 s | 5.66 s | 1.24× |

The 1/64 figure is a *FLOP-throughput* ratio, and it would bite on a
compute-bound kernel. This rollout is latency-bound: the GPU is idle most of the
time even in float32, so doubling the arithmetic width is nearly free. **float64
costs about 20 %, not 64×.**

Meanwhile the accuracy margin in float32 is thinner than comfortable. Divergence
from float64 over the full 40 s fixtures at `dt = 0.02`:

| fixture | max Δv | max Δx |
|---|---|---|
| `regime_brake_release` | **0.030124 m/s** | 0.020441 m |
| `regime_startup` | 0.017563 m/s | 0.004194 m |
| `regime_stretch_brake` | 0.013750 m/s | 0.021953 m |
| `regime_coast_to_brake` | 0.012955 m/s | 0.008141 m |
| `regime_emergency` | 0.012287 m/s | 0.030206 m |

`test_float32_error_is_acceptable` passes — it checks `regime_cruise`, which is
benign. But the worst fixture consumes **60 % of the 0.05 m/s tolerance in
rounding error alone**, on a 40 s run. Dataset scenarios are 300–600 s, and that
error accumulates.

**Recommendation: generate the dataset in float64.** It costs 20 % on a job
that now takes minutes, and it keeps the entire error budget available for
physics rather than spending most of it on rounding. Store as float32 if the
files need to be smaller — downcasting the *output* is a different decision from
integrating in reduced precision.

The module default is left at float32 to match the spec's stated API; the
recommendation above is a call for the dataset driver to make explicitly.

---

## What was left alone

**The coupler deadband keeps its exact piecewise form.** Spec §4.8 says to report
and stop if fixed-step error at deadband crossings turns out unacceptable. It is
acceptable: on `adversarial_slack_cycling`, the fixture built specifically to
cycle in and out of slack, engagement agreement with the reference is above the
98 % the contract demands, and peak coupler force — the safety-critical quantity
— is within 5 % on every fixture. No softening was applied, and none is needed
on these grounds. (The *gradient-conditioning* argument for softening it is
separate and still open; see below.)

**Two properties of the NumPy oracle are reproduced deliberately rather than
corrected**, per §4.2. Both are flagged here as the spec asks:

1. **The grade force applies `sin()` to a field that already holds
   `sin(theta)`.** `forces.grade_force` calls `np.sin` on the return value of
   `RouteProfile.sin_theta_at`. So the model computes `m·g·sin(sin θ)`, not
   `m·g·sin θ`. At the grades in play (`|sin θ| ≤ 0.04`) the two differ by about
   1e-5 relative, which is far below anything the dataset resolves — but it is
   in the fixtures, in Chapter 4, and now in the torch port, so all three agree.
   Worth a deliberate decision before the dataset is built rather than after.

2. **Davis resistance uses a hard `sign(v)` with a 1e-9 threshold**, a genuine
   discontinuity at `v = 0`. Reproduced exactly. As spec §9 anticipated, this is
   a candidate for the same `tanh` treatment the brake force got in August: it
   is the remaining hard discontinuity in the RHS, and it sits exactly where the
   dataset spends a lot of its time (held brake at standstill). Gradients through
   it are zero rather than wrong, so it degrades Neural-ODE conditioning rather
   than breaking it.

---

## Test-contract changes

Both flagged per spec §7. Neither loosens a tolerance; both fix tests that could
not pass against *any* implementation. The contract was red at import before the
port existed, so these three tests had never actually executed.

**1. `torch.as_tensor()` does not accept `requires_grad`** — and never has. It
raises `TypeError`, so `test_rhs_gradcheck`,
`test_gradients_flow_through_rollout` and `test_no_inplace_breaks_autograd` all
errored before reaching an assertion. Switched to `torch.tensor()`. No semantic
change.

**2. `test_rhs_gradcheck` evaluated at a point where the model is not
differentiable.** The RHS is piecewise in three places — `max(z, 0)` on the
actuator states, the piecewise-linear route field, and the coupler deadband —
and every fixture's `y0` lands on at least one of them. Actuators start settled
at a zero command, and `x0 = 3000, 2980, …` falls on exact multiples of the
route's 10 m node spacing.

A central difference across a kink returns the mean of the two one-sided slopes,
which cannot equal any single analytic subgradient. Measured on `FIXTURES[0]`:
**7 of 576 Jacobian entries mismatched, every one of them `d(dv)/d(z_brk)` or
`d(dv)/d(z_trac)`, and every numerical value exactly half the analytic one** —
the signature of `relu` at zero. Moving to a fixture with the actuators off zero
relocated the failures to `d(dv)/d(x)`, the route lattice.

Now evaluated at `regime_stretch_brake` — the one fixture holding brake and
traction simultaneously, so both actuator states are strictly positive — with
positions offset 3.7 m off the route lattice. Tolerances are untouched
(`eps=1e-6, atol=1e-6, rtol=1e-4`). The test then checks differentiability where
the model is differentiable, which is its intent.

This is worth noting beyond the test fix: **the RHS has kinks at every route
node**, and a surrogate trained through `F_phys` will see them. If gradient
quality becomes a problem, smoothing the route interpolation (or the actuator
`relu`) is the same class of change as the Davis `tanh` above.

---

## Open items this port does not address

Unchanged from `NEXT_STEPS.md`, and all still gating the dataset:

1. **`pyarrow` is not installed** — needed for `index.parquet` at step 4.
2. **The per-scenario wall-clock budget needs a Windows implementation**
   (`SIGALRM` is Unix-only).
3. **Splits are still `{train: 419, ood_size: 81}`** — no `val`, `test_id`,
   `ood_grade`, `ood_corridor` or `control_eval`. With 5 corridors in one
   region, `ood_corridor` and `ood_grade` are not buildable, and fixing that
   needs the route pipeline with OSM + DEM network access.
4. ~~**`k_curv_scale` is still 0.0 by default**~~ — resolved after this report
   was first written. Investigating the calibration showed the *form* was
   wrong, not just the constant: real curve resistance is speed-independent
   while the proxy scales with `v²`, so no single `k` is correct. A
   `curvature_model` flag now offers `roeckl` and `linear` alongside the
   original `proxy_v2` (still the default, so nothing here changes). See
   `NEXT_STEPS.md` item 3. The port skips the interpolation entirely when the
   scale is zero, exactly as the NumPy path does, and enabling a model costs
   12–20 % throughput on the synthetic benchmark — about 3 % on a real build.
