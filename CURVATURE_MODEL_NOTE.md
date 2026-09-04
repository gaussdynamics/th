# Curvature Resistance — Investigation and Open Decision

_Written 2026-09-03. Closes `NEXT_STEPS.md` item 3 as an investigation; the
modelling decision it sets up is still open. Source material for the reference
simulator chapter, which currently describes only the `proxy_v2` form._

---

## Summary

The reference simulator has always carried a curvature term, disabled by
default, with a note that its magnitude "is not calibrated". Investigating the
calibration showed the problem is not the constant but the **shape**: the
implemented law scales with `v²` while real curve resistance is
speed-independent, so no single scale factor is correct at more than one speed.

Three models are now selectable via `curvature_model`. `proxy_v2` remains the
default, so nothing previously recorded changes. The decision still to be made
is which becomes the model of record for the thesis.

---

## The original law and why it cannot be calibrated

`route.curvature_force_longitudinal`, `model="proxy_v2"`:

```
F_curv = k · m · v² · |κ(x)| · sgn(v)
```

Standard railway practice gives curve resistance as a fraction of vehicle
weight, **independent of speed**:

- **Röckl**: `w_c = 650/(R − 55)` N/kN for `R ≥ 300 m`, `500/(R − 30)` below
- **AREMA**: 0.8 lb per ton per degree of curve, i.e. `w_c ≈ 0.6986·κ`

Both give roughly `6.8·m·κ` newtons. Setting the proxy equal to that fixes `k`
only at one speed:

| radius | Röckl | AREMA | proxy `k=0.5` | proxy `k=0.03` |
|---|---|---|---|---|
| 1000 m | 0.0067 | 0.0069 | 0.0125 / **0.1125** / 0.3125 | 0.0008 / **0.0067** / 0.0187 |
| 300 m | 0.0260 | 0.0228 | 0.0417 / **0.3750** / 1.0417 | 0.0025 / **0.0225** / 0.0625 |

*(N per kg of consist mass; the three proxy values are at v = 5 / **15** / 25 m/s.)*

`k ≈ 0.03` matches the standard models at 15 m/s, and is then **9× too small at
5 m/s and 2.8× too large at 25 m/s**. The exponent is wrong, not the constant.

**The one value that existed in the repo was far off.** `k = 0.5`, in the
`curvature_enabled` reference fixture, gives 13.3 kN on a 100 t car at the
corridors' p95 curvature — 16× the standard value, and 1.4× the force of a 1%
grade. It was evidently chosen to make the term visible in a test, not to be
physical, and it is the only place a non-zero `k` appears.

---

## What was added

`curvature_model ∈ {proxy_v2, roeckl, linear}`, threaded through `route.py`,
`rhs.py`, `torch_rhs.py`, `randomize.py`, the dataset writer and the build
driver. Strictly additive: **`proxy_v2` remains the default**, so Chapter 4, the
reference fixtures and every previously recorded result are untouched, and the
full pre-existing suite still passes.

For `roeckl` and `linear`, `k_curv_scale` becomes a dimensionless multiplier on
the standard formula — **1.0 is the textbook value**. That is what turns an
uncalibrated tuning knob into a calibrated one.

Implementation details worth knowing:

- Both are expressed in `κ` rather than `R = 1/κ`, so tangent track is not a
  division by zero.
- Curvature is clamped at a 60 m radius (`MIN_CURVE_RADIUS_M`). Röckl's
  denominators vanish at `R = 55 m` and `30 m`; real track does not go near that
  — the sharpest curve in the current corridor set is 80 m — but a noisy DEM
  curvature field must not be able to put an infinity into the RHS.
- The torch batch carries `curvature_model` as one string for the whole batch,
  not a per-scenario tensor, so the traced graph keeps a Python constant instead
  of a select that never varies. Mixed-model batches are **rejected**, not
  silently resolved.

**Verification.** Torch and NumPy agree to **7.6e-17 relative** on the RHS for
every model — machine precision. 14 tests in
`simulator2/tests/test_curvature_models.py` cover the published formulae, the
speed-independence property, sign behaviour, the clamp, torch/NumPy parity,
differentiability, and mixed-batch rejection.

---

## Röckl's discontinuity

Röckl is genuinely **discontinuous at R = 300 m**, where the two branches meet:

| R | `w_c` |
|---|---|
| 301.0 m | 0.002642 |
| 300.0 m | 0.002653 |
| 299.9 m | 0.001853 |
| 299.0 m | 0.001859 |

A ~30% jump. This is a property of the published formula, not of the
implementation, and `test_roeckl_is_discontinuous_at_300m` pins it so it cannot
be "fixed" by accident.

It matters here more than it would elsewhere. The physics-residual design
evaluates `F_phys` inside autograd on every training step, so a step
discontinuity in the RHS is a step in the gradient. This project has
consistently gone the other way — the brake sign uses a `tanh` blend rather than
`sgn` precisely so the RHS stays smooth for the solver and for autograd, and
`TORCH_PORT_REPORT.md` flags the remaining hard `sign(v)` in Davis resistance as
a candidate for the same treatment.

`linear` (AREMA) tracks Röckl within about 15% over 300–2000 m, is smooth
everywhere, and is marginally cheaper. **It is the recommended model.**

---

## How much it changes

Measured on the full 10,000-scenario corpus by building `data/v1` (curvature
off) and `data/v2` (`linear`, `k = 1.0`) from the same seed and the same commit,
so the two share scenarios exactly and only the force law differs.

**Trajectories** (500 scenarios diffed in full):

| | p50 | p90 | max |
|---|---|---|---|
| max \|Δv\| | **0.374 m/s** | 1.56 | 14.4 |
| max \|Δx\| | 16.6 m | 164 | 1564 |

**Summary quantities** (all 10,000):

| | p50 | p90 | p99 | max |
|---|---|---|---|---|
| ΔF_max | 0.50% | 4.6% | 20.9% | 116% |
| ΔE_trip | 0.86% | 10.3% | 86% | — |
| Δdistance | 1.3% | 10.9% | 78% | — |

Reading:

- **The trajectories move well beyond the project's own tolerance.** The median
  0.374 m/s is ~7× the 0.05 m/s the torch port was held to. Omitting curvature
  is not a small perturbation of where the train ends up.
- **The safety-critical quantity barely moves.** `F_max` shifts 0.50% at the
  median and 4.6% at p90 — inside the 5% gate the port's own contract uses.
- **The mechanism is integration, not sensitivity.** 809 N on a 100 t car is
  0.008 m/s²; over a 600 s window that is 4.8 m/s and 1.4 km. A small persistent
  force accumulating over a long run, not chaotic divergence. This is why an
  instantaneous force-ratio argument (which is what "curvature is only ~8% of a
  1% grade" amounts to) gives the wrong answer, and why the comparison had to be
  integrated rather than reasoned about.
- **Curvature slightly suppresses runaways.** Scenarios exceeding the route
  speed limit by >5 m/s fall from 18.6% to 17.8%, since curve drag opposes
  downhill acceleration.

### The `F_max` tail

`ΔF_max` reaches 20.9% at p99 and 116% at worst. Those are concentrated in the
overspeed population on the clamped 4% grades, where adding drag changes *when*
a consist runs away rather than perturbing a stable trajectory. Filtering on
`v_over_limit_max_mps` removes most of that tail along with the runaways
themselves.

### A note on the earlier estimate

A first sensitivity probe used the `proxy_v2` law at `k = 0.03` and reported a
median of 0.14 m/s with a maximum of 25.5 m/s. The correct law gives a *larger*
median (0.374) and a *much smaller* maximum (14.4). The proxy's extreme tail was
an artifact of `v²` amplifying on runaway scenarios — at the 50+ m/s a runaway
reaches, the proxy returns 11× its 15 m/s value. The physically correct law is
more uniformly significant and far less pathological, which is itself an
argument for it.

---

## The open decision

Three options, in the order I would rank them:

1. **Adopt `linear` at `k = 1.0`.** Physically standard, smooth, calibrated,
   ~3% build cost. Requires a Chapter 4 amendment describing the law, and the
   `curvature_enabled` reference fixture would need regenerating if it is to
   exercise the new default. `data/v2` is already built this way.
2. **Adopt `roeckl` at `k = 1.0`.** Closer to the published railway formula, at
   the price of a step discontinuity in a gradient the surrogate differentiates
   through.
3. **Leave curvature off and drop `kappa` from the model's inputs.** Defensible
   — the corridors are mostly straight (pooled median κ = 3.0e-5, R ≈ 33 km) —
   but currently `route_kappa` is stored and normalized while having provably
   zero effect on the target, so any route encoder will learn to ignore a
   channel that is not ignorable on real track. If curvature stays off, the
   channel should go too, and that should be stated rather than left implicit.

What should **not** happen is keeping `proxy_v2` with a fitted `k`: it bakes a
known-wrong speed dependence into the corpus to gain a term that is a few
percent of the grade force.

## Cost

Measured at `B=256, N=130`, float64, 120 s simulated:

| | s | vs off |
|---|---|---|
| off | 4.95 | — |
| `linear` | 5.55 | 1.12× |
| `roeckl` | 5.94 | 1.20× |

On a real build the overhead is smaller — 1336 s to 1389 s, about 3% — because
real scenarios spend proportionally more time in the rest of the RHS.

---

## Corridor curvature, for reference

| corridor | median κ | p95 κ | max κ | min radius |
|---|---|---|---|---|
| `route_line0_49km` | 3.97e-05 | 1.16e-03 | 3.64e-03 | 275 m |
| `route_line2_33km` | 4.88e-04 | 1.41e-03 | 1.25e-02 | 80 m |
| `route_line3_29km` | 1.74e-05 | 1.15e-03 | 4.09e-03 | 245 m |
| `route_line4_27km` | 5.95e-06 | 4.22e-04 | 3.68e-03 | 272 m |
| `route_line5_22km` | 2.26e-05 | 1.05e-03 | 2.03e-03 | 494 m |

Pooled: median 3.04e-05 (R ≈ 33 km), p95 1.19e-03 (R ≈ 840 m).

For scale, on one 100 t car at 15 m/s at the pooled p95 curvature:

```
Davis resistance          938 N
grade at 1%              9810 N
Röckl curve resistance    809 N     <- 86 % of Davis, 8 % of a 1 % grade
proxy k=0.5            13,337 N
proxy k=0.03              800 N
```
