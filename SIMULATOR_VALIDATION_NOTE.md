# Simulator Validation — "How do you know the simulator is accurate?"

_Opened 2026-09-26. **Major emphasis**: this runs alongside the surrogate and
route work, not after them. Companion to `PROJECT_DIRECTION.md` (whose
"calibration caveat" this replaces with evidence) and Chapter 4._

## The question

At the September committee meeting the committee's mathematician asked how
we can know the simulator is accurate. Nearly every number in the thesis
except the route dataset comes from it: the corpora, the surrogate's
training targets and error metrics, and the control chapter's fuel figures.

The answer given in the meeting was that the contribution is the
system-identification framework, not a simulator, so simulator accuracy is
not required. That is defensible, but it is currently asserted rather than
shown. The aim here is to (a) show as much accuracy as can be shown, (b) fix
what is wrong before the next data regeneration, and (c) turn the fallback
argument into a measured result.

## Three questions that must be kept apart

| | Question | Status |
|---|---|---|
| **Verification** | Does the code solve the equations Chapter 4 writes down? | **Done for every force law — see Tier 1.** |
| **Validation** | Do those equations describe a real freight train? | Not established. Parameters are uncited and several model-form simplifications are known (below). |
| **Claim dependence** | Which thesis claims need validation at all? | Surrogate-vs-simulator claims do not. Transfer to real trains and the control chapter's fuel numbers do. |

## Plan

1. **Tier 1 — verification against exact solutions.** _Done 2026-09-26._
2. **Tier 2 — sourced parameters and physics corrections.** Replace illustrative values with cited ones and fix the unphysical items in the findings table. Do this before the next data regeneration, so the data is regenerated once, not twice.
3. **Tier 3 — code-to-code benchmark.** The international longitudinal train dynamics simulator benchmark: Spiryagin, Wu & Cole, *VSD* 55(4) 2017 (questions, already in `references.bib`), and the companion results paper, Wu, Spiryagin, Cole et al., *VSD* 56(3):343–365, 2018, doi:10.1080/00423114.2017.1377840 (confirmed 2026-09-26). Nine simulators took part: TABLDSS, UM, CRE-LTS, TDEAS, PoliTo, TsDyn, CARS, BODYSIM and VOCO. The abstract reports basic agreement on locomotive, resistance and grade forces, with the major differences in the **draft gear models**, which is exactly F3. Reproduce its cases; land inside the spread of the participating simulators. First step: read which model features the cases require, since that decides how much of Tier 2 must come first.
4. **Tier 4 — measured field data.** Scarce publicly. State it as a limitation unless something turns up.
5. **The fallback as a result.** Train and evaluate the surrogate on illustrative-parameter and sourced-parameter data. If surrogate error barely changes, "the framework does not need an accurate simulator" becomes a measured result. Pair it with the hysteretic draft gear from Tier 2: that is a harder, history-dependent nonlinearity, and surviving it answers "would this work on a real train?"
6. **Control chapter framing.** Fuel savings are savings *within the model*, stated as such.

## Tier 3 scope — from the benchmark questions paper (read 2026-09-26)

Source: `paper/refs/papers/internatl_benchmarking_ltd_simulators.pdf`
(Spiryagin, Wu & Cole 2017). The paper is fully self-contained: formulas,
the track (Appendix 1) and the controls (Appendix 2) are all printed.

**The cases.** Four freight trains, 5000 s each:

| Train | Consist | Mass (t) | Length (m) |
|---|---|---|---|
| 1 | 2 locos (type 1) + 50 wagons | 6,668 | 796 |
| 2 | Train 1 twice, distributed power | 13,336 | 1,592 |
| 3 | 3 locos (type 2) + 240 wagons | 38,985 | 2,713 |
| 4 | 2 locos (type 2) + 120 wagons + 1 loco (type 2) + 120 wagons | 38,985 | 2,713 |

- Vehicles: locomotives of 134 t and 195 t (6 axles); wagons of 128 t and 160 t (4 axles).
- Wagons run in pairs joined by a slackless drawbar. All other connections are couplers with 10 mm total slack.

**What the benchmark specifies, against what we have:**

| Item | Benchmark | Ours | Work |
|---|---|---|---|
| Air brakes | **Excluded on purpose** (§2) | load-independent, no propagation | none for the benchmark. F1/F2 stay a Tier 2 item and remain unvalidated by it |
| Draft gear | Friction-rubber gear from a drop-hammer loading/unloading cross plot (Fig. 3: ~85 mm travel, ~2.1 MN peak); participants build their own model | linear Kelvin–Voigt spring-damper, no hysteresis | **New model: the main job.** The results paper says draft gear is where simulators disagree most |
| Drawbars | zero slack | — | Edge type with no slack |
| Traction | 8 notches: `F = (N/8)·Te_max` if `F·v < (N/8)²·P_max`, else `(N/8)²·P_max/v` (Tables 4, 6) | `min(u·F_max, P/v)` | Same form. Only notch → (force, power) mapping needed |
| Dynamic brake | 8 notches, piecewise in speed (Tables 5, 6) | **absent** | New force channel (F5) |
| Notch rate limit | not specified, participants choose | first-order lag, τ = 5 s | Keep; state it |
| Propulsion resistance | `Q·m·(2.943 + 89.2/m_a + 0.0306V + 0.122V²/m)` N, V in km/h, Q = 3.2 lead loco else 1 | `A + Bv + Cv²` per vehicle | Exact conversion to per-vehicle A, B, C |
| Curving resistance | `6116/R` N per tonne; curves 50 m transition + 300 m arc + 50 m transition | AREMA `linear` law, 6.853 N/t per 1/R | `k_curv_scale = 0.8925` reproduces it exactly |
| Grade | Appendix 1 gradient table, 50 km, extended flat beyond | `sin θ(s)` field | Transcribe |
| Distributed power | remote locomotives get the lead's notch 3 s later | per-vehicle command arrays | Expressible directly |
| Outputs | lead-loco position and speed; coupler force and draft-gear deflection at set positions (Table 7); max tension/compression per coupler; 10 ms output step | — | Output writer |

**Takeaway.** The benchmark tests couplers, traction, dynamic brake,
resistance and grade: the in-train force problem without air brakes. To take
part we need a hysteretic draft gear (which also fixes F3 and the
contact-event convergence issue), a dynamic brake, notch tables and
drawbars. It does **not** validate air braking, so F1/F2 need their own
literature-based justification.

**Results paper** (Wu et al. 2018) obtained 2026-10-04:
`paper/refs/papers/internatl_benchmarking_ltd_results.pdf`, bib key
`Wu2018BenchmarkResults`. Comparison in the 2026-10-04 log entry.

**Corrigendum** (*VSD* 56(2):341, 2018, doi:10.1080/00423114.2017.1397356;
`paper/refs/papers/Corrigendum.pdf`): its only change is to Table 2. The
middle locomotive of Train 4 is **type 2**, not type 1 as the original
table printed it. The paper's stated totals (38,985 t, 2713.05 m) already
match the corrected consist (3 × 195 + 240 × 160 t; 3 × 24.35 + 240 × 11 m),
so only the table was wrong. Nothing else (draft gear, traction, track,
driving) is amended, so the Fig. 3 digitization stands.

## Tier 1 results

Code: `simulator2/verification.py`, `simulator2/tests/test_verification.py`
(18 tests, ~50 s), `scripts/verify_simulator_convergence.py`.

Everything runs the production path: `torch_rhs` under fixed-step RK4 at
`dt = 0.02 s`, in float64. Each case is compared with something that shares
no code with it. Short timescales (light masses) are used where the closed
form allows, to keep the runs fast.

**Mutation check.** Two bugs were planted to confirm the tests can fail:
- Coupler force directions swapped: caught by the oscillator test.
- Draft damper removed: caught by the energy and contact tests.

| Case | Exact reference | Error at production `dt` |
|---|---|---|
| Coastdown (Davis only) | arctangent closed form | 5e-14 m/s |
| Constant grade | linear in time | 9e-13 m/s |
| Balancing speed, force-limited and power-limited | quadratic / cubic root | < 1e-6 relative |
| Actuator lag | exponential | 8e-13 relative |
| Brake stop with `tanh` blend | `v0²/2a + ε²π²/12a` | 1e-6 m over 100 m |
| Deadband oscillator (slack, asymmetric k, undamped) | piecewise-harmonic period and buff amplitude | 6e-4 period, 1.7e-3 amplitude; → 3e-7 at `dt = 0.001` |
| Damped contact | Kelvin–Voigt restitution `exp(−ζπ/√(1−ζ²))` = 0.283 | 0.3 % at `dt = 0.001` |
| Chain normal modes, N = 10 | `ω_j = 2√(k/m) sin(jπ/2N)`, cosine shapes | leakage 1e-11; phase error equals RK4's own dispersion `(ωh)⁴/120` |
| Momentum (couplers internal) | Σ m v′ = Σ external force | 2e-15 relative |
| Energy audit, realistic 10-car consist | KE + PE_grade + PE_spring vs work | 1e-7 of energy throughput |

**Convergence** on the realistic consist, against a converged reference
(`dt = 0.000625 s`):

| dt [s] | max \|Δv\| | p99 \|Δv\| | median \|Δv\| | final \|Δx\| |
|---|---|---|---|---|
| 0.02 (production) | 7.5e-3 | 1.0e-3 | 7.5e-7 | 2.5e-5 |
| 0.01 | 1.1e-2 | 1.5e-3 | 1.3e-6 | 4.0e-5 |
| 0.005 | 2.6e-3 | 3.7e-4 | 3.7e-7 | 1.1e-5 |
| 0.0025 | 3.3e-3 | 4.4e-4 | 4.1e-7 | 1.2e-5 |

The reference agrees with a `dt = 0.00125` run to 6e-4 m/s (maximum).

- **Typical** velocity error at the production step is about **1e-6 m/s** (median).
- **Maximum** error is **7.5e-3 m/s**, and it does **not** fall steadily as dt shrinks. The p99 behaves the same way. The maximum comes from a handful of coupler contact events: the damper engages discontinuously (the force jumps by `c·δ̇`), so the moment of impact is resolved poorly, and small timing shifts change which impact produces the largest error.
- Chapter 7's dt sweep figure of 0.0063 m/s is this maximum, and should be described as a contact-event bound rather than a convergence result.
- Final position error after 60 s is 2e-5 m.

**Conclusion for Tier 1:** the code solves its equations. Integration error
at the production step is negligible away from contact events. At contact
events it is bounded, ~1e-2 m/s at worst, but not cleanly convergent. That is
a property of the discontinuous contact model (F3), not a bug in the
integrator. A hysteretic draft gear that engages without a force jump would
remove it.

## Findings: behaviour that is verified but physically questionable

Tier 1 pins each of these down with a `..._characterized` test, so changing
the physics fails loudly. The list comes from the code review of 2026-09-26.

| # | Finding | Where | Severity | Tier 2 action |
|---|---|---|---|---|
| F1 | Brake force 250 kN on every vehicle regardless of load, no adhesion cap: an empty car can decelerate at ~0.85 g | `constants.py`, `catalog.py` | **High** | Brake force from AAR net braking ratio × load; cap at adhesion |
| F2 | No brake-pipe propagation: every car brakes at once. Propagation delay is the main source of in-train force in braking, and coupler force is the control chapter's safety constraint | `dataset.py:158-160` | **High** | Serial application delay along the consist (propagation speed from the literature) |
| F3 | Coupler is Kelvin–Voigt with a deadband and no hysteresis. On unloading it pushes while still stretched (~25 % of peak pull, measured). Real draft gear dissipates mainly through friction hysteresis | `torch_rhs.py:670-687` | **High** (model form) | **Built 2026-09-26**: stateful friction gear on the benchmark's drop-hammer curve, `simulator2/draft_gear.py`, opt-in. Not yet in the corpora |
| F4 | Davis A and B about 2–4× below the original Davis values for a 100 t, 4-axle car (≈1.1 kN, 49 N·s/m against 600 N, 12). C is plausible | `constants.py:7-16` | Medium | AREMA / modified Davis, per axle and per ton |
| F5 | No tractive-effort curve and no dynamic brake. SD70 is 130 t and 300 kN (real: ~180 t, 600 kN+). P_max is 3.5 MW for every locomotive type | `constants.py:26-29`, `catalog.py` | Medium | Manufacturer TE curves; dynamic brake as its own channel |
| F6 | Resistance takes a hard `sign(v)`, so a stopped vehicle chatters about v = 0 instead of resting; nothing holds a stopped train on a grade (22 % of scenarios roll backwards) | `forces.py:12-19` | Low–medium | Static friction / holding logic, or exclude standstill from the corpus |
| F7 | Grade force is `sin(sin θ)`: −st²/6 relative, 2.7e-4 at the 4 % clip. The port report's "about 1e-5" is the 1 % value | `torch_rhs.py:623-633` | Negligible | Fix alongside the next regeneration |
| F8 | Speed limits not enforced (17.8 % of scenarios exceed by > 5 m/s) | ch.07:279 | Scenario realism, not physics | Speed-aware control profiles |
| F9 | Point-mass grade (no averaging over the car length); no rotational-inertia allowance | `torch_rhs.py` | Low | Rotating-mass factor ~2–5 %; length averaging optional |

## Log

- **2026-09-26** — Question raised at the committee meeting. Code review of
  the physics done. Tier 1 verification suite written and passing; mutation
  check confirms it can fail. Findings F1–F9 recorded. Convergence study run:
  the maximum error at contact events is not cleanly convergent (see Tier 1).
  Read the benchmark questions paper and wrote the Tier 3 scope above.
  Air brakes are excluded from the benchmark. The work is draft gear,
  dynamic brake, notches and drawbars. Results paper and corrigendum still
  to be obtained.
- **2026-09-26 (draft gear)** — Digitized benchmark Fig. 3 by pixel colour:
  `scripts/digitize_draft_gear.py` → `simulator2/data/draft_gear_benchmark.csv`
  (peak 2126 kN at 82.4 mm; overlay checked visually). Built the stateless
  look-up-table model (`simulator2/draft_gear.py`). It is opt-in:
  `coupler_model="draft_gear"` on the torch batch, default unchanged.
  The table and force-law properties are tested in `test_draft_gear.py`, and
  torch matches NumPy to 2e-10 N.
  **Finding: the stateless velocity switch is unfit at the production step.**
  Its slope `(F_L − F_U)/(2 v_t)` acts as a damper of ~4e7 N·s/m at
  `v_t = 5 mm/s`, about 80× the KV damper and far outside explicit RK4
  stability at `dt = 0.02`. Energy audit on the 10-car realistic case, 30 s:

  | v_t [m/s] | dt = 0.02 | dt = 0.005 | dt = 0.00125 | converged dissipation |
  |---|---|---|---|---|
  | 0.005 | resid 4.7e-3, dissipates **333 kJ** | 2.5e-4, 28 kJ | 1.6e-8, 11 kJ | 11 kJ |
  | 0.02 | 1.4e-3, 100 kJ | 1.6e-8 | 1.6e-8 | 6.2 kJ |
  | 0.1 | 1.6e-8 | 1.6e-8 | 1.6e-8 | 4.3 kJ |

  At the production step the switch manufactures ~30× the true dissipation.
  Even converged, the dissipation depends on `v_t`, which has no physical
  meaning.
- **2026-09-26 (draft gear, decision)** — **Chose the stateful friction
  model.** Each connection carries its gear force `z` as a state. Between the
  curves the gear sticks (`ż = k_t·ė`); at a curve it slides along it. The
  output is `z` clamped to the envelope. It is opt-in: under
  `coupler_model="draft_gear"`, `y` grows from `4N` to `4N + N − 1`; the
  default path, corpora and surrogate are untouched.
  - **Energy.** The audit closes to 1.6e-8 at `dt = 0.02` (same as fine
    steps). Coupler dissipation is within 0.4 % of converged at the
    production step (velocity switch: 30× off).
  - **Behaviour tests** (`test_draft_gear.py`, 13 passing):
    - It follows the loading curve out and the elastic line → unloading curve back.
    - It holds its force at rest.
    - Small reversals inside the envelope are elastic (no net work).
    - The force never leaves the envelope.
    - Momentum is exact.
    - It is stable on 26 t empties at `dt = 0.02`.
  - **The one modelling choice, `k_t`, quantified.** The drop-hammer unloading
    drop implies ~7e8 N/m per connection. The fraction of the loop area
    captured:

    | k_t [N/m] | 25 % travel | full travel |
    |---|---|---|
    | 1e8 (default) | 90 % | 86 % |
    | 3e8 | 97 % | 97 % |
    | 7e8 | 99 % | 100 % |

    The default trades 10–14 % of dissipation for RK4 stability at
    `dt = 0.02` on empties. The benchmark trains are all loaded (128–195 t)
    and ask for 10 ms output, so they use 7e8 at `dt = 0.01`.
  - **Convergence: worse than Kelvin–Voigt.** I had predicted it would be
    better, because the force no longer jumps at the slack edge; that was
    wrong. `ż` switches between stick and slip, and the curve slopes jump at
    every 0.5 mm table knot, so in contact the RHS is discontinuous almost
    everywhere. Realistic 10-car case against `dt = 0.000625`:

    | dt [s] | max \|Δv\| | p99 | median | final \|Δx\| |
    |---|---|---|---|---|
    | 0.02 | 2.5e-2 | 5.2e-3 | 9.6e-5 | 1.6e-4 |
    | 0.01 | 2.8e-2 | 4.7e-3 | 9.2e-5 | 1.3e-4 |
    | 0.005 | 8.8e-3 | 2.0e-3 | 3.7e-5 | 4.1e-5 |
    | 0.0025 | 9.5e-3 | 1.8e-3 | 3.2e-5 | 5.7e-5 |

    The reference itself agrees with `dt = 0.00125` only to 4.4e-3 m/s, so
    it is not converged. Energy closes and dissipation converges;
    trajectories do not converge at RK4's rate. The published simulators use
    far smaller steps for friction gear. Open options:
    - smooth the table (a spline with continuous slope) to remove the knot
      jumps;
    - an event-aware or implicit integrator for the gear state;
    - accept a smaller dt for gear runs.

    Whatever is chosen, the benchmark comparison will show whether the
    remaining error matters at the level the nine simulators agree to.
  - **Implication for the surrogate** (when the corpus is regenerated with
    this gear): the coupler becomes history-dependent, so the GNS must carry
    an edge state. This is a harder and more realistic learning problem.
- **2026-09-30** — Results paper (Wu et al. 2018) requested through
  interlibrary loan; expected in a few days. Corrigendum obtained and
  read: it only corrects the Train 4 consist (middle loco is type 2).
  Train table above updated.
- **2026-09-30 (benchmark harness)** — Built and ran the Tier 3 harness.
  - `scripts/extract_ltd_benchmark_tables.py` transcribes Appendix 1 (86
    gradient points) and Appendix 2 (notch sequences) from the PDF into
    `simulator2/data/ltd_benchmark/`.
  - `simulator2/ltd_benchmark.py` holds the trains (corrected Train 4),
    drawbar pairs, the Eq. 1 resistance as exact Davis terms, Eq. 2 curving
    (`k_curv_scale = 0.8925`, curves 50 + 300 + 50 m), the 3 s DP delay, the
    unrolled control cycles, and a chunked runner that reduces to the five
    section 3.6 outputs at 10 ms.
  - `torch_rhs` gains an opt-in `traction_model="notch"`: Table 4 traction and
    the dynamic brake (F5, benchmark form). Default path unchanged; full
    simulator suite passes (291), 40 new tests in `test_ltd_benchmark.py`.
  - **Interpretations.** Table 5 is inconsistent as printed for DB notches
    1-6; read against Figs 4b/5b (plateau `DB_max n/8.7` for 1-3, `n/7.5` for
    4-6, common ramp, power cap), which reproduces the figures' crossovers.
    Appendix 1 gradients are steps: that integrates to Fig. 6a (93.6 m valley,
    145.1 m summit, 113.3 m at 50 km); linear interpolation ends ~5 m high.
  - **Choices left open by the paper:** notch lag 5 s; friction gear at
    `k_t = 7e8`, `dt = 0.01 s`; start at rest, lead front at 0 m.
  - **First full run** (`scripts/run_ltd_benchmark.py`, 21 min on the 4090,
    all four trains in one batch). Only check available before the results
    paper: the controls were generated to stay within 110 % of the Fig. 7
    speed board.

    | Train | Distance at 5000 s | Worst vs board | Max tension | Max compression |
    |---|---|---|---|---|
    | 1 | 101.0 km | **140 %** at 5.0 km | 572 kN | 425 kN |
    | 2 | 100.1 km | **116 %** at 5.0 km | 573 kN | 415 kN |
    | 3 | 71.5 km | 103 % | 1638 kN | 1120 kN |
    | 4 | 72.6 km | 104 % | 1319 kN | 1026 kN |

    Trains 3 and 4 stay inside the board. Trains 1 and 2 overspeed in the
    60 km/h zone on the -0.9 % descent before 5 km: Train 1 is already at
    62 km/h at 393 s, where the controls call notch 8, and reaches 84 km/h.
    Either our light trains accelerate faster than the reference simulator
    in the first 400 s, or the driving algorithm anticipated the 80 km/h
    board. Not resolvable until the results paper's speed traces arrive.
- **2026-10-04 (results paper; comparison)** — Read Wu et al. 2018.
  - **Start location.** Section 2: all trains start at 3 km and only
    3-50 km is compared. The questions paper never says so; our harness
    started at 0 m, which caused the Train 1/2 overspeed above (controls
    played on the wrong stretch; average speed ~10 % high on all trains).
    `START_M = 3000` is now the default. The track behind 3 km falls, so with
    no air brakes or starting resistance the train rolls before notch 1.
  - Tables 3-6 transcribed to `simulator2/data/ltd_benchmark/wu2018_tables.csv`
    (checked against page images; text extraction shifts the speed columns).
    Fig. 2 digitized to `wu2018_fig2_speed.csv` (maxima inside each table's
    range). Comparison: `scripts/compare_ltd_benchmark.py`.
  - **Result, notch lag 5 s (headline):** speed matches on all trains (Fig. 2
    RMS 0.4-0.6 km/h). 23 of 32 table quantities inside the range of the
    paper's core six (TABLDSS, UM, CRE-LTS, TDEAS, TsDyn, CARS), 6 more
    within 2 %. Largest-force positions match. **Misses: Train 1 peak tension
    486 kN vs 549-734 for all nine (−11 %); Train 1 mean max tension −8 %.**
    The peak is at start-up (41 s, 10 km/h) and depends on the notch lag,
    which the benchmark leaves open: 486 / 541 / 565 / 622 kN at 5 / 2 / 1 /
    0.2 s. Full run at 1 s: 28 of 32 inside, the rest within 4 %. Kept 5 s as
    the headline since it predates the results.
  - **Deflection** is undefined in the paper; read as the whole connection
    (two gears + slack) since participants exceed one gear's 82.5 mm.
  - **Caveat for the thesis:** this validates the opt-in configuration
    (friction gear, notch traction, benchmark parameters), not the default
    path the corpora were generated on. Air brakes (F1/F2) untested. No dt
    sensitivity of the force results yet.
  - Write-up: https://claude.ai/artifact/B3H96WhB8rK56FpzqT2BHk
- **2026-10-04 (air brake, first implementation)** — Empirical propagation
  model (Wu, Spiryagin & Cole 2016, section 4.2, which recommends this class
  when the air brake is not the object of study): `simulator2/air_brake.py`.
  **All parameters are placeholders**; `AirBrakeParams.source` is `None` and
  `require_sourced()` refuses them, so no data can be generated from them yet.
  - Brake wave at `c` (placeholder 250 m/s) shifts each vehicle's command by
    `d/c`; build-up keeps the existing first-order lag with a time constant
    linear from head (3 s) to tail (10 s); fixed shoe force from design gross
    weight times shoe friction, capped at adhesion on actual weight; direct
    release with a recharge time.
  - **F1 reframed:** conventional cars have *fixed* shoe force, load
    independent. The defect is the 250 kN magnitude (about 6x the placeholder
    loaded-car value) and the missing adhesion cap. With the placeholders a
    26 t empty sits right at the adhesion limit (~0.15 g).
  - RHS change: optional `TorchScenarioBatch.tau_brk_node` (B, N); `None`
    keeps the scalar path (a uniform `tau_brk_node` is bit-identical to it).
    Everything else is command-side and opt-in; corpora untouched.
  - Tests (`test_air_brake.py`, 14): uncoupled vehicles match the closed-form
    build-up to 1e-5 m/s; coupled-train momentum matches the summed brake
    impulse to 1e-7; direct-release rules; capacity and cap. Mutation check:
    dropping the per-vehicle tau fails 2 tests, dropping the delay fails 3.
    Full simulator suite: 306 passed.
  - **F2 quantified:** 60 uniform cars, full application. Instantaneous
    train-wide brake (today's corpus model): **0 kN** run-in, every car
    decelerates together. With propagation: 440-500 kN peak buff force.
    The corpora cannot contain braking run-in on uniform consists.
  - Direct release barely changes the corpus regimes (94-100 % of braking
    impulse kept), except `DYNAMIC_BRAKE_DESCENT` (85 %, ~1 forced release per
    run), which modulates the *air* brake as a stand-in for the dynamic brake
    that did not exist then. Move it to the notch dynamic brake at regeneration.
  - **Not done yet:** wiring into `build_dataset.py`. The dataset files,
    `scripts/validate_dataset.py` and `surrogate/data.py` assume a scalar
    `tau_brk_s`; they need the per-vehicle array. Separate release time
    constant not modelled. Parameters need sources: Wei et al. 2017 (VSD
    55(4):517-533) for propagation and cylinder curves; AAR for net braking
    ratio.

- **2026-10-05 (air brake, timing calibrated to Wei et al. 2017)** — VSD
  55(4):517-533, measured curves of a Shuo-Huang 10,000 t train (HXD1 + 116
  C80, Chinese 120 valve), 4th and 116th car. The figures are vector
  graphics, so `scripts/digitize_air_brake.py` reads the curves from the PDF's
  line segments (output in `simulator2/data/wei2017_air_brake/`). Emergency
  onset spread comes out 7.59 s against 7.7 s in the paper's text, which
  checks the extraction.
  - Application wave: 7.03 s (Fig. 5) and 8.25 s (Fig. 7) over 112 car
    lengths → **176 m/s** (163-191). Release wave: 12.82 s → **105 m/s**,
    now separate (decreases propagate at the release speed).
  - Build-up tau: **10.2 s** at the head, **30.6 s** at the 116th car
    (Fig. 7, 140 kPa step), now linear *per metre* (0.0152 s/m), not
    normalised to train length. The head is two-timescale (fast jump, slow
    creep; first-order rms 24 kPa of 400). Emergency (Fig. 9, rig) is
    faster and uniform, 4.3 / 5.2 s, but is not modelled.
  - Release tau **6.7 s** (6.89 / 6.52), new RHS field
    `tau_brk_release_node`, used while the force falls. Recharge **90 s**
    (head car; the tail is still ~20 kPa short at 206 s, so it is optimistic).
  - Provenance is per field (`AirBrakeParams.sources`). Still unsourced:
    net braking ratio, shoe friction, adhesion, design gross weight, and
    the C80 car length (12.0 m) that turns delays into speeds.
    `require_sourced` still refuses.
  - Caveat for the thesis: a Chinese 120 valve, not North American ABDX.
    Tests: 18 air brake (2 caught by mutation), full suite 310 passed.
    F2 run-in figures from 2026-10-04 used the placeholders and need a re-run.
