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

## Air brake reading list (2026-10-05)

What to obtain to finish sourcing `AirBrakeParams` (see the 2026-10-05 log
entry). ✔ = appears in a reference list of a paper already in
`paper/refs/papers/`, so the citation is confirmed. The rest are standard
works cited from memory: check edition and chapter when requesting them.
**Start with 1, 2, 5 and 6.**

*Force parameters (still placeholders):*

1. AAR, *Manual of Standards and Recommended Practices*, Section E (Brakes
   and Brake Equipment): **S-401** (Brake Design Requirements) for net
   braking ratio; **M-926** (composite brake shoes) for shoe friction. Paid;
   try the library. Check S-401's definition of the ratio: if its shoe force
   already includes rigging losses, the model must not apply them again.
2. Spiryagin, Wolfs, Cole, Spiryagin, Sun & McSweeney, *Design and
   Simulation of Heavy Haul Locomotives and Trains*, CRC Press, 2016. Heavy
   haul brake modelling, adhesion, typical values.
3. Iwnicki (ed.), *Handbook of Railway Vehicle Dynamics*, CRC Press (2nd ed.
   2019; 1st ed. 2006): Cole's "Longitudinal Train Dynamics" chapter.
4. Hay, *Railroad Engineering*, 2nd ed., Wiley, 1982. Origin of
   F = Rb·W·e·f (the form of `brake_capacity_n`); shoe friction vs speed.
   ✔ (Wang & Rakha 2018)

*North American valve (the timing is from a Chinese 120 valve):*

5. Specchia, Afshari, Shabana et al., "A train air brake force model:
   locomotive automatic brake valve and brake pipe flow formulations,"
   *Proc IMechE F* 227(1):19–37 (2012). ✔ (Wei ref. 31)
6. Afshari, Specchia, Shabana et al., "A train air brake force model: car
   control unit and numerical results," *Proc IMechE F* 227(1):38–55 (2012).
   ✔ (Wei ref. 32)
7. Johnson, Booth & Mattoon, "Development of practical techniques for the
   simulation of train air brake operation," ASME 86-WA/RT-4. ✔ (Wei ref. 7)
8. Abdol Hamid, Limbert, Gauthier et al., "Simulation of freight train air
   brake system," ASME 86-WA/RT-15 (ABD/ABDW). ✔ (Wei ref. 6)
9. Blaine & Hengel (1971), brake system operation and testing and their
   effect on train performance. ✔ (Wang & Rakha 2018; full citation there)

*Model form and calibration:*

10. Murtaza, "Railway air brake simulation: an empirical approach,"
    *Proc IMechE F* 207:51–56 (1993). The first-order cylinder lag; cite
    directly, not only through the survey. ✔ (Wei ref. 14)
11. Wei, Ahmadian & Zhang, "Heavy haul train simulation of air brake system
    and longitudinal dynamics," 2014 IEEE/ASME Joint Rail Conference. Most
    likely source for the C80 car length (`C80_PITCH_M`). ✔ (Wei ref. 34)
12. Cantone, "TrainDy: the new UIC software for freight train
    interoperability," *Proc IMechE F* (2011). Independent empirical air
    brake model to compare timing against.

*Fallback for braking adhesion only (European):*

13. UIC leaflet 544-1 (Brakes – braking performance); EN 14531 (railway
    braking calculations).

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
- **2026-10-04 (step-size check)** — Trains 1 and 3, against a dt = 0.005 s
  reference with the same gear stiffness k_t
  (`scripts/ltd_benchmark_dt_check.py`; runner gained `--out-dt`,
  `--k-transition`). Max change in any table quantity:

  | Run | Train 1 | Train 3 |
  |---|---|---|
  | dt 0.01, k_t 7e8 (the benchmark run) | 0.1 % | 0.3 % |
  | **dt 0.02, k_t 1e8 (production)** | **0.0 %** | **0.6 %** |
  | dt 0.02, k_t 7e8 | **+9 to +43 %** | 1.0 % |
  | k_t 1e8 vs 7e8, both dt 0.005 (model change) | 0.1 % | 0.6 % |

  - The benchmark comparison stands: the dt = 0.01 run is converged.
  - **The production setting reproduces the benchmark numbers.** k_t = 1e8 at
    dt = 0.02 matches the fine-step reference to 0.6 %, and k_t 1e8 vs 7e8
    changes the table quantities by under 1 %. The 10-14 % dissipation loss
    measured on 2026-09-26 does not show up in these outputs.
  - **k_t = 7e8 at dt = 0.02 chatters** on Train 1 (128 t wagons): a bounded
    stick-slip limit cycle, not a blow-up. Force second-difference RMS 65 kN
    against 0.7 kN at dt = 0.01; steady -317 kN reads -277 to -450 kN. The
    run finishes without error, so only a convergence check exposes it.
    Never pair the benchmark stiffness with the production step.
  - Speed and position are step-insensitive everywhere (< 0.4 km/h, < 2 m).
  - Instantaneous forces still differ at single impact instants (p99 2-15 kN,
    max ~500 kN at one moment on Train 3; the compression envelope differs by up to
    107 kN at one position). Peaks over the run are robust; exact impact timing
    is not, as with the 2026-09-26 convergence study.
  - Empties: the production gear at dt = 0.02 shows no chatter on 26 t
    wagons (same high-frequency content as dt = 0.0025), but a hard 3 m/s
    impact cascade does not converge (0.5 m/s speed difference against fine
    step, non-monotone in dt). `test_stable_on_empty_wagons_at_production_step`
    only checks boundedness and energy, so it would not catch chatter.
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
- **2026-10-06 (reading list items 2 and 3: previews only)** — The two PDFs
  in `paper/refs/papers/` are publisher previews, not the books.
  `design_simulation_heavy_trains.pdf` (48 pp.) stops partway through §2.5.1
  (~p. 30). `handbook_railway_vehicle_dynamics.pdf` (92 pp.) stops in §3.3
  (~p. 70). Neither sources a placeholder field.
  - Missing from the previews, and still needed: Spiryagin et al. §2.7.2 *Air
    Brake Systems* (p. 41) and §5.2.2 *Modelling Vehicle Inputs*
    (pp. 161–175); Handbook 2nd ed. Ch. 13, Cole, *Longitudinal Train
    Dynamics…* (pp. 457–520).
  - What the previews do give is qualitative support for the model's form.
    Handbook §3.1.4 (pp. 53–54): skidding when brake force exceeds the
    wheel-rail force (supports the adhesion cap); an empty wagon can weigh
    5× less than a laden one; shoe friction is nonlinear in speed (the model
    uses a constant, so this simplification needs stating); empty-load
    devices exist (the model assumes cars without them); non-synchronous
    braking causes run-in (F2).
  - Spiryagin et al. §2.4.3: >40 % adhesion only for a *locomotive* under
    traction control on dry rail. This is not a braking adhesion for wagons
    and must not source `adhesion`.
  - Spiryagin et al. §1.3.4 (Daqin): 210 wagons + 2 HXD1 in 2.7 km; 320
    wagons + 4 HXD1 in "almost 4 km". Wagon type is not stated, so these give
    only upper bounds of 12.9 / 12.5 m per wagon, with the locomotives
    included. They are consistent with `C80_PITCH_M` = 12.0 but do not
    source it.
- **2026-10-07 (reading list items 2 and 3: full books)** — Full copies now
  in `paper/refs/papers/` (Spiryagin et al. 2017, 465 pp.; Handbook 2nd ed.,
  915 pp.). Handbook Ch. 13 §13.2.2.5 (pp. 470–473) is the same text and
  figures as Spiryagin §5.2.2.5 (pp. 171–175): one source, not two.
  - **Force parameters: no source in either book.** Neither gives a net
    braking ratio, a shoe friction value, a design gross weight or a wagon
    braking adhesion. Brake force is "converted by means of rigging factors
    and shoe friction coefficients" with no numbers. `require_sourced` still
    refuses. Items 1 and 4 (AAR S-401/M-926, Hay) remain the route, or a
    Chinese source for C80 to match the timing.
  - Adhesion values are all *locomotive traction* (Spiryagin p. 163–164:
    ~0.20 without modern control, ~0.35 daily, 0.1 in frost/snow). Usable
    only as a bracket, not as `adhesion` for braked wagons.
  - Plausibility check, not a source: typical train deceleration
    0.1–0.6 m/s² (Handbook p. 515, citing its ref. [21]); braking rates are
    held well below the adhesion limit to avoid wheel flats.
  - Timing and logic: propagation 250–300 m/s for a well-designed system
    (sonic limit 318–349 m/s); our 176 m/s (Wei, 120 valve) is below this
    range, so state that in the thesis. Spiryagin §5.5 (pp. 212–213):
    minimum application = 30 % to wagons; service >30–100 %; emergency at
    higher cylinder pressure on AAR systems; **no graduated release** in
    Australian and North American systems (supports direct release).
  - **New measured timing data:** Spiryagin Figs. 5.67 and 5.68 (pp. 215–216,
    PDF pp. 235–236) are *operational data* for a 100-wagon train with a
    ~2000 m pipe, minimum and full-service applications, BP and BC at three
    positions (16/996/1996 m head-end; 32/968/1943 m head–tail). They are
    vector paths (~7000 segments per curve), so `digitize_air_brake.py` can
    read them. An independent, non-Chinese check on propagation speed and
    the per-metre build-up tau.
  - C80 length: not in either book (Spiryagin uses 11 m for an Australian
    ore wagon, p. 213, as an assumption).
- **2026-10-07 (Spiryagin Figs 5.67–5.68 digitized)** —
  `scripts/digitize_spiryagin_brake.py` → `simulator2/data/spiryagin2017_air_brake/`
  (`t_s,p_kPa,curve`). 5.67a/b and 5.68a are vector (filled stroke outlines,
  curves told apart by draw order); 5.68b is a raster image, BC only, and
  the two end cylinders overlap there (`BC32+1943`). All four panels were
  checked against overlays of the rendered page.
  - **Propagation: 266–273 m/s** (head-end, BC and BP onset, both
    applications), inside the books' 250–300 m/s. Wei's C80 / 120 valve gives
    176 m/s. So 176 m/s is a property of that train, or of the unsourced
    12.0 m car length, not a general value.
  - **Head build-up agrees:** BC16 full service τ 9.7 s (rms 18 kPa) against
    Wei's 10.2 s, with the same two-timescale shape (jump to ~70 kPa in
    ~1.3 s, then a ramp).
  - **Behind the head it does not:** τ 55 s at 996 m and 62 s at 1996 m;
    the linear per-metre model gives 25 / 40 s. The rise is front-loaded,
    not linear in distance: the cylinders follow the pipe, which at 90 s
    is still at 385–391 kPa at 996/1996 m against 350 kPa at the head (the
    book's "limited by the rate of pressure drop in the pipe"). With the
    pipe exhausted at both ends (5.68b) the mid-train τ is 19 s.
  - Final pressures: 125 kPa for minimum (50 kPa pipe reduction) and
    350 kPa for full service (150 kPa): about 2.4:1. Minimum ≈ 36 % of full
    service (the book's text says 30 %).
  - Caveats: the valve and railway are not named; the curves are captioned
    "operational data" but look very smooth (the books' other brake figures
    are simulations). One valve per wagon (5.67) vs one per tandem pair (5.68).
  - **Implication for `air_brake.py`:** keep the head τ. The per-metre τ
    from two points of a 1.3 km span underestimates tail lag on a 2 km
    head-end train by ~2× at mid-train. Options: (a) a saturating τ(d)
    fitted to both sources; (b) the tail cylinder tracking a pipe-pressure
    lag (closer to the mechanism). Decide before regenerating; propagation
    speed is now a choice between sources and must be stated.
- **2026-10-07 (air brake: brake pipe lag replaces the per-metre τ)** —
  Option (b) from the entry above. `simulator2/air_brake.py` now computes
  each cylinder on the command side; the RHS only adds a 0.25 s numerical
  lag on application (`actuator_tau_s`) and the 6.7 s exhaust on release.
  No RHS change; the corpora are untouched (the air brake is still not wired
  into `build_dataset.py`).
  - **Model:** the front arrives at d/c (c = 270 m/s), and quick service
    vents 0.187 of a full-service reduction there. The rest of the
    reduction diffuses along the pipe from the exhaust (linear diffusion,
    level held at the lead, no flow at the tail), solved exactly as a sum of
    first-order modes (`pipe_modes`). The control valve fills towards the
    local reduction, upward only (it laps): inshot to 0.194 at 0.130/s,
    then 0.043/s (68 kPa, then 15 kPa/s on 350 kPa). Release front, direct
    release and recharge are unchanged. Pressures are fractions of full
    service, linear in the driver's brake fraction and in force.
  - **Calibration:** `scripts/calibrate_air_brake.py` fits the five
    pipe/valve parameters through `cylinder_level` to all of Spiryagin
    Figs 5.67–5.68 (11 curves, head-end and head–tail, minimum and full
    service): **rms 15.8 kPa**, head cylinders 1.7–4.9 kPa. D = 24,800 m²/s,
    so the slowest pipe mode is 4L²/(π²D) = 65 s at 2000 m.
  - **Out of sample: Wei Fig. 7, 116th car: rms 10.4 kPa**, no refit
    (pipe ~1430 m, own front speed). The 4th car is 26.7 kPa off: the 120
    valve's head fill (fast jump, slow creep) differs from Spiryagin's.
    One diffusivity explains a 2 km and a 1.4 km train and the head–tail
    case. Linear per-metre τ could not.
  - **Residuals:** cylinders behind the head under-predict by 20–28 kPa at
    the end of the minimum applications (5.67a 996/1996, 5.68a 968) and at
    1996 m in full service (5.67b). The measured tail fills faster than
    linear diffusion allows, which may be pipe nonlinearity or vent valves.
    The "max" form of quick service was chosen over an additive blend (same
    rms, 15.8 vs 16.1) because the blend makes a spurious jump on a second
    application step.
  - **Sources:** propagation and the five pipe/valve fields now cite
    Spiryagin; the release speed, exhaust τ and recharge still cite Wei (the
    release speed via the unsourced C80 length). Unsourced: the four force
    fields. `require_sourced` still refuses.
  - **F2 re-run** (uniform 100 t cars, 15 m, full service, placeholder
    capacity): 60 cars **255 kN** peak buff, 130 cars (1.9 km) **831 kN**;
    0 kN if every car brakes at once.
  - Tests: 23 air brake tests, including pipe modes vs an independent
    finite-difference solve (head-end and vented at both ends), L² and
    venting scaling, the head valve in closed form, saturation along the
    train, a regression on Fig. 5.67b, and exact integrated impulse and
    momentum through the actuator lag. Mutation check: no pipe lag fails 2,
    no front delay 1, no valve rate limit 2. Cylinder values are held
    between 0.1 s grid points (interpolating showed brake 0.1 s early).
- **2026-10-07 (reading list item 11: Wei, Ahmadian & Zhang 2014)** — Added
  as `DRAFTJRC2014-3789-...-finalrevise.pdf`, an author draft (cite the
  published JRC2014-3789 proceedings version). **It does not give the C80
  car length**: no table, the consist is only "one HXD1 locomotive and 106
  loaded C80 railcars". `C80_PITCH_M` (and so `release_propagation_mps`)
  stays unsourced. Content is a fluid-dynamic 120-valve model with
  simulated curves only (100 kPa reduction, 330 kPa peak), so there is
  nothing to calibrate against.
  - Also added: Liu, Wang, Luo, Zhang, Liu & Zhang (2025), IEEE T-ITS
    26(8):11893. No car length either. Its Fig. 6(d) checks wave speeds
    against measurements in Wu, Cole, Spiryagin et al. (2017), "Railway air
    brake model and parallel computing scheme", J. Comput. Nonlinear Dyn.
    12(5):051017. That paper is a candidate for measured 150-wagon data and
    possibly C80 dimensions.
- **2026-10-07 (web search for the C80 length)** — No citable value found.
  - A vendor listing (Tongling Tieke, rail-wagons.com) gives "length
    13,976 mm" for a commercial "C80": not citable, not stated as length
    over couplers, and **inconsistent** with Spiryagin §1.3.4 (210 wagons +
    2 HXD1 in 2.7 km → ≤ 12.9 m per vehicle). Probably a body length or
    another variant. Do not use.
  - A 12,000 mm hit on the CRRC Qiqihar site was the C86 ore wagon; the
    CRRC site refused connections, so its C80 sheet was not checked.
  - Li et al. (2025), SV-JME 71(11-12):371-380, doi 10.5545/sv-jme.2025.1462,
    a C80 SIMPACK model: only "fixed wheelbase 8.2 m" (bogie centres), axle
    load 25 t, load 80 t. No length.
  - Next: Wu, Cole, Spiryagin, Wang, Ma & Wei (2017), "Railway air brake
    model and parallel computing scheme", J. Comput. Nonlinear Dyn.
    12(5):051017, doi 10.1115/1.4036421; or the CRRC Qiqihar C80 product
    sheet / TB standard via the library.
- **2026-10-07 (Wu, Cole, Spiryagin, Wang, Ma & Wei 2017, JCND 12(5):051017)** —
  `paper/refs/papers/air_brake_parallel_computing.pdf`. **No C80 length**,
  but it makes the C80 length unnecessary for the release speed. Measured on
  the CRRC Brake Science & Technology rig (Meishan): 150 wagon brake sets,
  600 kPa, 13.5 m of brake pipe per wagon, pipe > 2 km. Wagon type and
  valve are not named (Chinese heavy haul context, so presumably the 120
  family). Speeds are in pipe metres, so no car length enters.
  - **Release propagation, measured: 162.52 m/s** (simulated 163.31).
    Check: Table 3 release onsets 89.7 / 95.5 / 101.1 s at wagons 001/075/150
    give 11.4 s, and 149 × 13.5 m / 11.4 s = 176 m/s. That is 8 % off the
    stated figure, so the distance they used is unclear.
  - Application propagation, measured (Table 2, read column-wise): minimum
    224, full 217, emergency 254 m/s. Release/application ≈ 0.73–0.75 (Wei:
    105/176 = 0.60).
  - Quick service: pipe dip 30–40 kPa on a 170 kPa full-service reduction
    (0.18–0.24 of full). **Independent support for the fitted
    `quick_service_frac` = 0.187.** Cylinder quick-service jump ~60 kPa.
  - Table 2 also has cylinder ascending times (001/075/150): full service
    35.4 / 99.9 / 106.2 s. That is again saturating along the train
    (pipe-limited), supporting the diffusion form. Fig. 9 has the measured
    curves (minimum, full, emergency; BP and BC at 001/075/150) and could be
    digitized as a third validation set.
  - **New issue: pipe length vs train length.** Both calibration sources
    measure distance along the *pipe*: Wu has 13.5 m of pipe per wagon;
    Spiryagin has "pipe ~2000 m" for 100 wagons, and the book notes the pipe
    can be much longer than the train (p. 214). `air_brake.py` uses
    vehicle-centre distance as pipe distance, so in train metres the front
    speed and the pipe lag are optimistic by the pipe/train length ratio
    (unknown for the thesis cars; ≥ 1). State it, or add a pipe-length-per-
    vehicle factor.
- **2026-10-07 (release speed from Wu et al.; pipe length factor)** —
  `release_propagation_mps` = **162.52 m/s** (`WU2017_RELEASE_MPS`, measured,
  pipe metres) replaces Wei's 105 m/s. The C80 length no longer enters the
  model; `C80_PITCH_M` is kept only to place cars in the Wei out-of-sample
  check (`WEI2017_SPAN_M` and `WEI2017_RELEASE_DELAY_S` removed). Release now
  passes a pipe in 1/162.5 s per metre (was 1/105). Direct release stays
  exact for pipes up to ~37 km (was ~15 km of train).
  - New `pipe_length_factor` (pipe metres per metre of train, default 1.0).
    `vehicle_commands` multiplies the vehicle distances by it, so the front,
    release and pipe lag are all in pipe metres, as the sources measured.
    It is in `PHYSICAL_FIELDS` without a source: the 1.0 is an assumption,
    and `require_sourced` now lists it with the four force fields. Real
    pipes are longer (Wu: 13.5 m per wagon; Spiryagin: ~20 m per wagon on
    the 5.67 train), so the default makes braking propagate optimistically
    fast. A value for the thesis cars needs a source (AAR, or a car's
    piping drawing).
  - Calibration unchanged (it works in pipe metres directly): rms 15.8 kPa,
    Wei 116th car 10.4 kPa. A refit from the new defaults drifts inshot to
    0.216 at the same rms: inshot level and rate are weakly identified
    against each other. Defaults kept.
  - Tests: 24 air brake (new: the factor stretches front and pipe lag,
    exactly equal to scaled distances).
- **2026-10-07 (Tier 2 without the brake numbers: physics v3)** — A physics
  switch, `simulator2/physics_config.py`: `LEGACY` (data/v1-v2, unchanged:
  legacy files have exactly their old keys, fixture tests pass) and `V3` (the
  benchmark-validated configuration plus the fixes). `build_dataset.py
  --physics v3`; it refuses while `require_sourced` fails unless
  `--allow-unsourced-brake` (dry run; recorded in `build_config.json` and
  every file's metadata). v3 files carry `physics_json`, gear states, the
  locomotive table, inertia factors and brake time constants;
  `validate_dataset.py` reads them and rebuilds each model independently.
  The surrogate is untouched (item 2 decides how it carries the gear state).
  - **F7** `grade_model="exact"` (m g sin θ). **F4** modified Davis (USA),
    Handbook Table 13.3, the benchmark's Eq. 1 form incl. Q = 3.2 on the
    lead locomotive (`simulator2/vehicle_physics.py`; a loaded 120 t hopper:
    A/B/C 745 N / 13.8 / 1.64 against 600 / 12 / 0.65). **F9** inertia
    m (1 + J_f), J_f = 1.5 % wagons, 12.5 % locomotives (Handbook p. 459,
    midpoints of 1-2 % and 10-15 %).
  - **F5** loco_sd70 → SD70ACe (Spiryagin Table 2.9 + the benchmark's
    type 2 table), loco_gp38 → ES44DC (Table 2.3; P_db derived as DB_max x
    42 km/h). Notch traction and dynamic brake; fractions → nearest notch;
    `DYNAMIC_BRAKE_DESCENT` now uses the dynamic brake, not the air brake.
    Notch-up limited to one per 8 s, the benchmark's fastest (Appendix 2).
  - **F3** friction draft gear in the corpora, coupler slack 10 mm total (the
    benchmark's). F_cpl saved from the gear law with the saved gear states.
    **Integration step 0.01 s** under the draft gear: the end stop
    (k_lock 5e8) on a 26 t empty is ~200 rad/s, beyond RK4 at 0.02 s; a
    126-car train diverged at 0.02 and ran clean at 0.01 (the benchmark
    step). Non-finite scenarios are no longer written (`diverged_scenarios`).
  - **Equilibrium start.** Scenarios settle their actuators at the first
    command but started the gears unloaded: four SD70ACe at notch 6 then
    yanked ~20 m of gear travel out of a 126-car train and snatched the
    tail at 8.8-14 MN at every step size. v3 starts the gears in
    quasi-static equilibrium (`equilibrium_start`): peak 2.4 MN, converged
    across dt 0.01-0.00125.
  - **F1/F2** the air brake (`air_brake.vehicle_commands`) drives `u_brk`;
    capacities load-capped (placeholders).
  - **F6** Karnopp stiction below 0.1 m/s: friction = clip(drive + m v/τ,
    ±(brake + 34.3 N/t)), starting resistance from Handbook p. 468 (roller
    bearings, lower value). Legacy never stands still: a braked car on 1 %
    creeps back at ~3 cm/s (now a test). First version left the holding
    force unclamped, so held cars anchored a creeping train; fixed and pinned
    (Coulomb rate while sliding).
  - **F8** `simulator2/speed_planner.py`: point-mass pre-simulation along the
    route; allowed speed = lowest limit under the train, braking curve
    0.15 m/s² (Handbook p. 515 range); traction slewed off, dynamic brake,
    latched minimum air application; holds a stalled train at full service;
    no power against the brakes (notch 2 cap until the release has reached
    the tail and exhausted). v0 is capped by the limit under the whole train.
  - **Overload flag.** Index columns `coupler_over_yield` / `_ultimate`
    (AAR knuckle 1.8 / 2.9 MN, Spiryagin p. 213). The model has no coupler
    failure; flagged runs are trains that would have parted. Not dropped.
  - **Smoke corpus, 64 scenarios, 60 trains, seed 0** (validate: PASSED,
    dv/dt residual 2.7 m/s² on non-impact points; end-stop impacts are
    skipped as the 0.25 s output grid cannot resolve them):

    | | legacy (same seed) | v3, first full | v3, final |
    |---|---|---|---|
    | over limit by > 5 m/s | 11 % | 2 % | 0 % |
    | reversing < -1 m/s | 16 % | 3 % | 0 % (worst -0.47) |
    | F_max max | (Kelvin-Voigt) | 5.4 MN | 2.38 MN |
    | > 2.9 MN (would part) | | 2 | 0 (1 above yield) |
    | cost, batch 64, RTX 4090 | 231 ms/scenario | | 948 ms (4.1x) |

    The 4.1x is dt 0.01 (2x) plus the gear state and notch law; budget it
    for the regeneration. data/v2 still validates under the new validator.
  - **Still open for the regeneration:** the four brake force fields and
    `pipe_length_factor` (AAR request); the knuckle-overload policy (drop or
    keep the flagged runs); the surrogate's handling of the gear state
    (item 2). The regime library is unchanged apart from the dynamic brake
    and the notch limit; v3 trains are much stronger (850 / 632 kN starting
    vs 300 / 180 kN) with the legacy locomotive counts.
- **2026-10-08 (v3 power sizing and the overload rule)** — Two decisions
  for the regeneration, both v3 only (legacy unchanged).
  - **Locomotive counts per route** (`simulator2/power_assignment.py`,
    `PhysicsConfig.power_sizing = "ruling_grade"`). The generated consists
    were sized for the illustrative 300 / 180 kN units by an unsourced
    0.12 N/kg rule. v3 now re-sizes each scenario's power for its corridor,
    as a railroad assigns power: the fewest units whose maximum continuous
    tractive effort holds the whole train at minimum continuous speed on the
    ruling grade. Ratings from Spiryagin et al. 2017: SD70ACe 698 kN at
    14.16 km/h (Table 2.9), ES44DC 470.08 kN at 21 km/h (Table 2.3). Load =
    modified Davis at that speed (lead factor on the first unit) + m g sin θ
    for wagons and locomotives. Ruling grade = steepest climb averaged over
    the train's trailing length, over the whole corridor in the direction
    of travel (any segment can then be run). Curvature resistance is not
    included. The train keeps its layout: head-end stays head-end; a DP train
    keeps ≥ 1 head unit and splits the rest over the remote positions it had
    (as `train_generator`); one unit → head end only. More than 12 units →
    scenario skipped (none on the current corridors). The index gains
    `loco_count_manifest` and `ruling_grade_pct`; `loco_count`,
    `total_mass_kg` and `distributed_power` describe the sized consist.
    - All 500 trains × 5 corridors: mean count 2.53 → 1.66, range 1-5
      (5 SD70ACe on route_line0 at its 4 % clamp). route_line4 (net downhill
      at train length) is all single units.
  - **Knuckle overload rule** (decided before the regeneration). A run whose
    |F_cpl| peak exceeds the 2.9 MN ultimate is **not written** (the train
    would have parted; `over_knuckle_ultimate_scenarios` in
    `build_config.json`). Runs between 1.8 MN yield and 2.9 MN are **kept and
    flagged** (`coupler_over_yield`): they occur in service, dropping them
    would bias the corpus to gentle driving, and the control use needs the
    surrogate right where forces are high; report surrogate error on that
    subset separately. The rule uses |F|, so a buff peak above 2.9 MN is
    excluded too (conservative: buff capacity is not the knuckle's).
  - **Smoke corpus, same 64 scenarios, seed 0** (validate: PASSED):

    | | v3, manifest counts | v3, sized |
    |---|---|---|
    | F_max max | 2.38 MN | 1.51 MN |
    | > 1.8 MN yield / > 2.9 MN | 1 / 0 | 0 / 0 |
    | over limit > 5 m/s, reversing < -1 m/s | 0 % / 0 % | 0 % / 0 % |
    | cost, batch 64 | 948 ms | 978 ms |
    | dv/dt residual (0.25 s output) | 2.7 m/s² | 5.4 m/s² |

    Counts: 25 scenarios kept their manifest count, 39 dropped by 1-3.
    The residual rise is output resolution, not a mismatch: the worst run is
    a 130-car train on one SD70ACe in cruise, with slack oscillation through
    the train (vehicle speed spread 0.7-1 m/s) faster than the 0.25 s grid;
    rebuilt at 0.05 s output the corpus maximum falls to 0.64 m/s².
    Lighter power means more in-train content above the output Nyquist,
    which matters for what the surrogate can learn from 0.25 s samples.
  - **Caveat on the rule.** It is a minimum (tonnage-rating) sizing: a
    train can just climb the ruling grade at 14-21 km/h. On easy corridors
    that is one unit for 13,000 t, which is slow to accelerate and below what
    a railroad would assign for schedule; a horsepower-per-trailing-ton floor
    would fix that but needs a source.
- **2026-10-12 (reading list item 5: Specchia, Afshari, Shabana & Caldwell
  2012)** — `paper/refs/papers/6221471.pdf`, Proc IMechE F 227(1):19-37,
  doi 10.1177/0954409712447230. **No force parameters.** It models the 26C
  locomotive automatic brake valve (regulating, relay and cut-off valves)
  and the brake pipe (1-D isothermal continuity + momentum, FE in x, wall
  friction from a Reynolds-number factor, leakage and branch flows). The
  valve and pipe are given symbolically: no numerical tables, no results.
  No net braking ratio, shoe friction, design gross weight, adhesion or
  pipe length per car. Shoe force, the CCU and all numerical examples are
  deferred to the companion paper (its ref. 26 = reading list item 6,
  Afshari et al. 2012, 227(1):38-55), which is the one to get next.
  - Usable as a citation for the model's *form* only: a constant
    propagation speed "cannot be justified in many applications" and
    misses coupler forces in severe braking (their argument against
    Sanborn et al.). Our model no longer assumes that (diffusion lag on top
    of the front), so this supports the 2026-10-07 change.
  - The four force fields and `pipe_length_factor` remain unsourced;
    `require_sourced` still refuses.
- **2026-10-12 (brake force fields from public secondary sources)** — No
  reply yet to the AAR requests, so: public documents that quote S-401.
  Nothing wired in; `require_sourced` unchanged.
  - **NTSB, Casselton ND Train Braking Simulation Study** (docket, July
    2015; TOES simulations). §2.2.4.1 quotes the AAR definition: NBR = sum
    of *actual normal* shoe forces / actual weight on rail, full service from
    90 psi brake pipe. "The NBR for cars built after January 1, 2004 must
    fall within 11% to 14%" (S-401, its Table 8). Pre-2004 cars 8.5-13 %
    loaded, empty ≤ 38 %. Study uses **10 %** as the nominal fleet value
    ("older tank cars ... likely 9% to 10%"); BNSF ECP cars measured
    9.5-11.9 %. Car-to-car NBR variation raises in-train forces.
    Table 4: brake pipe vs train length for 4 consist lengths gives
    **pipe_length_factor = 1.081** in every case (tank car trains, 5 locos).
    Its shoe friction curve (Attachment 2, from AAR R-469, 1981) is not in
    the docket copy.
  - **Snyder, Lonsdale, McConnell & Rees (2000)**, "Testing for current
    freight car braking ratios on Union Pacific Railroad", 92nd Air Brake
    Assoc. Conf. S-401-99 table (high-friction composition shoes, 30 psi
    reduction from 90): loaded 8.5-13 % (TOFC/COFC 11-13 %), empty ≤ 38 %.
    Loaded ratio is on **gross rail load**, empty on light weight. Measured
    with Jim Shoe dynamometers at the shoe, so **rigging losses are inside
    the NBR**: the model's μ·NBR·W must not apply them again (answers the
    reading list item 1 question). Measured loaded 7.7-10.6 % on 9 in-service
    cars; aluminum hoppers with **GRL 286,000 lb** (sources
    `car_design_gross_kg` = 129,727 kg as a real rating). Adhesion demand =
    μ_shoe x NBR must stay below wet-rail adhesion.
  - **Sun, Nafis Ahmad, Persson, Spiryagin, Bernal Arango & Cole (2024)**,
    VSD 62(12):3255-3275 (CQU repository copy). Max brake force = NBR
    (S-401 table) x wagon weight, our form. "AAR formulas": shoe CoF
    **0.32 - 0.002 V/3.6 loaded**, 0.45 - 0.0015 V/3.6 empty (V km/h):
    0.32 at rest, 0.28 at 80 km/h; placeholder 0.30 sits inside. Secondary
    (AAR document not named). Wheel-rail: dry max ~0.36, wet ~0.25 (from
    Spiryagin, Polach & Cole 2013). Their Table 1 is the pre-2004 8.5-13 %
    band although it cites S-401 rev. 2018; NTSB's 11-14 % is preferred.
  - **Model-form issue found:** the empty ≤ 38 % rule. Shoe force fixed by
    GRL on a 26-30 t tare car is 0.10 x 129.7 / 26-30 = 43-50 % of tare,
    above the AAR maximum. Real light cars must carry an empty/load device,
    so "cars without them" is non-compliant for our tares. Cap empty shoe
    force at 0.38 x tare (or model the device) before regenerating.
  - With loaded NBR ≤ 14 % and μ_shoe ≤ 0.32, loaded adhesion demand is
    ≤ 0.045 and empty ≤ 0.12: the adhesion cap binds only on bad rail, so
    its value matters little once the empty cap is in.
- **2026-10-12 (brake force fields wired in; `require_sourced` passes)** —
  Values from the entry above are now the `AirBrakeParams` defaults, each
  with a per-field source: NBR 0.10 (NTSB nominal fleet), GRL 286,000 lb
  (129,727 kg), shoe friction 0.30 (mean of the AAR loaded form over
  0-80 km/h), adhesion 0.25 (wet), `pipe_length_factor` 1.081 (NTSB
  Table 4). New sourced field `max_empty_braking_ratio` = 0.38: car shoe
  force = min(NBR x GRL, 0.38 x actual mass). Capacities: loaded 286k car
  38.2 kN (0.29 m/s²), 26 t empty 29.1 kN; empty cars are now set by the
  AAR limit, not by adhesion. PDFs added to `paper/refs/papers/`
  (NTSB_2015_..., Snyder_2000_..., Sun_2024_...).
  - Tests: 25 air brake (new: empty cap, adhesion cap, all defaults
    sourced; the two timing tests now scale distance by the pipe factor).
    Full suite 352 passed. `--allow-unsourced-brake` is no longer needed.
  - Not done: NBR per car (8.5-14 %) as a randomised parameter; the F2
    run-in figures still use the old placeholder capacity.
- **2026-10-12 (item 2: how the surrogate carries the gear state — measured)** —
  v3 smoke corpus, 64 scenarios, seed 0, sourced brake (1.87 s/scenario).
  Script: `gear_state_info.py` (scratchpad; promote to `scripts/` if kept).
  Coupler samples in contact, 0.25 s steps: 4.2 M, F rms 218.7 kN; 74.6 %
  are *inside* the loop (sticking), envelope width rms 232.6 kN.
  - **Stateless edge inputs lose a quarter of the force.** Best predictor of
    F from (deflection, closing rate), binned conditional mean: residual
    **51.5 kN rms** (p99 205 kN), 24 % of F rms; from deflection alone
    60.1 kN. A lower bound (small bins overfit). No network on these
    inputs can do better: hysteresis is not a function of the present.
  - **A carried gear state recovers it.** With the true z at k and the true
    deflection at k+1: keeping z (clamped to the envelope at k+1) misses
    6.0 kN rms; the elastic return map
    `z1 = clamp(z0 + k_t (e1 - e0), lo(e1), hi(e1))` misses **1.7 kN rms**
    (p99 5.3 kN), 0.8 % of F rms, with nothing learned.
  - **Proposed form:** z becomes an edge state alongside delta. F_cpl =
    clamp(z, envelope) exactly, as `coupler_features` does for
    Kelvin-Voigt today; z advances by the return map on the predicted
    deflection, optionally plus a learned edge-head residual (the analogue
    of `ddelta_corr`). The "stateless" variant is the ablation for Ch. 6.
  - **Other v3 changes the surrogate must take before training on v3**
    (found while mapping the files): per-vehicle brake application/release
    time constants (`tau_brk_node`, `tau_brk_release_node`; `actuator_step`
    assumes one scalar tau), notch traction and dynamic brake through
    `loco_table`, `inertia_factor`, and the standstill model. Edge static
    columns k/c are Kelvin-Voigt values with no meaning under the gear.
- **2026-10-12 (item 2 implemented: surrogate on v3 corpora)** — `surrogate/`
  reads v3 files (detected from the arrays; partial v3 refused) and keeps
  the legacy path unchanged.
  - **Gear state carried, not predicted.** Edge input
    `[delta, delta_dot, F_cpl, z, lo, hi]` (+ static): `GearLaw` in
    `step.py` gives F_cpl = clamp(z, envelope) exactly; `advance` moves z
    by the elastic return map on the predicted stretch. No learned gear
    residual yet (the floor is 1.7 kN rms; add one if rollouts show it
    matters).
  - **Brake per vehicle:** `brake_step`, release tau chosen once per step
    from the mid-step command: 3.5e-4 N rms against the corpus (sub-stepping
    changes nothing). `z_trac` (lagged notch) stays exact.
  - **Node inputs +3:** locomotive force at the start and end notch through
    the notch tables (`_notch_force`), inertia factor. 20 node / 12 edge
    features (legacy 17 / 9). Checkpoints record physics and feature counts;
    old checkpoints load as legacy.
  - `_Block.carry/features/advance` is now the one path for training,
    rollout and `eval_gns1.py`. Rollouts also report coupler-force error
    (`F`, kN), legacy included.
  - **Checks:** legacy eval of `gns1_seed0` on data/v2 is bit-identical to
    the previous code on every metric. 8 new tests (`test_v3.py`, builds a
    12-scenario v3 corpus in ~25 s): true targets reproduce v, delta to
    1e-9, z_brk to < 0.05 N, gear force within the return-map floor; the
    stored F_cpl is reproduced from (delta, z); oracle rollout stays
    bounded. Surrogate suite 24 passed.
  - **Smoke training, v3 (45 train scenarios, 1500 steps): runs, learns
    poorly** (velocity 1.2x better than zero one-step, shape worse; loss
    rises once pushforward starts). Inputs are well scaled except closing
    rate (|max| 80 sd: end-stop impacts). Not a verdict: 45 scenarios
    against a legacy model trained on thousands. Judge on the regenerated
    corpus; if it persists, suspects are the impacts above the 0.25 s
    Nyquist (2026-10-08 entry) and the stateless-ablation comparison.
- **2026-10-16 (output spacing for the regeneration)** — The
  2026-10-08 entry left open whether lighter v3 power puts in-train content
  above the 0.25 s Nyquist. The v2 check (`SURROGATE_FORMULATION_NOTE.md`,
  "A negative result worth recording") does not carry over: friction gear,
  end stops and lighter power are all new. Re-measured on the same 64-scenario
  seed-0 smoke corpus, built at 0.01 s output and decimated
  (`scripts/check_output_rate.py`), so every difference is sampling. Built on
  CPU (M5 Pro, ~10 min), with the brake force values of the 2026-10-12
  entry (NTSB / Snyder / Sun; `require_sourced` passes).
  - **The 4001-sample cap.** `RandomizationConfig.max_time_samples` silently
    cut every run to its first 40 s at 0.01 s output (1000 s at 0.25 s, 200 s
    at 0.05 s). `build_dataset.py --max-time-samples` now exposes it; any
    finer output spacing needs it raised.
  - Results (63 scenarios with coupler force, 20 with a peak above 500 kN):

    | | 0.05 s | 0.10 s | 0.25 s |
    |---|---|---|---|
    | train peak missed, runs > 500 kN, worst | 0.07 % (0.4 kN) | 0.32 % | 1.12 % (6.6 kN) |
    | train peak missed, all runs, worst | 0.51 % | 7.8 % | 23 % (a 74 kN run) |
    | yield (1.8 MN) flips | 0 | 0 | 0 |
    | own peak missed, couplers > 500 kN, worst | 1.0 % | 4.5 % | 22 % (178 kN) |
    | coupler-force power above Nyquist, median / worst | 0.01 / 0.17 % | 0.04 / 0.78 % | 0.10 / 2.0 % |
    | in-train velocity power above Nyquist, median / worst | 0.03 / 0.27 % | 0.20 / 2.2 % | 0.87 / 8.9 % |
    | disk, compressed, median 324 s run | 9.0 MB | 4.6 MB | 1.85 MB |
    | per 100k scenarios | 904 GB | 457 GB | 185 GB |

  - **Where the content above Nyquist is.** Large relative misses at 0.25 s
    are on low-force runs (coast and notch-up at 25-80 kN). The 178 kN
    per-coupler miss and all 27 end-stop contacts (3 shorter than 0.25 s)
    are one run: a 132-car emergency application, slack run-in, peak 2.10 MN.
    The 0.25 s grid catches that train's peak to 0.05 %; it misses the impact
    peaks on couplers that are not the train's maximum.
  - **Recommendation: keep 0.25 s.** The quantities the control chapter and the
    knuckle rule use (the train's peak |F_cpl|) are captured to about 1 % where
    forces are high, and 5x the disk (900 GB at 100k) is not affordable. The
    stored state is complete, so the 0.25 s step is still a well-defined
    map; the cost is that impacts are hard to learn, not that they are absent.
  - Bears on the 2026-10-12 suspicion: end-stop impacts occur in 1 of 64
    smoke runs (emergency only), so they explain the closing-rate outliers but
    are unlikely to be why the 45-scenario v3 smoke model learned poorly.
    Report surrogate error on emergency runs separately.
- **2026-10-16 (speed planner: notch rate limit; reversal backstop; F2 re-run)** —
  Found while assembling Chapter 12's smoke-test table.
  - **Bug.** The v3 smoke corpus (64, seed 0, sourced brake) had one run
    reversing to **-19.97 m/s**: `route_line0_49km_run000059`, 11 vehicles,
    873 t, one SD70ACe, `throttle_modulation`, under a 5.36 m/s limit on
    grades up to 3.5 %. The governor cycled traction to stay under the limit;
    `rate_limit_notches` runs *after* the planner and cut the delivered mean
    notch from 2.75 to 1.98 (100-250 s). The train stalled; the planner's
    point mass, on the commanded notches, held 3.85 m/s, so the hold never
    triggered and the train rolled back 2 km, off the start of the route.
  - **Fix.** `physics_config.NotchLimiter`, one step of the rate limit;
    `rate_limit_notches` is it applied to an array, and the planner steps it
    on its own signed notches (traction and dynamic brake), so the two
    cannot drift. New test: a 0/8 notch command cycling every 2 s accelerates
    the planner's train no faster than steady notch 1 (fails on the old
    planner: 11.2 against 1.5 m/s). The notch *lag* is still not in the
    planner.
  - **Backstop.** `build_dataset.py` (v3 only) does not write a run with
    `v_min_mps < REVERSAL_LIMIT_MPS` (-1.0, the validator's threshold);
    `reversed_scenarios` in `build_config.json`.
  - **Smoke corpus after the fix** (same 64; validate PASSED): reversing 0
    (worst -0.69 m/s), over limit > 5 m/s 0, worst overspeed 1.6 m/s,
    F_max max 2.10 MN (the `emergency` run, 132 cars, kept and flagged above
    yield), mean locomotives 1.72, nothing refused. Legacy, same seed: 19 %
    over by > 5 m/s, 28 % reversing (worst -19.2), F_max max 1.40 MN.
  - **F2 re-run with the sourced capacity** (`scripts/air_brake_run_in.py`;
    100 t cars, 15 m, full service from 20 m/s, pipe factor 1.081). Peak buff,
    every car at once / brake pipe: 60 cars Kelvin-Voigt 0 / 281 kN, draft
    gear 0 / 178 kN; 130 cars Kelvin-Voigt 0 / 828 kN, draft gear 0 / 702 kN.
    The 2026-10-12 caveat (run-in figures on the placeholder capacity) is
    closed.
  - **Calibration figure** `paper/scripts/plot_air_brake.py`: defaults give
    rms 15.9 kPa over the 11 curves (head 2.3-5.3 kPa), Wei 116th car
    10.5 kPa, 4th car 26.7 kPa.
