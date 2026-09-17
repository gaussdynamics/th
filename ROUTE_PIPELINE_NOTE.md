# Route Pipeline — grade field, corridor QA, and the corridor expansion

_Written 2026-09-17. Companion to `NEXT_STEPS.md` item 1 and
`DATASET_BUILD_REPORT.md`. Phase A is done and measured; Phases B and C are
planned and scoped here._

---

## Correction first: the pipeline is not offline

`NEXT_STEPS.md` item 1 says the route pipeline "needs OSM + DEM network access,
which the device bridge does not have", and that line was repeated into several
later documents without being tested. **It is wrong.** Both services are
reachable:

- USGS 3DEP point query: OK, ~6 s, returns 1418.7 m at Pueblo CO.
- Overpass: OK through `routegen.osm`, which sets the descriptive User-Agent the
  OSM usage policy expects. A bare request gets HTTP 406, which is what the
  original note probably saw and read as "no network".

Nothing about corridor acquisition was ever blocked. That assumption cost real
time, and it is the reason this note leads with it.

---

## Phase A — the grade field (done)

### The bug was one corridor, and it was not a smoothing problem

Grade comes from a Savitzky-Golay fit through DEM samples. The obvious reading
of `route_line2`'s 2.46% rms grade and 22.7% clip-binding was under-smoothing.
It is not. The **raw elevation** is broken before any smoothing is applied:

| corridor | median Δz per 10 m step | steps jumping > 1 m | 2nd-difference roughness |
|---|---|---|---|
| route_line0 | 0.077 m | 0.7% | 0.021 m |
| **route_line2** | **0.234 m** | **19.3%** | **0.143 m** |
| route_line3 | 0.047 m | 1.5% | 0.040 m |
| route_line4 | 0.039 m | 1.1% | 0.036 m |
| route_line5 | 0.045 m | 1.0% | 0.041 m |

Real track cannot change elevation by more than a metre across 19% of its 10 m
segments — that is a 10% grade on a fifth of the line. The samples are not on
the track: either the OSM centreline has drifted off the alignment, or the DEM
does not resolve a cut or fill and is reading the ground beside it. Its
bridge/tunnel fraction is only 0.9%, so structure correction is not the cause.

**No smoothing window repairs that.** Smoothing wrong numbers averages them.
Chasing it with a wider window is how the sweep first landed on 1200 m — a
value that would have flattened every *healthy* corridor to rescue one broken
one, at the cost of erasing real terrain.

So `scripts/tune_grade_smoothing.py` now applies a raw-elevation QA gate
(`RAW_JUMP_TOL = 5%` of steps over 1 m) **before** choosing a window, and
selects over corridors that pass. `route_line2` is rejected.

### Settings chosen

| parameter | was | now | why |
|---|---|---|---|
| `smooth_window_m` | 200 | **800** | narrowest window where the clip stops binding on every corridor that passes QA |
| `grade_clip` | 0.04 | **0.025** | 4% is far above freight ruling grades and let a broken corridor through unnoticed for months; the clip is a tripwire, not a filter |
| corridors | 5 | **4** | `route_line2` fails elevation QA |

Resulting grade fields (`route_generator/route_profiles_v3/`):

| corridor | rms | p95 | max | at clip |
|---|---|---|---|---|
| route_line0_49km | 0.86% | 1.45% | 2.50% | 1.0% |
| route_line3_29km | 0.72% | 1.25% | 1.47% | 0.0% |
| route_line4_27km | 0.39% | 0.75% | 1.23% | 0.0% |
| route_line5_22km | 0.58% | 0.87% | 1.17% | 0.0% |

`route_line0` still touches the clip on 1.0% of its length, in short runs
(median 135 m) reaching 5% unclipped. Real ruling grades hold for kilometres,
so these are residual artifacts rather than terrain — under the tolerance, but
worth a targeted fix later: **a grade excursion shorter than a few hundred
metres is not a ruling grade**, and a minimum-length filter would be more
physical than more smoothing.

### The overspeed prediction was half right

The prediction was that fixing the grade field would largely clear the
overspeed population. 400 scenarios built on the new corridors, against the old
corpus restricted to the same four corridors so the comparison is like for
like:

| corpus | n | never over | > 5 m/s | > 20 m/s | worst | reversing |
|---|---|---|---|---|---|---|
| `data/v2`, all 5 corridors | 10000 | 72.9% | 17.8% | 2.4% | 67.8 m/s | 22.0% |
| `data/v2`, excluding route_line2 | 8001 | 77.8% | 13.3% | 1.1% | 36.5 m/s | 24.2% |
| **new grade field** | 400 | **80.8%** | **13.5%** | 1.5% | **25.7 m/s** | 23.0% |

**Confirmed:** the catastrophic tail is gone. Worst case falls 67.8 → 25.7 m/s,
and the 245 km/h freight trains were route_line2's artifact grades.

**Refuted:** the moderate population is untouched. Like for like, >5 m/s goes
13.3% → 13.5%. Reversals are unchanged at ~23%.

So there is a second cause, and it is not the terrain. On a sustained 1%
downgrade the grade force on a 100 t car is ~9.8 kN against ~1.6 kN of Davis
resistance at 30 m/s — a coasting train accelerates, correctly, without bound.
The real cause is that **control profiles are sampled independently of the
terrain they run on**: `dynamic_brake_descent` is one of eleven regimes drawn by
weight, not because the train is actually on a descent. A train that coasts
down a real grade is faithful physics applied to an operationally implausible
command sequence.

That is a randomizer fix, not a route fix, and it belongs with the corridor
rebuild rather than before it. Options, cheapest first: condition regime
sampling on the grade ahead of the start chainage; or add a light "driver"
that applies dynamic brake when the speed limit is approached; or keep them and
label them as hard cases. The fields to filter on are already recorded per
scenario in `index.parquet`.

---

## Phase B — corridor acquisition (in progress)

### Scope

The target is no longer "15-25 corridors". It is **the major freight network
across a wide multi-state area centred on Pueblo CO**, with the whole North
American network as the eventual goal. The reasoning is that a control system
trained on a real, connected network is closer to something an operator could
actually use than one trained on five hand-picked lines — and the acquisition
cost scales far better than the hand-picked approach did.

Junk rail is excluded by the existing `main_only` filter in `routegen.osm`,
which drops any way with a `service` tag (sidings, spurs, yards, crossovers)
and `usage` of industrial, tourism or military. That is the "ignore the junk"
rule already implemented; it keeps untagged main and branch lines.

### Seed set

`route_generator/seeds/seeds_pueblo_region.json`: 22 seeds within 900 km of
Pueblo — 13 from the existing national list plus 9 added for this region.

The additions are deliberate. **Raton Pass** (Trinidad CO / Raton NM) and
**Tennessee Pass** (Minturn CO) are among the steepest Class I freight grades in
North America. The current corpus has no honest steep corridor at all, which is
why `ood_grade` ended up built on measurement noise. These are the real thing.
La Junta, Dodge City and Amarillo sit on the BNSF Transcon; Alliance NE covers
Powder River Basin coal.

### Running it

    python build_dataset.py --seeds seeds/seeds_pueblo_region.json \
        --out data_pueblo --endpoint https://overpass.kumi.systems/api/interpreter \
        --seed-radius-km 30 --max-radius-km 250 --sleep 1.5 --min-len-km 5

Caches per seed under `raw/` and resumes, so it can be stopped and restarted,
and new seeds cost only their own crawl.

**Use the default endpoint, and do not trust the mirror advice in
`build_dataset.py`'s docstring.** It suggests kumi.systems as the permissive
option. Measured on one 8 km tile, all returning the identical 122 ways:

| endpoint | time |
|---|---|
| overpass-api.de (default) | **4.3 s** |
| maps.mail.ru | 44.5 s |
| overpass.private.coffee | 185 s |
| overpass.kumi.systems | **319 s** |

Following that docstring cost an hour of a crawl that appeared to hang and was
in fact running 74x slower than necessary. Mirror performance drifts; re-measure
rather than inheriting a recommendation.

### Two rules to hold to

**Pick the steep holdout on purpose.** `ood_grade` must be a corridor that is
genuinely steep, verified against the raw-elevation QA gate above, not whichever
corridor the noise inflated. That is exactly how the current corpus went wrong.

**Reserve `control_eval` before generating anything.** Decide which corridors
are held out for the control demo in writing, first. A held-out set chosen after
seeing results is not held out.

---

## Phase C — corpus rebuild, at a larger scale

10,000 scenarios took 25 minutes of GPU time. That was never the constraint, and
with a wider corridor set there is no reason to stay there. **Plan for
50,000-100,000 scenarios**, which is 2-4 hours of integration.

What actually binds at that scale is **disk, not compute**: `data/v2` is 12.4 GB
for 10,000 scenarios, so 100,000 is ~124 GB compressed. Worth deciding
deliberately before building — options include dropping `edge_dynamic` for
scenarios not used in training (it is derivable, though see the precision
caveat in `SURROGATE_FORMULATION_NOTE.md`), shorter durations for part of the
corpus, or simply provisioning the disk.

What the larger corpus buys, in order:

1. **A populated `control_eval` split** — Chapter 7 becomes possible at all.
2. **An honest `ood_grade`** on real mountain grade.
3. **Many more corridors in training**, so "generalizes to unseen corridors"
   rests on more than a three-to-one split in one region.
4. **Headroom for the surrogate**, which is not near convergence: doubling
   training steps nearly halved the 200-step rollout error.

---

## What changes for existing results

Every surrogate number so far was measured on `data/v2`. After a rebuild they
are numbers from a different dataset and must be re-run, not carried over. This
needs stating in the thesis the same way the seeding-reproducibility bug was
stated in `DATASET_BUILD_REPORT.md` — bluntly, with a list of what predates the
rebuild.

Specifically superseded by a rebuild: the one-step accuracy tables, the rollout
horizons, and the `ood_size` generalization figures. Not superseded: the
formulation decisions in `SURROGATE_FORMULATION_NOTE.md`, which are properties
of the physics and the step size rather than of any particular corpus.
