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

    python -u build_dataset.py --seeds seeds/seeds_pueblo_region.json \
        --out data_pueblo --seed-radius-km 30 --max-radius-km 250 \
        --sleep 1.5 --min-len-km 5

``-u`` matters: piped stdout is block-buffered, so without it a running
crawl looks identical to a hung one for many minutes.

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

## Phase B revised: NARN replaces the Overpass crawl

_Changed 2026-09-17 after the crawl failed. This supersedes the seed-and-crawl
approach described above; the seed file and `routegen.crawl` are kept because
OSM is still the only source of line speed._

### The crawl did not work, and my endpoint advice caused it

The run above failed five consecutive seeds with `HTTP 504 (busy)` over 4.5
hours, producing only `.error` files. It looked alive because it kept advancing
the seed counter.

The cause was the endpoint recommendation two sections up. That table timed
**one 8 km tile** and concluded overpass-api.de was 74x faster than the kumi
mirror. A single query does not predict a sustained crawl: the main instance
answers one request quickly and throttles hard once a crawl starts issuing
hundreds. **Benchmark the workload, not a sample of it.** The earlier table is
left in place because the numbers are real; what was wrong was the conclusion
drawn from them.

### The better source

USDOT/BTS publish the **North American Rail Network** as a queryable feature
service. It is a better fit than OSM on every axis that matters here:

| | Overpass crawl | NARN |
|---|---|---|
| topology | inferred by stitching heuristics | **explicit** (`FRFRANODE`/`TOFRANODE`) |
| junk rail | filtered by tag heuristics | **pre-classified** by `NET` |
| coverage | per-seed crawl, 250 km radius | **whole continent**, 302,771 segments |
| rate limit | 504s under load | none observed |
| Colorado main line | 5 seeds failed in 4.5 h | **1,307 segments, 4,407 km, 9 s** |
| corridor identity | `route_line2` | `MOFFAT TUNNEL`, owner `UP` |

`NET` classifies every segment, which is exactly the "ignore the junk rail"
rule, already applied by the people who own the data:

| code | meaning | segments |
|---|---|---|
| **M** | **Main sub network** | **95,936** |
| O | Other track (minor industrial leads) | 88,561 |
| Y | Yard tracks | 79,028 |
| I | Major industrial lead | 16,134 |
| S | Passing sidings over 4000 ft | 10,193 |
| X / A / R | Out of service, abandoned, removed | 11,431 |
| T | Trail on former right-of-way | 1,473 |
| F | Rail ferry connection | 15 |

Per-segment attributes the OSM path never had: owner (`RROWNER1`), subdivision
and branch names, track count, passenger flag, state, mileage. Corridors get
real identities, which matters for a thesis that wants to name the route its
control demo runs on.

### What NARN does not give

**Line speed.** `v_max(s)` comes from OSM `maxspeed`, so `routegen.osm` stays in
the pipeline for that, queried per corridor rather than crawled. Until then the
FRA class fallback in `derive_profile` applies.

**Elevation.** Unchanged: 3DEP, which was never the bottleneck. Its cost at
continental scale is the open question -- point queries run ~6 s, so batching is
required before this scales past a region. That is the next thing to measure.

**Correction, 2026-09-20: 3DEP does not cover the non-US network.** That is a
gap in NARN's usefulness, not just in its attributes: 935 Canadian and 336
Mexican corridors (23% of the total) have no usable elevation from this source.
Measured on a 60-corridor sample, 0 of 48 US corridors failed the raw-elevation
QA gate against 5 of 9 Canadian and 1 of 3 Mexican. Details below.

### Acquisition

`scripts/pull_narn.py`. Fetches object ids for a server-side filter, then pulls
geometry in chunks **by POST** -- an id list of a few hundred makes a URL long
enough that the service answers 404, which reads like a wrong endpoint rather
than an oversized request. Writes `segments.parquet` (one row per segment, with
a vertex slice) and `geometry.npz` (concatenated coordinates plus offsets),
which is far smaller and faster to load than per-segment GeoJSON.

    python scripts/pull_narn.py --out data/narn --where "NET='M'"

### What this does not change

The corridor selection *rules* from the original plan still hold, and matter
more now that there is more to choose from:

- **Pick the steep holdout on purpose**, verified against the raw-elevation QA
  gate. Raton and Tennessee Pass are in the seed list for that reason and are
  identifiable in NARN by subdivision name.
- **Reserve `control_eval` before generating anything**, in writing.

---

## The continental network, and what it costs to map

_Measured 2026-09-17. `scripts/pull_narn.py`, `scripts/bench_route_pipeline.py`._

### What was pulled

    95,936 segments   274,145 km   2,719,766 vertices   35 MB   in 13.5 min

| | |
|---|---|
| countries | US 81,533 · Canada 12,455 · Mexico 1,948 segments |
| graph | 92,787 nodes, 95,936 edges |
| named subdivisions | 2,492 |
| native vertex spacing | ~101 m |
| owners by km | UP 40,204 · BNSF 38,817 · CN 30,533 · CPKC 26,136 · CSXT 25,782 · NS 23,022 |

### The pipeline, confirmed

The four stages are the existing `routegen` ones, unchanged:

1. **resample** — alignment to a uniform chainage grid at `ds` metres.
2. **elevate** — 3DEP `getSamples`.
3. **derive_profile** — Savitzky-Golay grade, planform curvature, `v_max`.
4. **export** — the four-channel tensor `.npz` the simulator consumes.

Only stage 2 costs anything. Projected onto the real 274,145 km at 10 m
spacing, which is 27.4M points:

| stage | rate | hours |
|---|---|---|
| resample | 625 km/s | 0.12 |
| **elevation** | **16,427 pts/s** | **0.46** |
| profile | 4,264 km/s | 0.02 |
| **total** | | **0.60** |

**The entire North American main-line network, in simulator-ready form, is
about 36 minutes.**

> **Superseded 2026-09-20.** The elevation rate above does not reproduce;
> measured throughput plateaus near 1,700 pts/s burst and 421 pts/s sustained,
> which puts the continental total at 4.5-18 h rather than 36 minutes. See
> "Elevation throughput does not reproduce" below. The resample and profile
> rates are unaffected and were re-confirmed.

### Getting there needed two changes, and one of them I got wrong first

`BATCH_CHUNK` was 100 and chunks were issued **sequentially**; only the EPQS
*fallback* was threaded. Measured throughput against chunk size on real
alignment vertices:

| points/request | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| points/s | 435 | 929 | 1,385 | **2,729** | truncates to 1000 |

Latency is nearly flat in chunk size — 0.23 s at 100 points, 0.37 s at 1000 —
so the old value spent almost all its time on per-request overhead. 1000 is the
service ceiling, not a guess: at 2000 it silently returns only 1000.

Concurrency at 1000 points/request: 1 worker 2,196 pts/s, 4 → 5,785, 8 →
11,829, **16 → 16,427**. The request is latency-bound, so workers scale nearly
linearly.

Together: **17.5 hours → 0.46 hours** for continental elevation.

**The mistake worth recording.** The first attempt at this raised `BATCH_CHUNK`
to 1000 and made it concurrent, and made the pipeline *600x slower* — 4 pts/s,
with every point falling through to the per-point EPQS path.
`_sample_chunk_imageserver` issues a **GET**, and a 1000-point geometry is
~30 KB of JSON, far past what a URL carries; the service rejects it. The
benchmark had used POST and so never saw it.

This is the same URL-length failure as `scripts/pull_narn.py`, which had been
diagnosed and fixed an hour earlier and not carried across. Both are now POST.

What made it *silent* was worse than the bug: the concurrent wrapper caught
every exception and returned `None` for the chunk, which is indistinguishable
from a working batch path except for being 600x slower. Chunk failures are now
collected and raised as a `RuntimeWarning` naming the first error. **A fallback
path that silently absorbs a total failure of the primary is not a fallback, it
is a way to not find out.**

### The stage that did not exist — traversal (done 2026-09-20)

NARN gives a **graph**, not routes: 95,936 segments averaging 2.9 km, joined at
92,787 nodes. `routegen/narn.py` is the stage between that and `resample`, and
it is deliberately thin. NARN segments are adapted into `Way` objects, so the
corridor walk already written for the Overpass area pull
(`network._grow_corridor`) applies unchanged. Driver:
`scripts/build_narn_corridors.py`. Tests: `route_generator/tests/test_narn.py`,
16 of them, all offline.

NARN turned out to be *better* input than OSM for this, not merely different.
`FRFRANODE`/`TOFRANODE` are authoritative, so the coordinate-fallback and
gap-bridging heuristics in `centerline.py` never fire — connectivity is read,
not inferred. The whole continental traversal is **93 s**, offline.

#### The decisions, and why

**Unit of traversal: `(RROWNER1, SUBDIV)`, not `SUBDIV`.** 2,492 subdivision
names but 3,212 owner+name groups, so names are reused across railroads and the
name alone would weld unrelated track together. Median group is 33.5 km, which
already sits inside the 22–49 km band the old five-corridor set occupied.

**Geometry direction was verified, not assumed.** Vertices run
`FRFRANODE → TOFRANODE`: checked on 4,994 junction pairs — segment A's last
vertex against segment B's first, where A ends at the node B starts from — with
zero exceptions. Without that the node ids and the coordinates could disagree
and `Way.reversed()` would silently corrupt a corridor.

**A branch ends a chain; it is not resolved.** 1,138 of 3,212 groups contain a
degree-≥3 node. Both arms come out as separate corridors with real identities.
Picking "the" through-route would put a heuristic back into a pipeline whose
whole justification was that NARN states topology explicitly.

**Disconnected components are never stitched.** 430 groups are multi-component.
These are either a reused subdivision name in two places or a genuine hole in
NARN's geometry; the walk cannot tell, and cannot cross a gap, so both come out
as separate corridors. That is the right answer either way, but it does mean two
unrelated corridors can carry the same base name with different piece indices.

**Segments with no `SUBDIV` are excluded.** 5,230 segments, 10,433 km. They
carry no parent-line identity, so a corridor built from them could not be named
or reserved for a split. The cost is real: a corridor terminates early wherever
such trackage would have connected it.

**Cut long chains; drop short ones.** `min_length_m = 15,000`, because
`DATA_SCHEMA` wants 300–600 s runs and the randomizer clips duration so the
consist cannot leave the route — below ~15 km a full-length scenario does not
fit. `max_length_m = 60,000`, because the whole route array is resident per
scenario and disk, not compute, binds at Phase C scale.

**Cuts land on segment boundaries — with a vertex-level fallback.** Cutting at a
way boundary keeps `way_id` provenance intact and leaves every piece a whole
number of NARN segments, traceable to its `FRAARCID` set. The first
implementation did *only* that, and **337 of 5,564 corridors came out over the
cap, up to 178 km**, because NARN's segment-length tail is long: remote Canadian
and Mexican route runs to 178 km in a single segment, and a single segment has
no interior boundary to cut at. `split_way` now subdivides any segment longer
than a quarter of the cap at its own vertices, keeping the parent `FRAARCID`,
and the cap is enforced as a hard bound rather than a target. Result: **0 over
cap**. A two-vertex segment still cannot be split and is returned whole — none
exist at this scale, but the function does not pretend otherwise.

**Track class from `TRACKS` and `PASSNGR`.** NARN has no line speed, so
`derive_profile` falls back to an FRA freight class. Single track keeps **class
4 (60 mph)** — exactly the fallback `data/v1` and `data/v2` were built with, so
single-track `route_vmax`, and the overspeed statistics measured against it,
stay comparable across the rebuild. Two or more tracks, or any passenger code,
promotes to class 5 (80 mph); an absent or zero track count demotes to class 3
(40 mph), because an unknown track count is not evidence of a fast railroad.
This is the one place the traversal invents physical information, and it sets
`route_vmax` everywhere.

**Structure flags are all False.** NARN carries no
bridge/tunnel/cutting/embankment tags. They are provenance only — nothing in the
RHS reads them — so they are left absent rather than guessed. See the tunnel
finding below for why their absence nonetheless costs something.

#### What came out

| | |
|---|---|
| corridors | **5,621** — US 4,350 · Canada 935 · Mexico 336 |
| track retained | 248,892 km of 274,145 |
| length | min 15.0 km, median 47.6 km, max 60.0 km, none over cap |
| groups | 3,212 — 1,856 simple paths, 1,138 branched, 430 multi-component |
| chains grown | 8,522, of which 1,423 cut and 5,850 dropped as too short |
| traversal time | 93 s, offline |

The km balance closes: 248,892 retained + 14,594 dropped short + 10,433 with no
subdivision = 273,919 against NARN's own 274,145, a 0.08% residual between the
`KM` attribute and a haversine recomputation of the geometry.

Corridors have real identities at last: `bnsf_raton_p0_56km`,
`up_moffat_tunnel_p2_56km`, `rrrr_tennessee_pass_p0_39km`, `bnsf_cajon_p0_52km`.
5,621 is far more than the splits need. **Corridor supply is no longer the
constraint.**

---

### Carrying corridors through the pipeline

`scripts/build_narn_routes.py` runs corridors through the four existing stages
unchanged and writes the DATA_SCHEMA §F route tensor. The Phase A QA gate is
applied **between** elevation and profile — it reads sampled elevation, so it
cannot gate ahead of the spend — and a failing corridor is recorded and skipped
rather than smoothed into range.

Measured on a 12-corridor pilot (hand-picked to span the topology cases and both
extremes of terrain) and a 60-corridor deterministic sample spread evenly
through the index.

#### Elevation throughput does not reproduce

The 16,427 pts/s recorded above on 2026-09-17, and the "36 minutes for the whole
continent" that follows from it, **do not reproduce**. Measured 2026-09-20 on
the identical code path — `method="batch"`, zero EPQS fallback, so this is not
the silent-fallback bug returning:

| points in one call | 1,000 | 5,000 | 20,000 | 60,732 |
|---|---|---|---|---|
| pts/s | 204 | 596 | 1,367 | **1,680** |

Throughput rises with batch size and plateaus near 1,700 pts/s. Raising
`batch_workers` from 16 to 64 buys only 1.5x (1,287 → 1,952 pts/s on 20,000
points), so the service is rate-limiting rather than latency-bound — the
opposite of the regime the 16-worker setting was tuned in. Sustained throughput
is lower still: the 60-corridor run averaged **421 pts/s** over 679 s.

**Revised continental projection: 4.5 h at the burst rate, ~18 h at the
sustained one — not 0.46 h.** I have not established the cause and am not
claiming a regression in our code; the honest statement is that the earlier
figure is not reproducible today, and the earlier note's own instruction to
re-measure before trusting a throughput number applies to its own number too.

This does not block anything: a few hundred corridors, which is all the splits
need, is 10–20 minutes. But because throughput depends on batch size, sampling
per corridor (~5,000 points) runs at roughly half the plateau rate. **Batching
elevation across corridors is the obvious next optimization and is not done.**

#### 3DEP is a US dataset, and the rejections say so

The 60-corridor sample rejected 6, a 10% rate. Split by country it is not a rate
at all, it is a boundary:

| country | sampled | rejected |
|---|---|---|
| US | 48 | **0** |
| Canada | 9 | **5** (56%) |
| Mexico | 3 | **1** (33%) |

3DEP is a USGS product. The non-US network — 935 Canadian and 336 Mexican
corridors, 23% of the total — has no usable elevation from this source, and the
QA gate is the thing that noticed. **Restrict the build to `COUNTRY == 'US'`
unless a non-US DEM is sourced** (Canada's CDEM is the obvious candidate). That
still leaves 4,350 corridors, which is far more than needed, so this costs
nothing in practice — but it must be stated rather than left as an unexplained
10% failure rate, and it means the "North American" network is a US network
until a second DEM is added.

#### The tunnel finding

The raw-elevation gate also rejects corridors it was not designed for. In Phase
A it caught `route_line2`, whose DEM data was simply broken. On US mountain
route it is catching **tunnels**, where 3DEP correctly returns the mountain
surface above the bore instead of the track inside it.

The signature is unmistakable and it is not noise. On `up_moffat_tunnel_p0_56km`
(6.0% of steps over 1 m, against a 5% tolerance) the jumps are not spread along
the corridor — 224 of 336 fall in one 5.7 km stretch spanning 152 m of
elevation, and 71 more in a second 2.6 km stretch spanning 88 m. That is a
surface profile over a bore, not a track profile.

Rejected on this basis: both Tennessee Pass pieces, and Moffat Tunnel pieces p0,
p3 and p4. These are exactly the steep, tunnelled mountain corridors the
`ood_grade` split wants. **The gate is behaving correctly and the data is
genuinely unusable as sampled** — but "reject it" is the wrong remedy when the
cause is a tunnel rather than bad data, because it systematically removes
mountain route. Options, none taken yet: interpolate elevation linearly across a
detected tunnel run, source tunnel portal elevations, or accept the corridor
with the tunnel stretch flagged. This needs a decision before `ood_grade` is
built.

#### The 2.5% clip binds on real mountain grade

Phase A cut `grade_clip` from 4% to 2.5% and chose an 800 m smoothing window. On
the flat-to-rolling corridors that survive the gate the settings are excellent —
across the 54 passing corridors of the sample, median rms grade **0.45%**, p90
0.79%, worst 0.97%, and the clamp binds on more than 1% of length for only 3 of
54 (worst 3.63%). Against `route_line2` sitting at the old 4% clamp for 22.7% of
its length, that is the Phase A fix working.

But **`bnsf_raton_p3_56km` — the Raton Pass summit piece — sits at the 2.5%
clamp for 25.4% of its length** (rms 1.73%, p95 2.50%). That is `route_line2`'s
pathology with the opposite cause: Raton's ruling grade is genuinely about 3.5%,
so the clip is truncating real terrain rather than DEM noise.
`bnsf_cajon_p1_29km` (rms 1.92%) and `up_moffat_tunnel_p1_51km` come close
without binding much.

`grade_clip = 0.025` was selected against a corridor set that contained no real
mountain grade, and should not be carried onto one that does. The clip and the
raw-elevation tolerance both need re-tuning against the corridors that matter
for `ood_grade` — the same `tune_grade_smoothing.py` exercise, run on a mountain
sample instead of the Pueblo five.

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
