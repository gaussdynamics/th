# Route Generator

Builds the **research-grade route dataset** for the LTD surrogate (see
`../PROJECT_DIRECTION.md` and `../DATA_SCHEMA.md`). Routes are acquired from
OpenStreetMap, resampled to a uniform arc-length grid, fused with USGS 3DEP
elevation, and reduced to the grade / curvature / speed-limit profile the
`simulator2` dynamics consume.

The pipeline runs as four stages, surfaced as four tabs in the Streamlit app and
as a folder per stage:

```
acquire → saved_routes/   ─►  resample → resampled_routes/  ─►
  elevation → elevated_routes/  ─►  profile → route_profiles/  (.geojson + .npz)
  ─►  export bin → route_tensors_export.zip  (routes/*.npz + index.csv + manifest)
```

All controls live on each tab's page (no sidebar); tab 1 also carries the route
library / pipeline-status panel and the relation batch.

Built to run **in batches** — you fetch and vet routes one at a time, collect the
good ones, and generate the dataset for that batch rather than the whole US rail
network at once.

## Status — Part 1: Acquisition

**Selection is area-first.** Railway routes are *not* tidy closed objects in OSM:
track is mapped as open `railway=rail` ways split into many segments, and route
*relations* (which have IDs) are incomplete for US freight and don't guarantee a
single clean corridor. So the primary workflow is: pick a bounding box → pull all
rail → auto-split the network into corridors. Relation-ID fetch is kept as a
secondary option for the cases where a good relation exists.

Implemented:

- **Area fetch + corridor extraction** (`routegen/osm.py` `fetch_area`/`fetch_radius`,
  `routegen/network.py` `extract_corridors`) — pull all track ways in a **bounding
  box** or a **radius around a point** (`around` filter; ways returned un-clipped),
  then split the resulting graph into corridors: maximal paths through *continuation*
  nodes (degree 2), breaking at *line-ends* (degree 1) and *junctions* (degree ≥ 3).
- **Bounded outward crawl** (`routegen/crawl.py` `crawl_network`) — optional in radius
  mode: seed from the ball, then follow each line's *open ends* beyond the radius
  (querying Overpass just past them) until lines reach natural termini or a hard cap
  (max distance from centre / iterations / way count). This captures lines that only
  *pass through* the ball in full, while the cap prevents the crawl from flooding the
  fully-connected national network.
- **Parent-line grouping** (`routegen/network.py` `group_corridors`) — OSM splits a
  physical line into many ways (at junctions, bridges, tag changes) and sidings
  create extra junctions, so a long route fragments into pieces. Grouping re-links
  collinear fragments — across both junctions and micro-gaps (coincident-but-
  unshared endpoints) — by endpoint proximity + heading continuity, assigning each
  fragment a shared `line_group` and a `group_order`. **Nothing is merged or
  dropped**: fragments stay separate and reversible, just labelled so they can be
  stitched downstream.
- **Service-track flagging** — ways tagged `service=siding/spur/yard/crossover` are
  kept but marked (`is_service`, `service_kind`) and never folded into a mainline
  group. They show muted on the map and can be included/excluded per batch.
- **OSM relation fetch** (`routegen/osm.py` `fetch_relation`) — Overpass query for a
  route relation: tags + member ways with geometry and node ids.
- **Robust centerline assembly** (`routegen/centerline.py`) — stitches unordered /
  reversed member ways into one ordered polyline by shared endpoint nodes, with a
  coordinate fallback; **detects gaps and branches**; de-duplicates junction
  vertices; propagates per-vertex tags (maxspeed, bridge, tunnel, cutting,
  embankment); computes geodesic arc length. Shared with corridor extraction.
- **Streamlit UI** (`app/streamlit_app.py`) — Area mode: enter a bbox, fetch, see all
  corridors color-coded on a map, browse a sortable table, inspect any corridor's QA
  (gaps, branches, speed-limit coverage, bridge/tunnel fractions), and select which
  corridors to add to a batch. Relation mode: single-route fetch + inspect. Both
  export GeoJSON/CSV and feed a downloadable batch manifest.

## Status — Part 1.5: Resample (`routegen/resample.py`)

Rewrites a stored route onto a fixed-Δs chainage grid (default 10 m) so the
downstream DEM query gets uniform coverage on tangents — OSM digitizes *adaptively
for horizontal curvature* (dense on curves, sparse on straights), which
under-samples the **vertical** profile exactly where grade lives. Geometry +
feature bookkeeping only; no smoothing or grade.

- **Uniform resample** of `(lon, lat)` by linear interpolation along arc length.
- **Step features held, not ramped** — `maxspeed_kph`, `way_id` and the
  `is_bridge/tunnel/cutting/embankment` flags are piecewise-constant, so a
  resampled point inherits the value of the source segment it falls in.
- **Transition snapping** — inserts a sample exactly where a step feature changes,
  so a bridge entrance or speed-limit step is not smeared by up to a cell width.
- **Validation + provenance** — drops coincident vertices, enforces strictly
  increasing `s`, and tags each point with its source-segment length and an
  `is_long_segment` flag (interpolated across a sparse OSM span / bridged gap).

## Status — Part 2: Elevation (`routegen/elevation.py`)

Attaches a USGS 3DEP ground elevation to every sample → the set of 3-D points
`(lon, lat, z)` that approximate the route.

- **Batch + fallback** — the 3DEP ImageServer `getSamples` endpoint returns many
  points per request (chunked), with the per-point EPQS service as fallback for
  any vertex the batch can't resolve. Retries with backoff on busy/5xx.
- **Lossless to geometry** — geometry becomes 3-D `[lon, lat, z]`, an
  `elevation_m` array + `elevation_meta` (source, fill stats) are added, and all
  Part-1/1.5 per-vertex arrays are preserved. NaN where elevation is unknown.

## Status — Part 3: Route profile (`routegen/profile.py`)

Derives the fields `simulator2.route.RouteProfile` consumes — `sinθ(s)`, `κ(s)` —
plus `v_max(s)` carried alongside (RouteProfile has no field for it yet). Pure
NumPy (no SciPy); the whole stage is offline-testable.

- **Structure correction** — on bridges/tunnels the DEM returns the deck/ground,
  not the rail, so elevation there (and any NaN) is discarded and linearly
  interpolated across, using the propagated structure flags.
- **Smooth, then differentiate** — a Savitzky–Golay filter gives smoothed
  elevation *and* its analytic derivative from one operator (3DEP error ~1 m would
  make a raw `dz/ds` almost all noise). `sinθ = sin(arctan(dz/ds))` with a physical
  grade clamp.
- **Curvature** — `(lon, lat)` → local planar frame; SG first/second derivatives
  give `κ = (x'y'' − y'x'')/(x'²+y'²)^{3/2}`, clamped to a minimum radius.
- **Speed limit** — `maxspeed_kph` → m/s with an FRA track-class fallback where
  OSM coverage is missing.
- **Output** — a viewable `route_profiles/<name>.geojson` (arrays in properties)
  **and** a `DATA_SCHEMA`-compliant `route_profiles/<name>.npz`
  (`route_s`, `route_sin_theta`, `route_kappa`, `route_vmax`, float32) — the
  drop-in handoff for dataset generation. `validate_profile` constructs a real
  `RouteProfile` and samples it as a sanity check.

## Status — Export bin (`routegen/export.py`)

Collects fully-processed routes (sent over from the Route-profile tab) and emits
the **route-tensor** handoff for the dataset build. A route tensor is the
canonical arc-length field of `DATA_SCHEMA §F`, four named float32 arrays —
`route_s`, `route_sin_theta`, `route_kappa`, `route_vmax`. The Export action
bundles a `routes/<name>.npz` per route plus an `index.csv` summary (location,
start/end, distance, grade/curvature stats) and a `manifest.json` (channel legend
+ provenance) into one zip that mirrors the dataset-build layout.

`route_s/route_sin_theta/route_kappa` drop straight onto `simulator2.RouteProfile`;
`route_vmax` is included in the tensor but consuming it needs a small RouteProfile
extension (`v_max_nodes` + `v_max_at`) — **deferred** pending the route-tensor
representation regroup.

Not yet (next parts):

- **Part 4** — batch dataset generation + storage per `DATA_SCHEMA.md` (loop
  sampled consist/control over the curated route profiles; emit scenario `.npz` +
  `index.parquet`).

## Reliability & filtering (avoiding Overpass timeouts)

Large single queries (e.g. all rail within 30 km of a dense hub) time out. Three
measures fix this and are on by default:

- **Main lines only** — queries exclude `service=*` (yards/sidings/crossovers) and
  `usage=industrial/tourism/military` (factory/plant spurs), so only running
  freight/main lines are pulled. This both matches the study scope and *drastically*
  cuts data volume in metros. Toggle off with the app checkbox or CLI `--all-tracks`.
- **Tiled seed** — the seed pull is split into a grid of small `around` queries
  (`--seed-tile-km`, default 8 km) instead of one big one, so no single request is
  heavy. The bounded crawl then expands outward in small steps as before.
- **Retry + failover** — every request retries on timeout / busy (429/502/503/504)
  with exponential backoff + jitter and rotates across the known mirrors.

If runs are still slow, lower `--seed-tile-km`, lower `--max-radius-km`, raise
`--sleep`, or use `--no-crawl` for seed-only pulls.

## Simplest workflow: click & store (recommended for a few routes)

If you just want a handful of real routes near a city, skip the bulk runner:

1. Launch the app, use **Radius mode** centered on your city. Turn on **Main lines
   only** (default) so yards/industrial spurs are excluded, and enable **Follow
   lines beyond radius** so each line is captured end-to-end.
2. On the clickable map, **click a line** — its full route highlights in red
   (all fragments of that physical line, stitched).
3. Name it and hit **Take & store route**. It's written to `saved_routes/<name>.geojson`
   with every vertex's bridge/tunnel/cutting/embankment flags, speed-limit coverage,
   and arc length — i.e. complete with the features you need for Part 2.

Service tracks render grey and are excluded from stored mainline routes; click one
directly only if you specifically want it. (Click-to-select needs
`streamlit-folium`; if it's not installed the app falls back to a static map.)

## Bulk workflow: verify, then auto-pull

**1. Verify (interactive, in the app).** Run the Streamlit app, use Radius mode on a
known area, and confirm: lines render contiguous (one `line_group` colour per
physical line), gaps are short and at bridges, branches sit at real junctions,
service tracks are correctly greyed, and — cross-checking against
openrailwaymap.org — a corridor traces real track at a plausible length. Try the
bounded crawl on a small seed radius and confirm a line extends past the ball and
stops. Once two or three areas look right, acquisition is trusted.

**2. Bulk-pull (headless, `build_dataset.py`).** Crawl a curated list of seed points
into a deduplicated dataset, in resumable batches:

```bash
# Validate the pipeline on the first few seeds:
python build_dataset.py --seeds seeds/seeds_us.json --out data --limit 3 \
    --seed-radius-km 25 --max-radius-km 120 \
    --endpoint https://overpass.kumi.systems/api/interpreter

# Full national run (resumes automatically; cached seeds are skipped):
python build_dataset.py --seeds seeds/seeds_us.json --out data \
    --endpoint https://overpass.kumi.systems/api/interpreter --sleep 1.5
```

Each seed is crawled to full lines (bounded by `--max-radius-km`), grouped, and
stitched into one ordered line per `line_group`; lines pulled from overlapping
seeds are **deduplicated by shared OSM way ids** (longest kept). Output:
`data/routes/line_*.geojson`, `data/index.csv`, `data/manifest.json`, plus a
resumable `data/raw/` cache. Edit `seeds/seeds_us.json` (36 US rail hubs) to change
coverage. Re-run with `--dedup-only` to rebuild the dataset from the cache without
any network calls.

## Install & run

```bash
cd route_generator
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Default (Area) mode: enter a bounding box — read one off bboxfinder.com or the OSM
"Export" tab — and the app pulls all rail in it and lists the corridors. Relation
mode: enter an OSM **relation** ID for a `route=railway` relation. Live fetching
needs network access and runs on your machine.

## Tests (offline, no network)

```bash
python tests/test_centerline.py        # or: pytest tests/
```

The assembly path is fully unit-tested against a crafted Overpass fixture, so the
stitching logic is validated without any network calls.

## Layout

```
route_generator/
  routegen/
    osm.py          # Overpass fetch (relation + area) + parsing + tag/maxspeed parsing
    centerline.py   # robust way-stitching, gap/branch detection, shared builder
    network.py      # area graph -> corridor extraction (split at junctions/ends)
    geometry.py     # haversine length / arc length (pure NumPy)
    models.py       # Way, RelationData, Centerline, Gap, Branch
    io.py           # GeoJSON + per-vertex record export
  app/streamlit_app.py
  tests/            # offline tests (centerline + network) + Overpass fixture
```

## Design notes

- **Acquisition and assembly are network-free except `fetch_relation`.**
  `parse_overpass_json` + `assemble_centerline` are pure functions, which is why
  they can be tested offline and reused for cached extracts later.
- **Quality is surfaced, not hidden.** Gaps are bridged but reported with
  distances; branch points (route ambiguity) are flagged; disconnected ways are
  dropped and listed. You decide whether a route is clean enough for the dataset.
- **Dependency-light core.** The whole pipeline — acquisition, resample, elevation,
  and the Part-3 profile (Savitzky–Golay smoothing, curvature, grade) — runs on
  `numpy`/`requests` alone; no SciPy. The smoothing/derivative operator is a small
  pure-NumPy Savitzky–Golay implementation, so every stage stays offline-testable.
