"""Headless bulk dataset builder: crawl curated seed points into a route dataset.

Pipeline per `PROJECT_DIRECTION.md` / `DATA_SCHEMA.md`:

  Stage A (per seed, networked, **resumable**): crawl the bounded network around a
  seed point, extract + group corridors, stitch each parent line_group into one
  ordered line, and cache the seed's lines to ``raw/``. Re-runs skip cached seeds.

  Stage B (offline, cheap, re-runnable): load all cached seed lines, **dedupe**
  lines pulled from overlapping seeds by shared OSM way ids (keep the longest),
  and write ``routes/<id>.geojson`` + ``index.csv`` + ``manifest.json``.

The Overpass call is injected through ``fetcher`` (see crawl.crawl_network) so the
whole pipeline is unit-testable offline.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .crawl import crawl_network
from .geometry import cumulative_arclength_m, haversine_m
from .io import centerline_to_geojson
from .models import Centerline
from .network import _UnionFind, extract_corridors

_COINCIDENT_EPS_M = 0.5


@dataclass
class Seed:
    name: str
    lat: float
    lon: float
    radius_km: Optional[float] = None  # overrides the run default if set


def load_seeds(path: str) -> List[Seed]:
    """Load seeds from JSON (list of {name, lat, lon, radius_km?})."""
    with open(path) as fh:
        data = json.load(fh)
    return [Seed(name=d["name"], lat=float(d["lat"]), lon=float(d["lon"]),
                 radius_km=d.get("radius_km")) for d in data]


# --------------------------------------------------------------------------- #
# Stitching line_group fragments into one ordered line
# --------------------------------------------------------------------------- #
def _d(p, q) -> float:
    return float(haversine_m(p[0], p[1], q[0], q[1]))


def _seg_arrays(cl: Centerline) -> Dict:
    return {
        "coords": np.asarray(cl.coords, float),
        "way_id": np.asarray(cl.way_id),
        "maxspeed_kph": np.asarray(cl.maxspeed_kph, float),
        "is_bridge": np.asarray(cl.is_bridge, bool),
        "is_tunnel": np.asarray(cl.is_tunnel, bool),
        "is_cutting": np.asarray(cl.is_cutting, bool),
        "is_embankment": np.asarray(cl.is_embankment, bool),
    }


def _rev(seg: Dict) -> Dict:
    return {k: v[::-1] for k, v in seg.items()}


def stitch_line(fragments: List[Centerline]) -> Centerline:
    """Concatenate a line_group's fragments (in group_order) into one ordered line.

    Each fragment is oriented to connect to the running end; duplicate junction
    vertices are dropped. This is where fragments are finally materialised into a
    single continuous route (acquisition kept them separate and reversible).
    """
    frags = sorted(fragments, key=lambda c: (c.group_order if c.group_order is not None else 0))
    segs = [_seg_arrays(f) for f in frags]

    if len(segs) > 1:
        s0, nb = segs[0]["coords"], segs[1]["coords"]
        d_start = min(_d(s0[0], nb[0]), _d(s0[0], nb[-1]))
        d_end = min(_d(s0[-1], nb[0]), _d(s0[-1], nb[-1]))
        if d_start < d_end:
            segs[0] = _rev(segs[0])

    cols = {k: [segs[0][k]] for k in segs[0]}
    for s in segs[1:]:
        last = cols["coords"][-1][-1]
        if _d(last, s["coords"][-1]) < _d(last, s["coords"][0]):
            s = _rev(s)
        start = 1 if _d(last, s["coords"][0]) <= _COINCIDENT_EPS_M else 0
        for k in s:
            cols[k].append(s[k][start:])

    coords = np.vstack(cols["coords"])
    used = sorted({int(w) for f in frags for w in f.used_way_ids})
    return Centerline(
        relation_id=-1,
        relation_tags={"line_group": str(frags[0].line_group)},
        coords=coords,
        s_m=cumulative_arclength_m(coords),
        way_id=np.concatenate(cols["way_id"]),
        maxspeed_kph=np.concatenate(cols["maxspeed_kph"]),
        is_bridge=np.concatenate(cols["is_bridge"]),
        is_tunnel=np.concatenate(cols["is_tunnel"]),
        is_cutting=np.concatenate(cols["is_cutting"]),
        is_embankment=np.concatenate(cols["is_embankment"]),
        line_group=frags[0].line_group,
        used_way_ids=used,
    )


def lines_from_corridors(corridors: List[Centerline]) -> List[Centerline]:
    """Stitch each mainline line_group into one line. Service tracks are excluded."""
    groups: Dict[int, List[Centerline]] = defaultdict(list)
    for c in corridors:
        if not c.is_service and c.line_group is not None:
            groups[c.line_group].append(c)
    return [stitch_line(g) for g in groups.values()]


# --------------------------------------------------------------------------- #
# Dedup across seeds (operates on GeoJSON feature dicts)
# --------------------------------------------------------------------------- #
def dedup_features(features: List[Dict]) -> List[Dict]:
    """Collapse features that share any OSM way id (same physical track); keep longest."""
    n = len(features)
    uf = _UnionFind(n)
    wayid_to_idx: Dict[int, List[int]] = defaultdict(list)
    for i, f in enumerate(features):
        for w in f["properties"].get("used_way_ids", []):
            wayid_to_idx[w].append(i)
    for idxs in wayid_to_idx.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    comps: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comps[uf.find(i)].append(i)
    kept = []
    for members in comps.values():
        best = max(members, key=lambda i: features[i]["properties"].get("length_m", 0.0))
        kept.append(features[best])
    kept.sort(key=lambda f: f["properties"].get("length_m", 0.0), reverse=True)
    return kept


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _seed_filename(idx: int, seed: Seed) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in seed.name)[:40]
    return f"seed_{idx:04d}_{safe}.json"


def run_acquisition(
    seeds: List[Seed],
    outdir: str,
    seed_radius_km: float = 30.0,
    crawl: bool = True,
    max_radius_km: float = 250.0,
    expand_m: float = 60.0,
    max_iterations: int = 40,
    min_len_km: float = 1.0,
    main_only: bool = True,
    seed_tile_km: float = 8.0,
    retries: int = 3,
    endpoint: Optional[str] = None,
    railway_values=("rail", "narrow_gauge"),
    sleep_s: float = 1.0,
    limit: Optional[int] = None,
    resume: bool = True,
    dedup_only: bool = False,
    fetcher=None,
    log: Callable[[str], None] = print,
) -> Dict:
    """Run the full bulk pipeline. Returns a summary dict; writes files under outdir."""
    raw_dir = os.path.join(outdir, "raw")
    routes_dir = os.path.join(outdir, "routes")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(routes_dir, exist_ok=True)

    work = seeds[:limit] if limit else seeds

    # ---- Stage A: per-seed crawl (resumable) ----
    if not dedup_only:
        for idx, seed in enumerate(work):
            raw_path = os.path.join(raw_dir, _seed_filename(idx, seed))
            if resume and os.path.exists(raw_path):
                log(f"[{idx+1}/{len(work)}] skip (cached): {seed.name}")
                continue
            r_km = seed.radius_km or seed_radius_km
            log(f"[{idx+1}/{len(work)}] crawling {seed.name} ({seed.lat},{seed.lon}) r={r_km}km")
            try:
                rel = crawl_network(
                    seed.lat, seed.lon, r_km * 1000,
                    endpoint=endpoint or _default_endpoint(),
                    railway_values=railway_values,
                    max_radius_m=max_radius_km * 1000, expand_radius_m=expand_m,
                    max_iterations=(max_iterations if crawl else 0),
                    main_only=main_only, seed_tile_km=seed_tile_km, retries=retries,
                    fetcher=fetcher,
                )
                corridors = extract_corridors(rel, min_length_m=min_len_km * 1000)
                lines = lines_from_corridors(corridors)
                feats = []
                for ln in lines:
                    gj = centerline_to_geojson(ln)
                    gj["properties"]["seed"] = seed.name
                    gj["properties"]["seed_idx"] = idx
                    feats.append(gj)
                with open(raw_path, "w") as fh:
                    json.dump(feats, fh)
                log(f"    -> {len(feats)} line(s), {sum(len(c.coords) for c in corridors)} verts")
            except Exception as exc:  # noqa: BLE001 - keep going; record the failure
                log(f"    !! failed: {exc}")
                with open(raw_path + ".error", "w") as fh:
                    fh.write(str(exc))
            if sleep_s:
                time.sleep(sleep_s)

    # ---- Stage B: dedup + write dataset ----
    all_feats: List[Dict] = []
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, fn)) as fh:
            all_feats.extend(json.load(fh))

    deduped = dedup_features(all_feats)
    index_rows = []
    for k, f in enumerate(deduped):
        line_id = f"line_{k:05d}"
        with open(os.path.join(routes_dir, f"{line_id}.geojson"), "w") as fh:
            json.dump(f, fh)
        p = f["properties"]
        index_rows.append({
            "line_id": line_id,
            "length_km": round(p.get("length_m", 0.0) / 1000, 2),
            "num_vertices": p.get("num_vertices"),
            "maxspeed_coverage": p.get("maxspeed_coverage"),
            "n_ways": len(p.get("used_way_ids", [])),
            "origin_seed": p.get("seed"),
        })

    _write_index_csv(os.path.join(outdir, "index.csv"), index_rows)
    manifest = {
        "n_seeds": len(work),
        "n_lines_raw": len(all_feats),
        "n_lines_deduped": len(deduped),
        "total_km": round(sum(r["length_km"] for r in index_rows), 1),
        "params": {"seed_radius_km": seed_radius_km, "crawl": crawl,
                   "max_radius_km": max_radius_km, "min_len_km": min_len_km,
                   "main_only": main_only, "seed_tile_km": seed_tile_km,
                   "railway_values": list(railway_values)},
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"DONE: {len(all_feats)} raw lines -> {len(deduped)} deduped, "
        f"{manifest['total_km']} km. Dataset in {routes_dir}")
    return manifest


def _write_index_csv(path: str, rows: List[Dict]) -> None:
    import csv
    cols = ["line_id", "length_km", "num_vertices", "maxspeed_coverage", "n_ways", "origin_seed"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _default_endpoint() -> str:
    from .osm import DEFAULT_ENDPOINT
    return DEFAULT_ENDPOINT
