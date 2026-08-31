"""Offline tests for the bulk dataset builder (no network).

Uses a synthetic rail 'world' + a fake fetcher so the full crawl->extract->group
->stitch->dedup->write pipeline runs without Overpass.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from routegen.dataset import (  # noqa: E402
    Seed,
    dedup_features,
    lines_from_corridors,
    run_acquisition,
    stitch_line,
)
from routegen.geometry import haversine_m, polyline_length_m  # noqa: E402
from routegen.network import extract_corridors  # noqa: E402
from routegen.models import RelationData, Way  # noqa: E402

LAT = 39.0
LON0 = -104.0
L = 0.10
GAP = 0.0002
N = 16  # ~140 km line


def _world():
    ways = []
    for k in range(N):
        a = LON0 + k * L
        b = LON0 + (k + 1) * L - GAP
        ways.append(Way(id=k + 1, tags={"railway": "rail"},
                        node_ids=[10 * (k + 1), 10 * (k + 1) + 1],
                        coords=[(a, LAT), (b, LAT)]))
    return ways


def _fake_fetcher(world):
    def fetch(points_latlon, radius_m):
        out = []
        for w in world:
            for (wlon, wlat) in (w.coords[0], w.coords[-1]):
                if any(haversine_m(wlon, wlat, qlon, qlat) <= radius_m
                       for (qlat, qlon) in points_latlon):
                    out.append(w)
                    break
        return out
    return fetch


def test_stitch_line_concatenates_fragments_in_order():
    # Build fragments via extraction on a micro-gap world, then stitch the group.
    rel = RelationData(id=-1, tags={}, ways=_world()[:5])
    corridors = extract_corridors(rel, min_length_m=100.0)
    line = stitch_line([c for c in corridors if not c.is_service])
    # Monotonic longitudes and length ≈ sum of fragment lengths.
    lons = line.coords[:, 0]
    assert np.all(np.diff(lons) > 0)
    assert line.length_m > 30_000  # 5 × ~8.6 km
    assert len(line.used_way_ids) == 5


def test_dedup_features_keeps_longest_on_shared_ways():
    short = {"properties": {"used_way_ids": [1, 2], "length_m": 1000.0},
             "geometry": {}, "type": "Feature"}
    longer = {"properties": {"used_way_ids": [2, 3, 4], "length_m": 5000.0},
              "geometry": {}, "type": "Feature"}
    isolated = {"properties": {"used_way_ids": [99], "length_m": 2000.0},
                "geometry": {}, "type": "Feature"}
    kept = dedup_features([short, longer, isolated])
    lengths = sorted(f["properties"]["length_m"] for f in kept)
    assert lengths == [2000.0, 5000.0]  # short+longer merged -> longer kept; isolated stays


def test_run_acquisition_end_to_end_and_dedup_across_seeds():
    world = _world()
    fetch = _fake_fetcher(world)
    # Two seeds on the SAME line -> after dedup, one line should remain.
    seeds = [Seed("west", LAT, LON0 + 1 * L), Seed("east", LAT, LON0 + 12 * L)]
    with tempfile.TemporaryDirectory() as out:
        manifest = run_acquisition(
            seeds, out, seed_radius_km=5.0, crawl=True, max_radius_km=300.0,
            expand_m=60.0, max_iterations=60, min_len_km=1.0, sleep_s=0.0,
            fetcher=fetch, log=lambda *_: None)
        routes = os.listdir(os.path.join(out, "routes"))
        assert manifest["n_lines_deduped"] == 1, manifest
        assert len(routes) == 1
        # The deduped line spans most of the world (both seeds' crawls merged).
        feat = json.load(open(os.path.join(out, "routes", routes[0])))
        coords = feat["geometry"]["coordinates"]
        assert polyline_length_m(np.asarray(coords)) > 100_000
        assert os.path.exists(os.path.join(out, "index.csv"))
        assert os.path.exists(os.path.join(out, "manifest.json"))


def test_resume_skips_cached_seeds():
    world = _world()
    seeds = [Seed("west", LAT, LON0 + 1 * L)]
    with tempfile.TemporaryDirectory() as out:
        run_acquisition(seeds, out, seed_radius_km=5.0, max_radius_km=300.0,
                        max_iterations=60, sleep_s=0.0, fetcher=_fake_fetcher(world),
                        log=lambda *_: None)
        raw_files = os.listdir(os.path.join(out, "raw"))
        assert len(raw_files) == 1
        # Second run with a fetcher that would explode if called -> must be skipped.
        def boom(*a, **k):
            raise AssertionError("fetcher called despite cache")
        run_acquisition(seeds, out, seed_radius_km=5.0, max_radius_km=300.0,
                        max_iterations=60, sleep_s=0.0, fetcher=boom,
                        log=lambda *_: None)  # should not raise


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
