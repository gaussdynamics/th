"""Offline tests for the bounded outward crawl (no network).

A synthetic 'world' of end-to-end line segments stands in for OSM; a fake fetcher
returns world ways whose endpoints fall within the query radius. We verify the
crawl walks the line outward from the seed and that the distance cap stops it
before it consumes the whole line.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routegen.crawl import crawl_network, disk_grid, open_endpoints  # noqa: E402
from routegen.geometry import haversine_m  # noqa: E402
from routegen.models import Way  # noqa: E402

LAT = 39.0
LON0 = -104.0
L = 0.10        # ~8.6 km per segment at this latitude
GAP = 0.0002    # ~17 m micro-gap between segments (< expand radius)
N = 20          # total segments in the world (~170 km line)


def _world():
    ways = []
    for k in range(N):
        a = LON0 + k * L
        b = LON0 + (k + 1) * L - GAP
        ways.append(Way(id=k + 1, tags={"railway": "rail"}, node_ids=[10 * (k + 1), 10 * (k + 1) + 1],
                        coords=[(a, LAT), (b, LAT)]))
    return ways


def _fake_fetcher(world):
    """Return world ways with any endpoint within radius_m of any query point."""
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


def test_disk_grid_covers_disk_with_multiple_points():
    pts = disk_grid(39.0, -104.0, radius_m=30_000, step_m=8000)
    assert len(pts) > 10  # tiled into many small seed points
    # All grid points lie within radius + one step of the centre.
    assert all(haversine_m(-104.0, 39.0, lo, la) <= 30_000 + 8000 for (la, lo) in pts)


def test_open_endpoints_finds_only_extreme_ends():
    # Two micro-gap-joined segments: only the two outer ends are "open".
    ways = _world()[:2]
    opens = open_endpoints(ways, tol_m=60.0)
    assert len(opens) == 2


def test_crawl_walks_outward_and_cap_stops_it():
    world = _world()
    fetch = _fake_fetcher(world)
    # Seed at the west end with a tiny radius (only segment 1), cap at ~80 km.
    rel = crawl_network(LAT, LON0, seed_radius_m=2000, expand_radius_m=60.0,
                        max_radius_m=80_000.0, max_iterations=50, fetcher=fetch)
    got = sorted(w.id for w in rel.ways)
    # Crawl extended well past the seed...
    assert len(got) > 3
    # ...but the 80 km cap stopped it before the whole 170 km line was consumed.
    assert len(got) < N
    # Everything collected lies within the cap distance of the centre.
    for w in rel.ways:
        for (wlon, wlat) in (w.coords[0], w.coords[-1]):
            assert haversine_m(LON0, LAT, wlon, wlat) <= 80_000.0 + L * 111_000


def test_crawl_terminates_at_real_end_within_cap():
    world = _world()[:3]  # short line, well inside a big cap
    rel = crawl_network(LAT, LON0, seed_radius_m=2000, expand_radius_m=60.0,
                        max_radius_m=500_000.0, max_iterations=50, fetcher=_fake_fetcher(world))
    # The whole short line is captured, then it stops (no infinite loop).
    assert sorted(w.id for w in rel.ways) == [1, 2, 3]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
