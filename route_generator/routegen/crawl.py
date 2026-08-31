"""Bounded outward crawl: follow lines beyond the seed radius to their natural ends.

A radius pull captures only the ways near a point; corridors that continue past
the ball get cut. This module seeds from the radius, then iteratively expands the
*open ends* of the captured network — endpoints with no neighbouring track in the
current set — by querying Overpass just beyond them, until lines reach real termini
or a hard cap is hit (max distance from centre, iterations, or way count).

The cap is essential: the freight network is one connected graph, so an uncapped
crawl would eventually pull the whole continent.

The Overpass call is injected via ``fetcher`` so the crawl loop is unit-testable
offline against a synthetic network.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, List, Optional, Tuple

from .geometry import haversine_m
from .models import RelationData, Way
from .osm import (
    DEFAULT_ENDPOINT,
    _post_overpass,
    build_multi_around_query,
    endpoint_rotation,
    parse_overpass_json,
)

# fetcher signature: (points_latlon: list[(lat, lon)], radius_m: float) -> list[Way]
Fetcher = Callable[[List[Tuple[float, float]], float], List[Way]]


def open_endpoints(ways: List[Way], tol_m: float) -> List[Tuple[float, float]]:
    """Endpoints (lon, lat) with no *other* way-endpoint within ``tol_m``.

    These are the line's open ends — either true termini or places where the
    track continues but hasn't been fetched yet. Internal joints (shared nodes
    and micro-gaps alike) have a near neighbour and are excluded. Uses a coarse
    spatial grid for O(E) neighbour lookup.
    """
    eps: List[Tuple[float, float]] = []
    for w in ways:
        if len(w.coords) >= 2:
            eps.append(w.coords[0])
            eps.append(w.coords[-1])
    if not eps:
        return []

    cell = max(tol_m, 1.0) / 111_320.0  # ~tol-sized cells in degrees (lat)
    grid = defaultdict(list)

    def key(lon, lat):
        return (int(lat / cell), int(lon / cell))

    for idx, (lon, lat) in enumerate(eps):
        grid[key(lon, lat)].append(idx)

    opens: List[Tuple[float, float]] = []
    for idx, (lon, lat) in enumerate(eps):
        ky = key(lon, lat)
        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for j in grid.get((ky[0] + dy, ky[1] + dx), []):
                    if j == idx:
                        continue
                    lon2, lat2 = eps[j]
                    if haversine_m(lon, lat, lon2, lat2) <= tol_m:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            opens.append((lon, lat))
    return opens


def disk_grid(center_lat, center_lon, radius_m, step_m):
    """Grid of (lat, lon) points covering a disk, spaced ``step_m`` apart."""
    dlat = step_m / 111_320.0
    dlon = step_m / (111_320.0 * max(math.cos(math.radians(center_lat)), 0.1))
    n = int(math.ceil(radius_m / step_m))
    pts = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            la, lo = center_lat + i * dlat, center_lon + j * dlon
            if haversine_m(center_lon, center_lat, lo, la) <= radius_m + step_m:
                pts.append((la, lo))
    return pts


def _make_overpass_fetcher(endpoint, railway_values, timeout_s, session,
                           main_only=True, retries=3, chunk=8) -> Fetcher:
    """Robust fetcher: small chunked around-queries with retry + endpoint failover."""
    eps = endpoint_rotation(endpoint)

    def fetch(points_latlon, radius_m):
        ways: List[Way] = []
        seen = set()
        for i in range(0, len(points_latlon), chunk):
            q = build_multi_around_query(points_latlon[i:i + chunk], radius_m,
                                         railway_values=railway_values,
                                         timeout_s=timeout_s, main_only=main_only)
            data = _post_overpass(endpoint, q, timeout_s, session=session,
                                  retries=retries, endpoints=eps)
            for w in parse_overpass_json(data).ways:
                if w.id not in seen:
                    seen.add(w.id)
                    ways.append(w)
        return ways
    return fetch


def crawl_network(
    center_lat: float,
    center_lon: float,
    seed_radius_m: float,
    endpoint: str = DEFAULT_ENDPOINT,
    railway_values=("rail", "narrow_gauge"),
    max_radius_m: float = 150_000.0,
    expand_radius_m: float = 60.0,
    max_iterations: int = 25,
    max_ways: int = 5000,
    timeout_s: int = 90,
    main_only: bool = True,
    seed_tile_km: float = 8.0,
    retries: int = 3,
    session=None,
    fetcher: Optional[Fetcher] = None,
    progress: Optional[Callable[[int, int, int], None]] = None,
) -> RelationData:
    """Seed at a point, then follow open line-ends outward within hard caps.

    The seed is gathered as a *grid of small around-queries* covering the disk
    (``seed_tile_km`` spacing) rather than one large query, so dense areas don't
    time out. Returns a RelationData with all accumulated ways (id=-1). Stops when
    no open end remains within ``max_radius_m``, when an expansion adds nothing new
    (real termini), or when ``max_iterations`` / ``max_ways`` is reached.
    """
    fetch = fetcher or _make_overpass_fetcher(
        endpoint, railway_values, timeout_s, session, main_only=main_only, retries=retries)

    # Tiled seed: many small around-queries instead of one big one.
    step_m = seed_tile_km * 1000.0
    seed_pts = disk_grid(center_lat, center_lon, seed_radius_m, step_m)
    tile_radius_m = min(seed_radius_m, step_m * 0.8)
    known = {}
    for w in fetch(seed_pts, tile_radius_m):
        known[w.id] = w
    if progress:
        progress(0, len(known), 0)

    expanded = set()
    for it in range(1, max_iterations + 1):
        if len(known) >= max_ways:
            break
        frontier = []
        for lon, lat in open_endpoints(list(known.values()), expand_radius_m):
            if haversine_m(center_lon, center_lat, lon, lat) > max_radius_m:
                continue
            k = (round(lat, 5), round(lon, 5))
            if k in expanded:
                continue
            frontier.append((lon, lat, k))
        if not frontier:
            break

        new_ways = fetch([(lat, lon) for (lon, lat, _) in frontier], expand_radius_m)
        for (_, _, k) in frontier:
            expanded.add(k)
        added = 0
        for w in new_ways:
            if w.id not in known:
                known[w.id] = w
                added += 1
        if progress:
            progress(it, len(known), len(frontier))
        if added == 0:
            break

    return RelationData(
        id=-1,
        tags={"source": "radius_crawl", "center": f"{center_lat},{center_lon}",
              "seed_radius_m": str(seed_radius_m), "max_radius_m": str(max_radius_m)},
        ways=list(known.values()),
    )
