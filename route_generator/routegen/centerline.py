"""Assemble a continuous, ordered centerline from a route relation's member ways.

OSM route relations store member ways in arbitrary order and direction. This
module stitches them into a single ordered polyline by matching shared endpoint
nodes, with coordinate-based fallback, and reports gaps and branches so the
caller (and the UI) can judge route quality before committing it to the dataset.

Design goal: robust, but not excessively clever. Endpoint chaining handles the
overwhelming majority of well-formed railway routes; ambiguous topology is
flagged rather than silently guessed.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import cumulative_arclength_m, haversine_m
from .models import SERVICE_VALUES, Branch, Centerline, Gap, RelationData, Way
from .osm import parse_maxspeed_kph, tag_is_truthy

# Two endpoints closer than this are treated as the same junction (no gap).
COINCIDENT_EPS_M = 0.5
# Bridge a node-id mismatch silently if endpoints are within this distance.
DEFAULT_GAP_TOLERANCE_M = 5.0
# Never jump farther than this to continue the chain; remainder is dropped.
DEFAULT_MAX_JUMP_M = 2_000.0


def _orient_to_start_at(way: Way, node_id: int) -> Optional[Way]:
    """Return the way oriented so its first node is ``node_id``, or None."""
    if way.start_node == node_id:
        return way
    if way.end_node == node_id:
        return way.reversed()
    return None


def _endpoint_adjacency(ways: List[Way]) -> Dict[int, List[int]]:
    """Map endpoint node id -> list of way indices having it as start or end."""
    adj: Dict[int, List[int]] = {}
    for idx, w in enumerate(ways):
        for nid in (w.start_node, w.end_node):
            if nid is not None:
                adj.setdefault(nid, []).append(idx)
    return adj


def _detect_branches(ways: List[Way], adj: Dict[int, List[int]]) -> List[Branch]:
    branches: List[Branch] = []
    for nid, idxs in adj.items():
        if len(idxs) > 2:
            # Find a coordinate for this node from any incident way endpoint.
            at = None
            for i in idxs:
                w = ways[i]
                if w.start_node == nid:
                    at = w.coords[0]
                    break
                if w.end_node == nid:
                    at = w.coords[-1]
                    break
            branches.append(
                Branch(node_id=nid, degree=len(idxs), way_ids=[ways[i].id for i in idxs], at_lonlat=at)
            )
    return branches


def _choose_start_index(ways: List[Way], adj: Dict[int, List[int]]) -> Tuple[int, int]:
    """Pick a starting (way_index, node_id). Prefer a degree-1 terminal endpoint."""
    terminals = sorted(nid for nid, idxs in adj.items() if len(idxs) == 1)
    if terminals:
        start_node = terminals[0]
        start_idx = adj[start_node][0]
        return start_idx, start_node
    # No terminal (loop or fully connected): start at the lowest-id way's start.
    start_idx = min(range(len(ways)), key=lambda i: ways[i].id)
    return start_idx, ways[start_idx].start_node


def _endpoint_coord(way: Way, which: str) -> Tuple[float, float]:
    return way.coords[0] if which == "start" else way.coords[-1]


def assemble_centerline(
    rel: RelationData,
    gap_tolerance_m: float = DEFAULT_GAP_TOLERANCE_M,
    max_jump_m: float = DEFAULT_MAX_JUMP_M,
) -> Centerline:
    """Stitch ``rel.ways`` into an ordered :class:`Centerline` with QA metadata."""
    ways = [w for w in rel.ways if len(w.node_ids) >= 2]
    if not ways:
        raise ValueError("Relation has no usable ways (need >= 2 nodes each).")

    adj = _endpoint_adjacency(ways)
    branches = _detect_branches(ways, adj)

    start_idx, open_node = _choose_start_index(ways, adj)
    first = _orient_to_start_at(ways[start_idx], open_node) or ways[start_idx]

    order: List[Tuple[int, bool]] = []
    gaps: List[Gap] = []
    used = set()

    def append_way(w_oriented: Way, original: Way):
        order.append((original.id, w_oriented.node_ids != original.node_ids))

    chain: List[Way] = [first]
    append_way(first, ways[start_idx])
    used.add(start_idx)
    open_node = first.end_node
    open_coord = first.coords[-1]

    while True:
        # 1) Prefer an unused way sharing the open endpoint node id.
        candidates = [i for i in adj.get(open_node, []) if i not in used]
        if candidates:
            nxt = min(candidates, key=lambda i: ways[i].id)
            oriented = _orient_to_start_at(ways[nxt], open_node)
            if oriented is None:  # shouldn't happen given adjacency, but be safe
                oriented = ways[nxt]
            chain.append(oriented)
            append_way(oriented, ways[nxt])
            used.add(nxt)
            open_node = oriented.end_node
            open_coord = oriented.coords[-1]
            continue

        # 2) Coordinate fallback: nearest unused way endpoint to the open coord.
        best = _nearest_unused_endpoint(ways, used, open_coord)
        if best is None:
            break
        i, which, dist = best
        if dist > max_jump_m:
            break  # too far to be the same route; leave remainder for "dropped"
        w = ways[i]
        oriented = w if which == "start" else w.reversed()
        if dist > gap_tolerance_m and dist > COINCIDENT_EPS_M:
            gaps.append(
                Gap(
                    after_way_id=chain[-1].id,
                    before_way_id=w.id,
                    distance_m=float(dist),
                    at_lonlat=open_coord,
                )
            )
        chain.append(oriented)
        append_way(oriented, w)
        used.add(i)
        open_node = oriented.end_node
        open_coord = oriented.coords[-1]

    dropped = [ways[i].id for i in range(len(ways)) if i not in used]

    return make_centerline_from_chain(
        rel.id, rel.tags, chain,
        order=order, gaps=gaps, branches=branches,
        used_way_ids=[ways[i].id for i in sorted(used)],
        dropped_way_ids=dropped,
    )


def make_centerline_from_chain(
    relation_id: int,
    relation_tags: Dict[str, str],
    chain: List[Way],
    order: Optional[List[Tuple[int, bool]]] = None,
    gaps: Optional[List[Gap]] = None,
    branches: Optional[List[Branch]] = None,
    used_way_ids: Optional[List[int]] = None,
    dropped_way_ids: Optional[List[int]] = None,
) -> Centerline:
    """Build a :class:`Centerline` from an already-ordered chain of oriented ways.

    Shared by single-relation assembly and area-based corridor extraction. Tags
    are read from each chain way directly (chain ways carry their own tags).
    """
    coords, way_id_per_vertex = _concatenate_chain(chain)
    cl_coords = np.asarray(coords, dtype=float)
    s_m = cumulative_arclength_m(cl_coords)

    tag_by_way = {w.id: w.tags for w in chain}
    way_id_arr = np.asarray(way_id_per_vertex, dtype=np.int64)
    maxspeed = np.array(
        [parse_maxspeed_kph(tag_by_way[wid].get("maxspeed")) for wid in way_id_per_vertex],
        dtype=float,
    )
    is_bridge = np.array([tag_is_truthy(tag_by_way[w], "bridge") for w in way_id_per_vertex])
    is_tunnel = np.array([tag_is_truthy(tag_by_way[w], "tunnel") for w in way_id_per_vertex])
    is_cutting = np.array([tag_is_truthy(tag_by_way[w], "cutting") for w in way_id_per_vertex])
    is_embankment = np.array(
        [tag_is_truthy(tag_by_way[w], "embankment") for w in way_id_per_vertex]
    )

    # Service-track classification: a corridor is service if the majority of its
    # vertices come from ways tagged service=siding/spur/yard/crossover.
    svc_vals = [tag_by_way[w].get("service") for w in way_id_per_vertex]
    svc_counts = Counter(v for v in svc_vals if v in SERVICE_VALUES)
    is_service = False
    service_kind: Optional[str] = None
    if svc_counts and sum(svc_counts.values()) > 0.5 * len(svc_vals):
        is_service = True
        service_kind = svc_counts.most_common(1)[0][0] if len(svc_counts) == 1 else "mixed"

    return Centerline(
        relation_id=relation_id,
        relation_tags=relation_tags,
        coords=cl_coords,
        s_m=s_m,
        way_id=way_id_arr,
        maxspeed_kph=maxspeed,
        is_bridge=is_bridge,
        is_tunnel=is_tunnel,
        is_cutting=is_cutting,
        is_embankment=is_embankment,
        is_service=is_service,
        service_kind=service_kind,
        order=order if order is not None else [(w.id, False) for w in chain],
        gaps=gaps or [],
        branches=branches or [],
        used_way_ids=used_way_ids if used_way_ids is not None else [w.id for w in chain],
        dropped_way_ids=dropped_way_ids or [],
    )


def _nearest_unused_endpoint(ways, used, coord) -> Optional[Tuple[int, str, float]]:
    """Nearest (way_index, 'start'|'end', distance_m) among unused ways."""
    lon0, lat0 = coord
    best = None
    for i, w in enumerate(ways):
        if i in used:
            continue
        for which in ("start", "end"):
            lon, lat = _endpoint_coord(w, which)
            d = float(haversine_m(lon0, lat0, lon, lat))
            if best is None or d < best[2]:
                best = (i, which, d)
    return best


def _concatenate_chain(chain: List[Way]) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Join oriented ways, dropping the duplicate shared vertex at clean junctions."""
    coords: List[Tuple[float, float]] = []
    way_ids: List[int] = []
    for k, w in enumerate(chain):
        wc = w.coords
        if k == 0:
            coords.extend(wc)
            way_ids.extend([w.id] * len(wc))
            continue
        # Decide whether the first vertex duplicates the previous last vertex.
        prev_lon, prev_lat = coords[-1]
        cur_lon, cur_lat = wc[0]
        d = float(haversine_m(prev_lon, prev_lat, cur_lon, cur_lat))
        start = 1 if d <= COINCIDENT_EPS_M else 0
        coords.extend(wc[start:])
        way_ids.extend([w.id] * len(wc[start:]))
    return coords, way_ids
