"""Area-based acquisition: split a pulled rail *network* into individual corridors.

A bounding-box pull of ``railway=rail`` ways is a graph, not a route. This module
splits that graph into corridors: maximal paths that run through *continuation*
nodes (degree 2) and break at *line-ends* (degree 1) and *junctions* (degree >= 3).
Each corridor becomes its own ordered :class:`Centerline`, ready for the dataset.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .centerline import (
    _detect_branches,
    _endpoint_adjacency,
    _orient_to_start_at,
    make_centerline_from_chain,
)
from .geometry import angle_diff_deg, initial_bearing_deg, haversine_m
from .models import Centerline, RelationData, Way

# Endpoints within this distance are considered the same junction (covers shared
# nodes at 0 m and the common "coincident but unshared" micro-gaps).
GROUP_JOIN_TOL_M = 40.0
# Two endpoints form a straight through-connection if their outward tangents are
# within this many degrees of being opposite (i.e. the line continues straight).
GROUP_ANGLE_TOL_DEG = 35.0


def _grow_corridor(seed_idx: int, ways: List[Way], adj: Dict[int, List[int]],
                   degree: Dict[int, int], used: set) -> List[Way]:
    """Grow a maximal degree-2 chain through ``seed_idx`` in both directions."""
    seed = ways[seed_idx]
    chain: List[Way] = [seed]
    used.add(seed_idx)

    # Extend forward from the seed's end node.
    cur = seed.end_node
    while degree.get(cur, 0) == 2:
        nxts = [i for i in adj.get(cur, []) if i not in used]
        if not nxts:
            break
        i = nxts[0]
        oriented = _orient_to_start_at(ways[i], cur) or ways[i]
        chain.append(oriented)
        used.add(i)
        cur = oriented.end_node

    # Extend backward from the seed's start node.
    cur = seed.start_node
    while degree.get(cur, 0) == 2:
        prevs = [i for i in adj.get(cur, []) if i not in used]
        if not prevs:
            break
        i = prevs[0]
        w = ways[i]
        # Orient so the way *ends* at cur, then prepend.
        oriented = w if w.end_node == cur else w.reversed()
        chain.insert(0, oriented)
        used.add(i)
        cur = oriented.start_node

    return chain


def extract_corridors(
    rel: RelationData,
    min_length_m: float = 500.0,
    min_vertices: int = 2,
) -> List[Centerline]:
    """Split a pulled network into corridors, longest first.

    ``rel`` is any :class:`RelationData` carrying the ways (e.g. from an area pull,
    where ``rel.id == -1``). Corridors shorter than ``min_length_m`` or with fewer
    than ``min_vertices`` vertices are discarded as stubs.
    """
    ways = [w for w in rel.ways if len(w.node_ids) >= 2]
    if not ways:
        return []

    adj = _endpoint_adjacency(ways)
    degree = {node: len(idxs) for node, idxs in adj.items()}
    branches = _detect_branches(ways, adj)

    used: set = set()
    corridors: List[Centerline] = []
    # Deterministic seed order by way id.
    for seed_idx in sorted(range(len(ways)), key=lambda i: ways[i].id):
        if seed_idx in used:
            continue
        chain = _grow_corridor(seed_idx, ways, adj, degree, used)
        cl = make_centerline_from_chain(
            relation_id=rel.id,
            relation_tags=rel.tags,
            chain=chain,
            branches=branches,  # network-level junctions, shared for context
        )
        if cl.length_m >= min_length_m and cl.num_vertices >= min_vertices:
            corridors.append(cl)

    corridors.sort(key=lambda c: c.length_m, reverse=True)
    group_corridors(corridors)
    return corridors


# --------------------------------------------------------------------------- #
# Parent-line grouping: link collinear fragments without merging them.
# --------------------------------------------------------------------------- #
def _point_to_polyline_m(lon: float, lat: float, coords) -> float:
    """Min distance (m) from a point to a polyline, via local equirectangular projection.

    Uses point-to-*segment* distance (not just vertices), so a click mid-way along a
    long straight tangent with sparse vertices still maps to the line.
    """
    import numpy as np

    mx = 111_320.0 * math.cos(math.radians(lat))
    my = 110_540.0
    px = (coords[:, 0] - lon) * mx
    py = (coords[:, 1] - lat) * my
    if coords.shape[0] == 1:
        return float(math.hypot(px[0], py[0]))
    ax, ay = px[:-1], py[:-1]
    bx, by = px[1:], py[1:]
    dx, dy = bx - ax, by - ay
    seglen2 = dx * dx + dy * dy
    t = np.where(seglen2 > 0, -(ax * dx + ay * dy) / np.where(seglen2 > 0, seglen2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return float(np.sqrt(cx * cx + cy * cy).min())


def nearest_corridor(corridors: List[Centerline], lat: float, lon: float):
    """Return (index, distance_m) of the corridor closest to a clicked point.

    Used for click-to-select on a map: the click lat/lon maps to the line under it.
    """
    best = (-1, float("inf"))
    for i, c in enumerate(corridors):
        if c.num_vertices == 0:
            continue
        d = _point_to_polyline_m(lon, lat, c.coords)
        if d < best[1]:
            best = (i, d)
    return best


def _outward_tangents(cl: Centerline):
    """Bearings pointing *out* of the corridor at its start and end endpoints."""
    c = cl.coords
    t_start = initial_bearing_deg(c[1, 0], c[1, 1], c[0, 0], c[0, 1])
    t_end = initial_bearing_deg(c[-2, 0], c[-2, 1], c[-1, 0], c[-1, 1])
    return t_start, t_end


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def group_corridors(
    corridors: List[Centerline],
    join_tol_m: float = GROUP_JOIN_TOL_M,
    angle_tol_deg: float = GROUP_ANGLE_TOL_DEG,
) -> None:
    """Assign ``line_group`` / ``group_order`` in place, linking collinear fragments.

    Two mainline corridors are joined into a group when an endpoint of each lies
    within ``join_tol_m`` and their outward tangents are within ``angle_tol_deg``
    of opposite (the line runs straight through the junction/micro-gap). Service
    tracks are never merged into mainline groups; each gets its own group.
    Nothing is concatenated — only labelled — so the fragments stay reversible.
    """
    n = len(corridors)
    if n == 0:
        return

    # Collect mainline endpoints: (corridor_idx, 'start'|'end', coord, tangent).
    ends = []
    for i, c in enumerate(corridors):
        if c.num_vertices < 2 or c.is_service:
            continue
        t_s, t_e = _outward_tangents(c)
        ends.append((i, "start", (c.coords[0, 0], c.coords[0, 1]), t_s))
        ends.append((i, "end", (c.coords[-1, 0], c.coords[-1, 1]), t_e))

    uf = _UnionFind(n)
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a in range(len(ends)):
        ia, _, ca, ta = ends[a]
        for b in range(a + 1, len(ends)):
            ib, _, cb, tb = ends[b]
            if ia == ib:
                continue
            if haversine_m(ca[0], ca[1], cb[0], cb[1]) > join_tol_m:
                continue
            # Straight through-connection: outward tangents ~opposite.
            if abs(angle_diff_deg(ta, tb) - 180.0) <= angle_tol_deg:
                uf.union(ia, ib)
                adjacency[ia].append(ib)
                adjacency[ib].append(ia)

    # Component id per corridor, ordered by total component length (desc) for
    # deterministic, human-friendly group numbering. Mainlines first.
    comp_members: Dict[int, List[int]] = {}
    for i in range(n):
        comp_members.setdefault(uf.find(i), []).append(i)

    def comp_len(members):
        return sum(corridors[i].length_m for i in members)

    ordered = sorted(comp_members.values(), key=comp_len, reverse=True)
    for gid, members in enumerate(ordered):
        for order_idx, i in enumerate(_order_within_group(members, adjacency)):
            corridors[i].line_group = gid
            corridors[i].group_order = order_idx


def _order_within_group(members: List[int], adjacency: Dict[int, List[int]]) -> List[int]:
    """Best-effort ordering of corridors along a group by walking through-links."""
    member_set = set(members)
    if len(members) == 1:
        return list(members)
    # Start from an end of the chain (a member with a single in-group link).
    def deg(i):
        return sum(1 for j in adjacency.get(i, []) if j in member_set)
    start = min(members, key=lambda i: (deg(i), i))
    ordered, seen = [], set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        ordered.append(cur)
        for nb in adjacency.get(cur, []):
            if nb in member_set and nb not in seen:
                stack.append(nb)
    # Append any members not reached via links (shouldn't happen within a comp).
    ordered += [i for i in members if i not in seen]
    return ordered
