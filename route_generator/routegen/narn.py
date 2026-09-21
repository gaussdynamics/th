"""Traverse the NARN main-line *graph* into ordered corridors.

`scripts/pull_narn.py` leaves a graph on disk: 95,936 segments averaging 2.9 km,
joined at 92,787 shared nodes. The simulator needs continuous alignments with
monotonic chainage. This module is the stage between them, and it is deliberately
thin: NARN segments are adapted into :class:`~.models.Way` objects so the corridor
walk already written for OSM (`network._grow_corridor`) applies unchanged.

NARN is *better* input than OSM here. ``FRFRANODE``/``TOFRANODE`` are
authoritative, so the coordinate-fallback and gap-bridging heuristics in
`centerline.py` never run; connectivity is read, not inferred.

The unit of traversal is the **subdivision** -- how a railroad divides its own
network -- keyed by ``(RROWNER1, SUBDIV)`` because names are reused across owners.
Every decision this module makes is recorded in `ROUTE_PIPELINE_NOTE.md`, Phase B.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .centerline import make_centerline_from_chain
from .geometry import haversine_m
from .models import Centerline, Way
from .network import _endpoint_adjacency, _grow_corridor

# A corridor shorter than this cannot host a full-length scenario: DATA_SCHEMA
# wants 300-600 s runs, which at line speed is 8-15 km of track, and the
# randomizer clips duration so the consist cannot run off the end.
DEFAULT_MIN_LENGTH_M = 15_000.0
# Above this a corridor is cut into pieces. The existing five-corridor set spans
# 22-49 km; more importantly the whole route array is resident per scenario, and
# disk is the binding constraint at Phase C scale.
DEFAULT_MAX_LENGTH_M = 60_000.0

# Track-class assignment. NARN carries no line speed, so `profile.derive_profile`
# falls back to an FRA freight class. Single track keeps class 4 (60 mph), which
# is exactly the fallback `data/v1` and `data/v2` were built with, so single-track
# route_vmax -- and the overspeed statistics measured against it -- stay
# comparable across the rebuild. Only multi-track and passenger-carrying route
# deviates, upward; an unknown track count deviates downward.
FRA_CLASS_MULTI_TRACK = 5   # 80 mph: 2+ main tracks, or hosts passenger service
FRA_CLASS_SINGLE = 4        # 60 mph: one main track -- the v1/v2 baseline
FRA_CLASS_UNKNOWN = 3       # 40 mph: TRACKS absent or zero
_FRA_CLASS_MPH = {1: 10, 2: 25, 3: 40, 4: 60, 5: 80}


@dataclass
class CorridorMeta:
    """Provenance for one extracted corridor, written to the manifest."""

    corridor_id: str
    owner: str
    subdiv: str
    division: str
    states: str
    country: str
    length_m: float
    num_vertices: int
    num_segments: int
    piece_index: int
    num_pieces: int
    was_cut: bool
    fra_class_min: int
    fra_class_max: int
    tracks_max: float
    has_passenger: bool
    first_arcid: int


@dataclass
class TraversalReport:
    """What the traversal did to the whole network, for the build log."""

    n_segments_in: int = 0
    n_segments_no_subdiv: int = 0
    km_no_subdiv: float = 0.0
    n_groups: int = 0
    n_groups_simple_path: int = 0
    n_groups_branched: int = 0
    n_groups_multicomponent: int = 0
    n_chains: int = 0
    n_chains_too_short: int = 0
    n_chains_cut: int = 0
    n_corridors: int = 0
    n_corridors_over_cap: int = 0
    km_in: float = 0.0
    km_out: float = 0.0
    dropped_short_km: float = 0.0
    corridors: List[CorridorMeta] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_narn(narn_dir: str):
    """Load ``segments.parquet`` + ``geometry.npz`` written by `pull_narn.py`."""
    from pathlib import Path

    import pandas as pd

    d = Path(narn_dir)
    df = pd.read_parquet(d / "segments.parquet")
    z = np.load(d / "geometry.npz")
    return df, np.asarray(z["lon"], float), np.asarray(z["lat"], float)


# --------------------------------------------------------------------------- #
# Segment -> Way adaptation
# --------------------------------------------------------------------------- #
def _fra_class(tracks, passngr) -> int:
    """Pick an FRA freight track class from NARN's track count / passenger code."""
    has_pass = isinstance(passngr, str) and passngr.strip().lower() not in ("", "nan", "none")
    try:
        t = float(tracks)
    except (TypeError, ValueError):
        t = float("nan")
    if math.isnan(t) or t < 1:
        # An unknown track count is not evidence of a fast railroad.
        return FRA_CLASS_MULTI_TRACK if has_pass else FRA_CLASS_UNKNOWN
    if t >= 2 or has_pass:
        return FRA_CLASS_MULTI_TRACK
    return FRA_CLASS_SINGLE


def segment_to_way(row, lon: np.ndarray, lat: np.ndarray) -> Way:
    """Adapt one NARN segment row into a :class:`Way`.

    Vertex order runs ``FRFRANODE -> TOFRANODE``; verified against 4,994 junction
    pairs on the continental pull with zero exceptions, so the node ids and the
    geometry agree and `Way.reversed()` is meaningful.

    NARN has no structure tags, so ``bridge``/``tunnel``/``cutting``/``embankment``
    are simply absent and `tag_is_truthy` reports False for all of them. Those
    flags are provenance only -- nothing in the RHS reads them.
    """
    a, b = int(row.vertex_start), int(row.vertex_end)
    coords = [(float(x), float(y)) for x, y in zip(lon[a:b], lat[a:b])]
    cls = _fra_class(row.TRACKS, row.PASSNGR)
    tags = {
        # Written as an OSM-style maxspeed string so `make_centerline_from_chain`
        # picks it up through the existing parser rather than a parallel path.
        "maxspeed": "%d mph" % _FRA_CLASS_MPH[cls],
        "narn:fra_class": str(cls),
        "narn:owner": str(row.RROWNER1),
        "narn:subdiv": str(row.SUBDIV),
        "narn:tracks": str(row.TRACKS),
    }
    return Way(id=int(row.FRAARCID), tags=tags,
               node_ids=[int(row.FRFRANODE), int(row.TOFRANODE)], coords=coords)


# --------------------------------------------------------------------------- #
# Grouping and traversal
# --------------------------------------------------------------------------- #
def subdivision_groups(df) -> Iterator[Tuple[str, str, object]]:
    """Yield ``(owner, subdiv, rows)`` per subdivision, in deterministic order.

    Segments with no ``SUBDIV`` (5,230 of 95,936 on the continental pull) are
    excluded: they carry no parent-line identity, so a corridor built from them
    could not be named or reserved for a split. The cost is that corridors may
    terminate early where such trackage would have connected them.
    """
    sub = df[df["SUBDIV"].notna()]
    for (owner, subdiv), rows in sub.groupby(["RROWNER1", "SUBDIV"], sort=True):
        yield str(owner), str(subdiv), rows


def _classify_group(ways: List[Way], adj: Dict[int, List[int]]) -> Tuple[bool, bool]:
    """Return ``(has_branch, is_multicomponent)`` for a subdivision subgraph."""
    has_branch = any(len(v) > 2 for v in adj.values())
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for w in ways:
        for n in (w.start_node, w.end_node):
            parent.setdefault(n, n)
        ra, rb = find(w.start_node), find(w.end_node)
        if ra != rb:
            parent[ra] = rb
    ncomp = len({find(n) for n in parent})
    return has_branch, ncomp > 1


def traverse_group(ways: List[Way]) -> List[List[Way]]:
    """Walk one subdivision subgraph into maximal ordered chains.

    A branch node (degree >= 3) *ends* a chain rather than being resolved: both
    arms come out as separate corridors with real identities. Guessing which arm
    is "the" main line would put a heuristic into a pipeline whose whole point is
    that NARN gives topology explicitly. Disconnected components likewise fall
    out as separate chains -- the walk cannot cross a gap.
    """
    adj = _endpoint_adjacency(ways)
    degree = {node: len(idxs) for node, idxs in adj.items()}
    used: set = set()
    chains: List[List[Way]] = []
    for seed in sorted(range(len(ways)), key=lambda i: ways[i].id):
        if seed in used:
            continue
        chains.append(_grow_corridor(seed, ways, adj, degree, used))
    return chains


# --------------------------------------------------------------------------- #
# Cutting long chains
# --------------------------------------------------------------------------- #
def _way_length_m(w: Way) -> float:
    c = w.coords
    return sum(haversine_m(c[i][0], c[i][1], c[i + 1][0], c[i + 1][1])
               for i in range(len(c) - 1))


def _chain_length_m(chain: Sequence[Way]) -> float:
    return float(sum(_way_length_m(w) for w in chain))


def split_way(w: Way, max_length_m: float) -> List[Way]:
    """Split one over-long segment at its own vertices into near-equal sub-ways.

    NARN segments average 2.9 km but the tail is long -- remote Canadian and
    Mexican route runs to 178 km in a single segment -- so a cut that only lands
    on segment boundaries cannot bound corridor length. Sub-ways keep the parent
    ``FRAARCID`` as their id, so provenance survives; their ``node_ids`` are
    inherited and meaningless, which is safe because splitting happens strictly
    *after* traversal, when adjacency is no longer consulted.
    """
    total = _way_length_m(w)
    if total <= max_length_m or len(w.coords) < 3:
        return [w]
    n = int(math.ceil(total / max_length_m))
    target = total / n
    out: List[Way] = []
    cur = [w.coords[0]]
    acc = 0.0
    for i in range(len(w.coords) - 1):
        a, b = w.coords[i], w.coords[i + 1]
        acc += haversine_m(a[0], a[1], b[0], b[1])
        cur.append(b)
        if acc >= target and len(out) < n - 1 and len(cur) >= 2:
            out.append(Way(id=w.id, tags=w.tags, node_ids=list(w.node_ids), coords=cur))
            cur = [b]
            acc = 0.0
    if len(cur) >= 2:
        out.append(Way(id=w.id, tags=w.tags, node_ids=list(w.node_ids), coords=cur))
    return out


def cut_chain(chain: List[Way], max_length_m: float,
              min_length_m: float) -> List[List[Way]]:
    """Split an over-long chain into near-equal pieces, capped at ``max_length_m``.

    Cuts land on *segment boundaries* wherever they can: that keeps ``way_id``
    provenance intact and means a piece is a whole number of NARN segments,
    traceable back to its ``FRAARCID`` set. Where a single segment is itself
    longer than a quarter of the cap, `split_way` subdivides it first, so the
    boundary cutter always has fine enough granularity to respect the cap.

    The cap is a hard bound, not a target: a piece closes before taking on a way
    that would carry it over. Evenness is a secondary rule applied within that.
    """
    granularity = max_length_m / 4.0
    chain = [sw for w in chain for sw in split_way(w, granularity)]
    total = _chain_length_m(chain)
    if total <= max_length_m or len(chain) < 2:
        return [chain]
    n_pieces = int(math.ceil(total / max_length_m))
    target = total / n_pieces
    pieces: List[List[Way]] = []
    cur: List[Way] = []
    cur_len = 0.0
    for w in chain:
        wl = _way_length_m(w)
        over_cap = cur_len + wl > max_length_m
        # Secondary: close when adding this way would overshoot the even-split
        # target by more than leaving it out undershoots -- nearest boundary,
        # not first past it.
        past_target = (len(pieces) < n_pieces - 1 and cur_len >= min_length_m
                       and abs(cur_len + wl - target) > abs(cur_len - target))
        if cur and (over_cap or past_target):
            pieces.append(cur)
            cur, cur_len = [], 0.0
        cur.append(w)
        cur_len += wl
    if cur:
        pieces.append(cur)
    return pieces


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s or "unnamed"


def corridor_id(owner: str, subdiv: str, length_m: float,
                piece_index: int, num_pieces: int) -> str:
    """``bnsf_raton_49km``, or ``up_moffat_tunnel_p2_58km`` when cut.

    Corridors get real identities rather than `route_line2`, which was one of the
    stated reasons for the NARN pivot. Length is in the id because every existing
    route id carries it and the dataset index gets read by eye.
    """
    base = "%s_%s" % (_slug(owner), _slug(subdiv))
    if num_pieces > 1:
        base += "_p%d" % piece_index
    return "%s_%dkm" % (base, int(round(length_m / 1000.0)))


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #
def build_corridors(
    df,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    min_length_m: float = DEFAULT_MIN_LENGTH_M,
    max_length_m: float = DEFAULT_MAX_LENGTH_M,
    limit_groups: Optional[int] = None,
) -> Tuple[List[Centerline], TraversalReport]:
    """Traverse the whole NARN table into named, ordered corridors."""
    report = TraversalReport()
    report.n_segments_in = int(len(df))
    report.n_segments_no_subdiv = int(df["SUBDIV"].isna().sum())
    report.km_in = float(df["KM"].sum())
    report.km_no_subdiv = float(df.loc[df["SUBDIV"].isna(), "KM"].sum())

    out: List[Centerline] = []
    for gi, (owner, subdiv, rows) in enumerate(subdivision_groups(df)):
        if limit_groups is not None and gi >= limit_groups:
            break
        report.n_groups += 1
        ways = [segment_to_way(r, lon, lat) for r in rows.itertuples()
                if int(r.vertex_end) - int(r.vertex_start) >= 2]
        if not ways:
            continue

        adj = _endpoint_adjacency(ways)
        has_branch, multi = _classify_group(ways, adj)
        if has_branch:
            report.n_groups_branched += 1
        if multi:
            report.n_groups_multicomponent += 1
        if not has_branch and not multi:
            report.n_groups_simple_path += 1

        chains = traverse_group(ways)
        report.n_chains += len(chains)

        pieces: List[List[Way]] = []
        for chain in chains:
            cut = cut_chain(chain, max_length_m, min_length_m)
            if len(cut) > 1:
                report.n_chains_cut += 1
            pieces.extend(cut)

        kept: List[List[Way]] = []
        for piece in pieces:
            plen = _chain_length_m(piece)
            if plen < min_length_m:
                report.n_chains_too_short += 1
                report.dropped_short_km += plen / 1000.0
                continue
            kept.append(piece)

        for pi, piece in enumerate(kept):
            first = piece[0]
            cl = make_centerline_from_chain(
                relation_id=int(first.id),  # no relation exists; FRAARCID anchors it
                relation_tags={"source": "narn", "owner": owner, "subdiv": subdiv},
                chain=piece,
            )
            cid = corridor_id(owner, subdiv, cl.length_m, pi, len(kept))
            cl.relation_tags["corridor_id"] = cid
            classes = [int(w.tags["narn:fra_class"]) for w in piece]
            arcids = {w.id for w in piece}
            srows = rows[rows["FRAARCID"].isin(arcids)]
            tracks = srows["TRACKS"].to_numpy(dtype=float)

            def _mode(col: str) -> str:
                """Most common value, or "" -- a column can be entirely null."""
                m = srows[col].dropna().mode()
                return str(m.iat[0]) if len(m) else ""

            out.append(cl)
            report.corridors.append(CorridorMeta(
                corridor_id=cid, owner=owner, subdiv=subdiv,
                division=_mode("DIVISION"),
                states="|".join(sorted(set(srows["STATEAB"].dropna().astype(str)))),
                country=_mode("COUNTRY"),
                length_m=float(cl.length_m), num_vertices=int(cl.num_vertices),
                num_segments=len(piece), piece_index=pi, num_pieces=len(kept),
                was_cut=len(kept) > 1,
                fra_class_min=min(classes), fra_class_max=max(classes),
                tracks_max=float(np.nanmax(tracks)) if np.isfinite(tracks).any()
                else float("nan"),
                has_passenger=bool(srows["PASSNGR"].notna().any()) if len(srows) else False,
                first_arcid=int(first.id),
            ))
            report.km_out += cl.length_m / 1000.0

    report.n_corridors = len(out)
    report.n_corridors_over_cap = sum(
        1 for m in report.corridors if m.length_m > max_length_m)
    return out, report
