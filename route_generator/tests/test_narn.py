"""Offline tests for NARN graph -> corridor traversal (no network, no data files).

Runnable with pytest or directly: ``python tests/test_narn.py``.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routegen.narn import _way_length_m as _way_length_m_  # noqa: E402
from routegen.narn import (  # noqa: E402
    FRA_CLASS_MULTI_TRACK,
    FRA_CLASS_SINGLE,
    FRA_CLASS_UNKNOWN,
    _chain_length_m,
    _fra_class,
    build_corridors,
    corridor_id,
    cut_chain,
    segment_to_way,
    split_way,
    traverse_group,
)


# --------------------------------------------------------------------------- #
# A tiny in-memory NARN: rows carry the same fields the parquet does.
# --------------------------------------------------------------------------- #
class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _seg(arcid, a, b, vs, ve, tracks=1.0, passngr=None, subdiv="TEST",
         owner="XX", division="DIV", state="CO", country="US"):
    return _Row(FRAARCID=arcid, FRFRANODE=a, TOFRANODE=b, KM=0.0,
                vertex_start=vs, vertex_end=ve, TRACKS=tracks, PASSNGR=passngr,
                SUBDIV=subdiv, RROWNER1=owner, DIVISION=division,
                STATEAB=state, COUNTRY=country)


def _line(n_pts, lon0, dlon, lat=40.0):
    """A straight east-west run of ``n_pts`` vertices starting at ``lon0``."""
    return ([lon0 + i * dlon for i in range(n_pts)], [lat] * n_pts)


def _geom(*chunks):
    lon, lat = [], []
    bounds = []
    for c_lon, c_lat in chunks:
        bounds.append((len(lon), len(lon) + len(c_lon)))
        lon.extend(c_lon)
        lat.extend(c_lat)
    return np.array(lon), np.array(lat), bounds


# --------------------------------------------------------------------------- #
# Traversal
# --------------------------------------------------------------------------- #
def test_three_segments_become_one_ordered_corridor():
    """Segments given out of order must come back as one monotonic chain."""
    lon, lat, b = _geom(_line(3, 0.0, 0.01), _line(3, 0.02, 0.01), _line(3, 0.04, 0.01))
    rows = [_seg(300, 1, 2, *b[0]), _seg(301, 2, 3, *b[1]), _seg(302, 3, 4, *b[2])]
    # Present them shuffled; the walk must not depend on input order.
    ways = [segment_to_way(r, lon, lat) for r in [rows[2], rows[0], rows[1]]]
    chains = traverse_group(ways)
    assert len(chains) == 1
    assert [w.id for w in chains[0]] == [300, 301, 302]
    xs = [c[0] for w in chains[0] for c in w.coords]
    assert all(x2 >= x1 for x1, x2 in zip(xs, xs[1:])), "chainage must be monotonic"


def test_reversed_segment_is_flipped_into_the_chain():
    """A segment digitized against the run direction must be reversed, not dropped."""
    lon, lat, b = _geom(_line(3, 0.0, 0.01), _line(3, 0.04, -0.01))
    # Second segment runs 0.04 -> 0.02, i.e. node 3 -> node 2: backwards.
    rows = [_seg(300, 1, 2, *b[0]), _seg(301, 3, 2, *b[1])]
    ways = [segment_to_way(r, lon, lat) for r in rows]
    chains = traverse_group(ways)
    assert len(chains) == 1
    xs = [c[0] for w in chains[0] for c in w.coords]
    assert all(x2 >= x1 for x1, x2 in zip(xs, xs[1:])), "reversed way was not flipped"


def test_branch_splits_into_separate_corridors():
    """A Y must come back as separate arms, never as a guessed through-route."""
    lon, lat, b = _geom(_line(3, 0.0, 0.01),   # stem, into node 2
                        _line(3, 0.02, 0.01),  # arm A
                        _line(3, 0.02, 0.01))  # arm B
    rows = [_seg(300, 1, 2, *b[0]), _seg(301, 2, 3, *b[1]), _seg(302, 2, 4, *b[2])]
    ways = [segment_to_way(r, lon, lat) for r in rows]
    chains = traverse_group(ways)
    assert len(chains) == 3, "degree-3 node must terminate every incident chain"
    assert sorted(len(c) for c in chains) == [1, 1, 1]


def test_disconnected_components_are_not_stitched():
    """Two unconnected runs in one subdivision stay two corridors."""
    lon, lat, b = _geom(_line(3, 0.0, 0.01), _line(3, 5.0, 0.01))
    rows = [_seg(300, 1, 2, *b[0]), _seg(301, 7, 8, *b[1])]
    ways = [segment_to_way(r, lon, lat) for r in rows]
    chains = traverse_group(ways)
    assert len(chains) == 2, "the walk must not bridge a gap between components"


def test_traversal_is_deterministic_regardless_of_input_order():
    lon, lat, b = _geom(*[_line(3, 0.02 * i, 0.01) for i in range(5)])
    rows = [_seg(300 + i, i + 1, i + 2, *b[i]) for i in range(5)]
    ways = [segment_to_way(r, lon, lat) for r in rows]
    a = [[w.id for w in c] for c in traverse_group(list(ways))]
    bb = [[w.id for w in c] for c in traverse_group(list(reversed(ways)))]
    assert a == bb


# --------------------------------------------------------------------------- #
# Cutting
# --------------------------------------------------------------------------- #
def _uniform_chain(n_ways, dlon=0.03):
    """``n_ways`` collinear ways of equal length, laid end to end."""
    lon, lat, b = _geom(*[_line(2, i * dlon, dlon) for i in range(n_ways)])
    rows = [_seg(300 + i, i + 1, i + 2, *b[i]) for i in range(n_ways)]
    return [segment_to_way(r, lon, lat) for r in rows]


def test_short_chain_is_not_cut():
    chain = _uniform_chain(10)
    assert len(cut_chain(chain, _chain_length_m(chain) * 2, 0.0)) == 1


def test_long_chain_is_cut_into_pieces_within_bound():
    chain = _uniform_chain(100)
    total = _chain_length_m(chain)
    cap = total / 5.0
    pieces = cut_chain(chain, cap, cap / 4.0)
    assert len(pieces) >= 5
    for p in pieces:
        assert _chain_length_m(p) <= cap, "the cap is a hard bound, not a target"


def test_cut_preserves_every_segment_exactly_once_and_in_order():
    chain = _uniform_chain(100)
    total = _chain_length_m(chain)
    pieces = cut_chain(chain, total / 5.0, total / 20.0)
    flat = [w.id for p in pieces for w in p]
    assert flat == [w.id for w in chain], "cutting must not drop, duplicate or reorder"


def test_single_over_long_segment_is_split_at_its_vertices():
    """A 2-segment chain cannot be cut on boundaries alone; vertices must serve."""
    lon, lat, b = _geom(_line(200, 0.0, 0.01))
    w = segment_to_way(_seg(300, 1, 2, *b[0]), lon, lat)
    cap = _way_length_m_(w) / 4.0
    parts = split_way(w, cap)
    assert len(parts) >= 4
    assert all(_chain_length_m([q]) <= cap * 1.01 for q in parts)
    assert all(q.id == 300 for q in parts), "sub-ways must keep the parent FRAARCID"
    # Vertices are preserved end to end, with the cut vertex shared by both sides.
    rebuilt = [parts[0].coords[0]]
    for q in parts:
        rebuilt.extend(q.coords[1:])
    assert rebuilt == w.coords


def test_chain_of_one_giant_segment_still_respects_the_cap():
    """The 178 km single-segment corridors must come out bounded."""
    lon, lat, b = _geom(_line(400, 0.0, 0.01))
    chain = [segment_to_way(_seg(300, 1, 2, *b[0]), lon, lat)]
    total = _chain_length_m(chain)
    cap = total / 5.0
    pieces = cut_chain(chain, cap, cap / 4.0)
    assert len(pieces) >= 5
    for q in pieces:
        assert _chain_length_m(q) <= cap, "a single long segment escaped the cap"


def test_unsplittable_two_vertex_segment_is_left_whole():
    """With only two vertices there is nowhere to cut; return it rather than lie."""
    lon, lat, b = _geom(_line(2, 0.0, 5.0))
    w = segment_to_way(_seg(300, 1, 2, *b[0]), lon, lat)
    assert split_way(w, 1_000.0) == [w]


# --------------------------------------------------------------------------- #
# Attribute mapping
# --------------------------------------------------------------------------- #
def test_fra_class_from_tracks_and_passenger():
    assert _fra_class(1.0, None) == FRA_CLASS_SINGLE          # v1/v2 baseline
    assert _fra_class(2.0, None) == FRA_CLASS_MULTI_TRACK
    assert _fra_class(1.0, "A") == FRA_CLASS_MULTI_TRACK      # Amtrak route
    assert _fra_class(0.0, None) == FRA_CLASS_UNKNOWN
    assert _fra_class(float("nan"), None) == FRA_CLASS_UNKNOWN
    assert _fra_class(None, None) == FRA_CLASS_UNKNOWN


def test_single_track_maxspeed_matches_the_v1_v2_fallback():
    """Single track must resolve to 60 mph, or the rebuild is not comparable."""
    lon, lat, b = _geom(_line(2, 0.0, 0.01))
    w = segment_to_way(_seg(300, 1, 2, *b[0], tracks=1.0), lon, lat)
    from routegen.osm import parse_maxspeed_kph
    assert math.isclose(parse_maxspeed_kph(w.tags["maxspeed"]) / 3.6,
                        60 * 0.44704, rel_tol=1e-3)


def test_structure_flags_are_all_false():
    """NARN carries no structure tags; the flags must be absent, not guessed."""
    lon, lat, b = _geom(_line(3, 0.0, 0.01), _line(3, 0.02, 0.01))
    rows = [_seg(300, 1, 2, *b[0]), _seg(301, 2, 3, *b[1])]
    df = _FakeFrame(rows)
    cors, _ = build_corridors(df, lon, lat, min_length_m=0.0)
    cl = cors[0]
    assert not cl.is_bridge.any() and not cl.is_tunnel.any()
    assert not cl.is_cutting.any() and not cl.is_embankment.any()


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #
def test_corridor_id_is_a_real_identity():
    assert corridor_id("BNSF", "RATON", 49_200.0, 0, 1) == "bnsf_raton_49km"
    assert corridor_id("UP", "MOFFAT TUNNEL", 58_400.0, 2, 5) == "up_moffat_tunnel_p2_58km"
    assert corridor_id("CSXT", "S&NA NORTH", 30_000.0, 0, 1) == "csxt_s_na_north_30km"


# --------------------------------------------------------------------------- #
# End to end on a fake frame
# --------------------------------------------------------------------------- #
class _FakeFrame:
    """The slice of the pandas API `build_corridors` actually uses."""

    def __init__(self, rows):
        import pandas as pd
        self._df = pd.DataFrame([r.__dict__ for r in rows])

    def __getattr__(self, name):
        return getattr(self._df, name)

    def __getitem__(self, k):
        return self._df[k]

    def __len__(self):
        return len(self._df)


def test_end_to_end_reports_and_drops_short_corridors():
    lon, lat, b = _geom(*[_line(2, i * 0.05, 0.05) for i in range(3)])
    rows = [_seg(300 + i, i + 1, i + 2, *b[i]) for i in range(3)]
    rows[0].SUBDIV = None  # a subdivision-less segment must be excluded
    df = _FakeFrame(rows)
    cors, rep = build_corridors(df, lon, lat, min_length_m=1_000.0)
    assert rep.n_segments_no_subdiv == 1
    assert rep.n_corridors == len(cors) == 1
    assert cors[0].relation_tags["corridor_id"].startswith("xx_test_")
    assert np.all(np.diff(cors[0].s_m) > 0), "chainage must be strictly increasing"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print("FAIL", name, "--", exc)
    print("\n%d failure(s)" % fails)
    sys.exit(1 if fails else 0)
