"""Offline tests for OSM parsing and centerline assembly (no network).

Runnable either with pytest or directly: ``python tests/test_centerline.py``.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routegen import (  # noqa: E402
    assemble_centerline,
    filter_track_ways,
    parse_maxspeed_kph,
    parse_overpass_json,
)
from routegen.models import RelationData, Way  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_relation_overpass.json")


def _load_fixture() -> RelationData:
    with open(FIXTURE) as fh:
        return parse_overpass_json(json.load(fh))


def test_parse_and_filter():
    rel = _load_fixture()
    assert rel.id == 999001
    assert rel.tags.get("name") == "Fixture Test Line"
    assert len(rel.ways) == 4  # includes the platform way
    track = filter_track_ways(rel)
    assert len(track.ways) == 3  # platform (id 77) dropped
    assert {w.id for w in track.ways} == {10, 11, 12}


def test_assemble_linear_order_and_orientation():
    cl = assemble_centerline(filter_track_ways(_load_fixture()))
    # Ways must be reordered 10 -> 11 -> 12, with 11 reversed to connect at node 3.
    assert cl.order == [(10, False), (11, True), (12, False)]
    assert cl.num_vertices == 6  # shared junction vertices de-duplicated
    assert cl.dropped_way_ids == []
    assert cl.gaps == []
    assert cl.branches == []
    # Longitudes should increase monotonically 0.000 .. 0.005.
    lons = cl.coords[:, 0]
    assert np.allclose(lons, [0.0, 0.001, 0.002, 0.003, 0.004, 0.005])
    # ~111.32 m per 0.001 deg at the equator, 5 segments.
    assert 540.0 < cl.length_m < 575.0


def test_per_vertex_tag_channels():
    cl = assemble_centerline(filter_track_ways(_load_fixture()))
    # Source way per vertex: [10,10,10,11,11,12]
    assert cl.way_id.tolist() == [10, 10, 10, 11, 11, 12]
    # Bridge flag on way 11's two vertices only.
    assert cl.is_bridge.tolist() == [False, False, False, True, True, False]
    # maxspeed: 60 km/h (way10), NaN (way11, no maxspeed), 50 mph -> ~80.47 (way12)
    assert cl.maxspeed_kph[0] == 60.0
    assert math.isnan(cl.maxspeed_kph[3])
    assert abs(cl.maxspeed_kph[5] - 50 * 1.609344) < 1e-6
    # Coverage: 3 of 5 segments have known maxspeed.
    assert abs(cl.maxspeed_coverage_fraction() - 0.6) < 1e-6


def test_maxspeed_parsing():
    assert parse_maxspeed_kph("60") == 60.0
    assert abs(parse_maxspeed_kph("50 mph") - 80.4672) < 1e-3
    assert parse_maxspeed_kph("100 km/h") == 100.0
    assert math.isnan(parse_maxspeed_kph("none"))
    assert math.isnan(parse_maxspeed_kph(None))
    assert math.isnan(parse_maxspeed_kph(""))


def _way(wid, nodes, coords, tags=None):
    return Way(id=wid, tags=tags or {"railway": "rail"}, node_ids=nodes, coords=coords)


def test_gap_detection():
    # Two ways that do NOT share a node id and are ~22 m apart at the joint.
    a = _way(1, [1, 2], [(0.0, 0.0), (0.001, 0.0)])
    b = _way(2, [3, 4], [(0.0012, 0.0), (0.0022, 0.0)])
    rel = RelationData(id=1, tags={}, ways=[a, b])
    cl = assemble_centerline(rel, gap_tolerance_m=5.0)
    assert len(cl.gaps) == 1
    g = cl.gaps[0]
    assert g.after_way_id == 1 and g.before_way_id == 2
    assert 15.0 < g.distance_m < 30.0  # ~22 m
    assert cl.dropped_way_ids == []  # bridged, not dropped


def test_branch_detection():
    # Three ways meeting at node 2 -> a branch (degree 3).
    a = _way(1, [1, 2], [(0.0, 0.0), (0.001, 0.0)])
    b = _way(2, [2, 3], [(0.001, 0.0), (0.002, 0.0)])
    c = _way(3, [2, 4], [(0.001, 0.0), (0.001, 0.001)])
    rel = RelationData(id=2, tags={}, ways=[a, b, c])
    cl = assemble_centerline(rel)
    assert any(br.node_id == 2 and br.degree == 3 for br in cl.branches)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
