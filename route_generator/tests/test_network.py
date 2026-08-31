"""Offline tests for area-based corridor extraction (no network).

Runnable with pytest or directly: ``python tests/test_network.py``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routegen import extract_corridors, nearest_corridor  # noqa: E402
from routegen.models import RelationData, Way  # noqa: E402


def _way(wid, nodes, coords, tags=None):
    return Way(id=wid, tags=tags or {"railway": "rail"}, node_ids=nodes, coords=coords)


# Helper: build a straight east-west chain of n unit segments sharing node ids,
# so every interior node has endpoint-degree 2 (a continuation).
def _straight_chain():
    # node ids 1..4, ways (1-2),(2-3),(3-4); given out of order to test stitching
    w_a = _way(1, [1, 2], [(0.000, 0.0), (0.010, 0.0)])
    w_b = _way(2, [2, 3], [(0.010, 0.0), (0.020, 0.0)])
    w_c = _way(3, [3, 4], [(0.020, 0.0), (0.030, 0.0)])
    return RelationData(id=-1, tags={}, ways=[w_c, w_a, w_b])


def test_straight_chain_is_one_corridor():
    corridors = extract_corridors(_straight_chain(), min_length_m=100.0)
    assert len(corridors) == 1
    cl = corridors[0]
    # 3 ways, 4 unique vertices after junction de-dup.
    assert cl.num_vertices == 4
    assert sorted(cl.used_way_ids) == [1, 2, 3]


def test_y_junction_splits_into_three_corridors():
    # Three legs meeting at node 100 (degree 3 -> junction; corridors break there).
    # Each leg long enough to clear the min-length filter (~1.1 km per 0.01 deg).
    leg_a = _way(1, [1, 100], [(0.00, 0.00), (0.01, 0.00)])
    leg_b = _way(2, [100, 2], [(0.01, 0.00), (0.02, 0.00)])
    leg_c = _way(3, [100, 3], [(0.01, 0.00), (0.01, 0.01)])
    rel = RelationData(id=-1, tags={}, ways=[leg_a, leg_b, leg_c])
    corridors = extract_corridors(rel, min_length_m=100.0)
    assert len(corridors) == 3
    # Each corridor is a single leg (no continuation across the degree-3 node).
    assert all(c.num_vertices == 2 for c in corridors)


def test_min_length_filter_drops_stubs():
    long_leg = _way(1, [1, 2], [(0.0, 0.0), (0.05, 0.0)])   # ~5.5 km
    stub = _way(2, [3, 4], [(0.10, 0.0), (0.1001, 0.0)])    # ~11 m
    rel = RelationData(id=-1, tags={}, ways=[long_leg, stub])
    corridors = extract_corridors(rel, min_length_m=500.0)
    assert len(corridors) == 1
    assert corridors[0].used_way_ids == [1]


def test_service_track_is_flagged():
    siding = _way(1, [1, 2], [(-105.00, 39.0), (-105.01, 39.0)],
                  tags={"railway": "rail", "service": "siding"})
    rel = RelationData(id=-1, tags={}, ways=[siding])
    cl = extract_corridors(rel, min_length_m=100.0)[0]
    assert cl.is_service is True
    assert cl.service_kind == "siding"


def test_grouping_links_collinear_fragments_across_microgap():
    # Two collinear east-west fragments with NO shared node ids and a ~17 m gap.
    part1 = _way(1, [1, 2], [(-105.50, 39.10), (-105.40, 39.10)])
    part2 = _way(2, [3, 4], [(-105.3998, 39.10), (-105.30, 39.10)])
    # A branch heading south from the same junction area.
    branch = _way(3, [5, 6], [(-105.40, 39.10), (-105.40, 39.05)])
    rel = RelationData(id=-1, tags={}, ways=[part1, part2, branch])
    corridors = extract_corridors(rel, min_length_m=100.0)
    by_way = {c.used_way_ids[0]: c for c in corridors}
    # The two collinear fragments share a parent line; the branch does not.
    assert by_way[1].line_group == by_way[2].line_group
    assert by_way[3].line_group != by_way[1].line_group
    # Within-line ordering is assigned.
    assert {by_way[1].group_order, by_way[2].group_order} == {0, 1}


def test_service_track_not_merged_into_mainline():
    main = _way(1, [1, 2], [(-105.50, 39.10), (-105.40, 39.10)])
    cont = _way(2, [2, 3], [(-105.40, 39.10), (-105.30, 39.10)])
    siding = _way(3, [2, 9], [(-105.40, 39.10), (-105.40, 39.105)],
                  tags={"railway": "rail", "service": "siding"})
    rel = RelationData(id=-1, tags={}, ways=[main, cont, siding])
    corridors = extract_corridors(rel, min_length_m=100.0)
    svc = [c for c in corridors if c.is_service]
    mains = [c for c in corridors if not c.is_service]
    assert len(svc) == 1
    # Service corridor is in a different group from the mainline.
    assert svc[0].line_group not in {c.line_group for c in mains}


def test_nearest_corridor_picks_clicked_line():
    # Two separate lines; a click near the second should select it.
    a = _way(1, [1, 2], [(-105.50, 39.10), (-105.40, 39.10)])   # along lat 39.10
    b = _way(2, [3, 4], [(-105.50, 39.20), (-105.40, 39.20)])   # along lat 39.20
    rel = RelationData(id=-1, tags={}, ways=[a, b])
    corridors = extract_corridors(rel, min_length_m=100.0)
    # Click near line b (lat ~39.20).
    idx, dist = nearest_corridor(corridors, 39.199, -105.45)
    assert corridors[idx].used_way_ids == [2]
    assert dist < 200  # within ~150 m


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
