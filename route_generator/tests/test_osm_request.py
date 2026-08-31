"""Offline tests for the Overpass request/error layer (no real network).

Uses a fake session object so we exercise headers, error surfacing, and query
construction without hitting Overpass.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routegen.osm import (  # noqa: E402
    USER_AGENT,
    build_area_query,
    build_query,
    build_radius_query,
    fetch_area,
    fetch_radius,
)


class _Resp:
    def __init__(self, status_code, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Session:
    """Records the last request and returns a canned response."""

    def __init__(self, resp):
        self._resp = resp
        self.last = None

    def post(self, endpoint, data=None, headers=None, timeout=None):
        self.last = {"endpoint": endpoint, "data": data, "headers": headers}
        return self._resp


def test_area_query_uses_filter_union_not_regex():
    # The railway *value* must use equality, not the anchored regex that caused 406.
    q = build_area_query(39.0, -105.6, 39.3, -105.2, railway_values=("rail", "narrow_gauge"),
                         main_only=False)
    assert '"railway"~' not in q  # no regex on the railway value
    assert '"railway"="rail"' in q and '"railway"="narrow_gauge"' in q
    assert q.count("(39.0,-105.6,39.3,-105.2)") == 2  # bbox applied per filter


def test_relation_query_shape():
    q = build_query(123)
    assert "rel(123)" in q and "way(r)" in q and "out geom" in q


def test_radius_query_uses_around():
    q = build_radius_query(38.25445, -104.60914, 40000, railway_values=("rail", "narrow_gauge"),
                           main_only=False)
    assert "around:40000,38.25445,-104.60914" in q
    assert '"railway"="rail"' in q and '"railway"="narrow_gauge"' in q


def test_main_only_filter_excludes_service_and_industrial():
    q = build_radius_query(38.25, -104.6, 40000, main_only=True)
    assert '"service"!~"."' in q                              # drop sidings/yards
    assert '"usage"!~"industrial|tourism|military"' in q      # drop factory spurs
    q_all = build_radius_query(38.25, -104.6, 40000, main_only=False)
    assert "service" not in q_all and "usage" not in q_all


def test_retry_then_success_with_backoff_zero():
    # Two busy (503) responses, then a 200 -> should succeed after retries.
    seq = [_Resp(503, text="busy"), _Resp(503, text="busy"),
           _Resp(200, payload={"elements": []})]

    class _SeqSession:
        def __init__(self):
            self.calls = 0

        def post(self, *a, **k):
            r = seq[self.calls]
            self.calls += 1
            return r

    from routegen.osm import _post_overpass
    sess = _SeqSession()
    data = _post_overpass("http://x", "q", 10, session=sess, retries=3, backoff_s=0.0,
                          endpoints=["http://a", "http://b"])
    assert data == {"elements": []}
    assert sess.calls == 3  # retried twice, succeeded on third


def test_fetch_radius_sets_metadata_and_headers():
    sess = _Session(_Resp(200, payload={"elements": [
        {"type": "way", "id": 1, "tags": {"railway": "rail"},
         "nodes": [1, 2], "geometry": [{"lat": 38.25, "lon": -104.6}, {"lat": 38.26, "lon": -104.6}]}
    ]}))
    rel = fetch_radius(38.25445, -104.60914, 40000, session=sess)
    assert sess.last["headers"]["User-Agent"] == USER_AGENT
    assert rel.tags["source"] == "radius_query"
    assert rel.tags["center"] == "38.25445,-104.60914"


def test_request_sets_user_agent():
    sess = _Session(_Resp(200, payload={"elements": [
        {"type": "way", "id": 1, "tags": {"railway": "rail"},
         "nodes": [1, 2], "geometry": [{"lat": 39.1, "lon": -105.5}, {"lat": 39.1, "lon": -105.4}]}
    ]}))
    rel = fetch_area(39.0, -105.6, 39.3, -105.2, session=sess)
    assert sess.last["headers"]["User-Agent"] == USER_AGENT
    assert sess.last["headers"]["Accept"] == "application/json"
    assert len(rel.ways) == 1


def test_406_surfaces_body():
    sess = _Session(_Resp(406, text="Not Acceptable: bad something"))
    try:
        fetch_area(39.0, -105.6, 39.3, -105.2, session=sess)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        msg = str(exc)
        assert "406" in msg
        assert "Not Acceptable" in msg          # server body surfaced
        assert "try another endpoint" in msg    # actionable hint


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
