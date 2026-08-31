"""OSM acquisition via the Overpass API.

``fetch_relation`` performs the network call; ``parse_overpass_json`` is a pure
function (no network) so the parsing/assembly path is fully unit-testable offline.
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional

from .models import TRACK_RAILWAY_VALUES, RelationData, Way

DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"

# Public mirrors (the main instance is often busy / rate-limited). The UI offers
# these as alternatives; all speak the same Overpass QL.
KNOWN_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# OSM usage policy expects a descriptive User-Agent; the bare python-requests UA
# is frequently rejected (sometimes as HTTP 406/403). Identify the tool.
USER_AGENT = "route_generator/0.1 (railway LTD dataset; OSM Overpass client)"

# Truthy values for structure tags (anything not in the "false" set counts as present).
_FALSE_TAG_VALUES = frozenset({"", "no", "false", "0", "none"})

# Way-filter for "main lines only": drop any service track (siding/spur/yard/
# crossover) and industrial/tourism/military usage, so factory/plant spurs and
# yards are excluded at the query level. Negated regex tag-filters also match ways
# that lack the key, so untagged-usage main/branch lines are kept.
_MAIN_FILTER = '["service"!~"."]["usage"!~"industrial|tourism|military"]'

# HTTP statuses worth retrying (server busy / overloaded / transient).
_RETRY_STATUS = frozenset({429, 502, 503, 504})


def _way_filter(main_only: bool) -> str:
    return _MAIN_FILTER if main_only else ""


def endpoint_rotation(primary: str) -> List[str]:
    """Primary endpoint first, then the other known mirrors (for failover)."""
    return [primary] + [e for e in KNOWN_ENDPOINTS if e != primary]


def _post_overpass(
    endpoint: str,
    query: str,
    timeout_s: int,
    session=None,
    retries: int = 3,
    backoff_s: float = 5.0,
    endpoints: Optional[List[str]] = None,
) -> dict:
    """POST an Overpass QL query and return parsed JSON, with retry + failover.

    Retries on connection/read timeouts and busy statuses (429/502/503/504) with
    exponential backoff + jitter, rotating across ``endpoints`` (mirrors) each
    attempt. Non-retryable errors (e.g. 400 syntax) raise immediately with the
    server's response body included.
    """
    import requests  # local import: package imports without network deps loaded

    eps = endpoints or [endpoint]
    http = session or requests
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    last = "unknown error"

    for attempt in range(retries + 1):
        ep = eps[attempt % len(eps)]
        try:
            resp = http.post(ep, data={"data": query}, headers=headers,
                             timeout=(15, timeout_s + 30))  # (connect, read)
        except Exception as exc:  # noqa: BLE001 - includes Timeout/ConnectionError
            last = f"network error: {exc}"
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise RuntimeError("Overpass response was not valid JSON.") from exc
            body = (getattr(resp, "text", "") or "").strip().replace("\n", " ")
            if resp.status_code in _RETRY_STATUS:
                last = f"HTTP {resp.status_code} (busy)"
            else:
                hint = " (request rejected — try another endpoint)" \
                    if resp.status_code in (403, 406) else \
                    (" (query syntax error)" if resp.status_code == 400 else "")
                raise RuntimeError(f"Overpass HTTP {resp.status_code}{hint}. {body[:300]}")
        if attempt < retries:
            time.sleep(backoff_s * (2 ** attempt) + random.uniform(0.0, 1.5))

    raise RuntimeError(
        f"Overpass failed after {retries + 1} attempt(s): {last}. "
        f"The servers may be busy — retry later, lower the radius, or use --no-crawl."
    )


def build_query(relation_id: int, timeout_s: int = 90) -> str:
    """Overpass QL: relation tags + member ways with tags, node ids, and geometry."""
    return (
        f"[out:json][timeout:{timeout_s}];\n"
        f"rel({int(relation_id)});\n"
        f"out tags;\n"
        f"way(r);\n"
        f"out geom;\n"
    )


def build_area_query(
    south: float, west: float, north: float, east: float,
    railway_values=("rail", "narrow_gauge"),
    timeout_s: int = 120,
    main_only: bool = True,
) -> str:
    """Overpass QL: track ways within a bounding box (with geometry).

    Uses a union of plain equality filters rather than an anchored regex — the
    regex form is occasionally rejected by Overpass frontends/WAFs when
    form-encoded (a common cause of spurious HTTP 406). With ``main_only`` it
    excludes yards/sidings/industrial spurs (see ``_MAIN_FILTER``).
    """
    bbox = f"{south},{west},{north},{east}"
    flt = _way_filter(main_only)
    parts = "\n  ".join(f'way["railway"="{v}"]{flt}({bbox});' for v in railway_values)
    return (
        f"[out:json][timeout:{timeout_s}];\n"
        f"(\n  {parts}\n);\n"
        f"out geom;\n"
    )


def build_radius_query(
    lat: float, lon: float, radius_m: float,
    railway_values=("rail", "narrow_gauge"),
    timeout_s: int = 120,
    main_only: bool = True,
) -> str:
    """Overpass QL: track ways within ``radius_m`` of a point (with geometry).

    Uses the ``around`` filter. A way is selected if *any* part of it lies within
    the radius, and Overpass returns the whole way (geometry is not clipped), so a
    line passing through the ball is captured in full at the way level. With
    ``main_only`` it excludes yards/sidings/industrial spurs.
    """
    center = f"{radius_m},{lat},{lon}"
    flt = _way_filter(main_only)
    parts = "\n  ".join(f'way["railway"="{v}"]{flt}(around:{center});' for v in railway_values)
    return (
        f"[out:json][timeout:{timeout_s}];\n"
        f"(\n  {parts}\n);\n"
        f"out geom;\n"
    )


def build_multi_around_query(
    points_latlon,
    radius_m: float,
    railway_values=("rail", "narrow_gauge"),
    timeout_s: int = 120,
    main_only: bool = True,
) -> str:
    """Overpass QL: union of ``around`` filters at many points (for crawl/tiled seed)."""
    flt = _way_filter(main_only)
    clauses = []
    for lat, lon in points_latlon:
        for v in railway_values:
            clauses.append(f'way["railway"="{v}"]{flt}(around:{radius_m},{lat},{lon});')
    body = "\n  ".join(clauses)
    return f"[out:json][timeout:{timeout_s}];\n(\n  {body}\n);\nout geom;\n"


def parse_maxspeed_kph(value: Optional[str]) -> float:
    """Parse an OSM maxspeed string to km/h. Returns NaN if unknown/unparseable.

    Handles ``"60"`` (km/h by OSM convention), ``"60 mph"``, ``"100 km/h"``,
    and conservative numeric extraction. ``"none"`` -> NaN.
    """
    if value is None:
        return math.nan
    v = value.strip().lower()
    if v in _FALSE_TAG_VALUES:
        return math.nan
    is_mph = "mph" in v
    num = ""
    for ch in v:
        if ch.isdigit() or ch == ".":
            num += ch
        elif num:
            break
    if not num:
        return math.nan
    try:
        n = float(num)
    except ValueError:
        return math.nan
    return n * 1.609344 if is_mph else n


def tag_is_truthy(tags: Dict[str, str], key: str) -> bool:
    """True if ``key`` is present with a non-false value (bridge/tunnel/etc.)."""
    return str(tags.get(key, "")).strip().lower() not in _FALSE_TAG_VALUES


def parse_overpass_json(data: dict) -> RelationData:
    """Convert an Overpass JSON response into a :class:`RelationData` (no network)."""
    elements = data.get("elements", [])
    rel_id: Optional[int] = None
    rel_tags: Dict[str, str] = {}
    ways: List[Way] = []

    for el in elements:
        etype = el.get("type")
        if etype == "relation":
            rel_id = el.get("id")
            rel_tags = el.get("tags", {}) or {}
        elif etype == "way":
            geom = el.get("geometry") or []
            coords = [(g["lon"], g["lat"]) for g in geom]
            node_ids = el.get("nodes") or list(range(len(coords)))
            if len(node_ids) != len(coords):
                # Geometry/nodes mismatch (rare); fall back to positional ids.
                node_ids = list(range(len(coords)))
            ways.append(
                Way(
                    id=el.get("id"),
                    tags=el.get("tags", {}) or {},
                    node_ids=list(node_ids),
                    coords=coords,
                )
            )

    if rel_id is None:
        # Some responses omit the relation element if queried oddly; tolerate it.
        rel_id = -1
    return RelationData(id=rel_id, tags=rel_tags, ways=ways)


def filter_track_ways(rel: RelationData, allow_all: bool = False) -> RelationData:
    """Keep only ways whose ``railway`` tag is a track value.

    Route relations often include platform/station ways; those are not part of
    the running centerline. If filtering removes everything (and ``allow_all`` is
    False), the original ways are kept and the caller should warn.
    """
    if allow_all:
        return rel
    kept = [w for w in rel.ways if w.tags.get("railway") in TRACK_RAILWAY_VALUES]
    if not kept:
        return rel  # nothing matched; return as-is so caller can warn + fall back
    return RelationData(id=rel.id, tags=rel.tags, ways=kept)


def fetch_relation(
    relation_id: int,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout_s: int = 90,
    session=None,
) -> RelationData:
    """Fetch a route relation from Overpass. Requires network (runs on your machine).

    Raises a RuntimeError with a readable message on HTTP/network/parse failure.
    """
    query = build_query(relation_id, timeout_s=timeout_s)
    data = _post_overpass(endpoint, query, timeout_s, session=session)
    rel = parse_overpass_json(data)
    if not rel.ways:
        raise RuntimeError(
            f"Relation {relation_id} returned no member ways. "
            f"Check the relation ID and that it is a railway route."
        )
    return rel


def fetch_area(
    south: float, west: float, north: float, east: float,
    endpoint: str = DEFAULT_ENDPOINT,
    railway_values=("rail", "narrow_gauge"),
    timeout_s: int = 90,
    main_only: bool = True,
    session=None,
) -> RelationData:
    """Fetch track ways within a bbox. Returns RelationData with id=-1.

    Requires network (runs on your machine). Retries + endpoint failover built in.
    """
    query = build_area_query(south, west, north, east, railway_values=railway_values,
                             timeout_s=timeout_s, main_only=main_only)
    data = _post_overpass(endpoint, query, timeout_s, session=session,
                          endpoints=endpoint_rotation(endpoint))
    rel = parse_overpass_json(data)
    rel.tags = {"source": "area_query",
                "bbox": f"{south},{west},{north},{east}"}
    if not rel.ways:
        raise RuntimeError("No railway track ways found in that area.")
    return rel


def fetch_radius(
    lat: float, lon: float, radius_m: float,
    endpoint: str = DEFAULT_ENDPOINT,
    railway_values=("rail", "narrow_gauge"),
    timeout_s: int = 90,
    main_only: bool = True,
    session=None,
) -> RelationData:
    """Fetch all track ways within ``radius_m`` of a point. Returns RelationData id=-1.

    Ways are returned un-clipped, so any line intersecting the ball is captured in
    full at the way level. Requires network (runs on your machine).
    """
    query = build_radius_query(lat, lon, radius_m, railway_values=railway_values,
                               timeout_s=timeout_s, main_only=main_only)
    data = _post_overpass(endpoint, query, timeout_s, session=session,
                          endpoints=endpoint_rotation(endpoint))
    rel = parse_overpass_json(data)
    rel.tags = {"source": "radius_query",
                "center": f"{lat},{lon}", "radius_m": str(radius_m)}
    if not rel.ways:
        raise RuntimeError("No railway track ways found within that radius.")
    return rel
