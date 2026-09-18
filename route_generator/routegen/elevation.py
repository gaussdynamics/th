"""Part 2 — USGS 3DEP elevation sampling for assembled centerlines.

Takes the (lon, lat) vertices of a Part-1 route and attaches a ground elevation
to each one, producing the set of 3-D points ``(lon, lat, z)`` that downstream
curve-fitting (grade ``sinθ(s)``, curvature ``κ(s)``) will consume. The higher-
dimensional per-vertex feature vector (arc length, maxspeed, bridge/tunnel/…)
from Part 1 rides along untouched — this step only adds the ``z`` channel.

Two USGS sources, primary + fallback (both speak EPSG:4326 lon/lat):

* **3DEP ImageServer ``getSamples``** (primary) — one request returns elevations
  for *many* points, so a 500-vertex route costs a handful of requests, not 500.
  Points are sent in chunks (``BATCH_CHUNK``) to stay under the service's
  per-request sample cap.
* **EPQS point service** (fallback) — one request per point, queried with a small
  thread pool. Used for any vertex the batch path could not resolve (service
  hiccup, NoData pixel, etc.).

Pure-parse helpers (``parse_getsamples_json`` / ``parse_epqs_json``) take no
network so the parsing path is unit-testable offline.
"""

from __future__ import annotations

import json
import math
import os
import random
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Endpoints / constants
# --------------------------------------------------------------------------- #
IMAGESERVER_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/getSamples"
)
EPQS_URL = "https://epqs.nationalmap.gov/v1/json"

USER_AGENT = "route_generator/0.1 (railway LTD dataset; USGS 3DEP client)"

# Max points per ImageServer request. Measured 2026-09-17: the service returns
# a full result up to 1000 points and silently truncates to 1000 at 2000, so
# 1000 is the ceiling rather than a guess. Throughput against chunk size, on
# real alignment vertices:
#
#     100 pts/req ->   435 pts/s      500 pts/req -> 1,385 pts/s
#     250 pts/req ->   929 pts/s     1000 pts/req -> 2,729 pts/s
#
# Latency is nearly flat in chunk size (0.23 s at 100, 0.37 s at 1000), so the
# old value spent almost all of its time on per-request overhead.
BATCH_CHUNK = 1000

# Batch requests are issued concurrently. Measured at 1000 points/request:
# 1 worker 2,196 pts/s, 4 -> 5,785, 8 -> 11,829, 16 -> 16,427. Sequential
# issue was the single largest cost in mapping the network: it put the
# continental main line at 8.7 hours against 0.55 at this setting.
BATCH_WORKERS = 16

# EPQS sentinel for "no data at this location".
_EPQS_NODATA = -1000000.0

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


# --------------------------------------------------------------------------- #
# Pure parsers (no network — unit-testable)
# --------------------------------------------------------------------------- #
def _to_float(v) -> Optional[float]:
    """Parse a possibly-string elevation; NaN/None/sentinel/NoData -> None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or f <= _EPQS_NODATA:
        return None
    return f


def parse_getsamples_json(payload: Dict, n_points: int) -> List[Optional[float]]:
    """ImageServer ``getSamples`` JSON -> elevations aligned to input order.

    Each sample carries a ``locationId`` equal to the index of the input point;
    we scatter values back into an ``n_points``-long list (missing -> None).
    """
    out: List[Optional[float]] = [None] * n_points
    for s in (payload or {}).get("samples", []) or []:
        try:
            loc = int(s.get("locationId"))
        except (TypeError, ValueError):
            continue
        if 0 <= loc < n_points:
            out[loc] = _to_float(s.get("value"))
    return out


def parse_epqs_json(payload: Dict) -> Optional[float]:
    """EPQS JSON -> elevation (meters). Handles v1 and legacy nested shapes."""
    if not payload:
        return None
    if "value" in payload:
        return _to_float(payload["value"])
    # Legacy: {"USGS_Elevation_Point_Query_Service": {"Elevation_Query": {...}}}
    try:
        q = payload["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]
        return _to_float(q.get("Elevation"))
    except (KeyError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Network samplers
# --------------------------------------------------------------------------- #
def _chunks(seq: Sequence, n: int):
    for i in range(0, len(seq), n):
        yield i, seq[i : i + n]


def _sample_chunk_imageserver(
    lonlats: Sequence[Tuple[float, float]],
    session,
    timeout_s: int,
    retries: int,
    backoff_s: float,
) -> List[Optional[float]]:
    """One ImageServer request for a chunk of points -> elevations in order."""
    import requests  # local import: package imports without network deps

    http = session or requests
    geometry = {
        "points": [[float(lon), float(lat)] for lon, lat in lonlats],
        "spatialReference": {"wkid": 4326},
    }
    params = {
        "geometry": json.dumps(geometry),
        "geometryType": "esriGeometryMultipoint",
        "returnFirstValueOnly": "true",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "json",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    last = "unknown error"
    for attempt in range(retries + 1):
        try:
            # POST, not GET: a 1000-point geometry is ~30 KB of JSON, far past
            # what a URL can carry, and the service rejects it rather than
            # truncating. This was a silent 600x slowdown -- every chunk failed
            # and every point fell through to the per-point EPQS path.
            resp = http.post(IMAGESERVER_URL, data=params, headers=headers,
                             timeout=(15, timeout_s))
        except Exception as exc:  # noqa: BLE001 - Timeout/ConnectionError/…
            last = f"network error: {exc}"
        else:
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError as exc:
                    raise RuntimeError("getSamples response was not valid JSON.") from exc
                if isinstance(payload, dict) and payload.get("error"):
                    raise RuntimeError(f"getSamples error: {payload['error']}")
                return parse_getsamples_json(payload, len(lonlats))
            last = f"HTTP {resp.status_code}"
            if resp.status_code not in _RETRY_STATUS:
                body = (getattr(resp, "text", "") or "").strip().replace("\n", " ")
                raise RuntimeError(f"getSamples HTTP {resp.status_code}. {body[:200]}")
        if attempt < retries:
            time.sleep(backoff_s * (2 ** attempt) + random.uniform(0.0, 0.5))
    raise RuntimeError(f"getSamples failed after {retries + 1} attempt(s): {last}.")


def _sample_point_epqs(
    lon: float, lat: float, session, timeout_s: int, retries: int, backoff_s: float
) -> Optional[float]:
    """One EPQS request for a single point. Returns None on persistent failure."""
    import requests

    http = session or requests
    params = {"x": float(lon), "y": float(lat), "units": "Meters", "wkid": 4326}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(retries + 1):
        try:
            resp = http.get(EPQS_URL, params=params, headers=headers,
                            timeout=(15, timeout_s))
            if resp.status_code == 200:
                return parse_epqs_json(resp.json())
        except Exception:  # noqa: BLE001
            pass
        if attempt < retries:
            time.sleep(backoff_s * (2 ** attempt) + random.uniform(0.0, 0.5))
    return None


def sample_elevations(
    lonlats: Sequence[Tuple[float, float]],
    *,
    method: str = "batch",
    session=None,
    timeout_s: int = 60,
    retries: int = 3,
    backoff_s: float = 1.5,
    epqs_workers: int = 8,
    batch_workers: int = BATCH_WORKERS,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, Dict]:
    """Ground elevation (m) for each ``(lon, lat)``; gaps filled by EPQS.

    Parameters
    ----------
    method : {"batch", "epqs"}
        ``"batch"`` uses the ImageServer for the bulk pass then EPQS for any
        points it could not resolve. ``"epqs"`` uses EPQS for every point.
    progress : callable(done, total), optional
        Invoked as points are resolved (chunk- or point-granular).

    Returns
    -------
    (elev, stats)
        ``elev`` is ``float64[N]`` with ``np.nan`` where elevation is unknown.
        ``stats`` reports counts: ``n``, ``n_filled``, ``n_missing``,
        ``n_batch``, ``n_epqs_fallback``, ``method``.
    """
    pts = [(float(lon), float(lat)) for lon, lat in lonlats]
    n = len(pts)
    elev: List[Optional[float]] = [None] * n
    n_batch = 0

    if n == 0:
        return np.zeros(0, dtype=float), {
            "n": 0, "n_filled": 0, "n_missing": 0,
            "n_batch": 0, "n_epqs_fallback": 0, "method": method,
        }

    done = 0
    if method == "batch":
        # Issued concurrently: the request is latency-bound, not bandwidth- or
        # server-CPU-bound, so workers scale nearly linearly to 16. A failed
        # chunk leaves its points unresolved and they fall through to the EPQS
        # path below, which is what that path is for.
        jobs = list(_chunks(pts, BATCH_CHUNK))

        failures: List[str] = []

        def _one(job):
            base, chunk = job
            try:
                return base, _sample_chunk_imageserver(
                    chunk, session, timeout_s, retries, backoff_s)
            except Exception as exc:  # noqa: BLE001
                # Recorded, not swallowed. These points still fall through to
                # EPQS, but a batch path that fails for every chunk looks
                # identical to one that works except for being 600x slower,
                # and that is exactly how it was missed once already.
                failures.append(f"{type(exc).__name__}: {exc}")
                return base, [None] * len(chunk)

        with ThreadPoolExecutor(max_workers=max(1, batch_workers)) as ex:
            for base, vals in ex.map(_one, jobs):
                for j, v in enumerate(vals):
                    if v is not None:
                        elev[base + j] = v
                        n_batch += 1
                done += len(vals)
                if progress:
                    progress(done, n)

        if failures:
            warnings.warn(
                f"getSamples failed on {len(failures)}/{len(jobs)} chunk(s); "
                f"those points fall back to the slow per-point path. "
                f"First error: {failures[0][:160]}",
                RuntimeWarning, stacklevel=2,
            )

    # Fallback (or primary, when method == "epqs"): EPQS for unresolved points.
    missing_idx = [i for i in range(n) if elev[i] is None]
    n_epqs = 0
    if missing_idx:
        done_epqs = 0
        total_epqs = len(missing_idx)
        with ThreadPoolExecutor(max_workers=max(1, epqs_workers)) as ex:
            futs = {
                ex.submit(_sample_point_epqs, pts[i][0], pts[i][1],
                          session, timeout_s, retries, backoff_s): i
                for i in missing_idx
            }
            for fut in as_completed(futs):
                i = futs[fut]
                v = fut.result()
                if v is not None:
                    elev[i] = v
                    n_epqs += 1
                done_epqs += 1
                if progress and method == "epqs":
                    progress(done_epqs, total_epqs)

    arr = np.array([np.nan if v is None else float(v) for v in elev], dtype=float)
    n_missing = int(np.isnan(arr).sum())
    stats = {
        "n": n,
        "n_filled": n - n_missing,
        "n_missing": n_missing,
        "n_batch": n_batch,
        "n_epqs_fallback": n_epqs,
        "method": method,
    }
    return arr, stats


# --------------------------------------------------------------------------- #
# GeoJSON helpers (Part-1 saved_routes <-> Part-2 elevated_routes)
# --------------------------------------------------------------------------- #
def feature_lonlats(feature: Dict) -> List[Tuple[float, float]]:
    """(lon, lat) vertices from a GeoJSON LineString Feature."""
    return [(float(c[0]), float(c[1]))
            for c in feature["geometry"]["coordinates"]]


def elevate_feature(feature: Dict, elev: np.ndarray, stats: Optional[Dict] = None) -> Dict:
    """Return a copy of ``feature`` with elevation added.

    Geometry coordinates become 3-D ``[lon, lat, z]`` (z=null where unknown), an
    ``elevation_m`` per-vertex array is added to properties, and a small
    ``elevation_meta`` block records the source + fill stats. All Part-1 per-
    vertex arrays (s_m, maxspeed, structure flags, …) are preserved unchanged.
    """
    out = json.loads(json.dumps(feature))  # deep copy, JSON-clean
    coords = out["geometry"]["coordinates"]
    z_list: List[Optional[float]] = []
    for i, c in enumerate(coords):
        z = float(elev[i]) if (i < len(elev) and not math.isnan(elev[i])) else None
        coords[i] = [float(c[0]), float(c[1])] + ([z] if z is not None else [None])
        z_list.append(z)
    props = out.setdefault("properties", {})
    props["elevation_m"] = z_list
    meta = {
        "source": "USGS 3DEP (getSamples + EPQS)",
        "vertical_units": "m",
        "datum": "NAVD88 (3DEP native)",
    }
    if stats:
        meta.update(stats)
    props["elevation_meta"] = meta
    return out


def has_elevation(feature: Dict) -> bool:
    """True if a feature already carries an ``elevation_m`` array."""
    return bool(feature.get("properties", {}).get("elevation_m"))


def list_saved_routes(directory: str) -> List[str]:
    """Sorted ``*.geojson`` filenames in ``directory`` (Part-1 saved_routes)."""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(".geojson"))


def load_feature(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def save_feature(feature: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(feature, fh, indent=2)
    return path
