"""Part 1.5 — resample an assembled centerline onto a uniform arc-length grid.

OSM digitizes geometry *adaptively for horizontal curvature*: dense on curves,
sparse on long tangents. That under-samples the **vertical** profile exactly
where grade lives — a 2 km straight gets one segment, so any hill inside it is
invisible to a DEM query placed only at the OSM nodes. This stage rewrites a
route onto a fixed-Δs chainage grid so the later USGS step (and curve fitting)
see uniform coverage.

Strictly geometry + bookkeeping — **no smoothing, no grade**. That is phase 3.

What it does:

* **Uniform resample.** Interpolate ``(lon, lat)`` linearly along the polyline at
  every ``ds_m`` of arc length (linear is exact at 10 m scale).
* **Step features held, not ramped.** ``maxspeed_kph``, ``way_id`` and the
  ``is_bridge/is_tunnel/is_cutting/is_embankment`` flags are piecewise-constant:
  a resampled point inherits the value of the source segment it falls in
  (left/hold-last), never an interpolated value.
* **Transition snapping.** Inserts a sample exactly where any step feature
  changes, so a bridge entrance or speed-limit step is not smeared by up to a
  cell width.
* **Validation.** Drops coincident vertices, enforces strictly increasing ``s``.
* **Provenance.** Tags each point with its source-segment length and an
  ``is_long_segment`` flag (interpolated across a sparse OSM span / bridged gap),
  so synthetic points on big jumps stay distinguishable from surveyed track.

``resample_feature`` is pure (no network) and unit-testable.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List

import numpy as np

# Per-vertex step/categorical arrays carried by hold-last (segment membership).
STEP_KEYS = ("way_id", "maxspeed_kph", "is_bridge", "is_tunnel",
             "is_cutting", "is_embankment")


def _coverage(maxspeed: List) -> float:
    vals = [v for v in maxspeed if v is not None]
    return (len(vals) / len(maxspeed)) if maxspeed else 0.0


def resample_feature(
    feature: Dict,
    ds_m: float = 10.0,
    *,
    snap_transitions: bool = True,
    long_segment_m: float = 300.0,
) -> Dict:
    """Resample a Part-1 GeoJSON LineString Feature onto a uniform Δs grid.

    Returns a new Feature (2-D ``[lon, lat]`` geometry) whose per-vertex arrays
    (``s_m``, ``way_id``, ``maxspeed_kph``, structure flags) are aligned to the
    new grid, plus ``seg_len_m`` / ``is_long_segment`` provenance and a
    ``resample_meta`` block. Route-level scalar properties are preserved.
    """
    coords = feature["geometry"]["coordinates"]
    lon = np.array([c[0] for c in coords], dtype=float)
    lat = np.array([c[1] for c in coords], dtype=float)
    props = feature.get("properties", {})

    n_in = len(coords)
    if n_in < 2:
        raise ValueError("Route needs at least 2 vertices to resample.")

    s = np.asarray(props.get("s_m", []), dtype=float)
    if s.shape[0] != n_in:
        # Fall back to a geodesic chainage if s_m is absent/mismatched.
        s = _geodesic_chainage(lon, lat)

    # --- validation: drop coincident vertices, enforce strictly increasing s --
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    lon, lat, s = lon[keep], lat[keep], s[keep]
    step_in = {
        k: [props[k][i] for i in range(n_in) if keep[i]]
        for k in STEP_KEYS
        if isinstance(props.get(k), list) and len(props[k]) == n_in
    }
    nk = len(s)
    if nk < 2:
        raise ValueError("Route collapses to <2 unique vertices after dedup.")
    s_max = float(s[-1])

    # --- build the sample grid ----------------------------------------------
    grid = list(np.arange(0.0, s_max, float(ds_m)))
    if not grid or grid[-1] < s_max:
        grid.append(s_max)
    if snap_transitions:
        for vals in step_in.values():
            for i in range(1, nk):
                if vals[i] != vals[i - 1]:
                    grid.append(float(s[i]))
    grid = np.array(sorted({round(g, 6) for g in grid}), dtype=float)
    grid = grid[(grid >= 0.0) & (grid <= s_max)]

    # --- locate each grid point in a source segment [seg, seg+1] ------------
    # ``seg_feat`` is the hold-last (left/own-node) index for step features —
    # for a grid point on a node (incl. the endpoint) it is that node's own
    # index; for an interior point it is the left node. ``seg_i`` is clamped one
    # short so ``seg_i + 1`` stays in range for position interpolation.
    seg_feat = np.clip(np.searchsorted(s, grid, side="right") - 1, 0, nk - 1)
    seg_i = np.clip(seg_feat, 0, nk - 2)
    s0, s1 = s[seg_i], s[seg_i + 1]
    span = np.where(s1 > s0, s1 - s0, 1.0)
    t = (grid - s0) / span

    out_lon = lon[seg_i] + t * (lon[seg_i + 1] - lon[seg_i])
    out_lat = lat[seg_i] + t * (lat[seg_i + 1] - lat[seg_i])
    seg_len = (s1 - s0)

    # --- hold-last propagation of step features -----------------------------
    out_step = {k: [step_in[k][int(i)] for i in seg_feat] for k in step_in}

    # --- assemble output feature (preserve route-level scalars) -------------
    out = json.loads(json.dumps(feature))
    out["geometry"] = {
        "type": "LineString",
        "coordinates": [[float(x), float(y)] for x, y in zip(out_lon, out_lat)],
    }
    p = out.setdefault("properties", {})
    p["num_vertices"] = int(len(grid))
    p["length_m"] = s_max
    p["s_m"] = [float(g) for g in grid]
    for k in STEP_KEYS:
        if k in out_step:
            p[k] = out_step[k]
    if "maxspeed_kph" in out_step:
        p["maxspeed_coverage"] = _coverage(out_step["maxspeed_kph"])
    p["seg_len_m"] = [float(v) for v in seg_len]
    p["is_long_segment"] = [bool(v > long_segment_m) for v in seg_len]
    p["order"] = []  # stale after resample; vertex order is the array order
    p["resample_meta"] = {
        "ds_m": float(ds_m),
        "snap_transitions": bool(snap_transitions),
        "long_segment_m": float(long_segment_m),
        "n_in": int(n_in),
        "n_unique_in": int(nk),
        "n_out": int(len(grid)),
        "n_snapped": int(len(grid) - _n_uniform(s_max, ds_m)),
        "n_long_segment": int(np.sum(seg_len > long_segment_m)),
    }
    return out


def _n_uniform(s_max: float, ds_m: float) -> int:
    """How many points a pure uniform grid (no snapping) would have."""
    n = int(math.floor(s_max / ds_m)) + 1
    if (n - 1) * ds_m < s_max:
        n += 1
    return n


def _geodesic_chainage(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Cumulative great-circle arc length (m) — fallback when s_m is missing."""
    R = 6371008.8
    lo, la = np.radians(lon), np.radians(lat)
    dlo, dla = np.diff(lo), np.diff(la)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(a))
    return np.concatenate([[0.0], np.cumsum(d)])
