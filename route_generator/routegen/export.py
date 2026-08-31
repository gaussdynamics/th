"""Export bin — turn finished route profiles into the route-tensor handoff.

A "route tensor" here is the canonical arc-length route field the surrogate's
dynamics consume, stored per ``DATA_SCHEMA.md §F`` as four named arrays on a
uniform ``s`` grid:

    route_s        [R]  arc length (m)
    route_sin_theta[R]  sinθ(s) grade field
    route_kappa    [R]  curvature proxy κ(s) (1/m)
    route_vmax     [R]  speed limit v_max(s) (m/s)

``route_vmax`` is included per the agreed representation; consuming it in the
simulator still needs a small ``RouteProfile`` extension (a ``v_max_nodes`` field
+ ``v_max_at``) — that change is intentionally deferred to the regroup, so this
module only *produces* the array.

The bundle (``build_export_bundle``) writes a ``routes/`` folder of per-route
``.npz`` plus an ``index.csv`` summary and a ``manifest.json`` (channel legend +
provenance), zipped — mirroring the dataset-build layout so it drops in directly.
Pure stdlib + NumPy; no network.
"""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from typing import Dict, List

import numpy as np

# Canonical route-tensor channels (DATA_SCHEMA §F). Order is the column order
# used if the arrays are ever stacked into a single [R, C] matrix downstream.
TENSOR_CHANNELS = ("route_s", "route_sin_theta", "route_kappa", "route_vmax")
CHANNEL_LEGEND = {
    "route_s": "arc length s (m)",
    "route_sin_theta": "sin(theta(s)) — grade field (dimensionless)",
    "route_kappa": "curvature proxy kappa(s) (1/m)",
    "route_vmax": "speed limit v_max(s) (m/s)",
}


def _stem(name: str) -> str:
    return name[:-8] if name.endswith(".geojson") else name


def tensor_from_profile_geojson(feature: Dict) -> Dict[str, np.ndarray]:
    """Extract the named route-tensor arrays from a Part-3 profile GeoJSON.

    Reads the arrays Part 3 already wrote into ``properties`` (no re-derivation),
    so it works on both freshly-derived and on-disk profiles. Raises if the
    feature is not a derived profile.
    """
    p = feature.get("properties", {})
    required = ("s_m", "sin_theta", "kappa", "v_max_ms")
    missing = [k for k in required if k not in p]
    if missing:
        raise ValueError(f"Not a derived route profile (missing {missing}). "
                         "Run the Route-profile tab first.")
    return {
        "route_s": np.asarray(p["s_m"], dtype=np.float32),
        "route_sin_theta": np.asarray(p["sin_theta"], dtype=np.float32),
        "route_kappa": np.asarray(p["kappa"], dtype=np.float32),
        "route_vmax": np.asarray(p["v_max_ms"], dtype=np.float32),
    }


def route_summary(feature: Dict, name: str = "") -> Dict:
    """One-row summary of a route's headline data for the export index/table."""
    coords = feature["geometry"]["coordinates"]
    p = feature.get("properties", {})
    lon = np.array([c[0] for c in coords], dtype=float)
    lat = np.array([c[1] for c in coords], dtype=float)
    meta = p.get("profile_meta", {})
    s = np.asarray(p.get("s_m", []), dtype=float)
    dist_km = float(meta.get("length_km", (s[-1] / 1000.0) if s.size else 0.0))
    label = (p.get("name") or p.get("relation_tags", {}).get("name") or _stem(name))
    return {
        "route": _stem(name),
        "name": label,
        "n_points": len(coords),
        "distance_km": round(dist_km, 3),
        "start_lat": round(float(lat[0]), 6) if lat.size else None,
        "start_lon": round(float(lon[0]), 6) if lon.size else None,
        "end_lat": round(float(lat[-1]), 6) if lat.size else None,
        "end_lon": round(float(lon[-1]), 6) if lon.size else None,
        "center_lat": round(float(lat.mean()), 6) if lat.size else None,
        "center_lon": round(float(lon.mean()), 6) if lon.size else None,
        "grade_max_pct": round(float(meta.get("grade_max_pct", float("nan"))), 3),
        "grade_rms_pct": round(float(meta.get("grade_rms_pct", float("nan"))), 3),
        "min_radius_m": (None if not math.isfinite(meta.get("min_radius_m", float("inf")))
                         else round(float(meta["min_radius_m"]), 1)),
        "vmax_coverage_frac": round(float(meta.get("vmax_coverage_frac", float("nan"))), 3),
    }


def _npz_bytes(arrays: Dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _index_csv(summaries: List[Dict]) -> str:
    if not summaries:
        return ""
    cols = list(summaries[0].keys())
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=cols)
    w.writeheader()
    w.writerows(summaries)
    return out.getvalue()


def build_export_bundle(features: Dict[str, Dict]) -> bytes:
    """Zip a set of profiled routes into a dataset-ready bundle (bytes).

    Layout:
        routes/<stem>.npz   per-route tensor (TENSOR_CHANNELS, float32)
        index.csv           one summary row per route
        manifest.json       channel legend + provenance + v_max-pending note
    """
    summaries: List[Dict] = []
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, feature in features.items():
            stem = _stem(name)
            tensor = tensor_from_profile_geojson(feature)
            zf.writestr(f"routes/{stem}.npz", _npz_bytes(tensor))
            summaries.append(route_summary(feature, name))
        zf.writestr("index.csv", _index_csv(summaries))
        manifest = {
            "n_routes": len(features),
            "tensor_channels": list(TENSOR_CHANNELS),
            "channel_legend": CHANNEL_LEGEND,
            "dtype": "float32",
            "grid": "uniform arc-length s (per-route Δs in each profile)",
            "schema": "DATA_SCHEMA.md §F route fields; one .npz per route in routes/",
            "notes": ("route_vmax is included; consuming it requires adding "
                      "v_max_nodes to simulator2.RouteProfile (deferred to regroup). "
                      "route_s/route_sin_theta/route_kappa map directly onto the "
                      "existing RouteProfile."),
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    return zbuf.getvalue()
