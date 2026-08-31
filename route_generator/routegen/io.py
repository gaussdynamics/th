"""Serialization helpers for assembled centerlines (GeoJSON / per-vertex records)."""

from __future__ import annotations

import math
from typing import Dict, List

from .models import Centerline


def _clean(x: float):
    """JSON-safe float (NaN -> None)."""
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)


def centerline_records(cl: Centerline) -> List[Dict]:
    """One row per vertex; convenient for CSV / DataFrame export."""
    rows = []
    for i in range(cl.num_vertices):
        rows.append(
            {
                "vertex": i,
                "s_m": float(cl.s_m[i]),
                "lon": float(cl.coords[i, 0]),
                "lat": float(cl.coords[i, 1]),
                "way_id": int(cl.way_id[i]),
                "maxspeed_kph": _clean(cl.maxspeed_kph[i]),
                "is_bridge": bool(cl.is_bridge[i]),
                "is_tunnel": bool(cl.is_tunnel[i]),
                "is_cutting": bool(cl.is_cutting[i]),
                "is_embankment": bool(cl.is_embankment[i]),
            }
        )
    return rows


def centerline_to_geojson(cl: Centerline) -> Dict:
    """A GeoJSON LineString Feature with relation tags + QA summary in properties.

    Per-vertex arrays are included in properties so the raw acquisition is
    self-contained for the downstream DEM/profile step (Part 2).
    """
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(lon), float(lat)] for lon, lat in cl.coords],
        },
        "properties": {
            "relation_id": cl.relation_id,
            "relation_tags": cl.relation_tags,
            "length_m": cl.length_m,
            "num_vertices": cl.num_vertices,
            "maxspeed_coverage": cl.maxspeed_coverage_fraction(),
            "is_service": cl.is_service,
            "service_kind": cl.service_kind,
            "line_group": cl.line_group,
            "group_order": cl.group_order,
            "used_way_ids": [int(w) for w in cl.used_way_ids],
            "n_gaps": len(cl.gaps),
            "n_branches": len(cl.branches),
            "dropped_way_ids": cl.dropped_way_ids,
            "order": cl.order,
            "s_m": [float(s) for s in cl.s_m],
            "way_id": [int(w) for w in cl.way_id],
            "maxspeed_kph": [_clean(v) for v in cl.maxspeed_kph],
            "is_bridge": [bool(b) for b in cl.is_bridge],
            "is_tunnel": [bool(b) for b in cl.is_tunnel],
            "is_cutting": [bool(b) for b in cl.is_cutting],
            "is_embankment": [bool(b) for b in cl.is_embankment],
        },
    }
