"""routegen — OSM-based railway route acquisition for the LTD surrogate dataset.

Part 1 scope: acquire a route relation by OSM relation ID and assemble a robust,
ordered centerline with quality-assurance metadata (gaps, branches, tag coverage).
DEM sampling and the curve-fit profile engine arrive in Part 2.
"""

from __future__ import annotations

from .centerline import assemble_centerline, make_centerline_from_chain
from .crawl import crawl_network
from .dataset import (
    Seed,
    dedup_features,
    lines_from_corridors,
    load_seeds,
    run_acquisition,
    stitch_line,
)
from .elevation import (
    elevate_feature,
    feature_lonlats,
    has_elevation,
    list_saved_routes,
    load_feature,
    sample_elevations,
    save_feature,
)
from .export import (
    build_export_bundle,
    route_summary,
    tensor_from_profile_geojson,
)
from .io import centerline_records, centerline_to_geojson
from .models import Branch, Centerline, Gap, RelationData, Way
from .profile import (
    derive_profile,
    profile_to_geojson,
    profile_to_npz_dict,
    save_profile_npz,
    validate_profile,
)
from .resample import resample_feature
from .network import extract_corridors, group_corridors, nearest_corridor
from .osm import (
    DEFAULT_ENDPOINT,
    KNOWN_ENDPOINTS,
    fetch_area,
    fetch_radius,
    fetch_relation,
    filter_track_ways,
    parse_maxspeed_kph,
    parse_overpass_json,
)

__all__ = [
    "assemble_centerline",
    "make_centerline_from_chain",
    "extract_corridors",
    "group_corridors",
    "nearest_corridor",
    "crawl_network",
    "Seed",
    "load_seeds",
    "run_acquisition",
    "stitch_line",
    "lines_from_corridors",
    "dedup_features",
    "centerline_records",
    "centerline_to_geojson",
    "resample_feature",
    "derive_profile",
    "validate_profile",
    "profile_to_geojson",
    "profile_to_npz_dict",
    "save_profile_npz",
    "tensor_from_profile_geojson",
    "route_summary",
    "build_export_bundle",
    "sample_elevations",
    "elevate_feature",
    "feature_lonlats",
    "has_elevation",
    "list_saved_routes",
    "load_feature",
    "save_feature",
    "fetch_relation",
    "fetch_area",
    "fetch_radius",
    "filter_track_ways",
    "parse_overpass_json",
    "parse_maxspeed_kph",
    "DEFAULT_ENDPOINT",
    "KNOWN_ENDPOINTS",
    "RelationData",
    "Way",
    "Centerline",
    "Gap",
    "Branch",
]

__version__ = "0.1.0"
