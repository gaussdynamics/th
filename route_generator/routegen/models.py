"""Data models for OSM acquisition and assembled centerlines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# OSM railway track values we treat as routable centerline (excludes platforms,
# stations, abandoned/razed unless explicitly allowed by the caller).
TRACK_RAILWAY_VALUES = frozenset(
    {"rail", "light_rail", "subway", "tram", "narrow_gauge", "monorail", "funicular"}
)

# OSM service=* values that mark non-mainline track (kept, but flagged so they
# don't sever a running line). https://wiki.openstreetmap.org/wiki/Key:service
SERVICE_VALUES = frozenset({"siding", "spur", "yard", "crossover"})


@dataclass
class Way:
    """An OSM way member of a route relation, with inline geometry."""

    id: int
    tags: Dict[str, str]
    node_ids: List[int]
    coords: List[Tuple[float, float]]  # (lon, lat) per node, same order as node_ids
    role: str = ""

    @property
    def start_node(self) -> Optional[int]:
        return self.node_ids[0] if self.node_ids else None

    @property
    def end_node(self) -> Optional[int]:
        return self.node_ids[-1] if self.node_ids else None

    def reversed(self) -> "Way":
        return Way(
            id=self.id,
            tags=self.tags,
            node_ids=list(reversed(self.node_ids)),
            coords=list(reversed(self.coords)),
            role=self.role,
        )


@dataclass
class RelationData:
    """A fetched OSM route relation: its tags and ordered member ways."""

    id: int
    tags: Dict[str, str]
    ways: List[Way] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.tags.get("name", f"relation/{self.id}")


@dataclass
class Gap:
    """A discontinuity between two consecutive ways in the assembled chain."""

    after_way_id: int
    before_way_id: int
    distance_m: float
    at_lonlat: Tuple[float, float]


@dataclass
class Branch:
    """A node where more than two way-endpoints meet (route ambiguity)."""

    node_id: int
    degree: int
    way_ids: List[int]
    at_lonlat: Optional[Tuple[float, float]] = None


@dataclass
class Centerline:
    """A continuous, ordered centerline assembled from a relation's ways.

    Arrays are vertex-aligned (length P). Per-vertex tag flags are propagated
    from the source way of each vertex.
    """

    relation_id: int
    relation_tags: Dict[str, str]
    coords: np.ndarray  # [P, 2] (lon, lat)
    s_m: np.ndarray  # [P] cumulative arc length (metres)
    way_id: np.ndarray  # [P] source way id per vertex
    maxspeed_kph: np.ndarray  # [P] float, NaN where unknown
    is_bridge: np.ndarray  # [P] bool
    is_tunnel: np.ndarray  # [P] bool
    is_cutting: np.ndarray  # [P] bool
    is_embankment: np.ndarray  # [P] bool
    order: List[Tuple[int, bool]] = field(default_factory=list)  # (way_id, was_reversed)
    gaps: List[Gap] = field(default_factory=list)
    branches: List[Branch] = field(default_factory=list)
    used_way_ids: List[int] = field(default_factory=list)
    dropped_way_ids: List[int] = field(default_factory=list)
    # Service-track classification (mainline if is_service is False).
    is_service: bool = False
    service_kind: Optional[str] = None  # siding / spur / yard / crossover / mixed
    # Parent-line grouping (set by network.group_corridors). Fragments of one
    # physical line share a line_group; group_order is their order along it.
    line_group: Optional[int] = None
    group_order: Optional[int] = None

    @property
    def length_m(self) -> float:
        return float(self.s_m[-1]) if self.s_m.size else 0.0

    @property
    def num_vertices(self) -> int:
        return int(self.coords.shape[0])

    def maxspeed_coverage_fraction(self) -> float:
        """Fraction of total length whose source way has a known maxspeed."""
        if self.coords.shape[0] < 2:
            return 0.0
        seg_len = np.diff(self.s_m)
        known = ~np.isnan(self.maxspeed_kph[:-1])
        total = seg_len.sum()
        return float(seg_len[known].sum() / total) if total > 0 else 0.0
