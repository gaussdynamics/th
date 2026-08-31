"""Geodesic helpers (WGS84 haversine). Pure NumPy, no binary geo deps.

Accuracy note: haversine is well within tolerance for route-length and
arc-length bookkeeping. Higher-accuracy ellipsoidal distance (pyproj.Geod) can
replace this in Part 2 if needed; the interface returns metres throughout.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius (IUGG)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between two points (scalars or arrays)."""
    lon1, lat1, lon2, lat2 = map(np.asarray, (lon1, lat1, lon2, lat2))
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def segment_lengths_m(coords: np.ndarray) -> np.ndarray:
    """Per-segment lengths (metres) for a polyline ``coords`` of shape [P, 2] (lon, lat).

    Returns an array of length ``P-1``.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2:
        return np.zeros(0, dtype=float)
    lon, lat = coords[:, 0], coords[:, 1]
    return haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])


def cumulative_arclength_m(coords: np.ndarray) -> np.ndarray:
    """Cumulative arc length (metres) at each vertex; shape [P], starting at 0."""
    seg = segment_lengths_m(coords)
    s = np.zeros(coords.shape[0] if hasattr(coords, "shape") else len(coords), dtype=float)
    if seg.size:
        s[1:] = np.cumsum(seg)
    return s


def polyline_length_m(coords: np.ndarray) -> float:
    """Total polyline length in metres."""
    return float(segment_lengths_m(coords).sum())


def initial_bearing_deg(lon1, lat1, lon2, lat2) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees in [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlam)
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d
