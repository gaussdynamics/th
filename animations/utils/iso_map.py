"""Axonometric projection and terrain contours for the route map panel.

The map panel draws a real piece of corridor -- OSM alignment with USGS 3DEP
elevation on every vertex -- as a small isometric block, the way a site plan
draws a cutting. Plan geometry is kept true to scale; only elevation is
exaggerated, because 13 m of climb across 1.6 km is invisible otherwise.
"""

from __future__ import annotations

import numpy as np

#: Rotation of the plan about the vertical, degrees. Swings the corridor's long
#: axis onto the screen diagonal so it reads as running away from the viewer.
ISO_AZIMUTH_DEG = -52.0

#: Foreshortening of the depth axis. 0.5 is the familiar 2:1 "isometric" look.
ISO_TILT = 0.52

#: Elevation multiplier, applied before projection. Announced on screen -- the
#: relief is real, the vertical scale is not.
Z_EXAGGERATION = 34.0


def project(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, z_ref: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project local ENU metres onto the panel's 2-D axonometric frame.

    Returns coordinates still in metres-ish units; :func:`fit` turns them into
    scene units. ``z_ref`` is the elevation datum, so a point on the datum lands
    exactly on the ground plane.
    """
    a = np.radians(ISO_AZIMUTH_DEG)
    ca, sa = np.cos(a), np.sin(a)
    plan_u = x * ca - y * sa
    plan_v = x * sa + y * ca
    sx = plan_u
    sy = ISO_TILT * plan_v + Z_EXAGGERATION * (np.asarray(z, dtype=float) - z_ref)
    return sx, sy


def fit(
    sx: np.ndarray, sy: np.ndarray, width: float, height: float
) -> tuple[float, float, float]:
    """Uniform scale and centring that fit projected points into a box.

    Returns ``(scale, cx, cy)`` such that ``scale * (s - c)`` lands inside a box
    of ``width`` x ``height`` centred on the origin. The scale is shared by both
    axes so the plan is not stretched.
    """
    x0, x1 = float(np.min(sx)), float(np.max(sx))
    y0, y1 = float(np.min(sy)), float(np.max(sy))
    span_x = max(x1 - x0, 1e-9)
    span_y = max(y1 - y0, 1e-9)
    scale = min(width / span_x, height / span_y)
    return scale, 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def contour_levels(grid_z: np.ndarray, step_m: float = 2.0) -> np.ndarray:
    """Iso-elevation levels on a round ``step_m`` covering the terrain."""
    lo = np.floor(float(np.min(grid_z)) / step_m) * step_m
    hi = np.ceil(float(np.max(grid_z)) / step_m) * step_m
    return np.arange(lo + step_m, hi, step_m)


def contour_paths(
    grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray, levels: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    """Extract contour polylines as ``(level, [[x, y], ...])`` pairs.

    Uses contourpy directly -- it ships with matplotlib and needs no figure, so
    nothing here opens a plotting backend inside a Manim render.
    """
    from contourpy import contour_generator

    gen = contour_generator(x=grid_x, y=grid_y, z=grid_z)
    out: list[tuple[float, np.ndarray]] = []
    for level in levels:
        for seg in gen.lines(float(level)):
            seg = np.asarray(seg, dtype=float)
            if seg.shape[0] >= 2:
                out.append((float(level), seg))
    return out


def resample_polyline(pts: np.ndarray, max_points: int) -> np.ndarray:
    """Thin a polyline to at most ``max_points`` vertices, keeping both ends."""
    if pts.shape[0] <= max_points:
        return pts
    idx = np.unique(np.linspace(0, pts.shape[0] - 1, max_points).astype(int))
    return pts[idx]


def position_on_route(
    s_query: float,
    seg_s: np.ndarray,
    seg_x: np.ndarray,
    seg_y: np.ndarray,
    seg_z: np.ndarray,
) -> tuple[float, float, float]:
    """Interpolate the alignment at a chainage, in local ENU metres."""
    return (
        float(np.interp(s_query, seg_s, seg_x)),
        float(np.interp(s_query, seg_s, seg_y)),
        float(np.interp(s_query, seg_s, seg_z)),
    )
