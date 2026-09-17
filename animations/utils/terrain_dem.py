"""Real terrain for the map panel: a DEM patch around the animated segment.

The route tensors carry ground elevation only *along* the alignment, which is a
1-D curve -- interpolating a surface from it produces contours that hug the
track like contour-feathers and describe no real landform. This module instead
samples USGS 3DEP on a regular grid over the segment's bounding box, using the
same ``routegen.elevation`` client the route pipeline uses, so the contours in
the animation are surveyed ground.

The result is cached in ``animations/data/`` and committed, so rendering the
scene never needs the network. Delete the cache to re-fetch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "route_generator"))


def lonlat_from_enu(
    east: np.ndarray, north: np.ndarray, frame: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the local tangent-plane projection used by ``segment_run``."""
    lon_mean, lat_mean, lat0 = frame
    lon = east / (111_320.0 * np.cos(np.radians(lat0))) + lon_mean
    lat = north / 110_540.0 + lat_mean
    return lon, lat


def fetch_dem_grid(
    cache_path: Path,
    frame: tuple[float, float, float],
    origin: tuple[float, float],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    nx: int = 28,
    ny: int = 64,
    allow_network: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Grid of real ground elevation over the segment, cached on disk.

    ``x_bounds``/``y_bounds`` are in the segment's local ENU metres, whose origin
    sits at ``origin`` in the route-wide frame. The grid is deliberately not
    square: the segment box is roughly 1:5, so ``nx`` x ``ny`` keeps the sample
    spacing comparable on both axes for a given request budget (100 points per
    request). Returns ``(gx, gy, gz)``, or ``None`` if the DEM is unavailable
    and nothing is cached, so the caller can fall back.
    """
    want = np.array([*x_bounds, *y_bounds])
    if cache_path.exists():
        with np.load(cache_path) as c:
            if (int(c["nx"]), int(c["ny"])) == (nx, ny) and np.allclose(c["bounds"], want):
                return c["gx"], c["gy"], c["gz"]

    if not allow_network:
        return None

    gx = np.linspace(x_bounds[0], x_bounds[1], nx)
    gy = np.linspace(y_bounds[0], y_bounds[1], ny)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    lon, lat = lonlat_from_enu(mesh_x.ravel() + origin[0],
                               mesh_y.ravel() + origin[1], frame)

    from routegen.elevation import sample_elevations

    elev, stats = sample_elevations(list(zip(lon, lat)), method="batch")
    if not np.isfinite(elev).any():
        return None
    gz = elev.reshape(mesh_x.shape)

    # A few NoData cells are normal at tile edges; fill them from their finite
    # neighbours so the contour generator has a complete grid.
    if not np.isfinite(gz).all():
        gz = _fill_holes(gz)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, gx=gx, gy=gy, gz=gz, nx=nx, ny=ny, bounds=want,
        n_missing=stats.get("n_missing", 0),
    )
    return gx, gy, gz


def _fill_holes(gz: np.ndarray, passes: int = 12) -> np.ndarray:
    """Replace NaNs by the mean of their finite 4-neighbours, iterated."""
    out = gz.copy()
    for _ in range(passes):
        bad = ~np.isfinite(out)
        if not bad.any():
            break
        padded = np.pad(out, 1, mode="edge")
        stack = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1],
                          padded[1:-1, :-2], padded[1:-1, 2:]])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=0)
        out[bad] = mean[bad]
    return np.nan_to_num(out, nan=float(np.nanmean(gz)))
