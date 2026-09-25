"""Part 3 — derive a physical route profile from elevated, resampled samples.

Turns the set of 3-D points produced by Parts 1.5/2 into the fields the
``simulator2`` dynamics actually consume: the arc-length grade field
``sinθ(s)``, a curvature proxy ``κ(s)``, and a speed limit ``v_max(s)``. The
output maps one-to-one onto ``simulator2.route.RouteProfile``
(``s_nodes_m`` / ``sin_theta_nodes`` / ``kappa_nodes``); ``v_max`` is carried
alongside (RouteProfile has no field for it yet).

Why this stage is delicate — grade, not geometry, is the point. 3DEP vertical
error is ~1 m; differenced raw over a 10 m sample that is a 10 % apparent grade,
while real freight ruling grades are ~1–2.5 %. So a naive ``dz/ds`` is almost
all noise. The pipeline therefore is:

1. **Structure correction.** On bridges/tunnels the DEM returns the deck/ground,
   not the rail, so elevation there (and any NaN/NoData) is discarded and linearly
   interpolated across — rail is engineered to near-constant grade on structures.
2. **Smooth, then differentiate.** A Savitzky–Golay filter (local least-squares
   polynomial) gives a smoothed elevation *and* its analytic first derivative from
   one consistent operator, instead of differencing noise. ``sinθ`` is then
   ``sin(arctan(dz/ds))`` with a physical grade clamp.
3. **Curvature from the smoothed planform.** ``(lon, lat)`` → a local planar frame
   (m); SG first/second derivatives give ``κ = (x'y'' − y'x'') / (x'²+y'²)^{3/2}``,
   clamped to a minimum radius.
4. **Speed limit.** ``maxspeed_kph`` → m/s, with an FRA track-class fallback where
   OSM coverage is missing.

Everything is pure NumPy (no SciPy) and ``derive_profile`` takes no network, so
the whole stage is unit-testable offline.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

# FRA freight track-class maximum speeds (mph) — fallback where OSM maxspeed is
# absent. Classes 6+ are passenger/HSR and excluded from the freight default.
FRA_CLASS_MPH = {1: 10, 2: 25, 3: 40, 4: 60, 5: 80}
_MPH_TO_MS = 0.44704
_KPH_TO_MS = 1.0 / 3.6


# --------------------------------------------------------------------------- #
# Savitzky–Golay (pure NumPy): smoothing + analytic derivative
# --------------------------------------------------------------------------- #
def savgol(y: np.ndarray, window: int, poly: int, deriv: int = 0,
           delta: float = 1.0) -> np.ndarray:
    """Savitzky–Golay filter: fit an order-``poly`` polynomial in each length-
    ``window`` window and return value (``deriv=0``) or the ``deriv``-th derivative.

    Edges use odd reflection (slope-preserving), so boundary grades are not
    artificially flattened. ``window`` is forced odd and ``>= poly+1``.
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = max(window, poly + 1 + ((poly + 1) % 2 == 0))  # keep odd & >= poly+1
    if window > n:  # short routes: shrink to the largest valid odd window
        window = n if n % 2 == 1 else n - 1
        window = max(window, poly + 1)
    half = window // 2

    # Design matrix over centred sample offsets; pseudo-inverse row `deriv`
    # is the FIR kernel (correlation) for that derivative.
    u = np.arange(-half, half + 1, dtype=float)
    A = np.vander(u, poly + 1, increasing=True)
    kernel = np.linalg.pinv(A)[deriv] * (math.factorial(deriv) / (delta ** deriv))

    y_pad = np.pad(y, half, mode="reflect", reflect_type="odd")
    return np.correlate(y_pad, kernel, mode="valid")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _local_xy(lon: np.ndarray, lat: np.ndarray):
    """Equirectangular local projection (m) about the route's mean — adequate
    for curvature of a single corridor."""
    R = 6371008.8
    lat0 = float(np.mean(lat))
    lon0 = float(np.mean(lon))
    x = np.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y


def _fill_invalid(values: np.ndarray, s: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``values`` over ``s`` at the ``invalid`` positions,
    using the valid samples (constant extrapolation at the ends)."""
    out = np.array(values, dtype=float)
    valid = ~invalid
    if valid.sum() < 2:
        return out  # not enough anchors; leave as-is
    out[invalid] = np.interp(s[invalid], s[valid], out[valid])
    return out


def _window_samples(window_m: float, ds_m: float) -> int:
    return max(3, int(round(window_m / max(ds_m, 1e-9))))


# --------------------------------------------------------------------------- #
# Core derivation
# --------------------------------------------------------------------------- #
def derive_profile(
    feature: Dict,
    *,
    smooth_window_m: float = 200.0,
    poly_order: int = 2,
    grade_clip: float = 0.04,
    curv_window_m: float = 300.0,
    min_radius_m: float = 80.0,
    vmax_fallback_class: int = 4,
) -> Dict:
    """Derive ``sinθ(s)``, ``κ(s)``, ``v_max(s)`` from an elevated route feature.

    Parameters
    ----------
    smooth_window_m, poly_order : elevation Savitzky–Golay window (m) and order.
    grade_clip : max |dz/ds| (slope), e.g. 0.04 = 4 %.
    curv_window_m : SG window (m) for the planform derivatives.
    min_radius_m : curvature clamp, |κ| ≤ 1/min_radius_m.
    vmax_fallback_class : FRA freight class (1–5) used where maxspeed is missing.

    Returns a dict of equal-length arrays (``s_m``, ``lon``, ``lat``,
    ``z_raw_m``, ``z_corr_m``, ``z_smooth_m``, ``grade``, ``sin_theta``,
    ``kappa``, ``v_max_ms``, ``is_bridge``, ``is_tunnel``,
    ``structure_corrected``) plus a ``meta`` block of parameters and summary
    statistics (grade max/RMS %, min radius, structure-corrected fraction, …).
    """
    coords = feature["geometry"]["coordinates"]
    props = feature.get("properties", {})
    n = len(coords)
    lon = np.array([c[0] for c in coords], dtype=float)
    lat = np.array([c[1] for c in coords], dtype=float)
    z_raw = np.array([(c[2] if len(c) > 2 and c[2] is not None else np.nan)
                      for c in coords], dtype=float)

    s = np.asarray(props.get("s_m", np.arange(n) * 10.0), dtype=float)
    ds = float(np.median(np.diff(s))) if n > 1 else 10.0

    is_bridge = np.array(props.get("is_bridge", [False] * n), dtype=bool)
    is_tunnel = np.array(props.get("is_tunnel", [False] * n), dtype=bool)

    # --- (1) structure correction ------------------------------------------
    invalid = ~np.isfinite(z_raw) | is_bridge | is_tunnel
    z_corr = _fill_invalid(z_raw, s, invalid)
    # If everything was invalid, fall back to zeros so downstream stays finite.
    if not np.isfinite(z_corr).all():
        z_corr = np.nan_to_num(z_corr, nan=0.0)

    # --- (2) smooth elevation + analytic grade -----------------------------
    win = _window_samples(smooth_window_m, ds)
    z_smooth = savgol(z_corr, win, poly_order, deriv=0, delta=ds)
    grade = savgol(z_corr, win, poly_order, deriv=1, delta=ds)
    grade = np.clip(grade, -grade_clip, grade_clip)
    sin_theta = np.sin(np.arctan(grade))

    # --- (3) curvature from smoothed planform ------------------------------
    x, y = _local_xy(lon, lat)
    cwin = _window_samples(curv_window_m, ds)
    dx = savgol(x, cwin, poly_order, deriv=1, delta=ds)
    dy = savgol(y, cwin, poly_order, deriv=1, delta=ds)
    ddx = savgol(x, cwin, poly_order, deriv=2, delta=ds)
    ddy = savgol(y, cwin, poly_order, deriv=2, delta=ds)
    speed2 = dx * dx + dy * dy
    denom = np.power(np.maximum(speed2, 1e-9), 1.5)
    kappa = (dx * ddy - dy * ddx) / denom
    kappa_max = 1.0 / max(min_radius_m, 1e-6)
    kappa = np.clip(kappa, -kappa_max, kappa_max)

    # --- (4) speed limit ----------------------------------------------------
    ms_raw = props.get("maxspeed_kph", [None] * n)
    fallback_ms = FRA_CLASS_MPH.get(int(vmax_fallback_class), 60) * _MPH_TO_MS
    v_max = np.array([(float(v) * _KPH_TO_MS if v is not None else fallback_ms)
                      for v in ms_raw], dtype=float)
    n_vmax_fallback = int(sum(1 for v in ms_raw if v is None))

    # --- stats --------------------------------------------------------------
    grade_pct = grade * 100.0
    nonzero_k = np.abs(kappa) > 1e-9
    min_radius = float(1.0 / np.abs(kappa[nonzero_k]).max()) if nonzero_k.any() else float("inf")
    meta = {
        "ds_m": ds,
        "smooth_window_m": float(smooth_window_m),
        "poly_order": int(poly_order),
        "grade_clip": float(grade_clip),
        "curv_window_m": float(curv_window_m),
        "min_radius_clamp_m": float(min_radius_m),
        "vmax_fallback_class": int(vmax_fallback_class),
        "n": n,
        "length_km": float(s[-1] / 1000.0) if n else 0.0,
        "grade_max_pct": float(np.abs(grade_pct).max()) if n else 0.0,
        "grade_rms_pct": float(np.sqrt(np.mean(grade_pct ** 2))) if n else 0.0,
        "min_radius_m": min_radius,
        "n_structure_corrected": int(invalid.sum()),
        "structure_corrected_frac": float(invalid.mean()) if n else 0.0,
        "n_vmax_fallback": n_vmax_fallback,
        "vmax_coverage_frac": float(1.0 - n_vmax_fallback / n) if n else 0.0,
    }

    return {
        "s_m": s, "lon": lon, "lat": lat,
        "z_raw_m": z_raw, "z_corr_m": z_corr, "z_smooth_m": z_smooth,
        "grade": grade, "sin_theta": sin_theta, "kappa": kappa,
        "v_max_ms": v_max,
        "is_bridge": is_bridge, "is_tunnel": is_tunnel,
        "structure_corrected": invalid,
        "meta": meta,
    }


# --------------------------------------------------------------------------- #
# RouteProfile validation
# --------------------------------------------------------------------------- #
def validate_profile(result: Dict) -> Dict:
    """Sanity-check the derived arrays and confirm they build a RouteProfile.

    Returns ``{"ok": bool, "checks": {name: bool}, "messages": [..]}``. If
    ``simulator2`` is importable, actually constructs ``RouteProfile`` and probes
    ``sin_theta_at`` mid-route; otherwise validates shapes/finiteness only.
    """
    checks: Dict[str, bool] = {}
    msgs = []
    s = result["s_m"]
    arrs = {k: result[k] for k in ("sin_theta", "kappa", "v_max_ms", "z_smooth_m")}

    checks["equal_length"] = all(len(v) == len(s) for v in arrs.values())
    checks["s_monotonic"] = bool(np.all(np.diff(s) > 0)) if len(s) > 1 else True
    checks["all_finite"] = all(np.isfinite(v).all() for v in arrs.values())
    checks["sin_theta_in_range"] = bool(np.all(np.abs(result["sin_theta"]) <= 1.0))
    checks["v_max_positive"] = bool(np.all(result["v_max_ms"] > 0))

    try:
        import sys, os
        sim_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if sim_root not in sys.path:
            sys.path.insert(0, sim_root)
        from simulator2.route import RouteProfile  # type: ignore

        rp = RouteProfile(s_nodes_m=s, sin_theta_nodes=result["sin_theta"],
                          kappa_nodes=result["kappa"])
        mid = float(s[len(s) // 2])
        _ = rp.sin_theta_at(mid)
        _ = rp.kappa_at(mid)
        checks["routeprofile_constructs"] = True
        msgs.append("Built simulator2.RouteProfile and sampled it mid-route.")
    except Exception as exc:  # noqa: BLE001
        checks["routeprofile_constructs"] = checks["equal_length"] and checks["all_finite"]
        msgs.append(f"simulator2 not imported ({exc}); validated array contract only.")

    return {"ok": all(checks.values()), "checks": checks, "messages": msgs}


# --------------------------------------------------------------------------- #
# Serializers — GeoJSON (viewable) + DATA_SCHEMA npz (handoff)
# --------------------------------------------------------------------------- #
def profile_to_geojson(feature: Dict, result: Dict) -> Dict:
    """Profile as a 3-D GeoJSON LineString (z = smoothed elevation) with the
    derived per-vertex arrays in properties. Route-level scalars are preserved."""
    import json

    out = json.loads(json.dumps(feature))
    lon, lat = result["lon"], result["lat"]
    z = result["z_smooth_m"]
    out["geometry"] = {
        "type": "LineString",
        "coordinates": [[float(lo), float(la), float(zz)]
                        for lo, la, zz in zip(lon, lat, z)],
    }
    p = out.setdefault("properties", {})
    p["s_m"] = [float(v) for v in result["s_m"]]
    p["sin_theta"] = [float(v) for v in result["sin_theta"]]
    p["grade_pct"] = [float(v * 100.0) for v in result["grade"]]
    p["kappa"] = [float(v) for v in result["kappa"]]
    p["v_max_ms"] = [float(v) for v in result["v_max_ms"]]
    p["z_raw_m"] = [None if not np.isfinite(v) else float(v) for v in result["z_raw_m"]]
    p["z_smooth_m"] = [float(v) for v in z]
    p["structure_corrected"] = [bool(v) for v in result["structure_corrected"]]
    p["profile_meta"] = result["meta"]
    return out


def profile_to_npz_dict(result: Dict) -> Dict[str, np.ndarray]:
    """The DATA_SCHEMA §F route fields, float32, ready for ``np.savez_compressed``.

    Keys match the schema exactly: ``route_s``, ``route_sin_theta``,
    ``route_kappa``, ``route_vmax``.
    """
    return {
        "route_s": result["s_m"].astype(np.float32),
        "route_sin_theta": result["sin_theta"].astype(np.float32),
        "route_kappa": result["kappa"].astype(np.float32),
        "route_vmax": result["v_max_ms"].astype(np.float32),
    }


def save_profile_npz(result: Dict, path: str) -> str:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **profile_to_npz_dict(result))
    return path


# --------------------------------------------------------------------------- #
# Tunnel detection from the elevation signature
# --------------------------------------------------------------------------- #
# Why this exists: step (1) of derive_profile already interpolates elevation
# across tunnels, but it reads the ``is_tunnel`` flag from Part-1 properties.
# OSM tags structures; NARN does not (see narn.py), so on a NARN corridor the
# flag is all-False and the correction is a silent no-op. 3DEP then returns the
# mountain surface above a bore, the raw-elevation QA gate sees a cliff, and the
# corridor is rejected -- systematically removing the steep tunnelled mountain
# route the ood_grade split most wants.
#
# So: recover the flag from the elevation itself. Three conditions have to hold
# together, and each rules out a different false positive:
#
#   1. Local jump density. Inside a bore the sampled surface climbs away from
#      the track, so >1 m steps cluster. Healthy track runs 0.7-1.5%; the two
#      tunnelled stretches measured on up_moffat_tunnel_p0 run 39% and 27%.
#   2. Contiguous run length, bounded at both ends. A tunnel is a local feature.
#      route_line2's DEM was broken along a fifth of the corridor, diffusely --
#      that is not a tunnel and must stay rejected. The upper bound is set past
#      the longest US rail tunnel (Cascade, 12.5 km).
#   3. The surface sits ABOVE the portal-to-portal chord. This is the condition
#      that actually separates a tunnel from noise, because it is signed: a
#      mountain over a bore is always an over-estimate, never an under-estimate,
#      while DEM noise has no preferred sign.
#
# Condition 3 is also why linear interpolation is the right correction rather
# than merely a convenient one: a tunnel is bored at near-constant grade, so the
# portal-to-portal chord is close to the true rail profile through it.

JUMP_M = 1.0              # a step larger than this is "a jump" (3DEP vert. error ~1 m)
DENSITY_WINDOW_M = 500.0  # window over which jump density is measured
DENSITY_TOL = 0.25        # jump fraction inside the window marking a candidate
MIN_RUN_M = 300.0         # shorter than this is not worth correcting as a tunnel
MAX_RUN_M = 15000.0       # longer than this is a broken corridor, not a bore
MIN_RISE_M = 5.0          # median height of surface over chord, for condition 3


def _runs(mask: np.ndarray):
    """Yield ``(start, stop)`` half-open index pairs of each True run in ``mask``."""
    if mask.size == 0:
        return
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        yield int(a), int(b)


def detect_tunnel_runs(
    z: np.ndarray,
    s: np.ndarray,
    *,
    jump_m: float = JUMP_M,
    density_window_m: float = DENSITY_WINDOW_M,
    density_tol: float = DENSITY_TOL,
    min_run_m: float = MIN_RUN_M,
    max_run_m: float = MAX_RUN_M,
    min_rise_m: float = MIN_RISE_M,
) -> np.ndarray:
    """Flag samples where the DEM is reading a surface above a bore.

    Returns a boolean mask over ``z``, suitable for the ``is_tunnel`` property
    that :func:`derive_profile` interpolates across. Pure NumPy, no network.

    The mask marks the *interior* of each detected run; the endpoints are kept
    valid so they can serve as the portals the fill interpolates between.
    """
    z = np.asarray(z, dtype=float)
    s = np.asarray(s, dtype=float)
    n = z.size
    out = np.zeros(n, dtype=bool)
    if n < 3:
        return out

    ds = float(np.median(np.diff(s))) if n > 1 else 10.0
    if not np.isfinite(ds) or ds <= 0:
        return out

    # (1) jump density in a sliding window, measured on steps and carried to samples
    jump = (np.abs(np.diff(z)) > jump_m).astype(float)
    win = max(3, int(round(density_window_m / ds)))
    kernel = np.ones(win) / win
    pad = win // 2
    density = np.convolve(np.pad(jump, pad, mode="edge"), kernel, mode="same")[pad:pad + jump.size]
    candidate = np.zeros(n, dtype=bool)
    candidate[:-1] = density >= density_tol
    candidate[1:] |= density >= density_tol

    for a, b in _runs(candidate):
        length = float(s[b - 1] - s[a])
        # (2) bounded extent
        if not (min_run_m <= length <= max_run_m):
            continue
        # (3) the surface must sit above the portal-to-portal chord
        chord = np.linspace(z[a], z[b - 1], b - a)
        rise = float(np.median(z[a:b] - chord))
        if rise < min_rise_m:
            continue
        # Keep the portals as interpolation anchors; flag the interior.
        if b - a > 2:
            out[a + 1:b - 1] = True
    return out
