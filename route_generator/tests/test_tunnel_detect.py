"""Offline tests for tunnel detection from the elevation signature.

Runnable with pytest or directly: ``python tests/test_tunnel_detect.py``.

The detector exists because NARN carries no structure tags, so the structure
correction in ``derive_profile`` -- which already interpolates across tunnels --
is a silent no-op on every NARN corridor, and the raw-elevation QA gate then
rejects tunnelled mountain route for having a tunnel in it.

The property that matters most is the last one: the correction must not rescue
a corridor whose DEM is genuinely broken. ``route_line2`` is the case, and it
is tested against the committed corridor rather than a synthetic stand-in.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routegen.profile import (  # noqa: E402
    MAX_RUN_M,
    MIN_RUN_M,
    _fill_invalid,
    derive_profile,
    detect_tunnel_runs,
)

_ROUTES = os.path.join(os.path.dirname(__file__), "..", "elevated_routes")
DS = 10.0


# --------------------------------------------------------------------------- #
# Synthetic corridors
# --------------------------------------------------------------------------- #
def _track(n=3000, grade=0.01, noise=0.2, seed=0):
    """Healthy track: a constant grade under ~1 m 3DEP vertical error."""
    rng = np.random.default_rng(seed)
    s = np.arange(n) * DS
    return s, grade * s + rng.normal(0, noise, n)


def _add_bore(z, i0, i1, height=120.0, noise=3.0, seed=1, sign=1.0):
    """Overlay the surface a DEM reads above (or, sign=-1, below) the track."""
    rng = np.random.default_rng(seed)
    out = z.copy()
    hill = height * np.sin(np.linspace(0, np.pi, i1 - i0))
    out[i0:i1] += sign * (hill + rng.normal(0, noise, i1 - i0))
    return out


def test_healthy_track_is_not_flagged():
    s, z = _track()
    assert detect_tunnel_runs(z, s).sum() == 0


def test_bore_is_flagged_over_its_extent():
    s, z = _track()
    i0, i1 = 1000, 1400
    m = detect_tunnel_runs(_add_bore(z, i0, i1), s)
    assert m.any()
    idx = np.flatnonzero(m)
    # The run covers the bore; a little overhang onto adjacent track is fine
    # (it interpolates across track that is already near-straight) but it must
    # not wander off into the rest of the corridor.
    assert idx.min() <= i0 + 20 and idx.max() >= i1 - 20
    assert idx.min() > i0 - 100 and idx.max() < i1 + 100


def test_portals_stay_valid_as_interpolation_anchors():
    """The fill needs an anchor at each end, so the run's endpoints are kept."""
    s, z = _track()
    m = detect_tunnel_runs(_add_bore(z, 1000, 1400), s)
    idx = np.flatnonzero(m)
    assert not m[idx.min() - 1]
    assert not m[idx.max() + 1]


def test_dip_is_not_a_tunnel():
    """A cut or fill reads BELOW the chord. Only over-estimates are bores --
    this is the signed condition, and it is what rules out zero-mean noise."""
    s, z = _track()
    assert detect_tunnel_runs(_add_bore(z, 1000, 1400, sign=-1.0), s).sum() == 0


def test_diffuse_noise_is_not_a_tunnel():
    """route_line2's pathology in synthetic form: broken along the corridor,
    with no coherent rise. Tested at a length that clears the extent bound, so
    the signed condition is doing the work rather than the run-length cap."""
    rng = np.random.default_rng(2)
    n = 800  # 8 km, inside MAX_RUN_M
    s = np.arange(n) * DS
    for sd in (0.8, 1.2, 2.0):
        z = 0.01 * s + rng.normal(0, sd, n)
        assert detect_tunnel_runs(z, s).sum() == 0, sd


def test_run_shorter_than_the_minimum_is_ignored():
    s, z = _track()
    span = int(MIN_RUN_M / DS) // 3
    assert detect_tunnel_runs(_add_bore(z, 1000, 1000 + span), s).sum() == 0


def test_run_longer_than_the_maximum_is_ignored():
    """Past the longest US rail tunnel a corridor-wide defect is not a bore."""
    n = int(2 * MAX_RUN_M / DS)
    s, z = _track(n=n)
    assert detect_tunnel_runs(_add_bore(z, 1, n - 1, height=400.0), s).sum() == 0


def test_a_smooth_bore_is_not_detected_and_need_not_be():
    """Detection keys on jump density, which is also what the QA gate keys on.
    So a bore whose sampled surface happens to be smooth is not detected --
    and it is also not rejected, which is why the two criteria are the same
    one. The detector's job is exactly to rescue what the gate would throw
    out, no more. A smooth bore still biases the grade field; catching that
    would need a structure source rather than the elevation signature.
    """
    s, z = _track(noise=0.05)
    zb = _add_bore(z, 1000, 1400, noise=0.0)
    assert detect_tunnel_runs(zb, s).sum() == 0
    assert _jump_frac(zb) < 0.05          # and the gate passes it regardless


def test_degenerate_inputs_do_not_raise():
    for n in (0, 1, 2, 3):
        s = np.arange(n) * DS
        assert detect_tunnel_runs(np.zeros(n), s).shape == (n,)


def test_correction_removes_the_phantom_grade():
    """End to end: the flag feeds derive_profile's structure correction, and
    the grade spike a bore puts in the field goes away."""
    s, z = _track(noise=0.2)
    i0, i1 = 1000, 1400
    zb = _add_bore(z, i0, i1, noise=3.0)
    n = zb.size
    feat = {
        "geometry": {"coordinates": [[-104.0 + i * 1e-4, 38.0, float(zb[i])]
                                     for i in range(n)]},
        "properties": {"s_m": s.tolist()},
    }
    before = derive_profile(feat, smooth_window_m=800.0, grade_clip=0.04)
    feat["properties"]["is_tunnel"] = detect_tunnel_runs(zb, s).tolist()
    after = derive_profile(feat, smooth_window_m=800.0, grade_clip=0.04)

    g_before = np.abs(np.asarray(before["grade"])).max()
    g_after = np.abs(np.asarray(after["grade"])).max()
    assert g_before > 0.03           # the bore drives the field into the clamp
    assert g_after < 0.5 * g_before  # and the correction takes it back out


# --------------------------------------------------------------------------- #
# The committed corridors -- real 3DEP, real DEM defects
# --------------------------------------------------------------------------- #
def _load(name):
    with open(os.path.join(_ROUTES, name)) as fh:
        d = json.load(fh)
    coords = d["geometry"]["coordinates"]
    props = d.get("properties", {})
    n = len(coords)
    z = np.array([(c[2] if len(c) > 2 and c[2] is not None else np.nan)
                  for c in coords], dtype=float)
    s = np.asarray(props.get("s_m", np.arange(n) * DS), dtype=float)
    return s, z


def _jump_frac(z):
    z = z[np.isfinite(z)]
    return float((np.abs(np.diff(z)) > 1.0).mean())


def test_no_false_positives_on_healthy_corridors():
    """The four corridors that pass Phase A QA. None is tunnelled (their OSM
    is_tunnel arrays are empty), so any detection here is a false positive."""
    for name in ("route_line0_49km.geojson", "route_line3_29km.geojson",
                 "route_line4_27km.geojson", "route_line5_22km.geojson"):
        s, z = _load(name)
        assert detect_tunnel_runs(z, s).sum() == 0, name


def test_route_line2_stays_rejected_after_correction():
    """The load-bearing one. route_line2's DEM is broken, not tunnelled, and
    correcting elevation before the gate must not smuggle it back in."""
    s, z = _load("route_line2_33km.geojson")
    m = detect_tunnel_runs(z, s)
    z_corr = _fill_invalid(z, s, m | ~np.isfinite(z))
    assert _jump_frac(z) > 0.15
    assert _jump_frac(z_corr) > 0.05     # still far over RAW_JUMP_TOL
    assert m.sum() * DS < 0.05 * s[-1]   # and barely anything was touched


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok    {name}")
            except Exception:
                fails += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print("\n%s" % ("all passed" if not fails else "%d failed" % fails))
    sys.exit(1 if fails else 0)
