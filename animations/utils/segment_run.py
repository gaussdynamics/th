"""Build and export the three-vehicle corridor run the matrix scene animates.

A deliberately small run: one SD70 and two loaded coal hoppers over ~1.6 km of
``route_line5_22km``, chosen because that window has real relief (13 m of climb),
curves down to a 550 m radius, and a 20 m/s limit -- enough for the coupler
forces to swing from draft to buff without anything dramatic happening.

Three vehicles is the point. ``state`` is 3x4, ``node_static`` 3x7,
``edge_dynamic`` 2x3 and ``edge_static`` 2x6, so every matrix in the schema fits
on screen with its numbers legible, and each row can be wired by a leader line
to the vehicle or coupler it describes.

The route is carried three ways, all from the same corridor:

* the arc-length fields (``seg_s``/``sin_theta``/``kappa``/``vmax``) the RHS
  actually samples -- this is what the simulator sees;
* the real planform and ground elevation (``seg_x``/``seg_y``/``seg_z``), read
  from ``route_generator/elevated_routes/<id>.geojson``, which is OSM geometry
  with USGS 3DEP elevation on every vertex -- this is what the map view draws;
* a terrain field for contour lines, which is **interpolated from the track's
  own elevation** and is therefore real only along the alignment (see
  ``_terrain_grid``).

Run it from the repo root::

    python animations/utils/segment_run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simulator2.catalog import CarLibrary
from simulator2.consist import build_consist, from_blocks
from simulator2.route import RouteProfile
from simulator2.scenarios import make_scenario_from_consist
from simulator2.simulate import simulate_train_tensorized
from simulator2.state_schema import EdgeChannel, NodeChannel

from animations.utils.terrain_dem import fetch_dem_grid

#: Corridor window. Picked by scoring 1.6 km windows over every corridor for
#: grade spread and mean curvature, skipping ``route_line2_33km`` whose grade
#: field sits on the +-4 % clamp for a quarter of its length.
ROUTE_ID = "route_line5_22km"
SEG_S0_M = 4780.0
SEG_S1_M = 6370.0

#: The consist, head to tail.
BLOCKS = [("loco_sd70", 1), ("coal_hopper_loaded", 2)]

T_SPAN = (0.0, 100.0)
N_TIME_SAMPLES = 1001
V0_MPS = 13.0


def _control_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Breakpoints for a plain run: pull over the climb, ease off, brake down.

    Piecewise-linear in time and interpolated, so the control panel in the scene
    is reading the same numbers the RHS is. Nothing here is closed-loop -- the
    speed that comes out is whatever the grade allows.
    """
    t_knots = np.array([0.0, 12.0, 30.0, 46.0, 58.0, 72.0, 84.0, 100.0])
    # Fraction of the locomotive's F_trac_max (300 kN). A percent of grade costs
    # this consist 32 kN, so these notches are what it takes to hold line speed
    # over the two climbs and no more.
    trac = np.array([0.14, 0.20, 0.20, 0.12, 0.30, 0.06, 0.0, 0.0])
    # Fraction of each vehicle's F_brk_max, applied train-wide. Light: enough to
    # push the couplers into buff at the end without stopping the train.
    brake = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.08])
    return t_knots, trac, brake


def _load_planform(
    route_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Real (east, north, elevation) in metres for every route vertex.

    The elevated GeoJSON has one vertex per arc-length node, so these line up
    index-for-index with ``route_s``. Longitude/latitude are projected onto a
    local tangent plane at the route's mean latitude, which is accurate well
    past the 1.6 km we draw.
    """
    path = _REPO_ROOT / "route_generator" / "elevated_routes" / f"{route_id}.geojson"
    doc = json.loads(path.read_text(encoding="utf-8"))
    feature = doc["features"][0] if doc.get("type") == "FeatureCollection" else doc
    coords = np.asarray(feature["geometry"]["coordinates"], dtype=float)
    lon, lat, elev = coords[:, 0], coords[:, 1], coords[:, 2]
    lat0 = float(lat.mean())
    east = (lon - lon.mean()) * 111_320.0 * np.cos(np.radians(lat0))
    north = (lat - lat.mean()) * 110_540.0
    return east, north, elev, (float(lon.mean()), float(lat.mean()), lat0)


def _terrain_grid(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, n: int = 44, pad: float = 0.22
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A terrain surface for contour lines, interpolated from the alignment.

    There is no DEM in the repo -- only the ground elevation sampled *along* the
    track. This fills the rest of the box by inverse-distance weighting from
    those samples, so the contours are faithful where the track runs and are an
    interpolation everywhere else. They are drawn as map texture, not as
    surveyed terrain, and the scene labels them as interpolated.
    """
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    gx = np.linspace(x0 - px, x1 + px, n)
    gy = np.linspace(y0 - py, y1 + py, n)
    mesh_x, mesh_y = np.meshgrid(gx, gy)

    # Inverse-distance weighting, softened so the surface does not spike onto
    # each individual track vertex.
    d2 = (mesh_x[..., None] - x) ** 2 + (mesh_y[..., None] - y) ** 2
    span = max(x1 - x0, y1 - y0)
    w = 1.0 / (d2 + (0.035 * span) ** 2) ** 1.5
    grid_z = (w * z).sum(axis=-1) / w.sum(axis=-1)
    return gx, gy, grid_z


def _model_elevation(seg_s: np.ndarray, sin_theta: np.ndarray, z_datum: float) -> np.ndarray:
    """Elevation by integrating the route's own grade field.

    The raw 3DEP samples on the alignment are noisy at the metre level -- up to
    3 m between adjacent 10 m vertices, which is a 30 % grade where the route's
    fitted ``sin_theta`` says 3.5 %. Drawing those raw samples at 34x vertical
    exaggeration renders the noise, not the terrain. Integrating ``sin_theta``
    instead gives the profile the simulator actually climbs, and it is smooth
    because the route pipeline already fitted it. Anchored so the mean matches
    the DEM, so the axis still reads as real height above sea level.
    """
    dz = np.concatenate([[0.0], np.cumsum(np.diff(seg_s) * sin_theta[:-1])])
    return dz - dz.mean() + z_datum


def build_scenario():
    """The scenario object, ready to integrate."""
    library = CarLibrary.load()
    train = from_blocks("animation_three_vehicle", BLOCKS,
                        description="SD70 + 2 loaded coal hoppers")
    vehicles, couplers = build_consist(train, library)
    if len(vehicles) != 3:
        raise ValueError(f"expected a 3-vehicle consist, built {len(vehicles)}")

    route = RouteProfile.from_tensor_npz(
        _REPO_ROOT / "data" / "v2" / "routes" / f"{ROUTE_ID}.npz"
    )

    consist_len_m = sum(c.L0_m for c in couplers)
    x_lead_m = SEG_S0_M + consist_len_m

    t_knots, trac_frac, brake_frac = _control_tables()
    f_trac_max = float(vehicles[0].F_trac_max_N)

    def u_trac_cmd(t: float, i: int) -> float:
        if not vehicles[i].can_traction:
            return 0.0
        return float(np.interp(t, t_knots, trac_frac)) * f_trac_max

    def u_brk_cmd(t: float, i: int) -> float:
        return float(np.interp(t, t_knots, brake_frac)) * float(vehicles[i].F_brk_max_N)

    scenario = make_scenario_from_consist(
        route=route,
        vehicles=vehicles,
        couplers=couplers,
        x_lead_m=x_lead_m,
        t_span=T_SPAN,
        n_time_samples=N_TIME_SAMPLES,
        k_curv_scale=1.0,
        u_trac_cmd=u_trac_cmd,
        u_brk_cmd=u_brk_cmd,
    )
    # Match the dataset build: speed-independent curve resistance.
    scenario.curvature_model = "linear"

    # Roll in at line speed rather than from a standstill -- a 3-vehicle train
    # starting at rest spends the whole run accelerating and never settles.
    n = len(vehicles)
    scenario.y0[n : 2 * n] = V0_MPS
    return scenario, train, route


def export_segment_run(output_npz: Path) -> Path:
    scenario, train, route = build_scenario()
    result = simulate_train_tensorized(scenario)

    t = result.t
    h, e = result.H_hist, result.E_hist
    n = h.shape[1]

    state = h[:, :, : int(NodeChannel.MASS_KG)]                 # [T,3,4]
    node_static = h[0, :, int(NodeChannel.MASS_KG) :]           # [3,7]
    edge_dynamic = e[:, :, : int(EdgeChannel.L0_M)]             # [T,2,3]
    edge_static = e[0, :, int(EdgeChannel.L0_M) :]              # [2,6]
    edge_index = np.stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int32)

    u_trac = np.array([[scenario.u_trac_cmd(float(tk), i) for i in range(n)] for tk in t])
    u_brk = np.array([[scenario.u_brk_cmd(float(tk), i) for i in range(n)] for tk in t])

    # Route window, widened past the travelled distance so the map has margin.
    x_lead = state[:, 0, 0]
    s_lo = min(SEG_S0_M, float(x_lead.min()) - 120.0)
    s_hi = max(SEG_S1_M, float(x_lead.max()) + 120.0)
    s_full = route.s_nodes_m
    m = (s_full >= s_lo) & (s_full <= s_hi)

    east, north, elev, frame = _load_planform(ROUTE_ID)
    if east.size != s_full.size:
        raise ValueError(
            f"planform has {east.size} vertices but route field has {s_full.size}"
        )
    # Local frame: origin at the window's start vertex.
    x_off, y_off = float(east[m][0]), float(north[m][0])
    seg_x, seg_y, seg_z = east[m] - x_off, north[m] - y_off, elev[m]

    seg_z_model = _model_elevation(
        s_full[m], route.sin_theta_nodes[m], float(np.mean(seg_z))
    )
    pad = 0.25
    x_b = (float(seg_x.min() - pad * np.ptp(seg_x)),
           float(seg_x.max() + pad * np.ptp(seg_x)))
    y_b = (float(seg_y.min() - pad * np.ptp(seg_y)),
           float(seg_y.max() + pad * np.ptp(seg_y)))
    dem = fetch_dem_grid(
        _REPO_ROOT / "animations" / "data" / f"terrain_{ROUTE_ID}.npz",
        frame, (x_off, y_off), x_b, y_b,
    )
    if dem is None:
        # No DEM and no cache: fall back to the alignment-only interpolation so
        # the scene still renders, and say so in the metadata.
        grid_x, grid_y, grid_z = _terrain_grid(seg_x, seg_y, seg_z_model)
        terrain_source = "interpolated from the track elevation (no DEM available)"
    else:
        grid_x, grid_y, grid_z = dem
        terrain_source = "USGS 3DEP, sampled on a grid over the segment"

    payload = {
        "t": t.astype(np.float32),
        "state": state.astype(np.float32),
        "node_static": node_static.astype(np.float32),
        "edge_dynamic": edge_dynamic.astype(np.float32),
        "edge_static": edge_static.astype(np.float32),
        "edge_index": edge_index,
        "u_trac": u_trac.astype(np.float32),
        "u_brk": u_brk.astype(np.float32),
        "seg_s": s_full[m].astype(np.float32),
        "seg_sin_theta": route.sin_theta_nodes[m].astype(np.float32),
        "seg_kappa": route.kappa_nodes[m].astype(np.float32),
        "seg_vmax": route.v_max_nodes[m].astype(np.float32),
        "seg_x": seg_x.astype(np.float32),
        "seg_y": seg_y.astype(np.float32),
        "seg_z": seg_z.astype(np.float32),
        "seg_z_model": seg_z_model.astype(np.float32),
        "terrain_x": grid_x.astype(np.float32),
        "terrain_y": grid_y.astype(np.float32),
        "terrain_z": grid_z.astype(np.float32),
        "node_feature_names": np.array(
            ["x", "v", "z_brk", "z_trac"]
            + ["mass", "davisA", "davisB", "davisC", "trac?", "Ftrac", "Fbrk"],
            dtype=object,
        ),
        "edge_feature_names": np.array(
            ["delta", "d_dot", "F_cpl", "L0", "slack", "k_dr", "c_dr", "k_bf", "c_bf"],
            dtype=object,
        ),
        "vehicle_labels": np.array([c.car_type_id for c in train.cars], dtype=object),
        "metadata_json": json.dumps(
            {
                "source": "animations.utils.segment_run",
                "route_id": ROUTE_ID,
                "segment_s_m": [SEG_S0_M, SEG_S1_M],
                "n_vehicles": int(n),
                "n_time_steps": int(t.size),
                "curvature_model": scenario.curvature_model,
                "k_curv_scale": float(scenario.k_curv_scale),
                "v0_mps": V0_MPS,
                "terrain_source": terrain_source,
                "elevation_note": "seg_z is raw 3DEP; seg_z_model integrates route sin_theta and is what the scene draws",
            }
        ),
    }

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    return output_npz


def main() -> None:
    out = _REPO_ROOT / "animations" / "data" / "three_vehicle_segment.npz"
    written = export_segment_run(out)
    with np.load(written, allow_pickle=True) as d:
        state, t = d["state"], d["t"]
        f = d["edge_dynamic"][:, :, 2]
        print(f"wrote {written}")
        print(f"  T={t.size}  N={state.shape[1]}  duration {t[-1]:.0f} s")
        print(f"  lead chainage {state[0, 0, 0]:.0f} -> {state[-1, 0, 0]:.0f} m "
              f"({state[-1, 0, 0] - state[0, 0, 0]:.0f} m travelled)")
        print(f"  speed {state[:, 0, 1].min():.1f} .. {state[:, 0, 1].max():.1f} m/s")
        print(f"  coupler force {f.min() / 1e3:+.1f} .. {f.max() / 1e3:+.1f} kN")


if __name__ == "__main__":
    main()
