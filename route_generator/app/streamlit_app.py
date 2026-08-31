"""Route Generator UI: acquire centerlines (Tab 1) and sample elevation (Tab 2).

Run from the route_generator/ folder:

    streamlit run app/streamlit_app.py

**Tab 1 — Route acquisition.** Select an **area** (radius or bounding box). The
app pulls all rail in the box, splits the network into corridors at
junctions/line-ends, and lets you pick which corridors to add to a batch.
Secondary mode: fetch a single route by OSM **relation ID**. Picked routes are
stored to ``saved_routes/``.

**Tab 2 — Stage-2 elevation (USGS 3DEP).** Take a stored route's centerline
samples and attach a ground elevation to each ``(lat, lon)`` via USGS 3DEP,
yielding the set of 3-D points ``(lon, lat, z)`` that approximate the route.
(Curve fitting → grade/curvature is a later stage; this step only adds ``z``.)
Results are written to ``elevated_routes/``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routegen import (  # noqa: E402
    DEFAULT_ENDPOINT,
    assemble_centerline,
    centerline_records,
    centerline_to_geojson,
    build_export_bundle,
    crawl_network,
    derive_profile,
    elevate_feature,
    extract_corridors,
    feature_lonlats,
    fetch_area,
    fetch_radius,
    fetch_relation,
    filter_track_ways,
    has_elevation,
    list_saved_routes,
    load_feature,
    nearest_corridor,
    profile_to_geojson,
    resample_feature,
    route_summary,
    sample_elevations,
    save_feature,
    save_profile_npz,
    stitch_line,
    validate_profile,
)
from routegen.osm import KNOWN_ENDPOINTS  # noqa: E402
from routegen.models import TRACK_RAILWAY_VALUES  # noqa: E402

try:
    import folium  # noqa: E402
    from streamlit_folium import st_folium  # noqa: E402
    HAS_FOLIUM = True
except Exception:  # noqa: BLE001
    HAS_FOLIUM = False

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_routes"))
RESAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resampled_routes"))
ELEV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "elevated_routes"))
PROFILE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "route_profiles"))
_PALETTE_HEX = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

st.set_page_config(page_title="Route Generator", layout="wide")

_PALETTE = [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
            [148, 103, 189], [140, 86, 75], [227, 119, 194], [23, 190, 207]]

if "batch" not in st.session_state:
    st.session_state.batch = {}        # label -> geojson dict
    st.session_state.corridors = []    # list[Centerline] from last area pull
    st.session_state.current = None    # single Centerline from relation mode
    st.session_state.selected_group = None  # clicked line_group (int) or "svc_<idx>"
    st.session_state.saved = []        # names of stored routes
    st.session_state.resampled = {}    # filename -> {"feature":dict, "meta":dict}
    st.session_state.elevated = {}     # filename -> {"feature":dict, "stats":dict}
    st.session_state.profiles = {}     # filename -> {"feature":dict, "result":dict}
    st.session_state.export_bin = []   # filenames sent to the export bin (tab 5)


# --------------------------------------------------------------------------- #
# Map renderers
# --------------------------------------------------------------------------- #
def _path(cl):
    return [[float(lon), float(lat)] for lon, lat in cl.coords]


def _render_one(cl):
    """Single centerline with tunnel/bridge/gap highlights."""
    path = _path(cl)
    clat, clon = float(cl.coords[:, 1].mean()), float(cl.coords[:, 0].mean())
    try:
        import pydeck as pdk

        layers = [pdk.Layer("PathLayer", data=[{"path": path}], get_path="path",
                            get_color=[120, 120, 120], width_min_pixels=3)]

        def _pts(mask, color):
            idx = [i for i in range(cl.num_vertices) if mask[i]]
            return (pdk.Layer("ScatterplotLayer",
                              data=[{"position": path[i]} for i in idx],
                              get_position="position", get_fill_color=color,
                              get_radius=30, radius_min_pixels=2) if idx else None)

        for layer in (_pts(cl.is_tunnel, [150, 60, 200]), _pts(cl.is_bridge, [40, 120, 220])):
            if layer:
                layers.append(layer)
        if cl.gaps:
            layers.append(pdk.Layer("ScatterplotLayer",
                          data=[{"position": [g.at_lonlat[0], g.at_lonlat[1]]} for g in cl.gaps],
                          get_position="position", get_fill_color=[230, 50, 50],
                          get_radius=60, radius_min_pixels=4))
        st.pydeck_chart(pdk.Deck(layers=layers, map_style=None,
                        initial_view_state=pdk.ViewState(latitude=clat, longitude=clon, zoom=9),
                        tooltip={"text": "tunnel ●purple  bridge ●blue  gap ●red"}))
    except Exception:
        st.map(pd.DataFrame(cl.coords, columns=["lon", "lat"]))


def _render_corridors(corridors, highlight=None):
    """All corridors from an area pull, color-cycled; optional highlighted index."""
    try:
        import pydeck as pdk

        data, all_lat, all_lon = [], [], []
        for i, c in enumerate(corridors):
            p = _path(c)
            all_lat += [pt[1] for pt in p]
            all_lon += [pt[0] for pt in p]
            if highlight is not None:
                color = [220, 30, 30] if i == highlight else [205, 205, 205]
            elif c.is_service:
                color = [170, 170, 170]  # service tracks muted grey
            else:
                # Color by parent line group so fragments of one line share a hue.
                color = _PALETTE[(c.line_group or 0) % len(_PALETTE)]
            tag = f" [{c.service_kind}]" if c.is_service else f" · line {c.line_group}"
            data.append({"path": p, "color": color,
                         "name": f"#{i} · {c.length_m/1000:.1f} km{tag}"})
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("PathLayer", data=data, get_path="path",
                              get_color="color", width_min_pixels=3, pickable=True)],
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=sum(all_lat) / len(all_lat),
                longitude=sum(all_lon) / len(all_lon), zoom=8),
            tooltip={"text": "{name}"}))
    except Exception:
        import numpy as np
        pts = np.vstack([c.coords for c in corridors])
        st.map(pd.DataFrame(pts, columns=["lon", "lat"]))


def _qa_metrics(cl):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Length", f"{cl.length_m/1000:.1f} km")
    c2.metric("Ways", len(cl.used_way_ids))
    c3.metric("Vertices", cl.num_vertices)
    c4.metric("Maxspeed cov.", f"{cl.maxspeed_coverage_fraction()*100:.0f}%")
    c5.metric("Gaps", len(cl.gaps))
    c6.metric("Branches", len(cl.branches))


def _structure_table(cl):
    rec_df = pd.DataFrame(centerline_records(cl))
    seg = rec_df["s_m"].diff().fillna(0.0)
    frac = lambda m: f"{100*seg[m].sum()/seg.sum():.1f}%" if seg.sum() else "0%"
    s1, s2, s3 = st.columns(3)
    s1.metric("On bridges", frac(rec_df["is_bridge"]))
    s2.metric("In tunnels", frac(rec_df["is_tunnel"]))
    s3.metric("Cut/embankment", frac(rec_df["is_cutting"] | rec_df["is_embankment"]))
    return rec_df


def _add_to_batch(label, cl):
    gj = centerline_to_geojson(cl)
    gj["properties"]["batch_label"] = label
    st.session_state.batch[label] = gj


def _folium_map(corridors, highlight_group):
    """Clickable folium map: lines coloured by line_group; selected line in red."""
    lat0 = float(np.mean([c.coords[:, 1].mean() for c in corridors]))
    lon0 = float(np.mean([c.coords[:, 0].mean() for c in corridors]))
    m = folium.Map(location=[lat0, lon0], zoom_start=10, tiles="cartodbpositron")
    for c in corridors:
        pts = [[float(lat), float(lon)] for lon, lat in c.coords]
        is_sel = (not c.is_service) and c.line_group == highlight_group
        if is_sel:
            color, weight = "#e6194b", 6
        elif c.is_service:
            color, weight = "#999999", 2
        else:
            color, weight = _PALETTE_HEX[(c.line_group or 0) % len(_PALETTE_HEX)], 4
        tip = (f"siding/{c.service_kind}" if c.is_service
               else f"line {c.line_group} · {c.length_m/1000:.1f} km (click to select)")
        folium.PolyLine(pts, color=color, weight=weight, opacity=0.9, tooltip=tip).add_to(m)
    return m


def _selected_line(corridors, sel):
    """Resolve a selection to (member corridors, stitched line Centerline)."""
    if isinstance(sel, str) and sel.startswith("svc_"):
        c = corridors[int(sel[4:])]
        return [c], c
    members = sorted((c for c in corridors if not c.is_service and c.line_group == sel),
                     key=lambda c: (c.group_order if c.group_order is not None else 0))
    if not members:
        return [], None
    return members, stitch_line(members)


def _save_route(line, name):
    os.makedirs(SAVE_DIR, exist_ok=True)
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in name) or "route"
    path = os.path.join(SAVE_DIR, f"{safe}.geojson")
    gj = centerline_to_geojson(line)
    gj["properties"]["name"] = name
    with open(path, "w") as fh:
        json.dump(gj, fh, indent=2)
    return path


# --------------------------------------------------------------------------- #
# Tab 1 — acquisition body (controls now live on-page, not in a sidebar)
# --------------------------------------------------------------------------- #
def _acquisition_controls():
    """Build the on-page acquisition controls; return all inputs as a dict."""
    with st.expander("🛰  Acquire — fetch routes from OSM", expanded=True):
        t0, t1 = st.columns([3, 2])
        _ep_choice = t0.selectbox("Overpass endpoint", KNOWN_ENDPOINTS + ["Custom…"])
        endpoint = (t0.text_input("Custom endpoint", value=DEFAULT_ENDPOINT)
                    if _ep_choice == "Custom…" else _ep_choice)
        t0.caption("If one endpoint returns 406/429, switch to another.")
        mode = t1.radio("Selection mode",
                        ["Radius (around point)", "Bounding box", "Relation ID"])

        # Defaults so every name exists regardless of the selected mode.
        v = dict(endpoint=endpoint, mode=mode, go=False, types=[], main_only=True,
                 min_km=1.0, lat=None, lon=None, radius_km=None, crawl=False,
                 max_radius_km=150, expand_m=60, max_iter=30,
                 west=None, south=None, east=None, north=None,
                 rel_id="", gap_tol=5.0, max_jump=2000.0, allow_all=False)

        if mode == "Radius (around point)":
            st.caption("Pulls every track way intersecting the ball, un-clipped. "
                       "Default centre: Pueblo, CO.")
            c = st.columns(3)
            v["lat"] = c[0].number_input("Centre latitude", value=38.25445, format="%.5f")
            v["lon"] = c[1].number_input("Centre longitude", value=-104.60914, format="%.5f")
            v["radius_km"] = c[2].slider("Radius (km)", 1.0, 150.0, 40.0, 1.0)
            c2 = st.columns([2, 1, 2])
            v["types"] = c2[0].multiselect("Railway types", sorted(TRACK_RAILWAY_VALUES),
                                           default=["rail", "narrow_gauge"])
            v["min_km"] = c2[1].slider("Min corridor (km)", 0.0, 20.0, 1.0, 0.5)
            v["main_only"] = c2[2].checkbox("Main lines only (exclude yards/spurs)", value=True)
            v["crawl"] = st.checkbox("Follow lines beyond radius (bounded crawl)", value=False,
                                     help="Extend intersecting lines past the ball to their "
                                          "natural ends, capped so it can't flood the network.")
            if v["crawl"]:
                cc = st.columns(3)
                v["max_radius_km"] = cc[0].slider("Max crawl radius (km)", 10, 400, 150, 10)
                v["expand_m"] = cc[1].number_input("Expansion step (m)", 20, 500, 60, 10)
                v["max_iter"] = cc[2].number_input("Max iterations", 1, 100, 30, 1)
            v["go"] = st.button("Fetch radius & extract corridors", type="primary",
                                width="stretch")

        elif mode == "Bounding box":
            st.caption("Tip: read a bbox off bboxfinder.com or the OSM 'Export' tab.")
            c = st.columns(4)
            v["west"] = c[0].number_input("West (min lon)", value=-109.06, format="%.5f")
            v["south"] = c[1].number_input("South (min lat)", value=36.99, format="%.5f")
            v["east"] = c[2].number_input("East (max lon)", value=-102.04, format="%.5f")
            v["north"] = c[3].number_input("North (max lat)", value=41.00, format="%.5f")
            c2 = st.columns([2, 1, 2])
            v["types"] = c2[0].multiselect("Railway types", sorted(TRACK_RAILWAY_VALUES),
                                           default=["rail", "narrow_gauge"])
            v["min_km"] = c2[1].slider("Min corridor (km)", 0.0, 20.0, 1.0, 0.5)
            v["main_only"] = c2[2].checkbox("Main lines only (exclude yards/spurs)", value=True)
            v["go"] = st.button("Fetch area & extract corridors", type="primary",
                                width="stretch")
        else:
            v["rel_id"] = st.text_input("OSM relation ID", placeholder="e.g. 1700865")
            cc = st.columns(3)
            v["gap_tol"] = cc[0].number_input("Gap tolerance (m)", 0.0, 100.0, 5.0, 1.0)
            v["max_jump"] = cc[1].number_input("Max jump (m)", 100.0, 20000.0, 2000.0, 100.0)
            v["allow_all"] = cc[2].checkbox("Keep all ways (no track filter)", value=False)
            v["go"] = st.button("Fetch & assemble", type="primary", width="stretch")
    return v


def _pipeline_status_panel():
    """Stored-routes status + relation batch (relocated from the old sidebar)."""
    on_disk = list_saved_routes(SAVE_DIR)
    with st.expander(f"📚  Library & pipeline status — {len(on_disk)} stored route(s)"):
        if on_disk:
            resampled = set(list_saved_routes(RESAMPLE_DIR))
            elevated = set(list_saved_routes(ELEV_DIR))
            profiled = set(list_saved_routes(PROFILE_DIR))
            rows = [{
                "route": fn,
                "resampled": "✅" if fn in resampled else "—",
                "elevated": "✅" if fn in elevated else "—",
                "profiled": "✅" if fn in profiled else "—",
            } for fn in on_disk]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("Click a route on the map above and 'Take & store' to save it here.")

        if st.session_state.batch:
            st.markdown(f"**Relation batch ({len(st.session_state.batch)})**")
            for label in list(st.session_state.batch):
                cols = st.columns([5, 1])
                cols[0].write(label)
                if cols[1].button("✕", key=f"rm_{label}"):
                    del st.session_state.batch[label]
                    st.rerun()
            manifest = {"features": list(st.session_state.batch.values()),
                        "labels": list(st.session_state.batch.keys())}
            st.download_button("Download batch manifest", json.dumps(manifest, indent=2),
                               file_name="route_batch.geojson.json", mime="application/json")


def render_acquisition_tab():
    st.header("Railway route acquisition")
    st.caption("Part 1: OSM → robust centerlines + QA. Pick routes here, then send them "
               "down the pipeline in the next tabs.")

    ctl = _acquisition_controls()
    endpoint, mode, go = ctl["endpoint"], ctl["mode"], ctl["go"]
    types, main_only, min_km = ctl["types"], ctl["main_only"], ctl["min_km"]
    lat, lon, radius_km = ctl["lat"], ctl["lon"], ctl["radius_km"]
    crawl, max_radius_km = ctl["crawl"], ctl["max_radius_km"]
    expand_m, max_iter = ctl["expand_m"], ctl["max_iter"]
    west, south, east, north = ctl["west"], ctl["south"], ctl["east"], ctl["north"]
    rel_id, gap_tol, max_jump, allow_all = (ctl["rel_id"], ctl["gap_tol"],
                                            ctl["max_jump"], ctl["allow_all"])

    if go and mode == "Radius (around point)":
        if not types:
            st.error("Pick at least one railway type.")
        else:
            try:
                # The seed is always tiled (many small queries) to avoid timeouts;
                # crawl off == 0 expansion iterations.
                bar = st.progress(0.0)
                note = st.empty()
                mi = int(max_iter) if crawl else 0

                def _cb(it, nways, nfront):
                    msg = (f"Crawl iteration {it}: {nways} ways, {nfront} open ends"
                           if it else f"Tiled seed: {nways} ways")
                    note.write(msg)
                    bar.progress(min(it / float(mi), 1.0) if mi else 0.5)

                rel = crawl_network(lat, lon, radius_km * 1000, endpoint=endpoint,
                                    railway_values=tuple(types),
                                    max_radius_m=(max_radius_km * 1000 if crawl else radius_km * 1000),
                                    expand_radius_m=(expand_m if crawl else 60.0),
                                    max_iterations=mi, main_only=main_only, progress=_cb)
                bar.empty()
                note.empty()
                st.session_state.corridors = extract_corridors(rel, min_length_m=min_km * 1000)
                st.session_state.current = None
            except Exception as exc:  # noqa: BLE001
                st.session_state.corridors = []
                st.error(str(exc))

    if go and mode == "Bounding box":
        if not types:
            st.error("Pick at least one railway type.")
        elif not (south < north and west < east):
            st.error("Bounding box invalid: need South<North and West<East.")
        else:
            with st.spinner("Querying Overpass and extracting corridors…"):
                try:
                    rel = fetch_area(south, west, north, east, endpoint=endpoint,
                                     railway_values=tuple(types), main_only=main_only)
                    st.session_state.corridors = extract_corridors(rel, min_length_m=min_km * 1000)
                    st.session_state.current = None
                except Exception as exc:  # noqa: BLE001
                    st.session_state.corridors = []
                    st.error(str(exc))

    if go and mode == "Relation ID":
        if not rel_id.strip().isdigit():
            st.error("Enter a numeric OSM relation ID.")
        else:
            with st.spinner("Querying Overpass and assembling centerline…"):
                try:
                    rel = fetch_relation(int(rel_id), endpoint=endpoint)
                    has_track = any(w.tags.get("railway") in TRACK_RAILWAY_VALUES for w in rel.ways)
                    track = rel if (allow_all or not has_track) else filter_track_ways(rel)
                    if not allow_all and not has_track:
                        st.warning("No track-type ways matched; kept all ways. Inspect carefully.")
                    st.session_state.current = assemble_centerline(
                        track, gap_tolerance_m=gap_tol, max_jump_m=max_jump)
                    st.session_state.corridors = []
                except Exception as exc:  # noqa: BLE001
                    st.session_state.current = None
                    st.error(str(exc))

    # ----------------------------------------------------------------------- #
    # Area results
    # ----------------------------------------------------------------------- #
    corridors = st.session_state.corridors
    if corridors:
        n_lines = len({c.line_group for c in corridors if not c.is_service})
        n_service = sum(1 for c in corridors if c.is_service)
        total_km = sum(c.length_m for c in corridors) / 1000
        st.subheader(f"{n_lines} lines · {n_service} service tracks · {total_km:.0f} km total")

        if HAS_FOLIUM:
            st.caption("**Click a line** to select its full route (red), then store it below. "
                       "Coloured = main/branch lines · grey = service track.")
            sel = st.session_state.selected_group
            fmap = _folium_map(corridors, sel)
            state = st_folium(fmap, height=560, use_container_width=True,
                              returned_objects=["last_clicked"], key="selmap")
            click = (state or {}).get("last_clicked")
            if click:
                idx, dist = nearest_corridor(corridors, click["lat"], click["lng"])
                if idx >= 0 and dist <= 1500:
                    c = corridors[idx]
                    new_sel = f"svc_{idx}" if c.is_service else c.line_group
                    if new_sel != sel:
                        st.session_state.selected_group = new_sel
                        st.rerun()

            sel = st.session_state.selected_group
            members, line = _selected_line(corridors, sel) if sel is not None else ([], None)
            if line is not None:
                st.markdown(f"### Selected route — {line.length_m/1000:.1f} km")
                _qa_metrics(line)
                _structure_table(line)
                default_name = f"route_line{sel}_{line.length_m/1000:.0f}km" \
                    if not (isinstance(sel, str)) else f"siding_{line.length_m/1000:.1f}km"
                name = st.text_input("Route name", value=default_name)
                b1, b2 = st.columns(2)
                if b1.button("✅ Take & store route", type="primary", width="stretch"):
                    path = _save_route(line, name)
                    if name not in st.session_state.saved:
                        st.session_state.saved.append(name)
                    st.success(f"Stored '{name}' ({line.length_m/1000:.1f} km, "
                               f"{line.num_vertices} pts) → saved_routes/{os.path.basename(path)}")
                b2.download_button("Download GeoJSON", json.dumps(centerline_to_geojson(line), indent=2),
                                   file_name=f"{name}.geojson", mime="application/geo+json",
                                   width="stretch")
            else:
                st.info("Click a route on the map to select it.")
        else:
            st.warning("For click-to-select, install: `pip install streamlit-folium folium`. "
                       "Showing the static map for now.")
            _render_corridors(corridors)

        with st.expander("All fragments / lines table"):
            table = pd.DataFrame([{
                "idx": i, "line_group": c.line_group, "group_order": c.group_order,
                "service": c.service_kind or "", "length_km": round(c.length_m / 1000, 2),
                "vertices": c.num_vertices, "maxspeed_cov": round(c.maxspeed_coverage_fraction(), 2),
            } for i, c in enumerate(corridors)]).sort_values(["line_group", "group_order"])
            st.dataframe(table, width="stretch", hide_index=True, height=240)

    # ----------------------------------------------------------------------- #
    # Relation result
    # ----------------------------------------------------------------------- #
    cl = st.session_state.current
    if cl is not None:
        st.subheader(cl.relation_tags.get("name", f"relation/{cl.relation_id}"))
        _qa_metrics(cl)
        _render_one(cl)
        if cl.gaps:
            with st.expander(f"⚠ {len(cl.gaps)} gap(s) bridged", expanded=True):
                st.dataframe(pd.DataFrame([{
                    "after_way": g.after_way_id, "before_way": g.before_way_id,
                    "distance_m": round(g.distance_m, 1)} for g in cl.gaps]),
                    width="stretch", hide_index=True)
        if cl.branches:
            with st.expander(f"⚠ {len(cl.branches)} branch point(s)"):
                st.dataframe(pd.DataFrame([{
                    "node_id": b.node_id, "degree": b.degree,
                    "way_ids": ", ".join(map(str, b.way_ids))} for b in cl.branches]),
                    width="stretch", hide_index=True)
        if cl.dropped_way_ids:
            st.info(f"Dropped {len(cl.dropped_way_ids)} disconnected way(s): {cl.dropped_way_ids}")
        rec_df = _structure_table(cl)
        with st.expander("Per-vertex table"):
            st.dataframe(rec_df, width="stretch", height=300)
        e1, e2, e3 = st.columns(3)
        gj = centerline_to_geojson(cl)
        e1.download_button("Download GeoJSON", json.dumps(gj, indent=2),
                           file_name=f"relation_{cl.relation_id}.geojson", mime="application/geo+json")
        e2.download_button("Download vertices CSV", rec_df.to_csv(index=False),
                           file_name=f"relation_{cl.relation_id}_vertices.csv", mime="text/csv")
        if e3.button("➕ Add to batch", width="stretch"):
            _add_to_batch(f"relation_{cl.relation_id}", cl)
            st.success(f"Added. Batch now {len(st.session_state.batch)}.")

    st.divider()
    _pipeline_status_panel()


# --------------------------------------------------------------------------- #
# Tab 2 — Resample to a uniform arc-length grid
# --------------------------------------------------------------------------- #
def _render_resampled(feature, name):
    """2-D plan-view graph + spacing chart + table for one resampled route."""
    coords = feature["geometry"]["coordinates"]
    lon = np.array([c[0] for c in coords], dtype=float)
    lat = np.array([c[1] for c in coords], dtype=float)
    p = feature["properties"]
    s = np.array(p.get("s_m", list(range(len(coords)))), dtype=float)
    seg = np.diff(s)
    meta = p.get("resample_meta", {})
    is_long = np.array(p.get("is_long_segment", [False] * len(coords)), dtype=bool)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points", f"{len(coords)}", help="After resampling")
    c2.metric("Δs", f"{meta.get('ds_m', float('nan')):.0f} m")
    c3.metric("Length", f"{p.get('length_m', s[-1])/1000:.1f} km")
    c4.metric("Interp. (long-seg) pts", int(is_long.sum()),
              help="Points interpolated across sparse OSM spans / bridged gaps")

    # --- 2-D plan-view graph (lon vs lat) -----------------------------------
    st.markdown("**Plan view** — resampled centerline (lon vs lat). Orange = points "
                "interpolated across long OSM gaps.")
    plan = pd.DataFrame({
        "lon": lon, "lat": lat,
        "kind": np.where(is_long, "interpolated", "on-node"),
    })
    # Fit the viewport to the route with a small margin (Altair otherwise forces
    # zero onto the axes, leaving the route as a dot in the corner). zoom/pan kept.
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    pad_x = max((lon_max - lon_min) * 0.06, 1e-4)
    pad_y = max((lat_max - lat_min) * 0.06, 1e-4)
    try:
        import altair as alt
        chart = (
            alt.Chart(plan)
            .mark_circle(size=18)
            .encode(
                x=alt.X("lon:Q", title="lon", scale=alt.Scale(
                    domain=[lon_min - pad_x, lon_max + pad_x], zero=False, nice=False)),
                y=alt.Y("lat:Q", title="lat", scale=alt.Scale(
                    domain=[lat_min - pad_y, lat_max + pad_y], zero=False, nice=False)),
                color=alt.Color("kind:N", title=""),
                tooltip=["lon", "lat", "kind"],
            )
            .properties(height=420)
            .interactive()  # scroll to zoom, drag to pan
        )
        st.altair_chart(chart, width="stretch")
    except Exception:
        st.scatter_chart(plan, x="lon", y="lat", color="kind", height=420)

    # --- spacing check ------------------------------------------------------
    st.markdown("**Sample spacing** along the route (should sit flat at Δs).")
    st.line_chart(pd.DataFrame({"distance_km": s[1:] / 1000.0, "spacing_m": seg}
                               ).set_index("distance_km"), height=200)

    with st.expander("Per-vertex table"):
        cols = {"vertex": np.arange(len(coords)), "s_km": np.round(s / 1000.0, 4),
                "lon": np.round(lon, 6), "lat": np.round(lat, 6),
                "seg_len_m": np.round(np.array(p.get("seg_len_m", [np.nan]*len(coords))), 1),
                "is_long_segment": is_long}
        if "maxspeed_kph" in p:
            cols["maxspeed_kph"] = p["maxspeed_kph"]
        st.dataframe(pd.DataFrame(cols), width="stretch", hide_index=True, height=300)

    st.download_button("Download resampled GeoJSON", json.dumps(feature, indent=2),
                       file_name=name, mime="application/geo+json", width="stretch")


def render_resample_tab():
    st.header("Stage 1.5 — Resample to a uniform grid")
    st.caption("Rewrite a stored route onto a fixed Δs chainage grid so the elevation "
               "step gets uniform coverage on tangents (where OSM places few nodes but "
               "grade still lives). Geometry + feature bookkeeping only — no smoothing or "
               "grade; that's the curve-fit stage.")

    saved = list_saved_routes(SAVE_DIR)
    if not saved:
        st.info("No stored routes yet. Pick and store routes in the **Route acquisition** "
                "tab first.")
        return

    done = set(list_saved_routes(RESAMPLE_DIR))
    label = lambda f: f + ("  ✅ resampled" if f in done else "")
    picks = st.multiselect("Routes to resample", saved, default=saved[:1], format_func=label)

    cc1, cc2, cc3 = st.columns([2, 2, 1])
    ds_m = cc1.slider("Δs — sample spacing (m)", 2.0, 50.0, 10.0, 1.0,
                      help="10 m matches 3DEP (~1/3 arc-second ≈ 10 m). Finer buys "
                           "correlated samples, not new terrain detail.")
    snap = cc2.checkbox("Snap feature transitions", value=True,
                        help="Insert a sample exactly where maxspeed / bridge / tunnel "
                             "changes, so step features aren't smeared by a cell width.")
    overwrite = cc3.checkbox("Re-run", value=True, help="Re-resample if already done.")

    if st.button("📐 Resample", type="primary", width="stretch", disabled=not picks):
        for fname in picks:
            if (fname in done) and not overwrite:
                continue
            try:
                feat = load_feature(os.path.join(SAVE_DIR, fname))
                out = resample_feature(feat, ds_m=float(ds_m), snap_transitions=snap)
                save_feature(out, os.path.join(RESAMPLE_DIR, fname))
                st.session_state.resampled[fname] = {"feature": out,
                                                     "meta": out["properties"]["resample_meta"]}
                m = out["properties"]["resample_meta"]
                st.write(f"{fname}: {m['n_in']} → {m['n_out']} pts at {m['ds_m']:.0f} m "
                         f"({m['n_snapped']} snapped, {m['n_long_segment']} long-seg) "
                         f"→ resampled_routes/{fname}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{fname}: {exc}")
        st.success("Done. Resampled routes written to resampled_routes/.")

    # --- inspect a resampled route ------------------------------------------
    available = {f: None for f in list_saved_routes(RESAMPLE_DIR)}
    available.update({f: st.session_state.resampled[f]["feature"]
                      for f in st.session_state.resampled})
    if not available:
        st.caption("Run resampling above to see the 2-D plan view and spacing here.")
        return

    st.divider()
    choice = st.selectbox("Inspect resampled route", sorted(available), key="resample_pick")
    feature = st.session_state.resampled.get(choice, {}).get("feature")
    if feature is None:
        feature = load_feature(os.path.join(RESAMPLE_DIR, choice))
    _render_resampled(feature, choice)


# --------------------------------------------------------------------------- #
# Tab 3 — Stage-2 elevation (USGS 3DEP)
# --------------------------------------------------------------------------- #
def _elev_color(t: float):
    """Normalized elevation t∈[0,1] -> blue(low)→red(high) RGB."""
    t = 0.0 if t != t else max(0.0, min(1.0, t))  # clamp, NaN->0
    return [int(255 * t), 70, int(255 * (1 - t))]


def _render_elevated(feature, name):
    """3-D point view + elevation profile + per-vertex table for one route."""
    coords = feature["geometry"]["coordinates"]
    lon = np.array([c[0] for c in coords], dtype=float)
    lat = np.array([c[1] for c in coords], dtype=float)
    z = np.array([(c[2] if len(c) > 2 and c[2] is not None else np.nan) for c in coords],
                 dtype=float)
    s_m = np.array(feature["properties"].get("s_m", list(range(len(coords)))), dtype=float)
    finite = np.isfinite(z)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vertices", len(coords))
    c2.metric("With elevation", int(finite.sum()))
    c3.metric("Missing", int((~finite).sum()))
    if finite.any():
        c4.metric("Elevation range", f"{np.nanmin(z):.0f}–{np.nanmax(z):.0f} m")
        c5.metric("Total relief", f"{(np.nanmax(z) - np.nanmin(z)):.0f} m")

    # --- 3-D point cloud (lon, lat, z) coloured by elevation ----------------
    st.markdown("**3-D centerline points** (colour = elevation, low ●blue → high ●red)")
    st.caption("Drag to pan · scroll to zoom · **hold ⌃/⌘ (or right-drag) and drag to "
               "rotate & tilt** the 3-D view.")
    try:
        import pydeck as pdk

        if finite.any():
            zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
            span = (zmax - zmin) or 1.0
        else:
            zmin, span = 0.0, 1.0
        pts = []
        for i in range(len(coords)):
            zi = z[i]
            t = (zi - zmin) / span if np.isfinite(zi) else float("nan")
            pts.append({
                "position": [float(lon[i]), float(lat[i]),
                             float(zi) if np.isfinite(zi) else zmin],
                "color": _elev_color(t),
                "tip": (f"s={s_m[i]/1000:.2f} km · "
                        + (f"{zi:.0f} m" if np.isfinite(zi) else "no data")),
            })
        path = [[float(lon[i]), float(lat[i])] for i in range(len(coords))]
        layers = [
            pdk.Layer("PathLayer", data=[{"path": path}], get_path="path",
                      get_color=[150, 150, 150], width_min_pixels=2),
            pdk.Layer("ScatterplotLayer", data=pts, get_position="position",
                      get_fill_color="color", get_radius=40, radius_min_pixels=2,
                      pickable=True),
        ]
        st.pydeck_chart(pdk.Deck(
            layers=layers, map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=float(lat.mean()), longitude=float(lon.mean()),
                zoom=9, pitch=55, bearing=0),
            tooltip={"text": "{tip}"}))
    except Exception:
        st.map(pd.DataFrame({"lon": lon, "lat": lat}))

    # --- elevation profile vs arc length ------------------------------------
    st.markdown("**Elevation profile** (ground elevation vs arc length)")
    prof = pd.DataFrame({"distance_km": s_m / 1000.0, "elevation_m": z}).set_index("distance_km")
    st.line_chart(prof, height=260)

    # --- per-vertex table ---------------------------------------------------
    with st.expander("Per-vertex table (lon, lat, elevation)"):
        tbl = pd.DataFrame({
            "vertex": np.arange(len(coords)),
            "s_km": np.round(s_m / 1000.0, 4),
            "lon": np.round(lon, 6),
            "lat": np.round(lat, 6),
            "elevation_m": np.round(z, 2),
        })
        st.dataframe(tbl, width="stretch", hide_index=True, height=300)

    st.download_button(
        "Download elevated GeoJSON (3-D)", json.dumps(feature, indent=2),
        file_name=name, mime="application/geo+json", width="stretch")


def render_elevation_tab():
    st.header("Stage 2 — Elevation (USGS 3DEP)")
    st.caption("Attach a ground elevation to every centerline sample. Output = the set of "
               "3-D points (lon, lat, z) that approximate the route. Sample the **resampled** "
               "routes so tangents get coverage; raw saved routes also work. Curve fitting → "
               "grade/curvature comes in a later stage.")

    has_resampled = bool(list_saved_routes(RESAMPLE_DIR))
    src_label = st.radio(
        "Source", ["Resampled (uniform Δs) — recommended", "Raw saved routes"],
        index=0 if has_resampled else 1, horizontal=True)
    src_dir = RESAMPLE_DIR if src_label.startswith("Resampled") else SAVE_DIR

    saved = list_saved_routes(src_dir)
    if not saved:
        if src_dir == RESAMPLE_DIR:
            st.info("No resampled routes yet. Run the **Resample** tab first, or switch the "
                    "source above to raw saved routes.")
        else:
            st.info("No stored routes yet. Pick and store routes in the **Route acquisition** "
                    "tab first.")
        return

    done = set(list_saved_routes(ELEV_DIR))
    label = lambda f: f"{f}" + ("  ✅ elevated" if f in done else "")
    picks = st.multiselect("Routes to process", saved, default=saved[:1],
                           format_func=label)

    cc1, cc2, cc3 = st.columns([2, 1, 1])
    method = cc1.radio("USGS source", ["batch", "epqs"], horizontal=True,
                       help="batch = 3DEP ImageServer getSamples (fast, many points per "
                            "request) with per-point EPQS fallback. epqs = EPQS for every "
                            "point (slower, simplest service).")
    workers = cc2.slider("EPQS workers", 1, 16, 8, help="Concurrency for the EPQS fallback.")
    overwrite = cc3.checkbox("Re-run if already elevated", value=False)

    if st.button("⛰ Compute elevation (USGS 3DEP)", type="primary",
                 width="stretch", disabled=not picks):
        prog = st.progress(0.0)
        status = st.empty()
        for k, fname in enumerate(picks):
            if (fname in done) and not overwrite:
                status.write(f"Skipping {fname} (already elevated; tick re-run to force).")
                continue
            feature = load_feature(os.path.join(src_dir, fname))
            lonlats = feature_lonlats(feature)
            status.write(f"[{k+1}/{len(picks)}] {fname}: sampling {len(lonlats)} points…")

            def _cb(d, t, _f=fname, _k=k):
                frac = (d / t) if t else 1.0
                prog.progress(min((k + frac) / len(picks), 1.0))

            try:
                elev, stats = sample_elevations(lonlats, method=method,
                                                epqs_workers=int(workers), progress=_cb)
                out = elevate_feature(feature, elev, stats=stats)
                save_feature(out, os.path.join(ELEV_DIR, fname))
                st.session_state.elevated[fname] = {"feature": out, "stats": stats}
                if stats["n_missing"]:
                    status.write(f"{fname}: {stats['n_filled']}/{stats['n']} resolved "
                                 f"({stats['n_missing']} missing) → elevated_routes/{fname}")
                else:
                    status.write(f"{fname}: all {stats['n']} points resolved "
                                 f"(batch {stats['n_batch']}, EPQS {stats['n_epqs_fallback']}) "
                                 f"→ elevated_routes/{fname}")
            except Exception as exc:  # noqa: BLE001
                status.write("")
                st.error(f"{fname}: {exc}")
        prog.progress(1.0)
        st.success(f"Done. Elevated routes written to elevated_routes/.")

    # --- inspect a processed route ------------------------------------------
    available = {}
    available.update({f: None for f in list_saved_routes(ELEV_DIR)})  # on disk
    available.update({f: st.session_state.elevated[f]["feature"]      # in session
                      for f in st.session_state.elevated})
    if not available:
        st.caption("Run elevation above to see the 3-D points and profile here.")
        return

    st.divider()
    choice = st.selectbox("Inspect elevated route", sorted(available))
    feature = st.session_state.elevated.get(choice, {}).get("feature")
    if feature is None:
        feature = load_feature(os.path.join(ELEV_DIR, choice))
    if not has_elevation(feature):
        st.warning("That file has no elevation array — re-run elevation for it.")
        return
    _render_elevated(feature, choice)


# --------------------------------------------------------------------------- #
# Tab 4 — Route profile (Phase 3: grade / curvature / speed limit)
# --------------------------------------------------------------------------- #
def _result_from_geojson(feature):
    """Rebuild the arrays needed for plots/validation from a profile GeoJSON."""
    coords = feature["geometry"]["coordinates"]
    p = feature["properties"]
    g = lambda k: np.array(p.get(k, [np.nan] * len(coords)), dtype=float)
    z_raw = np.array([(v if v is not None else np.nan) for v in
                      p.get("z_raw_m", [None] * len(coords))], dtype=float)
    return {
        "s_m": g("s_m"), "sin_theta": g("sin_theta"), "kappa": g("kappa"),
        "v_max_ms": g("v_max_ms"), "grade": g("grade_pct") / 100.0,
        "z_raw_m": z_raw, "z_smooth_m": g("z_smooth_m"),
        "structure_corrected": np.array(p.get("structure_corrected",
                                              [False] * len(coords)), dtype=bool),
        "meta": p.get("profile_meta", {}),
    }


def _render_profile(feature, name):
    """Validation + raw/smoothed elevation, grade, curvature and v_max profiles."""
    res = _result_from_geojson(feature)
    s_km = res["s_m"] / 1000.0
    meta = res["meta"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Length", f"{meta.get('length_km', s_km[-1] if len(s_km) else 0):.1f} km")
    m2.metric("Grade max", f"{meta.get('grade_max_pct', float('nan')):.2f} %")
    m3.metric("Grade RMS", f"{meta.get('grade_rms_pct', float('nan')):.2f} %")
    mr = meta.get("min_radius_m", float("inf"))
    m4.metric("Min radius", "—" if not np.isfinite(mr) else f"{mr:.0f} m")
    m5.metric("v_max coverage", f"{meta.get('vmax_coverage_frac', 0)*100:.0f} %")

    val = validate_profile(res)
    cells = st.columns(len(val["checks"]))
    for col, (k, ok) in zip(cells, val["checks"].items()):
        col.markdown(("✅ " if ok else "❌ ") + f"`{k}`")
    st.caption(("✅ " if val["ok"] else "⚠ ") + " · ".join(val["messages"]))

    st.markdown("**Elevation** — raw DEM vs structure-corrected vs smoothed (m)")
    st.line_chart(pd.DataFrame({
        "distance_km": s_km, "raw": res["z_raw_m"], "smoothed": res["z_smooth_m"],
    }).set_index("distance_km"), height=240)

    st.markdown("**Grade** `sinθ(s)` as % — the dominant forcing term")
    st.line_chart(pd.DataFrame({"distance_km": s_km,
                                "grade_pct": res["grade"] * 100.0}
                               ).set_index("distance_km"), height=200)

    cprof, vprof = st.columns(2)
    with cprof:
        st.markdown("**Curvature** `κ(s)` (1/m)")
        st.line_chart(pd.DataFrame({"distance_km": s_km, "kappa": res["kappa"]}
                                   ).set_index("distance_km"), height=200)
    with vprof:
        st.markdown("**Speed limit** `v_max(s)` (m/s)")
        st.line_chart(pd.DataFrame({"distance_km": s_km, "v_max_ms": res["v_max_ms"]}
                                   ).set_index("distance_km"), height=200)

    d1, d2 = st.columns(2)
    d1.download_button("Download profile GeoJSON", json.dumps(feature, indent=2),
                       file_name=name, mime="application/geo+json",
                       width="stretch")
    npz_path = os.path.join(PROFILE_DIR, name.replace(".geojson", ".npz"))
    if os.path.exists(npz_path):
        with open(npz_path, "rb") as fh:
            d2.download_button("Download RouteProfile .npz", fh.read(),
                               file_name=os.path.basename(npz_path),
                               mime="application/octet-stream", width="stretch")


def render_profile_tab():
    st.header("Stage 3 — Route profile (grade · curvature · speed limit)")
    st.caption("Derive the fields simulator2 consumes — sinθ(s), κ(s), v_max(s) — from the "
               "elevated samples. DEM noise is ~1 m, so grade is smoothed (Savitzky–Golay), "
               "not differenced raw; bridges/tunnels are corrected first. Output maps 1:1 onto "
               "RouteProfile, plus a v_max array carried alongside.")

    elevated = list_saved_routes(ELEV_DIR)
    if not elevated:
        st.info("No elevated routes yet. Run the **Elevation** tab first.")
        return

    done = set(f for f in list_saved_routes(PROFILE_DIR))
    label = lambda f: f + ("  ✅ profiled" if f in done else "")
    picks = st.multiselect("Routes to profile", elevated, default=elevated[:1],
                           format_func=label)

    with st.expander("Smoothing & physical limits", expanded=True):
        a, b, c = st.columns(3)
        smooth_window_m = a.slider("Elevation smooth window (m)", 30, 600, 200, 10,
                                   help="Longer = smoother grade, less detail. ~200 m "
                                        "suppresses ~1 m DEM noise without killing real grade.")
        poly_order = a.select_slider("Polynomial order", [2, 3, 4], value=2)
        grade_clip_pct = b.slider("Grade clamp (%)", 1.0, 6.0, 4.0, 0.5,
                                  help="Physical cap on |dz/ds|; freight ruling grades are ~1–2.5 %.")
        curv_window_m = b.slider("Curvature smooth window (m)", 50, 800, 300, 10)
        min_radius_m = c.slider("Min curve radius (m)", 30, 400, 80, 10,
                                help="Curvature clamp |κ| ≤ 1/R_min.")
        fra_class = c.select_slider("FRA fallback class (v_max)", [1, 2, 3, 4, 5], value=4,
                                    help="Track-class speed used where OSM maxspeed is missing. "
                                         "Class 4 = 60 mph freight.")

    if st.button("📈 Derive route profile", type="primary",
                 width="stretch", disabled=not picks):
        for fname in picks:
            try:
                feat = load_feature(os.path.join(ELEV_DIR, fname))
                if not has_elevation(feat):
                    st.warning(f"{fname}: no elevation array — re-run the Elevation tab.")
                    continue
                result = derive_profile(
                    feat, smooth_window_m=float(smooth_window_m), poly_order=int(poly_order),
                    grade_clip=grade_clip_pct / 100.0, curv_window_m=float(curv_window_m),
                    min_radius_m=float(min_radius_m), vmax_fallback_class=int(fra_class))
                gj = profile_to_geojson(feat, result)
                save_feature(gj, os.path.join(PROFILE_DIR, fname))
                save_profile_npz(result, os.path.join(PROFILE_DIR, fname.replace(".geojson", ".npz")))
                st.session_state.profiles[fname] = {"feature": gj, "result": result}
                m = result["meta"]
                st.write(f"{fname}: grade max {m['grade_max_pct']:.2f} % · RMS "
                         f"{m['grade_rms_pct']:.2f} % · min R "
                         f"{m['min_radius_m']:.0f} m · {m['n_structure_corrected']} structure-"
                         f"corrected pts → route_profiles/{fname} (+ .npz)")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{fname}: {exc}")
        st.success("Done. Profiles written to route_profiles/ (GeoJSON + RouteProfile .npz).")

    available = {f: None for f in list_saved_routes(PROFILE_DIR)}
    available.update({f: st.session_state.profiles[f]["feature"]
                      for f in st.session_state.profiles})
    if not available:
        st.caption("Derive a profile above to see grade / curvature / v_max here.")
        return

    st.divider()
    choice = st.selectbox("Inspect profile", sorted(available), key="profile_pick")
    feature = st.session_state.profiles.get(choice, {}).get("feature")
    if feature is None:
        feature = load_feature(os.path.join(PROFILE_DIR, choice))
    _render_profile(feature, choice)

    # --- send fully-processed routes to the export bin (tab 5) --------------
    st.divider()
    s1, s2 = st.columns(2)
    in_bin = choice in st.session_state.export_bin
    if s1.button(("✓ In export bin" if in_bin else "📦 Send this route to export bin"),
                 width="stretch", disabled=in_bin):
        st.session_state.export_bin.append(choice)
        st.rerun()
    if s2.button("📦 Send all profiled routes to export bin", width="stretch"):
        for f in available:
            if f not in st.session_state.export_bin:
                st.session_state.export_bin.append(f)
        st.rerun()
    if st.session_state.export_bin:
        st.caption(f"Export bin holds {len(st.session_state.export_bin)} route(s) — "
                   "open the **Export bin** tab to review and bundle.")


# --------------------------------------------------------------------------- #
# Tab 5 — Export bin (route-tensor handoff)
# --------------------------------------------------------------------------- #
def _load_profile_feature(fname):
    """Resolve a binned route to its profile feature (session first, then disk)."""
    feat = st.session_state.profiles.get(fname, {}).get("feature")
    if feat is None:
        path = os.path.join(PROFILE_DIR, fname)
        feat = load_feature(path) if os.path.exists(path) else None
    return feat


def render_export_tab():
    st.header("Stage 5 — Export bin (route tensor)")
    st.caption("Collected, fully-processed routes ready for the dataset build. Each exports "
               "as the canonical route tensor — s, sinθ, κ, v_max on a uniform arc-length "
               "grid (DATA_SCHEMA §F) — bundled as routes/*.npz + index.csv + manifest.json.")

    # keep the bin to routes that still have a derived profile available
    bin_routes = [f for f in st.session_state.export_bin if _load_profile_feature(f) is not None]
    st.session_state.export_bin = bin_routes
    if not bin_routes:
        st.info("Export bin is empty. In the **Route profile** tab, derive a profile and "
                "click *Send to export bin*.")
        return

    features = {f: _load_profile_feature(f) for f in bin_routes}
    summaries = [route_summary(features[f], f) for f in bin_routes]
    table = pd.DataFrame(summaries)[[
        "name", "distance_km", "n_points", "start_lat", "start_lon",
        "end_lat", "end_lon", "center_lat", "center_lon",
        "grade_max_pct", "grade_rms_pct", "min_radius_m", "vmax_coverage_frac",
    ]]
    st.subheader(f"{len(bin_routes)} route(s) · {table['distance_km'].sum():.1f} km total")
    st.dataframe(table, width="stretch", hide_index=True)

    # remove individual routes from the bin
    with st.expander("Manage bin"):
        for f in list(bin_routes):
            cols = st.columns([6, 1])
            cols[0].write(f)
            if cols[1].button("Remove", key=f"unbin_{f}"):
                st.session_state.export_bin.remove(f)
                st.rerun()

    st.divider()
    st.markdown("**Route tensor** — `[route_s, route_sin_theta, route_kappa, route_vmax]`, "
                "float32, one `.npz` per route. `v_max` is included; wiring it into the "
                "simulator's `RouteProfile` is the deferred change we'll make after regroup.")
    bundle = build_export_bundle(features)
    st.download_button("⬇ Download export bundle (.zip)", bundle,
                       file_name="route_tensors_export.zip", mime="application/zip",
                       type="primary", width="stretch")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
st.title("Route Generator")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① Route acquisition", "② Resample (uniform Δs)", "③ Elevation (USGS 3DEP)",
     "④ Route profile", "⑤ Export bin"])
with tab1:
    render_acquisition_tab()
with tab2:
    render_resample_tab()
with tab3:
    render_elevation_tab()
with tab4:
    render_profile_tab()
with tab5:
    render_export_tab()
