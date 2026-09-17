"""Three vehicles on a real corridor, and the five matrices that describe them.

One locomotive and two loaded hoppers run 1.5 km of ``route_line5_22km`` while
the schema's five arrays fill in beside them: the three node matrices (dynamic
state, static vehicle parameters, control input) and the two edge matrices
(dynamic coupler state, static coupler parameters). Every row is tagged with
the vehicle or coupler it describes, in that object's colour, so the point of
the layout -- that a row *is* an object -- is visible rather than asserted.

Static panels never change and are drawn muted; dynamic panels are heat-mapped
and their numbers run. That contrast is the reason the dataset stores them
separately (see ``DATA_SCHEMA.md`` design principle 1).

Render::

    python animations/render.py --scene three --quality l
    manim -qh animations/scenes/three_vehicle_matrices.py ThreeVehicleMatrices

Data comes from ``animations/utils/segment_run.py``; run that first if
``animations/data/three_vehicle_segment.npz`` is missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Circle,
    DecimalNumber,
    Dot,
    Line,
    Polygon,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    VMobject,
    ValueTracker,
    color_gradient,
    linear,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from animations.utils import iso_map
from animations.utils.matrix_panel import Column, MatrixPanel
from animations.utils.text_style import label

DATA = _REPO_ROOT / "animations" / "data" / "three_vehicle_segment.npz"

# -- playback ------------------------------------------------------------
#: Simulated seconds per wall-clock second. The run is 100 s; at 4x it plays in
#: 25 s, which is long enough to watch the numbers move and short enough to sit
#: through. Announced on screen.
PLAYBACK_RATE = 4.0

# -- palette -------------------------------------------------------------
BG = "#0b0f14"
INK = "#e6edf3"
DIM = "#8b9aa7"
FAINT = "#3a4650"
LOCO_COLOR = "#f2a33c"
CAR_COLORS = ["#f2a33c", "#69b3e7", "#7fd1a6"]
RAIL = "#55646f"
TRACK_COLOR = "#e8eef4"
TRAVELLED = "#f2a33c"
CONTOUR = "#4a6275"
TRAC_COLOR = "#7fd1a6"
BRK_COLOR = "#e2685f"

#: Exaggeration of coupler travel in the consist drawing. Real slack is +-25 mm
#: on a 29 m train; at true scale nothing would move.
SLACK_GAIN = 40.0

#: Metres of track per scene unit in the consist panel.
CONSIST_M_PER_UNIT = 1.0 / 0.145
TIE_SPACING_M = 5.0


def _diverging(n: int = 65):
    return color_gradient(["#4b8fd4", "#1b2229", "#e0714d"], n)


def _sequential(n: int = 65):
    return color_gradient(["#16202a", "#2f6f8f", "#7fd1a6"], n)


class _Heat:
    """Per-column colour normalisation held fixed over the whole run."""

    def __init__(self, lo: float, hi: float, diverging: bool) -> None:
        if diverging:
            m = max(abs(lo), abs(hi), 1e-9)
            self.lo, self.hi = -m, m
            self.ramp = _diverging()
        else:
            self.lo, self.hi = float(lo), float(hi)
            self.ramp = _sequential()
        if self.hi - self.lo < 1e-12:
            self.hi = self.lo + 1.0

    def __call__(self, value: float):
        a = (float(value) - self.lo) / (self.hi - self.lo)
        a = min(1.0, max(0.0, a))
        return self.ramp[int(round(a * (len(self.ramp) - 1)))]


def _sample(arr: np.ndarray, t: np.ndarray, t_now: float) -> np.ndarray:
    """Linear interpolation of a ``[T, ...]`` array at ``t_now``."""
    i = int(np.clip(np.searchsorted(t, t_now) - 1, 0, t.size - 2))
    span = t[i + 1] - t[i]
    w = 0.0 if span <= 0 else float((t_now - t[i]) / span)
    w = min(1.0, max(0.0, w))
    return arr[i] + w * (arr[i + 1] - arr[i])


def _polyline(points: np.ndarray, color, width: float, opacity: float = 1.0) -> VMobject:
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners([np.array([p[0], p[1], 0.0]) for p in points])
    return m


class ThreeVehicleMatrices(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG
        d = np.load(DATA, allow_pickle=True)
        meta = json.loads(str(d["metadata_json"]))

        t = d["t"].astype(float)
        state = d["state"].astype(float)              # [T,3,4]
        node_static = d["node_static"].astype(float)  # [3,7]
        edge_dyn = d["edge_dynamic"].astype(float)    # [T,2,3]
        edge_static = d["edge_static"].astype(float)  # [2,6]
        u_trac = d["u_trac"].astype(float)            # [T,3]
        u_brk = d["u_brk"].astype(float)              # [T,3]
        n_veh = state.shape[1]

        seg_s = d["seg_s"].astype(float)
        seg_sin = d["seg_sin_theta"].astype(float)
        seg_kap = d["seg_kappa"].astype(float)
        seg_vmax = d["seg_vmax"].astype(float)
        seg_x, seg_y = d["seg_x"].astype(float), d["seg_y"].astype(float)
        # Raw 3DEP samples are metre-noisy; the scene draws the profile the
        # simulator climbs (route sin_theta integrated). See segment_run.py.
        seg_z = d["seg_z_model"].astype(float)
        ter_x, ter_y, ter_z = (d[k].astype(float) for k in ("terrain_x", "terrain_y", "terrain_z"))

        tracker = ValueTracker(float(t[0]))

        self._header(meta)
        self._consist_panel(node_static, edge_static)
        self._map_panel(seg_x, seg_y, seg_z, ter_x, ter_y, ter_z, meta)
        self._param_panel()
        self._node_panels(node_static)
        self._edge_panels(edge_static)
        self._control_panel(t, u_trac, u_brk)
        self._legend()

        heats = {
            "x": _Heat(state[:, :, 0].min(), state[:, :, 0].max(), False),
            "v": _Heat(state[:, :, 1].min(), state[:, :, 1].max(), False),
            "zb": _Heat(0.0, max(state[:, :, 2].max(), 1.0), False),
            "zt": _Heat(0.0, max(state[:, :, 3].max(), 1.0), False),
            "ut": _Heat(0.0, max(u_trac.max(), 1.0), False),
            "ub": _Heat(0.0, max(u_brk.max(), 1.0), False),
            "delta": _Heat(edge_dyn[:, :, 0].min(), edge_dyn[:, :, 0].max(), True),
            "ddot": _Heat(edge_dyn[:, :, 1].min(), edge_dyn[:, :, 1].max(), True),
            "fcpl": _Heat(edge_dyn[:, :, 2].min(), edge_dyn[:, :, 2].max(), True),
        }
        self.force_heat = heats["fcpl"]

        def update(_m) -> None:
            now = tracker.get_value()
            st = _sample(state, t, now)          # [3,4]
            ed = _sample(edge_dyn, t, now)       # [2,3]
            ut = _sample(u_trac, t, now)         # [3]
            ub = _sample(u_brk, t, now)          # [3]

            self.time_num.set_value(now)

            # -- node matrices ---------------------------------------------
            self.p_state.set_values(st)
            ctrl = np.stack([ut, ub], axis=1)
            self.p_ctrl.set_values(ctrl)
            for i in range(n_veh):
                self.p_state.set_heat(i, 0, heats["x"](st[i, 0]))
                self.p_state.set_heat(i, 1, heats["v"](st[i, 1]))
                self.p_state.set_heat(i, 2, heats["zb"](st[i, 2]))
                self.p_state.set_heat(i, 3, heats["zt"](st[i, 3]))
                self.p_ctrl.set_heat(i, 0, heats["ut"](ut[i]))
                self.p_ctrl.set_heat(i, 1, heats["ub"](ub[i]))

            # -- edge matrices ---------------------------------------------
            self.p_edge.set_values(ed)
            for j in range(n_veh - 1):
                self.p_edge.set_heat(j, 0, heats["delta"](ed[j, 0]))
                self.p_edge.set_heat(j, 1, heats["ddot"](ed[j, 1]))
                self.p_edge.set_heat(j, 2, heats["fcpl"](ed[j, 2]))

            # -- consist ----------------------------------------------------
            self._update_consist(st, ed)

            # -- map + route parameters ------------------------------------
            s_lead = float(st[0, 0])
            self._update_map(s_lead, seg_s, seg_x, seg_y, seg_z)
            grade = float(np.interp(s_lead, seg_s, seg_sin))
            kappa = float(np.interp(s_lead, seg_s, seg_kap))
            elev = float(np.interp(s_lead, seg_s, seg_z))
            vmax = float(np.interp(s_lead, seg_s, seg_vmax))
            self.pv_s.set_value(s_lead)
            self.pv_grade.set_value(100.0 * grade)
            self.pv_kappa.set_value(1e3 * kappa)
            # Tangent track has no finite radius; clamp rather than print 1e6.
            self.pv_radius.set_value(min(1.0 / max(abs(kappa), 1e-9), 9999.0))
            self.pv_vmax.set_value(vmax)
            self.pv_elev.set_value(elev)

            # -- control playhead -------------------------------------------
            self._update_control(now, float(ut[0]), float(ub[0]))

        # Manim treats a mobject as static unless it is animated or carries an
        # updater, and renders static mobjects once into a cached background.
        # The updater below drives mobjects it does not own, so every panel has
        # to hang off the mobject that carries it -- otherwise the panels are
        # baked into that background and the run plays back frozen.
        root = VGroup(*self.mobjects)
        self.remove(*self.mobjects)
        root.add_updater(update)
        self.add(root)

        update(root)
        self.wait(0.6)
        self.play(
            tracker.animate.set_value(float(t[-1])),
            run_time=float(t[-1] - t[0]) / PLAYBACK_RATE,
            rate_func=linear,
        )
        self.wait(0.8)

    # ------------------------------------------------------------------ #
    # header                                                             #
    # ------------------------------------------------------------------ #
    def _header(self, meta: dict) -> None:
        title = label("Three Vehicle Consist", font_size=21, color=INK)
        title.to_edge(UP, buff=0.16).to_edge(LEFT, buff=0.3)
        self.add(title)

        # Fixed columns, with the value right-aligned: the readout crosses from
        # "9.9" to "10.0" mid-run, and arranging it once would let that shove
        # the unit sideways.
        y = 3.72
        t_lbl = label("t =", font_size=20, color=DIM)
        t_lbl.move_to(np.array([5.42, y, 0.0])).align_to(np.array([5.42, 0, 0]), LEFT)
        self.time_num = DecimalNumber(0.0, num_decimal_places=1, mob_class=Text,
                                      font_size=24, color=INK, edge_to_fix=RIGHT)
        self.time_num.move_to(np.array([6.78, y, 0.0])).align_to(
            np.array([6.78, 0, 0]), RIGHT)
        unit = label("s", font_size=20, color=DIM)
        unit.move_to(np.array([6.90, y, 0.0])).align_to(np.array([6.90, 0, 0]), LEFT)
        badge = label(f"playback x{PLAYBACK_RATE:g}", font_size=14, color=DIM)
        badge.move_to(np.array([7.02, y - 0.30, 0.0])).align_to(
            np.array([7.02, 0, 0]), RIGHT)
        self.add(t_lbl, self.time_num, unit, badge)

    # ------------------------------------------------------------------ #
    # consist                                                            #
    # ------------------------------------------------------------------ #
    def _consist_panel(self, node_static: np.ndarray, edge_static: np.ndarray) -> VGroup:
        anchor = np.array([-3.95, 2.35, 0.0])
        self.consist_anchor = anchor
        self.coupler_l0 = edge_static[:, 0]

        panel = VGroup()
        head = label(f"consist   coupler travel x{SLACK_GAIN:g}", font_size=16, color=DIM)
        head.move_to(anchor + np.array([0.0, 1.05, 0.0]))
        head.align_to(np.array([-7.02, 0.0, 0.0]), LEFT)
        panel.add(head)

        # Rail and scrolling ties give the run a ground speed to read against.
        rail = Line(anchor + 3.05 * LEFT + 0.52 * DOWN, anchor + 3.05 * RIGHT + 0.52 * DOWN,
                    stroke_color=RAIL, stroke_width=2.2)
        self.ties = VGroup(*[
            Line(ORIGIN, 0.13 * DOWN, stroke_color=RAIL, stroke_width=1.6)
            for _ in range(26)
        ])
        panel.add(rail, self.ties)

        self.cars: list[VGroup] = []
        for i in range(3):
            self.cars.append(self._make_vehicle(i, CAR_COLORS[i]))
            panel.add(self.cars[i])

        self.coupler_bars: list[VGroup] = []
        self.coupler_tags: list[Text] = []
        for j in range(2):
            bar = VGroup(
                Rectangle(width=0.3, height=0.075, stroke_width=0, fill_opacity=1.0),
                Rectangle(width=0.1, height=0.16, stroke_width=0, fill_opacity=1.0),
            )
            tag = label(f"C{j}", font_size=13, color=DIM)
            self.coupler_bars.append(bar)
            self.coupler_tags.append(tag)
            panel.add(bar, tag)

        self.add(panel)
        return panel

    def _make_vehicle(self, i: int, color: str) -> VGroup:
        """A locomotive or a hopper -- boxes, but boxes with a silhouette."""
        g = VGroup()
        if i == 0:
            body = RoundedRectangle(corner_radius=0.05, width=1.30, height=0.36,
                                    stroke_color=color, stroke_width=1.6,
                                    fill_color=color, fill_opacity=0.32)
            cab = Polygon(
                np.array([0.30, 0.18, 0.0]), np.array([0.62, 0.18, 0.0]),
                np.array([0.56, 0.40, 0.0]), np.array([0.34, 0.40, 0.0]),
                stroke_color=color, stroke_width=1.6, fill_color=color, fill_opacity=0.5,
            )
            stack = Rectangle(width=0.09, height=0.09, stroke_width=0,
                              fill_color=color, fill_opacity=0.7)
            stack.move_to(np.array([-0.32, 0.22, 0.0]))
            g.add(body, cab, stack)
            wheel_xs = [-0.44, -0.20, 0.22, 0.46]
        else:
            body = Polygon(
                np.array([-0.55, -0.18, 0.0]), np.array([0.55, -0.18, 0.0]),
                np.array([0.43, 0.20, 0.0]), np.array([-0.43, 0.20, 0.0]),
                stroke_color=color, stroke_width=1.6, fill_color=color, fill_opacity=0.24,
            )
            lip = Line(np.array([-0.45, 0.20, 0.0]), np.array([0.45, 0.20, 0.0]),
                       stroke_color=color, stroke_width=2.0)
            g.add(body, lip)
            wheel_xs = [-0.34, 0.34]
        for wx in wheel_xs:
            g.add(Circle(radius=0.058, stroke_color=color, stroke_width=1.3,
                         fill_color=BG, fill_opacity=1.0).move_to(np.array([wx, -0.30, 0.0])))
        g.add(label(f"V{i}", font_size=13, color=color).move_to(np.array([0.0, -0.02, 0.0])))
        return g

    def _update_consist(self, st: np.ndarray, ed: np.ndarray) -> None:
        anchor = self.consist_anchor
        upm = 1.0 / CONSIST_M_PER_UNIT  # scene units per metre

        # Car 0 is pinned; the others hang off it at their true spacing with the
        # coupler travel magnified, so bunching and stretching are visible.
        xs = [0.0]
        for j in range(2):
            gap_m = float(self.coupler_l0[j]) + SLACK_GAIN * float(ed[j, 0])
            xs.append(xs[-1] - gap_m * upm)
        centre = 0.5 * (xs[0] + xs[-1])
        for i, car in enumerate(self.cars):
            car.move_to(anchor + np.array([xs[i] - centre, 0.0, 0.0]))

        for j, bar in enumerate(self.coupler_bars):
            left = self.cars[j + 1].get_right() + 0.02 * RIGHT
            right = self.cars[j].get_left() + 0.02 * LEFT
            mid = 0.5 * (left + right)
            length = max(float(right[0] - left[0]), 0.04)
            col = self.force_heat(float(ed[j, 2]))
            bar[0].stretch_to_fit_width(length).move_to(mid).set_fill(col, opacity=1.0)
            bar[1].move_to(mid).set_fill(col, opacity=1.0)
            self.coupler_tags[j].move_to(mid + 0.26 * DOWN)

        # Ties scroll with the lead vehicle so the panel has a ground speed.
        s_lead = float(st[0, 0])
        phase = (s_lead % TIE_SPACING_M) * (1.0 / CONSIST_M_PER_UNIT)
        for k, tie in enumerate(self.ties):
            x = 3.05 - phase - k * TIE_SPACING_M * (1.0 / CONSIST_M_PER_UNIT)
            tie.move_to(anchor + np.array([x, -0.46, 0.0]))
            tie.set_opacity(1.0 if -3.05 <= x <= 3.05 else 0.0)

    # ------------------------------------------------------------------ #
    # map                                                                #
    # ------------------------------------------------------------------ #
    def _map_panel(self, seg_x, seg_y, seg_z, ter_x, ter_y, ter_z, meta) -> VGroup:
        box_c = np.array([1.62, 2.12, 0.0])
        box_w, box_h = 4.05, 2.45
        self.map_centre = box_c

        frame = Rectangle(width=box_w + 0.2, height=box_h + 0.42,
                          stroke_color=FAINT, stroke_width=1.2)
        frame.move_to(box_c)

        z_ref = float(np.min(ter_z))
        # Project terrain contours and the alignment in one frame, then fit all
        # of it into the box with a single shared scale.
        levels = iso_map.contour_levels(ter_z, step_m=2.0)
        paths = iso_map.contour_paths(ter_x, ter_y, ter_z, levels)

        all_sx, all_sy = [], []
        proj_paths = []
        for level, seg in paths:
            seg = iso_map.resample_polyline(seg, 90)
            sx, sy = iso_map.project(seg[:, 0], seg[:, 1],
                                     np.full(seg.shape[0], level), z_ref)
            proj_paths.append((level, sx, sy))
            all_sx.append(sx)
            all_sy.append(sy)
        tx, ty = iso_map.project(seg_x, seg_y, seg_z, z_ref)
        all_sx.append(tx)
        all_sy.append(ty)

        scale, cx, cy = iso_map.fit(np.concatenate(all_sx), np.concatenate(all_sy),
                                    box_w, box_h)

        def to_scene(sx, sy):
            return np.stack([scale * (sx - cx) + box_c[0],
                             scale * (sy - cy) + box_c[1]], axis=1)

        self._map_to_scene = to_scene
        self._map_zref = z_ref

        group = VGroup(frame)
        for level, sx, sy in proj_paths:
            pts = to_scene(sx, sy)
            # Every 10 m contour a touch brighter, the way a map indexes them.
            emphasised = abs(level / 10.0 - round(level / 10.0)) < 1e-6
            group.add(_polyline(pts, CONTOUR, 1.5 if emphasised else 0.9,
                                0.95 if emphasised else 0.6))

        track_pts = to_scene(tx, ty)
        self._track_pts = track_pts
        self._seg_s = None  # filled by caller via _update_map
        group.add(_polyline(track_pts, TRACK_COLOR, 2.4, 0.9))

        self.travelled = _polyline(track_pts[:2], TRAVELLED, 3.2, 1.0)
        group.add(self.travelled)

        self.train_dot = Dot(radius=0.065, color=TRAVELLED)
        self.train_halo = Circle(radius=0.14, stroke_color=TRAVELLED,
                                 stroke_width=1.4, stroke_opacity=0.65)
        group.add(self.train_dot, self.train_halo)

        cap = VGroup(
            label(f"{meta['route_id']}  |  isometric, elevation x{iso_map.Z_EXAGGERATION:g}",
                 font_size=13, color=DIM),
            label(f"2 m contours, {meta['terrain_source']}",
                 font_size=11, color=FAINT),
        ).arrange(DOWN, buff=0.04, aligned_edge=LEFT)
        cap.next_to(frame, DOWN, buff=0.06).align_to(frame, LEFT)
        group.add(cap)

        self.add(group)
        return group

    def _update_map(self, s_lead, seg_s, seg_x, seg_y, seg_z) -> None:
        x, y, z = iso_map.position_on_route(s_lead, seg_s, seg_x, seg_y, seg_z)
        sx, sy = iso_map.project(np.array([x]), np.array([y]), np.array([z]), self._map_zref)
        pt = self._map_to_scene(sx, sy)[0]
        p3 = np.array([pt[0], pt[1], 0.0])
        self.train_dot.move_to(p3)
        self.train_halo.move_to(p3)

        k = int(np.clip(np.searchsorted(seg_s, s_lead), 2, seg_s.size))
        pts = np.vstack([self._track_pts[:k], pt[None, :]])
        self.travelled.set_points_as_corners(
            [np.array([p[0], p[1], 0.0]) for p in pts]
        )

    # ------------------------------------------------------------------ #
    # route parameter readout                                            #
    # ------------------------------------------------------------------ #
    def _param_panel(self) -> VGroup:
        """Route fields sampled at the lead vehicle, as a plain readout.

        Values are right-aligned on a fixed column (``edge_to_fix=RIGHT``) so a
        number growing a digit does not shove its unit sideways mid-run.
        """
        label_x, value_x, unit_x, top_y = 3.98, 6.42, 6.52, 2.94

        title = label("route at V0", font_size=16, color=INK)
        title.move_to(np.array([label_x, top_y, 0.0])).align_to(
            np.array([label_x, 0.0, 0.0]), LEFT)

        rows = [
            ("chainage s", "m", 1, "pv_s"),
            ("grade", "%", 2, "pv_grade"),
            ("curvature", "1/km", 3, "pv_kappa"),
            ("radius", "m", 0, "pv_radius"),
            ("speed limit", "m/s", 1, "pv_vmax"),
            ("elevation", "m", 1, "pv_elev"),
        ]
        grp = VGroup(title)
        for k, (name, unit, dp, attr) in enumerate(rows):
            y = top_y - 0.33 * (k + 1)
            lbl = label(name, font_size=13, color=DIM)
            lbl.move_to(np.array([label_x, y, 0.0])).align_to(
                np.array([label_x, 0.0, 0.0]), LEFT)
            num = DecimalNumber(
                0.0, num_decimal_places=dp, mob_class=Text,
                font_size=16, color=INK, edge_to_fix=RIGHT,
            )
            num.move_to(np.array([value_x, y, 0.0])).align_to(
                np.array([value_x, 0.0, 0.0]), RIGHT)
            uni = label(unit, font_size=11, color=FAINT)
            uni.move_to(np.array([unit_x, y, 0.0])).align_to(
                np.array([unit_x, 0.0, 0.0]), LEFT)
            setattr(self, attr, num)
            grp.add(lbl, num, uni)

        self.add(grp)
        return grp

    # ------------------------------------------------------------------ #
    # matrices                                                           #
    # ------------------------------------------------------------------ #
    def _node_panels(self, node_static: np.ndarray) -> VGroup:
        row_labels = [f"V{i}" for i in range(3)]

        self.p_state = MatrixPanel(
            "State",
            [Column("x [m]", 1.0, 0, commas=False), Column("v [m/s]", 1.0, 2),
             Column("z_brk [kN]", 1e-3, 1), Column("z_trc [kN]", 1e-3, 1)],
            row_labels, CAR_COLORS, static=False, cell_w=0.70, font_size=12,
        )
        self.p_static = MatrixPanel(
            "Node Static",
            [Column("m [t]", 1e-3, 0), Column("A [N]", 1.0, 0),
             Column("B", 1.0, 1), Column("C", 1.0, 2),
             Column("trac?", 1.0, 0), Column("Ftr [kN]", 1e-3, 0),
             Column("Fbr [kN]", 1e-3, 0)],
            row_labels, CAR_COLORS, static=True, cell_w=0.52,
        )
        self.p_ctrl = MatrixPanel(
            "Control",
            [Column("u_trc [kN]", 1e-3, 1), Column("u_brk [kN]", 1e-3, 1)],
            row_labels, CAR_COLORS, static=False, cell_w=0.68,
        )
        self.p_static.set_values(node_static)

        row = VGroup(self.p_state, self.p_static, self.p_ctrl).arrange(
            RIGHT, buff=0.46, aligned_edge=UP)
        row.move_to(np.array([0.0, -0.80, 0.0]))
        row.align_to(np.array([-7.02, 0.0, 0.0]), LEFT)
        self.add(row)
        return row

    def _edge_panels(self, edge_static: np.ndarray) -> VGroup:
        row_labels = ["C0", "C1"]
        tag_colors = [DIM, DIM]

        self.p_edge = MatrixPanel(
            "Edge Dynamic",
            [Column("delta [mm]", 1e3, 1), Column("d_dot [mm/s]", 1e3, 1),
             Column("F_cpl [kN]", 1e-3, 1)],
            row_labels, tag_colors, static=False, cell_w=0.78,
        )
        self.p_edge_static = MatrixPanel(
            "Edge Static",
            [Column("L0 [m]", 1.0, 1), Column("slk [mm]", 1e3, 0),
             Column("k_dr [MN/m]", 1e-6, 1), Column("c_dr [kNs/m]", 1e-3, 0),
             Column("k_bf [MN/m]", 1e-6, 1), Column("c_bf [kNs/m]", 1e-3, 0)],
            row_labels, tag_colors, static=True, cell_w=0.56,
        )
        self.p_edge_static.set_values(edge_static)

        row = VGroup(self.p_edge, self.p_edge_static).arrange(
            RIGHT, buff=0.46, aligned_edge=UP)
        row.move_to(np.array([0.0, -2.95, 0.0]))
        row.align_to(np.array([-7.02, 0.0, 0.0]), LEFT)
        self.add(row)
        return row

    # ------------------------------------------------------------------ #
    # control signal                                                     #
    # ------------------------------------------------------------------ #
    def _control_panel(self, t, u_trac, u_brk) -> VGroup:
        x0, x1 = 0.75, 6.90
        y0, y1 = -3.45, -2.20
        self._ctrl_box = (x0, x1, y0, y1)

        f_hi = max(float(u_trac.max()), float(u_brk.max()), 1.0) * 1.12
        self._ctrl_scale = (float(t[0]), float(t[-1]), 0.0, f_hi)

        frame = VGroup(
            Line(np.array([x0, y0, 0]), np.array([x1, y0, 0]),
                 stroke_color=FAINT, stroke_width=1.2),
            Line(np.array([x0, y0, 0]), np.array([x0, y1, 0]),
                 stroke_color=FAINT, stroke_width=1.2),
        )

        def to_xy(tt, ff):
            t_lo, t_hi, f_lo, f_hi_ = self._ctrl_scale
            sx = x0 + (np.asarray(tt) - t_lo) / (t_hi - t_lo) * (x1 - x0)
            sy = y0 + (np.asarray(ff) - f_lo) / (f_hi_ - f_lo) * (y1 - y0)
            return np.stack([sx, sy], axis=1)

        self._ctrl_to_xy = to_xy

        trac_line = _polyline(to_xy(t, u_trac[:, 0]), TRAC_COLOR, 2.0)
        brk_line = _polyline(to_xy(t, u_brk[:, 0]), BRK_COLOR, 2.0)

        title = label("control input u(t)", font_size=16, color=INK)
        title.move_to(np.array([x0, y1 + 0.26, 0.0])).align_to(
            np.array([x0, 0, 0]), LEFT)
        legend = VGroup(
            label("u_trac  V0", font_size=12, color=TRAC_COLOR),
            label("u_brk  all", font_size=12, color=BRK_COLOR),
        ).arrange(RIGHT, buff=0.35)
        legend.move_to(np.array([x1, y1 + 0.26, 0.0])).align_to(
            np.array([x1, 0, 0]), RIGHT)

        y_hi = label(f"{f_hi / 1e3:.0f} kN", font_size=11, color=FAINT)
        y_hi.next_to(np.array([x0, y1, 0]), LEFT, buff=0.08)
        y_lo = label("0", font_size=11, color=FAINT)
        y_lo.next_to(np.array([x0, y0, 0]), LEFT, buff=0.08)
        t_lo_l = label("0 s", font_size=11, color=FAINT)
        t_lo_l.next_to(np.array([x0, y0, 0]), DOWN, buff=0.1)
        t_hi_l = label(f"{t[-1]:.0f} s", font_size=11, color=FAINT)
        t_hi_l.next_to(np.array([x1, y0, 0]), DOWN, buff=0.1)

        self.playhead = Line(np.array([x0, y0, 0]), np.array([x0, y1, 0]),
                             stroke_color=INK, stroke_width=1.2, stroke_opacity=0.55)
        self.trac_dot = Dot(radius=0.045, color=TRAC_COLOR)
        self.brk_dot = Dot(radius=0.045, color=BRK_COLOR)

        panel = VGroup(frame, trac_line, brk_line, title, legend,
                       y_hi, y_lo, t_lo_l, t_hi_l,
                       self.playhead, self.trac_dot, self.brk_dot)
        self.add(panel)
        return panel

    def _update_control(self, now: float, ut: float, ub: float) -> None:
        x0, x1, y0, y1 = self._ctrl_box
        p_t = self._ctrl_to_xy(np.array([now]), np.array([ut]))[0]
        p_b = self._ctrl_to_xy(np.array([now]), np.array([ub]))[0]
        self.playhead.put_start_and_end_on(
            np.array([p_t[0], y0, 0.0]), np.array([p_t[0], y1, 0.0])
        )
        self.trac_dot.move_to(np.array([p_t[0], p_t[1], 0.0]))
        self.brk_dot.move_to(np.array([p_b[0], p_b[1], 0.0]))

    # ------------------------------------------------------------------ #
    # legend                                                             #
    # ------------------------------------------------------------------ #
    def _legend(self) -> None:
        ramp = _diverging()
        bar = VGroup()
        w = 0.055
        for k, col in enumerate(ramp):
            bar.add(Rectangle(width=w, height=0.11, stroke_width=0,
                              fill_color=col, fill_opacity=1.0)
                    .shift(k * w * RIGHT))
        bar.move_to(np.array([-4.6, 0.62, 0.0]))
        lo = label("buff", font_size=11, color=DIM).next_to(bar, LEFT, buff=0.1)
        hi = label("draft", font_size=11, color=DIM).next_to(bar, RIGHT, buff=0.1)
        cap = label("coupler force colour", font_size=11, color=DIM)
        cap.next_to(bar, DOWN, buff=0.06)
        self.add(bar, lo, hi, cap)
