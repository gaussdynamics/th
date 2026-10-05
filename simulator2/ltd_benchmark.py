"""The international LTD simulator benchmark, as a runnable scenario set.

Spiryagin, Wu & Cole, "International benchmarking of longitudinal train
dynamics simulators: benchmarking questions", *VSD* 55(4):450-463, 2017, with
the corrigendum in *VSD* 56(2):341, 2018 (Train 4's middle locomotive is
type 2). Section and table numbers below are the paper's. The participants'
results, to compare against, are in Wu, Spiryagin, Cole et al., *VSD*
56(3):343-365, 2018.

Everything the paper specifies is encoded here. The tables it prints are
transcribed from the PDF by ``scripts/extract_ltd_benchmark_tables.py`` into
``data/ltd_benchmark/``; the rest is short enough to write down directly.

Choices the paper leaves to participants, and the ones made here:

* **Draft gear model** (section 3.1). The stateful friction gear of
  :mod:`simulator2.draft_gear` on the digitized Fig. 3, with the drop-hammer
  transitional stiffness ``k_t = 7e8 N/m`` -- every benchmark vehicle is
  heavy enough for that at ``dt = 0.01 s``.
* **Notch rate limit** (section 3.2). A first-order lag on the notch,
  ``tau_notch`` (default 5 s, the simulator's traction lag). While the lag
  carries the notch between integers the force blends the neighbouring notch
  curves (:func:`simulator2.torch_rhs._notch_force`).
* **Dynamic brake law.** Table 5 is internally inconsistent as printed; it is
  read against Figs 4b and 5b. See :func:`simulator2.torch_rhs._db_at_notch`.
* **Starting resistance** (section 3.3) is not modelled, as the paper allows.
* **Initial state.** The questions paper does not give one; the results
  paper (Wu et al. 2018, section 2) says all trains start at the 3 km
  location and compares 3-50 km only. The train starts at rest, every
  connection centred in its slack, with the lead locomotive's front at
  ``START_M`` and the consist on the track behind it.
* **Gradient table** (Appendix 1) is read as steps: each gradient holds from
  its location to the next. Integrated from 105 m that gives the Fig. 6a
  profile (valley 93.6 m at 4.1 km, summit 145.1 m at 24.7 km, 113.3 m at
  50 km); linear interpolation between the points does not (117.8 m at 50 km).
* **Track before 0 m** is taken as level; beyond 50 km it keeps the last
  gradient and curvature, as section 3.4 says.

Air brakes are excluded by the benchmark (section 2), so ``F_brk_max = 0``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .constants import DEFAULT_V_BRAKE_EPS, DEFAULT_V_EPS, GRAVITY_MPS2
from .draft_gear import load_table as load_draft_gear_table
from .route import _AREMA_COEFF
from .torch_rhs import (
    D_EDGE_STATIC,
    D_LOCO_PARAM,
    D_NODE_STATIC,
    EdgeStatic,
    NodeStatic,
    TorchScenarioBatch,
)

DATA = Path(__file__).parent / "data" / "ltd_benchmark"

#: Simulated time and required output step (section 3.6).
T_END_S = 5000.0
OUTPUT_DT_S = 0.01

#: Lead locomotive's front at t = 0 (results paper, section 2).
START_M = 3000.0

#: Coupler slack: 10 mm total (section 3.1). Drawbars have none.
COUPLER_SLACK_HALF_M = 0.005
#: Transitional stiffness of the friction gear, from the drop-hammer plot.
K_TRANSITION_N_PER_M = 7.0e8

#: Remote locomotives follow the lead's notch after this delay (section 3.5).
DP_DELAY_S = 3.0
#: Frontal resistance factor of the leading locomotive (Eq. 1).
Q_LEAD = 3.2
#: Curving resistance per tonne at radius R is 6116/R N (Eq. 2).
CURVING_N_PER_T_M = 6116.0
#: Transition, arc and transition lengths of every curve (section 3.4).
CURVE_TRANSITION_M = 50.0
CURVE_ARC_M = 300.0

#: Curve start locations [m] and signed radii [m] (Fig. 6b).
CURVES: list[tuple[float, float]] = [
    (4000.0, 1000.0), (8000.0, 800.0), (12000.0, 600.0), (16000.0, 400.0),
    (20000.0, 300.0), (24000.0, 200.0), (28000.0, -200.0), (32000.0, -300.0),
    (36000.0, -400.0), (40000.0, -600.0), (44000.0, -800.0), (48000.0, -1000.0),
]

#: Track extent of the built route [m]. Long enough for the lead to stay
#: inside it for 5000 s at line speed, and for a 2.7 km train's tail at 0 s.
ROUTE_START_M = -5000.0
ROUTE_END_M = 200_000.0

#: A gradient step is a 1 cm ramp: the route field is piecewise linear.
_STEP_M = 0.01


@dataclass(frozen=True)
class VehicleType:
    """Table 3, plus Table 6 for locomotives."""

    name: str
    mass_t: float
    axle_load_t: float
    length_m: float
    loco: tuple[float, float, float, float, float, float] | None = None
    """Te_max N, P_max W, DB_max N, P_db_max W, v_db_min m/s, v_db_max m/s."""

    @property
    def is_loco(self) -> bool:
        return self.loco is not None


VEHICLE_TYPES = {
    "loco1": VehicleType("loco1", 134.0, 22.33, 22.95,
                         (600e3, 3207e3, 325e3, 3600e3, 1.39, 11.94)),
    "loco2": VehicleType("loco2", 195.0, 32.50, 24.35,
                         (850e3, 3207e3, 472e3, 5500e3, 1.39, 11.94)),
    "wagon1": VehicleType("wagon1", 128.0, 32.00, 15.00),
    "wagon2": VehicleType("wagon2", 160.0, 40.00, 11.00),
}

#: Table 2 as corrected by the corrigendum: (type, count) blocks, front first.
TRAINS: dict[int, list[tuple[str, int]]] = {
    1: [("loco1", 2), ("wagon1", 50)],
    2: [("loco1", 2), ("wagon1", 50), ("loco1", 2), ("wagon1", 50)],
    3: [("loco2", 3), ("wagon2", 240)],
    4: [("loco2", 2), ("wagon2", 120), ("loco2", 1), ("wagon2", 120)],
}

#: Table 7. Position n is the connection at the rear of vehicle n (1-based),
#: i.e. edge index n - 1.
REPORT_POSITIONS: dict[int, list[int]] = {
    1: [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 51],
    2: [2, 11, 20, 31, 40, 54, 61, 70, 81, 90, 101],
    3: [3, 24, 49, 72, 99, 122, 147, 170, 195, 218, 242],
    4: [2, 23, 48, 71, 98, 123, 146, 169, 194, 217, 242],
}

#: Repeating tail of the Train 1 and 2 controls (section 3.5): the notch
#: sequence from ``start`` repeats with ``period`` until the end of the run.
CONTROL_CYCLES: dict[int, tuple[float, float]] = {1: (2688.0, 408.0), 2: (2276.0, 118.0)}


# ---------------------------------------------------------------------------
# the consist
# ---------------------------------------------------------------------------

@dataclass
class Consist:
    """One benchmark train, front to back."""

    train: int
    types: list[VehicleType]
    remote: np.ndarray          # bool per vehicle: a DP locomotive
    drawbar: np.ndarray         # bool per connection: drawbar (no slack)

    @property
    def n(self) -> int:
        return len(self.types)

    @property
    def mass_kg(self) -> np.ndarray:
        return np.array([t.mass_t * 1e3 for t in self.types])

    @property
    def length_m(self) -> float:
        return float(sum(t.length_m for t in self.types))


def build_consist(train: int) -> Consist:
    """Vehicles, DP flags and connection types for one train.

    Wagons run in pairs, each pair joined by a drawbar; pairs, locomotives
    and locomotive-wagon joints use couplers (section 3.1). The first block
    of locomotives is the head end; any later locomotive is a remote unit.
    """
    types: list[VehicleType] = []
    remote: list[bool] = []
    drawbar: list[bool] = []
    for b, (name, count) in enumerate(TRAINS[train]):
        vt = VEHICLE_TYPES[name]
        if not vt.is_loco and count % 2:
            raise ValueError(f"train {train}: odd wagon block cannot form pairs")
        for k in range(count):
            if types:
                # The second wagon of a pair is drawn by a drawbar.
                drawbar.append(not vt.is_loco and k % 2 == 1)
            types.append(vt)
            remote.append(vt.is_loco and b > 0)
    return Consist(train, types, np.array(remote), np.array(drawbar))


def davis_coefficients(consist: Consist) -> np.ndarray:
    """Eq. 1 in per-vehicle ``A + B v + C v^2`` form, ``v`` in m/s, ``(N, 3)``.

    ``F = Q m (2.943 + 89.2/m_a + 0.0306 V + 0.122 V^2 / m)`` with ``m``,
    ``m_a`` in tonnes and ``V`` in km/h converts exactly:
    ``A = Q m (2.943 + 89.2/m_a)``, ``B = 3.6 * 0.0306 Q m``,
    ``C = 3.6^2 * 0.122 Q``.
    """
    out = np.zeros((consist.n, 3))
    for i, vt in enumerate(consist.types):
        q = Q_LEAD if i == 0 else 1.0
        out[i, 0] = q * vt.mass_t * (2.943 + 89.2 / vt.axle_load_t)
        out[i, 1] = 3.6 * 0.0306 * q * vt.mass_t
        out[i, 2] = 3.6 ** 2 * 0.122 * q
    return out


def k_curv_scale_linear() -> float:
    """Scale on the ``"linear"`` curvature law that gives Eq. 2 exactly:
    ``k m g c / R = 6116 m_t / R``."""
    return CURVING_N_PER_T_M / 1000.0 / (GRAVITY_MPS2 * _AREMA_COEFF)


# ---------------------------------------------------------------------------
# track
# ---------------------------------------------------------------------------

def load_gradient() -> np.ndarray:
    """Appendix 1, ``(K, 2)``: location [m], gradient [%] (rise positive)."""
    with open(DATA / "gradient.csv") as fh:
        rows = list(csv.reader(fh))[1:]
    return np.array([[float(a), float(b)] for a, b in rows])


def gradient_pct_at(s: np.ndarray, table: np.ndarray | None = None) -> np.ndarray:
    """Gradient [%] at locations ``s``: each value holds until the next
    location; level before the first and the last value beyond the end."""
    table = load_gradient() if table is None else table
    idx = np.searchsorted(table[:, 0], np.asarray(s, float), side="right") - 1
    return np.where(idx < 0, 0.0, table[np.clip(idx, 0, None), 1])


def curvature_at(s: np.ndarray) -> np.ndarray:
    """Signed curvature [1/m] at locations ``s``: each curve ramps linearly
    over its entry transition, holds over the arc, and ramps out."""
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    lt, la = CURVE_TRANSITION_M, CURVE_ARC_M
    for s0, r in CURVES:
        out = out + np.interp(s, [s0, s0 + lt, s0 + lt + la, s0 + 2 * lt + la],
                              [0.0, 1.0 / r, 1.0 / r, 0.0], left=0.0, right=0.0)
    return out


def build_route() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(s, sin_theta, kappa)`` on one grid holding every knot exactly."""
    grad = load_gradient()
    knots = {ROUTE_START_M, ROUTE_END_M}
    for s0 in grad[:, 0]:
        knots.update((s0 - _STEP_M, s0))
    for s0, _ in CURVES:
        lt, la = CURVE_TRANSITION_M, CURVE_ARC_M
        knots.update((s0, s0 + lt, s0 + lt + la, s0 + 2 * lt + la))
    s = np.array(sorted(knots))
    theta = np.arctan(gradient_pct_at(s, grad) / 100.0)
    return s, np.sin(theta), curvature_at(s)


def elevation_m(s: np.ndarray, z0: float = 105.0) -> np.ndarray:
    """Elevation integrated from the gradient table (for checking Fig. 6a)."""
    grad = load_gradient()
    s = np.asarray(s, float)
    knots = np.concatenate([[0.0], grad[:, 0]])
    g = np.concatenate([[0.0], grad[:, 1]]) / 100.0
    z_knot = z0 + np.concatenate([[0.0], np.cumsum(g[:-1] * np.diff(knots))])
    idx = np.clip(np.searchsorted(knots, s, side="right") - 1, 0, None)
    return z_knot[idx] + g[idx] * (s - knots[idx])


# ---------------------------------------------------------------------------
# driving controls
# ---------------------------------------------------------------------------

def load_controls(train: int, t_end: float = T_END_S) -> np.ndarray:
    """Appendix 2 for one train, ``(K, 2)``: time [s], notch, held until the
    next row. Trains 1 and 2 are printed only up to their first repeat; the
    cycle is unrolled here to ``t_end`` as section 3.5 instructs."""
    with open(DATA / f"controls_train{train}.csv") as fh:
        rows = np.array([[float(a), float(b)] for a, b in list(csv.reader(fh))[1:]])
    if train in CONTROL_CYCLES:
        start, period = CONTROL_CYCLES[train]
        cycle = rows[(rows[:, 0] >= start) & (rows[:, 0] < start + period)]
        if not np.isclose(rows[-1, 0], start + period) or rows[-1, 1] != cycle[0, 1]:
            raise ValueError(f"train {train}: printed table does not close its cycle")
        out = [rows[rows[:, 0] < start]]
        k = 0
        while start + k * period < t_end:
            seg = cycle.copy()
            seg[:, 0] += k * period
            out.append(seg[seg[:, 0] < t_end])
            k += 1
        rows = np.concatenate(out)
    return rows[rows[:, 0] <= t_end]


def notch_at(t: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Commanded notch at times ``t`` (a step function, right-continuous)."""
    idx = np.searchsorted(controls[:, 0], np.asarray(t, float), side="right") - 1
    return np.where(idx < 0, 0.0, controls[np.clip(idx, 0, None), 1])


def command_grid(train: int, consist: Consist, t_end: float = T_END_S,
                 step_s: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Per-vehicle notch commands ``(t, u)`` with ``u`` of shape ``(T, N)``.

    Each notch change becomes a ``step_s`` ramp on the grid, so the
    piecewise-linear command interpolation in the RHS reproduces the step.
    Remote locomotives get the lead's command ``DP_DELAY_S`` later; wagons
    get zero.
    """
    ctl = load_controls(train, t_end)
    changes = ctl[1:, 0]
    breaks = np.concatenate([changes, changes + DP_DELAY_S]) if consist.remote.any() else changes
    breaks = breaks[(breaks > 0) & (breaks < t_end)]
    t = np.unique(np.concatenate([[0.0, t_end], breaks - step_s, breaks]))
    lead = notch_at(t, ctl)
    delayed = notch_at(t - DP_DELAY_S, ctl)
    is_loco = np.array([vt.is_loco for vt in consist.types])
    u = np.where(consist.remote[None, :], delayed[:, None], lead[:, None])
    return t, u * is_loco[None, :]


# ---------------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------------

def initial_positions(consist: Consist, start_m: float = START_M) -> np.ndarray:
    """Vehicle centres [m], lead front at ``start_m``, every connection at
    rest length (centred in its slack)."""
    lengths = np.array([vt.length_m for vt in consist.types])
    return start_m - (np.cumsum(lengths) - 0.5 * lengths)


def build_batch(
    trains: tuple[int, ...] = (1, 2, 3, 4),
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    tau_notch_s: float = 5.0,
    k_transition: float = K_TRANSITION_N_PER_M,
    t_end: float = T_END_S,
    start_m: float = START_M,
) -> tuple[TorchScenarioBatch, list[Consist]]:
    """One padded batch holding the requested benchmark trains."""
    s, sin_theta, kappa = build_route()
    consists = [build_consist(k) for k in trains]
    records = []
    for train, c in zip(trains, consists):
        n = c.n
        node = np.zeros((n, D_NODE_STATIC))
        node[:, NodeStatic.MASS_KG] = c.mass_kg
        node[:, NodeStatic.DAVIS_A:NodeStatic.DAVIS_C + 1] = davis_coefficients(c)
        node[:, NodeStatic.CAN_TRACTION] = [float(vt.is_loco) for vt in c.types]
        # F_TRAC_MAX_N and F_BRK_MAX_N stay 0: traction comes from the notch
        # law, and air brakes are outside the benchmark.

        lengths = np.array([vt.length_m for vt in c.types])
        edge = np.zeros((n - 1, D_EDGE_STATIC))
        edge[:, EdgeStatic.L0_M] = 0.5 * (lengths[:-1] + lengths[1:])
        edge[:, EdgeStatic.SLACK_HALF_M] = np.where(c.drawbar, 0.0, COUPLER_SLACK_HALF_M)

        t_cmd, u = command_grid(train, c, t_end)
        y0 = np.zeros(4 * n)
        y0[:n] = initial_positions(c, start_m)
        records.append({
            "N": n, "t": t_cmd, "y0": y0,
            "node_static": node, "edge_static": edge,
            "u_trac": u, "u_brk": np.zeros_like(u),
            "route_s": s, "route_sin_theta": sin_theta, "route_kappa": kappa,
            "tau_brk_s": 1.0, "tau_trac_s": tau_notch_s, "p_max_w": 0.0,
            "k_curv_scale": k_curv_scale_linear(),
            "v_eps": DEFAULT_V_EPS, "v_brake_eps": DEFAULT_V_BRAKE_EPS,
            "brake_opposes_motion": True, "curvature_model": "linear",
        })
    base = TorchScenarioBatch._from_records(records, device=device, dtype=dtype)

    f_kw = {"dtype": dtype, "device": torch.device(device)}
    b, n_max = base.node_mask.shape
    loco = torch.zeros(b, n_max, D_LOCO_PARAM, **f_kw)
    for i, c in enumerate(consists):
        for j, vt in enumerate(c.types):
            if vt.is_loco:
                loco[i, j] = torch.tensor(vt.loco, **f_kw)

    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    # The draft gear carries one force state per connection after the four
    # node blocks; every gear starts unloaded.
    fields["y0"] = torch.cat([base.y0, torch.zeros(b, n_max - 1, **f_kw)], dim=1)
    fields.update(
        coupler_model="draft_gear",
        draft_gear_table=torch.as_tensor(load_draft_gear_table(), **f_kw),
        draft_gear_k_transition=k_transition,
        traction_model="notch",
        loco_table=loco,
    )
    return TorchScenarioBatch(**fields), consists


# ---------------------------------------------------------------------------
# running it
# ---------------------------------------------------------------------------

def run(
    batch: TorchScenarioBatch,
    consists: list[Consist],
    *,
    t_end: float = T_END_S,
    dt: float = 0.01,
    out_dt: float = OUTPUT_DT_S,
    chunk_s: float = 50.0,
    progress=None,
) -> list[dict[str, np.ndarray]]:
    """Integrate the batch and reduce it to the section 3.6 outputs.

    The full state at 10 ms over 5000 s is ~5 GB per long train, so the run
    is integrated in ``chunk_s`` pieces and each piece is reduced on the
    device before the next starts. One dict per train:

    ``t_s``                 output times
    ``lead_front_m``        location of the lead locomotive's front (output 1)
    ``lead_speed_mps``      its speed (output 2)
    ``force_n``             connection force at the Table 7 positions,
                            tension positive (output 3), ``(T, P)``
    ``gear_defl_m``         per-gear deflection there: travel beyond the slack
                            shared by the two gears in series, draft positive
                            (output 4), ``(T, P)``
    ``max_tension_n``,
    ``max_compression_n``   per connection over the run (output 5)
    ``positions``           the Table 7 positions (1-based vehicle numbers)

    Extremes are taken over the output samples, as the participants' were.
    """
    from .draft_gear import N_GEARS, draft_gear_stateful_torch
    from .torch_rhs import rollout_rk4

    n = batch.n_vehicles
    dev, dt_ = batch.device, batch.dtype
    es = batch.edge_static
    l0 = es[..., EdgeStatic.L0_M]
    slack = es[..., EdgeStatic.SLACK_HALF_M]
    n_out = int(round(t_end / out_dt)) + 1
    per_chunk = max(1, int(round(chunk_s / out_dt)))

    outs = []
    for c in consists:
        pos = REPORT_POSITIONS[c.train]
        outs.append({
            "t_s": np.arange(n_out) * out_dt,
            "lead_front_m": np.empty(n_out),
            "lead_speed_mps": np.empty(n_out),
            "force_n": np.empty((n_out, len(pos)), np.float32),
            "gear_defl_m": np.empty((n_out, len(pos)), np.float32),
            "max_tension_n": np.full(c.n - 1, -np.inf),
            "max_compression_n": np.full(c.n - 1, np.inf),
            "positions": np.array(pos),
        })
    report_idx = [torch.as_tensor(o["positions"] - 1, device=dev) for o in outs]
    half_lead = torch.tensor([c.types[0].length_m / 2 for c in consists], dtype=dt_, device=dev)

    y = batch.y0
    i0 = 0
    while i0 < n_out - 1:
        i1 = min(i0 + per_chunk, n_out - 1)
        t_grid = torch.arange(i0, i1 + 1, dtype=torch.float64, device=dev) * out_dt
        _, ys = rollout_rk4(batch, y, t_grid.to(dt_), dt=dt)
        if i0 > 0:
            ys = ys[:, 1:]          # the first sample is the last chunk's end
        lo = i0 if i0 == 0 else i0 + 1

        x, v = ys[..., :n], ys[..., n:2 * n]
        delta = x[..., :-1] - x[..., 1:] - l0[:, None]
        force, _ = draft_gear_stateful_torch(
            delta, v[..., :-1] - v[..., 1:], ys[..., 4 * n:5 * n - 1],
            slack[:, None], batch.draft_gear_table,
            k_transition=batch.draft_gear_k_transition, k_lock=batch.draft_gear_k_lock,
        )
        s = slack[:, None]
        travel = torch.where(delta > s, delta - s,
                             torch.where(delta < -s, delta + s, torch.zeros_like(delta)))
        f_hi = force.amax(dim=1).cpu().numpy()
        f_lo = force.amin(dim=1).cpu().numpy()
        lead_x = (x[..., 0] + half_lead[:, None]).cpu().numpy()
        lead_v = v[..., 0].cpu().numpy()

        for b, (c, o) in enumerate(zip(consists, outs)):
            hi = lo + ys.shape[1]
            o["lead_front_m"][lo:hi] = lead_x[b]
            o["lead_speed_mps"][lo:hi] = lead_v[b]
            o["force_n"][lo:hi] = force[b][:, report_idx[b]].cpu().numpy()
            o["gear_defl_m"][lo:hi] = (travel[b][:, report_idx[b]] / N_GEARS).cpu().numpy()
            m = c.n - 1
            o["max_tension_n"] = np.maximum(o["max_tension_n"], f_hi[b, :m])
            o["max_compression_n"] = np.minimum(o["max_compression_n"], f_lo[b, :m])

        if not torch.isfinite(ys[:, -1]).all():
            raise FloatingPointError(f"state went non-finite by t = {i1 * out_dt:g} s")
        y = ys[:, -1]
        i0 = i1
        if progress is not None:
            progress(i1 * out_dt)
    return outs


__all__ = [
    "CURVES", "REPORT_POSITIONS", "START_M", "TRAINS", "VEHICLE_TYPES",
    "build_batch", "build_consist", "build_route", "command_grid",
    "curvature_at", "davis_coefficients", "elevation_m", "gradient_pct_at",
    "initial_positions", "k_curv_scale_linear", "load_controls", "notch_at",
    "run",
]
