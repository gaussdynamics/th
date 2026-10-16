"""Physics configuration for corpus generation: which model of each force.

The corpora so far (data/v1, data/v2) were generated on the Chapter 4
physics, which ``SIMULATOR_VALIDATION_NOTE.md`` reviews as findings F1-F9.
The fixes, and the configuration the LTD benchmark validated (friction draft
gear, notch traction with dynamic brake), are switched on together by
:data:`V3`. :data:`LEGACY` reproduces the existing corpora exactly, so the
old data, ``scripts/validate_dataset.py`` and the surrogate keep working.

Each field names one model choice and the finding it answers. A field is
applied at the stage it belongs to:

* **consist** (when a scenario is built): ``resistance``, ``rotating_mass``,
  ``locomotives``, the air brake's capacities;
* **commands** (when commands are materialized): ``traction_model`` (notches),
  ``air_brake``, ``speed_planner``;
* **batch** (:func:`configure_batch`, after ``from_scenarios``):
  ``grade_model``, ``coupler_model``, ``traction_model``, ``standstill_model``,
  the per-vehicle brake time constants.

The configuration is written into every scenario file (``physics_json``) and
into ``build_config.json``; a file without it is legacy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any

__all__ = ["PhysicsConfig", "LEGACY", "V3", "PRESETS", "from_json", "physics_of"]


@dataclass(frozen=True)
class PhysicsConfig:
    """One model choice per force. See the module docstring for the stages."""

    #: Preset name, for files and logs.
    name: str = "legacy"
    #: F7. ``"double_sine"``: m g sin(sin theta), the Chapter 4 code (the
    #: route field already holds sin theta). ``"exact"``: m g sin theta.
    grade_model: str = "double_sine"
    #: F4. ``"catalog"``: fixed Davis A/B/C per car type. ``"modified_davis"``:
    #: per vehicle from mass and axles (Handbook Table 13.3).
    resistance: str = "catalog"
    #: F9. Add the rotating-mass allowance to each vehicle's inertia.
    rotating_mass: bool = False
    #: F5. ``"catalog"``: the illustrative SD70/GP38 entries. ``"sourced"``:
    #: manufacturer data (Spiryagin et al. 2017, Ch. 2).
    locomotives: str = "catalog"
    #: F5. ``"force_lag"`` (force command, one power cap) or ``"notch"``
    #: (8 notches and a dynamic brake from per-locomotive tables).
    traction_model: str = "force_lag"
    #: F3. ``"kelvin_voigt"`` or ``"draft_gear"`` (friction gear, benchmark curve).
    coupler_model: str = "kelvin_voigt"
    #: F1/F2. Per-vehicle air brake (:mod:`simulator2.air_brake`): propagation,
    #: pipe lag, valve fill, load-capped capacity.
    air_brake: bool = False
    #: F6. ``"none"`` or ``"breakaway"`` (static holding at standstill).
    standstill_model: str = "none"
    #: F8. Shape commands with a point-mass pre-simulation so runs respect
    #: the route speed limits.
    speed_planner: bool = False
    #: F5. ``"manifest"``: the locomotive count the consist was generated
    #: with. ``"ruling_grade"``: re-sized per scenario so the sourced units'
    #: continuous tractive effort holds the train on the route's ruling grade
    #: (:mod:`simulator2.power_assignment`).
    power_sizing: str = "manifest"

    def __post_init__(self) -> None:
        allowed = {
            "grade_model": ("double_sine", "exact"),
            "resistance": ("catalog", "modified_davis"),
            "locomotives": ("catalog", "sourced"),
            "traction_model": ("force_lag", "notch"),
            "coupler_model": ("kelvin_voigt", "draft_gear"),
            "standstill_model": ("none", "breakaway"),
            "power_sizing": ("manifest", "ruling_grade"),
        }
        for name, ok in allowed.items():
            if getattr(self, name) not in ok:
                raise ValueError(f"{name} must be one of {ok}, got {getattr(self, name)!r}")
        if self.traction_model == "notch" and self.locomotives != "sourced":
            raise ValueError("traction_model 'notch' needs locomotives 'sourced' "
                             "(the catalog has no notch tables)")
        if self.power_sizing == "ruling_grade" and self.locomotives != "sourced":
            raise ValueError("power_sizing 'ruling_grade' needs locomotives 'sourced' "
                             "(the catalog has no continuous ratings)")

    @property
    def default_dt_s(self) -> float:
        """Integration step for corpus generation. The friction draft gear's
        end stop (``k_lock`` 5e8 N/m) on a 26 t empty car is a ~200 rad/s mode,
        beyond explicit RK4's stability limit at 0.02 s (2.8 / 0.02 = 140):
        a 126-car v3 train diverged there and ran clean at 0.01 s, the step the
        benchmark validation used (SIMULATOR_VALIDATION_NOTE.md, 2026-10-07)."""
        return 0.01 if self.coupler_model == "draft_gear" else 0.02

    @property
    def is_legacy(self) -> bool:
        return self == LEGACY

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


LEGACY = PhysicsConfig()

V3 = PhysicsConfig(
    name="v3",
    grade_model="exact",
    resistance="modified_davis",
    rotating_mass=True,
    locomotives="sourced",
    traction_model="notch",
    coupler_model="draft_gear",
    air_brake=True,
    standstill_model="breakaway",
    speed_planner=True,
    power_sizing="ruling_grade",
)

PRESETS = {"legacy": LEGACY, "v3": V3}


def from_json(text: str) -> PhysicsConfig:
    """Inverse of :meth:`PhysicsConfig.to_json`; unknown keys are an error."""
    data = json.loads(text)
    names = {f.name for f in fields(PhysicsConfig)}
    extra = set(data) - names
    if extra:
        raise ValueError(f"unknown physics fields {sorted(extra)}")
    return PhysicsConfig(**data)


def physics_of(npz: Any) -> PhysicsConfig:
    """The configuration a scenario file was generated with (legacy if absent)."""
    files = getattr(npz, "files", npz)
    if "physics_json" not in files:
        return LEGACY
    return from_json(str(npz["physics_json"]))


#: AAR knuckle strength: 1.8 MN (400,000 lb) yield, 2.9 MN (650,000 lb)
#: ultimate in tension (Spiryagin et al. 2017, p. 213, citing the AAR
#: standard). The model has no coupler failure; a run above the ultimate is a
#: train that would have parted, and is flagged in the index.
KNUCKLE_YIELD_N = 1.8e6
KNUCKLE_ULTIMATE_N = 2.9e6


# -- consist stage ------------------------------------------------------------------

#: Coupler free slack under the friction draft gear: the LTD benchmark's 10 mm
#: total (Spiryagin, Wu & Cole 2017, section 2), the configuration the gear was
#: validated in. ``SLACK_HALF_M`` holds half of it.
V3_COUPLER_SLACK_HALF_M = 0.005


@dataclass
class ConsistPhysics:
    """A consist prepared for one physics configuration.

    ``vehicles`` and ``couplers`` feed the scenario as before; the arrays are
    the per-vehicle extras the batch stage needs (``None`` under legacy).
    """

    vehicles: list
    couplers: list
    car_types: list
    is_loco: Any = None            # (N,) bool
    loco_table: Any = None         # (N, 6) for traction_model "notch"
    inertia_factor: Any = None     # (N,)


def configure_consist(train: Any, library: Any, physics: PhysicsConfig,
                      air_brake_params: Any = None) -> ConsistPhysics:
    """Build a train's consist under ``physics`` (the consist-stage fields).

    Legacy is exactly :func:`simulator2.consist.build_consist`.
    """
    import numpy as np

    from .catalog import CarCategory
    from .consist import build_consist, build_consist_from_types, resolve_car_types
    from .vehicle_physics import (
        apply_modified_davis, axles_of_spec, inertia_factors, sourced_car_type,
    )

    resolved = resolve_car_types(train, library)
    if physics.is_legacy:
        vehicles, couplers = build_consist(train, library)
        return ConsistPhysics(vehicles, couplers, resolved)

    specs = [None] * len(resolved)
    if physics.locomotives == "sourced":
        pairs = [sourced_car_type(ct) for ct in resolved]
        resolved, specs = [p[0] for p in pairs], [p[1] for p in pairs]
    vehicles, couplers = build_consist_from_types(resolved, library)
    is_loco = np.array([ct.category == CarCategory.LOCOMOTIVE for ct in resolved])

    if physics.resistance == "modified_davis":
        vehicles = apply_modified_davis(
            vehicles, resolved, [axles_of_spec(ct, s) for ct, s in zip(resolved, specs)])
    if physics.coupler_model == "draft_gear":
        for c in couplers:
            c.slack_half_m = V3_COUPLER_SLACK_HALF_M
    if physics.air_brake:
        from dataclasses import replace as _replace

        from .air_brake import AirBrakeParams, brake_capacity_n
        p = air_brake_params or AirBrakeParams()
        f_max = brake_capacity_n(np.array([v.mass_kg for v in vehicles]), is_loco, p)
        vehicles = [_replace(v, F_brk_max_N=float(f)) for v, f in zip(vehicles, f_max)]

    loco_table = None
    if physics.traction_model == "notch":
        loco_table = np.zeros((len(vehicles), 6))
        for i, s in enumerate(specs):
            if s is not None:
                loco_table[i] = s.table
    inertia = (np.array(inertia_factors(resolved)) if physics.rotating_mass else None)
    return ConsistPhysics(vehicles, couplers, resolved, is_loco, loco_table, inertia)


# -- command stage ------------------------------------------------------------------

#: Regimes whose brake curve is the locomotives' dynamic brake under v3. The
#: legacy corpus drove the *air* brake there as a stand-in, for want of a
#: dynamic brake (SIMULATOR_VALIDATION_NOTE.md, 2026-10-04).
DYNAMIC_BRAKE_REGIMES = ("dynamic_brake_descent",)


@dataclass
class Commands:
    """Materialized commands, plus the per-vehicle brake time constants the
    air brake sets (``None`` otherwise)."""

    u_trac: Any                 # (T, N): newtons (force_lag) or signed notch (notch)
    u_brk: Any                  # (T, N): newtons
    tau_apply: Any = None       # (N,)
    tau_release: Any = None     # (N,)


def to_notches(frac: Any) -> Any:
    """A 0-1 command fraction as the nearest of the 8 notches (0-8)."""
    import numpy as np
    return np.rint(8.0 * np.clip(frac, 0.0, 1.0))


#: Fastest notch-up of the LTD benchmark's driving (Spiryagin, Wu & Cole 2017,
#: Appendix 2: one notch per 8-40 s on all four trains). Under v3 a notch's
#: magnitude (traction or dynamic brake) rises by at most one per interval;
#: it may fall at once. Without it, a regime ramping from notch 2 to 8 in
#: seconds after a brake release ran a 142-car train out to 6.6 MN.
NOTCH_UP_INTERVAL_S = 8.0


def rate_limit_notches(u: Any, t: Any, interval_s: float = NOTCH_UP_INTERVAL_S) -> Any:
    """Signed notches ``(T, N)`` with each step up in magnitude at least
    ``interval_s`` after the previous one; reductions pass through."""
    import numpy as np
    u = np.asarray(u, float)
    t = np.asarray(t, float)
    out = np.empty_like(u)
    cur = u[0].copy()
    last_up = np.full(u.shape[1], -np.inf)
    for k in range(u.shape[0]):
        want = u[k]
        flip = np.sign(want) * np.sign(cur) < 0
        cur = np.where(flip, 0.0, cur)                  # traction <-> brake via 0
        smaller = np.abs(want) <= np.abs(cur)
        can_step = (t[k] - last_up) >= interval_s - 1e-9
        step = np.where(want >= 0, 1.0, -1.0)
        up = ~smaller & can_step
        cur = np.where(smaller, want, np.where(up, cur + step, cur))
        last_up = np.where(up, t[k], last_up)
        out[k] = cur
    return out


def materialize_commands(profile: Any, consist: ConsistPhysics, t: Any, train: Any,
                         physics: PhysicsConfig, regime: str | None = None,
                         x0: Any = None, air_brake_params: Any = None,
                         route: Any = None, v0: float | None = None) -> Commands:
    """Commands for one scenario under ``physics`` (the command-stage fields).

    Legacy is exactly :func:`simulator2.dataset.materialize_commands_fast`.
    Under notch traction the traction fractions become notches, and in the
    :data:`DYNAMIC_BRAKE_REGIMES` the brake curve becomes negative (dynamic
    brake) notches on every locomotive instead of an air brake application.
    With the speed planner the fractions are first governed against the route
    limits (:func:`simulator2.speed_planner.plan_commands`; needs ``route``,
    ``x0`` and ``v0``). With the air brake, the driver's brake fraction goes
    through :func:`simulator2.air_brake.vehicle_commands` (``x0``: vehicle
    positions).
    """
    import numpy as np

    from .dataset import command_fractions, materialize_commands_fast

    vehicles = consist.vehicles
    if physics.is_legacy:
        return Commands(*materialize_commands_fast(profile, vehicles, t, train=train))

    trac_frac, brake_frac = command_fractions(profile, vehicles, t, train)
    notch = physics.traction_model == "notch"
    if notch and regime in DYNAMIC_BRAKE_REGIMES:
        db_frac, brake_frac = brake_frac, np.zeros_like(brake_frac)
    else:
        db_frac = np.zeros_like(brake_frac)
    if physics.speed_planner:
        from .air_brake import AirBrakeParams
        from .speed_planner import plan_commands
        if route is None or x0 is None or v0 is None:
            raise ValueError("the speed planner needs route, x0 and v0")
        abp = air_brake_params or AirBrakeParams()
        pipe_m = abp.pipe_length_factor * float(np.ptp(np.asarray(x0, float)))
        plan = plan_commands(np.asarray(t, float), trac_frac, brake_frac, db_frac,
                             consist, np.asarray(x0, float), float(v0), route, physics,
                             recharge_s=abp.recharge_s,
                             release_wait_s=(pipe_m / abp.release_propagation_mps
                                             + 3.0 * abp.tau_release_s))
        trac_frac, brake_frac, db_frac = plan["trac_frac"], plan["brake_frac"], plan["db_frac"]

    if notch:
        n_db = to_notches(db_frac)[:, None] * consist.is_loco[None, :]
        u_trac = rate_limit_notches(np.where(n_db > 0, -n_db, to_notches(trac_frac)), t)
    else:
        f_trac_max = np.array([v.F_trac_max_N for v in vehicles])
        u_trac = trac_frac * f_trac_max[None, :]

    if not physics.air_brake:
        f_brk = np.array([v.F_brk_max_N for v in vehicles])
        return Commands(u_trac, brake_frac[:, None] * f_brk[None, :])
    from .air_brake import AirBrakeParams, vehicle_commands
    if x0 is None:
        raise ValueError("the air brake needs the vehicle positions x0")
    bc = vehicle_commands(np.asarray(t, float), brake_frac, np.asarray(x0, float),
                          np.array([v.mass_kg for v in vehicles]), consist.is_loco,
                          air_brake_params or AirBrakeParams())
    return Commands(u_trac, bc.u_brk, bc.tau_apply, bc.tau_release)


# -- batch stage --------------------------------------------------------------------

def configure_batch(batch: Any, consists: list[ConsistPhysics], commands: list[Commands],
                    physics: PhysicsConfig) -> Any:
    """Set the batch-stage fields of ``physics`` on a batch built by
    ``TorchScenarioBatch.from_scenarios`` (legacy: returned unchanged).

    Per-vehicle arrays are padded to the batch width (1 where the RHS divides
    by them, 0 in the locomotive table). The actuator states in ``y0`` are
    reset to the commands at ``t = 0``, since the scenario settled them on the
    legacy force commands, and under the draft gear one gear state per
    connection is appended (:func:`draft_gear_initial_state`).
    """
    import torch

    if physics.is_legacy:
        return batch
    from .torch_rhs import TorchScenarioBatch

    b, n_max = batch.node_mask.shape
    f_kw = {"dtype": batch.y0.dtype, "device": batch.y0.device}

    def padded(rows, fill: float, width: int | None = None):
        shape = (b, n_max) if width is None else (b, n_max, width)
        out = torch.full(shape, fill, **f_kw)
        for i, r in enumerate(rows):
            r = torch.as_tensor(r, **f_kw)
            out[i, : r.shape[0]] = r
        return out

    fields = {f: getattr(batch, f) for f in batch.__dataclass_fields__}
    y0 = batch.y0.clone()
    for i, c in enumerate(commands):
        n = c.u_trac.shape[1]
        y0[i, 2 * n_max: 2 * n_max + n] = torch.as_tensor(c.u_brk[0], **f_kw)
        y0[i, 3 * n_max: 3 * n_max + n] = torch.as_tensor(c.u_trac[0], **f_kw)
    fields["grade_model"] = physics.grade_model

    if physics.rotating_mass:
        fields["inertia_factor"] = padded([c.inertia_factor for c in consists], 1.0)
    if physics.traction_model == "notch":
        fields["traction_model"] = "notch"
        fields["loco_table"] = padded([c.loco_table for c in consists], 0.0, width=6)
    if physics.air_brake:
        fields["tau_brk_node"] = padded([c.tau_apply for c in commands], 1.0)
        fields["tau_brk_release_node"] = padded([c.tau_release for c in commands], 1.0)
    if physics.standstill_model != "none":
        fields["standstill_model"] = physics.standstill_model
    if physics.coupler_model == "draft_gear":
        import numpy as np

        from .draft_gear import load_table
        table = load_table()
        fields.update(coupler_model="draft_gear",
                      draft_gear_table=torch.as_tensor(np.asarray(table), **f_kw))
        fields["y0"] = torch.cat([y0, torch.zeros(b, max(n_max - 1, 0), **f_kw)], dim=1)
        fields["y0"] = equilibrium_start(TorchScenarioBatch(**fields), consists, table)
    else:
        fields["y0"] = y0
    return TorchScenarioBatch(**fields)


#: Couplers whose equilibrium tension is below this keep their randomized
#: position in the slack [N].
_EQ_TENSION_MIN_N = 5.0e3


def equilibrium_start(batch: Any, consists: list[ConsistPhysics], table: Any) -> Any:
    """``y0`` with the draft gears in quasi-static equilibrium at ``t = 0``.

    A scenario starts with its actuators settled at the first command, which
    can be several locomotives at a high notch. Starting the gears unloaded
    then yanks the whole gear travel (~165 mm per connection, ~20 m along a
    126-car train) out within seconds and the tail snatches against the end
    stops: a 9 MN transient in the first v3 smoke test, at every step size.
    A train running under those forces would already be stretched, so:

    1. with every coupler unloaded, the RHS at ``t = 0`` gives each vehicle's
       external force ``F_i = m_i (1 + J_i) dv_i``;
    2. the train accelerates as one, ``a = sum F / sum m (1 + J)``;
    3. coupler ``j`` carries ``T_j = sum_{i <= j} (F_i - m_i (1 + J_i) a)``;
    4. each gear takes the deflection the loading curve gives for ``|T_j|``,
       its state is ``T_j``, and the vehicles are placed accordingly (the lead
       stays put). Couplers with ``|T_j|`` below 5 kN keep their randomized
       position in the slack.
    """
    import numpy as np
    import torch

    from .draft_gear import N_GEARS
    from .torch_rhs import NodeStatic, torch_rhs

    b, n_max = batch.node_mask.shape
    y = batch.y0.clone()
    edge = batch.edge_static.detach().cpu().double().numpy()
    x_rand = y[:, :n_max].detach().cpu().double().numpy()
    # unloaded: every connection at zero deflection, every gear at zero force
    x_zero = x_rand.copy()
    for i, c in enumerate(consists):
        n = len(c.vehicles)
        x_zero[i, 1:n] = x_zero[i, 0] - np.cumsum(edge[i, : n - 1, 0])
    y_zero = y.clone()
    y_zero[:, :n_max] = torch.as_tensor(x_zero, dtype=y.dtype, device=y.device)
    with torch.no_grad():
        dv = torch_rhs(0.0, y_zero, batch)[:, n_max:2 * n_max].detach().cpu().double().numpy()
    mass = batch.node_static[..., NodeStatic.MASS_KG].detach().cpu().double().numpy()
    inertia = mass * (1.0 if batch.inertia_factor is None
                      else batch.inertia_factor.detach().cpu().double().numpy())
    f_load = np.maximum.accumulate(np.asarray(table)[:, 1])
    g_tab = np.asarray(table)[:, 0]

    out = y.clone()
    for i, c in enumerate(consists):
        n = len(c.vehicles)
        if n < 2:
            continue
        f = inertia[i, :n] * dv[i, :n]
        a = f.sum() / inertia[i, :n].sum()
        tension = np.cumsum(f - inertia[i, :n] * a)[:-1]
        slack = edge[i, : n - 1, 1]
        gear = np.interp(np.abs(tension), f_load, g_tab)
        delta = np.sign(tension) * (slack + N_GEARS * gear)
        rand = np.clip(x_rand[i, : n - 1] - x_rand[i, 1:n] - edge[i, : n - 1, 0], -slack, slack)
        small = np.abs(tension) < _EQ_TENSION_MIN_N
        delta = np.where(small, rand, delta)
        z = np.where(small, 0.0, tension)
        x = np.empty(n)
        x[0] = x_rand[i, 0]
        x[1:] = x[0] - np.cumsum(edge[i, : n - 1, 0] + delta)
        out[i, :n] = torch.as_tensor(x, dtype=y.dtype, device=y.device)
        out[i, 4 * n_max: 4 * n_max + n - 1] = torch.as_tensor(z, dtype=y.dtype, device=y.device)
    return out


# -- outputs ------------------------------------------------------------------------

def scenario_outputs(physics: PhysicsConfig, state: Any, gear_state: Any,
                     edge_static: Any, consist: ConsistPhysics, commands: Commands,
                     batch: Any) -> tuple[Any, Any, dict]:
    """``(edge_dynamic, power, extras)`` for :func:`dataset.build_scenario_arrays`.

    Legacy: ``(None, None, {})``, i.e. the existing Kelvin-Voigt edges and
    force-lag power. Under the draft gear the coupler force is recomputed from
    the saved states with the RHS's own law, and the gear states are stored.
    Under notch traction the power is the locomotives' traction work rate.
    """
    import numpy as np
    import torch

    if physics.is_legacy:
        return None, None, {}
    f64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    x, v = state[:, :, 0], state[:, :, 1]
    extras: dict = {"physics_json": physics.to_json()}
    edge_dynamic = None
    if physics.coupler_model == "draft_gear":
        from .draft_gear import draft_gear_stateful_torch
        delta = x[:, :-1] - x[:, 1:] - edge_static[None, :, 0]
        ddot = v[:, :-1] - v[:, 1:]
        with torch.no_grad():
            f, _ = draft_gear_stateful_torch(
                f64(delta), f64(ddot), f64(gear_state), f64(edge_static[None, :, 1]),
                f64(batch.draft_gear_table.detach().cpu()),
                k_transition=batch.draft_gear_k_transition,
                k_lock=batch.draft_gear_k_lock)
        edge_dynamic = np.stack([delta, ddot, f.numpy()], axis=-1)
        extras["gear_state"] = gear_state
    power = None
    if physics.traction_model == "notch":
        from .torch_rhs import _notch_force
        lp = f64(consist.loco_table)[None].expand(state.shape[0], -1, -1)
        with torch.no_grad():
            force = _notch_force(f64(state[:, :, 3]), f64(v), lp,
                                 f64(float(batch.v_eps[0, 0]))).numpy()
        power = np.where(state[:, :, 3] > 0, force * v, 0.0).sum(axis=1)
        extras["loco_table"] = consist.loco_table
    if physics.rotating_mass:
        extras["inertia_factor"] = consist.inertia_factor
    if physics.air_brake:
        extras["tau_brk_node"] = commands.tau_apply
        extras["tau_brk_release_node"] = commands.tau_release
    if physics.coupler_model == "draft_gear":
        from .draft_gear import N_GEARS
        extras["draft_gear_end_m"] = float(
            N_GEARS * batch.draft_gear_table[-1, 0].detach().cpu())
    if physics.standstill_model == "breakaway":
        extras["standstill_resistance_npkg"] = float(batch.standstill_resistance_npkg)
        extras["standstill_tau_s"] = float(batch.standstill_tau_s)
    return edge_dynamic, power, extras
