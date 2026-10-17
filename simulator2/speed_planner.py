"""Point-mass speed planner: keep open-loop commands inside the speed limits.

Finding F8: the driving regimes are open loop and route-blind, and the batch
solver needs every command fixed in advance, so 17.8 % of legacy scenarios
exceeded the route limit by more than 5 m/s. This planner runs before the
full simulation. It simulates the train as one mass along its route (grade
taken at every vehicle's own position) under the regime's commands, and
governs them:

* **Allowed speed.** The limit that binds the lead at chainage ``x`` is the
  lowest limit anywhere under the train, ``[x - L, x]`` (the tail must clear a
  restriction too), backed off ahead of each lower limit by a braking curve
  ``v^2 = v_lim^2 + 2 a d``.
* **Governor.** Near the allowed speed, traction is scaled down (at a finite
  rate, as a driver notches off). Above it, the dynamic brake (notch
  traction) comes in proportionally, and beyond that an air brake
  application is latched at the minimum level and released well below the
  limit, never sooner than the recharge time after the last release.
* **Holding at a stop.** At or near standstill, if traction cannot beat
  grade and resistance, the train is held on a full-service application (a
  driver does not let a stopped train roll back; a minimum application let a
  10 kt train on 1.1 % roll back to -4.6 m/s while the pipe filled), released
  once traction wins. With the standstill model (F6) the brake then holds it.
* **No power against the brakes.** While holding, and after any release until
  the release has reached the tail and the cylinders have exhausted
  (``release_wait_s``), traction is capped at a low notch: powering up while
  the tail is still braked snatched a 143-car train to 7.6 MN.

The governed curves replace the regime's (traction scaled, brake and dynamic
brake taken as the larger of the regime's and the planner's), so commands
stay open loop and the regime's character is kept away from the limits. The
planner's model is cruder than the simulator (no slack action, no brake
propagation, no notch lag); margins absorb the difference, and
``speed_limit_stats`` measures what is left. It does take the notch-up rate
limit, since that changes how much traction a cycling governor delivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import DEFAULT_V_BRAKE_EPS, GRAVITY_MPS2
from .physics_config import NotchLimiter, to_notches
from .torch_rhs import STANDSTILL_RESISTANCE_NPKG, STANDSTILL_TAU_S

__all__ = ["PlannerParams", "allowed_speed_field", "plan_commands"]


@dataclass(frozen=True)
class PlannerParams:
    """Governor settings. Driving-policy choices, not physical parameters."""

    #: Start notching off this far below the allowed speed [m/s].
    traction_margin_mps: float = 1.5
    #: Traction scale slew rate [1/s] (full to zero in 5 s).
    traction_slew_per_s: float = 0.2
    #: Dynamic brake: zero this far below the allowed speed, full 1/gain above.
    db_margin_mps: float = 0.5
    db_gain_per_mps: float = 0.5
    #: Air brake latches on this far above the allowed speed, off this far below.
    air_on_mps: float = 1.0
    air_off_mps: float = 3.0
    #: Latched application level: the minimum application, 125 of 350 kPa in
    #: Spiryagin et al. (2017) Fig. 5.67.
    air_level: float = 125.0 / 350.0
    #: Braking curve ahead of lower limits [m/s^2]: the low end of the typical
    #: train decelerations, 0.1-0.6 m/s^2 (Handbook 2nd ed., p. 515).
    decel_mps2: float = 0.15
    #: Below this speed a train that cannot pull away is held on the brake [m/s].
    hold_speed_mps: float = 0.3
    #: Holding application: full service.
    hold_level: float = 1.0
    #: Traction cap while the brakes are on or still releasing: notch 2 of 8.
    release_traction_cap: float = 0.25


def allowed_speed_field(route: Any, train_length_m: float,
                        decel_mps2: float) -> tuple[np.ndarray, np.ndarray]:
    """``(s, v_allow)`` on the route nodes for the *lead* vehicle's chainage."""
    s = np.asarray(route.s_nodes_m, float)
    vlim = np.asarray(route.v_max_nodes, float)
    vlim = np.where(np.isfinite(vlim), vlim, 1e6)
    # lowest limit under the train: running minimum over the trailing window
    out = vlim.copy()
    j0 = 0
    for j in range(s.size):
        while s[j0] < s[j] - train_length_m:
            j0 += 1
        out[j] = vlim[j0:j + 1].min()
    # braking curve: back off ahead of every lower limit
    for j in range(s.size - 2, -1, -1):
        out[j] = min(out[j], np.sqrt(out[j + 1] ** 2 + 2 * decel_mps2 * (s[j + 1] - s[j])))
    return s, out


def _notch_traction(n: np.ndarray, v: float, table: np.ndarray) -> np.ndarray:
    frac = n / 8.0
    return np.minimum(frac * table[:, 0], frac * frac * table[:, 1] / max(abs(v), 0.1))


def _dynamic_brake(n: np.ndarray, v: float, table: np.ndarray) -> np.ndarray:
    """Full-notch dynamic brake curve scaled by n/8 (the planner only needs a
    conservative magnitude, not the benchmark's per-notch plateaus)."""
    db, p_db, v_min, v_max = table[:, 2], table[:, 3], table[:, 4], table[:, 5]
    speed = abs(v)
    full = np.where(speed <= v_min, speed / np.maximum(v_min, 1e-9) * db,
                    np.where(speed <= v_max, db - 10.0 * speed, p_db / max(speed, 0.1)))
    return np.clip(full, 0.0, None) * n / 8.0


def plan_commands(t: np.ndarray, trac_frac: np.ndarray, brake_frac: np.ndarray,
                  db_frac: np.ndarray, consist: Any, x0: np.ndarray, v0: float,
                  route: Any, physics: Any, params: PlannerParams | None = None,
                  recharge_s: float = 90.0, release_wait_s: float = 0.0) -> dict[str, np.ndarray]:
    """Governed ``trac_frac [T, N]``, ``brake_frac [T]``, ``db_frac [T]``, and
    the planner's own trace ``v_plan``, ``v_allow`` for diagnostics."""
    p = params or PlannerParams()
    t = np.asarray(t, float)
    vehicles = consist.vehicles
    mass = np.array([veh.mass_kg for veh in vehicles])
    inertia = mass * (consist.inertia_factor if consist.inertia_factor is not None else 1.0)
    m_eff = inertia.sum()
    davis = np.array([[veh.davis_A, veh.davis_B, veh.davis_C] for veh in vehicles]).sum(axis=0)
    f_brk = np.array([veh.F_brk_max_N for veh in vehicles]).sum()
    x0 = np.asarray(x0, float)
    offsets = x0 - x0[0]                                # <= 0 behind the lead
    length = float(-offsets.min()) if offsets.size else 0.0
    s_nodes, v_allow_nodes = allowed_speed_field(route, length, p.decel_mps2)
    s_route = np.asarray(route.s_nodes_m, float)
    st_route = np.asarray(route.sin_theta_nodes, float)
    exact = getattr(physics, "grade_model", "exact") == "exact"
    notch = getattr(physics, "traction_model", "force_lag") == "notch"
    is_loco = np.asarray(consist.is_loco, bool) if consist.is_loco is not None else \
        np.array([veh.can_traction for veh in vehicles])
    table = consist.loco_table[is_loco] if notch else None
    f_trac_max = np.array([veh.F_trac_max_N for veh in vehicles])

    n_t = t.size
    out_trac = np.array(trac_frac, float, copy=True)
    out_brake = np.array(brake_frac, float, copy=True)
    out_db = np.array(db_frac, float, copy=True)
    v_plan = np.zeros(n_t)
    v_allow = np.zeros(n_t)
    x_lead, v = float(x0[0]), float(v0)
    g, air_on, ready_at, holding = 1.0, False, -np.inf, False
    released_until = -np.inf
    limiter = None
    for k in range(n_t):
        va = float(np.interp(x_lead, s_nodes, v_allow_nodes))
        v_plan[k], v_allow[k] = v, va
        dt = t[k + 1] - t[k] if k + 1 < n_t else 0.0
        # governor
        g_target = 0.0 if v > va - p.traction_margin_mps else 1.0
        g += np.clip(g_target - g, -p.traction_slew_per_s * dt, p.traction_slew_per_s * dt) \
            if k else (g_target - g)
        db = np.clip(p.db_gain_per_mps * (v - (va - p.db_margin_mps)), 0.0, 1.0) if notch else 0.0
        if air_on and v < va - p.air_off_mps:
            air_on, ready_at = False, t[k] + recharge_s
            released_until = t[k] + release_wait_s
        elif not air_on and v > va + p.air_on_mps and t[k] >= ready_at:
            air_on = True
        out_trac[k] *= g
        out_db[k] = max(out_db[k], db)
        # whether to hold is decided on the commanded traction
        tf = out_trac[k][is_loco]
        if notch:
            f_cmd = _notch_traction(np.rint(8 * tf), v, table).sum()
        else:
            f_cmd = (tf * f_trac_max[is_loco]).sum()
        breakaway = getattr(physics, "standstill_model", "none") == "breakaway"
        f_start = STANDSTILL_RESISTANCE_NPKG * mass.sum() if breakaway else 0.0
        sin_th = np.interp(x_lead + offsets, s_route, st_route)
        f_grade = (mass * GRAVITY_MPS2 * (sin_th if exact else np.sin(sin_th))).sum()
        stuck = f_cmd <= f_grade + davis[0] + f_start
        if holding and not stuck:
            holding, ready_at = False, t[k] + recharge_s
            released_until = t[k] + release_wait_s
        elif not holding and stuck and v <= p.hold_speed_mps and t[k] >= ready_at:
            holding = True
        if holding or t[k] < released_until or air_on:
            out_trac[k] = np.minimum(out_trac[k], p.release_traction_cap)
        out_brake[k] = max(out_brake[k], p.air_level if air_on else 0.0,
                           p.hold_level if holding else 0.0)
        # forces under the governed, capped traction
        tf = out_trac[k][is_loco]
        if notch:
            # The notches the train gets, not the ones commanded: the command
            # stage rate-limits every rise (NOTCH_UP_INTERVAL_S). Planning on
            # the commanded notches, a governor cycling traction under a
            # 5.4 m/s limit on 3.5 % lost a quarter of its mean notch to the
            # limit, so the train stalled where the planner's did not, was
            # never held, and rolled back 2 km (2026-10-16).
            n_db = to_notches(np.full(tf.shape, out_db[k]))
            want = np.where(n_db > 0, -n_db, to_notches(tf))
            if limiter is None:
                limiter = NotchLimiter(want)
            delivered = limiter.step(want, t[k])
            f_t = _notch_traction(np.clip(delivered, 0.0, None), v, table).sum()
            f_db = _dynamic_brake(np.clip(-delivered, 0.0, None), v, table).sum()
        else:
            f_t = (tf * f_trac_max[is_loco]).sum()
            f_db = 0.0
        if dt <= 0.0:
            continue
        sgn = np.sign(v) if abs(v) > 1e-6 else 0.0
        f_res = (davis[0] + davis[1] * abs(v) + davis[2] * v * v) * sgn
        f_b = (out_brake[k] * f_brk + f_db) * sgn
        a = (f_t - f_b - f_res - f_grade) / m_eff
        if breakaway and abs(v) < DEFAULT_V_BRAKE_EPS:
            # the RHS's standstill model, for the train as one mass
            drive = f_t - f_grade
            f_static = out_brake[k] * f_brk + f_start
            friction = np.clip(drive + m_eff * v / STANDSTILL_TAU_S, -f_static, f_static)
            a = (drive - friction) / m_eff
        v_new = v + a * dt
        if v > 0 and v_new < 0 and f_t < f_grade:     # stops rather than reverses
            v_new = 0.0 if abs(f_grade - f_t) <= f_brk * out_brake[k] else v_new
        x_lead += 0.5 * (v + v_new) * dt
        v = v_new
    return {"trac_frac": out_trac, "brake_frac": out_brake, "db_frac": out_db,
            "v_plan": v_plan, "v_allow": v_allow}
