"""Empirical freight air brake: brake pipe lag and control valve fill.

Replaces the train-wide, instantaneous brake of Chapter 4 (findings F1 and F2
in ``SIMULATOR_VALIDATION_NOTE.md``) with a reduced brake-pipe model, in the
empirical class surveyed by Wu, Spiryagin & Cole (2016, VSD 54(12),
section 4.2) for studies where the air brake is not itself the object:

* **Brake front.** A reduction reaches the vehicle at pipe distance ``d``
  behind the exhaust point after ``d / c`` (Martin & Hay; Grebenyuk's Eq. 5 in the
  survey), and the control valve's quick service vents a fixed part of a
  full-service reduction there at once. Release travels the same way, more
  slowly.
* **Brake pipe lag.** The rest of the reduction drains along the pipe from
  the exhaust point. The pipe pressure is modelled as linear diffusion with
  the pressure held at the driver's target at the exhaust and no flow at the
  far end. Behind the head the cylinders are limited by the pipe, not by
  their valves (Spiryagin et al. 2017, p. 214), so the lag saturates with
  distance and scales with the square of the pipe length: the slowest mode
  is ``4 L^2 / (pi^2 D)``. A first-order lag that grows linearly with
  distance, the previous form, gets neither.
* **Control valve.** The cylinder follows the local pipe reduction upwards
  only: a fast inshot to a fixed level, then a choke-limited service rate,
  never faster than the pipe allows. Holding the pipe laps the valve.
* **Brake force.** Conventional North American cars without empty/load
  equipment carry a fixed shoe force set by the *design* gross weight, so an
  empty car gets the same retarding force as a loaded one, capped at
  wheel-rail adhesion on its actual weight. (F1 described the defect as load
  independence; the defect is the 250 kN magnitude and the missing cap.)
* **Direct release.** A freight control valve cannot release partially: any
  release is a full release, and the brake can be re-applied only once the
  brake pipe has recharged. The cylinder then exhausts with its own time
  constant.

Distances are along the brake pipe, which can be much longer than the
train (Spiryagin et al. 2017, p. 214; Wu et al. 2017 have 13.5 m of pipe per
wagon). The sources measure in pipe metres, so ``pipe_length_factor`` (pipe
metres per metre of train) converts vehicle positions. Its default of 1.0 is
an assumption without a source.

Pressures are carried as fractions of a full-service application, which
maps linearly onto the driver's brake fraction and onto the brake force.
Everything up to the cylinder is computed on the command side; the RHS
applies a short numerical lag on application (``actuator_tau_s``) and the
exhaust time constant on release (``TorchScenarioBatch.tau_brk_node`` and
``tau_brk_release_node``).

**Timing parameters** are fitted to Spiryagin et al. (2017), Figs 5.67 and
5.68: brake pipe and cylinder pressures of a 100-wagon train with a ~2000 m
pipe, minimum and full-service applications, vented at the head only and at
both ends (``scripts/digitize_spiryagin_brake.py`` digitizes them,
``scripts/calibrate_air_brake.py`` fits them). Positions there are given in
metres, so no car length enters. The fit predicts the measured 116th-car
cylinder of Wei et al. (2017), a different train and valve, to 10 kPa rms.
The valve type is not named. The release speed is measured by Wu et al.
(2017) on a 150-wagon test rig, also in pipe metres; the exhaust time
constant and recharge come from Wei et al. (Chinese 120 valve).

**Force parameters** (net braking ratio, shoe friction, adhesion, design
gross weight) are still placeholders: the sources report cylinder pressures,
not forces. :func:`require_sourced` names them and refuses to pass until they
have citations in ``AirBrakeParams.sources``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy.signal import lfilter

from .constants import GRAVITY_MPS2

# -- calibration: Spiryagin et al. (2017) -------------------------------------------

SPIRYAGIN2017 = ("Spiryagin, Wolfs, Cole, Spiryagin, Sun & McSweeney (2017), Design and "
                 "Simulation of Heavy Haul Locomotives and Trains, CRC Press")
_FIT = f"{SPIRYAGIN2017}, Figs 5.67-5.68, fitted by scripts/calibrate_air_brake.py"

#: Brake front speed [m/s]: mean of the BC and BP onset waves over 16-1996 m
#: in Fig. 5.67 (a) and (b), 266-273 m/s. Inside the 250-300 m/s the book
#: gives for a well-designed system (p. 173).
SPIRYAGIN2017_PROPAGATION_MPS = 270.0

# -- calibration: Wu et al. (2017) -------------------------------------------------

WU2017 = ("Wu, Cole, Spiryagin, Wang, Ma & Wei (2017), Railway air brake model and "
          "parallel computing scheme, J. Comput. Nonlinear Dyn. 12(5):051017")

#: Brake release propagation speed [m/s], measured on the CRRC Brake Science &
#: Technology rig: 150 wagon brake sets, 600 kPa, 13.5 m of pipe per wagon
#: (section 5). The paper's own release onsets (Table 3, 11.4 s from wagon
#: 001 to 150) over 149 x 13.5 m give 176 m/s; the stated figure is used.
WU2017_RELEASE_MPS = 162.52

# -- calibration: Wei et al. (2017) ------------------------------------------------

WEI2017 = ("Wei, Hu, Wu, Zhao, Zhang & Zhang (2017), An air brake model for "
           "longitudinal train dynamics studies, VSD 55(4):517-533")

#: C80 coal car length over couplers [m]. Not given in Wei et al. (2017), Wei,
#: Ahmadian & Zhang (2014) or Wu et al. (2017); unsourced. Used only to place
#: the cars in the out-of-sample check of scripts/calibrate_air_brake.py.
C80_PITCH_M = 12.0

#: Brake cylinder onset, 4th to 116th car [s]: Fig. 5 (small reduction) and
#: Fig. 7 (large reduction), measured curves. Kept for the out-of-sample check.
WEI2017_APPLY_DELAY_S = (7.03, 8.25)
#: Exhaust time constant down to piston retraction, 4th and 116th car [s]: Fig. 5.
WEI2017_TAU_RELEASE_S = (6.89, 6.52)


@dataclass(frozen=True)
class AirBrakeParams:
    """Empirical air brake parameters.

    Timing is calibrated (see ``sources``); force parameters are placeholders.
    Overriding a sourced field does not update ``sources``: pass a new one.
    """

    #: Brake front speed for a service application [m/s].
    propagation_mps: float = SPIRYAGIN2017_PROPAGATION_MPS
    #: Release-wave propagation speed [m/s]. Must not exceed ``propagation_mps``.
    release_propagation_mps: float = WU2017_RELEASE_MPS
    #: Brake pipe length per metre of train [-]: converts the distance between
    #: vehicle centres into pipe distance, in which the speeds and the
    #: diffusivity are measured. ASSUMPTION: 1.0 (pipe as long as the train);
    #: real pipes are longer, which slows the front and the pipe lag.
    pipe_length_factor: float = 1.0
    #: Brake pipe diffusivity of the pressure reduction [m^2/s]. Sets the pipe
    #: lag: the slowest mode of a pipe of length L is ``4 L^2 / (pi^2 D)``,
    #: 65 s for 2000 m.
    pipe_diffusivity_m2ps: float = 24_800.0
    #: Pipe reduction vented locally by quick service when the front arrives
    #: [fraction of a full-service reduction].
    quick_service_frac: float = 0.187
    #: Cylinder level the valve's inshot fills to quickly [fraction of full service].
    inshot_frac: float = 0.194
    #: Cylinder fill rate up to the inshot level [full service per second].
    inshot_rate_per_s: float = 0.130
    #: Choke-limited cylinder fill rate above the inshot level [full service per
    #: second]: ~15 kPa/s on a 350 kPa full service.
    service_rate_per_s: float = 0.0430
    #: Cylinder exhaust time constant on release [s]; no trend along the train.
    tau_release_s: float = float(np.mean(WEI2017_TAU_RELEASE_S))
    #: Design gross weight of a freight car: brake rigging is sized for it,
    #: so it sets the shoe force whatever the car carries [kg].
    #: PLACEHOLDER: 286,000 lb, the common North American rating.
    car_design_gross_kg: float = 129_700.0
    #: Net braking ratio at full service: total shoe force over design gross
    #: weight. PLACEHOLDER: needs the AAR requirement.
    net_braking_ratio: float = 0.10
    #: Brake shoe friction coefficient (composite shoe). PLACEHOLDER.
    shoe_friction: float = 0.30
    #: Wheel-rail adhesion available to braking. PLACEHOLDER.
    adhesion: float = 0.15
    #: Model the freight control valve's direct release.
    direct_release: bool = True
    #: Brake pipe recharge time before a released brake can re-apply [s].
    #: Wei et al.: at the 4th car, brake pipe and auxiliary reservoir pressures
    #: coincide 90 s after release (70 kPa reduction). At the 116th car the
    #: reservoir is still ~20 kPa short after 206 s, so this is optimistic for
    #: the tail of a long train and for deeper reductions.
    recharge_s: float = 90.0
    #: Command change below this counts as holding, not releasing [fraction].
    #: A modelling tolerance, not a physical parameter.
    release_tol: float = 0.02
    #: RHS lag on application [s]. Numerical, not physical: the command already
    #: carries the cylinder fill, this only smooths it for the integrator.
    actuator_tau_s: float = 0.25
    #: Time step of the command-side pipe and valve computation [s]. Numerical.
    dt_s: float = 0.1
    #: Brake pipe discretisation for the diffusion modes [m]. Numerical.
    pipe_dx_m: float = 20.0
    #: Citation per physical parameter. A field missing here is a placeholder.
    sources: dict[str, str] = field(default_factory=lambda: {
        "propagation_mps": f"{SPIRYAGIN2017}, Fig. 5.67 (onset waves)",
        "release_propagation_mps": f"{WU2017}, section 5 (measured)",
        "pipe_diffusivity_m2ps": _FIT,
        "quick_service_frac": _FIT,
        "inshot_frac": _FIT,
        "inshot_rate_per_s": _FIT,
        "service_rate_per_s": _FIT,
        "tau_release_s": f"{WEI2017}, Fig. 5",
        "recharge_s": f"{WEI2017}, section 4, point (2)",
    })


#: Fields that need a citation before data can be generated.
PHYSICAL_FIELDS = (
    "propagation_mps", "release_propagation_mps", "pipe_length_factor",
    "pipe_diffusivity_m2ps",
    "quick_service_frac", "inshot_frac", "inshot_rate_per_s", "service_rate_per_s",
    "tau_release_s", "car_design_gross_kg", "net_braking_ratio",
    "shoe_friction", "adhesion", "recharge_s",
)


def unsourced(p: AirBrakeParams) -> list[str]:
    """Physical parameters without a citation."""
    return [f for f in PHYSICAL_FIELDS if not p.sources.get(f)]


def require_sourced(p: AirBrakeParams) -> None:
    """Refuse placeholder parameters (call before generating data)."""
    missing = unsourced(p)
    if missing:
        raise ValueError(
            f"AirBrakeParams placeholders without a source: {', '.join(missing)}; "
            "replace them with cited values before generating data"
        )


def brake_capacity_n(mass_kg: np.ndarray, is_loco: np.ndarray,
                     p: AirBrakeParams) -> np.ndarray:
    """Full-service retarding force per vehicle [N].

    Cars: shoe force from the design gross weight (fixed per car), times the
    shoe friction coefficient. Locomotives: from their own weight. Both are
    capped at adhesion on the vehicle's actual weight.
    """
    mass = np.asarray(mass_kg, float)
    design = np.where(np.asarray(is_loco, bool), mass, p.car_design_gross_kg)
    retard = p.shoe_friction * p.net_braking_ratio * design * GRAVITY_MPS2
    return np.minimum(retard, p.adhesion * mass * GRAVITY_MPS2)


def distance_behind_lead_m(x0: np.ndarray) -> np.ndarray:
    """Distance of each vehicle's centre behind the lead's centre [m]."""
    x0 = np.asarray(x0, float)
    return x0[0] - x0


def onset_delay_s(dist_m: np.ndarray, p: AirBrakeParams) -> np.ndarray:
    """When the brake front reaches each pipe distance, after the driver acts [s]."""
    return np.asarray(dist_m, float) / p.propagation_mps


def release_delay_s(dist_m: np.ndarray, p: AirBrakeParams) -> np.ndarray:
    """When the release wave reaches each pipe distance, after the driver acts [s]."""
    return np.asarray(dist_m, float) / p.release_propagation_mps


def pipe_modes(dist_m: np.ndarray, length_m: float, diffusivity_m2ps: float,
               vent_tail: bool = False, dx_m: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Brake pipe response as a sum of first-order lags.

    The reduction ``w`` obeys ``w_t = D w_xx`` on ``[0, L]``, held at the
    driver's level at ``x = 0`` (and at ``x = L`` if ``vent_tail``), with no
    flow at an unvented end. On a grid of spacing ``~dx_m`` this linear system
    diagonalises exactly, so the reduction at each vehicle is

        w_i(t) = sum_k gain[i, k] * lag(level, tau[k])(t),

    with ``sum_k gain[i, k] = 1`` (a held level fills the pipe). The vented
    ends enter as a mode with ``tau = 0``. Returns ``(tau (K,), gain (N, K))``.
    """
    dist = np.asarray(dist_m, float)
    if length_m <= 0.0:
        return np.zeros(1), np.ones((dist.size, 1))
    m = max(1, int(np.ceil(length_m / dx_m)))
    h = length_m / m
    x = np.arange(m + 1) * h
    inner = np.arange(1, m if vent_tail else m + 1)
    n, k = inner.size, diffusivity_m2ps / h**2
    if n == 0:                                   # one cell, vented at both ends
        nodes = np.ones((m + 1, 1))
        return np.zeros(1), np.stack([np.interp(dist, x, nodes[:, 0])], axis=1)
    a = np.diag(np.full(n, -2.0 * k)) + np.diag(np.full(n - 1, k), 1) + np.diag(np.full(n - 1, k), -1)
    b = np.zeros(n)
    b[0] += k
    if vent_tail:
        b[-1] += k
    elif n > 1:
        a[-1, -2] = 2.0 * k                      # mirror node: no flow at the end
    else:
        a[0, 0] = -k                             # single interior node, no flow beyond
    lam, vec = np.linalg.eig(a)
    lam, vec = lam.real, vec.real
    amp = -np.linalg.solve(vec, b) / lam         # modal amplitude per unit level
    nodes = np.zeros((m + 1, n + 1))
    nodes[inner, :n] = vec * amp
    nodes[0, n] = 1.0
    if vent_tail:
        nodes[m, n] = 1.0
    gain = np.stack([np.interp(dist, x, nodes[:, c]) for c in range(n + 1)], axis=1)
    return np.r_[-1.0 / lam, 0.0], gain


def _lag(level: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """First-order lag of a sampled signal, settled at its first value."""
    if tau <= 0.0:
        return level.copy()
    a = np.exp(-dt / tau)
    out, _ = lfilter([1.0 - a], [1.0, -a], level, zi=[a * level[0]])
    return out


def cylinder_level(t: np.ndarray, level: np.ndarray, dist_m: np.ndarray,
                   p: AirBrakeParams, length_m: float | None = None,
                   vent_tail: bool = False) -> np.ndarray:
    """Brake cylinder level of each vehicle ``(T, N)`` [fraction of full service].

    ``level`` is the brake level the control valves can follow (after
    :func:`direct_release`, or the clipped driver fraction), sampled at ``t``.
    ``dist_m`` are distances along the pipe from the lead (see
    ``pipe_length_factor``). The pipe is ``length_m`` long (default: to the
    last vehicle) and is
    exhausted at the lead, and also at the tail if ``vent_tail``. Each vehicle:

    * sees the level arrive with the front, ``d / c`` after the driver; until
      the release wave (``d / c_release``) it holds the level it last saw;
    * has a pipe reduction of at least the quick-service part, otherwise the
      diffused one, never above the level that has arrived;
    * fills its cylinder towards that reduction at the inshot rate, then the
      service rate; it never empties while applied (the valve laps);
    * is empty once the level that has arrived is zero (released).

    Before ``t[0]`` the driver's first level is held and everything is settled.
    Computed on a ``dt_s`` grid and held between its points when sampled at ``t``.
    """
    t = np.asarray(t, float)
    dist = np.asarray(dist_m, float)
    if length_m is None:
        length_m = float(dist.max(initial=0.0))
    tg = np.arange(t[0], t[-1] + p.dt_s / 2, p.dt_s)
    lev = np.interp(tg, t, np.clip(np.asarray(level, float), 0.0, 1.0))

    tau, gain = pipe_modes(dist, length_m, p.pipe_diffusivity_m2ps, vent_tail, p.pipe_dx_m)
    w = np.stack([_lag(lev, p.dt_s, tk) for tk in tau], axis=1) @ gain.T     # (Tg, N)

    front = np.minimum(dist, length_m - dist) if vent_tail else dist
    arrived = np.stack(
        [np.maximum(np.interp(tg - da, tg, lev), np.interp(tg - dr, tg, lev))
         for da, dr in zip(onset_delay_s(front, p), release_delay_s(front, p))], axis=1)
    reduction = np.minimum(np.maximum(w, np.minimum(p.quick_service_frac, arrived)), arrived)

    cyl = np.empty_like(reduction)
    b = arrived[0].copy()
    for k in range(tg.size):
        rate = np.where(b < p.inshot_frac, p.inshot_rate_per_s, p.service_rate_per_s)
        fill = np.minimum(reduction[k], b + rate * p.dt_s)
        b = np.maximum(b, fill) if p.direct_release else fill
        b[arrived[k] <= 0.0] = 0.0
        cyl[k] = b
    # Hold each grid value until the next (causal: interpolating would show a
    # step up to ``dt_s`` before the level that causes it).
    return cyl[np.clip(np.searchsorted(tg, t + 1e-9, side="right") - 1, 0, tg.size - 1)]


def direct_release(t: np.ndarray, frac: np.ndarray, p: AirBrakeParams) -> np.ndarray:
    """The brake level the control valves can follow, given the driver's.

    Rising or holding commands pass through (graduated application). A drop
    of more than ``release_tol`` below the application's peak releases fully;
    the brake then stays off for ``recharge_s`` and re-applies at the
    driver's command once that is above ``release_tol`` again. Small dips
    hold the peak (the valve laps).
    """
    t = np.asarray(t, float)
    frac = np.clip(np.asarray(frac, float), 0.0, 1.0)
    out = np.zeros_like(frac)
    applying, peak, ready_at = False, 0.0, -np.inf
    for k, f in enumerate(frac):
        if applying:
            if f >= peak - p.release_tol:
                peak = max(peak, f)
                out[k] = peak
                continue
            applying, ready_at = False, t[k] + p.recharge_s
        if t[k] >= ready_at and f > p.release_tol:
            applying, peak = True, f
            out[k] = f
    return out


class BrakeCommands(NamedTuple):
    """Per-vehicle inputs for :class:`~simulator2.torch_rhs.TorchScenarioBatch`."""

    #: Commands ``(T, N)`` [N], for ``u_brk``: the cylinder level times capacity.
    u_brk: np.ndarray
    #: Application time constants ``(N,)`` [s], for ``tau_brk_node``.
    tau_apply: np.ndarray
    #: Exhaust time constants ``(N,)`` [s], for ``tau_brk_release_node``.
    tau_release: np.ndarray
    #: Capacities ``(N,)`` [N], for the ``F_BRK_MAX_N`` column.
    f_max: np.ndarray


def vehicle_commands(t: np.ndarray, driver_frac: np.ndarray, x0: np.ndarray,
                     mass_kg: np.ndarray, is_loco: np.ndarray,
                     p: AirBrakeParams) -> BrakeCommands:
    """Per-vehicle brake commands from the driver's brake fraction.

    The driver's valve exhausts the pipe at the lead vehicle; the pipe runs
    to the last vehicle and is ``pipe_length_factor`` times the distance
    between them. See :func:`cylinder_level`. The release wave must pass the
    whole train before the next application starts; direct release
    guarantees it for pipes shorter than
    ``recharge_s / (1/c_release - 1/c_apply)``, about 37 km with the
    calibrated values. ``t`` should resolve the command changes; the
    cylinder itself is computed on the ``dt_s`` grid.
    """
    if p.release_propagation_mps > p.propagation_mps:
        raise ValueError("release_propagation_mps must not exceed propagation_mps")
    t = np.asarray(t, float)
    level = (direct_release(t, driver_frac, p) if p.direct_release
             else np.clip(np.asarray(driver_frac, float), 0.0, 1.0))
    dist = p.pipe_length_factor * distance_behind_lead_m(x0)
    f_max = brake_capacity_n(mass_kg, is_loco, p)
    cyl = cylinder_level(t, level, dist, p)
    return BrakeCommands(cyl * f_max[None, :], np.full(dist.shape, p.actuator_tau_s),
                         np.full(dist.shape, p.tau_release_s), f_max)


__all__ = [
    "AirBrakeParams", "BrakeCommands", "C80_PITCH_M", "PHYSICAL_FIELDS",
    "brake_capacity_n", "cylinder_level", "direct_release", "distance_behind_lead_m",
    "onset_delay_s", "pipe_modes", "release_delay_s", "require_sourced",
    "unsourced", "vehicle_commands",
]
