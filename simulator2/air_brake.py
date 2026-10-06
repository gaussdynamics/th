"""Empirical freight air brake: brake-wave propagation along the train.

Replaces the train-wide, instantaneous brake of Chapter 4 (findings F1 and F2
in ``SIMULATOR_VALIDATION_NOTE.md``) with the empirical form surveyed by
Wu, Spiryagin & Cole (2016, VSD 54(12), section 4.2), the class the survey
recommends when the air brake is not itself the object of study:

* **Propagation.** The driver's command reaches the vehicle at distance ``d``
  behind the lead after ``d / c`` (Martin & Hay; Grebenyuk's Eq. 5 in the
  survey). Release travels the same way, but more slowly.
* **Cylinder build-up.** Each vehicle's brake force follows the existing
  first-order lag towards its command (Murtaza's form, the survey's Eq. 6),
  with a time constant that grows linearly with distance from the lead,
  because the rear cars fill from a brake pipe that is still being reduced.
  Release exhausts the cylinder with its own, shorter time constant.
* **Brake force.** Conventional North American cars without empty/load
  equipment carry a fixed shoe force set by the *design* gross weight, so an
  empty car gets the same retarding force as a loaded one, capped at
  wheel-rail adhesion on its actual weight. (F1 described the defect as load
  independence; the defect is the 250 kN magnitude and the missing cap.)
* **Direct release.** A freight control valve cannot release partially: any
  release is a full release, and the brake can be re-applied only once the
  brake pipe has recharged.

Everything here works on the command side. The RHS accepts one application
and one release time constant per vehicle (``TorchScenarioBatch.tau_brk_node``
and ``tau_brk_release_node``).

**Timing parameters** are calibrated to the measured service applications of
Wei et al. (2017), digitized by ``scripts/digitize_air_brake.py``: a 10,000 t
Shuo-Huang train, HXD1 + 116 C80 cars with Chinese 120 distribution valves,
pressures measured at the 4th and 116th car. The 120 valve is not the North
American ABD/ABDX family; it has the same direct release and graduated
application, but carrying its timing over is an assumption.

**Force parameters** (net braking ratio, shoe friction, adhesion, design
gross weight) are still placeholders: the paper reports cylinder pressures,
not forces. :func:`require_sourced` names them and refuses to pass until they
have citations in ``AirBrakeParams.sources``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .constants import GRAVITY_MPS2

# -- calibration: Wei et al. (2017) ------------------------------------------------

WEI2017 = ("Wei, Hu, Wu, Zhao, Zhang & Zhang (2017), An air brake model for "
           "longitudinal train dynamics studies, VSD 55(4):517-533")

#: C80 coal car length over couplers [m]. NOT given in Wei et al. (2017); it
#: needs its own citation. Every propagation speed below scales with it.
C80_PITCH_M = 12.0
#: Brake pipe length between the 4th and 116th car test sections [m].
WEI2017_SPAN_M = 112 * C80_PITCH_M

#: Brake cylinder onset, 4th to 116th car [s]: Fig. 5 (small reduction) and
#: Fig. 7 (large reduction), measured curves.
WEI2017_APPLY_DELAY_S = (7.03, 8.25)
#: Brake cylinder exhaust onset, 4th to 116th car [s]: Fig. 5.
WEI2017_RELEASE_DELAY_S = 12.82
#: First-order build-up time constant at the 4th and 116th car [s], second
#: (140 kPa) step of Fig. 7. The 4th car fits worse (rms 24 kPa of 400 kPa):
#: it jumps most of the way within seconds, then creeps.
WEI2017_TAU_APPLY_S = (10.2, 30.6)
#: Exhaust time constant down to piston retraction, 4th and 116th car [s]: Fig. 5.
WEI2017_TAU_RELEASE_S = (6.89, 6.52)

_CAR = "car length C80_PITCH_M unsourced"


@dataclass(frozen=True)
class AirBrakeParams:
    """Empirical air brake parameters.

    Timing is calibrated (see ``sources``); force parameters are placeholders.
    Overriding a sourced field does not update ``sources``: pass a new one.
    """

    #: Brake-wave propagation speed for a service application [m/s]: 112 car
    #: lengths over the mean measured onset delay (the two tests: 163-191 m/s).
    propagation_mps: float = WEI2017_SPAN_M / float(np.mean(WEI2017_APPLY_DELAY_S))
    #: Release-wave propagation speed [m/s]. Must not exceed ``propagation_mps``.
    release_propagation_mps: float = WEI2017_SPAN_M / WEI2017_RELEASE_DELAY_S
    #: Cylinder build-up time constant at the lead vehicle [s].
    tau_head_s: float = WEI2017_TAU_APPLY_S[0]
    #: Growth of the build-up time constant with distance from the lead [s/m].
    #: Per metre, not per train: a short train's tail fills like a long
    #: train's middle.
    tau_per_m_s: float = (WEI2017_TAU_APPLY_S[1] - WEI2017_TAU_APPLY_S[0]) / WEI2017_SPAN_M
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
    #: Citation per physical parameter. A field missing here is a placeholder.
    sources: dict[str, str] = field(default_factory=lambda: {
        "propagation_mps": f"{WEI2017}, Figs 5 and 7 ({_CAR})",
        "release_propagation_mps": f"{WEI2017}, Fig. 5 ({_CAR})",
        "tau_head_s": f"{WEI2017}, Fig. 7",
        "tau_per_m_s": f"{WEI2017}, Fig. 7 ({_CAR})",
        "tau_release_s": f"{WEI2017}, Fig. 5",
        "recharge_s": f"{WEI2017}, section 4, point (2)",
    })


#: Fields that need a citation before data can be generated.
PHYSICAL_FIELDS = (
    "propagation_mps", "release_propagation_mps", "tau_head_s", "tau_per_m_s",
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
    """When the brake wave reaches each vehicle, after the driver acts [s]."""
    return np.asarray(dist_m, float) / p.propagation_mps


def release_delay_s(dist_m: np.ndarray, p: AirBrakeParams) -> np.ndarray:
    """When the release wave reaches each vehicle, after the driver acts [s]."""
    return np.asarray(dist_m, float) / p.release_propagation_mps


def rise_tau_s(dist_m: np.ndarray, p: AirBrakeParams) -> np.ndarray:
    """Per-vehicle build-up time constant, linear in distance from the lead [s]."""
    return p.tau_head_s + p.tau_per_m_s * np.asarray(dist_m, float)


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

    #: Commands ``(T, N)`` [N], for ``u_brk``.
    u_brk: np.ndarray
    #: Build-up time constants ``(N,)`` [s], for ``tau_brk_node``.
    tau_apply: np.ndarray
    #: Exhaust time constants ``(N,)`` [s], for ``tau_brk_release_node``.
    tau_release: np.ndarray
    #: Capacities ``(N,)`` [N], for the ``F_BRK_MAX_N`` column.
    f_max: np.ndarray


def vehicle_commands(t: np.ndarray, driver_frac: np.ndarray, x0: np.ndarray,
                     mass_kg: np.ndarray, is_loco: np.ndarray,
                     p: AirBrakeParams) -> BrakeCommands:
    """Per-vehicle brake commands from the driver's brake fraction.

    Increases reach each vehicle with the brake wave, decreases with the
    slower release wave: each vehicle follows the larger of the two shifted
    driver levels. That is exact while each release has passed the whole
    train before the next application starts. Direct release guarantees it
    for trains shorter than ``recharge_s / (1/c_release - 1/c_apply)``,
    about 22 km with the calibrated values.
    Before ``t[0]`` each vehicle holds the driver's first command.
    """
    if p.release_propagation_mps > p.propagation_mps:
        raise ValueError("release_propagation_mps must not exceed propagation_mps")
    t = np.asarray(t, float)
    level = (direct_release(t, driver_frac, p) if p.direct_release
             else np.clip(np.asarray(driver_frac, float), 0.0, 1.0))
    dist = distance_behind_lead_m(x0)
    f_max = brake_capacity_n(mass_kg, is_loco, p)
    u = np.stack([np.maximum(np.interp(t - da, t, level), np.interp(t - dr, t, level))
                  for da, dr in zip(onset_delay_s(dist, p), release_delay_s(dist, p))],
                 axis=1)
    return BrakeCommands(u * f_max[None, :], rise_tau_s(dist, p),
                         np.full(dist.shape, p.tau_release_s), f_max)


__all__ = [
    "AirBrakeParams", "BrakeCommands", "C80_PITCH_M", "PHYSICAL_FIELDS",
    "brake_capacity_n", "direct_release", "distance_behind_lead_m",
    "onset_delay_s", "release_delay_s", "require_sourced", "rise_tau_s",
    "unsourced", "vehicle_commands",
]
