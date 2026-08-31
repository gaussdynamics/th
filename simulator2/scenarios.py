"""Preset train scenarios (notebook-style simple consist)."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from . import constants as C
from .forces import traction_ramp
from .io_types import ExtendedTrainScenario
from .params import CouplerParameters, VehicleParameters
from .route import RouteProfile


def make_simple_train(
    n_cars: int,
    l0_m: float = 20.0,
) -> Tuple[List[VehicleParameters], List[CouplerParameters]]:
    """One lead locomotive + ``n_cars`` freight cars (notebook ``make_simple_train``)."""
    loco = VehicleParameters(
        mass_kg=C.DEFAULT_M_LOCO_KG,
        davis_A=C.DEFAULT_DAVIS_A_LOCO,
        davis_B=C.DEFAULT_DAVIS_B_LOCO,
        davis_C=C.DEFAULT_DAVIS_C_LOCO,
        can_traction=True,
        F_trac_max_N=300_000.0,
    )
    car = VehicleParameters(
        mass_kg=C.DEFAULT_M_CAR_KG,
        davis_A=C.DEFAULT_DAVIS_A_CAR,
        davis_B=C.DEFAULT_DAVIS_B_CAR,
        davis_C=C.DEFAULT_DAVIS_C_CAR,
        can_traction=False,
    )
    vehicles = [loco] + [car] * n_cars
    cp = CouplerParameters(
        L0_m=l0_m,
        slack_half_m=C.DEFAULT_SLACK_HALF_M,
        k_draft=C.DEFAULT_K_DRAFT,
        c_draft=C.DEFAULT_C_DRAFT,
        k_buff=C.DEFAULT_K_BUFF,
        c_buff=C.DEFAULT_C_BUFF,
    )
    couplers = [cp] * (len(vehicles) - 1)
    return vehicles, couplers


def _initial_positions(vehicles: List[VehicleParameters], couplers: List[CouplerParameters], x_lead: float) -> np.ndarray:
    n = len(vehicles)
    x0 = np.zeros(n, dtype=float)
    x0[0] = x_lead
    for i in range(1, n):
        x0[i] = x0[i - 1] - couplers[i - 1].L0_m
    return x0


def make_simple_train_scenario(
    n_cars: int = 20,
    l0_m: float = 20.0,
    x_lead_m: float = 1000.0,
    t_span: Tuple[float, float] = (0.0, 120.0),
    n_time_samples: int = 2401,
) -> ExtendedTrainScenario:
    """Default extended scenario: traction ramp on lead, zero brake command, flat route.

    Matches the spirit of Stage 7 notebook demo (commands can be overridden by caller).
    """
    vehicles, couplers = make_simple_train(n_cars, l0_m=l0_m)
    n = len(vehicles)
    route = RouteProfile(
        s_nodes_m=np.array([0.0, 50_000.0]),
        sin_theta_nodes=np.array([0.0, 0.0]),
        kappa_nodes=None,
    )
    x0 = _initial_positions(vehicles, couplers, x_lead_m)
    v0 = np.zeros(n, dtype=float)
    z0 = np.zeros(n, dtype=float)
    y0 = np.concatenate([x0, v0, z0, z0])

    def u_trac_cmd(t: float, i: int) -> float:
        return traction_ramp(t, t0=5.0, t1=25.0, F_max=200_000.0) if i == 0 else 0.0

    def u_brk_cmd(t: float, i: int) -> float:
        return 0.0

    t_eval = np.linspace(t_span[0], t_span[1], n_time_samples)

    return ExtendedTrainScenario(
        vehicles=vehicles,
        couplers=couplers,
        route=route,
        y0=y0,
        u_trac_cmd=u_trac_cmd,
        u_brk_cmd=u_brk_cmd,
        k_curv_scale=0.0,
        tau_brk_s=C.DEFAULT_TAU_BRK_S,
        tau_trac_s=C.DEFAULT_TAU_TRAC_S,
        p_max_w=C.DEFAULT_P_MAX_W,
        t_span=t_span,
        t_eval=t_eval,
        rtol=C.DEFAULT_RTOL,
        atol=C.DEFAULT_ATOL,
        method=C.DEFAULT_METHOD,
        extra_metadata={"scenario": "make_simple_train_scenario"},
    )


def make_scenario_on_route(
    route: RouteProfile,
    n_cars: int = 20,
    l0_m: float = 20.0,
    x_lead_m: Optional[float] = None,
    t_span: Tuple[float, float] = (0.0, 120.0),
    n_time_samples: int = 2401,
    F_trac_max_N: float = 200_000.0,
    k_curv_scale: float = 0.0,
) -> ExtendedTrainScenario:
    """Build a runnable scenario for a **real route** (loaded ``RouteProfile``).

    Same default consist and lead-loco traction ramp as
    ``make_simple_train_scenario``, but driven over ``route`` instead of a flat
    field. ``x_lead_m`` defaults to placing the whole consist just inside the
    start of the route (chainage 0). Set ``k_curv_scale`` > 0 to engage the
    curvature resistance proxy using the route's ``kappa`` field.
    """
    vehicles, couplers = make_simple_train(n_cars, l0_m=l0_m)
    n = len(vehicles)
    consist_len_m = sum(c.L0_m for c in couplers)
    if x_lead_m is None:
        x_lead_m = consist_len_m + 10.0  # rear car sits at ~chainage 0

    x0 = _initial_positions(vehicles, couplers, x_lead_m)
    v0 = np.zeros(n, dtype=float)
    z0 = np.zeros(n, dtype=float)
    y0 = np.concatenate([x0, v0, z0, z0])

    def u_trac_cmd(t: float, i: int) -> float:
        return traction_ramp(t, t0=5.0, t1=25.0, F_max=F_trac_max_N) if i == 0 else 0.0

    def u_brk_cmd(t: float, i: int) -> float:
        return 0.0

    t_eval = np.linspace(t_span[0], t_span[1], n_time_samples)

    return ExtendedTrainScenario(
        vehicles=vehicles,
        couplers=couplers,
        route=route,
        y0=y0,
        u_trac_cmd=u_trac_cmd,
        u_brk_cmd=u_brk_cmd,
        k_curv_scale=k_curv_scale,
        tau_brk_s=C.DEFAULT_TAU_BRK_S,
        tau_trac_s=C.DEFAULT_TAU_TRAC_S,
        p_max_w=C.DEFAULT_P_MAX_W,
        t_span=t_span,
        t_eval=t_eval,
        rtol=C.DEFAULT_RTOL,
        atol=C.DEFAULT_ATOL,
        method=C.DEFAULT_METHOD,
        extra_metadata={"scenario": "make_scenario_on_route", "n_vehicles": n},
    )


def make_scenario_from_consist(
    route: RouteProfile,
    vehicles: List[VehicleParameters],
    couplers: List[CouplerParameters],
    x_lead_m: Optional[float] = None,
    t_span: Tuple[float, float] = (0.0, 120.0),
    n_time_samples: int = 2401,
    F_trac_max_N: Optional[float] = None,
    k_curv_scale: float = 0.0,
    u_trac_cmd: Optional[Callable[[float, int], float]] = None,
    u_brk_cmd: Optional[Callable[[float, int], float]] = None,
) -> ExtendedTrainScenario:
    """Build a runnable scenario for an arbitrary consist (e.g. from
    ``consist.build_consist``), instead of the fixed ``make_simple_train`` shape.

    By default, traction is driven by each vehicle's ``can_traction`` flag
    rather than a hardcoded lead index (so multiple or mid-train locomotives
    all ramp up together) and brake is always zero. Each powered vehicle ramps
    toward its own ``F_trac_max_N`` unless ``F_trac_max_N`` is given here to
    override every powered vehicle's ceiling. Pass ``u_trac_cmd``/``u_brk_cmd``
    (e.g. from ``control_profile.build_command_callables``) to drive the run
    with a stored control profile instead.
    """
    n = len(vehicles)
    if len(couplers) != n - 1:
        raise ValueError(f"Expected {n - 1} couplers for {n} vehicles, got {len(couplers)}")
    consist_len_m = sum(c.L0_m for c in couplers)
    if x_lead_m is None:
        x_lead_m = consist_len_m + 10.0  # rear car sits at ~chainage 0

    x0 = _initial_positions(vehicles, couplers, x_lead_m)
    v0 = np.zeros(n, dtype=float)
    z0 = np.zeros(n, dtype=float)
    y0 = np.concatenate([x0, v0, z0, z0])

    if u_trac_cmd is None:
        def u_trac_cmd(t: float, i: int) -> float:
            if not vehicles[i].can_traction:
                return 0.0
            f_max = F_trac_max_N if F_trac_max_N is not None else vehicles[i].F_trac_max_N
            return traction_ramp(t, t0=5.0, t1=25.0, F_max=f_max)

    if u_brk_cmd is None:
        def u_brk_cmd(t: float, i: int) -> float:
            return 0.0

    t_eval = np.linspace(t_span[0], t_span[1], n_time_samples)

    return ExtendedTrainScenario(
        vehicles=vehicles,
        couplers=couplers,
        route=route,
        y0=y0,
        u_trac_cmd=u_trac_cmd,
        u_brk_cmd=u_brk_cmd,
        k_curv_scale=k_curv_scale,
        tau_brk_s=C.DEFAULT_TAU_BRK_S,
        tau_trac_s=C.DEFAULT_TAU_TRAC_S,
        p_max_w=C.DEFAULT_P_MAX_W,
        t_span=t_span,
        t_eval=t_eval,
        rtol=C.DEFAULT_RTOL,
        atol=C.DEFAULT_ATOL,
        method=C.DEFAULT_METHOD,
        extra_metadata={"scenario": "make_scenario_from_consist", "n_vehicles": n},
    )
