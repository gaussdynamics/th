"""Generate reference trajectories for the torch RHS port's equivalence tests.

Produces one self-describing ``.npz`` per case under ``torch_reference/``. Each
file carries everything needed to reconstruct and re-run the scenario, so the
torch implementation never has to import the NumPy path to be tested against
it -- it reads the fixture, integrates, and compares.

Committed so the fixtures are reproducible. Regenerate with:

    python3 simulator2/tests/fixtures/generate_torch_fixtures.py

Cases cover all 11 driving regimes plus two deliberately adversarial ones: a
slack-cycling case that repeatedly crosses the coupler deadband (where a
fixed-step integrator is most likely to diverge from the adaptive reference),
and a brake-to-standstill case that exercises the ``brake_opposes_motion``
sign blend through v = 0.
"""
from __future__ import annotations

import time

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from simulator2 import constants as C
from simulator2.control_profile import (
    CommandCurve,
    CommandKnot,
    ControlProfile,
    build_command_callables,
)
from simulator2.driving_regimes import Regime, generate_regime_profile
from simulator2.io_types import ExtendedTrainScenario
from simulator2.params import CouplerParameters, VehicleParameters
from simulator2.route import RouteProfile
from simulator2.scenarios import _initial_positions
from simulator2.simulate import materialize_commands, simulate_train_tensorized

OUT = Path(__file__).parent / "torch_reference"
OUT.mkdir(parents=True, exist_ok=True)

N_VEH = 6
DURATION_S = 40.0
DT_OUT = 0.05


def make_consist(n: int = N_VEH):
    vehicles = [
        VehicleParameters(
            mass_kg=130_000.0 if i == 0 else 100_000.0 + 5_000.0 * i,
            davis_A=800.0 if i == 0 else 600.0,
            davis_B=15.0 if i == 0 else 12.0,
            davis_C=0.8 if i == 0 else 0.7,
            can_traction=(i == 0),
            F_trac_max_N=600_000.0 if i == 0 else 0.0,
            F_brk_max_N=250_000.0,
        )
        for i in range(n)
    ]
    couplers = [
        CouplerParameters(
            L0_m=20.0,
            slack_half_m=C.DEFAULT_SLACK_HALF_M,
            k_draft=C.DEFAULT_K_DRAFT,
            c_draft=C.DEFAULT_C_DRAFT,
            k_buff=C.DEFAULT_K_BUFF,
            c_buff=C.DEFAULT_C_BUFF,
        )
        for _ in range(n - 1)
    ]
    return vehicles, couplers


def make_route(kind: str = "rolling") -> RouteProfile:
    s = np.linspace(0.0, 12_000.0, 1201)
    if kind == "flat":
        sin_theta = np.zeros_like(s)
    elif kind == "descent":
        sin_theta = np.full_like(s, -0.015)
    else:  # rolling
        sin_theta = 0.012 * np.sin(2 * np.pi * s / 2500.0) + 0.004 * np.sin(2 * np.pi * s / 700.0)
    kappa = 0.0008 * np.sin(2 * np.pi * s / 1800.0)
    v_max = np.full_like(s, 25.0)
    return RouteProfile(s, sin_theta, kappa, v_max)


def _curve(points):
    return CommandCurve(knots=[CommandKnot(t_s=float(t), fraction=float(f)) for t, f in points])


def slack_cycling_profile() -> ControlProfile:
    """Alternating power and brake, tuned to drive the couplers in and out of
    the slack deadband repeatedly -- the hardest case for a fixed step."""
    pts_t, pts_b = [], []
    for k in range(8):
        t0 = k * (DURATION_S / 8.0)
        if k % 2 == 0:
            pts_t += [(t0, 0.0), (t0 + 2.0, 0.85)]
            pts_b += [(t0, 0.0), (t0 + 2.0, 0.0)]
        else:
            pts_t += [(t0, 0.85), (t0 + 2.0, 0.0)]
            pts_b += [(t0, 0.0), (t0 + 2.0, 0.35)]
    pts_t.append((DURATION_S, 0.0))
    pts_b.append((DURATION_S, 0.0))
    return ControlProfile(
        name="slack_cycling", description="adversarial: repeated deadband crossings",
        duration_s=DURATION_S, brake=_curve(pts_b), traction_head_end=_curve(pts_t),
        traction_dpu_mid=_curve([(0.0, 0.0), (DURATION_S, 0.0)]),
        traction_dpu_rear=_curve([(0.0, 0.0), (DURATION_S, 0.0)]),
        seed=None,
    )


def brake_to_stop_profile() -> ControlProfile:
    """Full brake from speed to a complete stop, then hold: exercises the
    tanh(v / v_brake_eps) sign blend through and beyond v = 0."""
    return ControlProfile(
        name="brake_to_stop", description="adversarial: standstill under brake",
        duration_s=DURATION_S,
        brake=_curve([(0.0, 0.0), (5.0, 0.0), (10.0, 0.7), (DURATION_S, 0.7)]),
        traction_head_end=_curve([(0.0, 0.0), (DURATION_S, 0.0)]),
        traction_dpu_mid=_curve([(0.0, 0.0), (DURATION_S, 0.0)]),
        traction_dpu_rear=_curve([(0.0, 0.0), (DURATION_S, 0.0)]),
        seed=None,
    )


def build_scenario(profile: ControlProfile, route: RouteProfile, v0: float,
                   x_lead: float, k_curv_scale: float,
                   rtol: float = 1e-8, atol: float = 1e-10,
                   method: str = "RK45") -> ExtendedTrainScenario:
    vehicles, couplers = make_consist()
    n = len(vehicles)
    x0 = _initial_positions(vehicles, couplers, x_lead)
    v0_vec = np.full(n, v0, dtype=float)

    def u_trac_cmd(t: float, i: int) -> float:
        v = vehicles[i]
        if not v.can_traction:
            return 0.0
        return max(0.0, profile.traction_head_end.value_at(t)) * v.F_trac_max_N

    def u_brk_cmd(t: float, i: int) -> float:
        return max(0.0, profile.brake.value_at(t)) * vehicles[i].F_brk_max_N

    z_trac0 = np.array([u_trac_cmd(0.0, i) for i in range(n)])
    z_brk0 = np.array([u_brk_cmd(0.0, i) for i in range(n)])
    y0 = np.concatenate([x0, v0_vec, z_brk0, z_trac0])

    n_samples = int(round(DURATION_S / DT_OUT)) + 1
    return ExtendedTrainScenario(
        vehicles=vehicles, couplers=couplers, route=route, y0=y0,
        u_trac_cmd=u_trac_cmd, u_brk_cmd=u_brk_cmd,
        k_curv_scale=k_curv_scale,
        tau_brk_s=C.DEFAULT_TAU_BRK_S, tau_trac_s=C.DEFAULT_TAU_TRAC_S,
        p_max_w=C.DEFAULT_P_MAX_W, brake_opposes_motion=True,
        t_span=(0.0, DURATION_S),
        t_eval=np.linspace(0.0, DURATION_S, n_samples),
        rtol=rtol, atol=atol, method=method,
    )


def write_case(name: str, sc: ExtendedTrainScenario) -> None:
    _t0 = time.time()
    res = simulate_train_tensorized(sc)
    _w = time.time() - _t0
    u_trac, u_brk = materialize_commands(sc, res.t)
    n = len(sc.vehicles)
    node_static = np.array(
        [[v.mass_kg, v.davis_A, v.davis_B, v.davis_C,
          float(v.can_traction), v.F_trac_max_N, v.F_brk_max_N] for v in sc.vehicles]
    )
    edge_static = np.array(
        [[c.L0_m, c.slack_half_m, c.k_draft, c.c_draft, c.k_buff, c.c_buff] for c in sc.couplers]
    )
    np.savez_compressed(
        OUT / f"{name}.npz",
        t=res.t, y=res.sol.y, y0=sc.y0,
        H_hist=res.H_hist, E_hist=res.E_hist,
        node_static=node_static, edge_static=edge_static,
        u_trac=u_trac, u_brk=u_brk,
        route_s=sc.route.s_nodes_m, route_sin_theta=sc.route.sin_theta_nodes,
        route_kappa=sc.route.kappa_nodes, route_vmax=sc.route.v_max_nodes,
        tau_brk_s=sc.tau_brk_s, tau_trac_s=sc.tau_trac_s, p_max_w=sc.p_max_w,
        k_curv_scale=sc.k_curv_scale, v_eps=C.DEFAULT_V_EPS,
        v_brake_eps=C.DEFAULT_V_BRAKE_EPS,
        brake_opposes_motion=sc.brake_opposes_motion,
        N=n, duration_s=DURATION_S, nfev=res.sol.nfev,
        ref_rtol=sc.rtol, ref_atol=sc.atol, ref_method=sc.method,
    )
    v = res.H_hist[:, :, 1]
    print(f"  {name:28s} N={n} T={res.t.size} nfev={res.sol.nfev:8d} "
          f"v[{v.min():7.2f},{v.max():7.2f}] m/s {_w:6.1f}s")


# A consist held at a standstill under brake is genuinely hard for an adaptive
# solver: at v = 0 the tanh sign blend has slope 1/v_brake_eps, so error control
# chases vanishing oscillations about the equilibrium and the step size collapses.
# ``regime_emergency`` needed 431,018 RHS evaluations for 40 s of simulated time
# against ~2,700 for cruise, and LSODA did no better -- this is a discontinuity,
# not stiffness. Those references are therefore generated at the simulator's own
# default tolerance rather than a tightened one. Each fixture records
# ``ref_rtol`` / ``ref_atol`` / ``ref_method`` so a comparison can be scaled to
# the reference's actual accuracy.
#
# Worth noting for the port: a fixed-step integrator has no error control to
# collapse, so it handles this state at the same cost as any other. Held-brake
# standstill is common in real operation and will be common in the dataset.
STANDSTILL = dict(rtol=C.DEFAULT_RTOL, atol=C.DEFAULT_ATOL, method="RK45")


def main(only: str | None = None) -> None:
    print(f"writing fixtures to {OUT}")
    route_rolling, route_flat, route_desc = make_route("rolling"), make_route("flat"), make_route("descent")

    cases = {}
    for k, regime in enumerate(list(Regime)):
        prof = generate_regime_profile(regime, seed=1000 + k, duration_s=DURATION_S)
        v0 = 0.0 if regime is Regime.STARTUP else 14.0
        route = route_desc if regime is Regime.DYNAMIC_BRAKE_DESCENT else route_rolling
        extra = STANDSTILL if regime is Regime.EMERGENCY else {}
        cases[f"regime_{regime.value}"] = lambda p=prof, r=route, v=v0, e=extra: build_scenario(
            p, r, v0=v, x_lead=3000.0, k_curv_scale=0.0, **e)

    cases["adversarial_slack_cycling"] = lambda: build_scenario(
        slack_cycling_profile(), route_rolling, v0=8.0, x_lead=3000.0, k_curv_scale=0.0)
    cases["adversarial_brake_to_stop"] = lambda: build_scenario(
        brake_to_stop_profile(), route_flat, v0=18.0, x_lead=3000.0,
        k_curv_scale=0.0, **STANDSTILL)
    cases["curvature_enabled"] = lambda: build_scenario(
        generate_regime_profile(Regime.CRUISE, seed=77, duration_s=DURATION_S),
        route_rolling, v0=14.0, x_lead=3000.0, k_curv_scale=0.5)
    cases["brake_flag_disabled"] = lambda: _disable_flag(build_scenario(
        brake_to_stop_profile(), route_flat, v0=18.0, x_lead=3000.0,
        k_curv_scale=0.0, **STANDSTILL))

    for name, build in cases.items():
        if only and only not in name:
            continue
        write_case(name, build())
    print("done")


def _disable_flag(sc: ExtendedTrainScenario) -> ExtendedTrainScenario:
    sc.brake_opposes_motion = False
    return sc


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
