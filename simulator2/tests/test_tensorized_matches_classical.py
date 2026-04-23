"""Tensorized RHS and trajectories match classical ``train_rhs_extended``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from simulator2.constants import DEFAULT_P_MAX_W, DEFAULT_V_EPS
from simulator2.forces import traction_ramp
from simulator2.params import CouplerParameters, VehicleParameters
from simulator2.route import RouteProfile
from simulator2.rhs import train_rhs_extended, train_rhs_tensorized
from simulator2.scenarios import make_simple_train


def _rhs_extended_pack(vehicles, couplers, route, u_trac, u_brk, kc, tb, tt, pw):
    def f(t, y):
        return train_rhs_extended(
            t, y, vehicles, couplers, route, u_trac, u_brk, kc, tb, tt, pw
        )

    return f


def _rhs_tensor_pack(vehicles, couplers, route, u_trac, u_brk, kc, tb, tt, pw):
    def f(t, y):
        return train_rhs_tensorized(
            t, y, vehicles, couplers, route, u_trac, u_brk, kc, tb, tt, pw
        )

    return f


def test_rhs_vectors_agree_random_state():
    rng = np.random.default_rng(42)
    vehicles, couplers = make_simple_train(4)
    n = len(vehicles)
    route = RouteProfile(
        np.array([0.0, 1e5]),
        np.array([0.0, 0.05]),
        kappa_nodes=np.array([0.0, 0.001]),
    )
    y = rng.standard_normal(4 * n) * 0.1 + 1.0
    y[n : 2 * n] *= 5.0

    def u_trac(t, i):
        return traction_ramp(t) if i == 0 else 0.0

    def u_brk(t, i):
        return 50_000.0 if t > 10.0 else 0.0

    t = 3.0
    a = train_rhs_extended(
        t,
        y,
        vehicles,
        couplers,
        route,
        u_trac,
        u_brk,
        0.01,
        2.5,
        4.0,
        DEFAULT_P_MAX_W,
        DEFAULT_V_EPS,
    )
    b = train_rhs_tensorized(
        t,
        y,
        vehicles,
        couplers,
        route,
        u_trac,
        u_brk,
        0.01,
        2.5,
        4.0,
        DEFAULT_P_MAX_W,
        DEFAULT_V_EPS,
    )
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_rollout_lead_rear_and_coupler_force():
    vehicles, couplers = make_simple_train(6)
    n = len(vehicles)
    route = RouteProfile(
        np.array([0.0, 50_000.0]),
        np.array([0.0, 0.0]),
        None,
    )
    x0 = np.array([1000.0 - i * couplers[0].L0_m for i in range(n)], dtype=float)
    v0 = np.zeros(n)
    z0 = np.zeros(n)
    y0 = np.concatenate([x0, v0, z0, z0])

    def u_trac(t, i):
        return traction_ramp(t, t0=5.0, t1=25.0, F_max=200_000.0) if i == 0 else 0.0

    def u_brk(t, i):
        return 80_000.0 if t > 60.0 else 0.0

    tb, tt, pw = 3.0, 5.0, DEFAULT_P_MAX_W
    kc = 0.0
    t_span = (0.0, 40.0)
    te = np.linspace(*t_span, 201)

    sol_c = solve_ivp(
        _rhs_extended_pack(
            vehicles, couplers, route, u_trac, u_brk, kc, tb, tt, pw
        ),
        t_span,
        y0,
        t_eval=te,
        rtol=1e-6,
        atol=1e-8,
    )
    sol_t = solve_ivp(
        _rhs_tensor_pack(vehicles, couplers, route, u_trac, u_brk, kc, tb, tt, pw),
        t_span,
        y0,
        t_eval=te,
        rtol=1e-6,
        atol=1e-8,
    )
    assert sol_c.success and sol_t.success
    np.testing.assert_allclose(sol_c.y, sol_t.y, rtol=1e-6, atol=1e-6)

    from simulator2.forces import coupler_force_slack_asymmetric

    for k in (0, len(te) // 2, len(te) - 1):
        x = sol_c.y[0:n, k]
        v = sol_c.y[n : 2 * n, k]
        c0 = couplers[0]
        dlt = (x[0] - x[1]) - c0.L0_m
        dd = v[0] - v[1]
        f0 = float(
            coupler_force_slack_asymmetric(
                np.array([dlt]),
                np.array([dd]),
                c0.slack_half_m,
                c0.k_draft,
                c0.c_draft,
                c0.k_buff,
                c0.c_buff,
            )[0]
        )
        assert np.isfinite(f0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
