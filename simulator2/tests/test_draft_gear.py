"""Benchmark draft gear: the digitized table and the force law.

Properties that hold whatever closing-velocity switch is used. The switch
itself is an open decision (``SIMULATOR_VALIDATION_NOTE.md``, Tier 3): at the
production step it is stiff enough to manufacture dissipation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simulator2.draft_gear import (
    N_GEARS,
    draft_gear_force,
    draft_gear_force_torch,
    load_table,
)

S = 0.005   # benchmark coupler: 10 mm total slack


@pytest.fixture(scope="module")
def table() -> np.ndarray:
    return load_table()


def test_table_matches_the_published_figure(table) -> None:
    """Spot values read off Spiryagin et al. (2017) Fig. 3."""
    g_mm, f_l, f_u = table[:, 0] * 1e3, table[:, 1] / 1e3, table[:, 2] / 1e3
    assert g_mm[-1] == pytest.approx(82.4, abs=0.5)        # full travel
    assert f_l[-1] == pytest.approx(2100, rel=0.03)        # peak
    for mm, kn in [(15, 310), (30, 570), (45, 850), (60, 990)]:
        assert np.interp(mm, g_mm, f_l) == pytest.approx(kn, rel=0.06)
    assert np.interp(30, g_mm, f_u) == pytest.approx(85, rel=0.15)
    assert np.all(f_u <= f_l + 1e-9)
    assert np.all(np.diff(g_mm) > 0)


def test_curves_bound_the_force(table) -> None:
    """Closing fast follows loading, opening fast follows unloading."""
    g = np.linspace(0.001, table[-1, 0] - 1e-4, 50)
    delta = S + N_GEARS * g
    f_l = np.interp(g, table[:, 0], table[:, 1])
    f_u = np.interp(g, table[:, 0], table[:, 2])
    assert np.allclose(draft_gear_force(delta, np.full_like(g, 1.0), S, table), f_l, rtol=1e-6)
    assert np.allclose(draft_gear_force(delta, np.full_like(g, -1.0), S, table), f_u, rtol=1e-6)


def test_odd_in_draft_and_buff(table) -> None:
    d = np.linspace(-0.2, 0.2, 401)
    dd = np.linspace(-0.05, 0.05, 401)
    assert np.allclose(draft_gear_force(-d, -dd, S, table),
                       -draft_gear_force(d, dd, S, table))


def test_no_force_in_slack_and_no_jump_at_its_edge(table) -> None:
    """The Kelvin-Voigt coupler jumps by c * delta_dot on contact; this must
    not, at any closing speed."""
    for v in (-0.5, 0.0, 0.5):
        assert draft_gear_force(np.array([0.0, S * 0.99]), np.full(2, v), S, table) \
            == pytest.approx(0.0)
        edge = draft_gear_force(np.array([S + 1e-9]), np.array([v]), S, table)[0]
        assert abs(edge) < 1.0   # N


def test_lock_up_beyond_travel(table) -> None:
    solid = S + N_GEARS * table[-1, 0]
    f = draft_gear_force(np.array([solid, solid + 1e-3]), np.full(2, 1.0), S, table,
                         k_lock=5e8)
    assert f[1] - f[0] == pytest.approx(5e8 * 1e-3, rel=1e-6)


def test_torch_matches_numpy_and_is_differentiable(table) -> None:
    rng = np.random.default_rng(0)
    d = rng.uniform(-0.2, 0.2, 5000)
    dd = rng.uniform(-0.05, 0.05, 5000)
    ref = draft_gear_force(d, dd, S, table)
    td = torch.tensor(d, requires_grad=True)
    tdd = torch.tensor(dd, requires_grad=True)
    out = draft_gear_force_torch(td, tdd, torch.tensor(S, dtype=torch.float64),
                                 torch.tensor(table))
    assert np.abs(out.detach().numpy() - ref).max() < 1e-6
    out.sum().backward()
    assert torch.isfinite(td.grad).all() and torch.isfinite(tdd.grad).all()


# ---------------------------------------------------------------------------
# the stateful gear (the model the RHS uses)
# ---------------------------------------------------------------------------

from simulator2.draft_gear import (  # noqa: E402
    DEFAULT_K_TRANSITION,
    draft_gear_stateful_torch,
)
from simulator2.verification import (  # noqa: E402
    Case,
    energy_audit,
    momentum_residual,
    realistic_case,
    rk4_fixed,
)


def _drive(table, path: np.ndarray, h: float = 1e-3, z0: float = 0.0):
    """Push a connection along a prescribed extension path; return forces.
    ``path`` is sampled every ``h`` seconds; the state is advanced by
    explicit Euler, which is exact enough at this resolution."""
    tab = torch.tensor(table)
    s = torch.tensor(S, dtype=torch.float64)
    z = torch.tensor([z0], dtype=torch.float64)
    out = []
    for k in range(len(path) - 1):
        d = torch.tensor([path[k]], dtype=torch.float64)
        dd = torch.tensor([(path[k + 1] - path[k]) / h], dtype=torch.float64)
        f, dz = draft_gear_stateful_torch(d, dd, z, s, tab)
        out.append(float(f[0]))
        z = z + h * dz
    return np.asarray(out)


def test_full_cycle_traces_both_curves_and_its_area(table) -> None:
    """Close to full travel and back, slowly. Out: the loading curve. Back:
    the elastic line of slope ``k_t`` from the peak until it meets the
    unloading curve, then that curve. The work absorbed is the area of that
    loop, which is less than the full area between the published curves by
    the corner the elastic line cuts off (``k_t`` is softer than the
    near-vertical drop in the drop-hammer plot)."""
    e_max = N_GEARS * table[-1, 0]
    up = np.linspace(S, S + e_max, 20_001)
    path = np.concatenate([up, up[::-1][1:]])
    f = _drive(table, path)
    g = (path[:-1] - S) / N_GEARS
    closing = np.arange(len(f)) < len(up) - 1
    f_l = np.interp(g, table[:, 0], table[:, 1])
    f_u = np.interp(g, table[:, 0], table[:, 2])
    f_back = np.maximum(f_u, table[-1, 1] - DEFAULT_K_TRANSITION * (S + e_max - path[:-1]))
    mid = (g > 0.02 * table[-1, 0]) & (g < 0.98 * table[-1, 0])
    tol = 0.01 * table[-1, 1]
    assert np.abs(f - f_l)[closing & mid].max() < tol
    assert np.abs(f - f_back)[~closing & mid].max() < tol

    work = np.sum(f * np.diff(path))
    model_loop = np.sum(np.where(closing, f_l, -f_back) * np.abs(np.diff(path)))
    published_loop = N_GEARS * np.trapezoid(table[:, 1] - table[:, 2], table[:, 0])
    assert work == pytest.approx(model_loop, rel=0.01)
    assert work < published_loop


def test_gear_holds_its_force_at_rest(table) -> None:
    """Friction gear stopped part-way keeps the force it reached. The
    velocity-switch model would drop to the midpoint of its curves."""
    e_half = N_GEARS * 0.5 * table[-1, 0]
    path = np.concatenate([np.linspace(S, S + e_half, 5_001), np.full(5_000, S + e_half)])
    f = _drive(table, path)
    f_l = np.interp(0.5 * table[-1, 0], table[:, 0], table[:, 1])
    assert f[-1] == pytest.approx(f_l, rel=0.01)
    assert f[-1] == pytest.approx(f[5_010], rel=1e-9)


def test_small_reversal_inside_the_envelope_is_elastic(table) -> None:
    """Back off less than the stick band and return: the force comes back
    to where it was, and the excursion absorbs no net work."""
    e1 = N_GEARS * 0.4 * table[-1, 0]
    back = 0.2 * (np.interp(0.4 * table[-1, 0], table[:, 0], table[:, 1])
                  - np.interp(0.4 * table[-1, 0], table[:, 0], table[:, 2])) / DEFAULT_K_TRANSITION
    load = np.linspace(S, S + e1, 5_001)
    wiggle = np.concatenate([np.linspace(S + e1, S + e1 - back, 501)[1:],
                             np.linspace(S + e1 - back, S + e1, 501)[1:]])
    path = np.concatenate([load, wiggle])
    f = _drive(table, path)
    i0 = len(load) - 1
    assert f[-1] == pytest.approx(f[i0], rel=1e-3)
    assert abs(np.sum(f[i0:] * np.diff(path[i0:]))) < 1e-3 * abs(np.sum(f[:i0] * np.diff(path[:i0 + 1])))


def test_force_never_leaves_the_envelope(table) -> None:
    rng = np.random.default_rng(1)
    path = S + np.cumsum(rng.normal(0, 2e-4, 40_000))
    path = np.clip(path, -0.2, 0.2)
    f = _drive(table, path)
    e = np.where(path > S, path - S, np.where(path < -S, path + S, 0.0))[:-1]
    g = np.minimum(np.abs(e) / N_GEARS, table[-1, 0])
    f_l = np.interp(g, table[:, 0], table[:, 1])
    f_u = np.interp(g, table[:, 0], table[:, 2])
    mag = np.abs(f)
    inside = np.abs(e) < N_GEARS * table[-1, 0]
    assert np.all(mag[inside] <= f_l[inside] + 1e-6)
    assert np.all(mag[inside] >= f_u[inside] - 1e-6)
    assert np.all(np.sign(f[np.abs(f) > 1]) == np.sign(e[np.abs(f) > 1]))


@pytest.fixture(scope="module")
def gear_run():
    case = realistic_case(duration_s=60.0, coupler_model="draft_gear")
    batch = case.batch()
    t, y = rk4_fixed(batch, 60.0, 0.02, every=50)
    return batch, t, y


def test_rhs_momentum_with_draft_gear(gear_run) -> None:
    batch, t, y = gear_run
    worst = max(momentum_residual(tt, torch.as_tensor(yy[None]), batch)
                for tt, yy in zip(t, y))
    assert worst < 1e-12


def test_rhs_energy_balance_with_draft_gear() -> None:
    """At the production step, unlike the velocity-switch model, which made
    up ~30x the true dissipation (validation note, 2026-09-26)."""
    batch = realistic_case(duration_s=30.0, coupler_model="draft_gear").batch()
    fine = energy_audit(batch, 30.0, 0.005)
    prod = energy_audit(batch, 30.0, 0.02)
    assert prod["rel_residual"] < 1e-6
    assert prod["work_damper"] == pytest.approx(fine["work_damper"], rel=0.02)
    assert prod["work_damper"] > 0.0


def test_stable_on_empty_wagons_at_production_step(table) -> None:
    """The lightest stock in the corpus (26 t empties) hit hard enough to
    close the gears: RK4 at dt = 0.02 must stay bounded with the default
    transitional stiffness."""
    n, m = 12, 26_000.0
    case = Case(mass=np.full(n, m), x0=-15.0 * np.arange(n),
                v0=np.where(np.arange(n) < 6, 0.0, 3.0), L0=15.0, slack=S,
                coupler_model="draft_gear", draft_gear_table=table)
    _, y = rk4_fixed(case.batch(), 20.0, 0.02, every=10)
    assert np.all(np.isfinite(y))
    ke0 = 0.5 * m * 6 * 3.0 ** 2
    ke = 0.5 * m * (y[:, n:2 * n] ** 2).sum(1)
    assert ke.max() <= ke0 * (1 + 1e-6)
