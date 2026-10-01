"""The LTD benchmark harness: inputs against the questions paper, and the
notch traction and dynamic brake law against its published curves.

Spiryagin, Wu & Cole (2017) with the 2018 corrigendum. Each expected number
below is printed in the paper or read off its figures; nothing is taken from
our own output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simulator2 import ltd_benchmark as lb
from simulator2.torch_rhs import (
    LocoParam,
    _curvature_force,
    _db_at_notch,
    _notch_force,
    _traction_at_notch,
)

KMH = 1 / 3.6


def _lp(loco: str) -> torch.Tensor:
    return torch.tensor(lb.VEHICLE_TYPES[loco].loco, dtype=torch.float64)


def _eps() -> torch.Tensor:
    return torch.tensor(0.1, dtype=torch.float64)


def _db(loco: str, notch: int, kmh) -> np.ndarray:
    v = torch.as_tensor(np.atleast_1d(kmh) * KMH, dtype=torch.float64)
    n = torch.full_like(v, float(notch))
    return _db_at_notch(n, v, _lp(loco)[None].expand(len(v), -1), _eps()).numpy()


def _te(loco: str, notch: int, kmh) -> np.ndarray:
    v = torch.as_tensor(np.atleast_1d(kmh) * KMH, dtype=torch.float64)
    n = torch.full_like(v, float(notch))
    return _traction_at_notch(n, v, _lp(loco)[None].expand(len(v), -1), _eps()).numpy()


# -- consists -----------------------------------------------------------------

@pytest.mark.parametrize("train, mass_t, length_m", [
    (1, 6668, 795.90), (2, 13336, 1591.80), (3, 38985, 2713.05), (4, 38985, 2713.05),
])
def test_consist_totals_match_section_3_1(train, mass_t, length_m) -> None:
    c = lb.build_consist(train)
    assert c.mass_kg.sum() / 1e3 == pytest.approx(mass_t)
    assert c.length_m == pytest.approx(length_m)


def test_train_4_follows_the_corrigendum() -> None:
    c = lb.build_consist(4)
    assert [vt.name for vt in c.types if vt.is_loco] == ["loco2"] * 3


@pytest.mark.parametrize("train, remote", [(1, []), (2, [53, 54]), (3, []), (4, [123])])
def test_remote_units(train, remote) -> None:
    assert list(np.where(lb.build_consist(train).remote)[0] + 1) == remote


def test_wagons_run_in_drawbar_pairs() -> None:
    c = lb.build_consist(4)
    names = [vt.name for vt in c.types]
    for j, is_bar in enumerate(c.drawbar):
        front, rear = names[j], names[j + 1]
        if is_bar:
            assert front == rear == "wagon2"
    # 240 wagons make 120 pairs; every other connection is a coupler.
    assert c.drawbar.sum() == 120
    assert (~c.drawbar).sum() == c.n - 1 - 120


def test_report_positions_are_connections() -> None:
    for train, pos in lb.REPORT_POSITIONS.items():
        assert max(pos) <= lb.build_consist(train).n - 1


# -- resistance -----------------------------------------------------------------

@pytest.mark.parametrize("train", [1, 3])
def test_davis_conversion_reproduces_equation_1(train) -> None:
    c = lb.build_consist(train)
    abc = lb.davis_coefficients(c)
    for V in (0.0, 20.0, 60.0, 110.0):
        v = V * KMH
        for i in (0, 1, c.n - 1):
            vt = c.types[i]
            q = 3.2 if i == 0 else 1.0
            eq1 = q * vt.mass_t * (2.943 + 89.2 / vt.axle_load_t + 0.0306 * V
                                   + 0.122 * V ** 2 / vt.mass_t)
            assert abc[i, 0] + abc[i, 1] * v + abc[i, 2] * v * v == pytest.approx(eq1)


@pytest.mark.parametrize("radius", [200.0, 600.0, 1000.0])
def test_curving_resistance_is_equation_2(radius) -> None:
    mass = torch.tensor([[160e3]], dtype=torch.float64)
    one = torch.ones(1, 1, dtype=torch.float64)
    f = _curvature_force(one * lb.k_curv_scale_linear(), mass, one,
                         one / radius, one, "linear")
    assert f.item() == pytest.approx(160 * 6116 / radius, rel=1e-12)


# -- track ----------------------------------------------------------------------

def test_gradient_table_shape() -> None:
    g = lb.load_gradient()
    assert g.shape == (86, 2)
    assert g[0, 0] == 0 and g[-1, 0] == 50000
    assert np.all(np.diff(g[:, 0]) > 0)
    assert np.abs(g[:, 1]).max() < 1.0      # percent; the steepest is 0.91 %


def test_elevation_reproduces_figure_6a() -> None:
    """Read off Fig. 6a: valley ~94 m near 4 km, summit ~145 m near 25 km,
    ~113 m from 40 km on. Reading the table as steps gives these; linear
    interpolation would miss the end by ~5 m."""
    z = lb.elevation_m(np.array([0.0, 4080.0, 24690.0, 30000.0, 50000.0]))
    assert z == pytest.approx([105, 94, 145, 123, 113], abs=1.5)


def test_curves_follow_figure_6b() -> None:
    k = lb.curvature_at(np.array([24200.0, 28200.0, 4000.0, 4025.0, 4400.0, 4500.0]))
    assert k[0] == pytest.approx(1 / 200)
    assert k[1] == pytest.approx(-1 / 200)
    assert k[2] == 0.0 and k[5] == 0.0
    assert k[3] == pytest.approx(0.5 / 1000)        # halfway up the transition
    assert k[4] == 0.0                               # 50 + 300 + 50 m long


def test_route_holds_every_knot() -> None:
    s, sin_t, kap = lb.build_route()
    assert np.all(np.diff(s) > 0)
    g = lb.load_gradient()
    idx = np.searchsorted(s, g[:, 0])
    assert np.allclose(s[idx], g[:, 0])
    assert np.allclose(sin_t[idx], np.sin(np.arctan(g[:, 1] / 100)))


# -- controls -------------------------------------------------------------------

@pytest.mark.parametrize("train, n_rows, t_last", [(1, 132, 3096), (2, 110, 2394),
                                                   (3, 30, 5000), (4, 30, 5000)])
def test_appendix_2_as_printed(train, n_rows, t_last) -> None:
    raw = np.loadtxt(lb.DATA / f"controls_train{train}.csv", delimiter=",", skiprows=1)
    assert raw.shape == (n_rows, 2)
    assert raw[-1, 0] == t_last
    # One notch at a time. Trains 3 and 4 close with a row at 5000 s that
    # only marks the end of the run, holding notch 8.
    steps = np.abs(np.diff(raw[:, 1]))
    if t_last == 5000:
        assert steps[-1] == 0
        steps = steps[:-1]
    assert np.all(steps == 1)


@pytest.mark.parametrize("train, start, period", [(1, 2688.0, 408.0), (2, 2276.0, 118.0)])
def test_cycles_unroll_to_5000_s(train, start, period) -> None:
    ctl = lb.load_controls(train)
    assert ctl[-1, 0] < 5000.0 <= ctl[-1, 0] + period
    assert np.all(np.diff(ctl[:, 0]) > 0)
    assert np.all(np.abs(np.diff(ctl[:, 1])) == 1)
    t = np.arange(start, 5000.0, 0.5)
    assert np.array_equal(lb.notch_at(t, ctl), lb.notch_at(start + (t - start) % period, ctl))


def test_remote_units_lag_the_lead_by_3_s() -> None:
    c = lb.build_consist(2)
    t, u = lb.command_grid(2, c)
    ctl = lb.load_controls(2)
    probe = np.arange(0.0, 4990.0, 0.37)
    lead = np.interp(probe, t, u[:, 0])
    remote = np.interp(probe, t, u[:, 52])
    assert np.array_equal(lead, lb.notch_at(probe, ctl))
    assert np.array_equal(remote, lb.notch_at(probe - 3.0, ctl))
    assert not u[:, 2:52].any()                      # wagons get nothing


# -- traction and dynamic brake --------------------------------------------------

def test_traction_follows_table_4() -> None:
    # Type 1, notch 8: 600 kN until (P/Te) = 5.35 m/s = 19.2 km/h, then P/v.
    assert _te("loco1", 8, [5.0, 19.0]) == pytest.approx([600e3, 600e3])
    assert _te("loco1", 8, 60.0)[0] == pytest.approx(3207e3 / (60 * KMH))
    # Notch 4: half the force, a quarter of the power.
    assert _te("loco1", 4, 5.0)[0] == pytest.approx(300e3)
    assert _te("loco1", 4, 60.0)[0] == pytest.approx(0.25 * 3207e3 / (60 * KMH))
    assert _te("loco2", 8, 5.0)[0] == pytest.approx(850e3)
    assert _te("loco1", 0, 30.0)[0] == 0.0


@pytest.mark.parametrize("loco, db_max, p_db", [("loco1", 325e3, 3600e3), ("loco2", 472e3, 5500e3)])
def test_db_notches_7_and_8_follow_figures_4b_5b(loco, db_max, p_db) -> None:
    for n in (7, 8):
        assert _db(loco, n, 2.5)[0] == pytest.approx(db_max / 2, rel=0.01)  # on the ramp
        assert _db(loco, n, 30.0)[0] == pytest.approx(db_max - 10 * 30 * KMH)
        assert _db(loco, n, 44.0)[0] == pytest.approx(p_db / (44 * KMH))
    # The step at 43 km/h the figures show.
    assert _db(loco, 8, 42.9)[0] - _db(loco, 8, 43.1)[0] > 0.02 * db_max


@pytest.mark.parametrize("notch, crossover_kmh", [(6, 49.8), (5, 59.8), (4, 74.8)])
def test_db_partial_notch_plateaus_and_crossovers(notch, crossover_kmh) -> None:
    """Fig. 4b, type 1: notch n levels at 325 n/7.5 kN and meets the power
    hyperbola where 3600 kW / level = v."""
    level = 325e3 * notch / 7.5
    assert _db("loco1", notch, 20.0)[0] == pytest.approx(level, rel=1e-3)
    assert _db("loco1", notch, crossover_kmh - 3)[0] == pytest.approx(level, rel=2e-3)
    assert _db("loco1", notch, crossover_kmh + 5)[0] == pytest.approx(
        3600e3 / ((crossover_kmh + 5) * KMH))


@pytest.mark.parametrize("notch, kn", [(1, 37.4), (2, 74.7), (3, 112.1)])
def test_db_low_notches_stay_flat_to_115_kmh(notch, kn) -> None:
    f = _db("loco1", notch, [10.0, 60.0, 110.0]) / 1e3
    assert f == pytest.approx([kn] * 3, abs=0.5)


def test_db_is_monotone_in_notch() -> None:
    v = np.linspace(0.5, 120.0, 240)
    for loco in ("loco1", "loco2"):
        curves = np.array([_db(loco, n, v) for n in range(9)])
        assert np.all(np.diff(curves, axis=0) >= -1e-6)


def test_fractional_notch_blends_neighbours() -> None:
    lp = _lp("loco1")[None]
    v = torch.tensor([30 * KMH], dtype=torch.float64)

    def f(z):
        return _notch_force(torch.tensor([z], dtype=torch.float64), v, lp, _eps()).item()

    assert f(4.25) == pytest.approx(0.75 * f(4.0) + 0.25 * f(5.0))
    assert f(-6.5) == pytest.approx(0.5 * (f(-6.0) + f(-7.0)))
    assert f(-2.0) < 0 < f(2.0)
    # Continuous through every integer notch.
    for z in np.arange(-8.0, 8.5, 1.0):
        assert f(z - 1e-9) == pytest.approx(f(z + 1e-9), rel=1e-6, abs=1e-3)


def test_dynamic_brake_opposes_motion() -> None:
    lp = _lp("loco1")[None]
    z = torch.tensor([-8.0], dtype=torch.float64)
    fwd = _notch_force(z, torch.tensor([10.0], dtype=torch.float64), lp, _eps()).item()
    back = _notch_force(z, torch.tensor([-10.0], dtype=torch.float64), lp, _eps()).item()
    assert fwd == pytest.approx(-back) and fwd < 0


# -- the assembled batch ----------------------------------------------------------

def test_short_run_of_train_1() -> None:
    """60 s from rest: the train starts on notch 1 at 2 s, the gears load
    from the front, and every gear force stays inside its envelope."""
    batch, consists = lb.build_batch((1,), t_end=60.0)
    out = lb.run(batch, consists, t_end=60.0, chunk_s=20.0)[0]
    v = out["lead_speed_mps"]
    assert v[:200].max() < 1e-6                     # notch 0 until 2 s
    assert v[-1] > 0.5
    # The lead rocks back a fraction of a micron as the first gears take up
    # load; it never falls behind its furthest point by anything material.
    x = out["lead_front_m"]
    assert np.all(np.maximum.accumulate(x) - x < 1e-3)
    f = out["force_n"]
    assert f[-1, 0] > f[-1, -1] > 0                 # tension falls along the train
    assert np.all(np.abs(out["gear_defl_m"]) <= 0.0825)
    assert np.all(out["max_tension_n"] >= -1e-6)
