"""The update rule must be an exact inverse of the targets it is trained on.

If it is not, the model is chasing a target that does not reconstruct the data,
and every downstream metric is measured against the wrong thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from surrogate.step import (actuator_step, advance, coupler_state,
                            edge_targets, node_targets, reconstruct_x)

DATA = _REPO_ROOT / "data" / "v2"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="data/v2 not built")


@pytest.fixture(scope="module")
def scenario():
    """A deterministically chosen *moving* train scenario.

    Picking the first scenario by id lands on a near-stationary run -- the
    randomizer starts some consists at rest on flat track, and 22% of the corpus
    reverses at some point. Those are legitimate data but useless for testing
    properties of motion, so the fixture filters on mean speed first.
    """
    idx = pd.read_parquet(DATA / "index.parquet")
    moving = idx[(idx.split == "train") & (idx.v_mean_mps > 5.0)]
    assert not moving.empty
    r = moving.sort_values("scenario_id").iloc[0]
    with np.load(DATA / "scenarios" / r.route_id / f"{r.scenario_id}.npz") as f:
        d = {k: f[k] for k in f.files}
    d["_h"] = float(d["t"][1] - d["t"][0])
    return d


def _tensors(d, k, n_steps=64):
    st = torch.tensor(d["state"][k : k + n_steps + 1], dtype=torch.float64)
    u = torch.tensor(
        np.stack([d["u_trac"], d["u_brk"]], -1)[k : k + n_steps + 1], dtype=torch.float64)
    return st, u


def test_targets_advance_round_trip(scenario):
    """advance(targets(...)) reproduces the next state, to float64 precision.

    Velocity and coupler stretch are exact inverses by construction. Absolute
    `x` is rebuilt from the centroid plus the stretches, so it is only as good
    as the float32 positions it is compared against -- hence the looser bound.
    """
    d = scenario
    h = d["_h"]
    st, u = _tensors(d, 100)
    ed = torch.tensor(d["edge_dynamic"][100:165], dtype=torch.float64)
    l0 = torch.tensor(d["edge_static"][:, 0], dtype=torch.float64)
    s0, s1 = st[:-1], st[1:]
    dd0, dd1 = ed[:-1, :, 0], ed[1:, :, 0]

    abar = node_targets(s0, s1, h)
    corr = edge_targets(dd0, dd1, s0, s1, h)
    got, delta1 = advance(s0, dd0, abar, corr, u[:-1], u[1:], l0,
                          tau_brk=float(d["tau_brk_s"]),
                          tau_trac=float(d["tau_trac_s"]), h=h)

    assert torch.allclose(got[..., 1], s1[..., 1], atol=1e-9)   # v
    assert torch.allclose(delta1, dd1, atol=1e-9)               # coupler stretch
    # x: centroid trapezoid + exact shape, against float32-stored positions
    assert (got[..., 0] - s1[..., 0]).abs().max() < 5e-3


def test_reconstruct_x_inverts_the_coupler_geometry(scenario):
    """Positions rebuilt from stretches must match, up to the centroid anchor."""
    d = scenario
    st = torch.tensor(d["state"][:50, :, 0], dtype=torch.float64)
    ed = torch.tensor(d["edge_dynamic"][:50, :, 0], dtype=torch.float64)
    l0 = torch.tensor(d["edge_static"][:, 0], dtype=torch.float64)
    got = reconstruct_x(st.mean(dim=-1), ed, l0)
    assert (got - st).abs().max() < 5e-3


def test_edge_target_carries_real_signal(scenario):
    """The shape correction must sit well above float32, unlike its node form.

    This is the whole reason the correction head lives on the edges. Compare
    the same correction taken in absolute coordinates, where `x` ~ 15 km puts
    the float32 step at ~1 mm and swamps it.
    """
    d = scenario
    h = d["_h"]
    st = torch.tensor(d["state"], dtype=torch.float64)
    ed = torch.tensor(d["edge_dynamic"][:, :, 0], dtype=torch.float64)
    s0, s1 = st[:-1], st[1:]

    corr = edge_targets(ed[:-1], ed[1:], s0, s1, h)
    quantum_delta = float(np.spacing(np.float32(ed.abs().max())))
    assert float(corr.std()) > 1000.0 * quantum_delta

    v0, v1 = s0[..., 1], s1[..., 1]
    dx_corr = s1[..., 0] - (s0[..., 0] + h * 0.5 * (v0 + v1))
    quantum_x = float(np.spacing(np.float32(st[..., 0].abs().max())))
    assert float(dx_corr.std()) < 5.0 * quantum_x


def test_actuator_step_matches_corpus(scenario):
    """The analytic lag reproduces stored z to storage precision, not to a tolerance."""
    d = scenario
    h = d["_h"]
    for ch, u_key, tau_key in ((2, "u_brk", "tau_brk_s"), (3, "u_trac", "tau_trac_s")):
        z = torch.tensor(d["state"][:, :, ch], dtype=torch.float64)
        u = torch.tensor(d[u_key], dtype=torch.float64)
        got = actuator_step(z[:-1], u[:-1], u[1:], float(d[tau_key]), h)
        err = (got - z[1:]).abs().max()
        scale = z.abs().max().clamp_min(1.0)
        assert err / scale < 1e-6, f"channel {ch}: {err=} {scale=}"


def test_actuator_step_beats_euler(scenario):
    """Worth doing analytically: the lag is not so slow that Euler would do."""
    d = scenario
    h = d["_h"]
    z = torch.tensor(d["state"][:, :, 2], dtype=torch.float64)
    u = torch.tensor(d["u_brk"], dtype=torch.float64)
    tau = float(d["tau_brk_s"])
    exact = (actuator_step(z[:-1], u[:-1], u[1:], tau, h) - z[1:]).abs().max()
    euler = ((z[:-1] + h * (u[:-1] - z[:-1]) / tau) - z[1:]).abs().max()
    assert exact * 10 < euler


def test_coupler_state_is_float32_position_limited(scenario):
    """Deriving `delta` from stored `x` is limited by float32, not by the formula.

    `x` reaches ~11 km, where float32 spacing is ~1 mm, while `delta` lives on a
    +-66 mm scale against a 20 mm slack. So a delta differenced out of stored
    positions carries ~1 mm of quantization that the corpus's own
    `edge_dynamic` does not -- it was computed in float64 before being stored.

    This test pins the size of that floor rather than pretending it is absent.
    `delta_dot` is unaffected: velocities are O(10) m/s, so the same float32
    buys eight more digits.
    """
    d = scenario
    st = torch.tensor(d["state"][:200], dtype=torch.float64)
    es = torch.tensor(d["edge_static"], dtype=torch.float64)
    got = coupler_state(st, es)
    want = torch.tensor(d["edge_dynamic"][:200], dtype=torch.float64)

    x_max = float(torch.tensor(d["state"][:, :, 0]).abs().max())
    floor = 4.0 * float(np.spacing(np.float32(x_max)))
    delta_err = (got[..., 0] - want[..., 0]).abs().max()
    assert delta_err < floor, f"{delta_err=} exceeds the float32 floor {floor=}"

    ddot_err = (got[..., 1] - want[..., 1]).abs().max()
    assert ddot_err < 1e-5, f"delta_dot should be near-exact, got {ddot_err}"

    # The force inherits delta's error through the coupler stiffness.
    k_max = float(es[:, [2, 4]].max())
    f_err = (got[..., 2] - want[..., 2]).abs().max()
    assert f_err < k_max * floor * 4
