"""Tensor rollout shapes and expected channels."""

from __future__ import annotations

import numpy as np
import pytest

from simulator2.scenarios import make_simple_train_scenario
from simulator2.simulate import simulate_train_tensorized
from simulator2.state_schema import EdgeChannel, NodeChannel


def test_small_consist_shapes():
    sc = make_simple_train_scenario(n_cars=2, n_time_samples=51, t_span=(0.0, 5.0))
    res = simulate_train_tensorized(sc)
    n = len(sc.vehicles)
    t = res.t
    assert res.H_hist.shape == (len(t), n, res.metadata["d_node"])
    assert res.E_hist.shape == (len(t), n - 1, res.metadata["d_edge"])
    assert res.metadata["d_node"] == res.H_hist.shape[2]
    assert res.metadata["d_edge"] == res.E_hist.shape[2]


def test_larger_consist_shapes():
    sc = make_simple_train_scenario(n_cars=15, n_time_samples=101, t_span=(0.0, 10.0))
    res = simulate_train_tensorized(sc)
    n = len(sc.vehicles)
    assert res.H_hist.shape[1] == n
    assert res.E_hist.shape[1] == n - 1


def test_hist_contains_position_and_coupler_channel():
    sc = make_simple_train_scenario(n_cars=3, n_time_samples=31, t_span=(0.0, 4.0))
    res = simulate_train_tensorized(sc)
    k = 10
    H = res.H_hist[k]
    E = res.E_hist[k]
    assert np.all(np.isfinite(H))
    assert np.all(np.isfinite(E))
    # Positions should decrease along the train (lead index 0 largest x)
    x = H[:, int(NodeChannel.X)]
    assert x[0] > x[-1]
    assert E.shape[0] == H.shape[0] - 1
    assert np.all(E[:, int(EdgeChannel.L0_M)] == sc.couplers[0].L0_m)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
