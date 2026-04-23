"""Pack/unpack consistency and tensor shapes."""

from __future__ import annotations

import numpy as np
import pytest

from simulator2.params import CouplerParameters, VehicleParameters
from simulator2.state_schema import D_EDGE, D_NODE, EdgeChannel, NodeChannel
from simulator2.tensor_state import (
    compute_edge_tensor_from_nodes,
    extract_dynamic_node_tensor,
    pack_dynamic_state_from_tensor_state,
    unpack_to_tensor_state,
)


def _tiny_train():
    v = [
        VehicleParameters(
            1e5, 800.0, 15.0, 0.8, True, 3e5, 2e5
        ),
        VehicleParameters(9e4, 600.0, 12.0, 0.7, False, 0.0, 2e5),
    ]
    c = [
        CouplerParameters(20.0, 0.02, 9e6, 5e5, 12e6, 7e5),
    ]
    return v, c


def test_pack_unpack_roundtrip():
    vehicles, couplers = _tiny_train()
    n = len(vehicles)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(4 * n)
    state = unpack_to_tensor_state(y, vehicles, couplers)
    assert state.H.shape == (n, D_NODE)
    assert state.E.shape == (n - 1, D_EDGE)
    y2 = pack_dynamic_state_from_tensor_state(state)
    np.testing.assert_allclose(y, y2)

    dyn = extract_dynamic_node_tensor(state)
    assert dyn.shape == (n, 4)
    np.testing.assert_allclose(dyn[:, 0], y[0:n])
    np.testing.assert_allclose(dyn[:, 1], y[n : 2 * n])


def test_column_semantics():
    vehicles, couplers = _tiny_train()
    n = len(vehicles)
    x = np.array([100.0, 80.0])
    v = np.array([1.0, 0.5])
    zb = np.zeros(n)
    zt = np.zeros(n)
    y = np.concatenate([x, v, zb, zt])
    state = unpack_to_tensor_state(y, vehicles, couplers)
    assert state.H[0, NodeChannel.CAN_TRACTION] == 1.0
    assert state.H[1, NodeChannel.CAN_TRACTION] == 0.0
    assert state.H[0, NodeChannel.MASS_KG] == vehicles[0].mass_kg
    E = compute_edge_tensor_from_nodes(state.H, couplers)
    dlt = (x[0] - x[1]) - couplers[0].L0_m
    assert abs(E[0, EdgeChannel.DELTA] - dlt) < 1e-9
    assert abs(E[0, EdgeChannel.DELTA_DOT] - (v[0] - v[1])) < 1e-9


def test_empty_edges_single_vehicle():
    v = [VehicleParameters(1e5, 1.0, 1.0, 1.0, False)]
    c: list = []
    y = np.zeros(4)
    st = unpack_to_tensor_state(y, v, c)
    assert st.E.shape == (0, D_EDGE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
