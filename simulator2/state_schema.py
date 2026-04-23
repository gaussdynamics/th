"""Named indices for tensorized node and edge feature channels."""

from __future__ import annotations

from enum import IntEnum


class NodeChannel(IntEnum):
    """Per-vehicle (node) channels in ``H[..., d_node]``."""

    X = 0
    V = 1
    Z_BRK = 2
    Z_TRAC = 3
    MASS_KG = 4
    DAVIS_A = 5
    DAVIS_B = 6
    DAVIS_C = 7
    CAN_TRACTION = 8
    F_TRAC_MAX_N = 9
    F_BRK_MAX_N = 10


class EdgeChannel(IntEnum):
    """Per-coupler (edge) channels in ``E[..., d_edge]``."""

    DELTA = 0
    DELTA_DOT = 1
    F_CPL = 2
    L0_M = 3
    SLACK_HALF_M = 4
    K_DRAFT = 5
    C_DRAFT = 6
    K_BUFF = 7
    C_BUFF = 8


D_NODE = len(NodeChannel)
D_EDGE = len(EdgeChannel)


def node_channel_dict() -> dict[str, int]:
    return {c.name.lower(): int(c) for c in NodeChannel}


def edge_channel_dict() -> dict[str, int]:
    return {c.name.lower(): int(c) for c in EdgeChannel}
