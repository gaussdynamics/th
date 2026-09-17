"""Message passing on the consist chain.

A consist is a *path* graph: every vehicle has at most two neighbours, and
always the same two. That makes a graph library unnecessary -- message passing
on a path is slicing and padding, which is exactly how ``simulator2.torch_rhs``
already computes coupler forces::

    delta = x[:, :-1] - x[:, 1:]     # every edge reads its two endpoints
    f_in  = pad(f_cpl, (1, 0))       # every node gathers its two edges
    f_out = pad(f_cpl, (0, 1))

:class:`ChainGNN` is that pattern with the spring law replaced by an MLP and
repeated ``rounds`` times, in the same padded ``[B, N]`` layout the rest of the
project uses. No ``torch_geometric``, no scatter kernels, no CUDA-version pin.

**Depth is a physics number, not a hyperparameter.** Information travels one
vehicle per round, and a coupler force wave travels 167-193 m/s (median, by
branch) against a 17 m car pitch -- about 3 vehicles per 0.25 s step, 5 at p99.
So ``rounds=5`` covers the physical domain of dependence of one step, and does
not grow with ``N``. Measured from ``node_static``/``edge_static`` over 60
scenarios; see ``SURROGATE_FORMULATION_NOTE.md``.

Left and right messages are concatenated rather than summed. The coupler law is
asymmetric -- draft and buff have different stiffnesses -- so a vehicle being
pulled from ahead is in a different state from one being pushed from behind,
and summing would throw that away.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.SiLU(),
        nn.Linear(d_hidden, d_out),
    )


class ChainGNN(nn.Module):
    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        *,
        hidden: int = 128,
        rounds: int = 5,
        n_node_out: int = 1,
        n_edge_out: int = 1,
    ) -> None:
        super().__init__()
        self.rounds = rounds
        self.enc_node = _mlp(n_node_features, hidden, hidden)
        self.enc_edge = _mlp(n_edge_features, hidden, hidden)
        # One set of weights per round, shared across every vehicle and every
        # coupler -- which is what makes the model indifferent to N.
        self.edge_mlp = nn.ModuleList(
            [_mlp(3 * hidden, hidden, hidden) for _ in range(rounds)])
        self.node_mlp = nn.ModuleList(
            [_mlp(3 * hidden, hidden, hidden) for _ in range(rounds)])
        self.norm_edge = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(rounds)])
        self.norm_node = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(rounds)])
        # Two heads: acceleration lives on vehicles, the shape correction on
        # couplers. They are different quantities in different coordinates, so
        # they get separate decoders off the shared message-passed state.
        self.decode_node = _mlp(hidden, hidden, n_node_out)
        self.decode_edge = _mlp(hidden, hidden, n_edge_out)

    def forward(self, node: Tensor, edge: Tensor) -> tuple[Tensor, Tensor]:
        """``node`` ``[B, N, Fn]``, ``edge`` ``[B, N-1, Fe]``.

        Returns ``(abar [B, N], ddelta_corr [B, N-1])``, both normalized.
        """
        n = self.enc_node(node)
        e = self.enc_edge(edge)

        for r in range(self.rounds):
            # edge update: each coupler sees itself and the vehicles it joins
            e = e + self.norm_edge[r](
                self.edge_mlp[r](torch.cat([e, n[:, :-1], n[:, 1:]], dim=-1)))
            # node update: each vehicle gathers the coupler ahead and behind.
            # The end vehicles have no coupler on one side; zero padding is the
            # correct "no neighbour" signal and matches the RHS's f_in/f_out.
            left = F.pad(e, (0, 0, 1, 0))
            right = F.pad(e, (0, 0, 0, 1))
            n = n + self.norm_node[r](
                self.node_mlp[r](torch.cat([n, left, right], dim=-1)))

        return self.decode_node(n).squeeze(-1), self.decode_edge(e).squeeze(-1)
