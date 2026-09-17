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
        zero_mean_edge: bool = False,
    ) -> None:
        super().__init__()
        self.rounds = rounds
        self.zero_mean_edge = zero_mean_edge
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

        d = self.decode_edge(e).squeeze(-1)
        if self.zero_mean_edge:
            # sum_j delta_j = (x_first - x_last) - sum L0, so the sum of the
            # corrections over a consist is pinned by the end vehicles' motion,
            # which the trapezoid already carries. Measured on the corpus, the
            # true mean has a std of 0.026 mm on train and 0.008 mm on
            # ood_size, against a 1.04 mm per-coupler scale -- 2.5% and 0.6%.
            #
            # Left free, that mean is an unbounded drift mode: the model
            # stretches the whole train a little every step and nothing pushes
            # back. It showed up as a +3.9 mm bias after 200 steps on 150-car
            # consists, a quarter of the total error there, with the rollout
            # growing as n^0.66 instead of the n^0.37 seen on val.
            #
            # MEASURED AND REJECTED as a default, 2026-09-17. Imposing it cost
            # ~47% on val shape error at 200 steps in all three seeds tried
            # (7.30 mm against 4.95 mm mean) and changed ood_size not at all
            # (39.3 mm either way). The drift it removes is only 3% of the rms
            # there; the dominant term is variance, not bias. Plausibly the
            # centering couples every coupler's gradient to every other one,
            # which hurts more than the freed drift mode costs.
            #
            # Kept behind the flag because it is a clean ablation for Ch6, not
            # because it helps. Targets are left uncentered: the component the
            # model then cannot emit is 0.026 mm against a 1.04 mm scale, an
            # irreducible loss floor of well under 0.1%.
            d = d - d.mean(dim=-1, keepdim=True)
        return self.decode_node(n).squeeze(-1), d
