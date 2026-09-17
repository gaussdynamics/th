"""The rollout machinery must be exact when the predictions are.

If a rollout fed perfect targets does not reproduce the stored trajectory, then
any rollout error measured later is partly the harness, not the model -- and
that is invisible in a plot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from surrogate.data import ScenarioSet, interp_route
from surrogate.rollout import add_noise, noisy_targets, rollout
from surrogate.step import advance, edge_targets, node_targets

DATA = _REPO_ROOT / "data" / "v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="data/v2 not built")


@pytest.fixture(scope="module")
def ds():
    return ScenarioSet(DATA, "val", budget_gib=0.2, device=DEVICE, seed=0)


class _Oracle:
    """Returns the true targets for each step, normalized as the model would."""

    def __init__(self, blk, si, k0, h, tgt_std):
        self.blk, self.si, self.k0, self.h, self.std = blk, si, k0, h, tgt_std
        self.j = 0

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, node, edge):
        k = self.k0 + self.j
        st0, st1 = self.blk.state[self.si, k], self.blk.state[self.si, k + 1]
        d0 = self.blk.edge_dyn[self.si, k][..., 0]
        d1 = self.blk.edge_dyn[self.si, k + 1][..., 0]
        a = node_targets(st0, st1, self.h) / self.std["abar"]
        d = edge_targets(d0, d1, st0, st1, self.h) / self.std["dcorr"]
        self.j += 1
        return a, d


def test_oracle_rollout_reproduces_the_trajectory(ds):
    blk = ds.blocks[ds.block_keys[0]]
    si = torch.arange(min(8, blk.state.shape[0]), device=DEVICE)
    k0 = torch.full_like(si, 5)
    std = {"abar": torch.tensor(1.0, device=DEVICE),
           "dcorr": torch.tensor(1.0, device=DEVICE)}
    out = rollout(_Oracle(blk, si, k0, ds.h, std), blk, si, k0, 30,
                  route_s=ds.route_s, route_f=ds.route_f, nrm=ds.norm,
                  h=ds.h, tau_brk=ds.tau_brk, tau_trac=ds.tau_trac, tgt_std=std)
    # float32 accumulation over 30 steps, on velocities of order 10 m/s
    assert float(out["v"].max()) < 1e-3
    assert float(out["delta"].max()) < 1e-4


def test_interp_route_matches_numpy(ds):
    import numpy as np

    blk = ds.blocks[ds.block_keys[0]]
    si = torch.arange(min(4, blk.state.shape[0]), device=DEVICE)
    x = blk.state[si, 3][..., 0]
    got = interp_route(ds.route_s[blk.corridor[si]], ds.route_f[blk.corridor[si]], x)
    want = blk.route[si, 3]
    assert (got - want).abs().max() < 1e-3


def test_noisy_targets_land_on_the_true_next_state(ds):
    """A perturbed start plus its recomputed target must still reach the truth."""
    blk = ds.blocks[ds.block_keys[0]]
    si = torch.arange(min(8, blk.state.shape[0]), device=DEVICE)
    k = 7
    st0 = blk.state[si, k].double()
    st1 = blk.state[si, k + 1].double()
    d0 = blk.edge_dyn[si, k][..., 0].double()
    d1 = blk.edge_dyn[si, k + 1][..., 0].double()
    es = blk.edge_static[si].double()
    u0, u1 = blk.u[si, k].double(), blk.u[si, k + 1].double()

    gen = torch.Generator(device=DEVICE).manual_seed(0)
    st_p, d_p = add_noise(st0, d0, gen, v_std=0.05, delta_std=0.005)
    assert (st_p[..., 1] - st0[..., 1]).abs().max() > 1e-3   # noise really applied

    a, c = noisy_targets(st_p, d_p, st1, d1, ds.h)
    got, got_d = advance(st_p, d_p, a, c, u0, u1, es[..., 0],
                         tau_brk=ds.tau_brk, tau_trac=ds.tau_trac, h=ds.h)
    assert (got[..., 1] - st1[..., 1]).abs().max() < 1e-9
    assert (got_d - d1).abs().max() < 1e-9
