"""The v3 port: gear state carried, brake lags per vehicle, notch force as input.

Same standard as the legacy tests -- whatever the update rule claims to solve
exactly must reproduce the corpus to storage precision -- plus the one place
where it is not exact by design: the gear return map, pinned at its measured
floor. The module builds its own small v3 corpus (~25 s) so the tests do not
depend on a regeneration having happened.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from surrogate.data import N_EDGE_FEATURES_V3, N_NODE_FEATURES_V3, ScenarioSet
from surrogate.rollout import rollout, training_batch
from surrogate.step import GearLaw, brake_step, coupler_features, edge_targets, node_targets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ONE = {"abar": torch.tensor(1.0, device=DEVICE), "dcorr": torch.tensor(1.0, device=DEVICE)}


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    out = tmp_path_factory.mktemp("v3") / "corpus"
    subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "build_dataset.py"),
         "--n-scenarios", "12", "--batch-size", "12", "--seed", "1",
         "--physics", "v3", "--duration-range", "60", "120", "--out", str(out)],
        check=True, capture_output=True, cwd=_REPO_ROOT)
    return ScenarioSet(out, "train", budget_gib=1.0, device=DEVICE, seed=0)


def _starts(ds, n=8, k=5):
    blk = max(ds.blocks.values(), key=lambda b: b.n)     # couplers in contact
    si = torch.arange(min(n, blk.state.shape[0]), device=DEVICE)
    return blk, si, torch.full_like(si, k)


class _Oracle:
    """True targets for each step, as the legacy test's oracle."""

    def __init__(self, blk, si, k0, h):
        self.blk, self.si, self.k0, self.h, self.j = blk, si, k0, h, 0

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, node, edge):
        k = self.k0 + self.j
        st0, st1 = self.blk.state[self.si, k], self.blk.state[self.si, k + 1]
        d0 = self.blk.edge_dyn[self.si, k][..., 0]
        d1 = self.blk.edge_dyn[self.si, k + 1][..., 0]
        self.j += 1
        return node_targets(st0, st1, self.h), edge_targets(d0, d1, st0, st1, self.h)


def test_loader_detects_v3_and_widens_the_features(ds):
    assert ds.v3 and ds.law is not None
    assert (ds.n_node_features, ds.n_edge_features) == (N_NODE_FEATURES_V3, N_EDGE_FEATURES_V3)
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    b = ds.sample(32, gen, None)
    n = b.n_vehicles
    assert b.node.shape == (32, n, N_NODE_FEATURES_V3)
    assert b.edge.shape == (32, n - 1, N_EDGE_FEATURES_V3)
    assert b.gear0.shape == (32, n - 1)
    assert torch.isfinite(b.node).all() and torch.isfinite(b.edge).all()


def test_gear_law_reproduces_the_stored_coupler_force(ds):
    """F_cpl = clamp(z, envelope) from the stored stretch and gear state is the
    corpus's own force, so the gear columns describe the real loop."""
    for blk in ds.blocks.values():
        st, ed, g = blk.state, blk.edge_dyn, blk.gear
        f = coupler_features(ed[..., 0], st[..., 1], blk.edge_static[:, None],
                             g, blk.law)[..., 2]
        # float32 delta through gear stiffness up to k_lock: allow 1 N per kN
        assert (f - ed[..., 2]).abs().max() <= 1e-3 * ed[..., 2].abs().max() + 50.0


def test_true_targets_advance_onto_the_stored_state(ds):
    """v, delta: exact. z_trac and the switched z_brk: storage precision.
    The gear state: the return map's floor, measured at 1.7 kN rms."""
    blk, si, k = _starts(ds, k=0)
    errs = {"v": [], "d": [], "zb": [], "zt": [], "g": []}
    for kk in range(int(blk.lengths[si].min()) - 1):
        k = torch.full_like(si, kk)
        st0, d0, g0 = (t.double() for t in blk.carry(si, k))
        st1, d1, g1 = (t.double() for t in blk.carry(si, k + 1))
        a = node_targets(st0, st1, ds.h)
        c = edge_targets(d0, d1, st0, st1, ds.h)
        got, got_d, got_g = blk.advance(st0, d0, g0, a, c, si, k)
        errs["v"].append((got[..., 1] - st1[..., 1]).abs().max())
        errs["d"].append((got_d - d1).abs().max())
        errs["zb"].append((got[..., 2] - st1[..., 2]).abs().max())
        errs["zt"].append((got[..., 3] - st1[..., 3]).abs().max())
        lo, hi = blk.law.envelope(d1, blk.edge_static[si][..., 1].double())
        f_true = torch.minimum(torch.maximum(g1, lo), hi)
        f_got = torch.minimum(torch.maximum(got_g, lo), hi)
        errs["g"].append((f_got - f_true).pow(2).mean())
    g = torch.stack(errs.pop("g"))
    m = {k: float(torch.stack(v).max()) for k, v in errs.items()}
    assert m["v"] < 1e-9 and m["d"] < 1e-9
    assert m["zt"] < 1e-5                       # notch units
    assert m["zb"] < 0.05                       # N, on a ~57 kN scale
    # Gear force from the return map: pooled rms near the 1.7 kN measured on
    # the 64-scenario smoke corpus; single steps run higher (11.6 kN seen).
    assert float(g.mean().sqrt()) < 3e3
    assert float(g.max().sqrt()) < 25e3


def test_oracle_rollout_keeps_the_force_near_the_return_map_floor(ds):
    blk, si, k0 = _starts(ds)
    horizon = min(30, int(blk.lengths[si].min()) - 6)
    out = rollout(_Oracle(blk, si, k0, ds.h), blk, si, k0, horizon,
                  route_s=ds.route_s, route_f=ds.route_f, tgt_std=ONE)
    assert float(out["v"].max()) < 1e-3
    assert float(out["delta"].max()) < 1e-4
    # Gear error does not feed back into v or delta under the oracle, and the
    # envelope bounds it, so it must stay bounded rather than drift.
    assert float(out["F"].max()) < 50e3
    assert float(out["F"].median()) < float(out["F_hold"].median())


def test_brake_step_switches_to_the_release_lag():
    z = torch.tensor([1000.0, 1000.0], dtype=torch.float64)
    u = torch.tensor([0.0, 2000.0], dtype=torch.float64)          # release, apply
    ta = torch.full_like(z, 0.25)
    tr = torch.full_like(z, 6.7)
    got = brake_step(z, u, u, ta, tr, 0.25)
    want = u + (z - u) * torch.exp(-0.25 / torch.tensor([6.7, 0.25], dtype=torch.float64))
    assert torch.allclose(got, want, rtol=1e-12)


def test_gear_return_map_sticks_then_slides():
    law = GearLaw.default(dtype=torch.float64)
    s = torch.tensor(0.005, dtype=torch.float64)
    d0 = torch.tensor(0.030, dtype=torch.float64)                 # 25 mm into the gear
    lo, hi = law.envelope(d0, s)
    z0 = 0.5 * (lo + hi)                                          # inside the loop
    small = law.advance(z0, d0, d0 + 1e-5, s)                     # sticks: elastic
    assert torch.isclose(small - z0, torch.tensor(law.k_transition * 1e-5, dtype=torch.float64))
    big = law.advance(z0, d0, d0 + 0.02, s)                       # slides: on the loading curve
    assert torch.isclose(big, law.envelope(d0 + 0.02, s)[1])
    slack = law.advance(z0, d0, torch.tensor(0.0, dtype=torch.float64), s)
    assert slack == 0.0                                           # back in the slack


def test_training_batch_runs_v3_with_pushforward(ds):
    got = ds.sample_starts(8, torch.Generator(device=DEVICE).manual_seed(0), 4)
    assert got is not None
    blk, si, k0 = got

    class _Zero(torch.nn.Module):
        def forward(self, node, edge):
            return (torch.zeros(node.shape[:-1], device=node.device),
                    torch.zeros(edge.shape[:-1], device=edge.device))

    for n_push in (0, 3):
        node, edge, ta, td = training_batch(
            _Zero(), ds, blk, si, k0, n_push=n_push,
            gen=torch.Generator(device=DEVICE).manual_seed(1),
            v_std=0.005, delta_std=0.001, tgt_std=ONE)
        assert node.shape[-1] == N_NODE_FEATURES_V3 and edge.shape[-1] == N_EDGE_FEATURES_V3
        assert torch.isfinite(node).all() and torch.isfinite(edge).all()
        assert torch.isfinite(ta).all() and torch.isfinite(td).all()


def test_locomotive_force_feature_is_zero_on_wagons(ds):
    blk, si, k = _starts(ds)
    st = blk.state[si, k]
    extra = blk.node_extra(st, blk.u[si, k], blk.u[si, k + 1], si)
    wagons = blk.loco_table[si][..., 0] == 0
    assert wagons.any() and (~wagons).any()
    assert (extra[..., :2][wagons] == 0).all()
    assert torch.allclose(extra[..., 2][wagons], torch.full_like(extra[..., 2][wagons], 0.15))
