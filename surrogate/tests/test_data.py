"""The bucketed loader must keep batches rectangular and honour its budget."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from surrogate.data import N_EDGE_FEATURES, N_NODE_FEATURES, ScenarioSet

DATA = _REPO_ROOT / "data" / "v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="data/v2 not built")


@pytest.fixture(scope="module")
def small():
    return ScenarioSet(DATA, "val", budget_gib=0.25, device=DEVICE, seed=0)


def test_buckets_are_pure(small):
    """Every scenario in a block has that block's consist size."""
    for n, blk in small.blocks.items():
        assert blk.state.shape[2] == n
        assert blk.edge_static.shape[1] == n - 1
        assert blk.edge_dyn.shape[2] == n - 1


def test_batches_are_rectangular_and_shaped(small):
    """A batch is one consist size, so no padding and no mask is needed."""
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    seen = set()
    for _ in range(25):
        b = small.sample(64, gen, None)
        n = b.n_vehicles
        seen.add(n)
        assert b.node.shape == (64, n, N_NODE_FEATURES)
        assert b.edge.shape == (64, n - 1, N_EDGE_FEATURES)
        assert b.abar.shape == (64, n)
        assert b.dcorr.shape == (64, n - 1)
        assert torch.isfinite(b.node).all() and torch.isfinite(b.edge).all()
    assert len(seen) > 1, "sampling never left a single bucket"


def test_budget_is_respected(small):
    assert small.gib <= 0.25


def test_iter_all_covers_every_transition(small):
    total = sum(b.abar.shape[0] for b in small.iter_all(4096, None))
    assert total == len(small)


def test_ood_size_is_disjoint_from_train():
    """The size split is an extrapolation test, not an interpolation one.

    If these ever overlap, the Chapter 6 universality result is measuring the
    wrong thing, so it is worth failing loudly here.
    """
    import pandas as pd

    idx = pd.read_parquet(DATA / "index.parquet")
    train_n = set(idx[idx.split == "train"].N.unique())
    ood_n = set(idx[idx.split == "ood_size"].N.unique())
    assert train_n and ood_n
    assert not (train_n & ood_n)
    assert min(ood_n) > max(train_n)
