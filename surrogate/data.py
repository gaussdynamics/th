"""Variable-``N`` transition loader.

Scenarios are grouped into one block per consist size, and every batch is drawn
from a single block. That keeps batches rectangular with no padding and no
masks -- which matters more here than usual, because ``N`` spans 10 to 150 and
padding everything up to 150 would waste most of the tensor on the small
consists that dominate the corpus.

The model itself is indifferent to ``N``: message passing shares one set of
weights across every vehicle and every coupler, so a block of 12-car consists
and a block of 140-car consists exercise exactly the same parameters. Bucketing
is a data-layout decision, not a modelling one.

Why whole scenarios rather than individual transitions: a scenario ``.npz`` is
zlib-compressed and costs ~62 ms to open (``DATASET_BUILD_REPORT.md``), so a
per-sample read would be far slower than the training step it feeds. Scenarios
are decompressed once and held on the device.

The full train split is ~10.4 GiB laid out this way, so ``budget_gib`` caps what
is loaded and subsamples scenarios to fit. Raise it, or pass ``max_scenarios``,
to trade coverage against memory.

Normalization comes from the corpus ``norm_stats.json`` (train split only). The
two *target* statistics are not in that file -- the targets did not exist when
it was written -- so they are computed here and reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from surrogate.step import edge_targets, node_targets

#: Node channels fed to the network. Absolute chainage ``x`` is deliberately
#: absent: it ranges over 0-30 km and carries no transferable information. What
#: the physics reads at ``x`` -- grade, curvature, speed limit -- is supplied
#: instead, already sampled per vehicle.
N_NODE_FEATURES = 3 + 4 + 7 + 3   # (v, z_brk, z_trac) + u(k,k+1) + static + route
N_EDGE_FEATURES = 3 + 6           # (delta, delta_dot, F_cpl) + static

#: Bytes per vehicle-step held on the device: state(4) + u(2) + route(3) +
#: edge_dynamic(~3) float32 channels. Used to size the subsample before loading.
BYTES_PER_VEHICLE_STEP = 48

_ROUTE_FIELDS = ("route_sin_theta", "route_kappa", "route_vmax")


@dataclass
class Batch:
    node: Tensor        # [B, N, N_NODE_FEATURES]
    edge: Tensor        # [B, N-1, N_EDGE_FEATURES]
    abar: Tensor        # [B, N]    target, normalized
    dcorr: Tensor       # [B, N-1]  target, normalized
    n_vehicles: int


class _Block:
    """Every loaded scenario of one consist size, resident on ``device``."""

    def __init__(self, n: int, recs: list[dict], norm: dict[str, Tensor],
                 device: str) -> None:
        self.n = n
        self.device = device
        self.norm = norm
        lengths = [r["state"].shape[0] for r in recs]
        t_max = max(lengths)

        def pad(key: str, tail: tuple[int, ...]) -> Tensor:
            out = np.zeros((len(recs), t_max, *tail), dtype=np.float32)
            for i, r in enumerate(recs):
                out[i, : r[key].shape[0]] = r[key]
            return torch.from_numpy(out).to(device)

        self.state = pad("state", (n, 4))
        self.u = pad("u", (n, 2))
        self.route = pad("route", (n, 3))
        self.edge_dyn = pad("edge_dynamic", (n - 1, 3))
        self.node_static = torch.from_numpy(
            np.stack([r["node_static"] for r in recs])).to(device)
        self.edge_static = torch.from_numpy(
            np.stack([r["edge_static"] for r in recs])).to(device)

        pairs = [(i, k) for i, L in enumerate(lengths) for k in range(L - 1)]
        self.pairs = torch.tensor(pairs, dtype=torch.long, device=device)

    def __len__(self) -> int:
        return self.pairs.shape[0]

    def assemble(self, si: Tensor, k: Tensor, h: float,
                 tgt_std: dict[str, Tensor] | None) -> Batch:
        st0, st1 = self.state[si, k], self.state[si, k + 1]
        u0, u1 = self.u[si, k], self.u[si, k + 1]
        es = self.edge_static[si]
        ed0, ed1 = self.edge_dyn[si, k], self.edge_dyn[si, k + 1]
        nrm = self.norm

        node = torch.cat([
            (st0[..., 1:4] - nrm["state_mean"][1:4]) / nrm["state_std"][1:4],
            (u0 - nrm["u_mean"]) / nrm["u_std"],
            (u1 - nrm["u_mean"]) / nrm["u_std"],
            (self.node_static[si] - nrm["node_static_mean"]) / nrm["node_static_std"],
            (self.route[si, k] - nrm["route_mean"]) / nrm["route_std"],
        ], dim=-1)
        edge = torch.cat([
            (ed0 - nrm["edge_dyn_mean"]) / nrm["edge_dyn_std"],
            (es - nrm["edge_static_mean"]) / nrm["edge_static_std"],
        ], dim=-1)

        abar = node_targets(st0, st1, h)
        dcorr = edge_targets(ed0[..., 0], ed1[..., 0], st0, st1, h)
        if tgt_std is not None:
            abar = abar / tgt_std["abar"]
            dcorr = dcorr / tgt_std["dcorr"]
        return Batch(node=node, edge=edge, abar=abar, dcorr=dcorr, n_vehicles=self.n)


class ScenarioSet:
    """Scenarios from one split, bucketed by consist size."""

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        n_values: list[int] | None = None,
        max_scenarios: int | None = None,
        budget_gib: float = 6.0,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.device = device

        idx = pd.read_parquet(self.root / "index.parquet")
        rows = idx[idx.split == split]
        if n_values is not None:
            rows = rows[rows.N.isin(n_values)]
        if rows.empty:
            raise ValueError(f"no {split} scenarios in {root} for N={n_values}")

        # Shuffle first, then fill the budget, so the subsample is not biased
        # toward whichever consist sizes happen to sort first.
        rows = rows.sample(frac=1.0, random_state=seed)
        cost = (BYTES_PER_VEHICLE_STEP * rows.N * rows["T"]).to_numpy()
        keep = np.cumsum(cost) <= budget_gib * 2 ** 30
        rows = rows[keep]
        if max_scenarios is not None:
            rows = rows.iloc[:max_scenarios]
        if rows.empty:
            raise ValueError(f"budget_gib={budget_gib} is too small for one scenario")
        self.rows = rows.reset_index(drop=True)
        self.gib = float(cost[keep][: len(self.rows)].sum() / 2 ** 30)

        norm = _load_norm(self.root, device)
        routes: dict[str, dict] = {}
        by_n: dict[int, list[dict]] = {}
        h = tau_brk = tau_trac = None

        for _, r in self.rows.iterrows():
            p = self.root / "scenarios" / r.route_id / f"{r.scenario_id}.npz"
            with np.load(p) as f:
                rec = {
                    "state": f["state"].astype(np.float32),
                    "u": np.stack([f["u_trac"], f["u_brk"]], -1).astype(np.float32),
                    "node_static": f["node_static"].astype(np.float32),
                    "edge_static": f["edge_static"].astype(np.float32),
                    # The corpus's only precise record of consist shape: delta
                    # differenced out of float32 `x` at 15 km carries ~1 mm of
                    # quantization, larger than the signal being learned.
                    "edge_dynamic": f["edge_dynamic"].astype(np.float32),
                }
                if h is None:
                    t = f["t"].astype(np.float64)
                    h = float(t[1] - t[0])
                    tau_brk, tau_trac = float(f["tau_brk_s"]), float(f["tau_trac_s"])
            if r.route_id not in routes:
                with np.load(self.root / "routes" / f"{r.route_id}.npz") as f:
                    routes[r.route_id] = {k: f[k].astype(np.float64)
                                          for k in ("route_s", *_ROUTE_FIELDS)}
            rt = routes[r.route_id]
            xq = rec["state"][:, :, 0].astype(np.float64)
            rec["route"] = np.stack(
                [np.interp(xq, rt["route_s"], rt[k]).astype(np.float32)
                 for k in _ROUTE_FIELDS], axis=-1)
            by_n.setdefault(int(r.N), []).append(rec)

        self.h, self.tau_brk, self.tau_trac = float(h), float(tau_brk), float(tau_trac)
        self.blocks = {n: _Block(n, recs, norm, device)
                       for n, recs in sorted(by_n.items())}
        counts = torch.tensor([len(b) for b in self.blocks.values()],
                              dtype=torch.float64, device=device)
        # Sample a block in proportion to how many transitions it holds, so a
        # consist size is represented by its share of the data rather than by
        # how many distinct sizes happen to exist.
        self.block_p = (counts / counts.sum()).float()
        self.block_keys = list(self.blocks.keys())

    def __len__(self) -> int:
        return sum(len(b) for b in self.blocks.values())

    @property
    def n_range(self) -> tuple[int, int]:
        return min(self.block_keys), max(self.block_keys)

    def sample(self, batch_size: int, gen: torch.Generator,
               tgt_std: dict[str, Tensor] | None) -> Batch:
        bi = int(torch.multinomial(self.block_p, 1, generator=gen).item())
        blk = self.blocks[self.block_keys[bi]]
        j = torch.randint(len(blk), (batch_size,), generator=gen, device=self.device)
        p = blk.pairs[j]
        return blk.assemble(p[:, 0], p[:, 1], self.h, tgt_std)

    def iter_all(self, batch_size: int, tgt_std: dict[str, Tensor] | None):
        for blk in self.blocks.values():
            for a in range(0, len(blk), batch_size):
                p = blk.pairs[a : a + batch_size]
                yield blk.assemble(p[:, 0], p[:, 1], self.h, tgt_std)

    def target_std(self, n_batches: int = 64, batch_size: int = 2048) -> dict[str, Tensor]:
        """Std of each target over this set. Not present in ``norm_stats.json``."""
        a, d = [], []
        gen = torch.Generator(device=self.device).manual_seed(0)
        for _ in range(n_batches):
            b = self.sample(batch_size, gen, None)
            a.append(b.abar.flatten())
            d.append(b.dcorr.flatten())
        return {"abar": torch.cat(a).std(), "dcorr": torch.cat(d).std()}


def _load_norm(root: Path, device: str) -> dict[str, Tensor]:
    raw = json.loads((root / "norm_stats.json").read_text(encoding="utf-8"))
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
    return {
        "state_mean": t(raw["state"]["mean"]), "state_std": t(raw["state"]["std"]),
        "node_static_mean": t(raw["node_static"]["mean"]),
        "node_static_std": t(raw["node_static"]["std"]),
        "edge_static_mean": t(raw["edge_static"]["mean"]),
        "edge_static_std": t(raw["edge_static"]["std"]),
        "edge_dyn_mean": t(raw["edge_dynamic"]["mean"]),
        "edge_dyn_std": t(raw["edge_dynamic"]["std"]),
        "u_mean": t([raw["u_trac"]["mean"][0], raw["u_brk"]["mean"][0]]),
        "u_std": t([raw["u_trac"]["std"][0], raw["u_brk"]["std"][0]]),
        "route_mean": t([raw[f"route_{k}"]["mean"][0]
                         for k in ("sin_theta", "kappa", "vmax")]),
        "route_std": t([raw[f"route_{k}"]["std"][0]
                        for k in ("sin_theta", "kappa", "vmax")]),
    }
