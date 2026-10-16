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

from surrogate.step import (GearLaw, actuator_step, advance, coupler_features,
                            edge_targets, node_targets)

#: Node channels fed to the network. Absolute chainage ``x`` is deliberately
#: absent: it ranges over 0-30 km and carries no transferable information. What
#: the physics reads at ``x`` -- grade, curvature, speed limit -- is supplied
#: instead, already sampled per vehicle.
N_NODE_FEATURES = 3 + 4 + 7 + 3   # (v, z_brk, z_trac) + u(k,k+1) + static + route
N_EDGE_FEATURES = 3 + 6           # (delta, delta_dot, F_cpl) + static

#: v3 corpora (``simulator2.physics_config``) add, per vehicle, the locomotive
#: force at the start and at the end of the step (``z_trac`` is a lagged notch
#: there, read through the notch tables) and the inertia factor; per coupler,
#: the gear state and its envelope. See :func:`build_features`.
N_NODE_FEATURES_V3 = N_NODE_FEATURES + 3
N_EDGE_FEATURES_V3 = N_EDGE_FEATURES + 3

#: ``(inertia_factor - 1)`` scale: the corpus has 0.015 (wagons) and 0.125
#: (locomotives), Handbook p. 459.
INERTIA_SCALE = 0.1

#: Bytes per vehicle-step held on the device: state(4) + u(2) + route(3) +
#: edge_dynamic(~3) float32 channels. Used to size the subsample before loading.
BYTES_PER_VEHICLE_STEP = 48

_ROUTE_FIELDS = ("route_sin_theta", "route_kappa", "route_vmax")


def build_features(
    st: Tensor, ed: Tensor, node_static: Tensor, edge_static: Tensor,
    route: Tensor, u0: Tensor, u1: Tensor, nrm: dict[str, Tensor],
    node_extra: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Network inputs from raw tensors.

    Shared by the training sampler and the rollout so the two cannot drift
    apart -- a train/inference feature mismatch is the classic silent way for a
    step model to look fine one-step and fall apart rolled out.

    v3: ``ed`` has the six gear columns of :func:`coupler_features`; the gear
    state and envelope are forces, so they take the ``F_cpl`` statistics.
    ``node_extra`` is already scaled (:meth:`_Block.node_extra`).
    """
    parts = [
        (st[..., 1:4] - nrm["state_mean"][1:4]) / nrm["state_std"][1:4],
        (u0 - nrm["u_mean"]) / nrm["u_std"],
        (u1 - nrm["u_mean"]) / nrm["u_std"],
        (node_static - nrm["node_static_mean"]) / nrm["node_static_std"],
        (route - nrm["route_mean"]) / nrm["route_std"],
    ]
    if node_extra is not None:
        parts.append(node_extra)
    node = torch.cat(parts, dim=-1)
    edge_parts = [
        (ed[..., :3] - nrm["edge_dyn_mean"]) / nrm["edge_dyn_std"],
        (edge_static - nrm["edge_static_mean"]) / nrm["edge_static_std"],
    ]
    if ed.shape[-1] > 3:
        edge_parts.append((ed[..., 3:] - nrm["edge_dyn_mean"][2]) / nrm["edge_dyn_std"][2])
    edge = torch.cat(edge_parts, dim=-1)
    return node, edge


def interp_route(route_s: Tensor, route_f: Tensor, x: Tensor) -> Tensor:
    """Piecewise-linear lookup of route fields at arbitrary positions.

    ``route_s`` ``[B, R]``, ``route_f`` ``[B, R, F]``, ``x`` ``[B, N]`` ->
    ``[B, N, F]``. Needed because a rollout moves the train to positions that
    were never tabulated; the grid is 10 m, so the centimetre-scale position
    error a rollout accumulates is far below its resolution.
    """
    i = torch.searchsorted(route_s, x.contiguous()).clamp(1, route_s.shape[-1] - 1)
    s0 = torch.gather(route_s, 1, i - 1)
    s1 = torch.gather(route_s, 1, i)
    w = ((x - s0) / (s1 - s0).clamp_min(1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    idx = (i - 1).unsqueeze(-1).expand(-1, -1, route_f.shape[-1])
    f0 = torch.gather(route_f, 1, idx)
    f1 = torch.gather(route_f, 1, idx + 1)
    return f0 + w * (f1 - f0)


@dataclass
class Batch:
    node: Tensor        # [B, N, N_NODE_FEATURES]
    edge: Tensor        # [B, N-1, N_EDGE_FEATURES]
    abar: Tensor        # [B, N]    target, normalized
    dcorr: Tensor       # [B, N-1]  target, normalized
    n_vehicles: int
    st0: Tensor         # [B, N, 4]   raw state at k
    ed0: Tensor         # [B, N-1, 3] raw edge_dynamic at k
    # Kept so a batch can be rebuilt from a perturbed start state without
    # re-gathering from the block (noise injection, surrogate.rollout).
    node_static: Tensor
    edge_static: Tensor
    route: Tensor
    u0: Tensor
    u1: Tensor
    si: Tensor | None = None       # [B] scenario index within the block
    gear0: Tensor | None = None    # [B, N-1] gear state at k (v3)


class _Block:
    """Every loaded scenario of one consist size, resident on ``device``.

    Also the one place that knows how this corpus's physics turns a carried
    state into network inputs (:meth:`features`) and advances it
    (:meth:`advance`), so training, rollout and evaluation cannot disagree.
    """

    def __init__(self, n: int, recs: list[dict], norm: dict[str, Tensor],
                 device: str, *, h: float, tau_brk: float, tau_trac: float,
                 law: GearLaw | None = None) -> None:
        self.n = n
        self.device = device
        self.norm = norm
        self.h, self.tau_brk, self.tau_trac = h, tau_brk, tau_trac
        self.law = law
        self.v3 = law is not None
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
        if self.v3:
            self.gear = pad("gear_state", (n - 1,))
            stack = lambda key: torch.from_numpy(  # noqa: E731
                np.stack([r[key] for r in recs])).to(device)
            self.tau_apply = stack("tau_brk_node")
            self.tau_release = stack("tau_brk_release_node")
            self.loco_table = stack("loco_table")
            self.inertia = stack("inertia_factor")
            self.v_eps = torch.tensor([r["v_eps"] for r in recs],
                                      dtype=torch.float32, device=device)

        self.corridor = torch.tensor([r["corridor"] for r in recs],
                                     dtype=torch.long, device=device)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        pairs = [(i, k) for i, L in enumerate(lengths) for k in range(L - 1)]
        self.pairs = torch.tensor(pairs, dtype=torch.long, device=device)

    def __len__(self) -> int:
        return self.pairs.shape[0]

    # -- the corpus's physics on carried state -------------------------------------

    def carry(self, si: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Stored ``(state, delta, gear)`` at step ``k``; ``gear`` is None for legacy."""
        gear = self.gear[si, k] if self.v3 else None
        return self.state[si, k], self.edge_dyn[si, k][..., 0], gear

    def node_extra(self, st: Tensor, u0: Tensor, u1: Tensor, si: Tensor) -> Tensor | None:
        """v3 node inputs: locomotive force at the step's start and end notch, inertia.

        ``z_trac`` is a lagged notch under v3, and the force behind it is a
        table lookup in notch and speed (benchmark Tables 4-6), so the force
        is handed over rather than left for the network to learn. The end
        notch is exact (:func:`actuator_step`); both forces are taken at the
        start speed, the only one known. Scaled by the corpus's spread of
        ``F_trac_max_N``.
        """
        if not self.v3:
            return None
        from simulator2.torch_rhs import _notch_force

        lp = self.loco_table[si].to(st.dtype)
        v = st[..., 1]
        v_eps = self.v_eps[si, None].to(st.dtype)
        zt1 = actuator_step(st[..., 3], u0[..., 0], u1[..., 0], self.tau_trac, self.h)
        scale = self.norm["node_static_std"][5]
        return torch.stack([
            _notch_force(st[..., 3], v, lp, v_eps) / scale,
            _notch_force(zt1, v, lp, v_eps) / scale,
            (self.inertia[si].to(st.dtype) - 1.0) / INERTIA_SCALE,
        ], dim=-1)

    def features(self, st: Tensor, delta: Tensor, gear: Tensor | None, si: Tensor,
                 k: Tensor, route: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(node, edge, u0, u1)`` for a carried state at step ``k``."""
        u0, u1 = self.u[si, k], self.u[si, k + 1]
        es = self.edge_static[si]
        ed = coupler_features(delta, st[..., 1], es, gear, self.law)
        node, edge = build_features(st, ed, self.node_static[si], es, route, u0, u1,
                                    self.norm, self.node_extra(st, u0, u1, si))
        return node, edge, u0, u1

    def advance(self, st: Tensor, delta: Tensor, gear: Tensor | None, abar: Tensor,
                dcorr: Tensor, si: Tensor, k: Tensor,
                ) -> tuple[Tensor, Tensor, Tensor | None]:
        """One step from physical (de-normalized) predictions."""
        u0, u1 = self.u[si, k], self.u[si, k + 1]
        es = self.edge_static[si]
        if not self.v3:
            return advance(st, delta, abar, dcorr, u0, u1, es[..., 0],
                           tau_brk=self.tau_brk, tau_trac=self.tau_trac, h=self.h)
        return advance(st, delta, abar, dcorr, u0, u1, es[..., 0],
                       tau_brk=self.tau_apply[si], tau_brk_release=self.tau_release[si],
                       tau_trac=self.tau_trac, h=self.h,
                       gear=gear, law=self.law, slack_half=es[..., 1])

    def assemble(self, si: Tensor, k: Tensor, h: float,
                 tgt_std: dict[str, Tensor] | None) -> Batch:
        st0, st1 = self.state[si, k], self.state[si, k + 1]
        u0, u1 = self.u[si, k], self.u[si, k + 1]
        es = self.edge_static[si]
        ed0, ed1 = self.edge_dyn[si, k], self.edge_dyn[si, k + 1]
        nrm = self.norm
        gear0 = None
        ed_in = ed0
        if self.v3:
            # Stored edge channels as in legacy, plus the gear columns.
            gear0 = self.gear[si, k]
            lo, hi = self.law.envelope(ed0[..., 0], es[..., 1])
            ed_in = torch.cat([ed0, torch.stack([gear0, lo, hi], dim=-1)], dim=-1)

        node, edge = build_features(st0, ed_in, self.node_static[si], es,
                                    self.route[si, k], u0, u1, nrm,
                                    self.node_extra(st0, u0, u1, si))
        abar = node_targets(st0, st1, h)
        dcorr = edge_targets(ed0[..., 0], ed1[..., 0], st0, st1, h)
        if tgt_std is not None:
            abar = abar / tgt_std["abar"]
            dcorr = dcorr / tgt_std["dcorr"]
        return Batch(node=node, edge=edge, abar=abar, dcorr=dcorr,
                     n_vehicles=self.n, st0=st0, ed0=ed0,
                     node_static=self.node_static[si], edge_static=es,
                     route=self.route[si, k], u0=u0, u1=u1, si=si, gear0=gear0)


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
        self.v3 = _is_v3(self.root / "scenarios" / rows.iloc[0].route_id
                         / f"{rows.iloc[0].scenario_id}.npz")
        per_step = BYTES_PER_VEHICLE_STEP + (BYTES_GEAR_PER_STEP if self.v3 else 0)
        cost = (per_step * rows.N * rows["T"]).to_numpy()
        keep = np.cumsum(cost) <= budget_gib * 2 ** 30
        rows = rows[keep]
        if max_scenarios is not None:
            rows = rows.iloc[:max_scenarios]
        if rows.empty:
            raise ValueError(f"budget_gib={budget_gib} is too small for one scenario")
        self.rows = rows.reset_index(drop=True)
        self.gib = float(cost[keep][: len(self.rows)].sum() / 2 ** 30)

        norm = _load_norm(self.root, device)
        self.norm = norm
        routes: dict[str, dict] = {}
        corridor_id: dict[str, int] = {}
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
                if _is_v3_keys(f.files) != self.v3:
                    raise ValueError(f"{p.name}: mixes legacy and v3 physics in one set")
                if self.v3:
                    for key in V3_KEYS:
                        rec[key] = f[key].astype(np.float32)
                    rec["v_eps"] = float(f["v_eps"])
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
            rec["corridor"] = corridor_id.setdefault(r.route_id, len(corridor_id))
            by_n.setdefault(int(r.N), []).append(rec)

        # The five corridors, once, on the device: a rollout looks route fields
        # up at positions it predicts rather than positions the corpus stored.
        r_max = max(v["route_s"].shape[0] for v in routes.values())
        n_cor = len(corridor_id)
        s_tab = np.zeros((n_cor, r_max), dtype=np.float32)
        f_tab = np.zeros((n_cor, r_max, len(_ROUTE_FIELDS)), dtype=np.float32)
        for rid, ci in corridor_id.items():
            rt = routes[rid]
            n_r = rt["route_s"].shape[0]
            s_tab[ci, :n_r] = rt["route_s"]
            # Pad by repeating the last sample so a clamped lookup past the end
            # returns the final value rather than a zero.
            s_tab[ci, n_r:] = rt["route_s"][-1]
            for j, k in enumerate(_ROUTE_FIELDS):
                f_tab[ci, :n_r, j] = rt[k]
                f_tab[ci, n_r:, j] = rt[k][-1]
        self.route_s = torch.from_numpy(s_tab).to(device)
        self.route_f = torch.from_numpy(f_tab).to(device)

        self.h, self.tau_brk, self.tau_trac = float(h), float(tau_brk), float(tau_trac)
        # tau_brk is the legacy scalar lag; v3 files still carry it, unused,
        # and brake per vehicle (_Block.advance).
        self.law = GearLaw.default(device) if self.v3 else None
        self.n_node_features = N_NODE_FEATURES_V3 if self.v3 else N_NODE_FEATURES
        self.n_edge_features = N_EDGE_FEATURES_V3 if self.v3 else N_EDGE_FEATURES
        self.blocks = {n: _Block(n, recs, norm, device, h=self.h, tau_brk=self.tau_brk,
                                 tau_trac=self.tau_trac, law=self.law)
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

    def sample_starts(self, batch_size: int, gen: torch.Generator, span: int):
        """Draw ``(block, scenario, k)`` with at least ``span`` steps left.

        Plain :meth:`sample` only guarantees one step of headroom. Pushforward
        and multi-step training need a window, and a scenario too short for it
        has to be excluded rather than silently truncating the window.
        """
        bi = int(torch.multinomial(self.block_p, 1, generator=gen).item())
        blk = self.blocks[self.block_keys[bi]]
        room = blk.lengths - span - 1
        ok = torch.nonzero(room > 0).squeeze(-1)
        if ok.numel() == 0:
            return None
        si = ok[torch.randint(ok.numel(), (batch_size,), generator=gen,
                              device=self.device)]
        k0 = (torch.rand(batch_size, generator=gen, device=self.device)
              * room[si].float()).long()
        return blk, si, k0

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


#: Per-scenario arrays a v3 corpus adds that the surrogate needs: the gear
#: state, the brake lags per vehicle, the notch tables and the inertia factor.
V3_KEYS = ("gear_state", "tau_brk_node", "tau_brk_release_node", "loco_table",
           "inertia_factor")
#: Extra bytes per vehicle-step held for the gear state (float32, per coupler).
BYTES_GEAR_PER_STEP = 4


def _is_v3_keys(keys) -> bool:
    """All v3 arrays or none: the feature layout depends on every one of them."""
    present = [k in keys for k in V3_KEYS]
    if any(present) and not all(present):
        missing = [k for k, p in zip(V3_KEYS, present) if not p]
        raise ValueError(f"partial v3 scenario, missing {missing}; the surrogate "
                         "supports the legacy and the full v3 physics only")
    return all(present)


def _is_v3(path: Path) -> bool:
    with np.load(path) as f:
        return _is_v3_keys(f.files)


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
