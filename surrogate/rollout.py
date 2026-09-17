"""Multi-step rollout: the model fed its own output.

Everything up to here is one-step and teacher-forced -- every prediction starts
from a state the simulator produced. A rollout starts from one true state and
then runs on its own predictions, which is what MPC will do and what the
Chapter 6 accuracy claim has to survive.

The failure mode is compounding: a step model is only ever trained on states
drawn from the true distribution, so as soon as its own small errors move it
off that distribution it sees inputs it was never shown, predicts worse, and
moves further off. :func:`add_noise` is the standard countermeasure from the
graph-network-simulator literature -- perturb the input state during training
and keep the target on the *true* next state, so the model spends training
learning to correct drift rather than only to continue clean trajectories.
"""

from __future__ import annotations

import torch
from torch import Tensor

from surrogate.data import build_features, interp_route
from surrogate.step import advance, coupler_features, edge_targets, node_targets


def add_noise(
    st: Tensor, delta: Tensor, gen: torch.Generator,
    *, v_std: float, delta_std: float,
) -> tuple[Tensor, Tensor]:
    """Perturb the carried state the way a rollout would have drifted.

    Noise goes on ``v`` and on the coupler stretch only. ``x`` is left alone:
    absolute position enters the model solely through route lookup on a 10 m
    grid, so perturbing it would do nothing, and the geometry that matters is
    carried in ``delta``. The actuator channels are exact and never drift.
    """
    if v_std <= 0.0 and delta_std <= 0.0:
        return st, delta
    st = st.clone()
    if v_std > 0.0:
        st[..., 1] += torch.randn(st[..., 1].shape, generator=gen,
                                  device=st.device, dtype=st.dtype) * v_std
    if delta_std > 0.0:
        delta = delta + torch.randn(delta.shape, generator=gen,
                                    device=delta.device, dtype=delta.dtype) * delta_std
    return st, delta


def noisy_targets(
    st_pert: Tensor, delta_pert: Tensor, st1: Tensor, delta1: Tensor, h: float,
) -> tuple[Tensor, Tensor]:
    """Targets that carry a perturbed state back onto the true next state.

    This is the whole point of the noise: the label is recomputed against the
    *perturbed* start, so the model is asked for the correction that lands on
    the truth, not for the correction it would have made from a clean state.
    """
    abar = node_targets(st_pert, st1, h)
    dcorr = edge_targets(delta_pert, delta1, st_pert, st1, h)
    return abar, dcorr


@torch.no_grad()
def rollout(
    model, blk, si: Tensor, k0: Tensor, horizon: int,
    *, route_s: Tensor, route_f: Tensor, nrm: dict[str, Tensor],
    h: float, tau_brk: float, tau_trac: float,
    tgt_std: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Run ``horizon`` steps from ``(si, k0)`` on the model's own predictions.

    Returns per-step absolute errors against the stored trajectory, shaped
    ``[horizon, B]``: ``v`` and ``delta`` as max-over-consist, plus the
    zero-order baseline of simply holding the initial state.
    """
    es = blk.edge_static[si]
    ns = blk.node_static[si]
    l0 = es[..., 0]
    rs = route_s[blk.corridor[si]]
    rf = route_f[blk.corridor[si]]

    st = blk.state[si, k0]
    delta = blk.edge_dyn[si, k0][..., 0]
    st_hold, delta_hold = st, delta

    v_err, d_err, v_hold, d_hold = [], [], [], []
    for j in range(horizon):
        kj = k0 + j
        u0, u1 = blk.u[si, kj], blk.u[si, kj + 1]
        ed = coupler_features(delta, st[..., 1], es)
        route = interp_route(rs, rf, st[..., 0])
        node, edge = build_features(st, ed, ns, es, route, u0, u1, nrm)

        a_hat, d_hat = model(node, edge)
        st, delta = advance(st, delta, a_hat * tgt_std["abar"],
                            d_hat * tgt_std["dcorr"], u0, u1, l0,
                            tau_brk=tau_brk, tau_trac=tau_trac, h=h)

        st_true = blk.state[si, kj + 1]
        d_true = blk.edge_dyn[si, kj + 1][..., 0]
        v_err.append((st[..., 1] - st_true[..., 1]).abs().amax(-1))
        d_err.append((delta - d_true).abs().amax(-1))
        v_hold.append((st_hold[..., 1] - st_true[..., 1]).abs().amax(-1))
        d_hold.append((delta_hold - d_true).abs().amax(-1))

    return {
        "v": torch.stack(v_err), "delta": torch.stack(d_err),
        "v_hold": torch.stack(v_hold), "delta_hold": torch.stack(d_hold),
    }


@torch.no_grad()
def rollout_metrics(
    model, ds, tgt_std, horizon: int, *, n_batches: int = 8,
    batch_size: int = 128, seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Median rollout error at a few horizons, against holding the start state.

    Start points are sampled with enough room left in the scenario, so a short
    scenario does not silently shorten the horizon.
    """
    model.eval()
    gen = torch.Generator(device=ds.device).manual_seed(seed)
    acc = {k: [] for k in ("v", "delta", "v_hold", "delta_hold")}
    for _ in range(n_batches):
        bi = int(torch.multinomial(ds.block_p, 1, generator=gen).item())
        blk = ds.blocks[ds.block_keys[bi]]
        room = blk.lengths - horizon - 1
        ok = torch.nonzero(room > 0).squeeze(-1)
        if ok.numel() == 0:
            continue
        si = ok[torch.randint(ok.numel(), (batch_size,), generator=gen,
                              device=ds.device)]
        k0 = (torch.rand(batch_size, generator=gen, device=ds.device)
              * room[si].float()).long()
        out = rollout(model, blk, si, k0, horizon,
                      route_s=ds.route_s, route_f=ds.route_f, nrm=ds.norm,
                      h=ds.h, tau_brk=ds.tau_brk, tau_trac=ds.tau_trac,
                      tgt_std=tgt_std)
        for k in acc:
            acc[k].append(out[k])
    model.train()
    if not acc["v"]:
        return {}
    cat = {k: torch.cat(v, dim=1) for k, v in acc.items()}   # [horizon, B*]
    return {
        j + 1: {
            "v": float(cat["v"][j].median()),
            "v_hold": float(cat["v_hold"][j].median()),
            "delta": float(cat["delta"][j].median()),
            "delta_hold": float(cat["delta_hold"][j].median()),
        }
        for j in range(cat["v"].shape[0])
    }


def training_batch(
    model, ds, blk, si: Tensor, k0: Tensor,
    *, n_push: int, gen: torch.Generator, v_std: float, delta_std: float,
    tgt_std: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One supervised step, optionally started from the model's own drift.

    With ``n_push = 0`` this is ordinary one-step training (plus Gaussian noise
    if asked for). With ``n_push > 0`` the model first takes that many steps on
    its own output with no gradient, and only the step *after* that is
    supervised -- the pushforward trick.

    The point is that Gaussian noise is a guess at what the model's error looks
    like, while the pushforward uses the real thing: the perturbation is drawn
    from the model's own error distribution at its current level of training,
    and it tightens automatically as the model improves. Gradients are not
    taken through the unrolled steps, so the cost is one extra forward pass per
    push and memory does not grow with horizon.
    """
    es = blk.edge_static[si]
    ns = blk.node_static[si]
    l0 = es[..., 0]
    rs = ds.route_s[blk.corridor[si]]
    rf = ds.route_f[blk.corridor[si]]

    st = blk.state[si, k0]
    delta = blk.edge_dyn[si, k0][..., 0]
    st, delta = add_noise(st, delta, gen, v_std=v_std, delta_std=delta_std)

    def features(state, d, kj):
        u0, u1 = blk.u[si, kj], blk.u[si, kj + 1]
        ed = coupler_features(d, state[..., 1], es)
        route = interp_route(rs, rf, state[..., 0])
        return (*build_features(state, ed, ns, es, route, u0, u1, ds.norm), u0, u1)

    if n_push > 0:
        with torch.no_grad():
            for j in range(n_push):
                node, edge, u0, u1 = features(st, delta, k0 + j)
                a, d = model(node, edge)
                st, delta = advance(st, delta, a * tgt_std["abar"],
                                    d * tgt_std["dcorr"], u0, u1, l0,
                                    tau_brk=ds.tau_brk, tau_trac=ds.tau_trac,
                                    h=ds.h)

    kj = k0 + n_push
    node, edge, _, _ = features(st, delta, kj)
    st1 = blk.state[si, kj + 1]
    d1 = blk.edge_dyn[si, kj + 1][..., 0]
    abar, dcorr = noisy_targets(st, delta, st1, d1, ds.h)
    return node, edge, abar / tgt_std["abar"], dcorr / tgt_std["dcorr"]
