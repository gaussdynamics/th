#!/usr/bin/env python3
"""Finding F2, measured: braking run-in with and without brake propagation.

A uniform train of loaded cars at 20 m/s takes a full-service application.
If every car brakes at once (the Chapter 4 model) and every car is the same,
every car decelerates together and the couplers carry no run-in force. With
the brake pipe of :mod:`simulator2.air_brake`, the head brakes first and the
tail runs in against it. The script reports the peak compressive (buff)
coupler force both ways, for the legacy Kelvin-Voigt coupler and the friction
draft gear, at two train lengths.

    python scripts/air_brake_run_in.py

Cars are 100 t on a 15 m pitch, as the 2026-10-04 and 2026-10-07 runs in
SIMULATOR_VALIDATION_NOTE.md, so the figures compare with those; the brake
capacity is the sourced default (``AirBrakeParams``), 38.2 kN per loaded car.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator2.air_brake import AirBrakeParams, vehicle_commands  # noqa: E402
from simulator2.draft_gear import draft_gear_stateful_torch, load_table  # noqa: E402
from simulator2.physics_config import V3_COUPLER_SLACK_HALF_M  # noqa: E402
from simulator2.verification import Case, rk4_fixed  # noqa: E402

MASS_KG = 100_000.0
PITCH_M = 15.0
V0_MPS = 20.0
T_END_S = 60.0
DT_S = 0.005

#: The legacy coupler of the Chapter 4 corpora (Kelvin-Voigt with a deadband).
LEGACY_COUPLER = dict(slack=0.0127, k_draft=9e6, c_draft=5e5, k_buff=12e6, c_buff=7e5)


def at_once(p: AirBrakeParams) -> AirBrakeParams:
    """The same brake, reaching every car at the same instant."""
    return replace(p, propagation_mps=1e9, release_propagation_mps=1e8,
                   pipe_diffusivity_m2ps=1e12)


def peak_buff_kn(n: int, p: AirBrakeParams, coupler: str) -> tuple[float, float]:
    """Peak buff force over the train [kN], and the car where it occurs."""
    t = np.arange(0.0, T_END_S, 0.01)
    driver = np.where(t >= 1.0, 1.0, 0.0)
    x0 = -PITCH_M * np.arange(n)
    mass = np.full(n, MASS_KG)
    u, tau, tau_rel, _ = vehicle_commands(t, driver, x0, mass, np.zeros(n, bool), p)
    kw = dict(mass=mass, x0=x0, v0=np.full(n, V0_MPS), L0=PITCH_M, cmd_t=t,
              u_trac=np.zeros_like(u), u_brk=u, f_brk_max=1e12,
              tau_brk_node=tau, tau_brk_release_node=tau_rel)
    if coupler == "kelvin_voigt":
        case = Case(**kw, **LEGACY_COUPLER)
    else:
        case = Case(**kw, coupler_model="draft_gear", slack=V3_COUPLER_SLACK_HALF_M,
                    draft_gear_table=np.asarray(load_table()))
    _, y = rk4_fixed(case.batch(), T_END_S, DT_S, every=4)
    if coupler == "kelvin_voigt":
        delta = y[:, :n][:, :-1] - y[:, :n][:, 1:] - PITCH_M
        f = -np.clip(-(delta + LEGACY_COUPLER["slack"]), 0.0, None) * LEGACY_COUPLER["k_buff"]
    else:
        x, v = y[:, :n], y[:, n:2 * n]
        f64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
        with torch.no_grad():
            f, _ = draft_gear_stateful_torch(
                f64(x[:, :-1] - x[:, 1:] - PITCH_M), f64(v[:, :-1] - v[:, 1:]),
                f64(y[:, 4 * n:4 * n + n - 1]), f64(np.full(n - 1, V3_COUPLER_SLACK_HALF_M)),
                f64(load_table()), k_transition=case.draft_gear_k_transition,
                k_lock=case.draft_gear_k_lock)
        f = f.numpy()                                        # tension > 0
    buff = np.clip(-f, 0.0, None)
    k = int(np.argmax(buff.max(axis=0)))
    return float(buff.max()) / 1e3, k + 1


def main() -> None:
    p = AirBrakeParams()
    print(f"loaded {MASS_KG / 1e3:.0f} t cars, {PITCH_M:.0f} m pitch, full service from "
          f"{V0_MPS:.0f} m/s, pipe factor {p.pipe_length_factor}")
    print(f"{'cars':>5} {'coupler':>13} {'at once':>10} {'propagated':>12}  (peak buff, kN)")
    for n in (60, 130):
        for coupler in ("kelvin_voigt", "draft_gear"):
            once, _ = peak_buff_kn(n, at_once(p), coupler)
            prop, where = peak_buff_kn(n, p, coupler)
            print(f"{n:>5} {coupler:>13} {once:>10.1f} {prop:>12.1f}  (coupler {where})")


if __name__ == "__main__":
    main()
