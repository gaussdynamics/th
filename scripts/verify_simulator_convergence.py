#!/usr/bin/env python3
"""Step-size convergence of the production integrator on a realistic consist.

Companion to ``simulator2/tests/test_verification.py``, too slow to be a test
(~3 min). Runs ``realistic_case`` under fixed-step RK4 at a ladder of steps
and compares each against a reference at ``dt = 0.000625 s``, itself checked
against ``0.00125 s``.

The distinction the numbers make: away from coupler contact events RK4 is
fourth order and the typical error is ~1e-6 m/s; at a contact event the
force jumps by ``c * delta_dot`` (the damper engages discontinuously), the
impact is resolved poorly, and the *maximum* error is set by a few such events
and does not fall steadily with ``dt``. Report both, never only one.

Run::

    python scripts/verify_simulator_convergence.py                 # Kelvin-Voigt
    python scripts/verify_simulator_convergence.py --draft-gear    # friction gear
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator2.verification import realistic_case, rk4_fixed  # noqa: E402

T_END = 60.0
SAMPLE = 0.02       # compare on the production grid
N = 10


def run(dt: float, coupler_model: str) -> np.ndarray:
    every = max(1, int(round(SAMPLE / dt)))
    case = realistic_case(N, T_END, coupler_model=coupler_model)
    _, y = rk4_fixed(case.batch(), T_END, dt, every=every)
    return y


def main() -> int:
    model = "draft_gear" if "--draft-gear" in sys.argv else "kelvin_voigt"
    run_ = lambda dt: run(dt, model)   # noqa: E731
    print(f"coupler model: {model}")
    ref = run_(0.000625)
    ref2 = run_(0.00125)
    print(f"reference self-check: max|dv| {np.abs(ref2 - ref)[:, N:2*N].max():.2e} m/s\n")
    print(f"{'dt [s]':>8} {'max |dv|':>10} {'p99 |dv|':>10} {'median |dv|':>12} {'final |dx|':>11}")
    for dt in (0.02, 0.01, 0.005, 0.0025):
        y = run_(dt)
        e = np.abs(y[:, N:2 * N] - ref[:, N:2 * N])
        print(f"{dt:8.4f} {e.max():10.2e} {np.percentile(e, 99):10.2e} "
              f"{np.median(e):12.2e} {abs(y[-1, 0] - ref[-1, 0]):11.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
