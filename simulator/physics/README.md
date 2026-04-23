# Rail Longitudinal Dynamics Simulator (Physics Core)

Python package for simulating single-locomotive longitudinal dynamics along a route with grade and curvature. Built for batched rollouts on CPU/GPU with PyTorch; designed to extend to multiple vehicles and couplers later.

## What it does

- **Simulate** a single locomotive along a route with grade/curvature profiles (piecewise-linear).
- **Output** trajectories (position, speed) and per-force diagnostics for dataset generation.
- **Run** fast batched rollouts on CPU or GPU; all dynamics are vectorized over batch dimension B.
- **Sanity-check** via a small physics test suite (coastdown, grade, equilibrium, device parity).

## Requirements

- Python 3.11+
- PyTorch (CPU or CUDA)

## Install

From repo root (or from `simulator/physics`):

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Run tests

From `simulator/physics` (or repo root with `python -m pytest simulator/physics/tests`):

```bash
pytest -q
```

Or with verbose output:

```bash
pytest simulator/physics/tests -v
```

## Run demos

From **repo root** (`thesis/`):

```bash
python simulator/physics/scripts/demo_coastdown.py
python simulator/physics/scripts/demo_grade_climb.py
python simulator/physics/scripts/demo_constant_throttle.py
```

Each script prints numeric results; coastdown can optionally save a plot.

## Parameter meanings and units

See **UNITS.md** for the enforced unit system. Main parameter groups:

- **Locomotive**: mass (kg), Davis resistance A/B/C (N, N/(m/s), N/(m/s)²), traction limits (F_max N, P_max W, eta, mu, v_eps), brake (F_brake_max N, mu_brake).
- **Route**: grade and curvature as piecewise-linear functions of position x (m); grade in decimal, curvature in 1/m.
- **Control**: throttle and brake in [0, 1]; effective throttle = throttle*(1-brake), effective brake = brake*(1-throttle).

## Architecture

- `core/railphysics/`: library (route, params, forces, dynamics, integrators, rollout, diagnostics).
- `tests/`: pytest sanity tests.
- `scripts/`: standalone demos (no UI).

No transformer or UI in this package; physics core only.
