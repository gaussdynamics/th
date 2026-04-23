"""
Physics sanity tests: deterministic, fixed seeds.
"""
import pytest
import torch

from railphysics import (
    State,
    LocomotiveParams,
    default_loco_params,
    RouteProfileTorch,
    level_route,
    rollout_constant_action,
    step,
    compute_forces,
)


def _set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def level_route_fixture():
    return level_route(length_m=100_000.0)


@pytest.fixture
def params_fixture():
    return default_loco_params(batch_size=1)


def test_coastdown_on_level(level_route_fixture, params_fixture):
    """Coastdown: throttle=0, brake=0, v0=20. v must be non-increasing and end near 0 (< 1 m/s)."""
    _set_seed(42)
    B = 1
    x0 = torch.zeros(B)
    v0 = torch.full((B,), 20.0)
    state = State(x=x0, v=v0)
    dt = 0.5
    # Enough steps for Davis resistance to bring 20 m/s down to < 1 m/s
    num_steps = 2000
    states, _ = rollout_constant_action(
        state, throttle=0.0, brake=0.0,
        route=level_route_fixture, params=params_fixture,
        dt=dt, num_steps=num_steps,
    )
    v_traj = states["v"]
    # v is monotonically non-increasing
    for i in range(v_traj.shape[0] - 1):
        assert (v_traj[i + 1] <= v_traj[i] + 1e-6).all(), f"v increased at step {i}"
    # final speed near 0
    assert v_traj[-1].item() < 1.0, f"final v should be < 1 m/s, got {v_traj[-1].item()}"


def test_grade_reduces_acceleration(params_fixture):
    """One step: same throttle, grade=0 vs grade=0.01. v_next_grade < v_next_level."""
    _set_seed(42)
    B = 1
    x0 = torch.zeros(B)
    v0 = torch.full((B,), 5.0)
    throttle = torch.full((B,), 0.3)
    brake = torch.zeros(B)
    dt = 0.5

    route_level = level_route(length_m=10_000.0)
    route_grade = RouteProfileTorch(
        x_m=torch.tensor([0.0, 10_000.0]),
        grade=torch.tensor([0.01, 0.01]),
        curvature=torch.tensor([0.0, 0.0]),
    )

    _, v_level, _ = step(x0, v0, throttle, brake, route_level, params_fixture, dt)
    _, v_grade, _ = step(x0, v0, throttle, brake, route_grade, params_fixture, dt)

    assert (v_grade < v_level).all(), "grade should reduce acceleration (v_next lower)"


def test_constant_throttle_equilibrium(level_route_fixture, params_fixture):
    """Level track, small constant throttle, long run. Mean |dv/dt| in last 20 steps should be small."""
    _set_seed(42)
    B = 1
    x0 = torch.zeros(B)
    v0 = torch.zeros(B)
    dt = 0.5
    num_steps = 800
    states, breakdowns = rollout_constant_action(
        initial_state=State(x=x0, v=v0),
        throttle=0.2,
        brake=0.0,
        route=level_route_fixture,
        params=params_fixture,
        dt=dt,
        num_steps=num_steps,
    )
    # acceleration from last 20 steps: a = F_net / m
    p = params_fixture.for_batch(B)
    a_last = torch.stack([breakdowns[i].F_net / p.mass for i in range(num_steps - 20, num_steps)])
    mean_abs_a = a_last.abs().mean().item()
    tol = 0.08
    assert mean_abs_a < tol, f"mean |dv/dt| in last 20 steps should be < {tol}, got {mean_abs_a}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_parity():
    """Short rollout on CPU and GPU; results should be close."""
    _set_seed(42)
    B = 2
    num_steps = 20
    dt = 0.5
    route_cpu = level_route(length_m=50_000.0)
    params_cpu = default_loco_params(device=torch.device("cpu"), batch_size=B)
    state_cpu = State(
        x=torch.zeros(B),
        v=torch.full((B,), 10.0),
    )
    states_cpu, _ = rollout_constant_action(
        state_cpu, throttle=0.25, brake=0.0,
        route=route_cpu, params=params_cpu,
        dt=dt, num_steps=num_steps,
    )

    route_gpu = level_route(length_m=50_000.0, device=torch.device("cuda"))
    params_gpu = default_loco_params(device=torch.device("cuda"), batch_size=B)
    state_gpu = State(
        x=torch.zeros(B, device="cuda"),
        v=torch.full((B,), 10.0, device="cuda"),
    )
    states_gpu, _ = rollout_constant_action(
        state_gpu, throttle=0.25, brake=0.0,
        route=route_gpu, params=params_gpu,
        dt=dt, num_steps=num_steps,
    )

    torch.testing.assert_close(states_cpu["x"], states_gpu["x"].cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(states_cpu["v"], states_gpu["v"].cpu(), rtol=1e-5, atol=1e-5)
