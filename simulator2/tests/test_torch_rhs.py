"""Acceptance contract for the batched torch RHS port.

**These tests are expected to FAIL until ``simulator2/torch_rhs.py`` exists.**
They are the definition of "done" for the port, written before it, so that
"faster" cannot quietly mean "different". Do not weaken a tolerance to make a
test pass -- if a tolerance is genuinely wrong, say so and change it
deliberately, with the reason recorded.

See ``TORCH_PORT_SPEC.md`` for the design rationale behind each requirement.

The reference trajectories live in ``fixtures/torch_reference/*.npz``, generated
from the NumPy path by ``fixtures/generate_torch_fixtures.py``. Each fixture is
self-describing: it carries the consist, route, materialized commands, physics
constants, and the reference solution, so nothing here imports the NumPy
simulator to check the torch one.

Expected API (this is the contract):

    from simulator2.torch_rhs import TorchScenarioBatch, torch_rhs, rollout_rk4

    batch = TorchScenarioBatch.from_fixtures(paths, device="cuda", dtype=torch.float32)
    dy    = torch_rhs(t, y, batch)                 # y: (B, 4N) -> dy: (B, 4N)
    t_out, y_out = rollout_rk4(batch, y0, t_grid, dt)   # y_out: (B, T, 4N)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "torch_reference").glob("*.npz"))

torch = pytest.importorskip("torch", reason="torch is required for the port's tests")

try:  # noqa: SIM105
    from simulator2.torch_rhs import TorchScenarioBatch, rollout_rk4, torch_rhs

    HAVE_PORT = True
    IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    HAVE_PORT = False
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

NOT_BUILT = f"simulator2/torch_rhs.py not implemented yet ({IMPORT_ERROR})"
requires_port = pytest.mark.skipif(not HAVE_PORT, reason=NOT_BUILT)
requires_cuda = pytest.mark.skipif(
    not (HAVE_PORT and torch.cuda.is_available()), reason="CUDA device required"
)

# Fixtures whose reference was generated at the simulator's default tolerance
# because the adaptive solver cannot integrate a held-brake standstill
# efficiently. Their references are themselves noisier, so comparisons are
# scaled accordingly -- see the note in generate_torch_fixtures.py.
LOOSE = {"regime_emergency", "adversarial_brake_to_stop", "brake_flag_disabled",
         "regime_brake_release", "regime_startup"}

# Accuracy targets. Calibrated to "good enough to train a surrogate on", not to
# machine precision: the point of the fixed step is to trade a little accuracy
# for a bounded, regime-independent cost.
TOL_V_MPS = 0.05          # max abs velocity error vs reference
TOL_X_M = 1.0             # max abs position error over the window
TOL_FPEAK_REL = 0.05      # peak |F_coupler| within 5 % -- the safety-critical quantity
LOOSE_SCALE = 4.0


def _name(p: Path) -> str:
    return p.stem


def _tol(path: Path, base: float) -> float:
    return base * (LOOSE_SCALE if _name(path) in LOOSE else 1.0)


def _load(path: Path) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def test_fixtures_present() -> None:
    """Guards against a silently empty test suite."""
    assert len(FIXTURES) >= 13, f"expected the full fixture set, found {len(FIXTURES)}"
    names = {_name(p) for p in FIXTURES}
    assert "adversarial_slack_cycling" in names
    assert "adversarial_brake_to_stop" in names
    assert "brake_flag_disabled" in names


def test_port_module_exists() -> None:
    assert HAVE_PORT, NOT_BUILT


# ---------------------------------------------------------------------------
# RHS equivalence
# ---------------------------------------------------------------------------

@requires_port
@pytest.mark.parametrize("path", FIXTURES, ids=_name)
def test_rhs_matches_numpy_reference(path: Path) -> None:
    """The torch RHS must reproduce the NumPy RHS elementwise in float64.

    Evaluated along the reference trajectory, not just at t=0, so deadband
    states, standstill, and saturated actuators are all covered.
    """
    fx = _load(path)
    batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=torch.float64)
    t = fx["t"]
    y = fx["y"].T  # (T, 4N)
    idx = np.linspace(0, len(t) - 1, 40).astype(int)
    for k in idx:
        yk = torch.as_tensor(y[k][None, :], dtype=torch.float64)
        dy = torch_rhs(float(t[k]), yk, batch).detach().cpu().numpy()[0]
        ref = _numpy_rhs_from_fixture(fx, float(t[k]), y[k])
        scale = max(1.0, float(np.abs(ref).max()))
        assert np.abs(dy - ref).max() / scale < 1e-9, f"t={t[k]:.2f}"


@requires_port
def test_batched_rhs_equals_individual_calls() -> None:
    """Batching must not couple scenarios."""
    paths = FIXTURES[:6]
    batch = TorchScenarioBatch.from_fixtures(paths, device="cpu", dtype=torch.float64)
    ys = [torch.as_tensor(_load(p)["y0"], dtype=torch.float64) for p in paths]
    n_max = max(y.numel() for y in ys)
    y_pad = torch.zeros(len(paths), n_max, dtype=torch.float64)
    for i, y in enumerate(ys):
        y_pad[i, : y.numel()] = y
    dy_batched = torch_rhs(0.0, y_pad, batch)
    for i, p in enumerate(paths):
        single = TorchScenarioBatch.from_fixtures([p], device="cpu", dtype=torch.float64)
        dy_single = torch_rhs(0.0, ys[i][None, :], single)
        m = ys[i].numel()
        assert torch.allclose(dy_batched[i, :m], dy_single[0, :m], atol=1e-12)


@requires_port
def test_padding_does_not_leak() -> None:
    """Mixed-N batches are padded; padded slots must contribute nothing."""
    paths = FIXTURES[:4]
    batch = TorchScenarioBatch.from_fixtures(paths, device="cpu", dtype=torch.float64)
    assert hasattr(batch, "node_mask"), "batch must expose node_mask for ragged N"
    y0 = batch.y0.clone()
    dy = torch_rhs(0.0, y0, batch)
    b, n = batch.node_mask.shape
    dv = dy[:, n : 2 * n]
    assert torch.all(dv[~batch.node_mask] == 0.0), "padded vehicles produced acceleration"


# ---------------------------------------------------------------------------
# Rollout accuracy
# ---------------------------------------------------------------------------

@requires_port
@pytest.mark.parametrize("path", FIXTURES, ids=_name)
def test_rollout_matches_reference_trajectory(path: Path) -> None:
    fx = _load(path)
    batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=torch.float64)
    t_grid = torch.as_tensor(fx["t"], dtype=torch.float64)
    y0 = torch.as_tensor(fx["y0"][None, :], dtype=torch.float64)
    _, y_out = rollout_rk4(batch, y0, t_grid, dt=0.02)
    y_out = y_out.detach().cpu().numpy()[0]  # (T, 4N)
    n = int(fx["N"])
    ref = fx["y"].T
    dx = np.abs(y_out[:, :n] - ref[:, :n]).max()
    dv = np.abs(y_out[:, n : 2 * n] - ref[:, n : 2 * n]).max()
    assert dv < _tol(path, TOL_V_MPS), f"velocity error {dv:.4f} m/s"
    assert dx < _tol(path, TOL_X_M), f"position error {dx:.4f} m"


@requires_port
@pytest.mark.parametrize("path", FIXTURES, ids=_name)
def test_peak_coupler_force_preserved(path: Path) -> None:
    """Peak coupler force is the safety constraint the control chapter uses,
    so it gets its own check rather than hiding inside a trajectory norm."""
    fx = _load(path)
    batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=torch.float64)
    t_grid = torch.as_tensor(fx["t"], dtype=torch.float64)
    y0 = torch.as_tensor(fx["y0"][None, :], dtype=torch.float64)
    _, y_out = rollout_rk4(batch, y0, t_grid, dt=0.02)
    f_ref = np.abs(fx["E_hist"][:, :, 2]).max()
    f_got = _coupler_force_from_state(fx, y_out.detach().cpu().numpy()[0])
    rel = abs(f_got - f_ref) / max(f_ref, 1.0)
    assert rel < _tol(path, TOL_FPEAK_REL), f"peak |F| {f_got:.0f} vs {f_ref:.0f} N ({rel:.1%})"


@requires_port
def test_deadband_crossings_tracked() -> None:
    """The slack-cycling fixture is the case a fixed step is most likely to
    miss: repeated entry and exit from the coupler deadband."""
    path = next(p for p in FIXTURES if _name(p) == "adversarial_slack_cycling")
    fx = _load(path)
    batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=torch.float64)
    t_grid = torch.as_tensor(fx["t"], dtype=torch.float64)
    y0 = torch.as_tensor(fx["y0"][None, :], dtype=torch.float64)
    _, y_out = rollout_rk4(batch, y0, t_grid, dt=0.02)
    got = y_out.detach().cpu().numpy()[0]
    n = int(fx["N"])
    slack = float(fx["edge_static"][0, 1])
    ref_delta = _deltas(fx, fx["y"].T)
    got_delta = _deltas(fx, got)
    ref_engaged = np.abs(ref_delta) > slack
    got_engaged = np.abs(got_delta) > slack
    agree = float((ref_engaged == got_engaged).mean())
    assert agree > 0.98, f"deadband engagement agreed on only {agree:.1%} of samples"


@requires_port
def test_standstill_is_held_not_reversed() -> None:
    path = next(p for p in FIXTURES if _name(p) == "adversarial_brake_to_stop")
    fx = _load(path)
    batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=torch.float64)
    t_grid = torch.as_tensor(fx["t"], dtype=torch.float64)
    y0 = torch.as_tensor(fx["y0"][None, :], dtype=torch.float64)
    _, y_out = rollout_rk4(batch, y0, t_grid, dt=0.02)
    n = int(fx["N"])
    v = y_out.detach().cpu().numpy()[0][:, n : 2 * n]
    assert v.min() > -0.20, f"consist reversed to {v.min():.3f} m/s under brake"
    assert abs(v[-1].mean()) < 0.20, "consist did not settle at standstill"


@requires_port
def test_brake_flag_disabled_reproduces_historical_behaviour() -> None:
    """The flag must still switch the physics, or old results are irreproducible.
    This fixture's reference reverses to about -26 m/s, which is the bug."""
    path = next(p for p in FIXTURES if _name(p) == "brake_flag_disabled")
    fx = _load(path)
    assert not bool(fx["brake_opposes_motion"])
    n = int(fx["N"])
    assert fx["y"].T[:, n : 2 * n].min() < -20.0, "fixture no longer captures the old behaviour"
    batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=torch.float64)
    t_grid = torch.as_tensor(fx["t"], dtype=torch.float64)
    y0 = torch.as_tensor(fx["y0"][None, :], dtype=torch.float64)
    _, y_out = rollout_rk4(batch, y0, t_grid, dt=0.02)
    v = y_out.detach().cpu().numpy()[0][:, n : 2 * n]
    assert v.min() < -20.0, "flag=False did not reproduce the historical reversal"


# ---------------------------------------------------------------------------
# Differentiability -- the reason this port exists at all
# ---------------------------------------------------------------------------

@requires_port
def test_rhs_gradcheck() -> None:
    """Analytic gradients must match numerical ones.

    Evaluated at a deliberately perturbed state rather than straight at a
    fixture's ``y0``. The model is piecewise in three places -- ``max(z, 0)``
    on the actuator states, the piecewise-linear route field, and the coupler
    deadband -- and every fixture's ``y0`` sits exactly on at least one of
    those kinks: actuators start settled at a zero command, and vehicles are
    placed at exact multiples of the 10 m route-node spacing (``x0`` is
    3000, 2980, ... on a route sampled every 10 m). A central difference
    straddling a kink returns the average of the two one-sided slopes, which
    cannot equal any single analytic subgradient, so gradcheck at those points
    fails for *any* faithful implementation rather than testing anything.

    ``regime_stretch_brake`` is the one fixture holding brake and traction
    simultaneously, so both actuator states are strictly positive; the
    position offset moves the consist off the route lattice. Together they put
    the evaluation point where the model is genuinely differentiable, which is
    what this test is meant to check.
    """
    path = next(p for p in FIXTURES if _name(p) == "regime_stretch_brake")
    fx = _load(path)
    batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=torch.float64)
    y0 = fx["y0"][None, :].copy()
    y0[:, : int(fx["N"])] += 3.7  # off the 10 m route-node lattice
    y = torch.tensor(y0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda yy: torch_rhs(1.0, yy, batch), (y,), eps=1e-6, atol=1e-6, rtol=1e-4
    )


@requires_port
def test_gradients_flow_through_rollout() -> None:
    """A short rollout must be differentiable end to end with finite grads --
    this is what the physics-residual surrogate and the MPC demo depend on."""
    path = next(p for p in FIXTURES if _name(p) == "regime_cruise")
    fx = _load(path)
    batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=torch.float64)
    y0 = torch.tensor(fx["y0"][None, :], dtype=torch.float64, requires_grad=True)
    t_grid = torch.as_tensor(fx["t"][:101], dtype=torch.float64)
    _, y_out = rollout_rk4(batch, y0, t_grid, dt=0.02)
    y_out.pow(2).sum().backward()
    assert y0.grad is not None and torch.isfinite(y0.grad).all()
    assert y0.grad.abs().sum() > 0


@requires_port
def test_no_inplace_breaks_autograd() -> None:
    path = FIXTURES[0]
    batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=torch.float64)
    y = torch.tensor(_load(path)["y0"][None, :], dtype=torch.float64, requires_grad=True)
    with torch.autograd.set_detect_anomaly(True):
        torch_rhs(1.0, y, batch).sum().backward()


# ---------------------------------------------------------------------------
# Device, dtype, determinism
# ---------------------------------------------------------------------------

@requires_port
def test_determinism() -> None:
    path = FIXTURES[0]
    fx = _load(path)
    batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=torch.float64)
    y0 = torch.as_tensor(fx["y0"][None, :], dtype=torch.float64)
    t_grid = torch.as_tensor(fx["t"][:201], dtype=torch.float64)
    a = rollout_rk4(batch, y0, t_grid, dt=0.02)[1]
    b = rollout_rk4(batch, y0, t_grid, dt=0.02)[1]
    assert torch.equal(a, b)


@requires_cuda
def test_cuda_matches_cpu() -> None:
    path = next(p for p in FIXTURES if _name(p) == "regime_cruise")
    fx = _load(path)
    t_grid_np, y0_np = fx["t"][:201], fx["y0"][None, :]
    out = {}
    for dev in ("cpu", "cuda"):
        batch = TorchScenarioBatch.from_fixtures([path], device=dev, dtype=torch.float64)
        y0 = torch.as_tensor(y0_np, dtype=torch.float64, device=dev)
        t_grid = torch.as_tensor(t_grid_np, dtype=torch.float64, device=dev)
        out[dev] = rollout_rk4(batch, y0, t_grid, dt=0.02)[1].detach().cpu()
    assert torch.allclose(out["cpu"], out["cuda"], atol=1e-8, rtol=1e-8)


@requires_port
@pytest.mark.parametrize("path", [p for p in FIXTURES if _name(p) == "regime_cruise"], ids=_name)
def test_float32_error_is_acceptable(path: Path) -> None:
    """A 4090 runs float64 at roughly 1/64 of float32 throughput, so the port
    is expected to run in float32. That is only acceptable if the error stays
    within the training-data tolerance -- measured here, not assumed.
    """
    fx = _load(path)
    t_grid_np, y0_np = fx["t"], fx["y0"][None, :]
    res = {}
    for dt_ in (torch.float32, torch.float64):
        batch = TorchScenarioBatch.from_fixtures([path], device="cpu", dtype=dt_)
        y0 = torch.as_tensor(y0_np, dtype=dt_)
        t_grid = torch.as_tensor(t_grid_np, dtype=dt_)
        res[dt_] = rollout_rk4(batch, y0, t_grid, dt=0.02)[1].double().cpu().numpy()[0]
    n = int(fx["N"])
    dv = np.abs(res[torch.float32][:, n : 2 * n] - res[torch.float64][:, n : 2 * n]).max()
    assert dv < TOL_V_MPS, (
        f"float32 diverges from float64 by {dv:.4f} m/s. If this fails, report it "
        "and switch the default dtype rather than loosening the tolerance."
    )


# ---------------------------------------------------------------------------
# Performance gate
# ---------------------------------------------------------------------------

@requires_cuda
def test_throughput_gate_on_gpu() -> None:
    """The port exists to make dataset generation feasible at large N.

    Baseline to beat: the NumPy path could not finish a single N=130, 120 s run
    inside 30 s. This gate asks for a batch of 64 such runs in under 20 s, i.e.
    roughly 0.3 s per scenario -- about a 100x per-scenario improvement, which
    is deliberately conservative against what the hardware should manage.
    """
    import time

    b, n, sim_s, dt = 64, 130, 120.0, 0.02
    batch = TorchScenarioBatch.synthetic(batch_size=b, n_vehicles=n, device="cuda",
                                         dtype=torch.float32)
    y0 = batch.y0
    t_grid = torch.linspace(0.0, sim_s, int(sim_s / 0.25) + 1, device="cuda",
                            dtype=torch.float32)
    rollout_rk4(batch, y0, t_grid[:5], dt=dt)  # warm up kernels / compile
    torch.cuda.synchronize()
    t0 = time.time()
    rollout_rk4(batch, y0, t_grid, dt=dt)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    per_scenario = elapsed / b
    print(f"\n  {b} x N={n}, {sim_s}s sim @ dt={dt}: {elapsed:.2f}s total, "
          f"{per_scenario * 1000:.1f} ms/scenario")
    assert elapsed < 20.0, f"{elapsed:.1f}s for {b} scenarios (target < 20 s)"


# ---------------------------------------------------------------------------
# helpers -- reference physics, recomputed from the fixture without importing
# the NumPy simulator, so the two implementations stay independent
# ---------------------------------------------------------------------------

def _deltas(fx: dict, y: np.ndarray) -> np.ndarray:
    n = int(fx["N"])
    x = y[:, :n]
    l0 = fx["edge_static"][:, 0]
    return (x[:, :-1] - x[:, 1:]) - l0[None, :]


def _coupler_force_from_state(fx: dict, y: np.ndarray) -> float:
    n = int(fx["N"])
    v = y[:, n : 2 * n]
    delta = _deltas(fx, y)
    ddot = v[:, :-1] - v[:, 1:]
    es = fx["edge_static"]
    s, kd, cd, kb, cb = es[:, 1], es[:, 2], es[:, 3], es[:, 4], es[:, 5]
    f = np.zeros_like(delta)
    draft = delta > s[None, :]
    buff = delta < -s[None, :]
    f = np.where(draft, kd[None, :] * (delta - s[None, :]) + cd[None, :] * ddot, f)
    f = np.where(buff, kb[None, :] * (delta + s[None, :]) + cb[None, :] * ddot, f)
    return float(np.abs(f).max())


def _numpy_rhs_from_fixture(fx: dict, t: float, y: np.ndarray) -> np.ndarray:
    """Independent reference implementation of the RHS, straight from the
    fixture arrays. Deliberately a separate implementation from
    ``simulator2.rhs`` so a shared bug cannot make both sides agree."""
    n = int(fx["N"])
    x, v = y[:n], y[n : 2 * n]
    z_brk, z_trac = y[2 * n : 3 * n], y[3 * n : 4 * n]
    ns, es = fx["node_static"], fx["edge_static"]
    mass, dA, dB, dC = ns[:, 0], ns[:, 1], ns[:, 2], ns[:, 3]
    can_trac, f_brk_max = ns[:, 4], ns[:, 6]

    delta = (x[:-1] - x[1:]) - es[:, 0]
    ddot = v[:-1] - v[1:]
    s, kd, cd, kb, cb = es[:, 1], es[:, 2], es[:, 3], es[:, 4], es[:, 5]
    f_c = np.zeros(n - 1)
    f_c = np.where(delta > s, kd * (delta - s) + cd * ddot, f_c)
    f_c = np.where(delta < -s, kb * (delta + s) + cb * ddot, f_c)

    ut = np.array([np.interp(t, fx["t"], fx["u_trac"][:, i]) for i in range(n)])
    ub = np.array([np.interp(t, fx["t"], fx["u_brk"][:, i]) for i in range(n)])
    ut = np.where(can_trac > 0, ut, 0.0)

    sgn = np.where(np.abs(v) < 1e-9, 0.0, np.sign(v))
    r = (dA + dB * np.abs(v) + dC * v * v) * sgn
    g = mass * 9.81 * np.sin(np.interp(x, fx["route_s"], fx["route_sin_theta"]))
    kap = np.interp(x, fx["route_s"], fx["route_kappa"])
    kcs = float(fx["k_curv_scale"])
    c_cur = np.where(np.abs(v) < 1e-9, 0.0, kcs * mass * v * v * np.abs(kap) * np.sign(v))

    zt, zb = np.maximum(z_trac, 0.0), np.maximum(z_brk, 0.0)
    f_tr = np.where(can_trac > 0,
                    np.minimum(zt, float(fx["p_max_w"]) / np.maximum(np.abs(v), float(fx["v_eps"]))),
                    0.0)
    f_br = np.minimum(zb, f_brk_max)
    if bool(fx["brake_opposes_motion"]):
        f_br = f_br * np.tanh(v / float(fx["v_brake_eps"]))

    f_in = np.concatenate([[0.0], f_c])
    f_out = np.concatenate([f_c, [0.0]])
    dv = (f_tr - f_br - r - g - c_cur + f_in - f_out) / mass
    dz_trac = (ut - z_trac) / float(fx["tau_trac_s"])
    dz_brk = (ub - z_brk) / float(fx["tau_brk_s"])
    return np.concatenate([v, dv, dz_brk, dz_trac])


def test_reference_helper_agrees_with_the_stored_solution() -> None:
    """The independent reference RHS above is only useful if it is right;
    check it reproduces the derivative implied by the stored trajectory."""
    fx = _load(next(p for p in FIXTURES if _name(p) == "regime_cruise"))
    t, y = fx["t"], fx["y"].T
    k = len(t) // 2
    dy = _numpy_rhs_from_fixture(fx, float(t[k]), y[k])
    n = int(fx["N"])
    dv_fd = (y[k + 1] - y[k - 1]) / (t[k + 1] - t[k - 1])
    assert np.abs(dy[:n] - dv_fd[:n]).max() < 1e-3
    assert np.abs(dy[n : 2 * n] - dv_fd[n : 2 * n]).max() < 1e-2
