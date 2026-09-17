"""Export simulator2 tensor rollouts into animation-friendly NPZ bundles."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simulator2.scenarios import make_simple_train_scenario
from simulator2.simulate import simulate_train_tensorized
from simulator2.state_schema import EdgeChannel, NodeChannel


def _sample_controls(
    t: np.ndarray,
    n_vehicles: int,
    u_trac_cmd: Any,
    u_brk_cmd: Any,
) -> tuple[np.ndarray, np.ndarray]:
    u_trac = np.zeros((t.size, n_vehicles), dtype=float)
    u_brk = np.zeros((t.size, n_vehicles), dtype=float)
    for k, tk in enumerate(t):
        for i in range(n_vehicles):
            u_trac[k, i] = float(u_trac_cmd(float(tk), i))
            u_brk[k, i] = float(u_brk_cmd(float(tk), i))
    return u_trac, u_brk


def export_demo_run(
    output_npz: Path,
    n_cars: int = 8,
    t_span: tuple[float, float] = (0.0, 60.0),
    n_time_samples: int = 601,
) -> Path:
    """Run simulator2 and export core arrays for Manim scenes."""
    scenario = make_simple_train_scenario(
        n_cars=n_cars,
        n_time_samples=n_time_samples,
        t_span=t_span,
    )
    result = simulate_train_tensorized(scenario)
    n = int(result.metadata["N"])

    h_hist = result.H_hist
    e_hist = result.E_hist
    t = result.t

    pos = h_hist[:, :, int(NodeChannel.X)]
    vel = h_hist[:, :, int(NodeChannel.V)]
    z_brk = h_hist[:, :, int(NodeChannel.Z_BRK)]
    z_trac = h_hist[:, :, int(NodeChannel.Z_TRAC)]
    coupler_forces = (
        e_hist[:, :, int(EdgeChannel.F_CPL)] if e_hist.shape[1] > 0 else np.zeros((t.size, 0))
    )
    u_trac_cmd, u_brk_cmd = _sample_controls(t, n, scenario.u_trac_cmd, scenario.u_brk_cmd)

    tensor_slice_names = [
        "x_m",
        "v_m_per_s",
        "z_brk_n",
        "z_trac_n",
        "f_cpl_left_n",
    ]
    f_left = np.zeros_like(pos)
    if coupler_forces.shape[1] > 0:
        f_left[:, 1:] = coupler_forces
    tensor_slice = np.stack([pos, vel, z_brk, z_trac, f_left], axis=2)

    payload = {
        "time_s": t,
        "H_hist": h_hist,
        "E_hist": e_hist,
        "positions_m": pos,
        "velocities_mps": vel,
        "z_brk_n": z_brk,
        "z_trac_n": z_trac,
        "coupler_forces_n": coupler_forces,
        "u_trac_cmd_n": u_trac_cmd,
        "u_brk_cmd_n": u_brk_cmd,
        "tensor_slice": tensor_slice,
        "tensor_slice_feature_names": np.array(tensor_slice_names, dtype=object),
        "node_feature_names": np.array(
            [k for k, _ in sorted(result.metadata["node_channels"].items(), key=lambda kv: kv[1])],
            dtype=object,
        ),
        "edge_feature_names": np.array(
            [k for k, _ in sorted(result.metadata["edge_channels"].items(), key=lambda kv: kv[1])],
            dtype=object,
        ),
        "metadata_json": json.dumps(
            {
                "source": "simulator2.simulate.simulate_train_tensorized",
                "scenario": result.metadata.get("scenario", "make_simple_train_scenario"),
                "n_vehicles": n,
                "n_time_steps": int(t.size),
            }
        ),
    }

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **payload)
    return output_npz


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_path = repo_root / "animations" / "data" / "demo_tensor_run.npz"
    written = export_demo_run(out_path)
    print(f"Wrote animation demo data to: {written}")


if __name__ == "__main__":
    main()
