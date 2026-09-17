"""Load animation data exported from simulator2 rollouts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AnimationData:
    time_s: np.ndarray
    h_hist: np.ndarray
    e_hist: np.ndarray
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    z_brk_n: np.ndarray
    z_trac_n: np.ndarray
    coupler_forces_n: np.ndarray
    u_trac_cmd_n: np.ndarray
    u_brk_cmd_n: np.ndarray
    tensor_slice: np.ndarray
    tensor_slice_feature_names: list[str]
    node_feature_names: list[str]
    edge_feature_names: list[str]
    metadata: dict[str, object]

    @property
    def n_time(self) -> int:
        return int(self.time_s.shape[0])

    @property
    def n_vehicles(self) -> int:
        return int(self.positions_m.shape[1])


def load_animation_data(npz_path: Path) -> AnimationData:
    npz = np.load(npz_path, allow_pickle=True)
    metadata = json.loads(str(npz["metadata_json"]))
    return AnimationData(
        time_s=npz["time_s"],
        h_hist=npz["H_hist"],
        e_hist=npz["E_hist"],
        positions_m=npz["positions_m"],
        velocities_mps=npz["velocities_mps"],
        z_brk_n=npz["z_brk_n"],
        z_trac_n=npz["z_trac_n"],
        coupler_forces_n=npz["coupler_forces_n"],
        u_trac_cmd_n=npz["u_trac_cmd_n"],
        u_brk_cmd_n=npz["u_brk_cmd_n"],
        tensor_slice=npz["tensor_slice"],
        tensor_slice_feature_names=list(npz["tensor_slice_feature_names"]),
        node_feature_names=list(npz["node_feature_names"]),
        edge_feature_names=list(npz["edge_feature_names"]),
        metadata=metadata,
    )
