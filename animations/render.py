"""Simple render helper for all animation scenes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCENES = {
    "train": ("animations/scenes/train_tensor_side_by_side.py", "TrainTensorSideBySide"),
    "time": ("animations/scenes/tensor_time_evolution.py", "TensorTimeEvolution"),
    "coupler": ("animations/scenes/coupler_wave.py", "CouplerWave"),
    "advantage": ("animations/scenes/tensorization_advantage.py", "TensorizationAdvantage"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=list(SCENES.keys()) + ["all"], default="train")
    parser.add_argument("--quality", default="m", choices=["l", "m", "h", "k"])
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    keys = list(SCENES.keys()) if args.scene == "all" else [args.scene]
    for key in keys:
        scene_file, scene_name = SCENES[key]
        cmd = [
            sys.executable,
            "-m",
            "manim",
            f"-q{args.quality}",
            scene_file,
            scene_name,
        ]
        if args.preview:
            cmd.insert(3, "-p")
        subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
