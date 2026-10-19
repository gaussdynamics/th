"""Dataset builds on this machine: list them, read their status, start, pause
and resume them.

The builder (``scripts/build_dataset.py``) runs as its own process, never in
the console: a multi-day build must not die with the window, and the console
must not import torch. The two talk only through the files described in
``simulator2/build_progress.py``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from simulator2 import build_progress as bp

_REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = _REPO_ROOT / "scripts" / "build_dataset.py"
DATA_DIR = _REPO_ROOT / "data"

#: States in which a builder process is (or should be) alive.
ACTIVE = ("sampling", "running", "pausing", "finalizing")


@dataclass
class NewBuild:
    out: Path
    n_scenarios: int
    physics: str = "v3"
    seed: int = 0
    batch_size: int = 128
    route_dir: Optional[Path] = None
    split_file: Optional[Path] = None
    extra_args: str = ""

    def argv(self) -> List[str]:
        args = ["--out", str(self.out), "--n-scenarios", str(self.n_scenarios),
                "--physics", self.physics, "--seed", str(self.seed),
                "--batch-size", str(self.batch_size)]
        if self.route_dir:
            args += ["--route-dir", str(self.route_dir)]
        if self.split_file:
            args += ["--split-file", str(self.split_file)]
        return args + shlex.split(self.extra_args, posix=(os.name != "nt"))


class BuildService:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        # Builders this console launched, so a second click cannot start
        # another before the first has written its status.
        self._procs: dict[Path, subprocess.Popen] = {}

    # -- reading -------------------------------------------------------------
    def list_builds(self) -> List[Path]:
        """Build directories under ``data/``, most recently active first."""
        if not self.data_dir.exists():
            return []
        found = [p for p in self.data_dir.iterdir()
                 if p.is_dir() and (p / bp.JOURNAL).exists()]

        def last_active(p: Path) -> float:
            s = p / bp.STATUS
            return (s if s.exists() else p / bp.JOURNAL).stat().st_mtime

        return sorted(found, key=last_active, reverse=True)

    def status(self, out: Path) -> Optional[dict]:
        return bp.read_status(out)

    def is_active(self, out: Path) -> bool:
        proc = self._procs.get(Path(out).resolve())
        if proc is not None and proc.poll() is None:
            return True
        s = self.status(out)
        return bool(s and s.get("state") in ACTIVE)

    def log_tail(self, out: Path, max_bytes: int = 16_000) -> str:
        path = Path(out) / bp.LOG
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        return text if size <= max_bytes else text.split("\n", 1)[-1]

    # -- control -------------------------------------------------------------
    def start(self, build: NewBuild) -> subprocess.Popen:
        if (Path(build.out) / bp.JOURNAL).exists():
            raise ValueError(f"{build.out} already holds a build; resume it instead")
        return self._launch(build.argv(), Path(build.out))

    def resume(self, out: Path) -> subprocess.Popen:
        if self.is_active(out):
            raise ValueError(f"the build in {out} is still running")
        return self._launch(["--out", str(out), "--resume"], Path(out))

    def pause(self, out: Path) -> None:
        bp.request_pause(out)

    def _launch(self, args: List[str], out: Path) -> subprocess.Popen:
        """Start the builder detached from the console, its output appended to
        ``build.log``. On Windows it gets its own process group and no window,
        so closing the console (or a Ctrl-C in its terminal) leaves it running."""
        out.mkdir(parents=True, exist_ok=True)
        log = open(out / bp.LOG, "ab")
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"\n===== {stamp}  build_dataset.py {' '.join(args)}\n".encode())
        log.flush()
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                       | subprocess.CREATE_NO_WINDOW)
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(BUILD_SCRIPT), *args],
                cwd=_REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, **kwargs)
            self._procs[out.resolve()] = proc
            return proc
        finally:
            log.close()  # the child holds its own handle
