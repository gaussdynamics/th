"""A paused and resumed dataset build writes the same corpus as one run straight
through.

That is what lets a corpus built over several sessions stand in the thesis as
one build. The unit tests below cover the pieces a resume depends on: the
journal survives a kill mid-write, and a resume whose code or inputs changed is
refused.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulator2 import build_progress as bp

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD = [sys.executable, str(_REPO_ROOT / "scripts" / "build_dataset.py")]
# Two batches, so a pause after the first leaves one to resume.
_ARGS = ["--n-scenarios", "8", "--batch-size", "4", "--seed", "1", "--physics", "v3",
         "--duration-range", "60", "120"]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([*_BUILD, *args], check=True, capture_output=True, text=True,
                          cwd=_REPO_ROOT)


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume")
    straight, resumed = root / "straight", root / "resumed"
    _run(*_ARGS, "--out", str(straight))
    _run(*_ARGS, "--out", str(resumed), "--pause-after-batches", "1")
    paused = bp.read_status(resumed)
    _run("--out", str(resumed), "--resume")
    return straight, resumed, paused


def test_the_pause_stopped_after_one_batch(builds):
    _, resumed, paused = builds
    assert paused["state"] == "paused"
    assert (paused["batch_done"], paused["n_batches"]) == (1, 2)
    starts = [s for s in bp.Journal(resumed).load().sessions if s["event"] == "start"]
    assert [s["batch"] for s in starts] == [0, 1]


def test_resumed_build_writes_the_same_manifests(builds):
    straight, resumed, _ = builds
    a = pd.read_parquet(straight / "index.parquet").sort_values("scenario_id")
    b = pd.read_parquet(resumed / "index.parquet").sort_values("scenario_id")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    for name in ("splits.json", "norm_stats.json"):
        assert json.loads((straight / name).read_text()) == json.loads((resumed / name).read_text())
    cfg_a = json.loads((straight / "build_config.json").read_text())
    cfg_b = json.loads((resumed / "build_config.json").read_text())
    assert cfg_a["fingerprint"] == cfg_b["fingerprint"]


def test_resumed_build_writes_the_same_scenarios(builds):
    straight, resumed, _ = builds
    files = sorted(p.relative_to(straight) for p in (straight / "scenarios").rglob("*.npz"))
    assert files == sorted(p.relative_to(resumed) for p in (resumed / "scenarios").rglob("*.npz"))
    assert len(files) > 0
    for rel in files:
        with np.load(straight / rel) as a, np.load(resumed / rel) as b:
            assert sorted(a.files) == sorted(b.files)
            for k in a.files:
                np.testing.assert_array_equal(a[k], b[k], err_msg=f"{rel}:{k}")


def test_a_second_build_into_the_same_directory_is_refused(builds):
    straight, _, _ = builds
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run(*_ARGS, "--out", str(straight))
    assert "--resume" in exc.value.stderr


# ---------------------------------------------------------------------------
# the journal
# ---------------------------------------------------------------------------

def _journal(tmp_path) -> bp.Journal:
    j = bp.Journal(tmp_path)
    j.start({"seed": 0}, {"code": "a", "inputs": "b"}, n_batches=3, n_specs=9)
    j.add_batch(0, [{"scenario_id": "r_run000000"}], {"diverged": []}, 1.0, 3)
    return j


def test_journal_drops_a_line_torn_by_a_kill(tmp_path):
    _journal(tmp_path)
    path = tmp_path / bp.JOURNAL
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"kind": "batch", "batch": 1, "rec')     # the process died here
    j = bp.Journal(tmp_path).load()
    assert j.next_batch == 1
    assert path.read_text(encoding="utf-8").endswith("\n")  # cut, ready to append
    j.add_batch(1, [], {}, 1.0, 3)
    assert bp.Journal(tmp_path).load().next_batch == 2


def test_resume_is_refused_when_the_build_changed(tmp_path):
    _journal(tmp_path)
    j = bp.Journal(tmp_path).load()
    j.check({"code": "a", "inputs": "b"})                    # unchanged: fine
    with pytest.raises(SystemExit, match="simulator2/"):
        j.check({"code": "CHANGED", "inputs": "b"})


def test_fingerprint_sees_a_changed_input_file(tmp_path):
    f = tmp_path / "route.npz"
    f.write_bytes(b"one")
    before = bp.fingerprint({}, [], [f])
    f.write_bytes(b"two")
    after = bp.fingerprint({}, [], [f])
    assert before["inputs"] != after["inputs"]
    assert before["code"] == after["code"]


def test_status_goes_stale_when_the_heartbeat_stops(tmp_path):
    bp.write_json_atomic(tmp_path / bp.STATUS, {"state": "running", "updated": time.time()})
    assert bp.read_status(tmp_path)["state"] == "running"
    bp.write_json_atomic(tmp_path / bp.STATUS,
                         {"state": "running", "updated": time.time() - 2 * bp.STALE_AFTER_S})
    assert bp.read_status(tmp_path)["state"] == "stopped"


def test_pause_request_round_trip(tmp_path):
    assert not bp.pause_requested(tmp_path)
    bp.request_pause(tmp_path)
    assert bp.pause_requested(tmp_path)
    bp.clear_control(tmp_path)
    assert not bp.pause_requested(tmp_path)
