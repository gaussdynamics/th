"""Offline tests for corridor pool selection and split assignment.

Runnable with pytest or directly: ``python tests/test_split_selection.py``.

The property these guard is that a corridor's *split* depends only on its
measured grade, never on the proxy used to decide it was worth measuring. The
proxy is generous on purpose, so a flagged corridor that turns out flat must
land in the ordinary pool like any other.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from assign_split_corridors import evenly_spaced  # noqa: E402
from select_split_corridors import (  # noqa: E402
    is_mountain_candidate,
    matches_mountain_name,
    matches_mountain_state,
)


def _row(subdiv="MAIN LINE", states="KS", country="US"):
    return {"subdiv": subdiv, "states": states, "country": country}


# --------------------------------------------------------------------------- #
# The mountain proxy
# --------------------------------------------------------------------------- #
def test_proxy_flags_on_subdivision_name():
    assert is_mountain_candidate(_row(subdiv="RATON PASS"))
    assert is_mountain_candidate(_row(subdiv="Moffat Tunnel"))
    assert is_mountain_candidate(_row(subdiv="TENNESSEE PASS"))


def test_proxy_flags_on_state_even_with_a_plain_name():
    assert is_mountain_candidate(_row(subdiv="MAIN LINE", states="CO"))
    assert is_mountain_candidate(_row(subdiv="MAIN LINE", states="IA,WY"))


def test_proxy_ignores_flat_plains_corridors():
    assert not is_mountain_candidate(_row(subdiv="MAIN LINE", states="KS"))
    assert not is_mountain_candidate(_row(subdiv="RIVER SUB", states="IA,IL"))


def test_proxy_is_case_insensitive_and_handles_missing_fields():
    assert is_mountain_candidate({"subdiv": "cajon pass", "states": ""})
    assert not is_mountain_candidate({})


def test_name_and_state_flags_are_separable():
    """They are not equally informative and must not be pooled. The state flag
    hits 29% of the US network; pooling them and spacing across the union gave
    back a representative sample, and ood_grade came out +0.02 pp from train."""
    plain_in_mountain_state = _row(subdiv="MAIN LINE", states="CO")
    assert matches_mountain_state(plain_in_mountain_state)
    assert not matches_mountain_name(plain_in_mountain_state)

    named_on_the_plains = _row(subdiv="RATON PASS", states="KS")
    assert matches_mountain_name(named_on_the_plains)
    assert not matches_mountain_state(named_on_the_plains)


# --------------------------------------------------------------------------- #
# Deterministic spacing -- no RNG, so a rerun is the same set
# --------------------------------------------------------------------------- #
def test_evenly_spaced_is_deterministic_and_sized():
    items = list(range(100))
    a = evenly_spaced(items, 10)
    assert a == evenly_spaced(items, 10)
    assert len(a) == 10
    assert len(set(a)) == 10


def test_evenly_spaced_spans_the_range():
    """It must not take a prefix -- that would hand ood_corridor one end of the
    grade distribution instead of a cross-section of it."""
    picks = evenly_spaced(list(range(100)), 5)
    assert picks[0] < 20 and picks[-1] > 70


def test_evenly_spaced_returns_everything_when_asked_for_too_much():
    items = list(range(5))
    assert evenly_spaced(items, 10) == items


# --------------------------------------------------------------------------- #
# Grade recovery from the stored field
# --------------------------------------------------------------------------- #
def test_rms_grade_inverts_the_stored_sin_theta(tmp_path):
    """The field stores sin(arctan(g)); recovering g is tan(arcsin(.)). The
    naive arctan(.) agrees to O(g^3), so only an exact check catches it."""
    from assign_split_corridors import rms_grade_pct
    g = np.full(1000, 0.035)                    # a steady 3.5% grade
    p = tmp_path / "c.npz"
    np.savez(p, route_s=np.arange(1000) * 10.0,
             route_sin_theta=np.sin(np.arctan(g)),
             route_kappa=np.zeros(1000), route_vmax=np.full(1000, 20.0))
    assert abs(rms_grade_pct(p) - 3.5) < 1e-6


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                print(f"  skip  {name} (needs pytest tmp_path)")
                continue
            try:
                fn()
                print(f"  ok    {name}")
            except Exception:
                fails += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print("\n%s" % ("all passed" if not fails else "%d failed" % fails))
    sys.exit(1 if fails else 0)


# --------------------------------------------------------------------------- #
# Gap enforcement
# --------------------------------------------------------------------------- #
def _fake_routes(tmp_path, grades):
    """Write one .npz per corridor with a constant grade."""
    (tmp_path / "routes").mkdir(parents=True, exist_ok=True)
    for cid, g in grades.items():
        n = 500
        np.savez(tmp_path / "routes" / f"{cid}.npz",
                 route_s=np.arange(n) * 10.0,
                 route_sin_theta=np.full(n, np.sin(np.arctan(g / 100.0))),
                 route_kappa=np.zeros(n), route_vmax=np.full(n, 20.0))
    return tmp_path


def _run_assign(tmp_path, grades, **kw):
    import subprocess, json, sys as _s
    from pathlib import Path
    _fake_routes(tmp_path, grades)
    out = tmp_path / "split.json"
    args = [_s.executable, str(Path(__file__).parent.parent.parent
                               / "scripts" / "assign_split_corridors.py"),
            "--routes", str(tmp_path), "--out", str(out)]
    for k, v in kw.items():
        args += [f"--{k.replace('_','-')}", str(v)]
    subprocess.run(args, check=True, capture_output=True)
    return json.loads(out.read_text())


def test_training_pool_never_reaches_into_the_gap(tmp_path):
    """The buffer band must leave the training pool too. Leaving it in put the
    gap back exactly where it was meant to be opened."""
    grades = {f"c{i:03d}": 0.1 + i * 0.02 for i in range(100)}
    grades.update({f"steep{i}": 2.0 + i * 0.1 for i in range(10)})
    res = _run_assign(tmp_path, grades, n_ood_grade=10, n_ood_corridor=5,
                      n_control_eval=5, gap_pp=0.3)
    g = res["rms_grade_pct"]
    assert min(g[c] for c in res["ood_grade_corridors"]) \
        - max(g[c] for c in res["train_pool"]) >= 0.3 - 1e-9


def test_buffer_band_is_excluded_from_every_split(tmp_path):
    grades = {f"c{i:03d}": 0.1 + i * 0.02 for i in range(100)}
    grades.update({f"steep{i}": 2.0 + i * 0.1 for i in range(10)})
    res = _run_assign(tmp_path, grades, n_ood_grade=10, n_ood_corridor=5,
                      n_control_eval=5, gap_pp=0.3)
    band = set(res["buffer_band_excluded"])
    assert band
    for key in ("train_pool", "ood_grade_corridors", "ood_corridors",
                "control_eval_corridors"):
        assert not (band & set(res[key])), key


def test_splits_are_disjoint(tmp_path):
    """A control_eval corridor leaking into training would silently invalidate
    the control chapter."""
    grades = {f"c{i:03d}": 0.1 + i * 0.03 for i in range(120)}
    res = _run_assign(tmp_path, grades, n_ood_grade=10, n_ood_corridor=8,
                      n_control_eval=8, gap_pp=0.1)
    keys = ("train_pool", "ood_grade_corridors", "ood_corridors",
            "control_eval_corridors", "buffer_band_excluded")
    seen = set()
    for k in keys:
        cur = set(res[k])
        assert not (cur & seen), k
        seen |= cur
