"""Pause, resume and status for ``scripts/build_dataset.py``.

A corpus build runs for days, so it has to survive being stopped. Three files
in the output directory carry its state; the builder and the RailLab BUILD
screen talk only through them, so neither imports the other and the console
never imports torch.

``progress.jsonl``  the journal. A header line (the build's arguments and its
                    fingerprint), then one line per finished batch (its
                    records and its refused ids), appended and fsynced as each
                    batch ends. Resuming replays it and starts at the next
                    batch; a line cut short by a kill is dropped and that batch
                    is simply run again.
``status.json``     what the build is doing now, rewritten atomically after
                    each batch and every few seconds by a heartbeat thread.
``control.json``    a request to the build: ``{"command": "pause"}``. Read at
                    each batch boundary, so a pause takes effect when the
                    batch in flight finishes.

Resuming is only honest if the resumed build is the build that was started:
the same arguments, the same code and the same inputs. The fingerprint hashes
all three, and a resume with any of them changed is refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

JOURNAL = "progress.jsonl"
STATUS = "status.json"
CONTROL = "control.json"
LOG = "build.log"

#: A status this old while ``running`` means the builder is gone.
STALE_AFTER_S = 30.0
HEARTBEAT_S = 5.0

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Code that decides what the corpus contains. This module is left out: how a
# build is paused cannot change what it writes.
_SOURCE_GLOBS = ("simulator2/*.py", "scripts/build_dataset.py")
_NOT_SOURCE = {"simulator2/build_progress.py"}


# ---------------------------------------------------------------------------
# small file helpers
# ---------------------------------------------------------------------------

def _jsonable(x: Any) -> Any:
    """``json.dumps`` default for the numpy scalars and arrays in scenario meta."""
    if hasattr(x, "item") and getattr(x, "ndim", 0) == 0:
        return x.item()
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serializable: {type(x).__name__}")


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write via a temporary file and a rename, so a reader never sees half a
    file. On Windows the rename fails while a reader has the target open, so
    it is retried briefly."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[dict]:
    """The file's contents, or ``None`` if it is missing or mid-write."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------

def _sha(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()[:16]


def _files_sha(paths: Iterable[Path], root: Path) -> str:
    def chunks():
        for p in sorted(paths):
            yield str(p.relative_to(root)).replace("\\", "/").encode()
            yield p.read_bytes()
    return _sha(chunks())


def source_paths(repo_root: Path = _REPO_ROOT) -> List[Path]:
    out = []
    for pattern in _SOURCE_GLOBS:
        for p in repo_root.glob(pattern):
            if str(p.relative_to(repo_root)).replace("\\", "/") not in _NOT_SOURCE:
                out.append(p)
    return out


def fingerprint(
    build_args: Dict[str, Any],
    spec_keys: List[tuple],
    input_files: Iterable[Path],
    repo_root: Path = _REPO_ROOT,
) -> Dict[str, str]:
    """One hash per thing that must not change between pause and resume.

    ``spec_keys`` is one tuple per sampled scenario (id, split, T, N), so a
    sampler that draws differently is caught even when the code hash is not
    (e.g. a changed dependency). ``input_files`` are the route and consist
    files the scenarios are built from.
    """
    return {
        "args": _sha([json.dumps(build_args, sort_keys=True, default=_jsonable).encode()]),
        "code": _files_sha(source_paths(repo_root), repo_root),
        "inputs": _sha(p.read_bytes() for p in sorted(Path(f) for f in input_files)),
        "scenarios": _sha([json.dumps(spec_keys, default=_jsonable).encode()]),
    }


# ---------------------------------------------------------------------------
# journal
# ---------------------------------------------------------------------------

class Journal:
    """``progress.jsonl``: a header, then one line per finished batch."""

    def __init__(self, out_dir: Path) -> None:
        self.path = Path(out_dir) / JOURNAL
        self.header: Optional[dict] = None
        self.batches: List[dict] = []
        self.sessions: List[dict] = []

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> "Journal":
        """Read the journal back. A torn last line (the process died mid-write)
        is dropped and cut from the file, so the next append starts clean."""
        raw = self.path.read_bytes()
        good_end, lines = 0, []
        for line in raw.splitlines(keepends=True):
            try:
                lines.append(json.loads(line))
            except ValueError:
                break
            if not line.endswith(b"\n"):
                lines.pop()
                break
            good_end += len(line)
        if good_end < len(raw):
            with open(self.path, "r+b") as f:
                f.truncate(good_end)
        for entry in lines:
            kind = entry.get("kind")
            if kind == "header":
                self.header = entry
            elif kind == "batch":
                self.batches.append(entry)
            elif kind == "session":
                self.sessions.append(entry)
        if self.header is None:
            raise ValueError(f"{self.path} has no header line")
        for i, b in enumerate(self.batches):
            if b["batch"] != i:
                raise ValueError(f"{self.path}: batch {b['batch']} found where {i} was expected")
        return self

    def _append(self, entry: dict) -> None:
        line = json.dumps(entry, default=_jsonable) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def start(self, build_args: dict, fp: Dict[str, str], n_batches: int,
              n_specs: int) -> None:
        self.header = {"kind": "header", "build_args": build_args, "fingerprint": fp,
                       "n_batches": n_batches, "n_specs": n_specs,
                       "created": time.time()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._append(self.header)

    def check(self, fp: Dict[str, str]) -> None:
        """Refuse a resume whose build is not the one in the journal."""
        old = self.header["fingerprint"]
        changed = sorted(k for k in set(old) | set(fp) if old.get(k) != fp.get(k))
        if changed:
            why = {
                "args": "the build arguments",
                "code": "simulator2/ or scripts/build_dataset.py",
                "inputs": "the route or consist files",
                "scenarios": "the sampled scenarios",
            }
            raise SystemExit(
                f"cannot resume {self.path.parent}: "
                + ", ".join(why.get(k, k) for k in changed)
                + " changed since the build started. Resuming would mix two builds "
                  "in one corpus; restore them (e.g. check out the commit the build "
                  "started on) or start a new build in a new --out."
            )

    def add_session(self, info: dict) -> None:
        entry = {"kind": "session", **info}
        self.sessions.append(entry)
        self._append(entry)

    def add_batch(self, batch: int, records: List[dict], refused: Dict[str, List[str]],
                  seconds: float, n_in_batch: int) -> None:
        entry = {"kind": "batch", "batch": batch, "n": n_in_batch,
                 "seconds": round(seconds, 3), "records": records, "refused": refused}
        self._append(entry)
        self.batches.append(entry)

    # -- replay --------------------------------------------------------------
    @property
    def next_batch(self) -> int:
        return len(self.batches)

    def records(self) -> List[dict]:
        return [r for b in self.batches for r in b["records"]]

    def refused(self, kind: str) -> List[str]:
        return [sid for b in self.batches for sid in b["refused"].get(kind, [])]

    def batch_times(self) -> List[List[float]]:
        """``[batch, scenarios in it, seconds]`` per finished batch."""
        return [[b["batch"], b["n"], b["seconds"]] for b in self.batches]


# ---------------------------------------------------------------------------
# control
# ---------------------------------------------------------------------------

def request_pause(out_dir: Path) -> None:
    write_json_atomic(Path(out_dir) / CONTROL, {"command": "pause", "requested": time.time()})


def pause_requested(out_dir: Path) -> bool:
    c = read_json(Path(out_dir) / CONTROL)
    return bool(c and c.get("command") == "pause")


def clear_control(out_dir: Path) -> None:
    try:
        (Path(out_dir) / CONTROL).unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class StatusWriter:
    """Keeps ``status.json`` current. ``update`` sets fields and writes; a
    heartbeat thread rewrites ``updated`` every few seconds, so a reader can
    tell a long batch from a dead process. The heartbeat also notices a pause
    request as soon as it is made, and shows it as ``pausing``."""

    def __init__(self, out_dir: Path, **fields: Any) -> None:
        self.out_dir = Path(out_dir)
        self._lock = threading.Lock()
        self._status: Dict[str, Any] = {"pid": os.getpid(), "out": str(self.out_dir),
                                        "session_started": time.time(), **fields}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, daemon=True)

    def start(self) -> "StatusWriter":
        self._write()
        self._thread.start()
        return self

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._status.update(fields)
        self._write()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._status.get(key, default)

    def close(self, **fields: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=HEARTBEAT_S * 2)
        self.update(**fields)

    def _write(self) -> None:
        with self._lock:
            self._status["updated"] = time.time()
            snapshot = dict(self._status)
        try:
            write_json_atomic(self.out_dir / STATUS, snapshot)
        except OSError:
            pass  # a missed heartbeat is not worth killing a build over

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            with self._lock:
                running = self._status.get("state") == "running"
            if running and pause_requested(self.out_dir):
                with self._lock:
                    self._status["state"] = "pausing"
            self._write()


def read_status(out_dir: Path) -> Optional[dict]:
    """``status.json``, with ``state`` set to ``stopped`` when it claims to be
    working but the heartbeat has gone quiet (killed, crashed, power cut)."""
    s = read_json(Path(out_dir) / STATUS)
    if s is None:
        return None
    if s.get("state") in ("sampling", "running", "pausing", "finalizing") \
            and time.time() - float(s.get("updated", 0)) > STALE_AFTER_S:
        s["state"] = "stopped"
        s.setdefault("error", "the builder stopped without pausing; resume to continue")
    return s


def eta_seconds(batch_times: List[List[float]], batches_left: int,
                window: int = 20) -> Optional[float]:
    """Remaining time from the last ``window`` batches. Batches are sorted by
    duration, so later ones are slower: this underestimates early on."""
    recent = batch_times[-window:]
    if not recent:
        return None
    return batches_left * sum(t for _, _, t in recent) / len(recent)


# ---------------------------------------------------------------------------
# helpers for the builder loop
# ---------------------------------------------------------------------------

def remove_partial_files(out_dir: Path) -> int:
    """Delete the ``*.tmp`` files left by writes that a kill interrupted."""
    n = 0
    for p in (Path(out_dir) / "scenarios").rglob("*.tmp"):
        p.unlink()
        n += 1
    return n




# ---------------------------------------------------------------------------
# command line: python -m simulator2.build_progress {pause,status} OUT_DIR
# ---------------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Pause a running dataset build, or show its status.")
    ap.add_argument("command", choices=("pause", "status"))
    ap.add_argument("out", type=Path, help="the build's --out directory")
    args = ap.parse_args(argv)
    if args.command == "pause":
        request_pause(args.out)
        print(f"pause requested; {args.out} stops when the batch in flight finishes")
        return
    s = read_status(args.out)
    if s is None:
        raise SystemExit(f"no {STATUS} in {args.out}")
    print(f"{s.get('state')}: batch {s.get('batch_done', 0)}/{s.get('n_batches', '?')}, "
          f"{s.get('scenarios_written', 0)} scenarios written")
    if s.get("eta_s") is not None and s.get("state") in ("running", "pausing"):
        print(f"about {s['eta_s'] / 3600:.1f} h left")
    if s.get("error"):
        print(s["error"])


if __name__ == "__main__":
    _main()
