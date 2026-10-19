# RailLab — desktop research console

PySide6 UI over the existing `simulator2` / `routegen` backend. This is the
**skeleton** (build-order step 1 in `../APP_STACK_SKETCH.md`): app shell, nav
rail, four placeholder screens, and the cassette-futurism theme.

## Run

From the repo root (`th/`):

```bash
pip install PySide6 pyqtgraph
python -m app.main
```

## Headless self-test

Verifies the environment without a display (used in CI / sandboxes):

```bash
QT_QPA_PLATFORM=offscreen python -m app.main --selftest
```

Prints `SELFTEST OK` and exits 0 when the app constructs and renders.

## Layout

```
app/
├─ main.py          entry point (+ --selftest)
├─ main_window.py   nav rail + stacked screens + status bar
├─ views/           one screen per nav stop (placeholders for now)
├─ widgets/         reusable retro widgets (Led so far)
├─ theme/           cassette.qss
└─ resources/fonts  optional bundled fonts (VT323 / IBM Plex Mono)
```

## BUILD screen

Starts, watches, pauses and resumes `scripts/build_dataset.py` runs on this
machine. Builds live under `data/`; any directory there with a
`progress.jsonl` is listed. The builder is launched as its own process (output
in `<out>/build.log`), so closing the console does not stop a build, and
reopening it picks the build back up from its files. PAUSE takes effect when
the batch in flight finishes. Without the console:

```bash
python -m simulator2.build_progress pause data/v3     # or Ctrl-C in the build's terminal
python -m simulator2.build_progress status data/v3
python scripts/build_dataset.py --out data/v3 --resume
```

A resume is refused if `simulator2/`, `scripts/build_dataset.py`, the route or
consist files changed since the build started, so don't edit those (or switch
branches) while a build is paused.
