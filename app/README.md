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
