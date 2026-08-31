# Research Console — App Architecture Sketch

_A starting point to react to, not a spec. Cross things out, keep what fits._

Stack: **PySide6 (Qt for Python) + pyqtgraph + PyInstaller**, calling your existing
`simulator2` / `routegen` packages directly, in-process. One language (Python), real
dials (`QDial`), fully restylable for the cassette-futurism look, ships as a native
`.app` / `.exe`.

---

## 1. The core principle: the UI never imports scipy or torch

Your backend (`simulator2`, `routegen`, and the future surrogate/control code) stays
**exactly as it is**. The app talks to it through a thin **service layer**. Three reasons:

- You can keep running the backend headless (notebooks, scripts, the paper figures)
  with zero dependency on Qt.
- The UI thread only ever touches widgets. All heavy work (`solve_ivp`, dataset
  generation, training) runs on a worker thread and reports back via Qt signals.
- When the GNN/Neural-ODE surrogate and the control demo land, they plug in as two
  more services without touching the screens.

```
┌─────────────────────── app/ (PySide6) ───────────────────────┐
│  views/      ← screens: route browser, consist, sim, results  │
│  widgets/    ← retro reusable bits: knob, LED, CRT plot frame  │
│  models/     ← Qt models bridging your dataclasses to views    │
│  services/   ← thin adapters over the backend (the only place  │
│  workers/       that imports simulator2 / routegen / torch)    │
│  theme/      ← cassette.qss, palette, fonts                    │
└───────────────────────────────┬───────────────────────────────┘
                                 │  plain data in, Qt signals out
┌────────────────────────────────▼──────────────────────────────┐
│  simulator2/   routegen/   (future) surrogate/   control/      │
│  ← untouched. numpy/scipy today, torch later.                  │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Folder layout

Proposed home: a new `app/` package at the repo root (sibling to `simulator2/`,
`route_generator/`), so imports like `from simulator2.scenarios import ...` just work.

```
th/
├─ simulator2/                 # existing, untouched
├─ route_generator/routegen/   # existing, untouched
├─ surrogate/                  # future GNN/Neural-ODE (per PROJECT_DIRECTION.md)
├─ control/                    # future MPC / model-based RL
├─ simulator2/                 # + NEW backend modules (see §9):
│   ├─ catalog.py              #   CarType / CouplerType + default library
│   └─ consist.py              #   CarInstance, Train, build_consist(), JSON IO
├─ car_library/                # saved car types (JSON), like saved_routes/
│   ├─ loco_sd70.json
│   └─ tank_car_water.json
├─ saved_trains/               # saved trains (JSON): ordered consists
│   └─ heavy_coal_drag.json
└─ app/
   ├─ main.py                  # QApplication, load theme, show MainWindow
   ├─ main_window.py           # shell: nav rail + stacked screens + status bar
   ├─ services/
   │   ├─ route_service.py     # wraps routegen: list/load/generate routes
   │   ├─ catalog_service.py   # car-type library: list/load/save/duplicate
   │   ├─ consist_service.py   # build/load/save Trains -> (vehicles, couplers)
   │   ├─ sim_service.py       # wraps make_scenario_on_route + simulate_*
   │   ├─ training_service.py  # future: wraps surrogate training loop
   │   └─ control_service.py   # future: wraps MPC/RL rollout
   ├─ workers/
   │   ├─ worker.py            # generic QObject worker + signals base
   │   ├─ sim_worker.py        # runs a simulation off the UI thread
   │   └─ training_worker.py   # future: streams epoch/loss metrics
   ├─ models/
   │   ├─ route_list_model.py  # QAbstractListModel over saved_routes/
   │   ├─ car_library_model.py # car types grouped by category
   │   ├─ train_list_model.py  # saved trains
   │   └─ consist_model.py     # ordered, editable table of car instances
   ├─ views/
   │   ├─ route_view.py        # browse/select/preview a route
   │   ├─ consist_view.py      # car library + train builder + saved trains (§9)
   │   ├─ car_editor.py        # form to create/edit a CarType
   │   ├─ sim_view.py          # the "buttons and dials" control panel
   │   ├─ results_view.py      # pyqtgraph: speeds, coupler forces, positions
   │   ├─ training_view.py     # future: live loss/metric monitor
   │   └─ control_view.py      # future: control studio
   ├─ widgets/
   │   ├─ knob.py              # labeled QDial + value readout
   │   ├─ led.py               # status LED (idle/running/error)
   │   ├─ seven_seg.py         # numeric readout
   │   └─ crt_frame.py         # bezeled container for plots
   ├─ theme/
   │   ├─ cassette.qss
   │   └─ palette.py
   └─ resources/
       ├─ fonts/               # e.g. VT323 / IBM Plex Mono
       └─ icons/
```

---

## 3. How a screen wraps the backend (concrete)

The mapping to your real entry points:

| UI action | Backend call (already exists) |
|---|---|
| List routes | `routegen` saved files in `route_generator/saved_routes/` + `route_profiles/*.npz` |
| Load a route | `simulator2.route.load_route_profile(path)` → `RouteProfile` |
| List car types / trains | JSON in `car_library/` and `saved_trains/` (new — see §9) |
| Compile a train | `simulator2.consist.build_consist(train)` → `(vehicles, couplers)` (new — see §9) |
| Assemble a scenario | `make_scenario_from_consist(route, vehicles, couplers, t_span, k_curv_scale)` (new thin wrapper over the current `make_scenario_on_route` internals) |
| Run the simulation | `simulator2.simulate.simulate_train_tensorized(scenario)` → `TensorSimulationResult` (`H_hist`, `E_hist`, `F_coupler`) |
| Plot results | `pyqtgraph` over `H_hist` channels (speeds, `z_trac`, `z_brk`) and `F_coupler` |

### Service — the only file that imports your backend

```python
# app/services/sim_service.py
from simulator2.route import load_route_profile
from simulator2.scenarios import make_scenario_on_route
from simulator2.simulate import simulate_train_tensorized

class SimService:
    """Pure functions over the backend. No Qt in here — easy to unit test."""

    def build_scenario(self, route_path, *, n_cars, t_span,
                       f_trac_max_N, k_curv_scale):
        route = load_route_profile(route_path)
        return make_scenario_on_route(
            route, n_cars=n_cars, t_span=t_span,
            F_trac_max_N=f_trac_max_N, k_curv_scale=k_curv_scale,
        )

    def run(self, scenario):
        return simulate_train_tensorized(scenario)   # the blocking solve_ivp
```

### Worker — keeps the blocking solve off the UI thread

```python
# app/workers/sim_worker.py
from PySide6.QtCore import QObject, Signal, Slot

class SimWorker(QObject):
    finished = Signal(object)   # TensorSimulationResult
    failed   = Signal(str)
    progress = Signal(int)

    def __init__(self, svc, scenario):
        super().__init__()
        self._svc, self._scenario = svc, scenario

    @Slot()
    def run(self):
        try:
            self.finished.emit(self._svc.run(self._scenario))
        except Exception as e:          # e.g. solve_ivp RuntimeError(sol.message)
            self.failed.emit(str(e))
```

### View — wires dials to the service, never blocks

```python
# app/views/sim_view.py  (sketch)
def on_run_clicked(self):
    scenario = self.svc.build_scenario(
        self.route_path,
        n_cars=self.cars_knob.value(),
        t_span=(0.0, self.duration_knob.value()),
        f_trac_max_N=self.traction_knob.value() * 1000,
        k_curv_scale=self.curve_knob.value() / 100,
    )
    self.thread = QThread()
    self.worker = SimWorker(self.svc, scenario)
    self.worker.moveToThread(self.thread)
    self.thread.started.connect(self.worker.run)
    self.worker.finished.connect(self.results_view.show_result)  # queued → safe
    self.worker.failed.connect(self.status_led.set_error)
    self.thread.start()
    self.status_led.set_running()
```

---

## 4. The threading boundary — and one honest design tension

Rule: **widgets live on the UI thread; backend work lives on workers; they only
communicate through signals** (Qt delivers cross-thread signals as queued, so it's
safe). This holds for the future torch training loop too — the loop runs on a worker
and emits `(epoch, loss)`; the monitor plots it. Never call torch from the UI thread,
never touch a widget from the worker.

```
UI thread  ──build scenario──►  worker thread  ──►  simulator2 (solve_ivp)
   ▲                                  │
   └────────── finished(result) ◄─────┘   (queued signal, thread-safe)
```

**The tension to decide on now, because it shapes the UI:** `simulate_train_tensorized`
integrates the **whole time span in one `solve_ivp` call**. So a dial can't *yet* alter
a simulation while it's running — there's no per-step hook. Two honest options:

- **(A) Configure → Run → Play back.** Dials set scenario parameters; you hit Run; the
  worker computes the full rollout; results animate in `results_view`. Zero backend
  changes. This is the right first version.
- **(B) True live dials.** Requires a *stepwise* integrator in the backend (a loop you
  can pause/step and feed new inputs each tick). This arrives naturally with the
  learned Δt surrogate in `PROJECT_DIRECTION.md` ("graph network simulator" step), which
  is already a stepped model. So: build (A) now; (B) becomes possible for free once the
  surrogate exists.

My suggestion: ship (A), and design the sim view so the Run button and the dials are the
same controls you'll later wire to a live loop — the layout doesn't change, only what
happens between ticks.

---

## 5. Screens (kept deliberately few, so it isn't overwhelming)

A left nav rail with four stops, mirroring the research workflow:

1. **Route** — pick from `saved_routes/`, preview the grade/curvature profile.
2. **Consist** — the car-type library + train builder + saved trains. See §9.
3. **Simulate** — the control panel: traction knob, duration knob, curvature scale,
   Run / Stop, a status LED, and live readouts.
4. **Results** — pyqtgraph stack: car speeds, coupler forces, actuator states.

Two more dock in later with no restructuring: **Training** (loss/metric monitor) and
**Control** (MPC/RL studio).

---

## 6. The cassette-futurism look (starter QSS)

Qt Style Sheets get you most of the way. Pair an amber-on-charcoal palette with a bitmap
mono font, inset/outset bevels for that molded-plastic feel, and chunky focus states.

```css
/* app/theme/cassette.qss — starter, tune freely */
* { font-family: "VT323", "IBM Plex Mono", monospace; font-size: 16px; }

QMainWindow, QWidget { background: #1b1a17; color: #e8c170; }

/* molded panel */
QGroupBox {
    background: #2a2722;
    border: 2px solid #0d0c0a;
    border-top-color: #46413a;     /* light top/left = raised */
    border-left-color: #46413a;
    border-radius: 2px;
    margin-top: 14px; padding: 10px;
}
QGroupBox::title { color: #8a7d5a; subcontrol-origin: margin; left: 8px; }

/* chunky button with travel */
QPushButton {
    background: #2f2b25; color: #e8c170;
    border: 2px solid #0d0c0a;
    border-top-color: #514a40; border-left-color: #514a40;
    padding: 6px 14px;
}
QPushButton:pressed {                /* inset when pressed */
    border-top-color: #0d0c0a; border-left-color: #0d0c0a;
    border-bottom-color: #514a40; border-right-color: #514a40;
    background: #221f1a;
}
QPushButton#runButton { color: #9ef0a0; }   /* green "GO" */

/* the dials */
QDial { background: #2a2722; }

/* CRT-ish readout */
QLabel#readout {
    background: #0b1a0b; color: #6cff7a;
    border: 2px solid #0d0c0a; padding: 4px 8px;
}
```

Two retro widgets worth building first, since they carry the aesthetic:

- **`Knob`** = `QDial` + a label + a 7-seg-style value readout underneath. This *is* your
  "dial." Map each to one scenario parameter.
- **`Led`** = a small painted circle with idle (amber) / running (green, optionally
  blinking) / error (red) states, driven by worker signals.

For the scanline/CRT flourish, overlay a semi-transparent repeating-gradient widget on
the results pane — optional, and easy to add late.

A note on the route map: your Streamlit app uses folium for clickable selection. In Qt
you'd either embed Leaflet in a `QWebEngineView`, or (simpler, and more on-theme) draw
the route polyline natively in pyqtgraph. For v1 I'd skip the interactive map and just
list the pre-generated routes from `saved_routes/` — far less to build.

---

## 7. Dependencies & packaging

```
PySide6        # the GUI framework (Qt6)
pyqtgraph      # fast realtime plots
numpy, scipy   # already yours (simulator2)
# later: torch (surrogate/control) — runs on the worker thread only
```

Packaging to a standalone app: **PyInstaller** (`pyinstaller app/main.py --windowed
--name "RailLab"`). Bundle the `.qss` and fonts as data files; `--collect-all PySide6`
handles the Qt plugins. One `.app` on macOS, one `.exe` on Windows, no Python install
required for the end user.

---

## 8. Suggested build order (each step runs on its own)

1. **Skeleton:** `main.py` + `main_window.py` with the nav rail and an empty Simulate
   screen, loading `cassette.qss`. Proves the shell + theme.
2. **Vertical slice:** hardcode one route path, wire the Run button → `SimWorker` →
   one pyqtgraph speed plot. Proves the threading boundary end-to-end.
3. **Dials:** replace hardcoded values with `Knob` widgets bound to scenario params.
4. **Engine: consist system (§9).** Add `catalog.py` + `consist.py`, a small default
   car library, `build_consist()`, and `make_scenario_from_consist()`. Pure backend,
   unit-testable with no UI.
5. **Route + Consist screens:** `RouteListModel` over `saved_routes/`; the Consist tab
   (car library + train builder + saved trains).
6. **Wire it together:** Simulate tab selects a saved Train + a Route → compile → run.
7. **Polish:** LEDs, readouts, results tabs, packaging.
8. **Later:** drop in `training_view` + `TrainingWorker` when the surrogate exists; then
   the control studio.

---

## 9. Trains & car types — the consist system

The goal: build a train the way you'd actually describe one ("1 loco, 4 coal cars, 2 tank
cars"), tune each car individually, save reusable **car types** (e.g. *tank car filled with
water*) to a library, save whole **trains**, and run any saved train on any saved route.

This is built **into the engine**, not just the UI — three small dataclass layers on top of
your existing `VehicleParameters` / `CouplerParameters`, so the surrogate and control code
inherit the same vocabulary.

### Three layers

```
CarType        a named, reusable template ("tank_car_water", "loco_sd70")
   │           → knows how to emit a VehicleParameters + its physical length + coupler
   ▼
CarInstance    one car placed in a train: a CarType ref + optional per-car overrides
   │           (so you can fatten one hopper without touching the type)
   ▼
Train          an ordered list of CarInstances + a name + metadata
               → build_consist(train) -> (List[VehicleParameters], List[CouplerParameters])
```

`build_consist()` is the single bridge back to your current engine: everything downstream
(`make_scenario_from_consist`, `simulate_train_tensorized`) is unchanged.

### `simulator2/catalog.py` — the car-type library

```python
from dataclasses import dataclass, field
from enum import Enum

class CarCategory(str, Enum):
    LOCOMOTIVE = "locomotive"
    HOPPER     = "hopper"      # coal, ballast
    TANK       = "tank"        # liquids
    BOXCAR     = "boxcar"
    FLATCAR    = "flatcar"
    GONDOLA    = "gondola"

@dataclass
class CarType:
    """A reusable template. Mass is split tare + lading so 'tank car *filled with
    water*' is just a lading on the same shell."""
    id: str                       # "tank_car_water"  (filename stem)
    name: str                     # "Tank car — water (full)"
    category: CarCategory
    length_m: float               # physical length → sets default coupler spacing

    mass_tare_kg: float           # empty car
    lading: str = "empty"         # "water", "coal", "" ...
    mass_lading_kg: float = 0.0

    davis_A: float = 0.0          # rolling/mech resistance (N)
    davis_B: float = 0.0          # (N·s/m)
    davis_C: float = 0.0          # aero (N·s²/m²)

    can_traction: bool = False
    F_trac_max_N: float = 0.0
    F_brk_max_N: float = 250_000.0

    coupler_type: str = "standard_knuckle"   # ref into a CouplerType library
    notes: str = ""

    @property
    def mass_kg(self) -> float:
        return self.mass_tare_kg + self.mass_lading_kg

    def to_vehicle(self) -> "VehicleParameters":
        from .params import VehicleParameters
        return VehicleParameters(
            mass_kg=self.mass_kg,
            davis_A=self.davis_A, davis_B=self.davis_B, davis_C=self.davis_C,
            can_traction=self.can_traction,
            F_trac_max_N=self.F_trac_max_N, F_brk_max_N=self.F_brk_max_N,
        )
```

A parallel `CouplerType` wraps `CouplerParameters` the same way (named, e.g.
`standard_knuckle`), so coupler tuning is also reusable rather than re-typed per car.

### `simulator2/consist.py` — instances, trains, and the bridge

```python
@dataclass
class CarInstance:
    car_type_id: str                       # ref into car_library/
    overrides: dict = field(default_factory=dict)   # per-car tweaks, e.g.
                                                     # {"mass_lading_kg": 80_000}
    label: str = ""                        # optional, e.g. "head-end helper"

@dataclass
class Train:
    name: str
    cars: list[CarInstance]                # ORDERED, head → tail
    description: str = ""
    schema_version: int = 1

# --- build "1 loco, 4 coal, 2 tank" ergonomically -----------------------
def from_blocks(name, blocks: list[tuple[str, int]]) -> Train:
    """blocks = [("loco_sd70", 1), ("coal_hopper_loaded", 4), ("tank_water", 2)]"""
    cars = [CarInstance(tid) for tid, n in blocks for _ in range(n)]
    return Train(name=name, cars=cars)

# --- the single bridge back to the existing engine ----------------------
def build_consist(train, library) -> tuple[list, list]:
    vehicles, couplers = [], []
    for inst in train.cars:
        ct = library.get(inst.car_type_id).with_overrides(inst.overrides)
        vehicles.append(ct.to_vehicle())
    # one coupler between each adjacent pair; length from the lead car's coupler_type
    for lead, _trail in zip(train.cars, train.cars[1:]):
        couplers.append(library.coupler_for(lead).to_coupler())
    return vehicles, couplers
```

### Persistence — mirror `saved_routes/`

Plain human-readable JSON, versioned, one file per item, exactly like your route files.

```jsonc
// car_library/tank_car_water.json
{ "schema_version": 1, "id": "tank_car_water", "name": "Tank car — water (full)",
  "category": "tank", "length_m": 18.0,
  "mass_tare_kg": 30000, "lading": "water", "mass_lading_kg": 80000,
  "davis_A": 600, "davis_B": 12, "davis_C": 0.7,
  "can_traction": false, "F_brk_max_N": 250000,
  "coupler_type": "standard_knuckle" }

// saved_trains/heavy_coal_drag.json
{ "schema_version": 1, "name": "Heavy Coal Drag",
  "cars": [
    { "car_type_id": "loco_sd70" },
    { "car_type_id": "coal_hopper_loaded" },
    { "car_type_id": "coal_hopper_loaded" },
    { "car_type_id": "coal_hopper_loaded" },
    { "car_type_id": "coal_hopper_loaded", "overrides": { "mass_lading_kg": 95000 } },
    { "car_type_id": "tank_water" },
    { "car_type_id": "tank_water" }
  ] }
```

Trains reference car types by `id` (not by value), so editing *tank_car_water* once updates
every train that uses it — with per-car `overrides` as the escape hatch when you want one
car different. (If you'd rather trains be fully self-contained snapshots, that's a one-line
choice in `build_consist`; flagged in the open questions.)

### One engine change this forces (worth deciding now)

Today the traction command is hardcoded to car index 0 (`u_trac_cmd` returns the ramp only
for `i == 0`). With heterogeneous consists and possibly **multiple or mid-train locos**,
traction should be driven by each vehicle's `can_traction` flag, not its position. Small
change in the scenario builder; calling it out because it's the one place the new data model
reaches into the physics.

### Consist tab layout

Three panes, left→right, matching the build-then-save workflow:

1. **Car Library** (left rail): saved car types grouped by category. Buttons: New, Edit,
   Duplicate. Editing opens `car_editor.py` — a plain form over the `CarType` fields
   (tare/lading split, Davis A/B/C, traction, brake, length, coupler type).
2. **Train Builder** (center): the ordered consist as a table. Add a car type × count from
   the library; drag to reorder; remove. Each row expands to reveal per-car **overrides**.
   A readout strip shows live totals: car count, total mass, total length, powered units.
3. **Saved Trains** (right or top): list like saved_routes — Load, Save, Duplicate, Rename.

Then the **Simulate** tab gains two dropdowns at the top — *Train* and *Route* — so "send any
train on any route" is literally pick-two-and-Run.

### Default seed library (so the app isn't empty on first launch)

Ship a handful built from your existing constants: `loco_sd70` (from `DEFAULT_M_LOCO_KG`,
loco Davis, `can_traction=True`, `F_trac_max_N=300000`), `freight_car_generic` (from
`DEFAULT_M_CAR_KG`, car Davis), plus `coal_hopper_loaded` and `tank_water` as worked
examples of the tare+lading split.

---

### Open questions for you
- Home for the app — `app/` at the repo root (assumed here), or its own repo?
- v1 route selection: plain list from `saved_routes/` (recommended) vs. interactive map?
- Configure-and-run (A) for v1, with live dials (B) deferred to the surrogate — agreed?
- **Trains reference car types by id (edits propagate) vs. self-contained snapshots** —
  which do you want as the default?
- **Per-car overrides vs. promote-to-new-type** — keep both, or force every variation to be
  a saved type?
- Multi-loco traction via `can_traction` flags (recommended) — confirm before I touch the
  scenario builder.
