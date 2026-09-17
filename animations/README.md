# Train Tensor Animations (Manim)

This module adds a rendering layer on top of `simulator2` outputs so animation uses the **existing** longitudinal train dynamics and tensorized state definitions (`H_hist`, `E_hist`) instead of re-implementing physics in Manim.

## What It Includes

- Exporter adapter from `simulate_train_tensorized(...)` to animation-friendly `.npz`.
- Reusable data loader and tensor matrix visualizer.
- Main side-by-side scene: train model + evolving tensor slice.
- Additional scenes for time-index explanation, coupler-wave focus, and tensorization concept.

## Folder Layout

- `animations/data/` generated demo run files.
- `animations/scenes/` Manim scenes.
- `animations/utils/` exporter, loader, train shapes, color mapping, matrix visualizer.
- `animations/render.py` scene render helper.

## 1) Generate Demo Simulation Data

From repository root:

```bash
python animations/utils/simulation_exporter.py
```

This writes:

- `animations/data/demo_tensor_run.npz`

## 2) Visualized Simulator Fields

Exporter pulls simulator output from:

- `result.t`
- `result.H_hist`
- `result.E_hist`
- sampled `scenario.u_trac_cmd(t,i)` and `scenario.u_brk_cmd(t,i)`

Saved convenience channels include:

- `positions_m` from `H_hist[:,:,x]`
- `velocities_mps` from `H_hist[:,:,v]`
- `z_brk_n` from `H_hist[:,:,z_brk]`
- `z_trac_n` from `H_hist[:,:,z_trac]`
- `coupler_forces_n` from `E_hist[:,:,f_cpl]`
- `tensor_slice` shaped `[t, vehicle, feature]` with features:
  - `x_m`, `v_m_per_s`, `z_brk_n`, `z_trac_n`, `f_cpl_left_n`

## 3) Feature-to-Visual Mapping

- Train left panel:
  - car horizontal motion from `positions_m`
  - locomotive velocity arrow color from `velocities_mps`
  - coupler connector colors from `coupler_forces_n`
- Tensor right panel:
  - rows = vehicle index
  - cols = feature channel
  - cell colors from `tensor_slice[t,:,:]`

## 4) Color Coding

`animations/utils/color_mapping.py` centralizes color logic:

- Diverging map (`blue -> white -> red`) for signed values:
  - velocity
  - coupler force
- Sequential map (`blue -> green`) for nonnegative / magnitude channels.
- Normalization is fixed over the full loaded run for temporal consistency.

## 5) Render Scenes

Direct Manim commands:

```bash
manim -pqh animations/scenes/train_tensor_side_by_side.py TrainTensorSideBySide
manim -pql animations/scenes/tensor_time_evolution.py TensorTimeEvolution
manim -pql animations/scenes/coupler_wave.py CouplerWave
manim -pql animations/scenes/tensorization_advantage.py TensorizationAdvantage
manim -qh animations/scenes/three_vehicle_matrices.py ThreeVehicleMatrices
```

Using helper script:

```bash
python animations/render.py --scene train --quality l --preview
python animations/render.py --scene all --quality m
```

## 6) Scene Notes

- `TrainTensorSideBySide`: main thesis-ready explanatory scene.
- `TensorTimeEvolution`: explains `X[t, vehicle, feature]` slicing.
- `CouplerWave`: animates force propagation from `E[..., f_cpl]`; if unavailable, scene explains requirement.
- `TensorizationAdvantage`: conceptual sequential-vs-batched update comparison.
- `ThreeVehicleMatrices`: the schema explainer -- see the section below.

## Three-Vehicle Matrix Scene

`ThreeVehicleMatrices` is the schema explainer: one locomotive and two loaded
coal hoppers running 1.5 km of `route_line5_22km`, with all five schema arrays
filling in live beside them.

```bash
python animations/utils/segment_run.py            # build the run (once)
python animations/render.py --scene three --quality h
```

**What is on screen**

| Region | Shows |
|---|---|
| top left | the consist, coupler travel exaggerated x40, coupler bars heat-mapped by `F_cpl` |
| top centre | the corridor as an isometric block: real OSM alignment, real USGS 3DEP terrain contours, a marker at the lead vehicle |
| top right | route fields sampled at the lead vehicle (chainage, grade, curvature, radius, speed limit, elevation) |
| middle | the three node matrices: `state [3x4]`, `node_static [3x7]`, control `[3x2]` |
| lower left | the two edge matrices: `edge_dynamic [2x3]`, `edge_static [2x6]` |
| lower right | the control input `u(t)` with a playhead |

Dashed leader lines tie each matrix **row** to the vehicle or coupler it
describes. Static panels (`node_static`, `edge_static`) are drawn muted and
never change; dynamic panels are heat-mapped and run. That contrast is the
point -- it is why `DATA_SCHEMA.md` stores the two kinds separately.

**Three vehicles, and why**

`N = 3` is what makes every matrix legible at once: 3x4, 3x7, 3x2, 2x3 and 2x6
all fit on one frame with their numbers readable, and there are few enough rows
to wire each one to its object. The same scene at `N = 75` would be five grey
rectangles.

**What is real and what is not**

* The alignment, the grade field, the curvature and the speed limit are the
  corridor's own, and the dynamics are `simulator2` with the `linear` curvature
  model at `k_curv_scale = 1` -- the same configuration as the `data/v2` build.
* Terrain contours are **real**: USGS 3DEP sampled on a 28x64 grid over the
  segment by `animations/utils/terrain_dem.py`, cached in
  `animations/data/terrain_<route>.npz` so rendering needs no network. Delete
  the cache to re-fetch. If the service is unreachable and no cache exists, the
  exporter falls back to interpolating the track's own elevation and says so in
  the run metadata.
* The **drawn** elevation profile integrates the route's `sin_theta` field
  rather than using the raw 3DEP samples on the alignment. Those raw samples
  jump up to 3 m between adjacent 10 m vertices -- a 30 % grade where the fitted
  route field says 3.5 % -- and at 34x vertical exaggeration that renders noise
  instead of terrain. Both are exported (`seg_z` raw, `seg_z_model` integrated).
* Two exaggerations are applied and both are labelled on screen: elevation x34
  and coupler travel x40. Real slack is +-25 mm on a 29 m train.
* Playback is 4x: the 100 s run plays in 25 s.

**Choosing the segment**

`SEG_S0_M`/`SEG_S1_M` in `segment_run.py` were picked by scoring 1.6 km windows
across every corridor on grade spread and mean curvature, skipping
`route_line2_33km` (its grade field sits on the +-4 % clamp for a quarter of its
length). The chosen window climbs 12.7 m, curves to a 550 m radius and carries a
20 m/s limit. The control profile is four traction notches and a light brake at
the end, tuned so the train stays under the speed limit and finishes inside the
window.

## 7) Extending with New Features

1. Add channel extraction in `simulation_exporter.py`.
2. Append name to `tensor_slice_feature_names`.
3. Update column labels in scenes (or drive labels from data directly).
4. Add color rule in `color_mapping.py` if signed/magnitude behavior differs.

## Environment Notes

Install dependencies:

```bash
pip install -r animations/requirements.txt
```

If Manim system dependencies are missing, follow the official setup docs:
[https://docs.manim.community/](https://docs.manim.community/)
