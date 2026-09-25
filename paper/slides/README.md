# Defense deck

`main.tex` is the deck. Build it twice so overlay and frame counts settle:

```sh
pdflatex main.tex && pdflatex main.tex
```

Compiles clean with `pdflatex` on TeX Live 2026 — no shell-escape, no
lualatex, no fonts beyond what TeX Live ships.

## Layout

| File | Holds |
|---|---|
| `main.tex` | Document, title metadata, includes |
| `preamble.tex` | The Institutional style: palette, frametitle bar, footline, `\stattile` |
| `figures.tex` | TikZ figures |
| `charts.tex` | pgfplots charts, drawn natively so fonts match the deck |
| `acts/act1.tex` | Act I — Opening and Motivation, slides 1–8 |
| `acts/act2.tex` | Act II — Related Work and Positioning, slides 9–15 |
| `acts/act3.tex` | Act III — Problem Formulation, slides 16–27 |
| `acts/act4.tex` | Act IV — Reference Simulator, slides 28–40 |
| `acts/act5.tex` | Act V — Routes and Terrain, slides 41–48b |
| `acts/act6.tex` | Act VI — Tooling, GPU, and Data, slides 49–62 |
| `figures/` | Deck-only image assets (see its README) |
| `NOTES.md` | Open items, including what needs the workstation |
| `style-candidates/` | The three styles that were compared; B was chosen |

## Figures so far

- `\consistBand` — long consist as a quiet band. Title slide.
- `\trainFBD` — free-body diagram of a three-vehicle consist on a grade.
  `\trainFBD[1]` adds coupler-force annotations for plan slide 17.
- `\slackScene{1|2|3}` — bunched / take-up / run-out overlay build, with the
  coupler-force trace revealed progressively. Slide 3.
- `\surrogateFlow` — simulator → dataset → surrogate, with the validation loop
  back. Slide 6.
- `\actstrip{n}` — roadmap strip with act *n* highlighted. Slide 8, and reused
  by `\actdivider{n}{Title}` at the head of each later act.
- `\litMap{0|1}` — the four literatures as a 2×2 map; `1` reveals the "This
  Work" badge at the centre. Slide 9.
- `\mpStep{1|2|3}` — one message-passing step on a chain graph: the graph, the
  edge update, the node update. Slide 13; reused in the architecture act.
- `\benchRegimeChart` + `\benchLegend` — median wall clock per driving regime.
  Slide 5. Incomplete; see `NOTES.md`.
- `\taskMap` — three structured inputs → operator → four outputs. Slide 16.
- `\padWaste{1|2}` — a fixed token budget against a chain graph. Slide 18.
- `\nodeTensor`, `\edgeTensor` — the `H` and `E` feature matrices, channels
  exactly as `DATA_SCHEMA.md` lists them. Slides 19–20.
- `\rolloutTensor` — the `T × N × d` stack. Slide 21.
- `\imposedLearned{1|2|3}` — which state channels are learned and which are
  imposed. Slides 23 and 67; the central figure of the revised formulation.
- `\consistOnGrade` — one consist spanning several grade features. Slide 25.
- `\pipelineFig{n}` — the whole thesis as one flow, stage *n* highlighted.
  Slide 27, reused as the Act X divider and the closing slide.
- `\brakeBlendChart`, `\curvatureChart` — `sgn` against `tanh`, and Röckl's
  discontinuity at R = 300 m against the smooth AREMA law. Slide 24.
- `\stageLadder{n}` — the eight validation stages, rung *n* highlighted.
  Slide 28. `\stagepips{n}` is the compact in-slide version used on the stage
  slides themselves, where the ladder is too tall to sit beside the figure.
- `\scopeTable` — in-scope against out-of-scope. Slide 39, returned to at 94.
- `\couplerLawChart` — the constitutive law `F(δ)` with its ±20 mm deadband,
  from `simulator2/constants.py` defaults. `\couplerLawChartPlain` drops the
  text annotations for use as a small inset.
- `\routePipeline` — acquisition → resample → elevation → fields → export.
  Slide 42; fills the TODO placeholder at `05_route_acquisition.tex:14`.
- `\rawElevQAChart` — share of 10 m steps whose elevation jumps over a metre,
  per corridor, with the QA gate. Slide 48.
- `\routeFieldsChart` — grade, curvature and speed limit against chainage for
  the corrected Line 0. Slide 47.
- `\raillabLayers` — screens → services → backend. Slide 49; fills the TODO
  placeholder at `06_raillab.tex:24`.
- `\regimeGrid` — the eleven manoeuvre archetypes as small multiples,
  generated from `simulator2/driving_regimes.py`. Slide 53.
- `\throughputConsist`, `\throughputBatch` — wall clock flat in `N`, falling as
  `1/B`. Slide 56.
- `\diskLayout` — what a built corpus looks like on disk. Slide 59.
- `\fdConvergenceChart` — second-order convergence of the validator residual.
  Slide 61.

### Plot data

`figures/regime_*.dat` are the eleven control profiles generated locally from
`simulator2.driving_regimes` at seed 7 over 120 s — real output of the shipped
code, not a sketch. Regenerate with the snippet in `NOTES.md`.

`figures/grade_*.dat` and `figures/route_line0_fields.dat` are downsampled
route fields extracted from
`route_generator/route_profiles_v3/` on branch `surrogate-gns` and plotted
natively by pgfplots, so the fonts match the deck. The summary statistics on
slide 48b were recomputed from those arrays and reproduce
`ROUTE_PIPELINE_NOTE.md` exactly (rms 0.86 / 0.72 / 0.39 / 0.58 %).
Regenerate them with the snippet recorded in `NOTES.md` if the corridors
change.

Figures take their colors from five names the deck defines before
`\input{figures.tex}`: `fbCar`, `fbLoco`, `fbAccent`, `fbLine`, `fbMuted`.

## Conventions

- `\nextgroupplot` cannot be emitted from inside a `\foreach`. Unroll the
  panels, or the picture fails with a `Missing \endcsname` error.
- A comma inside a `\foreach` item splits the list. Brace-wrap any label that
  contains one: `2/{Two Bodies, Linear Coupler}`.
- `out` is a reserved TikZ key (curve exit angle). Never name a node style
  `out/.style` — it fails with a pgfkeys "requires a value" error.
- In pgfplots, a curve split into two `\addplot` calls consumes **two** legend
  entries. Mark the first `forget plot`, or the legend labels shift by one.
- A `decorate`/`brace` over a wide span renders almost flat. Use a thick
  coloured rule plus a label instead.
- TikZ's `\\` is **not** defined inside a nested `{\scriptsize ...}` group in
  node text. Apply the size change inline (`\\[0.15em]\scriptsize text`) rather
  than wrapping it in braces, or the picture fails to compile.
- Wrapped table columns use the `L{width}` column type (ragged right).
  Justified text at slide size opens visible rivers.
- Figure and axis labels are **Proper Case** — "Median Wall Clock per Run (s)",
  "Dynamic Brake Descent" — never the code's `snake_case` identifiers.
- Chart fills are `chartBlue` / `chartRed`, lighter steps of the deck's navy and
  crimson. The chrome colors are text colors and fail the categorical
  lightness and chroma checks as fills; the fills validate clean on all six
  (worst adjacent CVD ΔE 21.9 deutan, 32.2 tritan; normal-vision ΔE 28.0).
- `uniNavy` and `uniCrimson` are placeholders for the official CSU Pueblo
  values.
