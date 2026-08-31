# TRB Annual Meeting Paper — LTD Simulation Testbed

Manuscript covering the **preliminary / infrastructure phase** of the thesis: the
problem formulation, reference simulator, route-acquisition pipeline, and RailLab
console that together produce training data for the graph neural surrogate. No
surrogate is trained in this paper.

Source material: chapters 1–6 of the thesis in `../paper` (referenced only, never modified).

## Files

| File | Purpose |
|---|---|
| `ltd_surrogate_trb.tex` | Main manuscript — **this is the file to compile** |
| `ltd_surrogate_trb.bib` | BibTeX database (author–date, rendered by `trb.bst`) |
| `trbunofficial.cls` | TRB paper class (unchanged from template) |
| `trb.bst` | TRB bibliography style (unchanged from template) |
| `latexmkrc` | Sets the timezone so the title-page submission date follows TRB's event clock |
| `figures/` | Figures used in the paper |
| `figures/unused/` | Figures carried over from the thesis but cut for length — safe to delete |
| `_template_original/` | The pristine template files, kept for reference — safe to delete |

## Building

```bash
latexmk ltd_surrogate_trb.tex -pdf
```

On Overleaf: upload the whole folder and set `ltd_surrogate_trb.tex` as the main document.

## Compliance notes

- **Length: 19 pages.** TRB's 2027 limit is 20. The title page auto-reports the count
  via `\totalpagecount`. Adding roughly a page of text will push it over, so check the
  title-page number after any substantial edit.
- **Structured abstract** with the five required headings (Objectives, Methods,
  Findings, Novelty, Practical Applications).
- **Author–date citations** via `natbib` + `trb.bst`; reference list is alphabetical.
- **Line numbers** enabled through the `numbered` class option. Display equations are
  not line-numbered — this is inherent to the template and matches its own output.
- Figure captions are sentence case, table captions Title Case, both bold, per the class.

## Items to review before submission

The following are marked with `% TODO` comments in the `.tex`:

- ORCIDs are omitted from both `\TRBauthor` entries — add if desired.
- Author contributions (CRediT roles) — verify the split is accurate.
- Declaration of conflicting interests — currently "no potential conflicts."
- Funding — currently "no external financial support." Replace if a grant applies.

## Content trimmed for the page limit

Cut to fit the limit; restore from `../paper` or `figures/unused/` if space allows:

- The OSM base-map figure, the acquisition-panel screenshot, the RailLab controls
  screenshot, the resampled-route inspection view, the 3-D elevated route, the
  elevation profile, and the Stage 1 and Stage 6 simulator plots.
- The extended uncertainty-quantification literature review (thesis Ch. 2), compressed
  here to one subsection since UQ enters this work only as a coupler-force safety margin.
- The computational-cost table from thesis Ch. 1, inlined as prose in the introduction.

## Framing note

Thesis Ch. 1 still describes a "physics–transformer hybrid," while Ch. 3 and the thesis
abstract describe a graph neural network with the transformer demoted to an ablation.
This paper follows the latter, consistent with its title.
