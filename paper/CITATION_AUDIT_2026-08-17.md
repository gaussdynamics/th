# Thesis Citation Audit — August 17, 2026

Scope: `paper/` (the thesis). The TRB paper was audited previously; this pass brought the
thesis into line and re-verified everything independently against primary sources
(Crossref/DOI resolution, publisher records, arXiv, PMLR, agency report cover pages).

**Result: 9 of 33 citations were fabricated or materially wrong. All are now fixed.**
The document compiles clean with biber: 34 references, 0 undefined citations, 0 warnings.

---

## 1. Fabricated — source does not exist

| Key | What it claimed | Reality |
|---|---|---|
| `perez2016reduced` | Pérez, J. et al., "Reduced-order and lumped-parameter models for train dynamics," *Mechanical Systems and Signal Processing*, 2016 | No such article. Elsevier's Article Locator does not recognize the cited PII `S0888327016000126`. **Removed.** |
| `wu2019highfidelity` | Wu, Q. et al., "High-fidelity train longitudinal dynamics simulation," *Vehicle System Dynamics*, 2019, DOI `10.1080/00423114.2019.1589396` | DOI is unregistered — 404 at both doi.org and the Crossref API. No such VSD article. **Repointed** to the real paper: Wu, Cole & Spiryagin, "Parallel Computing Enables Whole-Trip Train Dynamics Optimizations," *J. Comput. Nonlinear Dyn.* 11(4):044503, 2016, DOI `10.1115/1.4032075`. |
| `Wu2018ComputingSchemesLTD` | Wu, Spiryagin & Cole, "Efficient computing schemes for longitudinal train dynamics," VSD 56(8):1237–1257, 2018, DOI `10.1080/00423114.2017.1415241` | DOI unregistered; no such article. The real paper is Wu & Cole, "Computing Schemes for Longitudinal Train Dynamics: Sequential, Parallel and Hybrid," *J. Comput. Nonlinear Dyn.* 10(6):064502, **2015**, DOI `10.1115/1.4029716`. **Corrected.** |
| `nrel2019cfd` | NREL, "High-Fidelity Computational Fluid Dynamics Simulation," NREL/TP-73474, 2019; note claiming "17 days on 672 CPU cores" | Report number does not exist. The PDF at that URL is **NREL/CP-5000-73474**, a conference preprint on extreme-load CFD for a *wave energy converter* — unrelated to rail. The 17-days/672-cores figure appears nowhere. **Removed.** |
| `nie_data_driven_dynamics` | Nie, Hao; Li, Zili; Iwnicki; Liu, Xian — VSD 2021 | Wrong authors, title, and year. Real: Nie, Tang, Liu, Chang & Zhang, "A data-driven dynamics simulation framework for railway vehicles," VSD 56(3):406–427, 2018, DOI `10.1080/00423114.2017.1381981`. **Corrected.** |
| `zhou_surrogate_integration` | Zhou, Yifan; Mei; Bruni — *Mech. Syst. Signal Process.*, 2022 | Wrong authors, journal, year. Real: Zhou, Meierhofer, Kugu, Xia & Grafinger, *Procedia CIRP* 119:345–350, 2023, DOI `10.1016/j.procir.2023.02.141`. **Corrected.** |
| `zhang_in_train_forces` | Zhang, Rui; Sun; Cole; Spiryagin — *IEEE T-ITS*, 2020 | Wrong authors, title, journal, year. Real: Zhang, Huang & Yan, "A data-driven approach for railway in-train forces monitoring," *Adv. Eng. Informatics* 59:102258, 2024, DOI `10.1016/j.aei.2023.102258`. **Corrected.** |
| `intrainnet` | Zhang, Rui; Sun; Spiryagin; Cole — *Proc. IMechE Part F*, 2021 | Wrong authors, subtitle, venue, year. Real: Zhang & Yan, "In-trainNet: A Two-Step Data-Driven Framework…," *Adv. Eng. Informatics* 65:103352, 2025, DOI `10.1016/j.aei.2025.103352`. **Corrected.** |
| `liu_rescheduling_surrogate` | Liu, Ziyang; Yang, Lixing; Gao, Ziyou — *Transportation Research Part C*, 2021 | Wrong authors, journal, year. Real: Liu, Cui, Dai, Yue & Yuan, *IEEE Trans. Autom. Sci. Eng.* 21(2):1107–1121, 2024, DOI `10.1109/TASE.2023.3338695`. **Corrected.** |

## 2. Real source, but cited for something it does not say

- **`versteeg2007introduction`** — the Versteeg & Malalasekera CFD textbook is genuine, but it was
  cited for a "50,000 CPU-hours" figure. It is an introductory finite-volume text and contains no
  CPU-hour benchmarks. **Removed** (the CFD rows left the table entirely — see §4).
- **`shabana_multibody`** — real book, but the entry said *Springer, 4th ed., 2020* with a fabricated
  Springer DOI (`10.1007/978-3-030-36579-7`, 404). It is **Cambridge University Press**; the 4th
  edition is 2013 and the **5th** is 2020. Corrected to 5th ed., CUP, 2020. It is no longer cited for
  a CPU-hour number; it is now cited in Ch. 2 for multibody formulation, which is what it covers.
- **`WheelRail2018TOES`** — the presentation is real and the URL is live, but the corporate author
  "{Wheel-Rail Interaction Seminar}" does not exist. Actual author: **Ralph Schorr (Amsted Rail)**,
  session PC07, WRI '18 Principles Course. Corrected, and the TOES row now also cites FRA report
  **DOT/FRA/ORD-23/19** as a stronger primary source.
- **`Cantone2008TrainDy`** — COMPRAIL 2008 was *Computers in Railways* **XI**, not X. Corrected to
  vol. 103, pp. 735–744, DOI `10.2495/CR080711` (which does resolve).
- **`Wei2014TDEAS`** — author list "Wei, W.; Hu, Y.; Wu, Q. et al." in *Proc. IMechE Part F* was wrong.
  Real: Wu, Luo & Cole, *J. Modern Transportation* 22(3):127–136, 2014, DOI `10.1007/s40534-014-0055-x`.
  Its abstract does introduce TDEAS as developed in China, so the table row's substance is correct.

## 3. Duplicate key

`Spiryagin2017BenchmarkQuestions` was defined **twice** in `references.bib` with conflicting data.
The 7-author / VSD 55(8):1205–1227 version is fabricated (its DOI is unregistered). The correct
entry is Spiryagin, Wu & Cole, VSD **55(4):450–463**, 2017, DOI `10.1080/00423114.2016.1270457`.
Duplicate deleted.

## 4. The Ch. 1 computational-cost table — rebuilt

This was the worst spot: all five CPU-hour values (0.05 / 50 / 500 / 50,000 / 275,000) were
attached to sources that state none of them, and two of the five sources did not exist.

You were mid-decision on this when the connection dropped, so I took the conservative route:
**the table now reports only figures stated directly in the cited works**, and it stays on rail
rather than detouring through CFD. Rows are now:

| Task | Reported cost | Source |
|---|---|---|
| Whole-trip LTD optimization study, sequential | ~18 months | Wu, Cole & Spiryagin 2016 |
| Same study, parallel across multiple processors | 11 days | Wu, Cole & Spiryagin 2016 |
| Coupler-force evaluation, share of LTD solver runtime | ~65% | Wu & Cole 2015 |
| Multibody vehicle–track simulation, 5 km segment | ~30 min | Zhou et al. 2023 |
| Learned surrogate, same 5 km segment | ~8 s | Zhou et al. 2023 |

Every figure above is quoted from the source's own abstract or text. The caption notes they are
not normalized to a common hardware baseline. A LaTeX comment above the table records what was
removed and why.

**If you preferred one of the other options** (drop the table for prose, or keep unsourced
order-of-magnitude estimates labeled as illustrative), the old version is recoverable from git and
the swap is a two-minute edit.

## 5. Prose corrected in Ch. 2

Six sentences described papers inaccurately once the real sources were identified:

1. **Nie et al.** — was "multibody vehicle–track simulations"; the paper has no track model
   (vertical and longitudinal vehicle dynamics, MATLAB/Simulink ↔ SIMPACK co-simulation). Also
   its surrogate is **Legendre polynomial regression** benchmarked against Kriging, not a neural net.
2. **Zhou et al.** — "These studies demonstrate speed increases multiple orders of magnitude" was
   plural on the strength of one reference, and unquantified. Now states the paper's actual figure:
   ~30 min (SIMPACK) → ~8 s (surrogate) for a 5 km segment.
3. **In-trainNet** — "combines shared feature extraction with configuration-specific adaptation"
   replaced with the paper's actual two-step structure (multi-task pre-training on several couplers
   of one configuration, then transfer to other configurations).
4. **Limitations paragraph** — the claim that force-estimation models exclude "braking modes,
   loading variations, or complex draft-gear hysteresis effects" is **not supported** by Zhang et al.
   (their inputs include traction/dynamic braking effort). Replaced with the verifiable limitation:
   single coupler position, single train configuration. Likewise the claim that In-trainNet degrades
   "where large strong nonlinear interactions between vehicles dominate" appears nowhere in that
   paper — replaced with its real residual limitation (still needs configuration-specific adaptation
   data).
5. **Liu et al.** — "decompose the problem into multiple surrogate submodules" → multi-surrogate
   search; added that its transfer learning addresses incomplete/imbalanced disruption data.
6. **Gal & Ghahramani** — "interpreting dropout at inference time as a variational approximation"
   inverts the argument. Dropout *training* is the variational approximation; keeping dropout on at
   test time draws Monte Carlo samples from the resulting approximate posterior. Reworded.

## 6. Housekeeping

- Removed 10 unused `placeholder_*` TODO stub entries.
- Converted `blundell2015weight`, `gal2016dropout`, `guo2017calibration`, `kuleshov2018accurate`
  from `@article`-with-proceedings-in-the-journal-field to proper `@inproceedings` with PMLR volume
  and page ranges. (All four are real; only the entry structure was wrong.)
- Added missing DOIs throughout; fixed the invalid ISBN on `Cole2006HandbookLTD`
  (9780849333215 → 9780849333217, checksum was wrong).
- `shafer2008tutorial`, `koenker1978regression`, `gneiting2007proper`, `romano2019conformalized`,
  `vaswani2017attention`, and the neural-operator group verified clean.

---

## Two things left for you

1. **`\printbibliography` is commented out in `main.tex`** (line ~123), along with `\backmatter`.
   Nothing renders as a reference list until you uncomment it. I left it as you had it.
2. **The abstract in `main.tex`** claims LTD simulator cost is "tens to hundreds of CPU-hours per
   scenario." That number has no citation and appears to trace back to the fabricated table. The
   sourced framing would be closer to: a *whole-trip optimization campaign* took ~18 months
   sequentially and 11 days in parallel — that is a campaign of many runs, not one scenario. Worth
   rewording before submission.

## Files changed

- `paper/refs/references.bib` (rewritten)
- `paper/chapters/01_introduction.tex` (cost table)
- `paper/chapters/02_related_work.tex` (six prose fixes, one added citation)
- `paper/tables/ltd_tools_comparison.tex` (TOES row)

## Verification performed

`pdflatex` → `biber` → `pdflatex` ×2 with `\printbibliography` enabled: **0 undefined citations,
0 biber warnings, 34 references rendered.** Every DOI in the final bibliography was resolved
against Crossref or the publisher; every URL was fetched and confirmed live.
