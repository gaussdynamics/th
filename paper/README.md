# Thesis: Hybrid Physics-Transformer Surrogate Modeling for Train System Dynamics

This repository contains a LaTeX thesis scaffold focused on an internal modeling framework (hybrid physics + transformer surrogate) for train system modeling.

## Repository Structure

```
thesis/
├── main.tex                    # Main document file
├── preamble.tex                # LaTeX packages, macros, and configuration
├── metadata.tex                # Thesis metadata (title, author, date, etc.)
├── chapters/                   # Chapter files
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_problem_formulation.tex
│   ├── 04_framework_architecture.tex
│   ├── 05_data_and_training.tex
│   ├── 06_evaluation_and_validation.tex
│   ├── 07_discussion.tex
│   └── 08_conclusion_future_work.tex
├── figures/                    # Figure files (PDF, PNG, etc.)
├── tables/                     # Table source files (if separate)
├── refs/                       # Bibliography
│   └── references.bib
└── scripts/                    # Build scripts and utilities
```

## Compilation

### Using latexmk (Recommended)

```bash
latexmk -pdf -pvc main.tex
```

The `-pvc` flag enables continuous preview mode (auto-recompiles on file changes).

### Using pdflatex + biber (Manual)

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Note: Multiple `pdflatex` runs are needed for cross-references to resolve correctly.

### Build Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Biber (for bibliography processing)
- Required packages (should be available in standard distributions):
  - amsmath, amssymb, amsthm
  - graphicx, booktabs, siunitx
  - physics (optional)
  - hyperref, cleveref
  - biblatex

## Editing Metadata

Edit `metadata.tex` to customize:
- Thesis title
- Author name
- Date
- Department
- University

## Adding Content

### Figures

1. Place figure files (PDF, PNG, etc.) in the `figures/` directory
2. Reference figures in chapters using:
   ```latex
   \begin{figure}[h]
       \centering
       \includegraphics[width=0.8\textwidth]{figures/filename.pdf}
       \caption{Figure caption}
       \label{fig:label}
   \end{figure}
   ```
3. Reference figures using `\cref{fig:label}`

### Tables

1. Create tables directly in chapter files or in separate files in `tables/`
2. Use `booktabs` package for professional formatting
3. Label tables with `\label{tab:label}` and reference with `\cref{tab:label}`

### Bibliography

1. Add BibTeX entries to `refs/references.bib`
2. Cite in text using `\cite{key}` or `\parencite{key}`
3. The bibliography will be automatically generated using biblatex

### Chapters

Each chapter file contains structured outlines with:
- Section and subsection headings
- Placeholder paragraphs (1-3 per section)
- TODO markers indicating what content to add
- Figure and table placeholders
- Cross-references using `\cref`

## Notation Macros

The preamble defines consistent notation macros:
- `\route`: Route profile $\mathcal{R}$
- `\consist`: Consist $\mathcal{C}$
- `\controlplan`: Control plan $\mathcal{U}$
- `\energy`: Energy $E$
- `\arrivaltime`: Arrival time $T$
- `\couplerforce{i}`: Coupler force $F_{\text{c},i}$
- `\quantile{p}`: Quantile $Q_p$
- `\exceedprob{X}{\tau}`: Exceedance probability $\mathbb{P}(X > \tau)$

See `preamble.tex` for the complete list.

## Labeling Conventions

Use consistent prefixes for labels:
- `ch:` for chapters (e.g., `\label{ch:intro}`)
- `sec:` for sections
- `fig:` for figures (e.g., `\label{fig:architecture}`)
- `tab:` for tables (e.g., `\label{tab:results}`)
- `eq:` for equations (e.g., `\label{eq:energy}`)

Reference using `\cref{label}` for automatic formatting.

## Next Steps Checklist

- [ ] Replace placeholder bibliography entries in `refs/references.bib` with actual citations
- [ ] Expand chapter content: replace TODO markers with actual text
- [ ] Add figures to `figures/` directory and update references
- [ ] Create tables with actual data and update placeholders
- [ ] Fill in metadata in `metadata.tex`
- [ ] Review and refine notation macros in `preamble.tex` as needed
- [ ] Compile and verify all cross-references resolve correctly
- [ ] Review formatting and adjust as needed
- [ ] Add acknowledgments section if desired
- [ ] Final proofreading and content review

## Troubleshooting

### Bibliography not appearing
- Ensure `biber` is installed and run after `pdflatex`
- Check that `references.bib` path is correct in `preamble.tex`
- Verify BibTeX keys match citation commands

### Cross-references show "??"
- Run `pdflatex` multiple times (usually 2-3 times)
- Check that labels are unique and properly formatted
- Ensure `cleveref` package is loaded

### Missing packages
- Install missing packages using your LaTeX distribution's package manager
- For TeX Live: `tlmgr install <package>`
- For MiKTeX: Package Manager GUI

## License

[Add your license information here]
