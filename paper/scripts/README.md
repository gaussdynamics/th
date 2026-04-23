# Scripts Directory

This directory is for build scripts and utilities.

## Potential Scripts

- Build scripts (e.g., `build.sh`, `build.bat`)
- Clean scripts to remove build artifacts
- Figure generation scripts
- Data processing scripts for tables

## Example Build Script

You might create a `build.sh` (Linux/Mac) or `build.bat` (Windows) script:

```bash
#!/bin/bash
# build.sh - Compile the thesis

latexmk -pdf -pvc main.tex
```

Or for Windows PowerShell:
```powershell
# build.ps1
latexmk -pdf -pvc main.tex
```

## Clean Script

A clean script to remove build artifacts:

```bash
#!/bin/bash
# clean.sh - Remove LaTeX build artifacts

rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.synctex.gz *.toc *.lof *.lot
```
