# CSE 4610 Design Project — Presentation Source

## Contents
- `presentation.tex` — main Beamer (Metropolis theme) source, 29 content slides + references
- `references.bib` — bibliography (all survey/paper references cited in the deck)
- `img/` — all figures
  - `make_charts.py` — regenerates the four data-driven charts from the project's own reported results
  - `*.png` — pre-generated chart images (already committed, no need to re-run the script)

## How to compile
Requires a TeX Live distribution with the `beamertheme-metropolis` and `pgfplots` packages
(both are in the default TeX Live "full" install).

```bash
pdflatex presentation.tex
bibtex presentation
pdflatex presentation.tex
pdflatex presentation.tex
```

The final `presentation.pdf` will contain the full slide deck.

## Regenerating the charts (optional)
If you update any of the numbers in `img/make_charts.py` (e.g. after finishing the remaining
6/8 paper configurations), just re-run:

```bash
cd img && python3 make_charts.py
```

This overwrites the four PNGs in place; no changes to the `.tex` file are needed.

## Editing tips
- Team / supervisor / course info: edit the `\author`, `\institute`, `\title` block near the top.
- Slide order follows: Context → Problem → Related Work → Baseline Paper & Reproduction →
  RL Formulation & Architecture → Results → Novelty & Future Work → Conclusion → References.
- All numeric results (success rate, collision rate, speeds, throughput) are taken directly
  from `README.md` / `Environment.md` and the milestone reports you provided — update the
  relevant `\includegraphics` charts and inline numbers together if these change.
