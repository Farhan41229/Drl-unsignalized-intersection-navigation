# CSE 4610 Design Project — Presentation Source

## Contents
- `presentation.tex` — main Beamer source (custom navy/blue theme, serif font, rounded blocks)
- `references.bib` — bibliography (all survey/paper references cited in the deck)
- `img/` — all figures
  - `make_charts.py` — regenerates the four data-driven charts from the project's own reported results
  - `*.png` — pre-generated chart images (already committed, no need to re-run the script)

## How to compile
Requires a standard TeX Live install (no special theme packages needed — the theme is built
from core `beamer` + `tikz` + `pgfplots` + `booktabs`/`tabularx`).

```bash
pdflatex presentation.tex
bibtex presentation
pdflatex presentation.tex
pdflatex presentation.tex
```

## Structure (36 slides)
1. **Introduction** — split into two subsections as requested:
   - *The Problem* — stat-infographic (36% / \$371B / 40%+) + short bullets
   - *Motivation* — AV vs. CAV diagram, Signal-Free AIM + Deep RL blocks
   - Core Challenge, Key Objectives
2. **Related Work** — taxonomy tree diagram, literature stats table, approach comparison table,
   two RL-literature tables (single-ego threads, graph/cooperative-AIM threads)
3. **Methodology** — roadmap diagram, reproduction target, disclaimer, current progress
   (with a visual 2/8 progress bar), milestone diagram, state/action/reward, pipeline diagram, SAC
4. **Implementation** — layered architecture-stack diagram, training-setup table, systems-engineering chart
5. **Experimental Results** — 3 result charts + simulation-physics-gap explanation + summary
6. **Challenges and Limitations** — 2x3 grid of short challenge cards (new section, filled in)
7. **Conclusion and Future Direction** — novelty, generalization-strategy diagram, next steps, conclusion
8. **GitHub Link and Individual Contribution** — repo box + contribution table (new section, filled in)
9. **References**

## Placeholders you must edit before presenting
- **GitHub link** (slide "GitHub & Individual Contributions"): currently
  `https://github.com/<team-org>/cav-intersection-drl` — replace with your real repository URL.
- **Individual contributions table** (same slide): the Name/ID rows are correct, but the
  "Key Contributions" column is an illustrative placeholder — edit it to reflect what each
  teammate actually did.

## Regenerating the charts (optional)
If you update any of the numbers in `img/make_charts.py` (e.g. after finishing the remaining
6/8 paper configurations), just re-run:

```bash
cd img && python3 make_charts.py
```

This overwrites the four PNGs in place (using the navy/accent-blue palette that matches the
slide theme); no changes to the `.tex` file are needed.

## Editing tips
- Team / supervisor / course info: edit the `\author`, `\institute`, `\title`, `\date` near the top.
- Color palette lives in one place near the top of the preamble (`mainblue`, `accentblue`,
  `blockbodyblue`, `mutedblue`) — change those `\definecolor` lines to re-theme everything,
  including the TikZ diagrams (which reuse the same colors via the `navy`/`teal` aliases).
- All numeric results (success rate, collision rate, speeds, throughput) are taken directly
  from `README.md` / `Environment.md` and the milestone reports you provided — update the
  relevant `\includegraphics` charts and inline numbers together if these change.
