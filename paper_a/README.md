# Paper A — Memory and chaos in a physics-constrained relaxation substrate

Substrate characterization paper (single author). Target: *International
Journal of Bifurcation and Chaos* (World Scientific) or *Chaos: An
Interdisciplinary Journal of Nonlinear Science* (AIP). Companion algorithm
paper: [`../paper_b/`](../paper_b/) (REDEM, target: *Neural Networks*).

## Contents

| File | Purpose |
|---|---|
| `PAPER_A.tex` | Submission-ready LaTeX (compiles standalone with `article`) |
| `PAPER_A.pdf` | Compiled PDF (12 pages, MiKTeX / pdflatex ×2, zero warnings) |
| `PAPER_A_draft.md` | Markdown draft (source prose; revision history in `../NEW_ALGORITHM_PLAN.md`) |
| `PAPER_A_sketch.md` | Outline, figure/table inventory, key numbers |
| `README.md` | This file |

## Compile

From this directory (figures resolve via `../figures/`):

```powershell
pdflatex PAPER_A.tex    # run twice for cross-references
# or, if Perl is installed:
latexmk -pdf PAPER_A.tex
```

## Submission checklist (Paper A)

- [ ] Replace `[Author Name]` / affiliation / email in `\author`.
- [ ] Swap the document class: `ws-ijbc.cls` for IJBC, or `revtex4-2` for
      Chaos (math, figures, and tables carry over unchanged).
- [ ] Keywords present: chaos; reservoir computing; memory capacity;
      forgetting kernel; homeostasis; relaxation substrate.
- [ ] Figures 1–4 + Supplementary Fig. S1 (all generated, 10 seeds, paired
      draws; vector `.pdf` in `../figures/`, `.png` preview twins):
      `../figures/paperA_fig1_substrate.pdf`,
      `../figures/substrate_phase_diagram_v2.pdf`,
      `../figures/forgetting_curve_theory.pdf`,
      `../figures/paperA_fig4_robustness.pdf`,
      `../figures/paperA_figS1_cv_sweep.pdf`.
- [ ] Tables: Table 1 (per-topology phase-diagram summary), Table 2
      (λ-homeostat gains); Supplementary Table S1 =
      `../data/substrate_phase_diagram_v2.csv`; Supplementary Note 1 (task-level
      CV sweep, `../data/s10_cv_sweep_v1.*`) with Fig. S1.
- [ ] Appendix A derivations complete: A.1 spectrum log-normality, A.2
      two-pass update order independence, A.3 kernel quadrature + median
      pinning + tail-slope asymptotics, A.4 Benettin iteration, A.5 MC
      estimator + linear-reservoir closed form.
- [ ] Companion citations: [4] prior Si₃N₄ pulse-encoding paper (device
      calibration); [12] Paper B (REDEM, [`../paper_b/`](../paper_b/)).
- [ ] Code availability statement:
      <https://github.com/huyamingc/REDEM> (private during review; public on
      acceptance).
- [ ] Cover letter: the per-pulse contrast coupling as a physical chaos knob;
      the analytic forgetting kernel (Pearson r = 0.97); the λ-homeostat
      restoring 8–18% of held-out memory after disturbances.
