# Paper A — Memory and chaos in a physics-constrained relaxation substrate

Substrate characterization paper (single author). Target: *International
Journal of Bifurcation and Chaos* (World Scientific) or *Chaos: An
Interdisciplinary Journal of Nonlinear Science* (AIP). Companion algorithm
paper: [`../paper_b/`](../paper_b/) (REDEM, target: *Neural Networks*).

## Contents

| File | Purpose |
|---|---|
| `PAPER_A.tex` | Submission-ready LaTeX (compiles standalone with `article`) |
| `PAPER_A.pdf` | Compiled PDF (14 pages, MiKTeX / pdflatex ×2, zero warnings) |
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

- [x] Author filled (2026-02-19): Yaming Hu, ORCID 0009-0003-1406-0485,
      Independent Researcher, Guiyang, Guizhou Province, China;
      64687555@qq.com. Cover letter: `COVER_LETTER.md`.
- [ ] Swap the document class: `ws-ijbc.cls` for IJBC, or `revtex4-2` for
      Chaos (math, figures, and tables carry over unchanged).
- [ ] Keywords present: chaos; reservoir computing; memory capacity;
      forgetting kernel; homeostasis; relaxation substrate.
- [ ] Figures 1–4 + Supplementary Fig. S1 (all generated, 10 seeds, paired
      draws; vector `.pdf` in `../figures/`):
      `../figures/paperA_fig1_substrate.pdf`,
      `../figures/substrate_phase_diagram_v2.pdf`,
      `../figures/forgetting_curve_theory.pdf`,
      `../figures/paperA_fig4_robustness.pdf`,
      `../figures/paperA_figS1_cv_sweep.pdf`.
- [ ] Tables: Table 1 (per-topology phase-diagram summary), Table 2
      (λ-homeostat gains); Supplementary Table S1 =
      `../data/substrate_phase_diagram_v2.csv`; Supplementary Note 1 (task-level
      CV sweep, `../data/s10_cv_sweep_v1.*`) with Fig. S1.
- [x] Follow-up evidence integrated (2026-08-24/25): §4.2 clip-range ablation ×
      fine κ grid (κ* = 25.3/27.4/27.9; transition coupling-driven — invariant
      to 5× clip widening for the memory-relevant topologies,
      `../data/s27_clip_kappa_fine_v1.*`); kernel shape stability under
      coupling (physical-kernel r = 0.91–0.99 over κ 15–30,
      `../data/kernel_coupling_shape_v1.*`; effective-median dilation reported
      qualitatively); λ-homeostat robust to FTLE estimation noise
      (σ up to 0.10, no significant degradation,
      `../data/s32_ftle_noise_v1.*`); Discussion adds the "spectrum knobs:
      location vs width" paragraph (CV ↔ τ_m ↔ log-uniform range across the
      companion papers).
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
