# REDEM — Physics-Grounded Online Learning Architecture

This repository contains the complete code, data, and figures for two
companion papers:

- **Paper A**: *"Memory and chaos in a physics-constrained relaxation
  substrate: phase diagram, multi-timescale forgetting, and disturbance
  robustness"* — substrate characterization (target: *International Journal of
  Bifurcation and Chaos* / *Chaos*)
  → See [`paper_a/PAPER_A.pdf`](paper_a/PAPER_A.pdf) and
  [`paper_a/PAPER_A.tex`](paper_a/PAPER_A.tex)

- **Paper B**: *"REDEM: Training-Inference Unified Learning with
  Meta-Adaptation and Structural Plasticity for Non-Stationary Environments"*
  — the online learning architecture (target: *Neural Networks*)
  → See [`paper_b/PAPER_B.pdf`](paper_b/PAPER_B.pdf) and
  [`paper_b/PAPER_B.tex`](paper_b/PAPER_B.tex)

All experiments are CPU-only and fully reproducible via the scripts in
`scripts/` (a shared CORE substrate and readout library supports both papers).
Data and figures live in `data/` and `figures/`.

- `README_REDEM.md` — detailed technical README: architecture, headline
  results (S1–S10), script inventory, reproduction commands.
- `NEW_ALGORITHM_PLAN.md` — the authoritative S0–S10 research plan and
  changelog.

## Repository layout

```
├── paper_a/     Paper A: substrate characterization (PDF, LaTeX, drafts)
├── paper_b/     Paper B: REDEM online learning architecture (PDF, LaTeX, drafts)
├── scripts/     Shared simulation code (CORE substrate, tasks, readouts, figure scripts)
├── data/        All experiment results (CSV + JSON, 10-seed means)
├── figures/     All publication figures (PNG)
└── *.md         Overview, technical README, research plan, coding standards (CLAUDE.md)
```

## Code availability

All simulation code and data required to reproduce the results are available
at <https://github.com/huyamingc/REDEM> (private during review; made public on
acceptance).
