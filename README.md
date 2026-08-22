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

**Two papers, one pipeline — two different stories.** Paper A is a physics /
nonlinear-dynamics theory paper about what the substrate *computes*; Paper B
is a machine-learning paper about how to *learn on* it. They share the same
simulation code and data but ask different questions:

| | Paper A — substrate characterization | Paper B — REDEM online learning |
|---|---|---|
| **Question** | What can this physical substrate compute? | How do you learn on top of it? |
| **Content** | Dynamics theory: memory–chaos phase diagram, forgetting kernel, λ-homeostat robustness (full derivations in Appendix A) | Learning algorithm + benchmarks: online RLS readout, dual-timescale metadata, chaos homeostat, structure plasticity, ablations |
| **Key results** | Order–chaos transition at κ*∈(25,30); held-out memory +24–53% just before it; forgetting kernel M(t)=∫p(τ)e^{−t/τ}dτ matches the measured memory curve (r=0.97); λ-homeostat restores 8–18% after disturbances | Tracks drift where frozen batch learners (GRU, transformer) fail permanently; integrated system matches or beats every ablation and the bare baseline (0.996 vs 0.988/0.994/0.973; N=1024: 0.998 vs 0.976); metadata is substrate-agnostic and transfers to a matched ESN |
| **Target journal** | IJBC / Chaos | Neural Networks |
| **Relationship** | Substrate theory; cites the prior Si₃N₄ pulse-encoding paper for device calibration | Builds on Paper A's substrate theory (cited as the companion in §2) |

All experiments are CPU-only and fully reproducible via the scripts in
`scripts/` (a shared CORE substrate and readout library supports both papers).
Data and figures live in `data/` and `figures/`.

- `README_REDEM.md` — detailed technical README: architecture, headline
  results (S1–S10), script inventory, reproduction commands.
- `NEW_ALGORITHM_PLAN.md` — the authoritative S0–S10 research plan and
  changelog.

## Repository layout

```
├── paper_a/     Paper A: substrate characterization (PDF, LaTeX, drafts, submission README)
├── paper_b/     Paper B: REDEM online learning architecture (PDF, LaTeX, drafts, submission README)
├── scripts/     Shared simulation code (CORE substrate, tasks, readouts, figure scripts)
├── data/        All experiment results (CSV + JSON, 10-seed means)
├── figures/     All publication figures (vector PDF; no raster twins)
└── *.md         Overview, technical README, research plan, coding standards (CLAUDE.md)
```

Each paper folder contains its own `README.md` with the submission checklist
(author placeholder, journal template swap, figures/tables inventory, cover
letter points).

## Scripts

All 22 committed scripts in `scripts/`, typed per `CLAUDE.md`
(ML > CORE > PAPER > FIG > EXPLORE). Two legacy scripts from the prior
Si₃N₄-pulse-encoding project are kept as shared dependencies (imported by the
new code) and are never modified.

| Script | Type | Purpose |
|---|---|---|
| `recurrent_substrate.py` | CORE | per-pulse contrast-coupled relaxation substrate (numba core); self-test 4/4 |
| `shallow_trap_array_simulator.py` | CORE (legacy) | Si₃N₄ shallow-trap device simulator; constants γ/τ₀/gen_tau_vec/preprogram_vec imported by 12 scripts |
| `fair_esn_comparison.py` | ML (legacy) | matched ESN reservoir class; imported by `baseline_showdown.py` and `esn_metadata_comparison.py` |
| `streaming_tasks.py` | CORE | task generators: drift_binary, narma10, mackey_glass, context_switch, regime_switch |
| `online_readout.py` | CORE | OnlineRLS, ThreeFactorReadout, ridge_fit, MC/accuracy metrics |
| `substrate_recurrence_characterization.py` | PAPER | S1: FTLE / held-out MC / separation vs κ sweep (610 runs) |
| `online_readout_streaming.py` | PAPER | S2: online RLS vs offline ridge on streaming tasks |
| `three_factor_online_readout.py` | PAPER | S3: reward-modulated Hebbian vs error-gated vs RLS (negative result) |
| `intrinsic_reward_experiment.py` | PAPER | S4: novelty intrinsic reward ablation (negative result) |
| `dual_timescale_metadata.py` | PAPER | S5: fast/dual/slow metadata on regime-switch |
| `chaos_regulator.py` | PAPER | S6: λ-homeostat under disturbances |
| `structure_plasticity.py` | PAPER | S7: correlation-guided rewiring |
| `integrated_benchmark.py` | PAPER | S8: full system vs ablations; N=1024 confirmation |
| `baseline_showdown.py` | ML | S9: vs matched ESN / GRU / tiny transformer (torch CPU) |
| `forgetting_curve_theory.py` | EXPLORE | S10: forgetting-kernel theory M(t), Gauss–Hermite, r=0.97 validation |
| `esn_metadata_comparison.py` | PAPER | S10: metadata transfer to a matched ESN |
| `cv_sweep.py` | PAPER | S10: task-level CV sweep |
| `gen_architecture_schematic.py` | FIG | Paper Fig. 1 schematics (substrate / REDEM, M4↔M5 loop) |
| `gen_substrate_phase_diagram.py` | FIG | S1 phase-diagram figure |
| `gen_s2_curves.py` | FIG | S2 learning curves |
| `gen_paper_figures.py` | FIG | batch: robustness / metadata / ablation / showdown |
| `gen_paperA_supp_figures.py` | FIG | Paper A Supplementary Fig. S1 (CV sweep) |

Each FIG script emits a single vector `.pdf` (journal submission); the papers
include the extension-less basename so `pdflatex` picks the vector file
automatically. Reproduction commands are in `README_REDEM.md`.

## Code availability

All simulation code and data required to reproduce the results are available
at <https://github.com/huyamingc/REDEM> (private during review; made public on
acceptance).

## License

Code: MIT (see [`LICENSE`](LICENSE)). The manuscripts in `paper_a/` and
`paper_b/` are the author's preprints: copyright is retained by the author
until journal publication, after which the journals' copyright terms apply.
