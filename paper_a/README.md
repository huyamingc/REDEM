# Paper A — Memory and chaos in a physics-constrained relaxation substrate

Paper A of the four-paper REDEM series (**role: physics**). It characterizes
the substrate the other three papers build on: the memory–chaos phase diagram
across five coupling topologies, the analytic multi-timescale forgetting
kernel, and the λ-homeostat that restores memory after disturbances.

- PDF: [`PAPER_A.pdf`](PAPER_A.pdf) | LaTeX: [`PAPER_A.tex`](PAPER_A.tex)
- Preprint: [doi:10.5281/zenodo.22109664](https://doi.org/10.5281/zenodo.22109664)
- Target: *Chaos, Solitons & Fractals* (Elsevier)
- Series overview and reading order (A → B → C → D):
  [`../README.md`](../README.md)

## Key results

- Order–chaos transition at κ* ∈ (25, 30); held-out memory capacity peaks
  24–53% above the uncoupled baseline just before the transition, and deep
  chaos destroys memory.
- Linear-memory decay follows the analytic forgetting kernel
  M(t) = ∫ p(τ) e^(−t/τ) dτ over the log-normal trap spectrum
  (Pearson r = 0.97); the 1/e horizon is pinned near τ0/⟨Δt⟩ ≈ 16 pulses
  regardless of spectrum width.
- The λ-homeostat restores 8–18% of post-disturbance held-out memory under
  temperature drift, edge damage, and readout noise.

## Contents

| File | Purpose |
|---|---|
| `PAPER_A.tex` | LaTeX source (compiles standalone with the `article` class) |
| `PAPER_A.pdf` | Compiled PDF (14 pages) |

## Compile

From this directory (figures resolve via `../figures/`):

```powershell
pdflatex PAPER_A.tex    # run twice for cross-references
```

## Figures and data anchors

| Item | File | Data |
|---|---|---|
| Fig. 1 substrate | `../figures/paperA_fig1_substrate.pdf` | — |
| Fig. 2 phase diagram | `../figures/substrate_phase_diagram_v2.pdf` | `../data/substrate_phase_diagram_v2.*` |
| Fig. 3 forgetting kernel | `../figures/forgetting_curve_theory.pdf` | `../data/forgetting_curve_theory_overlay_v1.json` |
| Fig. 4 robustness | `../figures/paperA_fig4_robustness.pdf` | `../data/s6_chaos_regulator_v1.*` |
| Supp. Fig. S1 CV sweep | `../figures/paperA_figS1_cv_sweep.pdf` | `../data/s10_cv_sweep_v1.*` |
| Supp. Table S1 | — | `../data/substrate_phase_diagram_v2.csv` |
| §4.2 clip × fine-κ grid | — | `../data/s27_clip_kappa_fine_v1.*` |
| kernel shape under coupling | — | `../data/kernel_coupling_shape_v1.*` |
| FTLE-noise robustness | — | `../data/s32_ftle_noise_v1.*` |

## Reproduce

CPU-only; uses the project virtual environment (`../.venv`). Run from this
directory:

```powershell
& ..\.venv\Scripts\python.exe ..\scripts\<script> [flags]
```

Experiment scripts accept `--quick` (reduced smoke run); `--sequential`
(where listed) disables the multiprocessing `Pool`. Each run regenerates the committed `../data/` files. The table below is the
experiment index: what each script does, what it writes, and the finding it
establishes (with the paper location).

| Script | What it does | Writes | Finding (paper anchor) |
|---|---|---|---|
| `substrate_recurrence_characterization.py` | S1: sweep coupling κ across five topologies; per run measure FTLE λ, held-out Jaeger MC, input separation (610 runs) | `../data/substrate_phase_diagram_v2.{csv,json}` | Sharp order–chaos transition at κ\* ∈ (25, 30); held-out MC peaks +24–53% just before the transition; deep chaos destroys memory (Fig. 2, Table 2) |
| `forgetting_curve_theory.py` | S10: evaluate the analytic forgetting kernel M(t) = ∫p(τ)e^(−t/τ)dτ over the log-normal trap spectrum; Gauss–Hermite validation | `../data/forgetting_curve_theory_overlay_v1.json` | Measured MC follows M(t), Pearson r = 0.97; the 1/e horizon is pinned at ≈ 16 pulses regardless of spectrum width — width only controls the tail weight (Fig. 3) |
| `kernel_coupling_shape.py` | Kernel–coupling shape stability: physical-kernel Pearson r against √MC(k) per operating point | `../data/kernel_coupling_shape_v1.{csv,json}` | r = 0.91–0.99 over κ 15–30 for the three mode-1 topologies — the forgetting-kernel law survives coupling |
| `cv_sweep.py` | S10: task-level CV (spectrum width) sweep, uncoupled vs. near-critical coupled | `../data/s10_cv_sweep_v1.{csv,json}` | Uncoupled MC +33% with CV; coupled near-critical falls — width helps only without coupling (Supp. Note 1, Fig. S1) |
| `s12_lambda_target_sweep.py` | E4: homeostat set-point λ_target × CV optimization sweep (5 seeds) | `../data/s12_lambda_target_sweep_v1.{csv,json}` | λ_target = 0 (edge of chaos) is optimal: +25%/+19%/+5% MC |
| `s27_clip_kappa_fine.py` | Clip-range ablation × fine κ grid (3 topologies × 3 α_max × 13 κ × 10 seeds) | `../data/s27_clip_kappa_fine_v1.{csv,json}` | κ\* = 25.3/27.4/27.9; invariant to 5× clip widening for lateral_ring/random_graph — chaos is coupling-driven, not clip-driven (ring_bidir shifts +3.4) |
| `s32_ftle_noise_robustness.py` | Homeostat with Gaussian noise injected on every FTLE estimate (5 levels × 10 seeds) | `../data/s32_ftle_noise_v1.{csv,json}` | No significant MC degradation up to σ = 0.10 — clipped proportional feedback integrates out estimation noise |
| `gen_substrate_phase_diagram.py` | FIG: render the κ–λ phase diagram | `../figures/substrate_phase_diagram_v2.pdf` | Fig. 2 |
| `gen_paperA_supp_figures.py` | FIG: render the supplementary CV-sweep figure | `../figures/paperA_figS1_cv_sweep.pdf` | Fig. S1 |

Seed discipline and the full S1–s35 pipeline:
[`../README_REDEM.md`](../README_REDEM.md).

## Companion papers

- Paper B (algorithm): [`../paper_b/`](../paper_b/) — REDEM online learning
- Paper C (dissection): [`../paper_c/`](../paper_c/) — mechanism disentanglement
- Paper D (architecture): [`../paper_d/`](../paper_d/) — state-space-native REDEM
