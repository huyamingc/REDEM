# Paper A — Memory and chaos in a physics-constrained relaxation substrate

Paper A of the four-paper REDEM series (**role: physics**). It characterizes
the substrate the other three papers build on: the memory–chaos phase diagram
across five coupling topologies, the analytic multi-timescale forgetting
kernel, and the λ-homeostat that restores memory after disturbances.

- PDF: [`PAPER_A.pdf`](PAPER_A.pdf) | LaTeX: [`PAPER_A.tex`](PAPER_A.tex)
- Preprint: [doi:10.5281/zenodo.22109665](https://doi.org/10.5281/zenodo.22109665)
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

CPU-only; uses the project virtual environment (`../.venv`).

```powershell
& ..\.venv\Scripts\python.exe ..\scripts\substrate_recurrence_characterization.py
& ..\.venv\Scripts\python.exe ..\scripts\forgetting_curve_theory.py
& ..\.venv\Scripts\python.exe ..\scripts\kernel_coupling_shape.py
& ..\.venv\Scripts\python.exe ..\scripts\cv_sweep.py
& ..\.venv\Scripts\python.exe ..\scripts\s12_lambda_target_sweep.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s27_clip_kappa_fine.py
& ..\.venv\Scripts\python.exe ..\scripts\s32_ftle_noise_robustness.py
& ..\.venv\Scripts\python.exe ..\scripts\gen_substrate_phase_diagram.py
& ..\.venv\Scripts\python.exe ..\scripts\gen_paperA_supp_figures.py
```

Every script accepts `--quick` for a smoke run. Full pipeline and seed
discipline: [`../README_REDEM.md`](../README_REDEM.md).

## Companion papers

- Paper B (algorithm): [`../paper_b/`](../paper_b/) — REDEM online learning
- Paper C (dissection): [`../paper_c/`](../paper_c/) — mechanism disentanglement
- Paper D (architecture): [`../paper_d/`](../paper_d/) — state-space-native REDEM
