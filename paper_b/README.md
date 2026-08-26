# Paper B — REDEM: Online Learning with Meta-Adaptation and Structural Plasticity for Non-Stationary Environments

Paper B of the four-paper REDEM series (**role: algorithm**). It builds the
training-inference-unified online learning architecture on Paper A's
substrate: a per-pulse RLS readout (M1), a slow metadata trace (M3),
functional-connectivity-guided rewiring (M4), and a chaos homeostat (M5) —
every learning signal local and derived from live inference.

- PDF: [`PAPER_B.pdf`](PAPER_B.pdf) | LaTeX: [`PAPER_B.tex`](PAPER_B.tex)
- Preprint: [doi:10.5281/zenodo.22110607](https://doi.org/10.5281/zenodo.22110607)
- Series overview and reading order (A → B → C → D):
  [`../README.md`](../README.md)

## Key results

- Full system 0.996 vs. bare baseline 0.973 (p < 0.0001); replicated at
  N = 1024 (0.9970 vs. 0.9753, paired t = 15.3).
- Tracks an abrupt class-interval inversion within 225–616 pulses, while
  frozen batch learners (GRU, tiny transformer) become systematically wrong
  and never recover.
- Dual-timescale metadata speeds regime adaptation by a controlled-measured
  factor of 1.3–2.4; the λ-homeostat restores 8–18% of post-disturbance
  memory (up to +25% at the edge of chaos, +32% after three sequential
  disturbances).
- Honest baseline position: competitive with a well-tuned ESN on standard
  tasks (drift accuracy 0.991 vs. 1.000; Mackey–Glass NMSE 0.0018 vs.
  3.6×10⁻⁵); the differentiated value is the mechanism set, not raw
  benchmark supremacy.

## Contents

| File | Purpose |
|---|---|
| `PAPER_B.tex` | LaTeX source (compiles standalone with the `article` class) |
| `PAPER_B.pdf` | Compiled PDF (11 pages) |

## Compile

From this directory (figures resolve via `../figures/`):

```powershell
pdflatex PAPER_B.tex    # run twice for cross-references
```

## Figures and data anchors

| Item | File | Data |
|---|---|---|
| Fig. 1 REDEM schematic | `../figures/paperB_fig1_redem.pdf` | — |
| Fig. 2 online vs. offline | `../figures/s2_online_readout_v1.pdf` | `../data/s2_online_readout_v1.*` |
| Fig. 3 metadata mechanism | `../figures/paperB_fig3_metadata.pdf` | `../data/s5_dual_timescale_v1.*` |
| Fig. 4 robustness | `../figures/paperA_fig4_robustness.pdf` (shared with Paper A) | `../data/s6_chaos_regulator_v1.*` |
| Fig. 5 ablation | `../figures/paperB_fig5_ablation.pdf` | `../data/s3_three_factor_v1.*`, `../data/s4_intrinsic_reward_v1.*` |
| Fig. 6 showdown | `../figures/paperB_fig6_showdown.pdf` | `../data/s10_esn_metadata_v1.*` |
| controlled adaptation | — | `../data/s5b_controlled_adaptation_v1.*` |
| disturbance chain | — | `../data/s11_disturbance_chain_v1.*` |
| causal audits | — | `../data/s13_causal_audit_v1.*`, `../data/s28_causal_audit_chain_v1.*`, `../data/s34_leak_sensitivity_v1.*` |
| M4–M5 coupling / gated plasticity | — | `../data/s24_homeo_plasticity_coupling_v1.*`, `../data/s25_reward_gated_plasticity_v1.*` |
| N=1024 replication | — | `../data/s30_integrated_1024_v1.*` |

## Reproduce

CPU-only; uses the project virtual environment (`../.venv`).

```powershell
& ..\.venv\Scripts\python.exe ..\scripts\online_readout_streaming.py
& ..\.venv\Scripts\python.exe ..\scripts\three_factor_online_readout.py
& ..\.venv\Scripts\python.exe ..\scripts\intrinsic_reward_experiment.py
& ..\.venv\Scripts\python.exe ..\scripts\dual_timescale_metadata.py
& ..\.venv\Scripts\python.exe ..\scripts\chaos_regulator.py
& ..\.venv\Scripts\python.exe ..\scripts\structure_plasticity.py
& ..\.venv\Scripts\python.exe ..\scripts\integrated_benchmark.py
& ..\.venv\Scripts\python.exe ..\scripts\baseline_showdown.py
& ..\.venv\Scripts\python.exe ..\scripts\esn_metadata_comparison.py
& ..\.venv\Scripts\python.exe ..\scripts\s5b_controlled_adaptation.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s24_homeo_plasticity_coupling.py
& ..\.venv\Scripts\python.exe ..\scripts\s25_reward_gated_plasticity.py
& ..\.venv\Scripts\python.exe ..\scripts\s28_causal_audit_chain.py
& ..\.venv\Scripts\python.exe ..\scripts\s30_integrated_1024.py --workers 4
& ..\.venv\Scripts\python.exe ..\scripts\s34_leak_sensitivity.py
& ..\.venv\Scripts\python.exe ..\scripts\gen_architecture_schematic.py
& ..\.venv\Scripts\python.exe ..\scripts\gen_s2_curves.py
& ..\.venv\Scripts\python.exe ..\scripts\gen_paper_figures.py
```

Every script accepts `--quick` for a smoke run. Full pipeline and seed
discipline: [`../README_REDEM.md`](../README_REDEM.md).

## Companion papers

- Paper A (physics): [`../paper_a/`](../paper_a/) — substrate characterization
- Paper C (dissection): [`../paper_c/`](../paper_c/) — mechanism disentanglement
- Paper D (architecture): [`../paper_d/`](../paper_d/) — state-space-native REDEM
