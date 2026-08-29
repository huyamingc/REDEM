# Paper B — REDEM: Online Learning with Meta-Adaptation and Structural Plasticity for Non-Stationary Environments

Paper B of the four-paper REDEM series (**role: algorithm**). It builds the
training-inference-unified online learning architecture on Paper A's
substrate: a per-pulse RLS readout (M1), a slow metadata trace (M3),
functional-connectivity-guided rewiring (M4), and a chaos homeostat (M5) —
every learning signal local and derived from live inference.

- PDF: [`PAPER_B.pdf`](PAPER_B.pdf) | LaTeX: [`PAPER_B.tex`](PAPER_B.tex)
- Preprint: [doi:10.5281/zenodo.22110606](https://doi.org/10.5281/zenodo.22110606)
- Target: *Neural Networks* (Elsevier)
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

CPU-only; uses the project virtual environment (`../.venv`). Run from this
directory:

```powershell
& ..\.venv\Scripts\python.exe ..\scripts\<script> [flags]
```

Experiment scripts accept `--quick` (reduced smoke run); `--sequential`
(where listed) disables the multiprocessing `Pool`, `--workers N` caps it.
Each run regenerates the committed `../data/` files. The table below is the
experiment index: what each script does, what it writes, and the finding it
establishes (with the paper location).

| Script | What it does | Writes | Finding (paper anchor) |
|---|---|---|---|
| `online_readout_streaming.py` | S2: online RLS readout vs. offline ridge on streaming tasks (drift / NARMA10 / Mackey–Glass) | `../data/s2_online_readout_v1.{csv,json}` | Acc 0.974–0.982 online; abrupt-inversion recovery within 225–616 pulses; MG NMSE 0.0018 — online learning matches offline where offline works and survives where it cannot (Fig. 2, Table 2) |
| `three_factor_online_readout.py` | S3: reward-modulated Hebbian vs. error-gated vs. RLS readouts on the inversion task | `../data/s3_three_factor_v1.{csv,json}` | Reward-only readouts collapse post-inversion (0.06–0.10) — an explicit error signal is necessary (Table 1) |
| `intrinsic_reward_experiment.py` | S4: novelty intrinsic-reward ablation on top of the reward-only readout | `../data/s4_intrinsic_reward_v1.{csv,json}` | Stream mean ≤ 0.514, never rescues the inversion — intrinsic reward is not a substitute for error (Table 1) |
| `dual_timescale_metadata.py` | S5: fast / dual / slow metadata traces on regime-switching tasks | `../data/s5_dual_timescale_v1.{csv,json}` | Dual-timescale metadata +1.3–2.1 pp (p < 0.0001) — a slow statistical trace extends the readout's horizon (Fig. 3) |
| `chaos_regulator.py` | S6: λ-homeostat under temperature drift, edge damage, readout noise | `../data/s6_chaos_regulator_v1.{csv,json}` | Held-out MC +8–18% post-disturbance vs. fixed κ (Fig. 4) |
| `structure_plasticity.py` | S7: correlation-guided functional rewiring at gentle vs. aggressive churn | `../data/s7_structure_plasticity_v1.{csv,json}` | Gentle 5% churn +7.8% MC (ring) and +11.3% prune repair; aggressive 20% churn −23% — structure must change slowly |
| `integrated_benchmark.py` | S8: full system vs. single-mechanism ablations vs. bare baseline | `../data/s8_integrated_v1.{csv,json}` | Full system 0.996 vs. baseline 0.973 (p < 0.0001) — mechanisms compose rather than compete (Table 1) |
| `baseline_showdown.py` | S9: matched ESN / GRU / tiny transformer baselines (torch CPU) | `../data/s9_baseline_showdown_v1.{csv,json}` | REDEM 0.991 vs. ESN 1.000 (honest) / GRU 0.394 / transformer 0.351 after drift — frozen batch learners never recover (Table 2, Fig. 6) |
| `esn_metadata_comparison.py` | S10: transfer the slow metadata trace to a matched ESN | `../data/s10_esn_metadata_v1.{csv,json}` | ESN+meta 0.998 ≈ ESN 0.996 ≈ REDEM 0.994 — the statistical-memory benefit is substrate-agnostic (Fig. 6) |
| `s11_disturbance_chain.py` | E3: three sequential disturbances (τ-drift → edge prune → readout noise), 10 seeds | `../data/s11_disturbance_chain_v1.{csv,json}` | Regulated r3 MC 8.47 vs. fixed 6.41 (+32%) — the homeostat's sequential-recovery anchor |
| `s13_causal_audit.py` | O4: causal leakage audit — leak each mechanism's signals into the arms it must not reach (7 arms, 10 seeds) | `../data/s13_causal_audit_v1.{csv,json}` | All leak arms ≤ 0.02 pp (0.9964–0.9962 vs. 0.9963) — mechanisms are causally clean |
| `s5b_controlled_adaptation.py --sequential` | S5 three-arm controlled adaptation re-measurement with known switch instants (2 substrates, 10 seeds) | `../data/s5b_controlled_adaptation_v1.{csv,json}` | Controlled adaptation factor 1.28–1.44 (T200), 1.79–2.40 (T40) — the honest, switch-relative version of the speedup claim |
| `s24_homeo_plasticity_coupling.py` | M4–M5 coupling loop under the S11 disturbance chain (4 arms, 10 seeds) | `../data/s24_homeo_plasticity_coupling_v1.{csv,json}` | Homeostat alone r3 MC 8.47 vs. coupled 5.27 (0/10) — rewiring during disturbance is harmful; the coupling loop stays a design note |
| `s25_reward_gated_plasticity.py` | Novelty-reward-gated vs. correlation-guided rewiring (4 arms, 10 seeds) | `../data/s25_reward_gated_plasticity_v1.{csv,json}` | Novelty-gated 14.59 vs. correlation-guided 12.43 (t = 4.5, 10/10) — intrinsic signals work as structure-level gates |
| `s28_causal_audit_chain.py` | Causal-leak audit re-run on the S11 disturbance chain (6 arms, 10 seeds) | `../data/s28_causal_audit_chain_v1.{csv,json}` | No leak arm improves recovery; the no-plasticity arm reproduces the 8.47 anchor exactly — causal cleanliness survives a non-ceilinged protocol |
| `s30_integrated_1024.py --workers 4` | N = 1024 integrated replication, baseline vs. full, paired (10 seeds) | `../data/s30_integrated_1024_v1.{csv,json}` | Full 0.9970 vs. baseline 0.9753 (paired t = 15.3, 10/10) — the composition claim holds at 4× scale |
| `s34_leak_sensitivity.py` | Leak-sensitivity scan on the S28 audit (6 leak configs, 10 seeds) | `../data/s34_leak_sensitivity_v1.{csv,json}` | 10× FTLE leak still NS (+0.21); 30% plasticity-correlation leak +0.57 (7/10, 95% CI includes 0) — the audit's resolution is bounded, claims scoped to operational leaks |
| `gen_architecture_schematic.py` | FIG: substrate and REDEM architecture schematics (M4↔M5 loop) | `../figures/paperA_fig1_substrate.pdf`, `../figures/paperB_fig1_redem.pdf` | Figs. 1 (shared with Paper A) |
| `gen_s2_curves.py` | FIG: S2 online-vs-offline learning curves | `../figures/s2_online_readout_v1.pdf` | Fig. 2 |
| `gen_paper_figures.py` | FIG: batch render robustness / metadata / ablation / showdown figures | `../figures/paperA_fig4_robustness.pdf`, `../figures/paperB_fig3_metadata.pdf`, `../figures/paperB_fig5_ablation.pdf`, `../figures/paperB_fig6_showdown.pdf` | Figs. 3–6 |

Seed discipline and the full S1–s35 pipeline:
[`../README_REDEM.md`](../README_REDEM.md).

## Companion papers

- Paper A (physics): [`../paper_a/`](../paper_a/) — substrate characterization
- Paper C (dissection): [`../paper_c/`](../paper_c/) — mechanism disentanglement
- Paper D (architecture): [`../paper_d/`](../paper_d/) — state-space-native REDEM
