# REDEM — Physics-Grounded Online Learning Architecture

This repository contains the complete code, data, and figures for three
companion papers (plus a fourth in development):

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

- **Paper C (in development)**: *"Dissecting Online Learning Mechanisms:
  Statistical Memory, Homeostatic Recovery, and Substrate Physics are
  Non-Transferable"* — three-mechanism disentanglement built on a falsifying
  experiment (s14: metadata does not transfer disturbance robustness to an
  ESN; s16: robust across τ_m ∈ [200, 2000]); §6 LLM extension written
  (s18: routing transfers, gating-only falsified, Fig 3); §7 conclusion
  point (6): host-boundary statement (the full framework needs a stateful
  host); drafted for Neurocomputing / Neural Networks short paper
  → See [`paper_c/PAPER_C.pdf`](paper_c/PAPER_C.pdf),
  [`paper_c/PAPER_C.tex`](paper_c/PAPER_C.tex),
  [`paper_c/DERIVATION.md`](paper_c/DERIVATION.md) and
  [`paper_c/PAPER_C_sketch.md`](paper_c/PAPER_C_sketch.md)

- **Paper D (in development)**: *"REDEM-SSM: A Foundation Model Architecture
  with Native Online Learning, Meta-Adaptation, and Structural
  Plasticity"* — a native SSM/Mamba-hosted foundation model instantiating
  M1/M3/M4/M5 from the ground up, motivated by Paper C's host-boundary
  result. P1–P4 (S19–S22) DONE: P1/P3a falsified with mechanisms isolated
  (linear state mixing; fixed gates don't fix it; M1 works on the additive
  input path, 10/10); P2 M3 routing transfers (forgetting −2.05, 10/10)
  while gating-only inverts on RLS readouts (readout-dynamics-dependent);
  P3 both supported (soft routing beats abrupt −1.82, 10/10; dormant
  covariance refresh flips routing to −1.72; M5 homeostat bounds the state
  and restores the full-state EMA detector 5/5); P4 benchmark (4-domain
  irregular switches) both hypotheses 10/10 (REDEM vs bare −2.25 stream,
  −4.47 forget; vs TF-A1 −9.28/−10.08) + real-text transfer (two Gutenberg
  books, REDEM vs bare −1.27, vs TF-A1 −5.23, 10/10). P5: PAPER_D.tex
  draft (11 pp, zero warnings; author info filled). (Target: NeurIPS/ICML stretch;
  arXiv + workshop first)
  → See [`paper_d/README.md`](paper_d/README.md),
  [`paper_d/PAPER_D_sketch.md`](paper_d/PAPER_D_sketch.md) and
  [`paper_d/PAPER_D.tex`](paper_d/PAPER_D.tex)

**Two papers, one pipeline — two different stories.** Paper A is a physics /
nonlinear-dynamics theory paper about what the substrate *computes*; Paper B
is a machine-learning paper about how to *learn on* it. They share the same
simulation code and data but ask different questions:

| | Paper A — substrate characterization | Paper B — REDEM online learning |
|---|---|---|
| **Question** | What can this physical substrate compute? | How do you learn on top of it? |
| **Content** | Dynamics theory: memory–chaos phase diagram, forgetting kernel, λ-homeostat robustness (full derivations in Appendix A) | Learning algorithm + benchmarks: online RLS readout, dual-timescale metadata, chaos homeostat, structure plasticity, ablations |
| **Key results** | Order–chaos transition at κ*∈(25,30); held-out memory +24–53% just before it; forgetting kernel M(t)=∫p(τ)e^{−t/τ}dτ matches the measured memory curve (r=0.97); λ-homeostat restores 8–18% after disturbances; edge of chaos (λ_target=0) identified as optimal target (+25%) | Tracks drift where frozen batch learners (GRU, transformer) fail permanently; integrated system matches or beats every ablation and the bare baseline (0.996 vs 0.988/0.994/0.973; N=1024: 0.998 vs 0.976); metadata is substrate-agnostic and transfers to a matched ESN; +32% memory after three sequential disturbances; causal audit confirms all mechanisms are causally clean |
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
├── paper_c/     Paper C (in development): three-mechanism disentanglement (derivation, sketch, README)
├── paper_d/     Paper D (sketch): native REDEM-SSM foundation model (README, design doc)
├── scripts/     Shared simulation code (CORE substrate, tasks, readouts, figure scripts)
├── data/        All experiment results (CSV + JSON, 10-seed means)
├── figures/     All publication figures (vector PDF; no raster twins)
└── *.md         Overview, technical README, research plan, coding standards (CLAUDE.md)
```

Each paper folder contains its own `README.md` with the submission checklist
(author placeholder, journal template swap, figures/tables inventory, cover
letter points).

## Scripts

All 43 committed scripts in `scripts/`, typed per `CLAUDE.md`
(ML > CORE > PAPER > FIG > EXPLORE). The **Paper** column marks which paper
each script serves (A / B / C / D; shared = library or figure used by more
than one paper). Two legacy scripts from the prior Si₃N₄-pulse-encoding
project are kept as shared dependencies (imported by the new code) and are
never modified. `README_REDEM.md` holds the full headline-results registry
(S1–s18) and reproduction commands.

| Script | Type | Paper | Purpose | Key result |
|---|---|---|---|---|
| `recurrent_substrate.py` | CORE | shared | per-pulse contrast-coupled relaxation substrate (numba core); self-test 4/4 | — |
| `shallow_trap_array_simulator.py` | CORE (legacy) | shared | Si₃N₄ shallow-trap device simulator; constants γ/τ₀/gen_tau_vec/preprogram_vec imported by 12 scripts | — |
| `fair_esn_comparison.py` | ML (legacy) | B | matched ESN reservoir class; imported by `baseline_showdown.py` and `esn_metadata_comparison.py` | — |
| `streaming_tasks.py` | CORE | shared | task generators: drift_binary, narma10, mackey_glass, context_switch, regime_switch | — |
| `online_readout.py` | CORE | shared | OnlineRLS, ThreeFactorReadout, ridge_fit, MC/accuracy metrics | — |
| `substrate_recurrence_characterization.py` | PAPER | A | S1: FTLE / held-out MC / separation vs κ sweep (610 runs) | κ* ∈ (25,30); held-out MC +24–53% at the edge |
| `online_readout_streaming.py` | PAPER | B | S2: online RLS vs offline ridge on streaming tasks | acc 0.974–0.982, recover 225–616; MG NMSE 0.0018 |
| `three_factor_online_readout.py` | PAPER | B | S3: reward-modulated Hebbian vs error-gated vs RLS (negative result) | reward-only post-inversion 0.06–0.10 |
| `intrinsic_reward_experiment.py` | PAPER | B | S4: novelty intrinsic reward ablation (negative result) | stream mean ≤ 0.514; never rescues |
| `dual_timescale_metadata.py` | PAPER | B | S5: fast/dual/slow metadata on regime-switch | +1.3–2.1 pp (p < 0.0001) |
| `chaos_regulator.py` | PAPER | B | S6: λ-homeostat under disturbances | held-out MC +8–18% |
| `structure_plasticity.py` | PAPER | B | S7: correlation-guided rewiring | gentle (5%) +8–11%; aggressive (20%) −23% |
| `integrated_benchmark.py` | PAPER | B | S8: full system vs ablations; N=1024 confirmation | 0.996 vs 0.973 (p<0.0001); N=1024 0.998 vs 0.976 |
| `baseline_showdown.py` | ML | B | S9: vs matched ESN / GRU / tiny transformer (torch CPU) | REDEM 0.991 vs GRU 0.371 / TF 0.351 (ESN 0.998, honest) |
| `forgetting_curve_theory.py` | EXPLORE | A | S10: forgetting-kernel theory M(t), Gauss–Hermite validation | measured MC follows M(t), r = 0.97 |
| `esn_metadata_comparison.py` | PAPER | B | S10: metadata transfer to a matched ESN | equalizes: ESN+meta 0.998 ≈ ESN 0.996 ≈ REDEM 0.994 |
| `cv_sweep.py` | PAPER | A | S10: task-level CV sweep (Paper A Supp. Note 1) | uncoupled MC +33% with CV; coupled near-critical falls |
| `s11_disturbance_chain.py` | PAPER | A | E3: sequential disturbance chain (3 rounds, 10 seeds) | regulated MC 8.47 vs fixed 6.41 (+32%) |
| `s12_lambda_target_sweep.py` | PAPER | A | E4: λ_target × CV optimization sweep (5 seeds) | λ_target=0 optimal: +25%/+19%/+5% MC |
| `s13_causal_audit.py` | PAPER | B | O4: causal leakage audit (7 arms, 3 seeds) | all mechanisms causally clean (<0.02 pp) |
| `s14_esn_disturbance_chain.py` | PAPER | C | ESN+metadata under the disturbance chain — falsifies metadata robustness transfer (3 arms, 10 seeds) | paired diffs −0.78/−0.76/−0.69, 0/10 seeds positive |
| `s16_tau_m_pressure_test.py` | PAPER | C | τ_m ∈ {200,500,1000,2000} stress test — falsification robust across metadata timescales (10 seeds) | 0/10 seeds positive at every τ_m (~5σ) |
| `s15_controlled_adaptation.py` | PAPER | C | controlled adaptation protocol with known switch instants (10 seeds) | true effect ~10 pulses + variance collapse (p90 76.5→42) |
| `s16b_falsification_stress_test.py` | PAPER | C | probe-protocol stress test (V0/V1/V2, 2 τ_m, 10 seeds) | sign robust 0/10; magnitude V0 −0.69 → V2 −0.01 |
| `s5b_controlled_adaptation.py` | PAPER | B | S5 three-arm controlled adaptation re-measurement (2 substrates, 10 seeds) | T200 factor 1.28–1.44, T40 1.79–2.40 (Paper B §4.3 revised) |
| `s17_substrate_stress.py` | PAPER | C | ESN substrate stress — equalizer gain at all configs (120 runs) | gain positive at all 6 ESN configs (+0.1 to +1.0 pp) |
| `s18_llm_drift_gate.py` | ML | C | §7 PoC: LLM drift gate (tiny transformer + LoRA, 90 runs, torch CPU) | A3 routing transfers −2.5 ppl (10/10); A2 gate falsified (0/10) |
| `s19_ssm_rls_readout.py` | ML | D | P1/P3a: per-token RLS readout on a hand-rolled diagonal SSM (7 arms, 70 runs, torch CPU) | state arms 58.5–115.9 (0/10) vs B-proj 11.75 (d −3.26, 10/10); oracle 7.25 |
| `s20_ssm_m3_routing.py` | ML | D | P2: M3 EMA metadata + drift detection + routing on the SSM host (A1/A2/A3, 90 runs, torch CPU) | A2 stream −1.52…−2.45 (10/10); A3 forget −2.05/−1.87/−1.20 (10/10, τ_m≤1000) |
| `s21_ssm_m4_m5.py` | ML | D | P3: M4 soft vs abrupt routing + M5 state-norm homeostat (E1/E2, 70 runs, torch CPU) | soft 8.22 vs abrupt 10.03 (−1.82, 10/10); M5 restores EMA detector 5/5, norm 11.3 vs 50.2 |
| `s22_ssm_p4_benchmark.py` | ML | D | P4: 4-domain irregular-switch benchmark (SSM-bare / SSM-REDEM / TF-A1, 30 runs, torch CPU) | REDEM vs bare −2.25/−4.47, vs TF −9.28/−10.08 (10/10) |
| `s23_ssm_p4_realtext.py` | ML | D | real-text benchmark: two Gutenberg books (Alice vs Dickens), 32-symbol char vocab (30 runs, torch CPU) | REDEM vs bare −1.27 (10/10), vs TF −5.23 (10/10) |
| `gen_paperD_fig1_p1_arms.py` | FIG | D | Paper D Fig. 1: P1/P3a state-readout falsification + input-path control | `figures/paperD_fig1_p1_arms.pdf` |
| `gen_paperD_fig2_routing.py` | FIG | D | Paper D Fig. 2: P2 routing retention + P3 soft vs abrupt | `figures/paperD_fig2_routing.pdf` |
| `gen_paperD_fig3_benchmark.py` | FIG | D | Paper D Fig. 3: P4 benchmark bars | `figures/paperD_fig3_benchmark.pdf` |
| `gen_architecture_schematic.py` | FIG | shared | Paper Fig. 1 schematics (substrate / REDEM, M4↔M5 loop) | `figures/paperA_fig1_substrate.pdf`, `figures/paperB_fig1_redem.pdf` |
| `gen_substrate_phase_diagram.py` | FIG | A | S1 phase-diagram figure | `figures/substrate_phase_diagram_v2.pdf` |
| `gen_s2_curves.py` | FIG | B | S2 learning curves | `figures/s2_online_readout_v1.pdf` |
| `gen_paper_figures.py` | FIG | shared | batch: robustness / metadata / ablation / showdown | `figures/paperA_fig4_robustness.pdf`, `figures/paperB_fig3_metadata.pdf`, `figures/paperB_fig5_ablation.pdf`, `figures/paperB_fig6_showdown.pdf` |
| `gen_paperA_supp_figures.py` | FIG | A | Paper A Supplementary Fig. S1 (CV sweep) | `figures/paperA_figS1_cv_sweep.pdf` |
| `gen_paperC_fig1_kernel.py` | FIG | C | Paper C Fig. 1: slow-trace kernel vs material forgetting kernel | `figures/paperC_fig1_kernel.pdf` |
| `gen_paperC_fig2_recovery.py` | FIG | C | Paper C Fig. 2: post-disturbance MC recovery vs τ_m (s16) | `figures/paperC_fig2_recovery.pdf` |
| `gen_paperC_fig3_llm.py` | FIG | C | Paper C Fig. 3: LLM drift-gate results (s18) | `figures/paperC_fig3_llm.pdf` |

Each FIG script emits a single vector `.pdf` (journal submission); the papers
include the extension-less basename so `pdflatex` picks the vector file
automatically. Reproduction commands are in `README_REDEM.md` (S1–s18) and
`paper_d/README.md` (s19–s23).

## Code availability

All simulation code and data required to reproduce the results are available
at <https://github.com/huyamingc/REDEM> (private during review; made public on
acceptance).

## License

Code: MIT (see [`LICENSE`](LICENSE)). The manuscripts in `paper_a/` and
`paper_b/` are the author's preprints: copyright is retained by the author
until journal publication, after which the journals' copyright terms apply.
