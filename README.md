# REDEM — Physics-Grounded Online Learning Architecture

This repository contains the complete code, data, and figures for three
companion papers (plus a fourth in development):

- **Paper A**: *"Memory and chaos in a physics-constrained relaxation
  substrate: phase diagram, multi-timescale forgetting, and disturbance
  robustness"* — substrate characterization (target: *International Journal of
  Bifurcation and Chaos* / *Chaos*)
  → See [`paper_a/PAPER_A.pdf`](paper_a/PAPER_A.pdf) and
  [`paper_a/PAPER_A.tex`](paper_a/PAPER_A.tex)

- **Paper B**: *"REDEM: Online Learning with Meta-Adaptation and Structural
  Plasticity for Non-Stationary Environments"*
  — the online learning architecture (target: *Neural Networks*)
  → See [`paper_b/PAPER_B.pdf`](paper_b/PAPER_B.pdf) and
  [`paper_b/PAPER_B.tex`](paper_b/PAPER_B.tex)

- **Paper C (in development)**: *"Dissecting Online Learning Mechanisms:
  Statistical Memory, Homeostatic Recovery, and Substrate Physics are
  Non-Substitutable"* — three-mechanism disentanglement built on a falsifying
  experiment (s14: metadata does not transfer disturbance robustness to an
  ESN; s16: robust across τ_m ∈ [200, 2000]); §6 LLM extension written
  (s18: routing transfers, gating-only falsified, Fig 3); §7 conclusion
  point (6): host-boundary statement (the full framework needs a stateful
  host); drafted for Neurocomputing / Neural Networks short paper
  → See [`paper_c/PAPER_C.pdf`](paper_c/PAPER_C.pdf),
  [`paper_c/PAPER_C.tex`](paper_c/PAPER_C.tex),
  [`paper_c/DERIVATION.md`](paper_c/DERIVATION.md) and
  [`paper_c/PAPER_C_sketch.md`](paper_c/PAPER_C_sketch.md)

- **Paper D (in development)**: *"REDEM-SSM: A State-Space Architecture with
  Native Online Learning, Meta-Adaptation, and Structural
  Plasticity"* — a native SSM-hosted architecture instantiating
  M1/M3/M4/M5 from the ground up, motivated by Paper C's host-boundary
  result. P1–P4 (S19–S22) DONE: P1/P3a falsified with mechanisms isolated
  (exact token recovery is a deconvolution impossibility (Prop. 1); the
  pooled state readout fails out of sample (ppl 58–116), 0/10, while M1 on
  the additive input path works 10/10 — boundary analysis (s35) scopes the
  failure: the current token IS linearly decodable from the state's fast
  channels (88.9–99.7% out-of-sample), the in-sample oracle is
  window-dependent and nested-violating, so the failure is a
  pooled-readout/metric property, not missing linear information); P2 M3
  routing transfers (forgetting −2.05, 10/10)
  while gating-only inverts on RLS readouts (readout-dynamics-dependent);
  P3 both supported (soft routing beats abrupt −1.81, 10/10; dormant
  covariance refresh flips the soft-routing stream result to −3.54; M5
  homeostat bounds the state
  and restores the full-state EMA detector 5/5); P4 benchmark (4-domain
  irregular switches) both hypotheses 10/10 (REDEM vs bare −2.25 stream,
  −4.47 forget; vs TF-A1 −9.28/−10.08) + real-text transfer (two Gutenberg
  books, REDEM vs bare −1.27, vs TF-A1 −5.23, 10/10). P5: PAPER_D.tex
  draft (14 pp, zero warnings; author info filled; P1 theorem, reference
  fairness, M5 negative and oracle integrated; P1 corollary revised with
  s35 boundary probes; related-work section added). (Target: NeurIPS/ICML
  stretch; arXiv + workshop first)
  → See [`paper_d/README.md`](paper_d/README.md),
  [`paper_d/PAPER_D_sketch.md`](paper_d/PAPER_D_sketch.md) and
  [`paper_d/PAPER_D.tex`](paper_d/PAPER_D.tex)

## Provenance: prior Si₃N₄ pulse-encoding paper

This repository continues the author's prior standalone paper, which serves
as the device-physics foundation ("Paper 0") of the A–D series:

> **Si₃N₄ Shallow-Trap Relaxation for Temporal Pattern Encoding: A
> Systematic Design-Space Analysis** — preprint, Zenodo
> [DOI 10.5281/zenodo.21753791](https://doi.org/10.5281/zenodo.21753791)
> (2026); target venue: *Neuromorphic Computing and Engineering*.

What the prior paper contributes to this repository:

- **Device calibration** — γ = ln 100, τ₀ = 174 µs (Eₐ = 0.55 eV,
  ν = 10¹³ s⁻¹, T = 300 K), α = 0.02, τ-spread CV = 0.20: the parameter
  set every Paper A–D simulation inherits, bit-for-bit.
- **Two legacy scripts** — `shallow_trap_array_simulator.py` and
  `fair_esn_comparison.py` are frozen from the prior project, imported by
  the current code, and never modified.
- **Scope split** — the prior paper studies a *parallel array of
  independent devices* (quasi-static block coupling, offline Ridge/MLP
  readout); Papers A–D study the *per-pulse coupled recurrent* substrate
  (online RLS readout, chaos homeostat). Paper A §1 states this gap
  explicitly and cites the prior work as `\cite{prior}`; no experimental
  results are shared between the two lines beyond the device model itself.

If the prior paper is accepted at *Neuromorphic Computing and Engineering*,
update the `\cite{prior}` entry in `paper_a/PAPER_A.tex` to the journal
version.

**Two papers, one pipeline — two different stories.** Paper A is a physics /
nonlinear-dynamics theory paper about what the substrate *computes*; Paper B
is a machine-learning paper about how to *learn on* it. They share the same
simulation code and data but ask different questions:

| | Paper A — substrate characterization | Paper B — REDEM online learning |
|---|---|---|
| **Question** | What can this physical substrate compute? | How do you learn on top of it? |
| **Content** | Dynamics theory: memory–chaos phase diagram, forgetting kernel, λ-homeostat robustness (full derivations in Appendix A) | Learning algorithm + benchmarks: online RLS readout, dual-timescale metadata, chaos homeostat, structure plasticity, ablations |
| **Key results** | Order–chaos transition at κ*∈(25,30); held-out memory +24–53% just before it; forgetting kernel M(t)=∫p(τ)e^{−t/τ}dτ matches the measured memory curve (r=0.97); λ-homeostat restores 8–18% after disturbances; edge of chaos (λ_target=0) identified as optimal target (+25%) | Tracks drift where frozen batch learners (GRU, transformer) fail permanently; integrated system matches or beats every ablation and the bare baseline (0.996 vs 0.988/0.994/0.973; N=1024, 10 seeds: 0.9970 vs 0.9753); metadata is substrate-agnostic — its statistical-memory benefit transfers to a matched ESN (robustness transfer falsified in Paper C); +32% memory after three sequential disturbances; causal audit confirms all mechanisms are causally clean |
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
├── paper_d/     Paper D (sketch): native REDEM-SSM architecture (README, design doc)
├── scripts/     Shared simulation code (CORE substrate, tasks, readouts, figure scripts)
├── data/        All experiment results (CSV + JSON, 10-seed means)
├── figures/     All publication figures (vector PDF; no raster twins)
└── *.md         Overview, technical README, research plan, coding standards (CLAUDE.md)
```

Each paper folder contains its own `README.md` with the submission checklist
(author placeholder, journal template swap, figures/tables inventory, cover
letter points).

## Scripts

All 55 committed scripts in `scripts/`, typed per `CLAUDE.md`
(ML > CORE > PAPER > FIG > EXPLORE). The **Paper** column marks which paper
each script serves (A / B / C / D; shared = library or figure used by more
than one paper). Two legacy scripts from the prior Si₃N₄-pulse-encoding
project are kept as shared dependencies (imported by the new code) and are
never modified. `README_REDEM.md` holds the full headline-results registry
(S1–s35) and reproduction commands.

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
| `integrated_benchmark.py` | PAPER | B | S8: full system vs ablations; N=1024 confirmation | 0.996 vs 0.973 (p<0.0001); N=1024 (s30, 10 seeds) 0.9970 vs 0.9753 |
| `baseline_showdown.py` | ML | B | S9: vs matched ESN / GRU / tiny transformer (torch CPU) | REDEM 0.991 vs GRU 0.394 / TF 0.351 (ESN 1.000, honest; z-scored features) |
| `forgetting_curve_theory.py` | EXPLORE | A | S10: forgetting-kernel theory M(t), Gauss–Hermite validation | measured MC follows M(t), r = 0.97 |
| `esn_metadata_comparison.py` | PAPER | B | S10: metadata transfer to a matched ESN | equalizes: ESN+meta 0.998 ≈ ESN 0.996 ≈ REDEM 0.994 |
| `cv_sweep.py` | PAPER | A | S10: task-level CV sweep (Paper A Supp. Note 1) | uncoupled MC +33% with CV; coupled near-critical falls |
| `s11_disturbance_chain.py` | PAPER | B, C | E3: sequential disturbance chain (3 rounds, 10 seeds; anchor for Paper C's ESN falsification) | regulated MC 8.47 vs fixed 6.41 (+32%) |
| `s12_lambda_target_sweep.py` | PAPER | A | E4: λ_target × CV optimization sweep (5 seeds) | λ_target=0 optimal: +25%/+19%/+5% MC |
| `s13_causal_audit.py` | PAPER | B | O4: causal leakage audit (7 arms, 10 seeds) | all mechanisms causally clean (≤0.02 pp; plasticity-correlation leak implemented, −0.01 pp) |
| `s14_esn_disturbance_chain.py` | PAPER | C | ESN+metadata under the disturbance chain — falsifies metadata robustness transfer (3 arms, 10 seeds) | paired diffs −0.78/−0.76/−0.69, 0/10 seeds positive |
| `s16_tau_m_pressure_test.py` | PAPER | C | τ_m ∈ {200,500,1000,2000} stress test — falsification robust across metadata timescales (10 seeds) | 0/10 seeds positive at every τ_m (~5σ) |
| `s15_controlled_adaptation.py` | PAPER | C | controlled adaptation protocol with known switch instants (10 seeds) | true effect ~10 pulses + variance collapse (p90 76.5→42) |
| `s16b_falsification_stress_test.py` | PAPER | C | probe-protocol stress test (V0/V1/V2, 4 τ_m, 10 seeds; v2 output) | sign robust ≤1/10 everywhere; magnitude V0 −0.69 → V2 −0.04/−0.01 across τ_m |
| `s5b_controlled_adaptation.py` | PAPER | B | S5 three-arm controlled adaptation re-measurement (2 substrates, 10 seeds) | T200 factor 1.28–1.44, T40 1.79–2.40 (Paper B §4.3 revised) |
| `s17_substrate_stress.py` | PAPER | C | ESN substrate stress — equalizer gain at all configs (120 runs) | gain positive at all 6 ESN configs (+0.1 to +1.0 pp) |
| `s18_llm_drift_gate.py` | ML | C | §7 PoC: LLM drift gate (tiny transformer + LoRA, 90 runs, torch CPU) | A3 routing transfers −2.5 ppl (10/10); A2 gate falsified (0/10) |
| `s19_ssm_rls_readout.py` | ML | D | P1/P3a: per-token RLS readout on a hand-rolled diagonal SSM (7 arms, 70 runs, torch CPU) | state arms 58.5–115.9 (0/10) vs B-proj 11.75 (d −3.26, 10/10); oracle 7.25 |
| `s20_ssm_m3_routing.py` | ML | D | P2: M3 EMA metadata + drift detection + routing on the SSM host (A1/A2/A3, 90 runs, torch CPU) | A2 stream −1.52…−2.45 (10/10); A3 forget −2.05/−1.87/−1.20 (10/10, τ_m≤1000) |
| `s21_ssm_m4_m5.py` | ML | D | P3: M4 soft vs abrupt routing + M5 state-norm homeostat (E1/E2, 70 runs, torch CPU) | soft 8.22 vs abrupt 10.03 (−1.82, 10/10); M5 restores EMA detector 5/5, norm 11.3 vs 50.2 |
| `s22_ssm_p4_benchmark.py` | ML | D | P4: 4-domain irregular-switch benchmark (SSM-bare / SSM-REDEM / TF-A1, 30 runs, torch CPU) | REDEM-SSM vs bare −2.25/−4.47, vs TF −9.28/−10.08 (10/10) |
| `s23_ssm_p4_realtext.py` | ML | D | real-text benchmark: two Gutenberg books (Alice vs Dickens), 32-symbol char vocab (30 runs, torch CPU) | REDEM-SSM vs bare −1.27 (10/10), vs TF −5.23 (10/10) |
| `s24_homeo_plasticity_coupling.py` | PAPER | B | M4-M5 coupling loop under the S11 disturbance chain (4 arms, 10 seeds) | coupling does not help: homeostat alone r3 MC 8.47 vs +fixed-churn 6.45 (t=−9.6, 0/10) / +coupled 5.27 (0/10) — rewiring during disturbance is harmful |
| `s25_reward_gated_plasticity.py` | PAPER | B | novelty-reward-gated vs correlation-guided rewiring (4 arms, 10 seeds, S7 protocol) | novelty-guided MC 14.59 vs corr 12.43 (+2.15, t=4.5, 10/10) — intrinsic signals are structure-level tools |
| `s26_ssm_p4_fair_tf.py` | ML | D | fair Transformer references for P4: tuned A1 grid (lr×rank) + 4-adapter A3 routing (9 arms, 10 seeds) | tuning cuts the stream gap to −1.68 (0/10) but collapses forgetting to 62.2; TF-A3 routing retains specialists (−8.17 forgetting, 10/10) yet stream stays 21.77 — mechanisms transfer, the host does not |
| `s27_clip_kappa_fine.py` | PAPER | A | clip-range ablation × fine κ grid (3 topologies × 3 α_max × 13 κ × 10 seeds) | κ* = 25.3/27.4/27.9; invariant to 5× clip widening for lateral_ring/random_graph (coupling-driven chaos), ring_bidir shifts +3.4 (clip contributes) |
| `kernel_coupling_shape.py` | PAPER | A | kernel-coupling shape stability: physical-kernel Pearson r against √MC(k) per operating point (from `substrate_phase_diagram_v2.json`) | r_phys = 0.91–0.99 over κ 15–30 of the three mode-1 topologies (0.98 at random_graph κ=25); effective-median dilation reported qualitatively |
| `s28_causal_audit_chain.py` | PAPER | B | causal leak audit re-run on the S11 disturbance chain (6 arms, 10 seeds) | no leak arm improves recovery (r3 MC deltas +0.00/+0.00/+0.28/+0.24, NS); no-plasticity r3 MC 8.47 = S11 anchor exactly — causal cleanliness survives a non-ceilinged protocol |
| `s30_integrated_1024.py` | PAPER | B | N=1024 integrated replication at 10 seeds (baseline vs full, paired) | full 0.9970 vs baseline 0.9753 (+2.2 pp, paired t=15.3, 10/10) — the scale-up claim now supports a paired test |
| `s31_char_bigram_oracle.py` | PAPER | D | char-bigram oracle on the real-text protocol (full-book vs ref-window fits, 10 seeds) | true first-order ceiling ppl 10.97±0.18; REDEM-SSM 12.07 sits within ~1.1 ppl of it — the "first-order regime" boundary is now quantitative |
| `s32_ftle_noise_robustness.py` | PAPER | A | homeostat with Gaussian noise on every FTLE estimate (5 levels × 10 seeds, S11 chain) | no significant MC degradation up to σ=0.10 (≈100% of estimate scale); settled κ 28.5–28.7 — clipped proportional feedback integrates out estimation noise |
| `s33_ssm_p4_m5.py` | ML | D | M5 state-norm homeostat added to the P4 stack (2 arms, 10 seeds) | M5 is significantly worse (stream +1.53, forgetting +2.90, t=−22.8/−26.5, 10/10) — Δt modulation breaks the Δt=1-calibrated whitening; S22's exclusion of M5 validated |
| `s34_leak_sensitivity.py` | PAPER | B | leak sensitivity scan on the S28 audit (6 leak configs, 10 seeds) | 10× FTLE leak still NS (+0.21); plasticity 30% future correlation +0.57 mean (7/10; paired t=2.04, 95% CI [−0.06,+1.20]) — audit resolution is bounded, "causally clean" scoped to operational leaks |
| `s35_readout_boundary_probe.py` | PAPER | D | P1 readout boundary probes (10 seeds, s19 host verbatim): full/half-window oracle, skip-vs-proj nested check, fast/slow token decoding, fast-channel next-token readouts vs static-table reference | full-window oracle 31.2 (matches s19, max diff 0.00) vs half-window 17.3 — oracle is window-dependent; skip 18.0 > proj 7.25 (nested violation, 10/10); token linearly decodable from fast channels (τ≤8) at 88.9–99.7% out-of-sample (chance 3.1%), slow channels at chance; yet fast-channel-only direct and two-stage next-token readouts still fail out-of-sample (ppl 68–104 vs static table 13.9–17.4) — "no useful linear map" is false, and no closed-form squared-loss state readout yields calibrated next-token probabilities; the P1 failure is a pooled-readout/metric property, not missing linear information |
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
