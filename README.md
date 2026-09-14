# REDEM — Physics-Grounded Online Learning Architecture

This repository contains the complete code, data, and figures for
**six companion preprints** that form one research program on online
learning in physics-constrained systems. The papers share a single
simulation pipeline and data set, cite each other as companions, and are
designed to be read in order (**A → B → C → D**, then the standalone
self-evolution extension **E** and the readout-form companion **F**):

| Paper | Role | Core question | Preprint |
| :--- | :--- | :--- | :--- |
| **A** | Physics | What does the substrate compute? (memory–chaos phase diagram, forgetting kernel, homeostat) | Zenodo [10.5281/zenodo.22109664](https://doi.org/10.5281/zenodo.22109664) |
| **B** | Algorithm | How do you learn on top of it? (REDEM: RLS readout, meta-adaptation, structural plasticity) | Zenodo [10.5281/zenodo.22110606](https://doi.org/10.5281/zenodo.22110606) |
| **C** | Dissection | Which mechanism does which job? (statistical memory ≠ robustness recovery) | Zenodo [10.5281/zenodo.22110618](https://doi.org/10.5281/zenodo.22110618) |
| **D** | Architecture | What host makes these mechanisms native? (state-space-native REDEM) | Zenodo [10.5281/zenodo.22110623](https://doi.org/10.5281/zenodo.22110623) |
| **E** | Self-evolution | Can the substrate detect and repair its own failures under ±1 reward? (sign-flip correction, reconstructed-label capacity sensing, snapshot-gated memory) | — (submission, target: *Neurocomputing*) |
| **F** | Readout form | Which selectivity pressure does a calibrated (clip-free) readout still need? (trained softmax, top-k, expert routing) | — (draft, target: *Neurocomputing* / TMLR) |

In one sentence each: **Paper A** establishes the physics (what the
substrate computes); **Paper B** builds the learning architecture on that
physics; **Paper C** dissects the mechanisms by falsifying transfers and
locates the host boundary; **Paper D** redesigns the host so the mechanisms
are native rather than retrofitted, motivated by C's boundary result;
**Paper E** extends the program to autonomous self-correction — the
substrate detecting, repairing, and remembering its own failures from a
sign-only reward, self-contained in its own directory with frozen
dependencies; **Paper F** is D's companion on the readout form —
replacing the squared-loss / clip-floor instrument with a trained
softmax, then asking which sparse-selectivity pressure the clean metric
still needs.

## Papers

- **Paper A — Physics**: *"Memory and chaos in a physics-constrained
  relaxation substrate: phase diagram, multi-timescale forgetting, and
  disturbance robustness"* — substrate characterization (target:
  *Chaos, Solitons & Fractals*)
  → [`paper_a/PAPER_A.pdf`](paper_a/PAPER_A.pdf) |
  [`paper_a/PAPER_A.tex`](paper_a/PAPER_A.tex) |
  [`paper_a/README.md`](paper_a/README.md) |
  Zenodo [10.5281/zenodo.22109664](https://doi.org/10.5281/zenodo.22109664)

- **Paper B — Algorithm**: *"REDEM: Online Learning with Meta-Adaptation and
  Structural Plasticity for Non-Stationary Environments"*
  — the online learning architecture (target: *Neural Networks*)
  → [`paper_b/PAPER_B.pdf`](paper_b/PAPER_B.pdf) |
  [`paper_b/PAPER_B.tex`](paper_b/PAPER_B.tex) |
  [`paper_b/README.md`](paper_b/README.md) |
  Zenodo [10.5281/zenodo.22110606](https://doi.org/10.5281/zenodo.22110606)

- **Paper C — Dissection**: *"Dissecting Online Learning Mechanisms:
  Statistical Memory, Homeostatic Recovery, and Substrate Physics are
  Non-Substitutable in the Directions and Conditions Tested"* — three-mechanism disentanglement built on a
  falsifying transfer experiment (target: *Neural Networks*, same journal as Paper B)
  → [`paper_c/PAPER_C.pdf`](paper_c/PAPER_C.pdf) |
  [`paper_c/PAPER_C.tex`](paper_c/PAPER_C.tex) |
  [`paper_c/README.md`](paper_c/README.md) |
  Zenodo [10.5281/zenodo.22110618](https://doi.org/10.5281/zenodo.22110618)

- **Paper D — Architecture**: *"REDEM-SSM: A State-Space Architecture with
  Native Online Learning, Meta-Adaptation, and Structural Plasticity"* —
  a native SSM-hosted architecture instantiating M1/M3/M4/M5 from the
  ground up, motivated by Paper C's host-boundary result (desk-rejected by
  *PRX Intelligence*: scope mismatch — the paper is CS/ML, not
  physics-advancing). Intermediate AI draft kept for rollback; **current
  submission = Neurocomputing** (`elsarticle`, 31 pp), with P4 M3/M4
  factor ablation and a classical ESN+RLS baseline
  → [`paper_d/PAPER_D_NC.pdf`](paper_d/PAPER_D_NC.pdf) |
  [`paper_d/PAPER_D_NC.tex`](paper_d/PAPER_D_NC.tex) |
  [`paper_d/README.md`](paper_d/README.md) |
  [`paper_d/CHANGELOG_PAPER_D_NC.md`](paper_d/CHANGELOG_PAPER_D_NC.md) |
  Zenodo [10.5281/zenodo.22110623](https://doi.org/10.5281/zenodo.22110623)

- **Paper E — Self-evolution**: *"Self-evolution under ±1 reward:
  sign-flip correction, reconstructed-label capacity sensing, and
  snapshot-gated memory in a recurrent relaxation reservoir"* — the
  autonomous correction stack: direction detection and sign-flip
  correction, a reconstructed-label capacity sense, snapshot-gated
  frozen-hypothesis memory with divergence protection, the multi-level
  expansion chain, the measured D2 timescale bound, content–rendering
  separation, multi-source validation, and an external EWC baseline
  (target: *Neurocomputing*; submission-ready elsarticle
  version). **Current manuscript = v3** (2026-09-10 stats/positioning
  repair + 2026-09-13 meta/clean-P0 wording): v3 corrects two false
  significance statements against the committed per-run data (cross-family
  ring gain `t=2.45 > 2.262`, `p≈0.037`; three of four per-segment memory
  gains significant), rewrites the Introduction positioning, tightens the
  Abstract (283 words on one counting convention; Neurocomputing states no
  abstract word limit), adds **EWC**
  (`scripts/s65_ewc_baseline.py`, bit-exact pairing to s52: whole-run
  −0.85±0.70 pp, t=−3.87, 0/10; only redistributes revisit retention,
  while frozen-snapshot memory gains +8.70±2.92 pp with no linear loss),
  and aligns title/abstract/cover letter with the retracted
  per-dimension-correction claim (uniform sign flip only); 68 pp
  → [`paper_e/PAPER_E_v3.pdf`](paper_e/PAPER_E_v3.pdf) |
  [`paper_e/PAPER_E_v3.tex`](paper_e/PAPER_E_v3.tex) |
  `paper_e/PAPER_E_v2.pdf` (v2, local only — not committed) |
  `paper_e/PAPER_E_v1.pdf` (reviewed baseline, local only — not committed) |
  [`paper_e/README.md`](paper_e/README.md) |
  [`review_workspace/modification_log_E_v3.md`](review_workspace/modification_log_E_v3.md)
  (v3 change log) |
  [`review_workspace/modification_log_EF_meta_20260913.md`](review_workspace/modification_log_EF_meta_20260913.md) |
  [`review_workspace/modification_log_EF_clean_p0.md`](review_workspace/modification_log_EF_clean_p0.md) |
  submission materials: `paper_e/COVER_LETTER.docx`, `paper_e/VITAE.docx`,
  `paper_e/Highlights.docx` (+ `Highlights.txt`), `paper_e/DECLARATION_OF_INTERESTS.docx`,
  `paper_e/Supplementary_Material_PaperE.zip` (**rebuilt for v3**: manuscript =
  `PAPER_E_v3.tex`, scripts s39–s65, post-rerun committed data, six figures)
  (Paper E is self-contained: its scripts `s39`–`s65` import the frozen
  `paper_e/deps/` modules, so it is reproducible from its own folder
  without the A–D pipeline.)

- **Paper F — Readout form** (companion to D):
  *"Learning the Readout Form: Objective-Level Calibration, Sparse
  Selectivity, and Expert Routing on a Diagonal State-Space Host"* —
  Paper D shows the squared-loss / clip-floor metric is load-bearing;
  Paper F *changes the instrument* (trained softmax readout) and then
  asks which selectivity pressure the clean metric still needs (top-k
  gates, expert routing). External Mamba-style selective-SSM baseline
  (s66) loses to Gate-C-topk on the same host/stream/seeds
  (stream 9.03 vs 7.03, 0/10 external better). The external arm's 24,864
  host parameters are **frozen at a random init**, so it trains *fewer*
  parameters than Gate-C-topk (8,224 vs 24,736), not more. Target:
  *Neurocomputing* / TMLR
  (draft, 13 pp)
  → [`paper_f/PAPER_F.pdf`](paper_f/PAPER_F.pdf) |
  [`paper_f/PAPER_F.tex`](paper_f/PAPER_F.tex) |
  [`paper_f/README.md`](paper_f/README.md) |
  [`review_workspace/modification_log_paper_f.md`](review_workspace/modification_log_paper_f.md) |
  [`review_workspace/S66_EXTERNAL_BASELINE_FINDINGS.md`](review_workspace/S66_EXTERNAL_BASELINE_FINDINGS.md)

## What each paper contributes to the series

| | Paper A | Paper B | Paper C | Paper D | Paper E | Paper F |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Question** | What can this substrate compute? | How do you learn on it? | Which mechanism does what? | What host makes it native? | How does it fix itself under ±1 reward? | Which readout form / selectivity pressure is still required under a clean metric? |
| **Core result** | κ*∈(25,30); held-out MC +24–53% just before it; forgetting kernel r=0.97; λ-homeostat +7.9–18% | 0.996 vs 0.973 (p<0.0001; N=1024: 0.9970 vs 0.9753); tracks drift where frozen learners never recover | the +32% sequential-recovery gain is the homeostat's, not the metadata's; metadata robustness transfer falsified (0/10 seeds at every τ_m); routing transfers, gating-only falsified | calibrated input path (state-mixture ranking is a clip-floor/calibration property); routing retains specialists; pause-learning does **not** pass the retention rubric once the floor is accounted for; full stack beats bare host on forgetting (−4.47/−4.53, 10/10) and TF+LoRA on reported axes; P4 factor ablation: hard M3 carries forgetting, soft M4 recovers stream; classical ESN+RLS fails out of sample (~98 vs bare 15.4 / lin 12.4) | reward-rate sign hides the direction; credit assignment = rule selection; ~~structural (per-dimension) gain correction required~~ **retracted** (only a uniform sign flip is demonstrated; s47 is a retracted design attempt, **not** a negative result); capacity sense idles/triggers by reconstruction; frozen memory beats re-adaptation; D2 timescale bound verified; decision-level autonomy + minimal operational self (persistence / functional identity / causal use); memory, not expansion, separates content–rendering (s63); majority rule collapses under a wrong majority but validation-driven selection finds the correct source (s64); s42 corrupt-reward is an honest negative (self-deception, not self-detectable); s58c multi-level expansion refuses the cubic rung and memory masks (not prevents) divergence; EWC does not improve the run and only redistributes revisit retention; hosts: E on relaxation reservoir, D/F on SSM, G is a port design (not in E) | trained softmax removes the clip-floor artifact (simplex projection does not); hard top-k is required for selective gates to retain; expert routing cuts forgetting further at stream parity; external selective-SSM baseline loses on stream (0/10) with ~1.34× Gate-C-topk params |
| **Target venue** | Chaos, Solitons & Fractals | Neural Networks | Neural Networks | Neurocomputing (NC retarget; PRX desk-reject) | Neurocomputing | Neurocomputing / TMLR |
| **Preprint DOI** | [10.5281/zenodo.22109664](https://doi.org/10.5281/zenodo.22109664) | [10.5281/zenodo.22110606](https://doi.org/10.5281/zenodo.22110606) | [10.5281/zenodo.22110618](https://doi.org/10.5281/zenodo.22110618) | [10.5281/zenodo.22110623](https://doi.org/10.5281/zenodo.22110623) | — (submission) | — (draft) |

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
  `fair_esn_comparison.py` are frozen from the prior project
  ([github.com/huyamingc/Si3N4-Pulse-Encoding](https://github.com/huyamingc/Si3N4-Pulse-Encoding)),
  imported by the current code, and never modified.
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
| **Key results** | Order–chaos transition at κ*∈(25,30); held-out memory +24–53% just before it; forgetting kernel M(t)=∫p(τ)e^{−t/τ}dτ matches the measured memory curve (r=0.97); λ-homeostat restores 7.9–18% after disturbances; edge of chaos (λ_target=0) identified as the best-supported target (+25%; main results run at the conservative −0.02) | Tracks drift where frozen batch learners (GRU, transformer) fail permanently; integrated system beats the bare baseline (0.996 vs 0.973, p<0.0001) and matches every ablation (homeostat removal ties; N=1024, 10 seeds: 0.9970 vs 0.9753); metadata is substrate-agnostic — its statistical-memory benefit transfers to a matched ESN (robustness transfer falsified in Paper C); +32% memory after three sequential disturbances; causal audit confirms all mechanisms are causally clean at the operational leak magnitudes tested |
| **Target journal** | Chaos, Solitons & Fractals | Neural Networks |
| **Relationship** | Substrate theory; cites the prior Si₃N₄ pulse-encoding paper for device calibration | Builds on Paper A's substrate theory (cited as the companion in §2) |

All experiments are CPU-only and fully reproducible via the scripts in
`scripts/` (a shared CORE substrate and readout library supports both papers).
Data and figures live in `data/` and `figures/`.

- `README_REDEM.md` — detailed technical README: architecture, headline
  results (S1–S10), script inventory, reproduction commands.

## Repository layout

```
├── paper_a/     Paper A: substrate characterization (PDF, LaTeX, README)
├── paper_b/     Paper B: REDEM online learning architecture (PDF, LaTeX, README)
├── paper_c/     Paper C: three-mechanism disentanglement (PDF, LaTeX, README)
├── paper_d/     Paper D: native REDEM-SSM architecture (PDF, LaTeX, README; current = Neurocomputing)
├── paper_e/     Paper E: self-evolution (submission-ready elsarticle, frozen deps/, submission materials)
├── paper_f/     Paper F: readout form / calibrated objective + sparse selectivity (draft, companion to D)
├── scripts/     Shared simulation code (CORE substrate, tasks, readouts, figure scripts; s39–s65 = Paper E; F uses s50–s54/s66)
├── data/        All experiment results (CSV + JSON, 10-seed means)
├── figures/     All publication figures (vector PDF; no raster twins)
└── *.md         Overview and technical README
```

Each paper folder contains its own `README.md` (key results, figures and
data anchors, compile instructions, and the reproduction commands scoped
to that paper), so a paper can be read and reproduced independently of the
series. Paper E additionally carries its **frozen dependency set** in
`paper_e/deps/` (the four core modules its scripts import), so it is fully
self-contained and reproducible from its own folder without the A–D
pipeline; its submission materials (cover letter, Vitae, Highlights,
declaration of interests, supplementary zip) live beside the manuscript.

## Scripts

All committed scripts in `scripts/`, typed
(ML > CORE > PAPER > FIG > EXPLORE). The **Paper** column marks which paper
each script serves (A / B / C / D; shared = library or figure used by more
than one paper). Two legacy scripts from the prior Si₃N₄-pulse-encoding
project are kept as shared dependencies (imported by the new code) and are
never modified. `README_REDEM.md` holds the full headline-results registry
(S1–s36) and reproduction commands for Papers A–D.

**Paper E chain (s39–s65).** The self-evolution experiments live in the
same `scripts/` directory but import the **frozen `paper_e/deps/` modules**
(they shadow the shared core), so they are reproducible independently of
Papers A–D. They are NOT enumerated in the A–D table below; see
[`paper_e/README.md`](paper_e/README.md) for the claim→script→data index.
Key members: `s39`–`s47` (direction, rule selection, sign-flip
correction), `s48`–`s58` (capacity sensing, memory, D2), `s58e`–`s62`
(timescale bounds), `s63` (content–rendering separation), `s64`
(multi-source validation), `s65` (external EWC baseline on the revisit
protocol), and
`gen_fig_self_evolution.py` (Paper E figures).

**Paper F chain (s50–s54, s53b, s66).** Readout-form experiments on the
Paper D diagonal-SSM host / stream protocol: `s50` (trained-softmax
pilot + controls), `s51` (learned gates incl. Gate-C-topk), `s52`
(soft/hard expert routing), `s53`/`s53b`/`s54` (N-scaling, k-sweep,
Mackey–Glass transfer), `s66` (external selective-SSM baseline).
See [`paper_f/README.md`](paper_f/README.md) and
[`review_workspace/modification_log_paper_f.md`](review_workspace/modification_log_paper_f.md).

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
| `chaos_regulator.py` | PAPER | B | S6: λ-homeostat under disturbances | held-out MC +7.9–18% |
| `structure_plasticity.py` | PAPER | B | S7: correlation-guided rewiring | gentle (5%) +8–11%; aggressive (20%) −23% |
| `integrated_benchmark.py` | PAPER | B | S8: full system vs ablations; N=1024 confirmation | 0.996 vs 0.973 (p<0.0001); N=1024 (s30, 10 seeds) 0.9970 vs 0.9753 |
| `baseline_showdown.py` | ML | B | S9: vs matched ESN / GRU / tiny transformer (torch CPU); fairness protocol: per-trial torch seeding precedes weight init | REDEM 0.991 vs GRU 0.371 / TF 0.353 (ESN 1.000, honest; z-scored features); MG NMSE GRU 1.80 / TF 0.77 |
| `forgetting_curve_theory.py` | EXPLORE | A | S10: forgetting-kernel theory M(t), Gauss–Hermite validation | measured MC follows M(t), r = 0.97 |
| `esn_metadata_comparison.py` | PAPER | B | S10: metadata transfer to a matched ESN | equalizes: ESN+meta 0.998 ≈ ESN 0.996 ≈ REDEM 0.994 |
| `cv_sweep.py` | PAPER | A | S10: task-level CV sweep (Paper A Supp. Note 1) | uncoupled MC +33% with CV; coupled near-critical falls |
| `s11_disturbance_chain.py` | PAPER | B, C | E3: sequential disturbance chain (3 rounds, 10 seeds; anchor for Paper C's ESN falsification) | regulated MC 8.47 vs fixed 6.41 (+32%) |
| `s12_lambda_target_sweep.py` | PAPER | A | E4: λ_target × CV optimization sweep (5 seeds) | λ_target=0 optimal: +25%/+19%/+5% MC |
| `s13_causal_audit.py` | PAPER | B | O4: causal leakage audit (7 arms, 10 seeds) | all mechanisms causally clean (≤0.02 pp; plasticity-correlation leak implemented, −0.01 pp) |
| `s14_esn_disturbance_chain.py` | PAPER | C | ESN+metadata under the disturbance chain — falsifies metadata robustness transfer (3 arms, 10 seeds) | paired diffs −0.78/−0.76/−0.69, 0/10 seeds positive |
| `s16_tau_m_pressure_test.py` | PAPER | C | τ_m ∈ {200,500,1000,2000} stress test — falsification robust across metadata timescales (10 seeds) | 0/10 seeds positive at every τ_m (~5σ) |
| `s15_controlled_adaptation.py` | PAPER | C | controlled adaptation protocol with known switch instants (10 seeds) | true effect ~10 pulses + variance collapse (p90 76.5→42) |
| `s16b_falsification_stress_test.py` | PAPER | C | probe-protocol stress test (V0/V1/V2, 4 τ_m, 10 seeds) | sign robust ≤1/10 everywhere; magnitude V0 −0.69 → V2 −0.04/−0.01 across τ_m |
| `s5b_controlled_adaptation.py` | PAPER | B | S5 three-arm controlled adaptation re-measurement (2 substrates, 10 seeds) | table-endpoint factor 1.25–2.7 (T200 1.25–1.50, T40 1.54–2.73; Paper B §4.3) |
| `s17_substrate_stress.py` | PAPER | C | ESN substrate stress — equalizer gain at all configs (120 runs) | gain positive at all 6 ESN configs (+0.1 to +1.0 pp) |
| `s18_llm_drift_gate.py` | ML | C | §7 PoC: LLM drift gate (tiny transformer + LoRA, 90 runs, torch CPU) | A3 routing transfers −2.5 ppl (10/10); A2 gate falsified (0/10) |
| `s19_ssm_rls_readout.py` | ML | D | P1/P3a: per-token RLS readout on a hand-rolled diagonal SSM (7 arms, 70 runs, torch CPU) | state arms 58.5–115.9 (0/10) vs B-proj 11.75 (d −3.26, 10/10); oracle 7.25 |
| `s20_ssm_m3_routing.py` | ML | D | P2: M3 EMA metadata + drift detection + routing on the SSM host (A1/A2/A3, 90 runs, torch CPU) | A2 stream −1.52…−2.45 (10/10); A3 forget −2.05/−1.87/−1.20 (10/10, τ_m≤1000) |
| `s21_ssm_m4_m5.py` | ML | D | P3: M4 soft vs abrupt routing + M5 state-norm homeostat (E1/E2, 70 runs, torch CPU) | soft 8.22 vs abrupt 10.03 (−1.81, 10/10); M5 restores EMA detector 5/5, norm 11.3 vs 50.2 |
| `s22_ssm_p4_benchmark.py` | ML | D | P4: 4-domain irregular-switch benchmark (SSM-bare / SSM-REDEM / TF-A1, 30 runs, torch CPU) | REDEM-SSM vs bare −2.25/−4.47, vs TF −9.28/−10.08 (10/10) |
| `s23_ssm_p4_realtext.py` | ML | D | real-text benchmark: two Gutenberg books (Alice vs Dickens), 32-symbol char vocab (30 runs, torch CPU) | REDEM-SSM vs bare −1.26 (10/10), vs TF −5.23 (10/10) |
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
| `s21_ssm_m4_m5.py --frozen-p-probe` | ML | D | dormant-covariance-refresh isolation (weights frozen, inverse P refreshed; writes s37) | stream/forget both exactly 32.0 (uniform) on all 10 seeds — refresh alone learns nothing |
| `s38_ssm_p4_m3m4_ablation.py` | ML | D | P4 factor ablation: bare / uniform / hard M3 / soft full stack (4 arms, 10 seeds) | Hard M3: +2.90/−5.24 (0/10, 10/10); soft vs hard: −5.14/+0.77 — M3 carries forgetting, soft M4 recovers stream |
| `s40_esn_rls_p4_baseline.py` | ML | D | classical ESN + online RLS on P4 (4 arms incl. one-hot control, 10 seeds) | ESN-128 stream 98.8 vs bare 15.4 / lin 12.4 (0/10 vs both); bare-vs-lin gap is the clip floor (unclipped bare ~11.5) |
| `s34_leak_sensitivity.py` | PAPER | B | leak sensitivity scan on the S28 audit (7 leak configs, 10 seeds); leak_k drives the FTLE leak horizon (s28 hard-codes 400) | 10× FTLE leak still NS at every tested horizon (+0.13/+0.39/+0.21 at horizons 50/200/400, CIs include 0); plasticity 30% future correlation +0.57 mean (7/10; paired t=2.04, 95% CI [−0.06,+1.20]) — audit resolution is bounded, "causally clean" scoped to operational leaks |
| `s35_readout_boundary_probe.py` | PAPER | D | P1 readout boundary probes (10 seeds, s19 host verbatim): full/half-window oracle, skip-vs-proj nested check, fast/slow token decoding, fast-channel next-token readouts vs static-table reference | full-window oracle 31.2 (matches s19, max diff 0.00) vs half-window 17.3 — oracle is window-dependent; skip 18.0 > proj 7.25 (nested violation, 10/10); token linearly decodable from fast channels (τ≤8) at 88.9–99.7% out-of-sample (chance 3.1%), slow channels at chance; yet fast-channel-only direct and two-stage next-token readouts still fail out-of-sample (ppl 68–104 vs static table 13.9–17.4) — "no useful linear map" is false, and no closed-form squared-loss state readout yields calibrated next-token probabilities; the P1 failure is a pooled-readout/metric property, not missing linear information |
| `s36_random_graph_instance_variability.py` | PAPER | A | random-graph instance variability: S1 random_graph sweep repeated on 9 Erdős–Rényi instances (8 fresh + original 777), 10 substrate seeds each (990 runs); reuses substrate_recurrence_characterization.run_single with per-task TOPO_SEED | peak held-out MC 13.47±0.57 (range 12.80–14.35), +48.5%±6.3% over uncoupled; κ*=25 (7/9) or 30 (2/9); original instance 13.91 (+53%) inside the envelope |
| `gen_paperD_fig1_p1_arms.py` | FIG | D | Paper D Fig. 1: P1/P3a state-readout falsification + input-path control; pooled-table ceiling computed from s19 `oracle_ppl` 10-seed mean 7.25 | `figures/paperD_fig1_p1_arms.pdf` |
| `gen_paperD_fig2_routing.py` | FIG | D | Paper D Fig. 2: P2 routing retention + P3 soft vs abrupt | `figures/paperD_fig2_routing.pdf` |
| `gen_paperD_fig3_benchmark.py` | FIG | D | Paper D Fig. 3: P4 benchmark bars | `figures/paperD_fig3_benchmark.pdf` |
| `gen_architecture_schematic.py` | FIG | shared | Paper Fig. 1 schematics (substrate / REDEM, M4↔M5 loop) | `figures/paperA_fig1_substrate.pdf`, `figures/paperB_fig1_redem.pdf` |
| `gen_substrate_phase_diagram.py` | FIG | A | S1 phase-diagram figure | `figures/substrate_phase_diagram_v2.pdf` |
| `gen_s2_curves.py` | FIG | B | S2 figure: drift-tracking curves + final-30% NMSE bars (reads the curves npz and the s2 JSON aggregates) | `figures/s2_online_readout_v1.pdf` |
| `gen_paper_figures.py` | FIG | shared | batch: robustness / metadata / ablation / showdown | `figures/paperA_fig4_robustness.pdf`, `figures/paperB_fig3_metadata.pdf`, `figures/paperB_fig5_ablation.pdf`, `figures/paperB_fig6_showdown.pdf` |
| `gen_paperA_supp_figures.py` | FIG | A | Paper A Supplementary Fig. S1 (CV sweep) | `figures/paperA_figS1_cv_sweep.pdf` |
| `gen_paperC_fig1_kernel.py` | FIG | C | Paper C Fig. 1: slow-trace kernel vs material forgetting kernel | `figures/paperC_fig1_kernel.pdf` |
| `gen_paperC_fig2_recovery.py` | FIG | C | Paper C Fig. 2: post-disturbance MC recovery vs τ_m (s16) | `figures/paperC_fig2_recovery.pdf` |
| `gen_paperC_fig3_llm.py` | FIG | C | Paper C Fig. 3: LLM drift-gate results (s18) | `figures/paperC_fig3_llm.pdf` |

Each FIG script emits a single vector `.pdf` (journal submission); the papers
include the extension-less basename so `pdflatex` picks the vector file
automatically. Reproduction commands (S1–s36 shared registry) are in
`README_REDEM.md`; Paper D follow-ups `s37`/`s38`/`s40` and the Paper E/F
chains are indexed in the per-paper READMEs.

## Code availability

All simulation code and data required to reproduce the results are openly
available at <https://github.com/huyamingc/REDEM>.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The manuscripts in `paper_a/` and
`paper_b/` are the author's preprints: copyright is retained by the author
until journal publication, after which the journals' copyright terms apply.
