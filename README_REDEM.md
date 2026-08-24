# REDEM — Physics-Grounded Online Learning Architecture

> Status: experimental program complete (S1–S9); papers drafted, revised and
> typeset (S10). The root `README.md` is the repository overview; the Si3N4
> pulse-encoding paper lives in its own repository
> (github.com/huyamingc/Si3N4-Pulse-Encoding) and is cited as the substrate
> calibration source in both papers. `NEW_ALGORITHM_PLAN.md` is the
> authoritative research plan and changelog for this project.

REDEM (working name; REward-gated Dual-timescale Eligibility Mechanism) is an
online "training == inference" learning architecture built on a
physics-constrained relaxation substrate: Si3N4-style shallow traps with a
log-normal time-constant spectrum, coupled by a per-pulse, topology-dependent
modulation of the injection coefficient. No backpropagation through time, no
GPU, no fabrication — everything runs numerically on CPU.

## Architecture

| Component | Mechanism | Role |
|---|---|---|
| Substrate | log-normal τ traps (median 174 µs, CV 0.20), per-pulse contrast coupling (κ knob) | nonlinear fading-memory kernel; chaos tuneable |
| Readout | Online RLS (λ=0.999, predict-before-update) | the online learner; error-driven, second-order |
| M3 Metadata | per-unit slow EMA of fast features (τ_m = 200–1000 pulses) | long-horizon statistical memory |
| M5 Chaos homeostat | Benettin FTLE estimate every 1000 pulses → κ step toward λ_target = −0.02 (manually tuned; E4 sweep identifies λ_target = 0 as optimal) | keeps substrate near the memory-optimal critical point |
| M4 Structure plasticity | every 2000 pulses, prune 5% lowest-|corr| edges, grow 5% highest-|corr| unconnected pairs | slow structural adaptation |

Two negative results delimit the design: reward-modulated Hebbian readouts
(without an error channel) cannot credit-assign through a class-interval
inversion; task-agnostic intrinsic rewards cannot rescue them. Error-driven
second-order learning is required at the readout; intrinsic signals belong at
the structure level (M4).

## Headline results (all 10 seeds unless noted)

| # | Result |
|---|---|
| S1 | Order–chaos transition at κ* ∈ (25,30); held-out memory +24–53% just before it; chaos destroys memory |
| S2 | Online RLS tracks class-inversion drift (mean acc 0.974–0.982, recover 225–616 pulses); frozen offline inverts and never recovers; near-critical coupling → Mackey-Glass NMSE 0.0018 (50× vs uncoupled) |
| S3/S4 | Reward-only three-factor fails at inversion (post 0.06–0.10); novelty intrinsic never rescues (stream mean ≤ 0.514) — negative results that fix the design |
| S5 | Dual-timescale metadata: regime task +1.3–2.1 pp (p<0.0001); adaptation re-measured by s5b (controlled switch-relative): T200 265–304 (fast) vs 202–211 (dual), factor 1.3–2.4 (the earlier "9–20× faster" ratio was a window-position metric artifact) |
| S6 | λ-homeostat: post-disturbance held-out memory +8–18% (τ-drift +18%) |
| S7 | Gentle (5%) correlation-guided rewiring +8–11%; aggressive (20%) −23%; pruning redundancy helps (de-homogenization) |
| S8 | Integrated system beats every ablation: 0.996 vs 0.973 (p<0.0001); persists at N=1024 (0.998 vs 0.976) |
| S9 | Online (REDEM 0.991, ESN 1.000) vs frozen batch (GRU 0.394, transformer 0.351, inverted post-swap); MG NMSE REDEM 0.0018 vs ESN 3.6e-5 vs GRU 1.56 vs TF 0.71; ESN edge on standard tasks reported honestly (2026-02-19 revision: z-scored ESN features, per-trial torch seed, causal transformer mask) |
| Theory | Forgetting kernel M(t)=∫p(τ)e^{−t/τ}dτ: 1/e horizon ≈16 pulses (median-pinned); tail steepness set by CV; measured MC curve follows it with r = 0.97 |
| S10-CV | Task-level CV sweep: uncoupled MC rises with CV (+33%, kernel theory); coupled near-critical MC falls (narrow CV best) — operating-regime-dependent knob |
| S10-ESN | Metadata transfer: ESN+meta 0.998 ≈ ESN 0.996 ≈ REDEM-full 0.994 on regime task — the mechanism equalizes the systems (adapt 11.22→0.24 in the S10 metric; controlled s15 shows the true effect is ~10 pulses + variance collapse) |
| E3 | Sequential disturbance chain (3 rounds: τ-drift → edge-prune → noise, 10 seeds): regulated arm maintains MC 8.47 vs fixed 6.41 (+32%) after all three; κ drifts 26.2→28.5 (active compensation) |
| E4 | λ_target sweep (4 values × 3 CV × 5 seeds): λ_target=0 (edge of chaos) is optimal — +25%/+19%/+5% MC gain over fixed at CV=0.1/0.2/0.4; monotonic improvement as λ→0 |
| O4 | Causal audit (7 arms × 10 seeds, re-run 2026-02-19 with the plasticity-correlation leak implemented): all adaptive mechanisms are causally clean — largest leak-arm deviation 0.012 pp (RLS target), plasticity leak −0.01 pp; causal-split protocol 0.9963 vs normal 0.9963 |
| s14 | ESN+metadata under the S11 disturbance chain (3 arms × 10 seeds): the slow trace does NOT transfer MC robustness to the ESN (paired diffs −0.78/−0.76/−0.69, 0/10 seeds positive) — the S11 +32% recovery is homeostat-driven; metadata still attenuates readout noise (r3 NMSE −9.5%, 10/10 seeds); redem_reg reproduces the S11 anchor exactly |
| s16 | τ_m pressure test (τ_m ∈ {200,500,1000,2000} × 10 seeds): the falsification is robust — esn_dual r3 MC ≤ esn_fast at every τ_m (0/10 seeds positive, ~5σ), no sensitive interval; noise-attenuation transfer holds at all τ_m. Paper C adopts the strong claim |
| s15 | Controlled adaptation (10 seeds × 5 switches, known switch instants): T40 40.6 (dual) vs 49.9 (fast) vs 52.7 (redem); T40 p90 42 vs 76.5 — the true metadata adaptation effect is ~10 pulses + variance collapse; the "47×"/"9–20×" ratios are metric artifacts |
| s16b | Probe-protocol stress test (10 seeds × 4 τ_m × 3 variants, v2): falsification robust in sign (≤1/10 positive in every cell; 0/10 in 11 of 12); magnitude protocol-dependent — V0 (readout noise) −0.69, V1 (std-slow) −0.66, V2 (state noise) −0.04…−0.01 across τ_m (EMA denoising nearly closes the gap at every timescale) |
| s5b | S5 three-arm controlled re-measurement (2 substrates × 5 arm configs × 10 seeds): T200 fast 264.8–303.9 vs dual/slow 202–211 pulses (factor 1.28–1.44; T40 factor 1.79–2.40); overall acc reproduces S5 exactly — Paper B §4.3/abstract revised |
| s17 | Substrate stress (3 spectral radii × 2 hetero × 10 seeds): equalizer gain positive at all 6 ESN configs (+0.1 to +1.0 pp), magnitude tracks the fast channel's timescale starvation; heterogeneity irrelevant |
| s18 | LLM drift-gate PoC (tiny transformer + LoRA, 90 runs): A3 domain routing transfers — forgetting −2.5 ppl (10/10 seeds, τ_m=200), stream ppl −1.07 (10/10) at τ_m≤500; A2 gate falsified (0/10); τ_m≥1000 sensitive interval reported |
| s19 | Paper D P1/P3a (diagonal SSM, per-token RLS, 70 runs): naive state readout falsified — state arms 58.5–115.9 ppl (0/10) vs B-projection input path 11.75 (d −3.26, 10/10); MLE ceiling probe 7.25 vs 7.35 B-proj oracle |
| s20 | Paper D P2 (M3 routing, 90 runs): gating-only inverts on the RLS host (stream −1.52…−2.45, 10/10 — what-is-paused qualifier); M3 routing transfers (forget −2.05/−1.87/−1.20 at τ_m≤1000, 10/10) |
| s21 | Paper D P3 (M4 soft routing + M5 homeostat, 70 runs): soft 8.22 vs abrupt 10.03 (−1.82, 10/10); dormant-covariance refresh flips routing to −1.72; M5 bounds whitened norm 11.3 vs 50.2 and restores the full-state EMA detector 5/5 |
| s22 | Paper D P4 benchmark (4-domain irregular switches, 30 runs): REDEM 13.18/8.93 vs bare SSM 15.43/13.41 vs TF-A1 22.46/19.01 (stream/forget) — all 10/10 |
| s23 | Paper D real-text transfer (two Gutenberg books, 30 runs): REDEM 12.07/11.85 vs bare 13.34/12.31 vs TF 17.31/13.86 — stream −1.27 vs bare (10/10), −5.23 vs TF (10/10) |
| s24 | M4-M5 coupling loop under the S11 disturbance chain (4 arms × 10 seeds): rewiring churn gated by the homeostat's κ deviation does NOT help recovery — homeostat alone r3 MC 8.47 vs fixed-churn 6.45 (t=−9.6, 0/10) / coupled 5.27 (t=−5.9, 0/10); rewiring during active disturbance compensation is harmful — recovery is the homeostat's job alone |
| s25 | Novelty-reward-gated rewiring vs correlation-guided (4 arms × 10 seeds, S7 protocol): novelty-guided MC_final 14.59 vs corr 12.43 (d=+2.15, t=4.5, 10/10) vs random 12.60 (t=0.3) vs ring 11.54 (t=8.0, 10/10) — intrinsic novelty is a structure-level tool, improving on M4's correlation heuristic |
| s26 | Fair Transformer references for P4 (9 arms × 10 seeds): tuned A1 grid — best stream 14.86 (lr 3e-3, r32) still loses to REDEM 13.18 (0/10, −1.68) and its forgetting collapses to 62.2; TF-A3 4-adapter routing — forgetting 10.85 vs 19.01 (−8.17, 10/10, −43%) but stream 21.77 loses 0/10 (+8.59) — the routing mechanism transfers to the Transformer host, the stream advantage does not |
| s27 | Clip-range ablation × fine κ grid (3 topologies × 3 α_max × 13 κ × 10 seeds): κ* = 25.3/27.4/27.9 (bootstrap CI ±0.1–0.8); 5× clip widening leaves κ* invariant for lateral_ring (+0.05) and random_graph (−0.21) — order–chaos is coupling-driven; ring_bidir shifts +3.4 (clip contributes); wider clip raises ring_bidir peak MC 11.9→17.7 |
| s28 | Causal leak audit re-run on the S11 disturbance chain (6 arms × 10 seeds): 1% future leaks into RLS/metadata/FTLE and 10% future correlation into M4 do not improve recovery (r3 MC deltas +0.00/+0.00/+0.28/+0.24, all NS; NMSE ≤0.004); no-plasticity r3 MC 8.47/κ 28.5 = S11 homeostat anchor exactly — causal cleanliness survives a non-ceilinged protocol |
| s30 | N=1024 integrated replication at 10 seeds (baseline vs full, paired per seed): full 0.9970 ± 0.0007 vs baseline 0.9753 ± 0.0044 (+2.2 pp, paired t=15.3, 10/10) — the scale-up claim now supports a paired test (S8 used 3 seeds) |
| s31 | Char-bigram oracle on the real-text protocol (full-book vs ref-window fits, 10 seeds): true first-order ceiling ppl 10.97 ± 0.18; SSM-REDEM 12.07 sits within ~1.1 ppl of its structural ceiling — the "first-order regime" boundary is now quantitative |
| s32 | FTLE-noise robustness of the homeostat (5 noise levels × 10 seeds, S11 chain): no significant r3 MC degradation up to σ=0.10 (~100% of the estimate scale; paired CIs include 0), settled κ 28.5–28.7 — the clipped proportional feedback integrates out estimation error |
| s33 | M5 in the P4 benchmark (2 arms × 10 seeds): SSM-REDEM+M5 is significantly worse (stream +1.53, forgetting +2.90, t=−22.8/−26.5, 10/10) — Δt modulation breaks the Δt=1-calibrated per-channel whitening on an already well-conditioned stream; S22's exclusion of M5 is validated (M5 stays a spike-regime safeguard, P3 E2) |
| s34 | Leak sensitivity scan on the S28 audit (6 leak configs × 10 seeds): 10× larger FTLE leaks stay NS (+0.21); the plasticity channel shows a genuine dose-response — 30% future correlation improves recovery (+0.57, CI [+0.08,+1.10], 7/10), 50% reverses it — "causally clean" is scoped to the operational leak magnitudes |

## Scripts (53 committed)

The authoritative script inventory — every script with its `CLAUDE.md` type,
**paper attribution (A/B/C/D/shared)**, purpose, and key result — is the
table in the root [`README.md`](README.md) (single source of truth; it is
kept in sync with `git ls-files scripts/`). Paper D experiments s19–s23, s26,
s31, s33 and
their figures are reproduced from `paper_d/README.md`.

## Data and figures

- Results: `data/substrate_phase_diagram_v2.*`, `data/s2_online_readout_v1.*`,
  `data/s3_three_factor_v1.*`, `data/s4_intrinsic_reward_v1.*`,
  `data/s5_dual_timescale_v1.*`, `data/s6_chaos_regulator_v1.*`,
  `data/s7_structure_plasticity_v1.*`, `data/s8_integrated_v1.*`,
  `data/s9_baseline_showdown_v1.*`, `data/forgetting_curve_theory.csv`,
  `data/s10_esn_metadata_v1.*`, `data/s10_cv_sweep_v1.*`,
  `data/s11_disturbance_chain_v1.*`, `data/s12_lambda_target_sweep_v1.*`,
  `data/s13_causal_audit_v1.*`, `data/s14_esn_disturbance_chain_v1.*`,
  `data/s16_tau_m_pressure_test_v1.*`, `data/s15_controlled_adaptation_v1.*`,
  `data/s16b_falsification_stress_test_v1.*`, `data/s5b_controlled_adaptation_v1.*`,
  `data/s17_substrate_stress_v1.*`, `data/s18_llm_drift_gate_v1.*`,
  `data/s24_homeo_plasticity_coupling_v1.*`,
  `data/s25_reward_gated_plasticity_v1.*`,
  `data/s26_ssm_p4_fair_tf_v1.*`, `data/s27_clip_kappa_fine_v1.*`,
  `data/s28_causal_audit_chain_v1.*`, `data/s30_integrated_1024_v1.*`,
  `data/s31_char_bigram_oracle_v1.*`, `data/s32_ftle_noise_v1.*`,
  `data/s33_ssm_p4_m5_v1.*`, `data/s34_leak_sensitivity_v1.*`,
  `data/s16b_falsification_stress_test_v2.*` (v2: all variants at all
  four τ_m; v1 kept for the original table)
  (CSV per run + JSON with params and per-cell aggregates).
  Paper D: `data/s19_ssm_rls_readout_v1.*`, `data/s20_ssm_m3_routing_v1.*`,
  `data/s21_ssm_m4_m5_v1.*`, `data/s22_ssm_p4_benchmark_v1.*`,
  `data/s23_ssm_p4_realtext_v1.*`; real-text corpus in
  `data/corpora/` (Gutenberg public domain, see `data/corpora/README.md`).
- Figures: `figures/substrate_phase_diagram_v2.pdf`,
  `figures/s2_online_readout_v1.pdf`, `figures/forgetting_curve_theory.pdf`,
  `figures/paperA_fig1_substrate.pdf`, `figures/paperA_fig4_robustness.pdf`,
  `figures/paperA_figS1_cv_sweep.pdf` (Supplementary Fig. S1),
  `figures/paperB_fig1_redem.pdf`, `figures/paperB_fig3_metadata.pdf`,
  `figures/paperB_fig5_ablation.pdf`, `figures/paperB_fig6_showdown.pdf`,
  `figures/paperC_fig1_kernel.pdf`, `figures/paperC_fig2_recovery.pdf`,
  `figures/paperC_fig3_llm.pdf` (Paper C, in development),
  `figures/paperD_fig1_p1_arms.pdf`, `figures/paperD_fig2_routing.pdf`,
  `figures/paperD_fig3_benchmark.pdf` (Paper D, in development)
  (vector PDF only; the papers include the extension-less basename so
  `pdflatex` picks the vector version).

## Reproduction

CPU-only; uses the project venv. numba (optional but recommended), numpy,
scipy, matplotlib, torch (CPU) for S9 only.

```powershell
# S1 phase diagram (610 runs, ~1 min with multiprocessing)
& .venv\Scripts\python.exe scripts\substrate_recurrence_characterization.py
# ... each step S1–S9 runs its script; every script accepts --quick for a smoke run:
& .venv\Scripts\python.exe scripts\online_readout_streaming.py --quick
& .venv\Scripts\python.exe scripts\three_factor_online_readout.py
& .venv\Scripts\python.exe scripts\intrinsic_reward_experiment.py
& .venv\Scripts\python.exe scripts\dual_timescale_metadata.py
& .venv\Scripts\python.exe scripts\chaos_regulator.py
& .venv\Scripts\python.exe scripts\structure_plasticity.py
& .venv\Scripts\python.exe scripts\integrated_benchmark.py
& .venv\Scripts\python.exe scripts\cv_sweep.py
# theory + figures
& .venv\Scripts\python.exe scripts\forgetting_curve_theory.py
& .venv\Scripts\python.exe scripts\esn_metadata_comparison.py
& .venv\Scripts\python.exe scripts\cv_sweep.py
# supplementary experiments (E3, E4, O4)
& .venv\Scripts\python.exe scripts\s11_disturbance_chain.py --sequential
& .venv\Scripts\python.exe scripts\s12_lambda_target_sweep.py --sequential
& .venv\Scripts\python.exe scripts\s13_causal_audit.py --sequential
# Paper C experiments (s14, s16, s15, s16b) + figures
& .venv\Scripts\python.exe scripts\s14_esn_disturbance_chain.py --sequential
& .venv\Scripts\python.exe scripts\s16_tau_m_pressure_test.py --sequential
& .venv\Scripts\python.exe scripts\s15_controlled_adaptation.py --sequential
& .venv\Scripts\python.exe scripts\s16b_falsification_stress_test.py --sequential
& .venv\Scripts\python.exe scripts\s17_substrate_stress.py --sequential
& .venv\Scripts\python.exe scripts\s18_llm_drift_gate.py --sequential
& .venv\Scripts\python.exe scripts\gen_paperC_fig3_llm.py
& .venv\Scripts\python.exe scripts\gen_paperC_fig1_kernel.py
& .venv\Scripts\python.exe scripts\gen_paperC_fig2_recovery.py
# Paper B revision: S5 arms controlled re-measurement
& .venv\Scripts\python.exe scripts\s5b_controlled_adaptation.py --sequential
& .venv\Scripts\python.exe scripts\gen_architecture_schematic.py
& .venv\Scripts\python.exe scripts\gen_paper_figures.py
# Paper D experiments (s19–s23, s26, s31, s33) + figures — see paper_d/README.md
& .venv\Scripts\python.exe scripts\s19_ssm_rls_readout.py --sequential
& .venv\Scripts\python.exe scripts\s20_ssm_m3_routing.py --sequential
& .venv\Scripts\python.exe scripts\s21_ssm_m4_m5.py --sequential
& .venv\Scripts\python.exe scripts\s22_ssm_p4_benchmark.py --sequential
& .venv\Scripts\python.exe scripts\s23_ssm_p4_realtext.py --sequential
& .venv\Scripts\python.exe scripts\s24_homeo_plasticity_coupling.py
& .venv\Scripts\python.exe scripts\s25_reward_gated_plasticity.py
& .venv\Scripts\python.exe scripts\s26_ssm_p4_fair_tf.py
& .venv\Scripts\python.exe scripts\s27_clip_kappa_fine.py
& .venv\Scripts\python.exe scripts\s28_causal_audit_chain.py
& .venv\Scripts\python.exe scripts\s30_integrated_1024.py --workers 4
& .venv\Scripts\python.exe scripts\s31_char_bigram_oracle.py
& .venv\Scripts\python.exe scripts\s32_ftle_noise_robustness.py
& .venv\Scripts\python.exe scripts\s33_ssm_p4_m5.py
& .venv\Scripts\python.exe scripts\s34_leak_sensitivity.py
& .venv\Scripts\python.exe scripts\gen_paperD_fig1_p1_arms.py
& .venv\Scripts\python.exe scripts\gen_paperD_fig2_routing.py
& .venv\Scripts\python.exe scripts\gen_paperD_fig3_benchmark.py
```

All experiments use fixed, per-run seeds (`run_idx * scale + offset`), paired
draws across compared configs, CSV+JSON dual output, and unbuffered progress
logging per `CLAUDE.md`. Multiprocessing `Pool` is used for Monte-Carlo trials
(Windows spawn; requires the process sandbox to allow named pipes).

## Compiling the papers

Both `.tex` files compile standalone with the article class (MiKTeX verified):
`pdflatex PAPER_A.tex` twice, `pdflatex PAPER_B.tex` twice (or `latexmk` if
Perl is installed). At submission, swap in the journal class:
ws-ijbc.cls / revtex4-2 for Paper A (IJBC/Chaos), elsarticle.cls for Paper B
(Neural Networks); fill the `[Author Name]` placeholders.

## Papers (drafted + revised, S10)

- `PAPER_A_draft.md` / `PAPER_A.tex` — substrate characterization: phase
  diagram, forgetting kernel theory (full derivations in Appendix A),
  λ-homeostat robustness; task-level CV sweep moved to Supplementary Note 1 /
  Fig. S1. Target: *International Journal of Bifurcation and Chaos* or *Chaos*.
  `PAPER_A.tex` compiles standalone (latexmk -pdf) with the article class;
  swap in ws-ijbc.cls (IJBC) or revtex4-2 (Chaos) at submission.
- `PAPER_B_draft.md` / `PAPER_B.tex` — REDEM architecture, mechanisms,
  ablations, baselines. Target: *Neural Networks* (Elsevier). Post-review
  revision applied: title without "physics" (S1), differentiators-first
  abstract (S2), compressed negative results (Table 1, S3), corrected
  metadata-transfer conclusion (T1), cross-reference to Paper A §2 (T3),
  Fig. 1 with M4↔M5 coupling loop (T2). `PAPER_B.tex` compiles standalone;
  swap in elsarticle.cls at submission.
- `PAPER_A_sketch.md` / `PAPER_B_sketch.md` — outlines, figure/table
  inventories, key-number tables.
- `paper_c/` (in development) — Paper C "Dissecting Online Learning
  Mechanisms: Statistical Memory, Homeostatic Recovery, and Substrate
  Physics are Non-Transferable". The three-mechanism disentanglement thesis
  is locked on s14/s16/s16b/s15/s5b; `paper_c/PAPER_C.tex` is drafted in
  elsarticle preprint format (13 pp, zero warnings) for Neurocomputing /
  Neural Networks short paper. See `paper_c/DERIVATION.md` and
  `paper_c/PAPER_C_sketch.md`.
- `paper_d/` (in development) — Paper D "REDEM-SSM": the SSM instantiation
  of the three mechanisms (M1 per-token RLS readout, M3 fast-channel EMA
  routing, M4 soft routing, M5 state-norm homeostat). P1–P5 DONE;
  `paper_d/PAPER_D.tex` drafted in elsarticle preprint format (13 pp, zero
  warnings; P1 theorem, reference fairness, M5 negative, oracle integrated).
  See `paper_d/README.md` and `paper_d/PAPER_D_sketch.md`.

## Roadmap status

| Step | Status |
|---|---|
| S0 plan freeze | done |
| S1 phase diagram | done |
| S2 online readout | done |
| S3 three-factor (negative) | done |
| S4 intrinsic reward (negative) | done |
| S5 dual-timescale metadata | done |
| S6 chaos homeostat | done |
| S7 structure plasticity | done |
| S8 integration + ablations + N=1024 | done |
| S9 baseline showdown | done |
| S10 papers | done (drafts + post-review revision pass; all figures complete) |
| E3 disturbance chain | done (10 seeds, +32% MC after 3 sequential disturbances) |
| E4 λ_target sweep | done (5 seeds × 4λ × 3CV, λ=0 optimal with +25% MC) |
| O4 causal audit | done (10 seeds × 7 arms, all mechanisms causally clean; plasticity-correlation leak implemented and verified 2026-02-19) |
| s14 ESN disturbance chain | done (10 seeds × 3 arms; metadata does not transfer MC robustness; +32% is homeostat) |
| s16 τ_m pressure test | done (10 seeds × 4 τ_m; falsification robust, strong claim adopted) |
| s15 controlled adaptation | done (10 seeds × 5 switches; true effect ~10 pulses + variance collapse; "47×" is a metric artifact) |
| s16b probe stress test | done (v2: 10 seeds × 4 τ_m × 3 variants; sign robust ≤1/10 in every cell, V0 −0.69 → V2 −0.04…−0.01) |
| s5b S5 controlled re-measurement | done (100 runs; T200 factor 1.28–1.44, T40 1.79–2.40; Paper B revised) |
| s17 substrate stress | done (120 runs; equalizer gain positive at all ESN configs) |
| s18 LLM drift gate | done (90 runs; A3 routing transfers, A2 gate falsified; §7 results in paper_c/LLM_EXTENSION.md) |
| Paper D s19 | done (70 runs; naive state readout falsified, B-proj input path works 10/10) |
| Paper D s20 | done (90 runs; gating-only inverts on RLS — what-is-paused qualifier; M3 routing transfers 10/10) |
| Paper D s21 | done (70 runs; soft routing beats abrupt; M5 homeostat restores detector 5/5) |
| Paper D s22 | done (30 runs; 4-domain benchmark, REDEM vs bare/TF 10/10) |
| Paper D s23 | done (30 runs; real-text Gutenberg transfer, REDEM vs bare/TF 10/10) |
| s24 M4-M5 coupling | done (40 runs; negative 0/10 — rewiring during disturbance compensation is harmful) |
| s25 reward-gated plasticity | done (40 runs; novelty-guided +2.15 vs correlation, 10/10) |
| s26 fair-TF references | done (90 runs; tuned grid closes stream gap to −1.68 (0/10) but collapses retention; TF-A3 routing transfers −43% forgetting) |
| s27 clip-kappa fine grid | done (1170 runs; κ* = 25.3/27.4/27.9, coupling-driven — invariant to 5× clip widening, ring_bidir excepted) |
| s28 chain-protocol causal audit | done (60 runs; no leak improves recovery; no-plasticity = 8.47 anchor exactly) |
| s30 N=1024 replication | done (20 runs; 10 seeds, paired t=15.3) |
| s31 char-bigram oracle | done (10 seeds; first-order ceiling ppl 10.97 ± 0.18) |
| s32 FTLE-noise robustness | done (50 runs; no significant degradation to σ=0.10) |
| s33 M5 in P4 | done (20 runs; honest negative 10/10 — M5 breaks Δt=1 whitening) |
| s34 leak sensitivity | done (60 runs; FTLE 10× NS; plasticity 30% significant +0.57 — audit resolution bounded) |
| Paper D P5 | done (PAPER_D.tex draft, 13 pp, zero warnings; revision pass 1: A2 qualifier + real-text results; pass 2: P1 theorem + reference fairness + M5 negative + oracle) |
| Paper C derivation | in progress (thesis locked; PAPER_C.tex submission-ready; §7 LLM PoC results available for the extension section) |

## Open items

- Author information (Yaming Hu, ORCID 0009-0003-1406-0485, Independent
  Researcher Guiyang, 64687555@qq.com) is filled in all four papers
  (A/B/C/D); the remaining placeholder-free check is the journal-template
  swap below.
- Swap journal document classes at submission (ws-ijbc / revtex4-2 for A;
  elsarticle for B).
- Paper C: `paper_c/PAPER_C.tex` drafted (elsarticle preprint, 13 pp, zero
  warnings). Venue decided: Neurocomputing / Neural Networks short paper.
  The LLM §7 PoC (s18) is DONE: A3 routing transfers (forgetting −28%,
  ppl −1.07 at τ_m=200), A2 gate falsified; results in
  `paper_c/LLM_EXTENSION.md` §7.7. Remaining before submission: cover
  letter.
- Paper D: `paper_d/PAPER_D.tex` drafted (13 pp, zero warnings). Open:
  title RESOLVED 2026-02-19 (revised to "A State-Space Architecture ...";
  "Foundation Model" dropped as overclaiming for the toy prototype),
  optional related-work paragraph +
  per-mechanism ablation table, cover letter.
- Paper B wording: revised (2026-02-17) - §4.3/§4.5/§4.6 and abstract use
  the controlled s15/s5b adaptation measurements (factor 1.3-2.4 instead
  of the artifact "9-20x").
