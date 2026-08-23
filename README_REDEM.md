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
| S9 | Online (REDEM 0.991, ESN 0.998) vs frozen batch (GRU 0.371, transformer 0.351, inverted post-swap); ESN edge on standard tasks reported honestly |
| Theory | Forgetting kernel M(t)=∫p(τ)e^{−t/τ}dτ: 1/e horizon ≈16 pulses (median-pinned); tail steepness set by CV; measured MC curve follows it with r = 0.97 |
| S10-CV | Task-level CV sweep: uncoupled MC rises with CV (+33%, kernel theory); coupled near-critical MC falls (narrow CV best) — operating-regime-dependent knob |
| S10-ESN | Metadata transfer: ESN+meta 0.998 ≈ ESN 0.996 ≈ REDEM-full 0.994 on regime task — the mechanism equalizes the systems (adapt 11.22→0.24 in the S10 metric; controlled s15 shows the true effect is ~10 pulses + variance collapse) |
| E3 | Sequential disturbance chain (3 rounds: τ-drift → edge-prune → noise, 10 seeds): regulated arm maintains MC 8.47 vs fixed 6.41 (+32%) after all three; κ drifts 26.2→28.5 (active compensation) |
| E4 | λ_target sweep (4 values × 3 CV × 5 seeds): λ_target=0 (edge of chaos) is optimal — +25%/+19%/+5% MC gain over fixed at CV=0.1/0.2/0.4; monotonic improvement as λ→0 |
| O4 | Causal audit (7 arms × 3 seeds): all adaptive mechanisms are causally clean — injecting 1% future data changes accuracy <0.02 pp; causal-split plasticity protocol within 0.01 pp of normal |
| s14 | ESN+metadata under the S11 disturbance chain (3 arms × 10 seeds): the slow trace does NOT transfer MC robustness to the ESN (paired diffs −0.78/−0.76/−0.69, 0/10 seeds positive) — the S11 +32% recovery is homeostat-driven; metadata still attenuates readout noise (r3 NMSE −9.5%, 10/10 seeds); redem_reg reproduces the S11 anchor exactly |
| s16 | τ_m pressure test (τ_m ∈ {200,500,1000,2000} × 10 seeds): the falsification is robust — esn_dual r3 MC ≤ esn_fast at every τ_m (0/10 seeds positive, ~5σ), no sensitive interval; noise-attenuation transfer holds at all τ_m. Paper C adopts the strong claim |
| s15 | Controlled adaptation (10 seeds × 5 switches, known switch instants): T40 40.6 (dual) vs 49.9 (fast) vs 52.7 (redem); T40 p90 42 vs 76.5 — the true metadata adaptation effect is ~10 pulses + variance collapse; the "47×"/"9–20×" ratios are metric artifacts |
| s16b | Probe-protocol stress test (10 seeds × 2 τ_m × 3 variants): falsification robust in sign (0/10 positive everywhere); magnitude protocol-dependent — V0 (readout noise) −0.69, V1 (std-slow) −0.66, V2 (state noise) −0.01 (EMA denoising nearly closes the gap) |
| s5b | S5 three-arm controlled re-measurement (2 substrates × 5 arm configs × 10 seeds): T200 fast 264.8–303.9 vs dual/slow 202–211 pulses (factor 1.28–1.44; T40 factor 1.79–2.40); overall acc reproduces S5 exactly — Paper B §4.3/abstract revised |
| s17 | Substrate stress (3 spectral radii × 2 hetero × 10 seeds): equalizer gain positive at all 6 ESN configs (+0.1 to +1.0 pp), magnitude tracks the fast channel's timescale starvation; heterogeneity irrelevant |
| s18 | LLM drift-gate PoC (tiny transformer + LoRA, 90 runs): A3 domain routing transfers — forgetting −2.5 ppl (10/10 seeds, τ_m=200), stream ppl −1.07 (10/10) at τ_m≤500; A2 gate falsified (0/10); τ_m≥1000 sensitive interval reported |
| s19 | Paper D P1/P3a (diagonal SSM, per-token RLS, 70 runs): naive state readout falsified — state arms 58.5–115.9 ppl (0/10) vs B-projection input path 11.75 (d −3.26, 10/10); MLE ceiling probe 7.25 vs 7.35 B-proj oracle |
| s20 | Paper D P2 (M3 routing, 90 runs): gating-only inverts on the RLS host (stream −1.52…−2.45, 10/10 — what-is-paused qualifier); M3 routing transfers (forget −2.05/−1.87/−1.20 at τ_m≤1000, 10/10) |
| s21 | Paper D P3 (M4 soft routing + M5 homeostat, 70 runs): soft 8.22 vs abrupt 10.03 (−1.82, 10/10); dormant-covariance refresh flips routing to −1.72; M5 bounds whitened norm 11.3 vs 50.2 and restores the full-state EMA detector 5/5 |
| s22 | Paper D P4 benchmark (4-domain irregular switches, 30 runs): REDEM 13.18/8.93 vs bare SSM 15.43/13.41 vs TF-A1 22.46/19.01 (stream/forget) — all 10/10 |
| s23 | Paper D real-text transfer (two Gutenberg books, 30 runs): REDEM 12.07/11.85 vs bare 13.34/12.31 vs TF 17.31/13.86 — stream −1.27 vs bare (10/10), −5.23 vs TF (10/10) |

## Scripts (43 committed)

The authoritative script inventory — every script with its `CLAUDE.md` type,
**paper attribution (A/B/C/D/shared)**, purpose, and key result — is the
table in the root [`README.md`](README.md) (single source of truth; it is
kept in sync with `git ls-files scripts/`). Paper D experiments s19–s23 and
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
  `data/s17_substrate_stress_v1.*`, `data/s18_llm_drift_gate_v1.*`
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
# Paper D experiments (s19–s23) + figures — see paper_d/README.md
& .venv\Scripts\python.exe scripts\s19_ssm_rls_readout.py --sequential
& .venv\Scripts\python.exe scripts\s20_ssm_m3_routing.py --sequential
& .venv\Scripts\python.exe scripts\s21_ssm_m4_m5.py --sequential
& .venv\Scripts\python.exe scripts\s22_ssm_p4_benchmark.py --sequential
& .venv\Scripts\python.exe scripts\s23_ssm_p4_realtext.py --sequential
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
  `paper_d/PAPER_D.tex` drafted in elsarticle preprint format (11 pp, zero
  warnings). See `paper_d/README.md` and `paper_d/PAPER_D_sketch.md`.

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
| O4 causal audit | done (3 seeds × 7 arms, all mechanisms causally clean) |
| s14 ESN disturbance chain | done (10 seeds × 3 arms; metadata does not transfer MC robustness; +32% is homeostat) |
| s16 τ_m pressure test | done (10 seeds × 4 τ_m; falsification robust, strong claim adopted) |
| s15 controlled adaptation | done (10 seeds × 5 switches; true effect ~10 pulses + variance collapse; "47×" is a metric artifact) |
| s16b probe stress test | done (10 seeds × 2 τ_m × 3 variants; sign robust, magnitude protocol-dependent) |
| s5b S5 controlled re-measurement | done (100 runs; T200 factor 1.28–1.44, T40 1.79–2.40; Paper B revised) |
| s17 substrate stress | done (120 runs; equalizer gain positive at all ESN configs) |
| s18 LLM drift gate | done (90 runs; A3 routing transfers, A2 gate falsified; §7 results in paper_c/LLM_EXTENSION.md) |
| Paper D s19 | done (70 runs; naive state readout falsified, B-proj input path works 10/10) |
| Paper D s20 | done (90 runs; gating-only inverts on RLS — what-is-paused qualifier; M3 routing transfers 10/10) |
| Paper D s21 | done (70 runs; soft routing beats abrupt; M5 homeostat restores detector 5/5) |
| Paper D s22 | done (30 runs; 4-domain benchmark, REDEM vs bare/TF 10/10) |
| Paper D s23 | done (30 runs; real-text Gutenberg transfer, REDEM vs bare/TF 10/10) |
| Paper D P5 | done (PAPER_D.tex draft, 11 pp, zero warnings; revision pass 1: A2 qualifier + real-text results) |
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
- Paper D: `paper_d/PAPER_D.tex` drafted (11 pp, zero warnings). Open:
  title wording (user decision — "Foundation Model Architecture" may
  overclaim for the toy prototype), optional related-work paragraph +
  per-mechanism ablation table, cover letter.
- Paper B wording: revised (2026-02-17) - §4.3/§4.5/§4.6 and abstract use
  the controlled s15/s5b adaptation measurements (factor 1.3-2.4 instead
  of the artifact "9-20x").
