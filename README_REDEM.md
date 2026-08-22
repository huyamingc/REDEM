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
| M5 Chaos homeostat | Benettin FTLE estimate every 1000 pulses → κ step toward λ_target = −0.02 | keeps substrate near the memory-optimal subcritical point |
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
| S5 | Dual-timescale metadata: regime task +1.3–2.1 pp (p<0.0001), 9–20× faster boundary adaptation |
| S6 | λ-homeostat: post-disturbance held-out memory +8–18% (τ-drift +18%) |
| S7 | Gentle (5%) correlation-guided rewiring +8–11%; aggressive (20%) −23%; pruning redundancy helps (de-homogenization) |
| S8 | Integrated system beats every ablation: 0.996 vs 0.973 (p<0.0001); persists at N=1024 (0.998 vs 0.976) |
| S9 | Online (REDEM 0.991, ESN 0.998) vs frozen batch (GRU 0.371, transformer 0.351, inverted post-swap); ESN edge on standard tasks reported honestly |
| Theory | Forgetting kernel M(t)=∫p(τ)e^{−t/τ}dτ: 1/e horizon ≈16 pulses (median-pinned); tail steepness set by CV; measured MC curve follows it with r = 0.97 |
| S10-CV | Task-level CV sweep: uncoupled MC rises with CV (+33%, kernel theory); coupled near-critical MC falls (narrow CV best) — operating-regime-dependent knob |
| S10-ESN | Metadata transfer: ESN+meta 0.998 ≈ ESN 0.996 ≈ REDEM-full 0.994 on regime task — the mechanism equalizes the systems |

## Scripts (22 committed; 2 legacy dependencies kept for compatibility)

| Script | Type | Purpose |
|---|---|---|
| `shallow_trap_array_simulator.py` | CORE (legacy) | Si₃N₄ shallow-trap device simulator from the prior paper; exports γ, τ₀, gen_tau_vec, preprogram_vec — imported by 12 scripts; never modified |
| `fair_esn_comparison.py` | ML (legacy) | matched ESN reservoir class from the prior paper; imported by `baseline_showdown.py`, `esn_metadata_comparison.py` |
| `recurrent_substrate.py` | CORE | per-pulse contrast-coupled relaxation substrate (numba core) |
| `substrate_recurrence_characterization.py` | PAPER | S1: FTLE / held-out MC / separation vs κ sweep (610 runs) |
| `gen_substrate_phase_diagram.py` | FIG | S1 phase diagram figure |
| `streaming_tasks.py` | CORE | task generators: drift_binary, narma10, mackey_glass, context_switch, regime_switch |
| `online_readout.py` | CORE | OnlineRLS, ThreeFactorReadout, ridge_fit, MC/accuracy metrics |
| `online_readout_streaming.py` | PAPER | S2: online RLS vs offline ridge on streaming tasks |
| `gen_s2_curves.py` | FIG | S2 learning curves |
| `three_factor_online_readout.py` | PAPER | S3: RMHL vs error-gated vs RLS (sparse/dense) |
| `intrinsic_reward_experiment.py` | PAPER | S4: novelty intrinsic × κ_int × reward-frequency ablation |
| `dual_timescale_metadata.py` | PAPER | S5: fast/dual/slow readouts on regime-switch |
| `chaos_regulator.py` | PAPER | S6: λ-homeostat vs fixed κ under disturbances |
| `structure_plasticity.py` | PAPER | S7: functional-connectivity rewiring vs fixed topologies |
| `integrated_benchmark.py` | PAPER | S8: full system vs ablations; N=1024 confirmation |
| `baseline_showdown.py` | ML | S9: vs ESN / GRU / tiny transformer (torch CPU) |
| `forgetting_curve_theory.py` | EXPLORE | S10 theory: M(t) kernel, Gauss–Hermite, r=0.97 validation |
| `esn_metadata_comparison.py` | PAPER | S10: ESN with/without metadata vs REDEM on regime task |
| `cv_sweep.py` | PAPER | S10: task-level CV sweep (CV∈{0.1,0.2,0.4} at optimal κ) |
| `gen_architecture_schematic.py` | FIG | Paper Fig 1 schematics (substrate / REDEM; M4↔M5 loop) |
| `gen_paperA_supp_figures.py` | FIG | Paper A Supplementary Fig. S1 (task-level CV sweep) |
| `gen_paper_figures.py` | FIG | Paper figure batch (robustness / metadata / ablation / showdown) |

## Data and figures

- Results: `data/substrate_phase_diagram_v2.*`, `data/s2_online_readout_v1.*`,
  `data/s3_three_factor_v1.*`, `data/s4_intrinsic_reward_v1.*`,
  `data/s5_dual_timescale_v1.*`, `data/s6_chaos_regulator_v1.*`,
  `data/s7_structure_plasticity_v1.*`, `data/s8_integrated_v1.*`,
  `data/s9_baseline_showdown_v1.*`, `data/forgetting_curve_theory.csv`,
  `data/s10_esn_metadata_v1.*`, `data/s10_cv_sweep_v1.*`
  (CSV per run + JSON with params and per-cell aggregates).
- Figures: `figures/substrate_phase_diagram_v2.pdf`,
  `figures/s2_online_readout_v1.pdf`, `figures/forgetting_curve_theory.pdf`,
  `figures/paperA_fig1_substrate.pdf`, `figures/paperA_fig4_robustness.pdf`,
  `figures/paperA_figS1_cv_sweep.pdf` (Supplementary Fig. S1),
  `figures/paperB_fig1_redem.pdf`, `figures/paperB_fig3_metadata.pdf`,
  `figures/paperB_fig5_ablation.pdf`, `figures/paperB_fig6_showdown.pdf`
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
& .venv\Scripts\python.exe scripts\baseline_showdown.py
# theory + figures
& .venv\Scripts\python.exe scripts\forgetting_curve_theory.py
& .venv\Scripts\python.exe scripts\esn_metadata_comparison.py
& .venv\Scripts\python.exe scripts\cv_sweep.py
& .venv\Scripts\python.exe scripts\gen_architecture_schematic.py
& .venv\Scripts\python.exe scripts\gen_paper_figures.py
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

## Open items

- Fill author name / affiliation / email placeholders in both `.tex` files.
- Swap journal document classes at submission (ws-ijbc / revtex4-2 for A;
  elsarticle for B).
- Final algorithm name (working name REDEM).
- Possible follow-ups: ESN-with-metadata fair comparison; S4's forward-model
  intrinsic variant.
