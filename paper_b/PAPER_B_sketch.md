# Paper B Sketch: REDEM — Training-Inference Unified Learning with Meta-Adaptation and Structural Plasticity

> Status: S10 draft skeleton v0.1 (numbers from the completed experiments);
> revised per the 8-point review (S1 title, S2 abstract order, S3 Table 1,
> T1 metadata conclusion, T2 Fig 1 M4↔M5 loop, T3 cross-reference).
> Target journal: *Neural Networks* (Elsevier); single author.
> Companion: Paper A (substrate theory, target: IJBC/Chaos).
> All figures/tables map to files under `data/` / `figures/` (see inventory).

---

## Working Title

**"REDEM: Training-Inference Unified Learning with Meta-Adaptation and Structural Plasticity for Non-Stationary Environments"**
*("physics-constrained" removed from the title per S1; the physical substrate remains as background motivation in §1.)*

## Proposed Abstract (draft)

Classical deep learning separates training from inference and relies on batch gradient computation; continuous learning under distribution drift remains an open problem. We introduce REDEM (REward-gated Dual-timescale Eligibility Mechanism — working name), an online learning architecture built on a physics-constrained relaxation substrate (Si3N4-style shallow traps, log-normal time constants, per-pulse contrast coupling; see the companion characterization paper). REDEM unifies training and inference: an online recursive-least-squares readout updates at every pulse from live prediction error (no BPTT, no separate training phase), while three substrate-level mechanisms operate concurrently — (M3) a slow metadata trace (per-unit exponential moving average of activity) that gives the readout long-horizon statistical memory; (M5) a chaos homeostat that estimates the finite-time Lyapunov exponent online and adjusts the coupling strength to a near-critical target; and (M4) slow, functional-connectivity-guided structure rewiring. On a drifting two-class interval task, the online readout tracks a class-interval inversion in ~600 pulses (accuracy 0.98 over the stream), while frozen batch learners (GRU, tiny transformer) — perfect before the drift — become systematically wrong after it (accuracy 0.00–0.15, never recovering). The dual-timescale metadata improves regime-switch classification by +1.3–2.1 pp with 1.3–2.4× faster adaptation under a controlled switch-relative measurement (paired p < 0.0001). The chaos homeostat restores post-disturbance memory by +8–18%. Gentle (5%) structure rewiring improves memory +8–11% while aggressive rewiring destabilizes (−23%). The integrated system beats every single-mechanism ablation (full 0.996 vs baseline 0.973, p < 0.0001) and the advantage persists at N = 1024. Compared with a matched echo-state network (256 units, heterogeneous leak rates, identical online readout), REDEM is competitive on standard tasks (Mackey-Glass NMSE 0.0018 vs 0.0002; drift accuracy 0.991 vs 0.998) while adding physical plausibility, long-horizon statistical memory, and disturbance robustness that the generic reservoir lacks. Two negative results delimit the design: reward-modulated Hebbian learning without an error signal cannot credit-assign through a mapping inversion (post-swap accuracy collapses to 0.06–0.10 and never recovers), and task-agnostic intrinsic rewards (novelty) cannot rescue it — error-driven second-order learning is necessary at the readout, while intrinsic signals find their proper role in structural exploration.

## Section Outline

### 1. Introduction
- Training-inference unification: online/continual learning, test-time adaptation; the stability-plasticity dilemma; fast weights; predictive coding / free energy (Friston 2010).
- Reservoir computing as a cheap, local substrate: fading memory, echo state property, edge of chaos; online readouts (RLS, LMS) as the simplest "training == inference" learners.
- The gap: (i) readout-only online learning does not adapt the substrate; (ii) reward-modulated Hebbian rules are often proposed as biological credit assignment but fail without an error channel; (iii) multi-timescale memory (metadata) and homeostatic regulation are under-explored in physical reservoirs.
- Contribution: REDEM — the full architecture; three substrate-level mechanisms (M3/M4/M5) validated individually and as ablations; honest benchmarking (online vs frozen; vs matched ESN); two negative results that sharpen the design space.

### 2. System architecture
- **Fig 1**: full-system schematic.
- Substrate: relaxation units, log-normal τ, per-pulse contrast coupling (summary of companion; cite Paper A).
- Readout (M1 readout-layer resolution): OnlineRLS with forgetting λ=0.999, Tikhonov regularization, trace cap; predict-before-update online protocol. **Eq. 1.**
- Metadata (M3): m_i(t) = (1−1/τ_m)m_i(t−1) + (1/τ_m) f_i(t); features = [fast, slow] concatenated; τ_m = 200–1000 pulses. **Eq. 2.**
- Chaos homeostat (M5): Benettin FTLE estimate every 1000 pulses (window 400), κ += η·clip(λ_target − λ, ±1), λ_target = −0.02, κ ∈ [1, 60]. **Eq. 3.**
- Structure plasticity (M4): every 2000 pulses, prune the 5% lowest-|corr| edges and grow the 5% highest-|corr| unconnected pairs (constant edge count; functional-connectivity Hebbian rewiring). **Eq. 4.**
- Reward/error signals: dense prediction error drives the readout; ±1 correctness rewards and novelty intrinsic signals were tested and REJECTED at the readout (S3/S4) — the design rationale section.

### 3. Tasks and metrics
- drift_binary: two-class interval blocks (20 pulses each, 10/60 μs), continuous interval random walk + abrupt class-interval swap every 1000 blocks (40k pulses); metrics: pre/post-swap steady accuracy, adaptation time, stream mean.
- mackey_glass: chaotic forecasting (a=0.2, b=0.1, τ=17); NMSE on last 30%.
- narma10: nonlinear memory composition; NMSE last 30%.
- regime_switch: 3 overlapping-regime classification (rare-event rates 0.12/0.20/0.28, identical marginal intervals); overall accuracy + boundary adaptation.
- All: 10 seeds, paired draws; readout-independent probes (held-out MC) where substrate quality is the target.

### 4. Results

#### 4.1 Online vs frozen: the training-inference unification case (S2, S9)
- **Fig 2 / Tab 1** (from `data/s2_online_readout_v1.csv/json`): drift stream — online RLS tracks (mean 0.982 parallel, 0.974 near-edge substrate; recovers 225–616 pulses after the swap), frozen offline ridge collapses to systematic inversion (post-swap 0.000–0.019, never recovers).
- **Tab 2** (from `data/s9_baseline_showdown_v1.csv/json`, 2026-02-19 revision): REDEM 0.991 vs GRU-frozen 0.394 (pre 0.860 → post 0.145) vs transformer-frozen 0.351 (pre 1.000 → post 0.000) vs ESN-online 1.000. MG NMSE: REDEM 0.0018 vs ESN 3.6e-5 vs GRU 1.56 vs transformer 0.71.
- Near-edge coupling boosts regression: MG 50× (0.0018 vs 0.090 parallel), NARMA +21% (0.431 vs 0.549).

#### 4.2 Negative results that shape the design (S3, S4)
- **Tab 3** (from `data/s3_three_factor_v1.csv/json`): RMHL (reward-only, ±1, no class label) learns the initial mapping (pre 0.89–0.94) but cannot recover from the inversion (post 0.06–0.10, mean 0.509); LMS (error-gated first-order, η=1e-4) tracks (mean 0.908–0.927) but is fragile (η≥1e-3 diverges) and weaker than RLS; RLS-sparse (class at block end) recovers robustly (post 0.93–0.95).
- **Tab 4** (from `data/s4_intrinsic_reward_v1.csv/json`): novelty intrinsic reward × κ_int × reward frequency — never rescues (best post 0.42–0.59 at the cost of destroying initial learning; stream mean never exceeds 0.514); intrinsic signals are re-purposed to structure exploration (M4).
- Conclusion: at the readout, error-driven second-order learning (RLS) is necessary and sufficient; reward/intrinsic gating belongs at the structure level.

#### 4.3 Dual-timescale metadata (M3, S5)
- **Fig 3 / Tab 5** (from `data/s5_dual_timescale_v1.csv/json`): regime task — dual vs fast-only +1.3 pp (parallel, t=19.6, p<0.0001) / +2.1 pp (near-edge, t=20.2); adaptation controlled re-measurement (`data/s5b_controlled_adaptation_v1.*`): T200 265–304 (fast) vs 202–211 (dual), factor 1.3–2.4; slow-only ≈ dual (statistics-dominated task).
- Forgetting-kernel link: the metadata implements a longer-timescale draw from the same log-normal spectrum (companion Paper A).

#### 4.4 Chaos homeostat (M5, S6)
- **Fig 4 / Tab 6** (from `data/s6_chaos_regulator_v1.csv/json`): post-disturbance held-out MC +7.6%/+18%/+7.9%/+11.8% (none/τ-drift/edge-prune/noise) at task-level NMSE parity; κ settles 26–27; the fixed activity-proxy target is not disturbance-invariant (negative design evidence).

#### 4.5 Structure plasticity (M4, S7)
- **Tab 7** (from `data/s7_structure_plasticity_v1.csv/json`): gentle (5%) correlation-guided rewiring +7.8% (ring start) and +11.3% (damage repair); aggressive (20%) −23% (destabilization — slow plasticity matters); pruning the dense random graph HELPS memory (12.68 vs 10.37 — de-homogenization); sparse random beats ring at equal density (13.52 vs 11.54).

#### 4.6 Integration and ablations (S8)
- **Fig 5 / Tab 8** (from `data/s8_integrated_v1.csv/json`): full 0.996 vs baseline 0.973 (p < 0.0001); marginal contributions: metadata +0.78 pp (largest), plasticity +0.21 pp, homeostat ≈0 on the no-disturbance task (its value is in S6's disturbed scenarios); N=1024: full 0.998 vs baseline 0.976.

### 5. Discussion
- The architecture's niche: online, local-rule, no-BPTT learning that adapts to drift and disturbances, on a physically plausible substrate — positioned against both batch deep learning (frozen = fragile under drift) and generic reservoirs (no metadata/homeostasis, no physical narrative).
- Honest limits: ESN beats REDEM on two standard tasks (readout-level parity, reservoir-level deficit on standard metrics); the value proposition is the mechanism set, not raw benchmark supremacy.
- Biological analogy: fast/slow memory (complementary learning systems), homeostatic regulation of excitability, slow structural plasticity — mapped one-to-one onto M3/M5/M4.
- Limitations: CPU-scale only; single-substrate simulation; reward signals without error channels fail (documented); the homeostat gains are substrate-level.

### 6. Conclusion
- A full training-inference-unified architecture on a physics substrate; three validated mechanisms; two negative results that delimit reward-only learning; honest benchmarking vs batch and reservoir baselines.

## Figure/Table Inventory

| # | Content | Source | Status |
|---|---|---|---|
| Fig 1 | REDEM system schematic | `figures/paperB_fig1_redem.png` | done |
| Fig 2 | Drift tracking curves (S2) | `figures/s2_online_readout_v1.png` | done |
| Fig 3 | Metadata regime results | `figures/paperB_fig3_metadata.png` | done |
| Fig 4 | Homeostat MC gains | `figures/paperB_fig4_robustness.png` (= paperA_fig4) | done |
| Fig 5 | Ablation curves | `figures/paperB_fig5_ablation.png` | done |
| Fig 6 | Showdown (S9) | `figures/paperB_fig6_showdown.png` | done |
| Tab 9 | ESN-metadata transfer (S10) | `data/s10_esn_metadata_v1.json` | done |
| Tab 1 | Online vs offline (S2) | `data/s2_online_readout_v1.json` | draft |
| Tab 2 | Showdown (S9) | `data/s9_baseline_showdown_v1.json` | draft |
| Tab 3 | Three-factor (S3) | `data/s3_three_factor_v1.json` | draft |
| Tab 4 | Intrinsic reward (S4) | `data/s4_intrinsic_reward_v1.json` | draft |
| Tab 5 | Dual-timescale (S5) | `data/s5_dual_timescale_v1.json` | draft |
| Tab 6 | Homeostat (S6) | `data/s6_chaos_regulator_v1.json` | draft |
| Tab 7 | Plasticity (S7) | `data/s7_structure_plasticity_v1.json` | draft |
| Tab 8 | Ablation (S8) | `data/s8_integrated_v1.json` | draft |

## Key Numbers

- Drift: online mean acc 0.974–0.982 (adapt 225–616 pulses); frozen post-swap 0.000–0.019.
- Showdown: REDEM 0.991 / ESN 1.000 / GRU 0.394 / trans 0.351 (drift); MG NMSE REDEM 0.0018 / ESN 3.6e-5 / GRU 1.56 / trans 0.71.
- RMHL negative: pre 0.89–0.94 → post 0.06–0.10, mean 0.509; LMS fragile (η window ~1e-4).
- Intrinsic negative: stream mean never > 0.514; κ_int ≥ 0.5 destroys initial learning.
- Metadata: +1.3–2.1 pp (p<0.0001), adaptation factor 1.3–2.4× (controlled re-measurement s5b/s15; the old "9–20×" was a window-position metric artifact).
- Homeostat: MC +7.6/+18/+7.9/+11.8%; κ settles 26–27.
- Plasticity: +7.8% (gentle), −23% (aggressive), repair +11.3%.
- Integration: full 0.996 vs baseline 0.973 (p<0.0001); N=1024 0.998 vs 0.976.

## TODO (writing pass)
- System schematic (Fig 1, manual drawing task): **done** — M4↔M5 coupling loop added (T2), verified by pixel check.
- Full intro literature pass.
- Decide naming: "REDEM" working name vs a final name (D5 decision from S0 — user's choice).
- Journal split: **decided** — Paper A → IJBC/Chaos; Paper B → *Neural Networks*.
- Negative results: **compressed** to Table 1 + 3 sentences (S3); data retained in `data/s3_three_factor_v1.*` / `data/s4_intrinsic_reward_v1.*`.
- Metadata-transfer conclusion: **rewritten** (T1) — mechanism is substrate-agnostic; REDEM's differentiation is the mechanism set (self-regulation, robustness, local sparse coupling, material-set spectrum); the "ESN needs BPTT" claim in the draft review was corrected as factually wrong.
