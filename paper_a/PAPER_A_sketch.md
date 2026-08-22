# Paper A Sketch: Memory–Chaos Phase Diagram and Multi-Timescale Forgetting of a Physics-Constrained Relaxation Substrate

> Status: S10 draft skeleton v0.1 (numbers from the completed experiments);
> revised per the 8-point review (S4: CV sweep → Supplementary; S5: Appendix A derivations).
> Target journal: *International Journal of Bifurcation and Chaos* (World Scientific)
> or *Chaos* (AIP); single author. Companion: Paper B (REDEM, target: *Neural Networks*).

---

## Working Title

**"Memory and chaos in a physics-constrained relaxation substrate: phase diagram, multi-timescale forgetting, and disturbance robustness"**

## Proposed Abstract (draft)

Reservoir computing exploits the fading-memory dynamics of a physical substrate, but the memory–chaos trade-off is usually studied on abstract recurrent networks with hand-tuned leak rates. We characterize the dynamics of a physics-constrained relaxation substrate inspired by Si3N4 shallow-trap charge storage: N units with a log-normal time-constant spectrum (median τ0 ≈ 174 μs, CV = 0.20) coupled by a per-pulse, topology-dependent modulation of the injection coefficient. Sweeping the coupling strength κ across topology families (ring, hub, lateral-inhibition, random graph) and measuring the finite-time Lyapunov exponent λ, the held-out Jaeger memory capacity MC, and input separation, we find: (i) weak coupling is a negative-feedback stabilizer (λ becomes more negative, states homogenize); (ii) a sharp order–chaos transition at κ* ∈ (25, 30), where the held-out memory capacity peaks 24–53% above the uncoupled baseline just before the transition; (iii) deep chaos destroys memory (MC collapses 3–7×), so the substrate's information processing is bounded by its physical parameter range; (iv) the separation–memory trade-off depends on the coupling sign — positive-feedback (excitatory) topologies self-limit at criticality (|λ| < 0.002 over a decade of κ) with maximal separation but zero linear memory, while negative-feedback topologies trade separation for linear decodability. We further derive the substrate's forgetting kernel M(t) = ∫ p(τ) e^{−t/τ} dτ: the 1/e memory horizon is pinned at ~16 pulses (set by the median τ0, matching the measured N_eff ≈ 17), while the spectrum width CV controls the tail weight (log-log slope from −60 at CV=0.02 to −7 at CV=1.0 — slower than a single exponential yet far from the power-law retention of multi-timescale cascade models). The measured memory-capacity curve follows this kernel with Pearson r = 0.97. Finally, a homeostatic regulator that estimates λ online (Benettin pairs every 1000 pulses) and adjusts κ to a near-critical target improves the post-disturbance held-out memory by 8–18% under temperature drift, edge damage, and readout noise, while a fixed coupling does not.

## Section Outline

### 1. Introduction
- Physical reservoir computing: device substrates as fading-memory kernels (Jaeger 2001; Maass 2002); edge-of-chaos computation (Bertschinger & Natschläger 2004); the memory–nonlinearity bound (Dambre et al. 2012).
- Multi-timescale memory theory: cascade models and power-law retention (Fusi, Drew & Abbott 2005; Benna & Fusi 2016); why physical trap spectra are a natural multi-timescale design knob.
- Gap: reservoir leak rates are hand-tuned scalars; a device-physics spectrum with a *designable width* (CV) has not been characterized as a forgetting kernel.
- Our contribution: (1) a per-pulse temporal generalization of the quasi-static topology coupling previously reported for Si3N4 pulse-encoding (cite the author's prior NCE paper as substrate-calibration source); (2) the first κ–MC–λ phase diagram for a physics-constrained relaxation substrate; (3) the analytic forgetting kernel with experimental validation; (4) a λ-homeostatic regulator robust to disturbances.

### 2. Substrate model
- Device physics: shallow-trap occupancy x_i ∈ [0,1], injection α(1−x), pulse-width and interval relaxation exp(−pw/τ), exp(−Δt/τ); current I = I_HRS·exp(γx) with γ = ln(100); log-normal τ with median τ0 = 174 μs, CV = 0.20 (E_a = 0.55 eV, ν = 10^13 s^-1, T = 300 K). **Eqs. 1-3.**
- Per-pulse contrast coupling: α_eff,i(t) = clip(α0(1 + κ·g_i(t)), α_min, α_max) with g_i a topology contrast of current ratios (mode 1: (nbr_mean − self)/self; mode 2: (self − nbr_mean)/nbr_mean); additive state coupling as a control. **Eqs. 4-6.** Two-pass order-independent update.
- Topology families: ring_bidir, ring_unidir, hub_star, lateral_ring, random_graph (Erdős–Rényi, avg degree 8).

### 3. Characterization methods
- Finite-time Lyapunov exponent: Benettin twin-trajectory method (renormalization every 10 pulses, ε = 1e-8), driven-system convention (per-pulse units). **Eq. 7.**
- Memory capacity: Jaeger MC(k) with ridge readout on current-ratio features, 70/30 chronological held-out split with a k_max leakage buffer; MC_total = Σ_{k=1..50} corr². **Eq. 8.** (Honest held-out protocol; the in-sample MC overstates memory in the chaotic regime.)
- Input separation: inter-stream RMS state distance.
- Activity proxy S: running std of the population-mean current ratio.

### 4. Phase diagram (results of S1)
- **Fig. 2** (existing: `figures/substrate_phase_diagram_v2.png`): κ vs FTLE, held-out MC, train MC, clip fraction.
- Table 1: per-topology κ* brackets and peak held-out MC.
- Findings:
  - Negative-feedback family (mode 1): stabilization for κ ≤ 10 (λ −0.065→−0.090), homogenization (|g| shrinks); transition to chaos at κ* ∈ (25, 30); held-out MC peaks just before the transition (random_graph: 9.07 → 13.91, +53%; ring_bidir +30%; lateral_ring +24%).
  - Positive-feedback family (mode 2): ring_unidir self-limits at criticality via the α_eff clip (λ ≈ −0.002 over κ ∈ [1, 10]), maximal separation (inter-RMS 0.085 vs 0.022 baseline) but zero held-out linear memory (0.12–0.39) — the separation–memory trade-off; hub_star freezes into bistable satellites (no feedback loop).
  - Deep chaos (κ = 50–100) destroys memory (held-out MC 1.85–4.4 vs 9–14 peak): the physical clip bounds make the chaos bounded but input-destructive.
  - Train-MC "explosion" in chaos is an overfitting artifact; the held-out protocol is essential (methodological contribution).
- **Table 2**: full phase-table numbers (from `data/substrate_phase_diagram_v2.csv/json`).

### 5. Multi-timescale forgetting theory (new theory section)
- Kernel: M(t) = ∫ p(τ) e^{−t/τ} dτ, p = log-normal(τ0, CV). **Eq. 9.**
- Gauss–Hermite evaluation; properties:
  - 1/e horizon ≈ τ0/Δt̄ ≈ 16 pulses independent of CV (median-pinned);
  - tail ln M(t) ~ −(ln t − μ)²/(2σ²): log-Gaussian, steeper than power-law, wider CV → heavier tail (log-log slope −60 at CV=0.02 → −7 at CV=1.0);
  - comparison to single-exponential (CV→0), stretched-exponential, and power-law (Benna–Fusi cascade ideal): the log-normal spectrum interpolates but does not reach power-law retention without an extremely wide spectrum.
- **Fig. 3** (new: `figures/forgetting_curve_theory.png`): M(t) curves, tail slopes, Weibull plot, S1 overlay.
- Validation: measured parallel-substrate sqrt(MC(k)) vs M(k·Δt̄): Pearson r = 0.97 (lag 10 nearly exact: 0.520 vs 0.523). **Fig. 3d.**
- Implication: CV is a device-physics design knob for the forgetting curve; the metadata slow-trace of the companion work (Paper B) exploits the same spectrum at a longer timescale.

### 6. Disturbance robustness and the λ-homeostat (results of S6)
- Setup: Mackey-Glass online forecasting (RLS readout) with abrupt disturbances at t = 10k: τ drift ×1.5, 40% edge pruning, readout noise σ = 0.1.
- Part 1: the disturbed MC–κ landscape — the optimal κ shifts with the disturbance (e.g., τ drift moves it right); a fixed activity-proxy target is not disturbance-invariant (motivates the λ-homeostat).
- Part 2: the λ-homeostat (Benettin estimate every 1000 pulses, λ target −0.02, κ step-clipped) settles at κ ≈ 26–27 and achieves post-disturbance held-out MC gains of +8–18% across all disturbance types (none +7.6%, τ drift +18%, edge prune +7.9%, noise +11.8%) at task-level NMSE parity.
- **Fig. 4 / Table 3**: from `data/s6_chaos_regulator_v1.csv/json` (needs a figure script — TODO).
- Self-repair narrative: substrate-level memory quality is restored by adapting the coupling strength, decoupled from the readout.

### 7. Discussion
- The physical parameter range bounds the achievable computation: κ*, the clip range [0.001, 0.10], and the τ spectrum jointly define the accessible memory–chaos envelope.
- Separation vs memory: linear readout benefits from negative-feedback (memory-rich) topologies; separation-rich critical states need nonlinear readouts (motivates the companion paper's online nonlinear readout work — actually RLS is linear; the separation-rich states are exploited in the companion's structure-plasticity work).
- The forgetting kernel as a design language: "device physics as a structured state-space kernel" (S4/Mamba analogy).
- Limitations: single-substrate simulation (no fabrication); CV sweep beyond 0.2 untested on tasks (only kernel-level); the λ-homeostat gains are substrate-level (readout compensates at task level).

## Figure/Table Inventory

| # | Content | Source | Status |
|---|---|---|---|
| Fig 1 | Substrate schematic (units, coupling, readout) | `figures/paperA_fig1_substrate.png` | done |
| Fig 2 | Phase diagram 2×2 (κ–FTLE–MC–clip) | `figures/substrate_phase_diagram_v2.png` | done |
| Fig 3 | Forgetting kernel (4 panels) | `figures/forgetting_curve_theory.png` | done |
| Fig 4 | λ-homeostat robustness (MC gains) | `figures/paperA_fig4_robustness.png` | done |
| Tab 1 | κ* brackets & peak MC per topology | `data/substrate_phase_diagram_v2.json` | draft numbers |
| Tab 2 | Phase table excerpt | `data/substrate_phase_diagram_v2.csv` | draft numbers |
| Tab 3 | Disturbance robustness table | `data/s6_chaos_regulator_v1.json` | draft numbers |

## Key Numbers (for the writing pass)

- MC peak: random_graph κ=25 held-out 13.91 vs parallel 9.07 (+53%); ring_bidir +30%; lateral_ring +24%.
- κ* ∈ (25, 30) for mode-1 family; ring_unidir self-limited λ≈−0.002 over κ ∈ [1,10]; hub_star frozen at clip=1.0.
- Deep chaos (κ=50/100): held-out MC 1.85–4.4.
- Forgetting: horizon ~12–16 pulses (all CV); tail slopes −60.6/−16.9/−7.1 (CV 0.2/0.5/1.0); Pearson r = 0.97 vs S1 MC.
- Robustness: MC +7.6%/+18%/+7.9%/+11.8% (none/tau_drift/edge_prune/noise), κ settled 26–27.

## TODO (writing pass)
- Fig 1 substrate schematic (manual drawing task).
- Literature anchoring pass for the exact citation list.
- CV sweep presentation: **decided** — task-level sweep moved to Supplementary
  Note 1 + Fig. S1 (`data/s10_cv_sweep_v1.csv/json`,
  `figures/paperA_figS1_cv_sweep.png`); main text keeps a one-line summary.
- Full derivations: **done** — Appendix A (A.1 spectrum log-normality, A.2
  two-pass update, A.3 forgetting-kernel quadrature + median pinning + tail
  asymptotics, A.4 Benettin steps, A.5 MC(k) estimator + linear closed form).
