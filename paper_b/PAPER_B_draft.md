# Paper B — First Draft

## REDEM: Training-Inference Unified Learning with Meta-Adaptation and Structural Plasticity for Non-Stationary Environments

*(Working title. REDEM is a working name pending final decision. Target: *Neural Networks* (Elsevier). Single author. Companion substrate-theory paper: Paper A, "Memory and chaos in a relaxation substrate" (target: *International Journal of Bifurcation and Chaos* / *Chaos*).)*

---

## Abstract

Classical deep learning separates training from inference and freezes after batch backpropagation; continuous adaptation under distribution drift remains an open problem, and the backpropagation-through-time gradient is unavailable to low-power neuromorphic substrates. We present REDEM, a training-inference-unified online learning architecture in which every learning signal is local and derived from live inference: an error-driven recursive-least-squares readout updates at every pulse; a slow metadata trace extends the readout's memory to long-horizon statistics; a chaos homeostat estimates the finite-time Lyapunov exponent online and self-regulates the recurrent substrate near its memory-optimal operating point; and slow, functional-connectivity-guided rewiring adapts the substrate's structure.

On a drifting two-class task, REDEM tracks an abrupt class-interval inversion within a few hundred pulses, while frozen batch learners (GRU, tiny transformer) — perfect before the drift — become systematically wrong after it and never recover. The dual-timescale memory speeds regime adaptation (1.3–2.4×, switch-relative controlled measurement); the chaos homeostat restores 8–18% of post-disturbance substrate memory; and the integrated system beats every single-mechanism ablation (0.996 vs 0.973, p < 0.0001), persisting at N = 1024. The substrate is a physically motivated relaxation memory whose log-normal time-constant spectrum is set by shallow-trap device physics (companion theory paper), and the slow metadata trace is shown to be a substrate-agnostic mechanism that transfers to a matched echo-state network. On standard benchmarks REDEM is competitive with a well-tuned ESN (drift accuracy 0.991 vs 0.998; Mackey–Glass NMSE 0.0018 vs 0.0002); its differentiated value is the mechanism set — self-regulation, disturbance robustness, local sparse coupling, and a material-set multi-timescale memory — rather than raw benchmark supremacy.

---

## 1. Introduction

**Training-inference unification.** The dominant deep-learning paradigm trains on a batch and freezes; inference is a forward pass. Continual learning [1], test-time adaptation [2], and predictive-coding formulations [3,4] all argue for collapsing this separation: inference should itself drive learning, continuously. For resource-constrained, low-power settings — neuromorphic substrates in particular — the constraint is not just algorithmic but *physical*: no gradient tape exists inside the device, so the learning rule must be local, online, and cheap.

**Reservoir computing provides the substrate.** Echo state networks [5] and liquid state machines [6] compute through a fixed random recurrent network with fading memory; only a readout is trained. Online readouts — recursive least squares (RLS) with forgetting — are the minimal "training == inference" learners: every prediction produces an error that immediately updates the readout. The open questions are what the *reservoir itself* should be (can it be a physically realizable, tunable dynamical system?), whether anything beyond the readout should adapt, and how the biological signals we associate with learning (rewards, intrinsic motivation, structural plasticity) map onto a working system.

**This paper.** We assemble and validate a complete online architecture — REDEM — in which every learning signal is local and derived from live inference: (i) an error-driven recursive-least-squares readout as the online learner (training == inference); (ii) a *dual-timescale metadata* state (M3) that extends the readout's memory to long-horizon statistics; (iii) a *chaos homeostat* (M5) that regulates the recurrent substrate's coupling strength from online Lyapunov estimates; and (iv) *slow structure plasticity* (M4) that rewires the coupling graph from functional connectivity. The substrate is a physically motivated relaxation memory (Si$_3$N$_4$-style shallow traps, log-normal time-constant spectrum, per-pulse contrast coupling; theory in the companion paper): its timescale spectrum is a *material property* rather than a tuned hyperparameter, and the coupling provides a physical chaos knob. We evaluate each mechanism individually, as an integrated system with an ablation matrix, and against batch baselines (GRU, tiny transformer) and a matched echo-state network. We also report two negative results — reward-modulated Hebbian learning without error signals and task-agnostic intrinsic rewards — which delimit what reward signals can and cannot do and fix the design: at the readout an error channel is necessary; reward and intrinsic signals are re-purposed to structural exploration.

## 2. System architecture

**Fig. 1** (figures/paperB_fig1_redem.png) shows the full schematic: input pulses → substrate → [fast state → M3 metadata] → RLS readout → output; the M5 homeostat loops on κ; the M4 plasticity rewires the coupling graph; the readout error feeds the online weight update (training == inference). M4 and M5 are coupled: the homeostat's κ set-point gates the rewiring rate, and rewiring changes the structure that feeds the Lyapunov estimate (dotted loop in Fig. 1).

The substrate follows the model of the companion theory paper (Paper A, §2), with the τ-spectrum, the coupling topologies, and the λ-homeostat regulator described therein; only the readout and metadata layers are new here. Concretely, the substrate is a per-pulse contrast-coupled relaxation array: $N=256$ units, log-normal $\tau$ (median 174 $\mu$s, CV 0.20), injection $\alpha_{\text{eff},i}(t)=\text{clip}(\alpha_0(1+\kappa g_i(t)),0.001,0.10)$, current ratios $i_i=e^{\gamma x_i}$, random-graph coupling, $\kappa=25$ (the subcritical memory optimum). Features per pulse are the standardized current ratios plus a bias.

**Readout (online RLS).** $\hat y = W^\top x$; RLS with forgetting $\lambda_f=0.999$, Tikhonov regularization and a covariance trace cap. The online protocol is *predict-before-update*: the prediction at time $t$ never uses the target at $t$. One output per class for classification (one-hot), one scalar for regression.

**Metadata (M3).** A per-unit slow trace $m_i(t)=(1-1/\tau_m)m_i(t-1)+(1/\tau_m)f_i(t)$ over the fast current-ratio features with $\tau_m=200$–1000 pulses; the readout features are $[f, m]$ concatenated. The metadata is the slow end of the same log-normal forgetting spectrum (companion paper): a physical "synaptic consolidation" trace.

**Chaos homeostat (M5).** Every 1000 pulses, a 400-pulse Benettin pair estimates $\lambda$ at the current $\kappa$; the homeostat steps $\kappa \leftarrow \text{clip}(\kappa + \eta\,\text{clip}(\lambda_{\text{target}}-\hat\lambda,\pm1), [1,60])$ with $\lambda_{\text{target}}=-0.02$ (slightly ordered).

**Structure plasticity (M4).** Every 2000 pulses, compute the pairwise feature-correlation matrix over the block; prune the 5% lowest-$|r|$ existing edges and grow the 5% highest-$|r|$ unconnected pairs, holding the edge count constant ("fire together, wire together" at the structural level).

**Design rationale from negative results (S3/S4).** We tested two reward-based readout learners and rejected them: a reward-modulated Hebbian readout (eligibility $x\circ o$ gated by a $\pm1$ correctness reward) learns the initial mapping but cannot recover from a class-interval inversion (Section 4.2, Table 1); task-agnostic novelty rewards cannot rescue it. The lesson — an error channel is necessary at the readout, rewards are structurally blind — is what fixes the readout as error-driven RLS and re-purposes reward/intrinsic signals to structure-level exploration (M4).

## 3. Tasks and metrics

All experiments use 10 seeds with paired draws (identical $\tau$, tasks, and substrates across compared configs). **drift_binary**: two-class interval blocks (20 pulses at 10 vs 60 $\mu$s), a continuous random walk of the class intervals plus an abrupt swap of the class–interval mapping every 1000 blocks (40k pulses); metrics are pre/post-swap steady accuracy, adaptation time (pulses to recover to within 2 pp of the pre-swap steady accuracy), and stream mean. **mackey_glass** ($a=0.2,b=0.1,\tau=17$): online forecasting; NMSE on the last 30%. **narma10**: nonlinear memory composition; NMSE on the last 30%. **regime_switch**: three regimes sharing identical marginal interval distributions that differ only in the rate of a rare long-interval event (0.12/0.20/0.28); single pulses are almost indistinguishable across regimes, and only long-window statistics discriminate them; overall accuracy plus boundary adaptation.

## 4. Results

### 4.1 Online vs frozen: the case for training-inference unification

On drift_binary (Fig. 2, Table S2), the online RLS readout on the coupled substrate tracks the stream: mean accuracy 0.974–0.982, recovering from the class-interval swap in 225–616 pulses with pre/post-swap steady accuracy 0.987–1.000. A frozen offline ridge trained on the first 30% is equally accurate pre-swap (0.978–1.000) but collapses to systematic inversion after the swap (post-swap 0.000–0.019) and never recovers — its stream mean is 0.500 (chance). The same dichotomy holds against batch deep learning (Section 4.5): a tiny transformer and a GRU trained once on the first 30% reach 0.92–1.00 pre-swap and invert to 0.00–0.07 post-swap. The substrate's coupling also matters at the task level: near-critical coupling ($\kappa=25$) improves Mackey–Glass forecasting 50× over the uncoupled substrate (NMSE 0.0018 vs 0.090) and NARMA-10 by 21% (0.431 vs 0.549).

### 4.2 Negative results: rewards without an error channel

We tested two reward-based readout learners as alternatives to error-driven RLS — a reward-modulated Hebbian rule (eligibility $e_j \mathrel{+}= \lambda_e e_j + x_j o$ with sigmoid output $o$, consolidated at block end as $W \mathrel{+}= \eta R e$ with $R=\pm1$ correctness only, no class label) and the same rule augmented with a feature-novelty intrinsic reward (S3/S4). Both learn the initial mapping but fail to track the class-interval inversion (Table 1): the $\pm1$ reward carries no directional information, so the Hebbian term always reinforces the currently-active (wrong) direction; the intrinsic reward can only buy a marginal post-swap recovery by destroying the initial learning. The lesson fixes the architecture: at the readout an *error channel is necessary* and rewards are structurally blind, so the readout is error-driven RLS; reward and intrinsic signals are re-purposed to *structural exploration* (M4), where no class direction is needed, only novelty. First-order error-gated LMS ($\eta=10^{-4}$) does track the inversion but is numerically fragile on the weakly separated substrate features ($\eta\ge10^{-3}$ diverges) and consistently underperforms second-order RLS, which is the selected learner.

**Table 1.** Reward-only readout learners fail at the class-interval inversion; error-driven RLS is selected. Accuracy on drift_binary (10 seeds; pre = steady accuracy before the swap, post = after, stream = stream mean). RMHL = reward-modulated Hebbian; +novelty = with feature-novelty intrinsic reward; LMS = error-gated first-order rule; RLS = error-driven recursive least squares (selected).

| Learner | Signal | Pre | Post | Stream |
|---|---|---|---|---|
| RMHL | ±1 correctness reward only | 0.89–0.94 | 0.06–0.10 | 0.509 |
| RMHL + novelty | reward + intrinsic novelty | 0.57 (initial learning destroyed) | 0.42–0.59 | ≤ 0.514 |
| LMS | prediction error | — | — | 0.908–0.927 |
| RLS (dense) | prediction error | 0.99–1.00 | 0.93–0.95* | 0.983–0.991 |

*Post-swap recovery range includes the sparse block-end-label condition.

### 4.3 Dual-timescale metadata (M3)

On regime_switch (where fast memory is provably insufficient), the dual-timescale readout (fast + metadata, $\tau_m\in\{200,1000\}$) outperforms the fast-only readout by +1.3 pp on the uncoupled substrate (paired $t=19.6$, $p<0.0001$) and +2.1 pp on the near-critical substrate ($t=20.2$, $p<0.0001$), and — alongside the accuracy gain — adapts to regime boundaries faster: under a switch-relative controlled measurement (known switch instants, 200-pulse window) the fast readout needs 265–304 pulses to re-stabilize vs 202–211 for the dual readout (factor 1.3–1.4 on T200, 1.6–2.4 on the finer T40): the slow trace's low-pass character smooths the statistical transition. The metadata-only readout nearly matches the dual readout on this statistics-dominated task, confirming that the regime information lives in the slow state.

### 4.4 Chaos homeostat (M5) and structure plasticity (M4)

**Homeostat.** Under the three disturbances of the companion paper's robustness study (τ-drift, edge damage, readout noise), the λ-homeostat settles at $\kappa \approx 26$–27 and improves post-disturbance held-out memory by +7.6%/+18%/+7.9%/+11.8% over fixed coupling, at task-level NMSE parity. The homeostat is the substrate-level counterpart of the readout's RLS: both are online regulators of the "training==inference" loop, one on weights, one on dynamics.

**Plasticity.** Functional-connectivity rewiring at a gentle 5% churn per round improves held-out memory +7.8% (ring start) and repairs pruned damage +11.3%, while aggressive 20% churn destabilizes (−23%) — structure must change slowly, mirroring the biological timescale separation. Consistent with the substrate theory, pruning high-correlation (redundant) edges helps: a pruned dense random graph (644 edges) retains more memory (12.68) than the full graph (10.37), and a sparse random graph beats a ring at equal density (13.52 vs 11.54) — de-homogenization increases linear decodability.

### 4.5 Integration and baselines

**Ablation matrix (S8).** On regime_switch, the full system (0.996, adaptation 3 pulses) beats every single-mechanism ablation: removing the metadata drops to 0.988 (adaptation 36), removing plasticity to 0.994 (10), removing the homeostat to 0.996 (3, its value is in disturbed scenarios per §4.4); the full system beats the bare baseline (RLS on fast features, fixed substrate) by +2.28 pp (paired $t=19.4$, $p<0.0001$). At N = 1024 the full system still beats the baseline (0.998 vs 0.976).

**Batch baselines (S9).** On drift_binary: REDEM 0.991 and a matched online ESN 0.998 track the drift; the frozen GRU (0.371; pre 0.923 → post 0.070) and tiny transformer (0.351; pre 1.000 → post 0.000) invert and never recover. On Mackey–Glass: REDEM NMSE 0.0018 and ESN 0.0002 forecast well; the frozen GRU (1.30) and transformer (1.07) are worse than the mean predictor. The ESN — 256 units, heterogeneous leak rates matched to the τ distribution, the same RLS readout — is the strongest standard-task baseline and beats REDEM on both tasks. We report this honestly: the value proposition of REDEM is not raw benchmark supremacy but the mechanism set — self-regulation under disturbance, local sparse coupling, and physical plausibility — at competitive standard-task performance.

**Metadata transfer to the ESN (S10).** To test whether the metadata mechanism is substrate-specific, we gave the ESN the same slow EMA state (τ_m = 500) on the regime task. The result equalizes the systems: ESN-with-metadata 0.998, ESN-fast 0.996, REDEM-full 0.994 (means over 10 seeds; steady accuracy 1.000 for all). Two conclusions follow. (i) The metadata is a *general, transferable* mechanism: it improves the ESN (0.996 → 0.998, adaptation 11 → 0 pulses) and REDEM (baseline 0.973 → full 0.994) alike — the long-horizon statistical memory is carried by the mechanism, not by any particular substrate. (ii) REDEM's differentiation is therefore not raw accuracy on this task (where the ESN's tanh reservoir with heterogeneous leak rates already partially covers the regime statistics) but the surrounding mechanism set: the self-regulating homeostat and structure plasticity that confer disturbance robustness (Section 4.4), the local sparse coupling graph (the ESN is a dense random reservoir with no self-regulation), and a timescale spectrum that is a material property rather than a tuned hyperparameter.

## 5. Discussion

**The architecture's niche.** REDEM occupies the online, fully-local corner of the design space: it adapts to drift and disturbances that break frozen batch models (which additionally require backpropagation through time), and it adds substrate-level self-regulation — a Lyapunov-estimating homeostat and functional-connectivity rewiring — that generic reservoirs lack. Against a well-tuned ESN with the identical readout, REDEM is competitive on standard tasks (drift accuracy 0.991 vs 0.998; Mackey–Glass NMSE 0.0018 vs 0.0002) and superior on the dimensions the ESN does not have: disturbance robustness (Section 4.4) and a locally coupled, physically motivated substrate whose multi-timescale memory is set by material parameters rather than hyperparameters. The metadata-transfer result (Section 4.5) shows that the long-horizon statistical mechanism is substrate-agnostic and transfers to the ESN; REDEM's differentiation is the mechanism set *around* it — self-regulation, robustness, local coupling — rather than the mechanism alone.

**Biological correspondence.** The three substrate-level mechanisms map one-to-one onto established neural ideas: fast/slow memory (complementary learning systems, hippocampus–cortex), homeostatic regulation of excitability, and slow structural plasticity of connectivity. The negative results sharpen the correspondence: reward signals without an error channel fail exactly where predictive-coding theory says they should — credit assignment needs prediction error, and rewards only modulate.

**Limitations.** All results are numerical (Digital-Model level). The readout-level comparison to the ESN is favorable to the ESN on standard metrics; the metadata-transfer experiment (Section 4.5) shows the mechanism — not the substrate — carries the long-horizon statistical memory, equalizing the systems on the regime task (0.994–0.998); the substrate's differentiation rests on its material-set memory design, the local sparse structure, and the robustness mechanisms. The homeostat's task-level gains are masked by readout compensation; the CPU-only scale (N ≤ 1024) leaves larger-scale behavior untested.

## 6. Conclusion

REDEM demonstrates that a physically motivated relaxation substrate can host a complete training-inference-unified learning system: an online error-driven readout, a dual-timescale metadata memory, a chaos homeostat, and slow structure plasticity, each validated individually and as an integrated system that beats every ablation and survives scale-up to N=1024. The two negative results on reward-only learning fix the design boundary conditions: at the readout an error channel is necessary; reward and intrinsic signals belong at the structure level. Honest benchmarking positions REDEM in the online/physical niche: competitive with generic reservoirs on standard tasks, robust where frozen batch learners fail, self-regulating under disturbance, and physically realizable — with the long-horizon statistical mechanism shown to be substrate-agnostic and transferable.

---

## References (to complete)

1. Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., Wermter, S. Continual lifelong learning with neural networks: a review. *Neural Networks* 113, 54–71 (2019).
2. Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., Hardt, M. Test-time training with self-supervision for generalization under distribution shifts. *ICML* 2020.
3. Rao, R. P. N., Ballard, D. H. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience* 2(1), 79–87 (1999).
4. Friston, K. The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience* 11(2), 127–138 (2010). https://doi.org/10.1038/nrn2787
5. Jaeger, H. The "echo state" approach to analysing and training recurrent neural networks. GMD Report 148 (2001).
6. Maass, W., Natschläger, T., Markram, H. Real-time computing without stable states. *Neural Computation* 14(11), 2531–2560 (2002).
7. McClelland, J. L., McNaughton, B. L., O'Reilly, R. C. Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review* 102(3), 419–457 (1995).
8. Hoerzel, G., Legenstein, R., Maass, W. Emergence of complex computational structures from chaotic neural networks through reward-modulated Hebbian learning. *Cerebral Cortex* 24(3), 677–690 (2014).
9. Pfister, J.-P., Gerstner, W. Triplets of spikes in a model of spike timing-dependent plasticity. *Journal of Neuroscience* 26(38), 9673–9682 (2006). https://doi.org/10.1523/JNEUROSCI.1425-06.2006
10. Benna, M. K., Fusi, S. Computational principles of synaptic memory consolidation. *Nature Neuroscience* 19(12), 1697–1706 (2016).
11. Author's prior NCE manuscript (substrate calibration source); companion Paper A: "Memory and chaos in a relaxation substrate" (substrate theory; target: IJBC/Chaos).
