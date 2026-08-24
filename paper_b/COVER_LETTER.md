# Cover Letter — Paper B

**Date**: 2026-02-19

**To**: The Editor-in-Chief, *Neural Networks* (Elsevier)

**Re**: Submission of the manuscript *"REDEM: Training-Inference Unified
Learning with Meta-Adaptation and Structural Plasticity for Non-Stationary
Environments"*

Dear Editor,

Please consider our manuscript for publication in *Neural Networks*.

**Author**: Yaming Hu (ORCID: 0009-0003-1406-0485), Independent Researcher,
Guiyang, Guizhou Province, China. E-mail: 64687555@qq.com.

**Summary.** We present REDEM, an online learning architecture in which
training and inference are unified: a recursive-least-squares readout
updates at every pulse from the live prediction error (local rule, no
backpropagation through time), on top of a physics-constrained relaxation
substrate with three concurrent substrate-level mechanisms:

- **M3 meta-adaptation** (dual-timescale statistical memory): a per-unit
  slow trace of fast activity, substrate-agnostic and transferable to a
  matched echo-state network;
- **M5 chaos homeostat**: online finite-time Lyapunov estimation that
  regulates the substrate near the memory-optimal critical point, restoring
  +8–18% of held-out memory after single disturbances and +32% after three
  sequential disturbances (10 seeds);
- **M4 structural plasticity**: gentle, correlation-guided rewiring
  (+8–11%) that destabilizes when aggressive (−23%); a novelty-reward-guided
  variant improves held-out memory further (14.59 vs 12.43, paired t = 4.5,
  10/10 seeds), confirming that intrinsic signals are structure-level tools;
- **M4–M5 coupling (negative result)**: under the sequential disturbance
  chain, rewiring churn gated by the homeostat's κ deviation does not help
  recovery (homeostat alone r3 MC 8.47 vs +fixed-churn 6.45 / +coupled 5.27,
  0/10 seeds) — disturbance recovery is the homeostat's job alone, and
  rewiring during active compensation is counterproductive.

The integrated system beats every single-mechanism ablation (0.996 vs 0.973,
p < 0.0001) and the advantage persists at N = 1024. A causal audit (10 seeds,
7 arms, including the plasticity-correlation leak) confirms all mechanisms
use only past information. We report honestly the negative results that
fixed the design (reward-only readouts cannot credit-assign through a class
inversion; task-agnostic intrinsic rewards cannot rescue them), and we
report honestly that a well-tuned online ESN edges REDEM on standard tasks
(drift accuracy 1.000 vs 0.991; Mackey–Glass NMSE 3.6e-5 vs 0.0018) — the
value proposition is the mechanism set, not raw benchmark supremacy.

**Fit to the journal.** The work addresses continual/online learning with
biologically motivated mechanisms (reward-gated eligibility, dual-timescale
memory, homeostatic regulation, structural plasticity) — central themes of
*Neural Networks*.

**Statements.** The manuscript is original, has not been published
elsewhere, and is not under consideration by any other journal. There are no
conflicts of interest. The full simulation code and data are available at
<https://github.com/huyamingc/REDEM> (private during review; made public on
acceptance). The companion substrate-characterization paper is cited as [12].

Yours sincerely,

Yaming Hu
