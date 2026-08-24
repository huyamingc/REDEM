# Cover Letter — Paper A

**Date**: 2026-02-19

**To**: The Editor-in-Chief, *International Journal of Bifurcation and Chaos*
(World Scientific) [alternative venue: *Chaos: An Interdisciplinary Journal
of Nonlinear Science*, AIP]

**Re**: Submission of the manuscript *"Memory and chaos in a
physics-constrained relaxation substrate: phase diagram, multi-timescale
forgetting, and disturbance robustness"*

Dear Editor,

Please consider our manuscript for publication in *International Journal of
Bifurcation and Chaos*.

**Author**: Yaming Hu (ORCID: 0009-0003-1406-0485), Independent Researcher,
Guiyang, Guizhou Province, China. E-mail: 64687555@qq.com.

**Summary.** We characterize the computational dynamics of a
physics-constrained relaxation substrate — an Si3N4-style shallow-trap
network with a log-normal time-constant spectrum and a per-pulse,
topology-dependent contrast coupling. The substrate is studied as a
nonlinear dynamical system with a tuneable chaos knob:

- **Order–chaos phase diagram**: a sharp transition at coupling strength
  κ* ∈ (25, 30) — a follow-up fine sweep (κ ∈ [20,30] at unit resolution)
  pins κ* = 25.3–27.9 with bootstrap CIs of ±0.1–0.8, and a clip-range
  ablation (α_max up to 5× wider) shows the transition is invariant to the
  physical clip for the memory-relevant topologies, i.e. coupling-driven
  — with held-out linear memory peaking +24–53% just before it;
- **Multi-timescale forgetting kernel**: the material memory kernel
  M(t) = ∫p(τ)e^{−t/τ}dτ matches the measured memory decay curve with
  Pearson r = 0.97, providing an analytic link between the trap spectrum and
  the computational memory horizon;
- **Disturbance robustness**: a λ-homeostat (finite-time Lyapunov estimator
  that adjusts the coupling toward a criticality target) restores +8–18% of
  held-out memory after timescale drift, structural pruning, and readout
  noise; a systematic target sweep identifies the edge of chaos
  (λ_target = 0) as the empirically optimal operating point;
- **Separation–memory trade-off**: near-critical states are information-rich
  but linearly undecodable, including a clean demonstration that a
  unidirectional-ring critical state carries near-zero linearly decodable
  memory despite maximal input separation.

All results are numerical (CPU-only), fully reproducible, and derived with a
10-seed paired protocol; complete derivations are given in the Appendix.

**Fit to the journal.** The paper sits at the intersection of nonlinear
dynamics, memory theory in physical substrates, and reservoir-computing
fundamentals — squarely within the scope of IJBC. The analytic forgetting
kernel and the homeostatic control of a physical chaos knob are the
distinctive contributions.

**Statements.** The manuscript is original, has not been published
elsewhere, and is not under consideration by any other journal. There are no
conflicts of interest. The full simulation code and data are available at
<https://github.com/huyamingc/REDEM> (private during review; made public on
acceptance).

Yours sincerely,

Yaming Hu
