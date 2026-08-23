# Cover Letter — Paper C

**Date**: 2026-02-19

**To**: The Editor-in-Chief, *Neurocomputing* (Elsevier)
[alternative venue: *Neural Networks* short paper]

**Re**: Submission of the manuscript *"Dissecting Online Learning Mechanisms:
Statistical Memory, Homeostatic Recovery, and Substrate Physics are
Non-Transferable"*

Dear Editor,

Please consider our manuscript for publication in *Neurocomputing*.

**Author**: Yaming Hu (ORCID: 0009-0003-1406-0485), Independent Researcher,
Guiyang, Guizhou Province, China. E-mail: 64687555@qq.com.

**Summary.** We ask a rarely tested question: which component of an online
learning system is responsible for which capability, and which capabilities
transfer to a different substrate? Using a falsifying-experiment design with
a 10-seed paired protocol (sign consistency reported throughout), we dissect
three mechanisms:

- **M3 statistical memory**: the slow exponential trace of reservoir states
  is a *synthesized forgetting kernel* — controllable horizon, exponential
  tail, substrate-independent by construction. It equalizes an ESN and the
  Si3N4 substrate on long-horizon statistical tasks and speeds post-switch
  adaptation, but it adds no raw memory capacity and does not transfer
  disturbance robustness to an ESN at any tested metadata timescale
  (0/10 seeds positive, ~5σ paired differences);
- **M5 homeostatic recovery**: sequential-disturbance recovery (+32%) is the
  homeostat's job, not the metadata's;
- **Substrate physics**: raw memory capacity is owned by the substrate's
  multi-timescale relaxation spectrum.

The falsification is stress-tested across metadata timescales (τ_m ∈
{200,500,1000,2000}), probe protocols (V0/V1/V2), and a Transformer/LoRA
proof of concept (routing transfers 10/10; gating-only is falsified 0/10).

**Fit to the journal.** The paper contributes a falsification methodology for
online-learning mechanism attribution, directly relevant to continual
learning, reservoir computing, and the design of adaptive systems.

**Statements.** The manuscript is original, has not been published
elsewhere, and is not under consideration by any other journal. There are no
conflicts of interest. The full simulation code and data are available at
<https://github.com/huyamingc/REDEM> (private during review; made public on
acceptance). Companion papers are cited as [A] (substrate theory) and [B]
(the integrated REDEM system).

Yours sincerely,

Yaming Hu
