# Cover Letter / Abstract Note — Paper D

**Date**: 2026-02-19

**Venue strategy**: arXiv preprint first, then a machine-learning workshop;
NeurIPS/ICML submission is the stretch goal. This letter is the arXiv
abstract note and the workshop submission note.

**Title**: *REDEM-SSM: A State-Space Architecture with Native Online
Learning, Meta-Adaptation, and Structural Plasticity*

**Author**: Yaming Hu (ORCID: 0009-0003-1406-0485), Independent Researcher,
Guiyang, Guizhou Province, China. E-mail: 64687555@qq.com.

**Abstract.**

> Paper C's host-boundary result shows that the full REDEM mechanism set
> (online RLS readout, statistical metadata, structural plasticity, stability
> regulation) does not instantiate on a frozen-feedforward host: the negative
> results were traced to the update policy, not to the host per se. This
> paper takes the constructive consequence and instantiates REDEM's
> mechanisms natively in a diagonal linear state-space model (SSM) host,
> from the ground up, on CPU.
>
> P1/P3a falsify the naive design: a linear readout on the state mixture
> cannot recover the current token (deconvolution impossibility for diagonal
> decay), with a B-projection control proving the RLS itself converges
> (11.75 ppl vs the 7.25 oracle ceiling). P2 shows M3's slow-trace metadata
> transfers to the SSM host: routing (two readouts, one per domain) improves
> forgetting at τ_m ≤ 1000 (10/10 seeds), while gating-only inverts on RLS
> readouts (readout-dynamics-dependent, with the what-is-paused qualifier).
> P3 supports both "gentle wins" hypotheses: soft routing beats abrupt
> switching (−1.82 ppl, 10/10), and a state-norm homeostat (M5 analogue)
> bounds the whitened state and restores the full-state EMA detector 5/5.
> P4 benchmarks the full stack on a 4-domain irregular-switch stream
> (10/10 vs bare SSM and vs a Transformer+LoRA reference) and on two
> real-text corpora (Alice vs Dickens: −1.27 stream ppl vs bare, 10/10).
>
> Everything is toy-scale by design (state dim 128, ~4k-parameter readouts,
> character-level vocabularies); we make no scaling or SOTA claims. The
> contribution is the three-mechanism division of labor transplanted onto a
> state-space host, with the honest falsification narrative preserved
> throughout.

**Honesty notes (workshop reviewers, please read)**. CPU-only; no
mamba-ssm dependency; the Transformer+LoRA reference reuses the Paper C §7
hyperparameters untuned for four domains; real-text gains are smaller than
synthetic-task gains because the two books share English bigram statistics.
The deconvolution impossibility is proven for *linear* readouts on the
diagonal-decay state mixture, and scoped as such.

**Statements.** The work is original and not under consideration elsewhere.
There are no conflicts of interest. Code and data are available at
<https://github.com/huyamingc/REDEM> (private during review; made public on
acceptance). Companion papers are cited as [A]–[C].

Yours sincerely,

Yaming Hu
