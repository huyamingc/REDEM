# Paper D - REDEM-SSM (working title, sketch stage)

**Working title (user proposal, adopted)**: REDEM-SSM: A Foundation Model
Architecture with Native Online Learning, Meta-Adaptation, and Structural
Plasticity

**Status**: P1 + P3a DONE 2026-02-18, both FALSIFIED with mechanisms
isolated. (1) P1: the bare-M1 linear readout on the diagonal-SSM state
cannot track (stream ppl 58-116 vs A1 15.01, 0/10 improved; oracle ~31 ~
uniform) because a linear readout cannot recover the current token from
the linearly-decayed mixture; M1 itself works (current-token-projection
control: 11.75, 10/10 better than A1, oracle 7.25 ~ the pooled-table
ceiling). (2) P3a: fixed Mamba-style gates (feature-level, gamma 1/5) do
NOT fix it (77.2/72.2, oracle 37.1/38.5, sharper worse) - the current
token must enter the readout as an ADDITIVE input feature; the state's
value is the domain level for M3 metadata (P2), and real selectivity must
be learned (gradient), a design boundary for the RLS-only constraint.
Next phases: P2 (M3 EMA second state + routing). Data:
`data/s19_ssm_rls_readout_v1.csv` (70 rows, 10 seeds).

## Origin and motivation

Paper C §7 conclusion point (6) (added 2026-02-18) states the host-boundary
result: the s18 negative result (A2 gating 0/10 seeds) together with the
narrow A3 routing window (stream-ppl benefit only at tau_m <= 500) show that
the *full* REDEM framework does not instantiate on a frozen-feedforward
host. The failure is attributed to the update policy (M1 removed, M4
switching made abrupt), not to the Transformer per se --- routing transferred
on most of the operating range. Paper D takes the constructive consequence:
design a foundation model whose host natively provides the three
requirements (a continuously active online readout, a slowly integrating
statistical memory, and gradually applied structural switches) instead of
retrofitting them.

## Non-overlap with Papers A, B, C (hard rules)

| Paper | Owns |
|---|---|
| A | substrate physics: log-normal tau spectrum, FTLE, material forgetting kernel M(t) |
| B | the REDEM algorithm as such (RLS readout, M3 EMA metadata, M4 plasticity, M5 homeostat) on the physical substrate |
| C | the three-mechanism disentanglement thesis + the Transformer transfer boundary (s18) |
| **D (this work)** | the **SSM instantiation** of the mechanisms (A-diagonal tau-spectrum mapping, per-token RLS readout, EMA second state, dynamic state-dimension/spectrum plasticity, eigenvalue-monitored dt regulation) and its empirical validation on streaming benchmarks |

Paper D cites A/B/C as foundations; Paper C §7 point (6) is its roadmap
entry. Paper D never re-derives substrate physics (A), the REDEM algorithm
(B), or the disentanglement thesis (C).

## Honesty / falsification discipline (carried from C)

1. **D2 boundary**: M5-on-SSM is *new research* (stability regulation), NOT
   validation of the substrate homeostat. Paper A's FTLE edge-of-chaos
   results do not port literally: a linear SSM is not chaotic. The M5
   analogue regulates the linear stability edge (|lambda| -> 1 / state-norm
   growth rate), which is an analogy, not an identity.
2. **"Training = inference" is not claimed**: SSM parallel training is
   offline gradient training; M1 is the per-token online RLS at inference.
   These are distinct and both are stated.
3. **Related-work honesty**: online learning inside SSMs already exists
   (Longhorn ICLR 2025, Titans NeurIPS 2025, TTT ICLR 2024, test-time SSM
   alignment for user-interest tracking). Paper D's claimed novelty is the
   three-mechanism *division of labor* + the *second-order (RLS)* update +
   *mechanism-level* stability regulation --- not "first online learning in
   an SSM".
4. **Scale honesty**: the PoC is tiny by design (state dim 64-256, ~1M
   params, torch CPU, s18 pattern). No SOTA or scaling claims. Benchmarks
   are synthetic multi-domain streams (s16 protocol generalized) + small
   char-level corpora (s18 style); WikiText-103 streaming is NOT feasible on
   CPU and is not promised.
5. **10-seed falsification discipline**: every claim is paired per-seed with
   sign-consistency counts (n_pos/10), exactly as Papers A-C.

## Contents

| File | Purpose |
|---|---|
| `README.md` | This file: status, non-overlap rules, honesty discipline |
| `PAPER_D_sketch.md` | The design doc: positioning, REDEM-SSM mapping, novelty, falsifiable predictions, roadmap P1-P5 |

## Roadmap (corrected estimates, part-time 3-4 months)

| Phase | Task | Deliverable | Estimate |
|---|---|---|---|
| P1 | Hand-rolled diagonal SSM (state dim 64-256) + per-token RLS on output projection; streaming tracking task | Working REDEM-SSM prototype (torch CPU, no mamba-ssm dependency) | 1-2 w |
| P2 | M3 EMA second state; drift detection on two-domain stream | M3-on-SSM drift detection + routing evidence | 1 w |
| P3 | M4 (state-dim activation / A-spectrum prune-grow, "gentle wins") + M5 (log|lambda| / state-norm monitor -> adjust dt) | Complete REDEM-SSM system | 2-3 w |
| P4 | Benchmarks: synthetic multi-domain streams + small char-level corpora vs bare SSM vs Transformer+LoRA reference (s18 reuse) | Full comparison data (10 seeds) | 1-2 w |
| P5 | Paper writing; arXiv + workshop first | Submission-ready draft | 2-3 w |

Total ~8-12 weeks full-time (optimistic), 3-4 months part-time. Phase
gates: each phase's exit criteria are defined in `PAPER_D_sketch.md`
Section 6; a falsified exit criterion stops the phase and is reported, not
silently worked around.

## Compile / reproduction

No scripts yet. When P1 starts, scripts live in `../scripts/` as `s19_*`
(ML class per CLAUDE.md) with data in `../data/s19_*_v1.csv/json` and
figures in `../figures/`.
