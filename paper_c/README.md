# Paper C - The Metadata Equalizer (working title)

Short paper (single author, ~6 pages). Status: derivation stage; the theory
is derived and anchored on committed data, but the new experiments needed to
publish (s14-s17) have not been run yet.

**Working title**: The Metadata Equalizer: Substrate-Agnostic Slow-Trace
Transfer and Disturbance Robustness in Online Reservoir Computing.

**Core claim**: a single algorithmic component - the slow exponential trace
of reservoir states (the M3 metadata of Paper B) - is a *synthesized
forgetting kernel*: controllable horizon, exponential tail, and
substrate-independent by construction. On long-horizon statistical tasks the
bottleneck is timescale coverage, not substrate physics; the same slow trace
equalizes an ESN and the Si3N4 substrate to a common performance band,
collapses post-switch adaptation ~47x, and is robust under sequential
disturbance. Honest position: we never claim REDEM beats the ESN - in S10 the
ESN+metadata arm has the best overall accuracy; the paper is about the
mechanism.

## Contents

| File | Purpose |
|---|---|
| `DERIVATION.md` | The theory: Propositions 1-4 (kernel synthesis, timescale coverage, adaptation collapse, disturbance attenuation), data anchors, required new experiments |
| `PAPER_C_sketch.md` | Section outline, figure/table inventory, key numbers, submission checklist |
| `README.md` | This file |

## Non-overlap with Papers A and B (hard rules)

- Paper A owns the *material* forgetting kernel M(t) and substrate theory.
- Paper B owns the integrated REDEM system (RLS + M3 + M4 + M5) and the
  online-vs-frozen and ESN showdowns.
- Paper C owns exactly one mechanism - the slow trace - and its transfer
  across substrates. M5/M4 appear at most as controlled ablations.

## Data anchors (committed, full-seed)

| File | Used for |
|---|---|
| `../data/s10_esn_metadata_v1.*` | S10 equalization: esn_fast 0.9955 / esn_dual 0.9979 / redem 0.9942; adapt 11.22 -> 0.24 (10 seeds) |
| `../data/s11_disturbance_chain_v1.*` | Sequential robustness anchor: MC 8.47 vs 6.41 (+32%), kappa 25.3 -> 28.5 (10 seeds) |
| `../scripts/esn_metadata_comparison.py` | S10 producer script |

## Required new experiments (not yet run)

1. `s14` - ESN+metadata under the S11 disturbance chain (decisive for the
   "sequential robustness transfers" claim).
2. `s15` - controlled adaptation protocol (known switch instants).
3. `s16` - tau_m sweep on ESN and REDEM (ceiling vs tau_m; MC(k) 1/e horizon
   shift, Proposition-1 prediction).
4. `s17` - substrate stress: ESN spectral radius {0.7, 0.9, 0.99}.

See `DERIVATION.md` Section 8 for rationale and order.

## Open decisions

- Venue: Neurocomputing vs Neural Networks (short/communication).
- Run s14-s17 before drafting? (Recommended: yes, s14 first.)
- Citation closure: cite Paper B Section 4.5 once B is public.
- Title / naming: "metadata equalizer" is a working name.

## Compile (once PAPER_C.tex exists)

```powershell
pdflatex PAPER_C.tex    # run twice for cross-references
```

## Submission checklist (Paper C)

- [ ] Replace `[Author Name]` / affiliation / email.
- [ ] Swap the document class to the target journal template.
- [ ] Honest-ESN wording locked (esn_dual best overall accuracy; state it).
- [ ] Cite Paper A Section 5 and Paper B Section 4.5.
- [ ] Code/data availability statement (REDEM repo, private during review).
