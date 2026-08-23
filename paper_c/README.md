# Paper C - The Metadata Equalizer (working title)

Short paper (single author, ~6 pages). Status: post-s14. The decisive
transfer experiment (s14) is DONE: it falsifies the original "sequential
robustness transfers" thesis and reframes the paper as a three-mechanism
disentanglement. s15-s16 remain before drafting. Theory in
`DERIVATION.md`; outline and numbers in `PAPER_C_sketch.md`.

**Working title** (proposed): The Metadata Equalizer: a Substrate-Agnostic
Statistical Memory for Online Reservoirs.

**Core claim**: the slow exponential trace of reservoir states (the M3
metadata of Paper B) is a *synthesized forgetting kernel* - controllable
horizon, exponential tail, substrate-independent by construction. On
long-horizon statistical tasks the bottleneck is timescale coverage, not
substrate physics; the same slow trace equalizes an ESN and the Si3N4
substrate on accuracy and collapses post-switch adaptation ~47x. It is a
*statistical* memory, not an episodic one: it adds no raw memory capacity
but attenuates readout noise on the online task. Sequential disturbance
recovery is the homeostat's job, not the metadata's. Honest position: we
never claim REDEM beats the ESN - in S10 the ESN+metadata arm has the best
overall accuracy.

## Contents

| File | Purpose |
|---|---|
| `DERIVATION.md` | The theory: Propositions 1-4, s14 results and the reframed disentanglement thesis, data anchors, remaining experiments |
| `PAPER_C_sketch.md` | Section outline, figure/table inventory, key numbers, submission checklist |
| `README.md` | This file |

## Non-overlap with Papers A and B (hard rules)

- Paper A owns the *material* forgetting kernel M(t) and substrate theory.
- Paper B owns the integrated REDEM system (RLS + M3 + M4 + M5) and the
  online-vs-frozen and ESN showdowns.
- Paper C owns exactly one mechanism - the slow trace - and its transfer
  across substrates, plus the three-mechanism disentanglement (metadata /
  homeostat / substrate physics). M5 appears only as the controlled
  redem_reg arm that isolates the homeostat's role.

## Data anchors (committed, full-seed)

| File | Used for |
|---|---|
| `../data/s10_esn_metadata_v1.*` | S10 equalization: esn_fast 0.9955 / esn_dual 0.9979 / redem 0.9942; adapt 11.22 -> 0.24 (10 seeds) |
| `../data/s11_disturbance_chain_v1.*` | Homeostat sequential recovery anchor: MC 8.47 vs 6.41 (+32%), kappa 25.3 -> 28.5 (10 seeds) |
| `../data/s14_esn_disturbance_chain_v1.*` | S14 transfer test: MC paired diffs -0.78/-0.76/-0.69 (0/10 positive), r3 NMSE -9.5% (10/10), redem_reg reproduces S11 anchor |
| `../scripts/esn_metadata_comparison.py` | S10 producer script |
| `../scripts/s14_esn_disturbance_chain.py` | S14 producer script (Type: PAPER) |

## Experiments status

- DONE: s14 (ESN+metadata under the disturbance chain, 10 seeds) - see
  `DERIVATION.md` Section 8 for the full reading.
- PENDING: s15 (controlled adaptation protocol), s16 (tau_m sweep +
  falsification stress test), s17 (substrate stress, optional).

## Open decisions (user gates)

- **Thesis reframing (needs user confirmation):** adopt the
  three-mechanism disentanglement thesis (metadata = statistical memory /
  homeostat = disturbance recovery / physics = raw memory)?
- Title: drop "Disturbance Robustness" if the disentanglement thesis is
  adopted.
- Venue: Neurocomputing vs Neural Networks (short/communication).
- Run s15-s16 before drafting? (Recommended: yes - s15 fixes the Prop. 3
  adaptation claim, s16 stress-tests the s14 falsification.)
- Citation closure: cite Paper B Section 4.5 once B is public.

## Compile (once PAPER_C.tex exists)

```powershell
pdflatex PAPER_C.tex    # run twice for cross-references
```

## Submission checklist (Paper C)

- [ ] Replace `[Author Name]` / affiliation / email.
- [ ] Swap the document class to the target journal template.
- [ ] Honest wording locked: equalizer is about statistical memory; +32% is
      homeostat; esn_dual best S10 accuracy - state all three.
- [ ] Cite Paper A Section 5 and Paper B Section 4.5.
- [ ] Code/data availability statement (REDEM repo, private during review).
