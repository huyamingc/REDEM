# Paper C - Three Mechanisms, Three Roles (working title)

Short paper (single author, ~6 pages). Status: post-s14, post-s16,
post-s15, post-s16b (all 10 seeds). The three-mechanism disentanglement
thesis is adopted (user gate, 2026-02-17): metadata is a substrate-agnostic
*statistical* memory; disturbance recovery is the homeostat's role; raw
memory capacity is the substrate's physics. Only s17 (optional substrate
stress) remains before drafting. Theory in `DERIVATION.md`; outline and
numbers in `PAPER_C_sketch.md`.

**Working title** (user proposal, adopted): Dissecting Online Learning
Mechanisms: Statistical Memory, Homeostatic Recovery, and Substrate Physics
are Non-Transferable.

**Core claim**: the slow exponential trace of reservoir states (the M3
metadata of Paper B) is a *synthesized forgetting kernel* - controllable
horizon, exponential tail, substrate-independent by construction. On
long-horizon statistical tasks the bottleneck is timescale coverage, not
substrate physics; the same slow trace equalizes an ESN and the Si3N4
substrate on accuracy and speeds post-switch adaptation (s15: ~10 pulses of
~200, variance collapse p90 76.5 -> 42). It is a *statistical* memory, not
an episodic one: it adds no raw memory capacity and does not transfer
disturbance robustness to an ESN at any metadata timescale (s14 + s16,
0/10 seeds positive; s16b: robust in sign to probe protocol), but it
denoises state-level corruption (s16b V2: gap -0.69 -> -0.01) and
attenuates readout noise on the online task. Sequential disturbance
recovery is the homeostat's job, not the metadata's. Honest position: we
never claim REDEM beats the ESN - in S10 the ESN+metadata arm has the best
overall accuracy.

## Contents

| File | Purpose |
|---|---|
| `DERIVATION.md` | The theory: Propositions 1-4, s14/s16 results, the locked disentanglement thesis, remaining experiments |
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
| `../data/s16_tau_m_pressure_test_v1.*` | S16 pressure test: r3 MC paired diffs -0.70/-0.69/-0.68/-0.68 at tau_m 200/500/1000/2000 (0/10 positive), strong claim locked |
| `../data/s15_controlled_adaptation_v1.*` | S15 controlled adaptation: T40 40.6 vs 49.9 (dual vs fast), p90 42 vs 76.5; "47x" ratio = metric artifact |
| `../data/s16b_falsification_stress_test_v1.*` | S16b probe stress: 0/10 positive at all (tau_m, variant); magnitude V0 -0.69 -> V2 -0.01 (state-noise denoising) |
| `../scripts/esn_metadata_comparison.py` | S10 producer script |
| `../scripts/s14_esn_disturbance_chain.py` | S14 producer script (Type: PAPER) |
| `../scripts/s16_tau_m_pressure_test.py` | S16 producer script (Type: PAPER) |
| `../scripts/s15_controlled_adaptation.py` | S15 producer script (Type: PAPER) |
| `../scripts/s16b_falsification_stress_test.py` | S16b producer script (Type: PAPER) |

## Experiments status

- DONE: s14 (ESN+metadata under the disturbance chain), s16 (tau_m
  pressure test), s15 (controlled adaptation), s16b (probe stress test) -
  all 10 seeds; see `DERIVATION.md` Section 8.
- PENDING: s17 (substrate stress, optional).

## Open decisions (user gates)

- **Thesis reframing (ADOPTED, 2026-02-17):** three-mechanism
  disentanglement; strong claim locked on s16 (no tau_m closes the MC gap).
- **Title (ADOPTED, 2026-02-17):** "Dissecting Online Learning Mechanisms:
  Statistical Memory, Homeostatic Recovery, and Substrate Physics are
  Non-Transferable".
- Venue: Neurocomputing vs Neural Networks (short/communication) - open.
- Paper B wording flag (user decision): the S10/S5 "47x"/"9-20x"
  adaptation ratios are window-position metric artifacts per s15; Paper B
  Section 4.3 and README headlines should be softened.
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
- [ ] Strong claim wording locked on s16 (0/10-seed sign consistency, ~5
      sigma paired diffs).
- [ ] Cite Paper A Section 5 and Paper B Section 4.5.
- [ ] Code/data availability statement (REDEM repo, private during review).
