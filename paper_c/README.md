# Paper C - Three Mechanisms, Three Roles (working title)

Short paper (single author, ~8 pages two-column equivalent; elsarticle
preprint renders 13 pages). Status: post-s14, post-s16, post-s15, post-s16b,
post-s5b, post-s17, post-s18 (all 10 seeds). The three-mechanism
disentanglement thesis is adopted (user gate, 2026-02-17): metadata is a
substrate-agnostic *statistical* memory; disturbance recovery is the
homeostat's role; raw memory capacity is the substrate's physics.
**PAPER_C.tex is drafted and submission-ready** (elsarticle preprint,
13 pp, zero warnings, incl. §6 LLM extension with s18 results + Fig 3 +
Table 6) for Neurocomputing / Neural Networks short paper. Theory in `DERIVATION.md`;
outline and numbers in `PAPER_C_sketch.md`.

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
0/10 seeds positive; s16b v2: sign robust across the probe protocol —
≤1/10 in every (τ_m, variant) cell over all four timescales), but it
denoises state-level corruption (s16b V2: gap -0.69 -> -0.04…-0.01 at
every τ_m) and
attenuates readout noise on the online task. Sequential disturbance
recovery is the homeostat's job, not the metadata's. Honest position: we
never claim REDEM beats the ESN - in S10 the ESN+metadata arm has the best
overall accuracy.

## Contents

| File | Purpose |
|---|---|
| `DERIVATION.md` | The theory: Propositions 1-4, s14/s15/s16/s16b/s5b/s17 results, the locked disentanglement thesis |
| `PAPER_C_sketch.md` | Section outline, figure/table inventory, key numbers, submission checklist |
| `LLM_EXTENSION.md` | Deduction: whether/how the disentanglement thesis transfers to Transformer/LoRA systems (PoC design, falsification discipline) |
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
| `../data/s16b_falsification_stress_test_v2.*` | S16b probe stress (v2: 10 seeds × 4 τ_m × 3 variants; ≤1/10 positive in every cell; magnitude V0 -0.69 -> V2 -0.04…-0.01, state-noise denoising at every τ_m; v1 kept for the original table) |
| `../data/s5b_controlled_adaptation_v1.*` | S5b substrate arms controlled: T200 265-304 (fast) vs 202-211 (dual), factor 1.3-2.4; overall acc reproduces S5 |
| `../data/s17_substrate_stress_v1.*` | S17 substrate stress: equalizer gain positive at all 6 ESN configs (+0.1 to +1.0 pp), magnitude tracks timescale starvation |
| `../data/s18_llm_drift_gate_v1.*` | S18 LLM PoC (90 runs): A3 routing transfers (forgetting −2.5 ppl, 10/10 seeds at τ_m=200; ppl −1.07 at τ_m≤500), A2 gate falsified (0/10); sensitive interval τ_m≥1000 |
| `../scripts/esn_metadata_comparison.py` | S10 producer script |
| `../scripts/s14_esn_disturbance_chain.py` | S14 producer script (Type: PAPER) |
| `../scripts/s16_tau_m_pressure_test.py` | S16 producer script (Type: PAPER) |
| `../scripts/s15_controlled_adaptation.py` | S15 producer script (Type: PAPER) |
| `../scripts/s16b_falsification_stress_test.py` | S16b producer script (Type: PAPER) |
| `../scripts/s5b_controlled_adaptation.py` | S5b producer script (Type: PAPER; Paper B revision evidence) |

## Experiments status

- DONE: s14 (ESN+metadata under the disturbance chain), s16 (tau_m
  pressure test), s15 (controlled adaptation, ESN arms), s16b (probe
  stress test), s5b (S5 substrate arms controlled), s17 (substrate stress),
  s18 (LLM drift-gate PoC: routing transfers, gate falsified) - all 10
  seeds; see `DERIVATION.md` Section 8 and `LLM_EXTENSION.md` §7.7.
- PAPER_C.tex drafted and submission-ready (elsarticle preprint, 13 pp,
  zero warnings). §6 LLM extension WRITTEN (s18 results + Fig 3 + Table 6;
  A2 gating falsified as design-lesson evidence, A3 routing transfers).
  §7 conclusion point (6) ADDED (host-boundary statement: the full
  framework needs a stateful host; calibrated to the §6 policy-level
  lesson, not an architecture trial).

## Open decisions (user gates)

- **Thesis reframing (ADOPTED, 2026-02-17):** three-mechanism
  disentanglement; strong claim locked on s16 (no tau_m closes the MC gap).
- **Title (ADOPTED, 2026-02-17):** "Dissecting Online Learning Mechanisms:
  Statistical Memory, Homeostatic Recovery, and Substrate Physics are
  Non-Transferable".
- Venue: **Neurocomputing / Neural Networks short paper (DECIDED,
  2026-02-17)** - both Elsevier; `PAPER_C.tex` now uses `elsarticle`
  preprint class.
- Paper B wording: resolved (2026-02-17) - §4.3/§4.5/§4.6 and abstract
  revised with the controlled s15/s5b measurements.
- Citation closure: cite Paper B Section 4.5 once B is public. (2026-08-24:
  companion papers are now formally cited as companion preprints —
  `\cite{companionA}` / `\cite{companionB}` — in PAPER_C.tex; swap to the
  published versions on acceptance.)

## Compile

```powershell
pdflatex PAPER_C.tex    # run three times for cross-references (elsarticle)
```

## Submission checklist (Paper C)

- [x] Author filled (2026-02-19): Yaming Hu, ORCID 0009-0003-1406-0485,
      Independent Researcher, Guiyang, Guizhou Province, China;
      64687555@qq.com. Cover letter: `COVER_LETTER.md`.
- [x] Document class: `elsarticle` preprint (Elsevier initial-submission
      format); switch to `\documentclass[final,3p]{elsarticle}` (two-col)
      or `[final,1p]` at acceptance per journal preference.
- [ ] Honest wording locked: equalizer is about statistical memory; +32% is
      homeostat; esn_dual best S10 accuracy - state all three (done in
      text).
- [ ] Strong claim wording locked on s16 (0/10-seed sign consistency, ~5
      sigma paired diffs) - done.
- [x] Cite Paper A Section 5 and Paper B Section 4.5 (companion refs in
      bibliography; `\cite{companionA}` at the substrate definition, 
      `\cite{companionB}` at the M3 introduction, `\cite{maass2002}` at the
      reservoir citation — added 2026-08-24 review audit, previously
      orphaned bibitems).
- [ ] Code/data availability statement (REDEM repo, private during review)
      - done.
- [ ] Keywords present (Elsevier format) - done.
- [x] §6 LLM extension written into PAPER_C.tex (s18 results + Fig 3 +
      Table 6; A2 falsification framed as design lesson) - done.
- [x] §7 conclusion point (6) host-boundary paragraph added (calibrated:
      host property + update-policy attribution + "stateful host" future
      work; NOT an architecture trial of the Transformer) - done.
- [ ] Cover letter: the falsifying-experiment angle (which mechanism does
      which job) + honest ESN position.
