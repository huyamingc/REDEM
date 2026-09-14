# Paradigm Note — Traps, Boundaries, and Rules from Papers D and E, and Paper F's Role

> Purpose: turn the falsification record of Papers D and E into a **design-rule set** for a
> self-evolving / self-correcting system, and fix Paper F's role inside that rule set.
> Status: internal planning document. All numeric claims below are reproducible from committed data.
> Language: body English; the same content is summarized in Chinese in the Paper F plan.

---

## 0. Why this note exists

Papers A–E were written as separate studies. Read together, D and E are not two papers but
**two falsification records at two different layers of the same system**:

| Layer | Paper | What was falsified |
|---|---|---|
| Output / readout | D | the readout form was **human-prescribed** (linear + clip); its metric artifact then shaped several conclusions |
| Decision | E | the autonomy built on top of that readout reaches only **rule selection** between two human-supplied correction laws |
| Measurement | F (this) | the artifact is a property of the **objective**, not of the readout's numeric range — and a post-hoc fix does not repair it |

The rule set below is the extraction. Paper F is the first paper that is *written as a rule*,
not as a mechanism demonstration.

---

## 1. Trap 1 (D) — "the artifact is in the numbers"

**Instance.** D's readout is squared-loss RLS evaluated as `CE = -ln clip(y_hat[target], 1e-12, 1)`.
Negative or near-zero scores are parked on the clip floor: 1.58% of stream tokens on the input-path
arm, 10.23% on the coverage-spectrum state arm (`data/s19_ssm_rls_readout_v1.csv`).

**What D got right.** D identified the floor as a *calibration* property, refused to claim the
floor-sensitive stream rankings, and left the fix as a prescription:
*a calibrated output (e.g. a non-negative readout or explicit simplex projection) that removes the
clip floor* (`PAPER_D_NC.tex` Conclusion L1044-1046).

**The residual trap.** D's framing still located the artifact in the readout's *range*
("non-negative", "simplex projection"). Paper F's simplex control falsifies that reading:

| comparison (same token set, stream) | dppl | seeds |
|---|---|---|
| B-softmax-sgd − B-lin-clip | **−3.11** | 10/10 |
| **B-simplex − B-lin-clip** | **+0.03** | **0/10** |
| Skip-simplex − Skip-lin-clip | +17.24 | 0/10 |

The post-hoc simplex projection **removes every negative output** (`neg_frac` 1.582% → 0,
`clip_frac` 1.582% → 0) and changes stream perplexity by **+0.03, in the wrong direction**.
Removing the negatives is not the repair. `data/s50_paper_f_pilot_nl_readout_v1.csv`, 80 rows,
8 arms x 10 seeds; pairing chain verified against `s19` (B-lin-clip ≡ s19 B-proj exactly, 10/10).

**Rule R1 — the instrument must be part of the learned system.**
A self-evaluation quantity must either be part of what training optimizes, or be shown invariant
to what training changes. A frozen human-prescribed criterion embedded in the measured quantity
produces artifacts that no amount of self-correction can remove, because the system is not
optimizing that quantity.

**Rule R2 — repair at the objective, not at the range.**
If a pathological output pattern is produced by an objective, post-processing the output cannot fix
it. Any "fix" that operates downstream of the loss is decoration. The test is cheap and mandatory:
implement the post-hoc fix as an arm, and report its dCE against the no-fix baseline. If the
post-hoc fix moves nothing, the pathology is in the objective.

---

## 2. Trap 2 (E) — "autonomy stops at rule selection"

**Instance.** E's positive claim is real and correctly stated in its own abstract:
*autonomy at the decision level* — the system finds the correct direction, repairs its own failures,
reuses frozen hypotheses, learns its own snapshot threshold, and knows from its own observation
scale when it cannot act (R1-R6, `PAPER_E_v3.tex` L44-68).

**The ceiling.** E's R1 mechanism is explicit that credit assignment under ±1 reward is a
**rule-selection problem** (L167): direction information is hidden in the sign of the aggregated
reward rate, and the system chooses between two human-supplied correction laws
(flip vs scale-with-reward). The system does not invent a correction law; it selects one of ours.

This is not a defect of E. It is a **boundary that E measured**:

> Autonomy in the decision layer is bounded by the breadth of the candidate set a human supplied.
> With a ±1 channel the gradient to a parameter is not available, so the only latitude left is
> rule selection. Every additional capability costs another mechanism: E needs a direction detector,
> a capacity probe, a frozen-hypothesis memory, a self-tuned gate, and a source-selection device.

**Rule R3 — autonomy is capped by the candidate set, so widen the candidate set, not the mechanism count.**
Adding mechanisms to a decision layer buys capabilities one at a time and never buys the ability to
generate a new hypothesis. The lever with a better return is to make the *form* of the system's own
output trainable, because then the candidate set is a function space rather than a two-element set,
and it is reached by the same gradient signal that trains everything else.

---

## 3. Why "port E's machinery onto the SSM host" is the wrong plan

E's decision-level mechanisms are **substrate-agnostic** (they act on readout, reward rate and
thresholds, not on the recurrence), so porting them to a state-space host is a re-run, not a
contribution. Worse, the port is predicted to underperform, and the reason is measurable:

- E's mechanisms exist **because** the feedback channel is ±1 and no gradient to a parameter exists.
- F's calibrated readout trains on per-token labels through cross-entropy. The gradient is available;
  the entire reason for the 5-mechanism stack disappears.
- At the same time F pays for it: the label channel is strictly more informative than ±1, and the
  calibrated arm's retention is **worse**, not better (forgetting same-token-set dppl +0.32, 1/10).

So the honest contrast is not "machinery A on host B" but **a feedback-budget trade**:

| | Paper E | Paper F |
|---|---|---|
| feedback | ±1 correctness only | per-token label |
| where autonomy lives | decision layer | readout layer |
| ceiling of autonomy | rule selection among human-supplied laws | form discovery inside a supplied family |
| mechanism cost | 5 mechanisms, each buying one capability | one objective + one falsifying control |
| measured price | sparse feedback | retention penalty measured in F |

**Rule R4 — state the feedback budget alongside every autonomy claim.**
An autonomy result is only interpretable together with the information the system was allowed to see.
"The system discovered X" is incomplete without "from what signal, and at what cost elsewhere."

---

## 4. The paradigm candidate

> **Self-improvement requires a trustworthy self-evaluation, and the evaluation instrument must be
> learnable rather than prescribed — otherwise the system optimizes a quantity that no longer tracks
> the capability it is improving.**

Both earlier papers are instances of the same failure at different layers: D prescribed the readout
form (artifact in the metric), E prescribed the correction laws (ceiling in the autonomy). F is the
first paper written under the rule instead of demonstrating another mechanism: it takes a signed,
non-probabilistic readout and lets the objective — not the designer — decide what the output form
becomes, then reports what that does to both the metric and the boundary.

**Paper F's four propositions (C1-C3 have data; C4 is the decisive open experiment):**

| ID | Proposition | Evidence |
|---|---|---|
| C1 | The artifact is a property of the objective, not of the numeric range: a trained probabilistic readout removes the floor and lowers stream CE by 0.307 nats/token (ppl −26%), with **no** clip, no floor, no post-processing | s50: 8.648 vs 11.754 (10/10, t=−92.3); accompanied honestly by the retention cost +0.32 (1/10) and mean forget 8.89 vs 8.39 |
| C2 | **Post-hoc calibration does not repair it** — the falsifying control that makes C1 a mechanism claim rather than a number | s50 B-simplex: dppl +0.03 (0/10) with neg_frac 1.582%→0 |
| C3 | Under a clean metric the boundary moves but does not open: state remains stream-neutral (+0.0029, 5/10, t=0.22) and retention-harmful (−1.348, 10/10, t=−6.73); dimension/optimizer is excluded as the cause | s50 + P-F0b (noise, substate arms) |
| C4 | Whether learned selectivity makes the state usable (R3 applied to the state path) | s51 unconstrained gates: **partial** — stream 10/10 better (Gate-C 7.741) but forget 0/10 (31.1 vs B 8.89); gate_mean≈0.72 |
| C4b | Sparsity-pressured gate retains stream win **and** closes retention | **SUPPORTED**: Gate-C-topk stream 7.033 (10/10 vs B, Skip-sub, Gate-C); forget 9.201 (7/10 ≤ Skip-sub; 10/10 better than unconstrained Gate-C; 3/10 vs B, mean +0.31 ≈ parity) |
| C5 | Soft routing over online softmax experts on the C4b base | **SUPPORTED (retention)**: Soft-topk forget 7.789 vs Single 9.201 (−1.41, 10/10, t=−10.4); stream 7.065 vs 7.033 (parity). Hard-topk forget 6.939 (10/10) with small stream cost. Full stack beats B-softmax on **both** axes (stream 7.07 vs 8.65; forget 7.79 vs 8.89). |

**Reporting rules F inherits and must obey** (from the traps):
1. Never quote `floor_frac = 0` for a softmax arm as a result — it is true by construction (softmax
   cannot emit 0). Report `clip_frac` and `neg_frac`; keep the two columns separate.
2. One non-negotiable column per headline: the **post-hoc-fix arm**. No C1-style claim without it.
3. Every upside gets its price in the same sentence (stream gain next to retention cost).
4. Autonomy language is allowed only where a *form* is learned, never where a *rule among supplied
   rules* is selected, and never where the designer chose the family without saying so.

---

## 5. The next boundary (F+1), stated now so F does not overclaim

Rules R1-R4 fix *how the system evaluates itself*. They do not give the system new hypotheses.
Candidate generation is still a human-supplied family (in F: the readout family). By the same
argument that produced Trap 2, the next ceiling is:

> **Once self-evaluation is trustworthy, the binding constraint becomes the breadth of the candidate
> space the system can generate.** Autonomy over *form* inside a supplied family is the last thing
> achieved by improving the readout; going past it requires a trainable mechanism for generating
> candidates rather than for scoring them.

Paper F must state this boundary in its own Discussion, in the same style as D and E: the rule it
establishes, plus the boundary at which the rule stops paying.

**Evidence gaps to declare in F** (no exceptions):
- No multi-step / generative / RL setting; every result is a next-token or classification quantity
  on a two-domain biased-bigram stream (D's s18/s19 protocol).
- CPU-scale: N = 128 state, ~4k readout parameters, one unswept host configuration.
- No external modern-SSM baseline (Mamba/DeltaNet/TTT) — F is a mechanism-instantiation study, not
  a state-of-the-art claim.
- No scaling validation until s53; no text benchmark until s52/s54.
- The decisive experiment (C4) has not run; F's form is therefore not yet fixed.

---

## 6. Status of the traps in the published record

| Item | State | Action |
|---|---|---|
| D's floor-sensitive framing | `PAPER_D_NC.tex` L76-79, L1043-1046 still locate the artifact in the readout range | Reword to scope the sensitivity to the squared-loss + clip readout family; record that F ran the prescription and that the post-hoc variant is falsified |
| E's abstract | Already states autonomy at the decision level as the contribution; 283 words on a 2026-09-14 re-measure (LaTeX and math stripped — the "242/250" figure recorded here earlier was not reproducible, and this journal states no abstract word limit) | No change. E's boundaries stand as scoping, which is what the abstract already says |
| F | No abstract exists (`preprint_titles_abstracts.txt` covers A-D only; E is also absent) | Write after C4's outcome fixes the form |
| F artifacts | `paper_f/`, `scripts/s50_*`, `data/s50_*`, `data/per_token/s50__*` are all git-untracked | Commit before any further runs |

---

## 7. Data provenance

- `data/s50_paper_f_pilot_nl_readout_v1.csv` / `.json` — 80 rows, 8 arms (B-lin-clip, B-simplex,
  B-softmax-sgd, Skip-lin-clip, Skip-simplex, Skip-softmax-sgd, B-noise-softmax, Skip-sub-softmax)
  x 10 seeds. Optimizer: AdaGrad lr=0.05, grad-clip 5.0, softmax T=1.0 (fixed), floor_eps 1e-12.
- `data/s19_ssm_rls_readout_v1.csv` — Paper D's readout arms; the pairing source.
- Read-only verification scripts: `review_workspace/_f_s50_verify.py`, `_f_s50_sameset.py`,
  `_f_s50_headtohead.py`, `_f_s50_stream_identity.py`, `_f_s50_skipdiff.py`,
  `_f_s50_controls_headtohead.py`.
- `paper_d/PAPER_D_NC.tex` (Prop 1, Conclusion L1043-1046), `paper_e/PAPER_E_v3.tex` (R1 L167,
  R8 L1649-1686, abstract L44-68).
