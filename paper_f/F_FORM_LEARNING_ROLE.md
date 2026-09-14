# Paper F as the Form-Learning Source of the Program

> Purpose: formalize Paper F's design rule, connect C1/C4b/C5 to the
> self + causality program, derive the counterfactual / world-model
> readings of the top-k gate, state the capability boundary from the
> width and k-sweeps, and position s66 as a scoped external check
> (not a SOTA claim) that points to s67 before any Paper H claim.
>
> Status: program-level derivation note. Numbers are the 10-seed
> (or 3–5 seed where noted) means from committed `data/s50_*`,
> `s51_*`, `s52_*`, `s53_*`, `s53b_*`, `s66_*`. Body English;
> conversation summaries may be Chinese.
>
> Citation prerequisite: `paper_f/` and its data/scripts must be git
> tracked before this note is cited from a submission package.

---

## 0. Role in one sentence

Paper F is the first paper in the REDEM program that makes the
**evaluation instrument** a learned object; it is therefore the
program's *form-learning* paper, and the source of every later claim
about generating new forms (Paper H) rather than scoring
human-supplied ones.

| Layer | Paper | What was prescribed | What F changes |
|---|---|---|---|
| Substrate physics | A | the substrate | — |
| Online algorithm | B | the mechanism set | — |
| Mechanism dissection | C | transfer targets | — |
| Host architecture | D | readout form (linear + clip) | diagnoses the trap |
| Decision autonomy | E | correction laws | measures the rule-selection ceiling |
| **Readout / measurement** | **F** | **nothing essential: softmax map, gate, experts are trained** | **the instrument is learnable** |

---

## 1. Design rule: formal statement

### 1.1 Informal rule (from the trap record)

> Self-improvement requires a trustworthy self-evaluation, and the
> evaluation instrument must be **learnable** rather than prescribed.

Papers D and E are two instances of the same failure at different
layers: D prescribed the readout form (metric artifact); E prescribed
the correction laws (autonomy ceiling). F is the first paper written
*under* the rule rather than demonstrating another mechanism.

### 1.2 Operational definition of "learnable instrument"

Let a stream system be a triple
\[
\mathcal{S} = (\theta_{\mathrm{host}},\; \theta_{\mathrm{inst}},\; \theta_{\mathrm{other}})
\]
where

* \(\theta_{\mathrm{host}}\) are host parameters (here: frozen diagonal
  SSM timescales; in s66, optionally trained selective-SSM params);
* \(\theta_{\mathrm{inst}}\) are the **instrument** parameters —
  anything that maps internal state to a *scored* predictive object
  (readout weights \(W\), gate logits \(C,b\), expert mixture
  \(\pi\));
* \(\theta_{\mathrm{other}}\) are remaining trainable objects.

Let \(J(\theta;\mathcal{D})\) be the reported evaluation functional on
data \(\mathcal{D}\) (stream CE, forgetting ppl, …). Call the
instrument **learnable** at a report time \(t\) if and only if:

**(L1) Gradient reachability.** There exists a training update
\[
\Delta\theta_{\mathrm{inst}} \neq 0
\]
that is a measurable function of the same per-token feedback used in
the report (here: the label \(e_t\) through \(\mathcal{L}_t =
-\ln p_t[e_t]\)). A frozen, designer-fixed map has
\(\Delta\theta_{\mathrm{inst}} \equiv 0\) and is *prescribed*.

**(L2) Objective alignment.** The quantity optimized on
\(\theta_{\mathrm{inst}}\) is (a monotone transform of, or the same
as) the quantity reported. In F: the trained objective is CE, and the
reported stream metric is CE / ppl under the same \(p_t\).

**(L3) Post-hoc non-substitutability.** There exists a falsifying
control that leaves \(\theta_{\mathrm{host}}\) and the squared-loss
solution \(W_{\mathrm{RLS}}\) fixed and applies only a range repair
\(\Pi\) after training. If
\[
J(\Pi \circ W_{\mathrm{RLS}}) \not\approx J(W_{\mathrm{trained}})
\]
then the defect lived in the objective, not in the numeric range, and
"make the output non-negative after the fact" is **not** a repair of
the instrument.

### 1.3 How C1 and C2 instantiate the definition

| Arm | What changes | L1 | L2 | L3 evidence |
|---|---|---|---|---|
| **B-softmax-sgd (C1)** | map family *and* objective: \(p=\mathrm{softmax}(W\phi)\), \(\mathcal{L}=-\ln p[e]\) | \(W\) receives CE gradient | report CE on \(p\) | stream \(\Delta\)CE \(=-0.307\) nats/token, ppl \(11.754\to8.648\), \(10/10\), paired \(t=-92.3\) |
| **B-simplex (C2)** | evaluation range only: \(\Pi_\Delta(\hat y)=\max(\hat y,0)/\sum_j\max(\hat y_j,0)\) after RLS | \(W\) still trained by squared error; \(\Pi\) has no parameters | report CE on \(\Pi(\hat y)\) | \(\mathrm{neg\_frac}: 1.582\%\to0\), stream \(\Delta\)ppl \(=+0.03\), \(0/10\) |

**Formal reading of C1.** C1 is an **objective-level** fix: it makes
the instrument gradient-reachable (L1) and aligned with the report
(L2). The clip floor vanishes by construction (softmax cannot emit
0); \(\texttt{floor\_frac}=0\) is structural and is never quoted as an
empirical finding.

**Formal reading of C2.** C2 is a **range-level** control that is
explicitly *not* learnable under (L1): \(\Pi_\Delta\) has no gradient
path into \(W_{\mathrm{RLS}}\). Its failure (\(0/10\)) is the
operational content of (L3): the artifact was in the objective, not
in the range. C2 is therefore what turns C1 from a number into a
mechanism claim.

**Corollary (reporting rule).** Any future "calibration" ablation
that only rescales, clips, renormalizes, or temperature-schedules the
output *after* training is a C2-class arm and must be reported as
such. If it moves nothing, the pathology is in the objective.

### 1.4 Feedback budget (inherited Rule R4)

Every form-learning statement is only interpretable with its
information budget. F uses **per-token labels** via CE — strictly
richer than E's \(\pm1\) channel. F does **not** port E's
five-mechanism decision tower: once a gradient exists, the reason
those mechanisms exist largely disappears. The price of the richer
budget is measured, not assumed: C1's stream gain is accompanied by a
retention cost (same-token-set \(\Delta\)ppl \(+0.32\), \(1/10\)).

---

## 2. From form learning to form generation: path to Paper H

### 2.1 What F actually automates

F learns form **inside** a human-supplied family
\[
\mathcal{F}_0 = \{\text{softmax maps},\; \text{sigmoid+TopK gates},\;
\text{two-expert soft/hard routes}\}.
\]
Within \(\mathcal{F}_0\), the gradient chooses the member. It does
**not** propose a new family. This is the same structure as E's
ceiling, one layer up:

| | Paper E | Paper F | Paper H (target) |
|---|---|---|---|
| Feedback | \(\pm1\) | per-token label | (must be stated; likely label or richer) |
| Autonomy locus | decision layer | readout / gate / routing form | **candidate-family generator** |
| Ceiling | rule selection among 2 human laws | form selection inside \(\mathcal{F}_0\) | generation of \(\mathcal{F}_1 \not\subseteq \mathcal{F}_0\) |
| What is scored | reward rate | CE under the learned instrument | CE (or task utility) under instruments the system itself proposed |

### 2.2 The F+1 step: "learn to generate form"

F already contains the seeds of a generator, but they are still
**selection**, not generation:

1. **Gate logits as a binary search over channel subsets.**
   \(g_t = \mathrm{TopK}_k(\sigma(C x_t+b))\) is a learned mask. The
   *support* \(\{i : g_{t,i}>0\}\) is a form choice. H must be able
   to *propose* new support geometries (block sparsity, timescale
   bands, cross terms), not only reweight channels of a fixed host.
2. **Expert count and references.** C5 hard-codes \(K=2\) domain
   references from Paper D's M3 metadata. H must be able to invent
   new experts and new reference statistics.
3. **Loss form.** F compares CE vs squared-loss+clip as two arms;
   the *choice* was still the experimenter's. H must be able to
   propose scoring rules and falsify them with a C2-class control.

**Definition (candidate-family generator).** A system is a *form
generator* relative to \(\mathcal{F}_0\) if it maintains a proposal
distribution
\[
q_\psi(\mathcal{F}) \quad\text{over instrument families}
\]
and an evaluation instrument \(J\) that is itself learnable under
§1.2, such that some \(\mathcal{F}\) with
\(q_\psi(\mathcal{F})>0\) is not in the closure of
\(\mathcal{F}_0\) under the experimenters' edit distance (new
topology, new cardinality rule, new loss, new expert birth).

### 2.3 Minimal H agenda implied by F's negatives

F's honest negatives already constrain H:

* **C2 → H must ship post-hoc controls for every new form.** A
  generator that proposes a range fix without an objective fix is
  predicted to fail like simplex.
* **C4 → unconstrained CE does not sparsify.** Any H proposal that
  hopes sparsity will "emerge" from task loss alone is predicted to
  fail; cardinality or equivalent pressure must be part of the
  proposed form.
* **s53/s53b → \(k \propto N\) fails.** A generator that scales
  sparsity with width without a budget rule will reproduce the
  \(N=512\) collapse. H needs an explicit sparsity/width law, or a
  search over \(k\) at fixed cost.
* **s66 frozen > joint.** Randomly initialized hosts that are trained
  jointly online can be *worse* than frozen random hosts. H's
  generator must include host-freezing as a first-class option, not
  only "train everything."

### 2.4 What F licenses and what it does not

**Licenses:** that the instrument can be moved inside the learned
system without destroying the metric; that sparsity pressure is
necessary and sufficient (at this scale) to buy stream without
sacrificing retention; that soft expert routing converts residual
stream–retention tension into retention.

**Does not license:** that the family \(\mathcal{F}_0\) is near the
optimal family; that wider hosts help the gated arms under the
current \(k(N)\) rule; that selective-SSM hosts are inferior; any
language-modeling or multi-step RL claim.

---

## 3. Self + causality: C1, C4b, C5 as seeds

The program's longer arc is a system that can **verify**, **select**,
and **organize** itself, then act causally on its own forms. F
supplies three concrete seeds. These are *seeds*, not claims of
autonomy at E's decision level.

### 3.1 C1 — "I can verify myself"

The trained softmax readout is a self-evaluation channel whose
gradient and report share the same object \(p_t\). Once C1 holds:

* a stream CE number is no longer partially an artifact of a
  prescribed clip;
* comparing two *forms* under CE is a comparison of predictive
  distributions, not of clipped scores;
* the system can, in principle, be scored by a map it is currently
  changing, without the metric silently drifting away from the
  capability being improved.

Causal content: **intervention on the objective** (squared-loss+clip
→ CE+softmax) moves the evaluation by \(-0.307\) nats/token, while
**intervention on the range only** (simplex) moves it by
\(\approx0\). That contrast is the causal signature of "the
instrument, not the cosmetic."

### 3.2 C4b — "I can select myself"

Hard top-\(k\) (\(k=32\)) is a learned *selection of which own
channels are allowed to speak* at each token:

\[
g_t = \mathrm{TopK}_k(\sigma(C x_t+b)),\qquad
\phi_t = [\,h^w_t \odot g_t;\; x_t;\; 1\,].
\]

Evidence (10 seeds): Gate-C-topk stream \(7.033\) vs B-softmax
\(8.648\) (\(10/10\)); forget \(9.201\) vs unconstrained Gate-C
\(31.075\) (\(10/10\) better than Gate-C; near B-softmax parity,
mean \(+0.31\)). Unconstrained CE gating improved stream but did not
sparsify (`gate_mean` \(\approx 0.72\)) and destroyed retention.

Causal content: **selection pressure** (cardinality), not capacity or
dimension, is what makes the state path usable under a clean metric.
C3 already ruled out "more dimensions help"; C4 ruled out "CE will
sparsify if you just wait"; C4b is the successful intervention.

### 3.3 C5 — "I can organize myself"

Soft routing over two online softmax experts on the C4b base:

\[
\pi_{t,k} = \mathrm{softmax}_k(-\|m_t-\mathrm{ref}_k\|/T_r),\qquad
p_t = \sum_k \pi_{t,k}\,\mathrm{softmax}(W_k\phi_t).
\]

Evidence (10 seeds): Soft-topk forget \(7.789\) vs single-topk
\(9.201\) (\(-1.41\), \(10/10\), \(t=-10.4\)) at stream parity
(\(7.065\) vs \(7.033\), \(t=0.68\)). Hard routing is the retention
specialist (\(6.939\)) at a small stream tax. Full stack beats
B-softmax on **both** axes (stream \(7.07\) vs \(8.65\); forget
\(7.79\) vs \(8.89\)).

Causal content: once each own specialist is selected (C4b), **organizing**
them under a responsibility-weighted mixture converts the remaining
stream–retention tension into retention. This mirrors Paper D's
soft-vs-abrupt finding under a different readout family — the
organizational principle survives the instrument change.

### 3.4 Program diagram (form layer)

```text
  D: prescribed readout ──diagnoses──► metric artifact
                                          │
  E: prescribed laws ──measures──► rule-selection ceiling
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │  F: learn the instrument                  │
                    │   C1  verify   (objective, not range)    │
                    │   C2  falsify range-fix (0/10)            │
                    │   C4b select   (hard top-k)               │
                    │   C5  organize (soft experts)             │
                    └─────────────────────┬─────────────────────┘
                                          │ boundary
                                          ▼
                    H: generate candidate families (not only score them)
```

---

## 4. Counterfactual engine and world model from the top-k gate

### 4.1 Top-k as a recorded counterfactual

At each token the gate commits to a support
\(S_t \subset \{1,\ldots,N\}\) with \(|S_t|=k\). The hard mask
implements an **implicit counterfactual pair**:

| | support \(S_t\) | complement \(\bar S_t\) |
|---|---|---|
| Feature used | \(h^w_t \odot g_t\) on \(S_t\) | zeroed |
| Gradient | flows into \(C,b,W\) on \(S_t\) | **blocked** through the hard mask |
| Predictive object | \(p_t = \mathrm{softmax}(W\phi_t)\) | not evaluated |

Two readings, both useful for H:

1. **Committed counterfactual.** The system chose \(S_t\) and thereby
   *declared* that the predictive content of \(\bar S_t\) is not
   needed for this token under the current instrument. The declaration
   is falsifiable: if a later domain switch makes retention fail,
   the held-out domain CE (forgetting) is a *measure of the cost of
   that counterfactual bet*.
2. **Logged counterfactual (cheap, not yet run).** Because
   \(\sigma(C x_t+b)\) produces a full soft gate before TopK, the
   ranked scores
   \[
   s_{t,(1)} \ge s_{t,(2)} \ge \cdots \ge s_{t,(N)}
   \]
   are an ordered list of "channels I almost used." The mass just
   below the cutoff is a continuous counterfactual ledger: high mass
   just below \(k\) means the choice was near-tied; a clean gap means
   the bet was sharp. Logging the cutoff gap
   \(\delta_t = s_{t,(k)} - s_{t,(k+1)}\) and the soft-mass fraction
   outside \(S_t\) would give a per-token *uncertainty of selection*
   without a second forward pass.

**What F does not yet do:** it does not evaluate the complement, does
not store \(\delta_t\), and does not train a critic on "what if a
different support had been used." Those are the natural F+1 / H
experiments for a true counterfactual engine. The *structure* is
already present; the *ledger* is not.

### 4.2 Softmax probabilities as a world-model prediction

Under C1 the readout is not a score but a categorical distribution
\[
p_t = \mathrm{softmax}(W\phi_t) \in \Delta^{|V|-1}.
\]
Interpretation as a **local world model**: conditioned on the
instrument's features \(\phi_t\) (gated state, clean input path,
bias), \(p_t(\cdot)\) is the model's one-step predictive belief about
the next symbol. Then:

* \(-\ln p_t[e_t]\) is **surprisal** of the realized symbol under
  that world model — the training signal and the report share this
  object (L2).
* \(\mathrm{CE}_{\mathrm{stream}}\) is the average surprisal of the
  realized stream under the evolving world model; it is a proper
  scoring rule, unlike clipped linear scores.
* Under C5, \(p_t = \sum_k \pi_{t,k} p_{t,k}\) is a **mixture of
  world models** with responsibilities \(\pi_{t,k}\). The
  responsibility-weighted CE gradient is standard mixture
  credit assignment: each expert is a specialist world model
  (domain-conditioned), and \(\pi\) is the inferred context.

**Causal / self-model reading.** The gate \(g_t\) is a policy over
which own state coordinates enter the world model at time \(t\).
C4b's success means: a sparse, learned *instrumentation policy* on
the host yields a better one-step world model on stream *without*
paying the retention bill of an open gate. That is a small, clean
instance of "the system chooses what to know in order to predict."

**Limit of the world-model claim.** \(p_t\) is one-step and
task-local (next symbol on a biased-bigram stream). It is not a
multi-step generative model, not an environment model for planning,
and not comparable to a dynamics model in an RL sense. Saying
"world model" here means *proper one-step predictive distribution
under a learned instrument*.

### 4.3 From counterfactual + world model to H

H's generator can reuse both structures:

* **Proposal = new support / new expert / new map.** Score with the
  C1 world model; accept if stream CE improves without a retention
  violation (C4b's dual-axis criterion).
* **Falsify with a C2-class control** for any proposed post-processing
  of the predictive object.
* **Record near-miss supports** (\(\delta_t\) ledger) as the proposal
  prior: channels repeatedly just below the cutoff are candidates for
  a new expert or a widened \(k\) in that regime only.

---

## 5. Capability boundary: width, k, and what N=512 taught

### 5.1 Facts from s53 / s53b

Width sweep (s53, 5 seeds, \(k=\max(8, N/4)\)):

| Arm | N=128 stream/forget | N=256 | N=512 |
|---|---|---|---|
| B-softmax-sgd | 8.63 / 8.75 | 7.84 / 9.59 | **7.25 / 11.09** |
| Gate-C-topk | **6.96 / 8.50** | 7.18 / 12.14 | 9.16 / 27.30 |
| Soft-topk | 6.88 / 7.12 | 7.58 / 9.92 | 16.37 / 50.06 |

Under the \(k\propto N\) rule the **calibrated input-path baseline
improves with width** while the sparse-gate and soft-routing arms
**fail at N=512**.

k-sweep at N=512 (s53b, 3 seeds):

| Arm | k=16 | k=32 | k=64 | k=128 |
|---|---|---|---|---|
| Gate-C-topk stream / forget | 6.71 / 9.12 | **6.79 / 9.89** | 7.31 / 13.83 | 9.16 / 31.29 |
| Soft-topk stream / forget | **6.59 / 6.27** | 6.87 / 7.62 | 7.78 / 11.07 | 16.14 / 50.80 |

vs N=512 baseline 7.25 / 11.09. **Restoring a fixed modest \(k\)
restores the C4b/C5 ordering and beats the baseline.** \(k=128\)
reproduces the collapse.

### 5.2 Interpretation: not a width impossibility

The N=512 failure of Gate-C-topk under \(k=N/4=128\) is a
**sparsity-budget failure**, not "wide hosts cannot be gated."
Empirical law at this protocol:

* keep \(k\) **fixed and modest** (16–32 at the tested widths), or
* define a **tuned \(k(N)\)** that does **not** grow linearly with
  \(N\);
* do **not** claim \(k \propto N\).

Possible mechanisms (hypotheses, not measured):

1. **Over-inclusion.** At \(k=128\) of \(N=512\), the gate is 25%
   open — enough for domain-mixture leakage that poisons retention
   (same qualitative failure as unconstrained Gate-C at N=128).
2. **Gradient dilution.** Straight-through gradients are split across
   \(k\) channels; larger \(k\) at fixed data length may slow the
   specialization that C4b depends on.
3. **Host spectrum vs budget.** Whitened channels have log-uniform
   timescales; a fixed \(k\) may correspond to a preferred number of
   *useful* timescales for this bigram task, independent of how many
   channels exist.

**F's claim boundary.** Primary tables remain at **N=128**. Scaling
statements must cite s53 *and* s53b together, and must state that
width helps the **ungated calibrated baseline** more reliably than it
helps the gated arms under the original \(k\) rule.

### 5.3 s67 (completed): frozen / pretrained / online host

s66's internal ordering — **random frozen selective-SSM ≫ jointly
trained selective-SSM** — required an immediate follow-up before any
Paper H claim that generates new host forms.

| Arm | stream | forget | note |
|---|---|---|---|
| F Gate-C-topk | **7.03** | 9.20 | best stream |
| F B-softmax-sgd | 8.65 | 8.89 | calibrated anchor |
| X-SelSSM-frozen | 9.03 | 10.38 | strongest external; loses to B on both axes |
| X-SelSSM-full | 10.19 | 115.24 | joint training destructive (matched lr) |

**s67 design** (`scripts/s67_host_freeze.py`,
`data/s67_host_freeze_v1.*`, 10 seeds, same protocol; anchors bit-exact
to s66/s50/s51, 80/80 pairs, `worst_rel=0`):

| Arm | stream | forget | params |
|---|---|---|---|
| X67-random-frozen | 9.030 | 10.378 | 33,088 |
| X67-pretrained-frozen | 9.581 | **84.714** | 33,088 |
| X67-online | 10.191 | **115.238** | 33,088 |
| X67-random-frozen-topk | 7.378 | 11.939 | 49,600 |
| X67-pretrained-frozen-topk | 7.177 | 14.978 | 49,600 |
| F-B-softmax-sgd | 8.648 | 8.891 | 4,128 |
| F-Gate-C-topk-softmax | **7.033** | 9.201 | 24,736 |

`pretrained-frozen` trains the host **only on the first segment**
(`t ≤ SEG_LEN=3000`, domain A), then freezes; readout stays online.
`online` uses matched `SOFTMAX_LR` only (s66's divergent 4× rate is
not re-run).

**Pre-registration outcomes (10 seeds):**

| ID | Question | Result |
|---|---|---|
| Q1 | pretrained-frozen better than random-frozen on forget? | **NO** — Δ +74.3, **0/10** (prediction falsified) |
| Q2 | pretrained-frozen better than online on forget? | **YES** — Δ −30.5, **10/10**, t=−8.5 |
| Q3 | random-frozen+topk stream not worse than Gate-C-topk by >+5%? | **NO** — Δ +0.35, **0/10** (≈5% worse, not parity) |
| Q4 | pretrained-frozen+topk stream ≥ Gate-C-topk? | **NO** — Δ +0.14, **3/10** (near-parity, not a win) |

**Interpretation (answers the pre-H questions):**

1. The frozen advantage is **optimisation interference**, not host
   quality. Even a *single segment* of joint CE on the host, then
   freeze, destroys retention (84.7 vs random-frozen 10.4, 0/10).
   "Pretrain then freeze" is worse than "never train the host" here.
2. Freezing after burn-in is still better than always-online (Q2),
   so post-burn-in joint updates are a large part of the damage — but
   they are not the only part.
3. F's top-k gate **recovers most of the stream gap** on a frozen
   selective host (9.03 → 7.38 / 7.18) but does **not** beat the
   diagonal-host Gate-C-topk (7.03). At this protocol the diagonal
   host remains the better substrate for the sparse gate.
4. Paper H must not default to online host training. If H proposes
   host updates, they need an explicit freeze / consolidation stage
   and a retention pre-commit.

s67 is the **last external-check experiment** implied by F. With s67
committed, Paper H may begin under the constraints above.

---

## 6. External baseline (s66): scoped check and the frozen-host surprise

### 6.1 What s66 is

A Mamba-style selective-SSM arm on the **identical** host protocol,
stream, 10 seeds, and metric as F, plus F's own anchors, in one
script (`s66_external_ssm_baseline`, `data/s66_external_ssm_baseline_v1.*`).
Anchor validation: bit-exact pairing to committed s50/s51 rows
(`status: ok`).

Committed means (10 seeds; external rows appear twice in the JSON
because of an lr-grid pooling — use the non-divergent matched-lr
block for the full arm):

| Arm | stream | forget | trainable params (approx) |
|---|---|---|---|
| F-B-lin-clip | 11.754 | 8.387 | 4,128 |
| F-B-softmax-sgd | 8.648 | 8.890 | 4,128 |
| F-Skip-sub-softmax | 8.142 | 11.231 | — |
| **F-Gate-C-topk-softmax** | **7.033** | 9.201 | 24,736 |
| X-SelSSM-frozen | 9.030 | 10.378 | ~33,088 (external stack) |
| X-SelSSM-full (matched lr) | **10.19** | **115.24** | ~33,088 |

External arm carries \(\sim1.34\times\) F's **Gate-C-topk** trainable
parameters (33,088 vs 24,736). That ratio is **not** \(1.34\times\) F's
ungated B-softmax-sgd anchor (4,128); against B the external arm is
much larger. The gate win on stream is therefore not a capacity fluke
*on this comparison*.

### 6.2 Positioning (must stay scoped)

s66 **closes** the program gap item
"no external modern-SSM baseline" **for the selective-SSM family
only**. It is a **scoped external check**, not a SOTA victory.

**Not claimed:** superiority of F over Mamba/TTT/DeltaNet in general;
scaling; language modeling; multi-step RL; any host architecture
ranking beyond this one configuration, one protocol family, N=128,
CPU scale.

**Closed gap:** `PAPER_F.tex` Limitations / Discussion no longer says
"no external selective-SSM baseline." The earlier generic "no
Mamba/DeltaNet/TTT" sentence remains only as a statement that DeltaNet,
gated linear attention, and a *viable* TTT arm were **not** run: the
TTT fast-weight arm scored worse than the uniform predictor and was
removed as non-viable.

**Disclosed defect (must remain in Limitations):** a \(4\times\)
larger host learning rate made joint training diverge; the committed
s66 JSON pools both rates under one arm name. The manuscript's quoted
full-arm stream \(10.19\) is the **matched-lr** value.

### 6.3 The frozen-host surprise (why it matters more than the win)

The informative result is **internal to the external pair**:

\[
\text{random frozen selectivity} \;\gg\; \text{jointly trained host}
\]
on both stream (9.03 vs 10.19) and especially retention (10.38 vs
115.24).

This is not "selective SSMs are bad." It is evidence that, at this
protocol scale and learning-rate regime, **online joint training of
the host is the destructive component**, while a frozen (even random)
selective feature map is a viable competitor that still loses to F's
learned top-k gate on stream (\(0/10\) external better on 10 paired
seeds).

That ordering is exactly why s67 (frozen / pretrained / online) is
ordered **before** Paper H: H must not build a generator that
jointly trains hosts by default.

### 6.4 One-line summary for the abstract/discussion sentence

> A selective-SSM baseline on the same stream, seeds, and metric does
> not close the gap to F's learned top-\(k\) gate (stream \(7.03\) vs
> \(9.03\), \(0/10\)); joint online training of that host is
> destructive; the frozen arm is the stronger competitor. We treat
> s66 as a scoped check on the selective-SSM parameterisation at this
> scale, not as a state-of-the-art comparison.

---

## 7. Limitations (capability boundary of F as form learning)

1. **Width-specific.** Primary claims at N=128. At N=512 the
   \(k\propto N\) rule fails (Gate-C-topk stream 9.16, forget 27.3);
   a fixed modest \(k\) restores the ordering (k=32: 6.79/9.89;
   Soft-topk k=16: 6.59/6.27). Claims require a fixed modest \(k\) or
   a tuned \(k(N)\), not \(k\propto N\).
2. **No standard benchmark.** Two-domain biased-bigram stream plus a
   Mackey–Glass bin task. No LM benchmark, no long-context suite, no
   RL benchmark.
3. **No DeltaNet / gated linear attention / viable TTT comparison.**
   s66 covers one selective-SSM parameterisation only; TTT was
   removed as non-viable at this scale.
4. **Host mostly frozen.** F's diagonal SSM timescales are not
   trained. s66 full and s67 online show joint host CE is destructive;
   s67 pretrained-frozen (1 segment then freeze) is also retention-
   destructive (forget 84.7, 0/10 vs random-frozen). The licensed
   statement is *do not train this host online under CE at this
   scale*, not a general host-design recommendation.
5. **Feedback budget is per-token labels.** Results are not
   comparable to E's \(\pm1\) autonomy without stating that budget.
6. **Family breadth is still human-supplied.** F learns form inside
   \(\mathcal{F}_0\); it does not generate new candidate families.
   That is H's job; s67 is now complete and licenses H under a
   default-frozen-host constraint.
7. **s66 rate-pooling disclosure.** Committed full-arm rows pool a
   matched-lr run and a divergent \(4\times\) lr run under one name;
   quotes must use the matched-lr value and disclose the pooling.
8. **CPU-scale proof of concept.** No scaling, no SOTA, no device
   claim. Softmax `floor_frac=0` is structural.

---

## 8. Immediate actions implied by this note

| Priority | Action | Why |
|---|---|---|
| P0 | **Git commit** paper F package + s67 | Citation prerequisite |
| P1 | Fix the informal "frozen beats B-softmax" phrasing wherever it appears; frozen loses to B on stream (9.03 vs 8.65) and on forget (10.38 vs 8.89) | Prevents an overclaim |
| P1 | Keep s66 Limitations wording as scoped check + lr-pooling disclosure | Already in `PAPER_F.tex`; do not "upgrade" to SOTA |
| P2 | ~~Design and run **s67**~~ **DONE** (bit-exact anchors; Q1/Q3/Q4 falsified, Q2 supported) | Last gate before Paper H |
| P3 | Optional F+1 logging: record top-k cutoff gap \(\delta_t\) and outside-mass as a counterfactual ledger | Cheap; feeds H's proposal prior |
| P3 | Paper H spec: candidate-family generator under §2.2, C2-class falsifying controls mandatory, **default host frozen** | Form generation, not another scorer |

---

## 9. Data anchors

| Experiment | File | Role |
|---|---|---|
| C1–C3 | `data/s50_paper_f_pilot_nl_readout_v1.*` | objective vs range; state boundary |
| C4/C4b | `data/s51_paper_f_learned_gate_v1.*` | unconstrained vs top-k vs L1 |
| C5 | `data/s52_paper_f_soft_route_experts_v1.*` | soft/hard expert routing |
| Width | `data/s53_paper_f_scaling_v1.*` | N=128/256/512, \(k=\max(8,N/4)\) |
| k-sweep | `data/s53b_paper_f_ksweep_v1.*` | N=512, k∈{16,32,64,128} |
| Transfer | `data/s54_paper_f_mackey_glass_v1.*` | Mackey–Glass bin task |
| External | `data/s66_external_ssm_baseline_v1.*` | selective-SSM scoped check |
| Host freeze | `data/s67_host_freeze_v1.*` | random / pretrained / online host policy |

Manuscript: `paper_f/PAPER_F.tex` (Design rule §Analysis; s66 +
limitations §Discussion; s53/s53b in Limitations).
Paradigm authority: `paper_f/PARADIGM_AND_RULES.md` (R1–R4).
