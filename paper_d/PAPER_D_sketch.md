# Paper D - Design Sketch: REDEM-SSM

**Working title**: REDEM-SSM: A Foundation Model Architecture with Native
Online Learning, Meta-Adaptation, and Structural Plasticity

**Status**: sketch (2026-02-18). No experiments yet. This document is the
design contract for the paper; every mechanism mapping below is a
*design claim* to be validated or falsified by the P1-P5 experiments, not
an established result.

---

## 1. Positioning and core question

Papers A-C establish three mechanisms (M1 online readout, M3 statistical
metadata, M4 structural plasticity, M5 stability homeostat) on a physical
reservoir, and show (Paper C §6, s18) that on a Transformer substrate only
a narrow instantiation transfers: domain routing helps (forgetting -28%,
stream ppl -1.07 at tau_m <= 500) while abrupt gating fails everywhere
(0/10 seeds). Paper C §7 point (6) concludes that the *full* framework does
not instantiate on a frozen-feedforward host.

**Paper D question**: what does a foundation model look like when the host
is chosen so that the REDEM mechanisms are *native operations* rather than
retrofits? We propose a state-space host (linear recurrence
h_t = A h_{t-1} + B x_t, Mamba-style selectivity) and instantiate M1/M3/M4/M5
at the mechanism level.

**Contribution claims (to be validated, not asserted)**:
1. The A matrix diagonal is a *synthesized tau spectrum*: Paper A's
   log-normal tau distribution becomes the A-diagonal, giving the host an
   explicit multi-timescale memory with a physically motivated prior.
2. A per-token RLS update on the output projection (O(r^2) in feature/rank
   dimension, per Paper B cost analysis) gives a continuously active online
   readout as a native operation of the linear recurrence.
3. An EMA of the hidden state as a *second state* (M3) provides
   timescale-controlled statistical memory that selects *which* readout is
   active (routing), not *when* to pause learning (the C §6 lesson).
4. Dynamic state-dimension activation and A-spectrum prune-grow (M4)
   implements structural plasticity with "gentle wins" (gradual switching).
5. An eigenvalue/state-norm monitor regulates the effective timescale (dt
   or A diagonal) to hold the host at a target stability band (M5) ---
   *new research*, the linear-stability analogue of the substrate homeostat
   (D2 boundary: not a validation of Paper A/B's FTLE homeostat).

---

## 2. Why SSM: host properties vs REDEM requirements

| REDEM requirement (from C §7 pt 6) | Transformer host | SSM host (Mamba-style) |
|---|---|---|
| Continuously active online readout (M1) | only via adapter retrofit (LoRA), update cost O(F^2) in adapter rank; works but bolted on | linear recurrence makes per-token output projection the natural update point; RLS is O(r^2) in projection rank |
| Slowly integrating statistical memory (M3) | limited by context window (4k-8k tokens); s18 showed the benefit dies at tau_m >= 1000 | unbounded hidden state with tunable decay; EMA second state is a native two-timescale hierarchy |
| Gradually applied structural switches (M4) | fixed dense attention; adapters are static ranks | A-spectrum / state-dimension are explicitly manipulable per step |
| Stability regulation (M5) | no timescale knob | eigenvalues of A (or effective dt) are a direct, measurable stability knob |

**Caveats stated up front**:
- A linear SSM is NOT chaotic; "edge of chaos" (Paper A) does not port. The
  M5 analogue is the *edge of stability* (|lambda| -> 1, bounded state-norm
  growth), a stability-edge controller, not a chaos controller.
- "Parallel training + linear inference" is NOT "training = inference".
  SSM parallel training is offline gradient training; M1 is the online
  per-token update at inference. Both are distinct and both are claimed.

---

## 3. Mechanism mapping (corrected design table)

| REDEM mechanism | Native SSM instantiation | Design claim (to test) |
|---|---|---|
| Substrate physics: log-normal tau spectrum | A diagonal = -1/tau_i, tau_i ~ LogNormal(tau0, CV) per Paper A; log|lambda| = effective decay rates | The tau spectrum prior gives the host multi-timescale coverage out of the box; coverage, not capacity, is the bottleneck (C thesis) |
| M1: online RLS readout | RLS on the output projection (or its low-rank factor): per-token covariance update, O(r^2) | Per-token RLS on an SSM tracks domain switches at least as well as on a Transformer+LoRA reference (s18 A1 baseline) |
| M3: metadata slow trace | EMA second state m_t = (1-1/tau_m) m_{t-1} + (1/tau_m) h_t | Routing on m_t transfers (positive n_pos/10 for forgetting) at tau_m in a broad window; gating-only on the SSM host FAILS like A2 (host-invariant policy lesson) |
| M4: structural plasticity | Dynamic state-dimension activation (prune low-contribution dims, grow new ones) and/or A-spectrum sparsity prune-grow; gradual ("gentle wins") application | Gradual structural changes beat abrupt ones on forgetting and stream ppl (10-seed sign consistency) |
| M5: stability homeostat | Monitor state-norm growth / log|lambda| of A; adjust effective dt (or A diagonal) to hold a target band | Regulation keeps stream performance stable under disturbance injection where a bare SSM degrades (new research, D2 boundary) |

---

## 4. Novelty vs prior art (to be verified before citing in the paper)

Online learning inside SSMs is an active area; Paper D must position
against it precisely:

| Work | What they do | Where Paper D differs |
|---|---|---|
| Longhorn (Liu et al., ICLR 2025; also NeurIPS 2024 virtual): "State Space Models are Amortized Online Learners" | SSM weights *amortize* an online-learning computation across training | D keeps an *explicit* per-token RLS readout at inference (no amortization claim) and adds mechanism-level M3/M4/M5 |
| Titans (Behrouz et al., NeurIPS 2025): "Learning to Memorize at Test Time" | Neural long-term memory module with surprise-based online update | D's memory is a *statistical* EMA (timescale-controlled, mechanism role = drift detection/routing per C thesis), not a surprise-triggered episodic buffer; D adds explicit stability regulation (M5) and structural plasticity (M4) |
| TTT (Sun et al., ICLR 2024): "Learning to (Learn at Test Time)" | Hidden states updated by gradient descent at test time | D uses second-order RLS (O(r^2), no backprop at inference) and a division-of-labor (three mechanisms) rather than a single self-supervised update rule |
| DeltaNet / RWKV / linear-attention families | Learnable linear recurrences with update-rule-like mechanics | D's contribution is the mechanism *division of labor* + stability-edge regulation, not the recurrence form |
| Test-time SSM alignment for user-interest shifts (sequential recommendation) | Test-time adaptation of SSM embeddings | Application-adjacent; D's task is the controlled drift/falsification protocol of C |

Claimed novelty, in one sentence: *a foundation-model architecture whose
host makes the C-thesis division of labor (M3 routes, M4 reshapes, M5
regulates, M1 always learns) native, validated under the 10-seed
falsification discipline.*

Bibliography (verify exact venue/year before citing in the paper):
- Longhorn: https://neurips.cc/virtual/2024/106432 ;
  https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-bo_liu_iclr_2025.html
- Titans (NeurIPS 2025): https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html
- TTT: arXiv:2406.07530 (Sun et al., ICLR 2024)
- Test-time SSM alignment (recsys): https://slideslive.com/39045832/testtime-alignment-with-state-space-model-for-tracking-user-interest-shifts-in-sequential-recommendation
- Mamba: arXiv:2312.00752 (Gu & Dao, 2023); S4: arXiv:2111.00396 (Gu et al., ICLR 2022)
- DeltaNet: arXiv:2406.06484 (Yang et al., 2024)

---

## 5. Falsifiable predictions (pre-registered, 10-seed sign consistency)

| # | Prediction | Falsification criterion | Phase |
|---|---|---|---|
| P1 | Per-token RLS on the SSM output projection tracks known domain switches at least as well as the s18 Transformer+LoRA A1 arm | RLS-on-SSM stream ppl worse than the A1 reference on every seed (0/10 improved) | P1 |
| P2 | Routing on the M3 EMA second state improves forgetting on the SSM host (like A3 on the Transformer); gating-only on the SSM host is falsified (like A2) | Routing forgetting diffs never negative (0/10 improved) at some tau_m, OR gating-only stream diffs never negative (0/10 improved) at every tau_m | P2 |
| P2 (result) | **PARTIALLY REPLICATED + one inversion** (10 seeds, 90 runs): routing improves forgetting at tau_m<=1000 (10/10, up to -2.05 ppl) - SUPPORTED; gating-only IMPROVES stream ppl 10/10 at every tau_m (opposite of s18's 0/10) - the "gating falsified" prediction itself is falsified. The pause-learning policy's effect is READOUT-DYNAMICS-DEPENDENT (Adam-LoRA: hurts; near-batch RLS: helps, stays near the pooled optimum). |
| P3 | Gradual M4 structural changes beat abrupt ones on forgetting and stream ppl | Abrupt M4 wins (10/10 positive) | P3 |
| P4 | M5 stability regulation keeps stream performance stable under disturbance injection where bare SSM degrades | Bare SSM and regulated SSM degrade equally (no separation, 10-seed) | P3 |
| P5 | The tau-spectrum prior (log-normal A diagonal) beats a uniform A diagonal on forgetting coverage at matched state dim | Uniform A wins (10/10 positive on forgetting) | P1/P4 |

Any falsified prediction is reported (0/10-style) and the mechanism
mapping revised --- same discipline as C's s14/s16/s18.

---

## 6. Roadmap with exit criteria

### P1 - RLS-on-SSM prototype (DONE 2026-02-18, FALSIFIED with mechanism isolated)

**Build**: `scripts/s19_ssm_rls_readout.py` (ML class, torch CPU) - diagonal
SSM (N=128), per-token RLS on the output projection (O(F^2), F=N+1),
identical s18 task/generators/seed rules (paired comparison valid).

**Result (10 seeds)**: P1 prediction FALSIFIED for every state-feature arm
- stream ppl 58-116 vs A1 15.01, improved 0/10 (LN-raw 115.0, LN-whiten
115.9, CV-whiten 81.3, CV-skip 58.5). The B-proj CONTROL (RLS on the
current-token projection B e_{t-1}, no state) reaches 11.75 - better than
A1 on 10/10 seeds (paired -3.26), forgetting -0.55 (10/10), oracle 7.25 =
the pooled-table ceiling. The in-sample ridge oracle on the SSM-state
features is ~31 (~uniform) vs 7.25 on the projection features.

**Mechanism (isolated)**: the failure is the STATE MIXING, not M1. A
linear readout cannot recover the current-token component from a
linearly-decayed state mixture: it would need U*Lambda^k = 0 for k>=1
while U*Lambda^0 != 0, impossible for a diagonal decay with A_i > 0
(deconvolution impossibility); older tokens contaminate at Lambda^k and
whitening fixes scale, not the mixing structure. Empirically the in-sample
ridge is equally stuck (oracle ~31 ~ uniform), and giving the readout the
current token directly (B-proj) restores tracking to the ceiling.

**Design implication (evidence-based)**: the diagonal-SSM host requires a
NONLINEAR/GATED output path (Mamba-style selectivity: y = C h (*) g(x))
to give the readout access to the current token - this is exactly the P3
subject, now quantitative. Additional findings: (a) the naive log-normal
spectrum (tau0=174, CV=0.20) has no fast channels and, unwhitened,
conditioning blows up (RLS P grows as (1/lambda)^t, W norm ~1.7e4,
catastrophic held-out ppl); (b) whitening fixes the conditioning but not
the representation; (c) a metric bug was found and fixed during P1 - the
squared-loss readout output is a linear MMSE distribution estimate, NOT a
softmax logit vector; softmaxing it squashed every arm toward uniform
(ppl ~26-31 ~ uniform vs the 7.34 MLE-table ceiling). All numbers above
use the corrected metric (CE = -ln(clip(y_hat[target]))).

**Exit**: prototype runs; no mamba-ssm dependency; P1 prediction tested
and reported honestly (falsified). Data: `data/s19_ssm_rls_readout_v1.csv`
/ `.json` (50 rows).

### P3a - input-gated (selective) readout (DONE 2026-02-18, FALSIFIED)

Executed BEFORE P2 on P1's evidence (user gate). Tests whether a
Mamba-style multiplicative gate makes the diagonal-SSM state readable by
the RLS readout: y = W (h_w * sigmoid(gamma * C x + b)), x = B e_{t-1},
C fixed random per seed, gamma in {1, 5}. Gate is applied at the FEATURE
level so the RLS readout stays linear-in-features/closed-form; the
literal output-gated form y = (W phi) (*) sigmoid(Cx+b) would break the
squared-loss readout's distribution interpretation (mass < 1) and is not
tested.

**Result (10 seeds, 70 runs)**: FALSIFIED - stream ppl 77.2 (CV-gate)
and 72.2 (CV-gate-g5) vs A1 15.01, improved 0/10; oracle 37.1 / 38.5
(sharper gate is WORSE - the negative is not a tuning artifact).
Feature-isolation diagnostics (seed 0): the current-token projection
B e_{t-1} alone reaches the pooled-table ceiling (oracle 7.34; RLS
11.75, 10/10 better than A1), and EVERY state-feature combination (fast
channels only, full state, gated at gamma 1/3/5, with/without B e)
degrades the pooled prediction (oracle 18.4-60.8) - the state carries
domain level and recent-token mixture that are uninformative noise for
the pooled-bigram readout, and near-constant slow columns poison
conditioning.

**Conclusion**: (1) the RLS readout's job is the INPUT path - the current
token must enter the readout as a clean ADDITIVE feature (B e); it
cannot be recovered from the linearly-mixed state by any FIXED gate; (2)
a fixed gate carries the prev code only multiplicatively inside a
time-varying state - noise, not signal; (3) Mamba's selectivity works
because the gate and state are LEARNED end-to-end (gradient), which
breaks the RLS-only constraint - a design boundary to flag, not to cross
silently; (4) the state's value is the DOMAIN LEVEL for M3 metadata
(P2), not the next-token readout.

### P2 - M3 EMA second state + drift detection (DONE 2026-02-18, partially replicated + one inversion)

**Build**: `scripts/s20_ssm_m3_routing.py` (ML, torch CPU). Host per
P1/P3a: readout = additive input path [B e; 1]; M3 metadata = per-token
EMA second state m_t = (1-1/tau_m) m_{t-1} + (1/tau_m) h_w,t[fast] (the
FAST-CHANNEL whitened state, tau<=8). Metadata feature note (bug found and
fixed during P2): the FULL whitened-state EMA is NOT a stationary domain
statistic - slow channels (tau up to 3000) accumulate over the whole
stream, so the EMA drifts away from any fixed per-domain reference and the
detector never flips; fast channels converge in a few tokens, are
stationary, and separate the domains cleanly (||ref0-ref1|| ~1.2, detector
tracks all 5 known switches with ~150-800-token lag). Arms: A1 bare
(single RLS readout), A2 gate-only (RLS error scaled 1.0 for tau_m after a
detected switch, 0.10 within a domain - the "pause-learning" claim), A3
routing (two RLS readouts, slow-trace selects active for prediction AND
update; inactive fully frozen). Task/seed rules/metrics verbatim from s18.

**Result (10 seeds, 90 runs)**: A1 reproduces s19 B-proj exactly
(11.754/8.387 - cross-check). A3 (routing) improves forgetting at
tau_m in {200,500,1000}: -2.05/-1.87/-1.20 ppl, 10/10 seeds; neutral at
2000 (-0.007, 5/10) - the "routing transfers" lesson HOLDS on the SSM
host. A3 stream ppl is worse than A1 (+0.38..+4.08, 0/10): the
domain-specialized readouts pay the detection-lag penalty (wrong
specialist predicts during the ~tau_m lag) - a specialization-vs-lag
tradeoff NOT seen on the transformer (s18 A3 improved stream at fast
tau_m). A2 (gating-only) IMPROVES stream ppl 10/10 at EVERY tau_m
(-1.52..-2.45) - the OPPOSITE of s18 (0/10) - with negligible forgetting
gain: on a near-batch RLS readout (lambda=0.9999), suppressing within-
domain updates keeps the readout near the pooled-table optimum, whereas on
s18's Adam-LoRA readout suppression let within-domain knowledge decay.

**Interpretation**: (1) M3 transfers: the fast-channel state EMA is a
working domain statistic on the SSM host, and routing retains domain
specialists (forgetting -2.05, 10/10). (2) The Paper C Sec 6.4 lesson
"the readout must remain active throughout the stream" is
READOUT-DYNAMICS-DEPENDENT, not host-invariant: it holds for fast
adaptive readouts (Adam/LoRA) but not for near-batch RLS readouts where
pausing is benign-to-beneficial. Paper C's wording may need a qualifier
(user decision). (3) Routing's stream-vs-forgetting tradeoff is
host-specific (specialists + detection lag). Data:
`data/s20_ssm_m3_routing_v1.csv` / `.json` (90 rows).

### P3 - M4 + M5 (2-3 w)
- M4: state-dimension activation scores -> gradual prune/grow; or
  A-spectrum sparsity. M5: state-norm growth / log|lambda| monitor ->
  adjust effective dt to hold a target band; disturbance injection
  protocol (C s11-style) adapted to tokens.
- Exit: P3 and P4 predictions tested.

### P4 - Benchmarks (1-2 w)
- Synthetic multi-domain streams (s16 protocol generalized to several
  domains/switch schedules) + small char-level corpora (s18 style, ~1MB
  text). Arms: bare SSM vs REDEM-SSM vs Transformer+LoRA reference (reuse
  s18 baseline data where possible). 10 seeds, paired diffs, n_pos/10.
- Honest scope: no WikiText-103, no scaling claims.

### P5 - Paper (2-3 w)
- arXiv + workshop first; NeurIPS/ICML stretch; mechanism-oriented journal
  fallback. Limitations section: toy scale, CPU, synthetic tasks, linear
  (non-chaotic) host, tau_m hyperparameter.

---

## 7. Risks and open questions

1. RLS on a *diagonal* SSM output may be near-trivial (the recurrence is
   linear); the interesting failure mode is that the mechanisms matter
   only under disturbance/switch pressure --- the P-tests are designed
   around that.
2. The "host-invariance" claim (P2) risks being trivially true; the
   informative comparison is *magnitude* (does the SSM host widen the
   tau_m window beyond the Transformer's tau_m <= 500?), reported honestly
   either way.
3. M5 as stability-edge regulation may be redundant with selective dt;
   the design must show separation (regulation helps where fixed
   selectivity fails).
4. Venue expectations: toy-scale PoC will not clear NeurIPS/ICML main
   track without scaling evidence; the honest fallback (workshop/arXiv +
   journal) is part of the plan, not an afterthought.

## 8. Decision log

- 2026-02-18: Paper D launched (user gate, option A). Scaffold created:
  README.md + PAPER_D_sketch.md. Calibrated host-boundary paragraph added
  to Paper C §7 point (6). Literature anchors verified via web search
  (Longhorn, Titans, TTT, test-time SSM alignment). Venue target
  NeurIPS/ICML recorded as stretch; arXiv+workshop first.
- 2026-02-18: **P1 (S19) DONE - FALSIFIED, mechanism isolated.** Stream
  ppl 58-116 on every state-feature arm (0/10 improved vs A1 15.01);
  oracle ~31 (~uniform) vs 7.25 on the current-token projection control
  (which reaches 11.75, 10/10 better than A1). Conclusion: M1 (RLS) works;
  the linear readout cannot use the linearly-mixed diagonal-SSM state
  (deconvolution impossibility); the host needs a nonlinear/gated output
  path (Mamba-style selectivity) - P3 is now evidence-driven. Also fixed a
  metric bug during P1 (double-softmax of the linear-MMSE readout output).
  Next: P2 (M3 EMA second state + routing on a host that can represent
  the current token), or P3 (selectivity) - user gate.
- 2026-02-18: **P3a (S19 extension) DONE - FALSIFIED.** Fixed Mamba-style
  multiplicative gates (gamma 1 and 5) do NOT make the state readable:
  stream ppl 77.2/72.2 vs A1 15.01 (0/10 improved), oracle 37.1/38.5
  (sharper gate worse). Feature-isolation shows the current-token
  projection B e is the only useful readout feature (oracle 7.34, RLS
  11.75 10/10) and every state-feature combination degrades the pooled
  prediction. Conclusion: the RLS readout needs the additive input path;
  fixed gates carry the prev code only as multiplicative noise inside the
  time-varying state; Mamba-style selectivity must be LEARNED (gradient),
  a design boundary for the RLS-only constraint; the state's value is the
  domain level for M3 (P2). Next: P2 (M3 EMA + routing) - user gate.
- 2026-02-18: **P2 (S20) DONE - partially replicated + one inversion.**
  M3 metadata (fast-channel state EMA) transfers to the SSM host: the
  detector tracks all 5 known switches, and routing (A3) improves
  forgetting at tau_m<=1000 (-2.05/-1.87/-1.20 ppl, 10/10), neutral at
  2000 - "routing transfers" HOLDS. Gating-only (A2) IMPROVES stream ppl
  10/10 at every tau_m (-1.52..-2.45) - the OPPOSITE of s18 (0/10): the
  pause-learning policy is readout-dynamics-dependent (near-batch RLS
  stays near the pooled optimum; Adam-LoRA decays). This qualifies Paper C
  Sec 6.4's "the readout must remain active" lesson (user decision on a
  qualifier). A3 stream is worse (+0.38..+4.08): specialization-vs-lag
  tradeoff not seen on the transformer. Also fixed during P2: the full
  whitened-state EMA is non-stationary (slow-channel accumulation, detector
  never flipped) - the metadata uses the fast (stationary) channels.
  Next: P3 (M4 + M5/learned selectivity), P4 benchmarks - user gate.
