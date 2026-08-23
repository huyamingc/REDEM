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
| P1 | Per-token RLS on the SSM output projection tracks known domain switches at least as well as the s18 Transformer+LoRA A1 arm | RLS-on-SSM stream ppl worse than A1 reference at every tau_m, 0/10 positive | P1 |
| P2 | Routing on the M3 EMA second state improves forgetting on the SSM host (like A3 on the Transformer); gating-only on the SSM host is falsified (like A2) | Routing forgetting diffs <= 0 at every tau_m (0/10 positive), OR gating-only stream ppl diffs <= 0 at every tau_m | P2 |
| P3 | Gradual M4 structural changes beat abrupt ones on forgetting and stream ppl | Abrupt M4 wins (10/10 positive) | P3 |
| P4 | M5 stability regulation keeps stream performance stable under disturbance injection where bare SSM degrades | Bare SSM and regulated SSM degrade equally (no separation, 10-seed) | P3 |
| P5 | The tau-spectrum prior (log-normal A diagonal) beats a uniform A diagonal on forgetting coverage at matched state dim | Uniform A wins (10/10 positive on forgetting) | P1/P4 |

Any falsified prediction is reported (0/10-style) and the mechanism
mapping revised --- same discipline as C's s14/s16/s18.

---

## 6. Roadmap with exit criteria

### P1 - RLS-on-SSM prototype (1-2 w)
- Build a hand-rolled diagonal SSM in torch CPU: state dim N in 64-256,
  d_model 64-128, ~1M params total. Diagonal A with log-normal tau prior
  (Paper A parameters: tau0 ~ 174 "steps" scaled to tokens, CV 0.20).
  Selective dt (Mamba-style, input-dependent) OPTIONAL for P1.
- Per-token RLS on the output projection, O(r^2), rank r = 16-64
  (CLAUDE.md: ML class, torch.manual_seed per trial, Pool only around
  independent trials, no @njit, CSV+JSON dual output).
- Task: s16-style two-domain streaming stream (known switches, 10 seeds).
- Exit: P1 prediction tested; prototype runs; no mamba-ssm dependency.

### P2 - M3 EMA second state + drift detection (1 w)
- EMA second state m_t on h_t; detector + routing arms (A3-analogue) and
  gating-only arm (A2-analogue) on the SSM host; tau_m in {200,500,1000,2000}.
- Exit: P2 prediction tested (host-invariance of the policy lesson).

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
