# Paper C — Extension Analysis: Can the Disentanglement Thesis Apply to Large Models?

**Status**: strategic analysis (no experiments run for this extension yet).
The deduction below is anchored on Paper C's *committed empirical results*
(S10/S11/S14/S15/S16/S16b/S5b), not on aspirations. The question: do the
three mechanisms — metadata (M3), homeostat (M5), substrate physics — and
the disentanglement thesis itself transfer to Transformer/LoRA systems, and
how would a proof-of-concept (PoC) be designed under this project's
CPU-only constraint?

---

## 1. What "applying Paper C to large models" can and cannot mean

Paper C's thesis is *negative in a precise way*: mechanisms are
non-transferable between jobs. The same discipline must apply to the LLM
extension. A PoC is not "another LoRA variant that beats Online-LoRA /
SLoRA"; it is a *validation of the system-architecture principles* —
which component should monitor drift, which should regulate stability,
which should learn content — plus an honest report of the dimensions where
transfer fails.

Practical constraint: this project is CPU-only (torch CPU, OpenBLAS 24
threads, no GPU). The repo already runs a tiny transformer baseline on CPU
(Paper B S9, `scripts/baseline_showdown.py`), so a small-scale LoRA-style
PoC is feasible; a GPT-2-124M-scale study is not (CPU fine-tuning of 124M
parameters is hours-to-days per run).

## 2. Mechanism mapping table

| Paper C mechanism | Paper C evidence (what it does) | LLM/Transformer counterpart | Transfer strength |
|---|---|---|---|
| M3 slow trace (EMA of states) | statistical memory: equalizes accuracy on regime tasks (S10, +0.24 pp), detects drift fast (S15/S5b: T40 49.9->40.6), denoises state-level corruption (S16b V2: gap -0.69 -> -0.01); NOT episodic (MC unchanged) | EMA of hidden states / attention outputs = cheap drift detector + domain-statistics feature | **Strong candidate** |
| M5 homeostat (FTLE-based kappa) | disturbance recovery ONLY: +32% sequential (S11), +18% r1->r3 (S14); needs a substrate Lyapunov estimate | needs an LLM "instability scalar" + a single "plasticity knob" (LoRA scale? LR? temperature?); no FTLE analog defined | **Weak / must be redefined** |
| M4 plasticity (correlation-guided rewiring) | gentle rewiring +8-11%, aggressive -23%; causally clean (O4) | LoRA module selection / adapter routing guided by activation correlation | Partial (principle only) |
| Substrate physics | raw memory capacity (MC 10.19 vs 6.17) | Transformer's own inductive memory (positional/attention) — NOT replaceable by metadata | Not transferable (by design) |

## 3. Deduction per mechanism (grounded in Paper C's numbers)

### D1. M3 transfers the most — as a drift detector and domain-statistics feature

- Paper C Prop. 2-3: on statistical tasks the bottleneck is timescale
  coverage; the slow trace re-centers on the new regime within ~tau_m +
  tau_eff and the readout's mapping stays stationary (S15: adaptation
  variance p90 76.5 -> 42.0). For an LLM, this maps directly to
  *streaming domain drift detection*: a slow trace of activations tells you
  "the domain switched" faster and more consistently than loss spikes.
- Paper C S16b V2: the metadata denoises state-level corruption. Relevant
  to LLM deployment under quantization/pruning noise.
- **Boundary from Paper C**: the slow trace is a *statistical* memory — it
  captures distributional identity, not content. It can gate WHEN to adapt
  and WHICH domain you are in; it cannot remember facts or events. Any
  PoC claiming "metadata improves knowledge retention" contradicts Paper C
  directly.

### D2. M5 does NOT transfer as-is — the principle may, the mechanism cannot

- Paper C S14/S11: the +32% sequential recovery is produced by the
  homeostat, and the homeostat is the *only* mechanism that delivers it.
  Its control signal is an FTLE estimate on the substrate dynamics — a
  quantity with no canonical Transformer analog.
- An LLM "M5" would require (a) an instability scalar (candidate: LoRA
  update norm, gradient curvature, activation drift — all heuristic) and
  (b) a single knob that trades plasticity vs stability (candidate: LoRA
  scale alpha). This is *new research*, not a validation of Paper C; the
  paper should say so explicitly rather than pretend the homeostat
  transfers.

### D3. M4 transfers only as a design principle

- Paper C's M4 evidence: gentle rewiring helps (+8-11%), aggressive churn
  destroys (-23%), and the mechanism is causally clean (O4). The LLM
  analog — routing/adapting a few LoRA modules guided by functional
  connectivity, pruning redundant adapters — inherits the "gentle wins"
  principle, but the mechanism (substrate edge rewiring) is structurally
  different from adapter selection. Treat as inspiration, not as carried
  evidence.

### D4. The disentanglement thesis itself is the transferable contribution

- Paper C's most robust claim is *which component does which job*:
  metadata = detect + statistical features; regulator = stability under
  disturbance; learning = content. The LLM PoC's scientific value is
  instantiating and *testing the same division of labor* on a Transformer —
  e.g., showing that a slow-trace drift gate improves streaming adaptation
  while a stability monitor prevents collapse, and that neither alone does
  the other's job. This is the §7 story.

### D5. PoC must carry Paper C's falsification discipline

Paper C's credibility comes from its negative-results structure. The LLM
extension must include the same:

1. **Known-switch protocol** (s15 lesson): the domain-switch times must be
   known and T_adapt reported switch-relative, not via window-position
   ratios.
2. **tau_m sweep** (s16 lesson): the drift-gate timescale must be swept
   (e.g., 200/500/1000/2000 steps); if the benefit flips sign at some
   tau_m, report the sensitive interval.
3. **Protocol stress test** (s16b lesson): the metric must be re-checked
   under plausible protocol variants (e.g., where the slow trace is
   computed relative to the noise injection).
4. **Seed discipline**: 10 seeds, paired differences, sign consistency.

## 4. Critical assessment of the proposed REDEM-LoRA plan

| Proposed element | Verdict against Paper C evidence |
|---|---|
| "Bare LoRA vs LoRA+M3 vs LoRA+M4, three arms" | Keep, but add a *stability-monitor* arm only if an LLM M5 is properly defined; otherwise drop M5 from the headline and place it in limitations |
| "M3 monitors drift -> decides whether to adapt" | Supported (S10/S15): this is the strongest transferable claim |
| "M4 decides which modules adapt" | Plausible principle (O4-clean, gentle>aggressive); needs new evidence, mark as such |
| "M5 maintains overall stability" | Not supported by Paper C's data for LLMs; must be redefined or deferred |
| "GPT-2 124M / LLaMA-3.2-1B on CPU" | Infeasible on this machine; use the repo's tiny transformer (S9 baseline) or GPT-2 tiny with a hand-rolled LoRA adapter |
| "The result doubles Paper C's value" | Plausible only if the §7 claim is scoped: "principle instantiation on a Transformer", not "new LoRA SOTA" |

## 5. Minimal PoC design (CPU-only, ~1-2 weeks)

- **Model**: the existing tiny transformer from `scripts/baseline_showdown.py`
  (torch CPU), extended with a hand-written LoRA adapter (low-rank delta on
  attention projections, no HF dependency). Tokenizer: char-level or
  byte-level for speed.
- **Task**: streaming domain drift — alternate two corpora (e.g., two
  different text styles/topics) every L steps, *known switch instants*
  (regime_switch analog). Perplexity is the quality metric; adaptation is
  the T_adapt-style metric.
- **Arms** (scoped to D1/D4):
  - bare online LoRA (update every step, small LR)
  - LoRA + slow-trace drift gate (EMA of activations; update adapter only
    after a detected drift, gate tau_m swept)
  - LoRA + slow-trace domain routing (two adapters, EMA routes the active
    one)
  - (optional, if an LLM M5 is defined) + stability monitor
- **Metrics**: stream perplexity; post-switch adaptation (known switches);
  forgetting of the previous domain (the LLM-specific metric Paper C does
  not have); per-arm tau_m sweep with 10 seeds and paired analysis.
- **Falsification pre-commitment** (mirrors S14): if the drift gate does
  not improve stream perplexity or adaptation over bare online LoRA with
  0/10-seed-sign-consistency across the tau_m sweep, the extension claim
  is reported as falsified — and that itself is a Paper C-consistent
  result (mechanisms do not transfer beyond their job).

## 6. Conclusion of the deduction

**Yes — Paper C can be applied to large models, but with boundaries that
Paper C's own evidence dictates:**

1. **What transfers (strong):** the slow-trace statistical memory as a
   drift detector and domain-statistics feature (S10/S15/S16b), and the
   *systems-architecture principle* of separating detect / regulate /
   learn (the disentanglement thesis).
2. **What transfers only as principle (weak):** M4's "gentle wins" for
   adapter routing; M5's stability role requires redefining an LLM
   "instability scalar + plasticity knob" — new research, not validation.
3. **What does not transfer (by design):** substrate physics / raw episodic
   memory; and the metadata's non-transferability of memory-capacity
   robustness carries over as a warning for any "metadata improves LLM
   knowledge retention" claim.
4. **How to do it:** a §7 PoC on the existing tiny transformer with a
   hand-rolled LoRA adapter, a known-switch streaming-drift task, the
   s15/s16/s16b falsification discipline, and honest reporting of the
   failing dimensions.

The extension upgrades Paper C from "mechanism paper on a physical
reservoir" to "system-architecture principles for online learning,
instantiated on two very different substrates" — provided the negative
dimensions are reported with the same honesty as the positive ones.
