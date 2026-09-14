# Self-Evolution Under ±1 Reward: A Verified Derivation Chain (S39–S58e)

> **Status**: research-narrative / provenance companion. The PAPER itself is
> [`PAPER_E.tex`](PAPER_E.tex) (claim-driven, R1–R6; this narrative's
> S-numbered walk-through is condensed there into a reproducibility index).
> Chinese traceability, per-script records, negative-result detail and the
> theorem list live in
> [`../review_workspace/derivation_record_s39-s44.md`](../review_workspace/derivation_record_s39-s44.md);
> the route map (what was tried, what remains) is in
> [`../review_workspace/derivation_route_map.md`](../review_workspace/derivation_route_map.md).
> Every number below is committed as CSV/JSON in `../data/` (s39…s62); the
> chain scripts import the frozen [`deps/`](deps/) modules.

## 0. Abstract (draft)

A recurrent spiking substrate with reward-modulated Hebbian plasticity
(RMHL) learns slowly and fails structurally under ±1 reward — it cannot
recover from mapping inversions (S3). We show that the missing signal is
present but hidden: reward-rate aggregation carries the direction
information, and a small meta-layer that detects the reward-rate sign and
inverts the readout weight recovers the task (S39–S41). We then identify
the boundary of the whole compensation family: credit assignment under ±1
reward is a *rule-selection* problem, not an information-theoretic limit
(S46); corrections must be *structural* (per-dimension gain), because a
scalar gain in the near-chance regime is a noise amplifier (S47). This
positions the derivation as: **an error-driven base (RLS readout) plus a
structural self-correction layer**, with the honest caveat that plain
error-driven learning is simpler and our contribution is the *autonomous*
part — the system detects its own representation insufficiency and repairs
it (capacity sense + basis expansion), preserves hypotheses across regime
revisits (memory), and does so continuously and hierarchically
(sense–expand–sense-again).

## 1. Main-line narrative

### 1.1 The Hebbian learner and its meta compensation (S39–S45)

- **S39**: direction information lives in the sign of the aggregated reward
  rate; an active complementary-hypothesis probe recovers the task
  (`probe_directed` 0.898/0.939) while passive variants fail.
- **S40**: the mechanism is *reward-rate sign detection + weight flip*
  (`passive_flip` matches the probe and detects faster, 39 vs 99 blocks);
  the complementary probe is a cost, not a benefit.
- **S41**: a level-3 controller learns its own flip policy online from
  experience (learned flip value v, no hand margin; envA recovery
  0.899/0.940 matching the hand-set policy; zero self-harm on a stable
  stream). Theorem D1: each meta-level learns from reward-rate *change*,
  which is convention-free.
- **S42/S43/S44/S45**: signal trust (D4) — a polluted reward signal drives
  the system to optimize the wrong signal and it cannot self-detect the
  pollution; the 1-D regime family is completely mapped (inversion-class
  regimes are flip-solvable; translation/compression regimes collapse for
  the whole RMHL family even though the task is learnable — S44 shows
  `rls_fresh` learns them); a weight norm-cap fixes the anti-learning
  (S45).

### 1.2 Credit assignment is rule selection; corrections must be structural (S46–S47)

- **S46**: with ±1 rewards and block-vote predictions, `r=+1 ⟹ y=vote` and
  `r=−1 ⟹ y=1−vote` — the ±1 stream is informationally equivalent to
  "label revealed after prediction". The RMHL failure is therefore a
  *rule-selection* failure (the Hebbian sign gate discards the label
  information), not an information-theoretic limit.
- **S47**: the correction operator must carry per-dimension gain; a scalar
  gain in the E[r]≈0 dead zone is a positive-feedback trap (noise
  amplifier).

### 1.3 The decodability base (S48–S49)

- **S48**: a smoothly rotating 2-D boundary is linearly decodable from the
  coupled substrate's features (random probe 0.91) and trackable online
  (rls ≈ ridge ceiling) — L2 rotation does not trigger representation
  insufficiency.
- **S49**: rotation *speed* is absorbed by the temporal dynamics (rls
  steady ≥ ridge ceiling at every speed) — speed is not an L3 trigger.

### 1.4 L3: capacity, stress and the corrected trigger picture (S50–S58a)

- **S50/S51 → S53 correction (methodology)**: the original "circle is
  undecodable, basis expansion is necessary" conclusion was a
  *probe-conditioning artifact* — ridge on the full 257-dim block-mean
  features with 500 samples is underdetermined (held-out 0.42). A
  well-conditioned probe (ridge on top-50 PCA components) shows the coupled
  substrate's *raw* features already decode the circle (0.80–0.88):
  representation insufficiency is real **only** on the uncoupled parallel
  substrate (0.48 linear, 0.57 quadratic). The capacity sense built from
  the corrected probe correctly idles on the sufficient substrate (9/10
  seeds no false trigger) and correctly triggers on the insufficient one
  (10/10, weak repair 0.52→0.60).
- **S55 → S57 correction**: input dimension is a *secondary* stress — the
  pure-quadratic ball degrades 2-D→3-D (probe 0.800→0.630) but at 4-D label
  balance forces radius > 0.5, the balanced ball clips the unit cube and
  reads *higher* (0.700), so dimension alone never collapses the coupled
  substrate. The annulus (two quadratic boundaries, balanced by
  construction) declines monotonically toward chance with dimension
  (probe 2-D ring 0.650 → 3-D shell 0.600 → 4-D shell 0.540).
- **S58a (honest negative)**: the complexity axis *saturates* above chance
  — even 6 quadratic surfaces (triple shell) cannot cross the chance line
  (4-D shell→double→triple rls 0.544→0.547→0.559, probe within noise). The
  coupled substrate retains an irreducible ~+4–6 pp edge: L3 on the
  coupled substrate is *asymptotic degradation, never collapse*; full
  collapse exists only on the uncoupled substrate (rich-kernel machine).

### 1.5 The self-evolution loop (S52–S56)

- **S52**: under A-B-A-B regime revisits, *frozen hypothesis memory +
  reward-rate selection* beats continuous re-adaptation (mean 0.888 vs
  0.838; circle revisit +14.9 pp). Hypotheses are identified *functionally*
  (prediction agreement), not geometrically.
- **S54**: memory boundaries — forgetting is eviction/capacity-driven, not
  time-driven; gain = capacity fit × revisit frequency.
- **S53**: the well-conditioned capacity sense + autonomous basis expansion
  form the self-evolution mechanism (idle at zero cost on sufficient
  substrate; triggers and weakly repairs on insufficient).
- **S56**: memory × basis expansion compose on orthogonal axes (joint ≥
  each single axis ≈ oracle); in dynamic environments the capacity sense
  must be *relative* (candidate basis vs current basis), because the
  absolute-threshold sense false-triggers at regime boundaries
  (boundary-mixing artifact).

### 1.6 The applicability boundary of the capacity sense (S58b)

The relative sense has a reliability cliff governed by the
*window/segment ratio × insufficient-exposure fraction*:

- reliability=(1−false)(correct): ABAB 0.90 (SEG=500) → 0.27 (250) → 0.00
  (125/60); the cliff sits at WIN_BLOCKS(240)/SEG ≥ 2 (the probe window
  permanently straddles ≥2 regimes).
- Window mixing acts two ways: on the sufficient substrate it creates
  delta *noise* spikes (one false trigger at SEG=250, held off elsewhere
  by the REL_HOLD gate); on the insufficient substrate it *dilutes* the
  genuine signal (all misses are margin/hold-gated — the candidate floor
  is never the blocker).
- The seed-4 miss of S56 is a REL_HOLD flicker (longest consecutive
  both-gate run 2<3; cq peak 0.733 ≫ floor; the segment mean of 0.041
  hides it).
- A denser 3-regime sequence does **not** degrade reliability — it
  improves it (extra insufficient exposures rescue flickering seeds).
- Parameter calibration cannot fix the cliff (S58f, honest negative): the
  probe-noise floor (random_graph best-delta ~0.13–0.14) overlaps the
  diluted genuine signal (parallel ~0.17–0.21), so raising REL_MARGIN from
  0.10 to 0.15/0.20 cuts deep into genuine triggers (SEG=500 reliability
  0.90→0.60→0.10) without reliably removing the false triggers at short
  segments — REL_MARGIN=0.10 is already the Pareto-optimal operating point,
  and the reliability cliff is a structural property of window mixing (the
  fix is structural: longer windows / more exposures).

### 1.7 Multi-level self-evolution and memory as divergence protection (S58c)

- Replacing the one-shot expansion gate with a cooldown yields a working
  sense–expand–sense-again chain: parallel L1 10/10, L2 5/10 (all on
  circle-revisit windows) while the ring is *correctly refused* 10/10 —
  the sense rejects an upgrade the substrate cannot support (cubic cannot
  decode the ring on parallel).
- Zero-padded memory survives two upgrades (the linear hypothesis is still
  exactly valid after double padding; linear revisit 0.95 vs 0.90 oracle).
- **New mechanism**: memory = divergence protection — a memoryless RLS
  readout *blows up* (weights to norm ~2800) when forced to chase an
  unfittable boundary (the ring) on the sufficient substrate, collapsing to
  chance (0.504); frozen-snapshot selection masks the identical worker
  divergence (0.801).

### 1.8 The substrate side of L3: rich-kernel saturation (S58d)

- The s58a irreducible margin is a monotone function of kernel *width* but
  an inverted-U of *coupling*: weakening the kernel (κ 25→15→10, N 256→64)
  compresses the margin toward chance (4-D shell 0.100→0.022→0.021;
  0.034); widening it (N→512) raises it (0.034→0.100→0.107); but κ=40
  over-coupling saturates the kernel and the margin collapses back to
  chance (4-D shell 0.008). Substrate capacity peaks at the optimal
  coupling κ*≈25–30 (S1 phase diagram) and falls on both sides.

### 1.9 The meta-layer timescale (S58e, theorem D2 verified)

- D2 ("a system can only correct changes slower than its own observation
  scale", marked theory since S45) is verified with a *sharp threshold* at
  τ_env/τ₂ ≈ 1–2: below it the meta layer cannot recover (post-swap
  accuracy ~0.5; flips fire on regime-mixed EMA estimates = noise
  amplification; no better than no meta); above it recovery is complete
  (post 0.90–0.94, success rate 1.0). The quantitative form of "τ_env ≫
  τ₂" is τ_env ≳ 2τ₂ (detection delay ~0.8τ₂ + flip-and-relearn).
  passive_fixed tracks meta_self exactly — the binding constraint is the
  detection timescale, not the value-learning component.

### 1.10 The memory layer tunes itself (G1/S59); structural repair is bounded (G2/S60)

- **G1 (positive)**: the last hand-set threshold in the memory layer (the
  snapshot gate SNAP_EMA_MIN=0.55) is REMOVED by generalising the s41
  value-learning: the system learns a self-tuned gate v_snap from the
  realized usefulness of memory (memory strictly beating the worker on
  selected blocks). In the dynamic A-B-A-B environment v_snap converges low
  (0.252 — memory is useful, keep snapshotting; performance matches the
  fixed-threshold memory), in the static single-regime environment it rises
  monotonically to 0.922 (snapshots self-suppress, no harm) — the learned
  gate reads the environment exactly like s41's learned flip value (+0.504
  vs 0.150). "When to keep experience" is now learned, not designed.
- **G2 (honest negative)**: the capacity sense's window is made adaptive
  (short window the moment hard content is detected), the candidate
  structural fix for the s58b reliability cliff. The short window genuinely
  strengthens the single-refit delta (clean windows), but its probe variance
  blocks the REL_HOLD=3 requirement: at SEG=250 reliability is unchanged
  (0.27) and at SEG=500 it gets WORSE (0.90→0.50). Probe conditioning
  (sample count) is the binding constraint. Together with s58f (margin
  cannot fix the cliff), this closes the parameter side: the cliff is
  unfixable by the sense's own parameters, and the only demonstrated repair
  is exposure count (the s58b ABCABC result — an experience-side property).

### 1.11 The generalization radius and the pipeline floor (G3/S61, G4/S62)

- **G3 (partial transfer)**: experience reuse across boundary families is
  PARTIAL and STRUCTURE-MATCHED. The frozen disk hypothesis partially decodes
  the annulus (0.576 on parallel, 0.619 on random_graph — both radial); on
  the uncoupled substrate, where the worker cannot learn the ring (0.504),
  memory is selected for 64% of the ring segment and gains +8.9 pp (0.593),
  while on the coupled substrate the gain is small (+1.3 pp) because the
  substrate already carries the ring. Linear hypotheses show no transfer to
  the ring (selection correctly discards them), and the foreign segment does
  NOT poison prior hypotheses (the seg5 linear revisit still gains). The
  reuse radius is "boundaries sharing structure", not arbitrary similar
  tasks.
- **G4 (honest negative, refines D2)**: the meta can adapt its own EMA
  timescale τ₂ — a working-end benefit (at ratio 2 the overall mean rises
  +13.4 pp by detecting swaps faster, post unchanged) — but it CANNOT soften
  the D2 threshold below it: the flip PIPELINE latency (check interval
  FLIP_CHECK=20 + value window W_EST=40 blocks) exceeds the fast regimes, and
  a too-fast EMA only adds decision noise. The quantitative form of
  "τ_env ≫ τ₂" is really "τ_env ≳ pipeline latency": τ₂ is adaptable, the
  decision pipeline is the true floor.

## 2. Methodology corrections record (honest reporting)

| Claim (earlier) | Correction (later) | Evidence |
|---|---|---|
| Circle undecodable; basis expansion necessary (S50/S51) | Probe-conditioning artifact: full-dim ridge underdetermined; well-conditioned probe decodes with raw features on the coupled substrate (0.80–0.88); insufficiency real only on parallel | S53 |
| Absolute capacity threshold (S53) valid dynamically | Absolute sense false-triggers at regime boundaries; the dynamic form is the RELATIVE dual-basis sense (v1→v2) | S56 |
| Dimension is the L3 trigger (S55) | Dimension is secondary: the 4-D balanced ball clips the cube and reads higher; boundary complexity (annulus) is the monotone stress axis | S57 |
| Complexity escalation should eventually trigger L3 (S58a prediction) | Negative: complexity saturates above chance; L3 is asymptotic degradation on the coupled substrate | S58a |
| Sense reliability uniform across segment lengths (implicit in S56) | Reliability cliff at window/segment ≥ 2; misses are margin/hold-gated; denser exposure rescues | S58b |
| One-shot expansion is enough (S53/S56) | Multi-level chain works (cooldown); memory = divergence protection; ring correctly refused | S58c |
| Higher κ → higher margin (S58d prediction) | Inverted-U: κ=40 over-coupling collapses the margin; capacity peaks at κ*≈25–30 | S58d |
| D2: "τ_env ≫ τ₂" (theory since S45) | Verified with sharp threshold ≈1–2; quantitative form τ_env ≳ 2τ₂ | S58e |
| Capacity-sense margin should be tuned above the noise floor (S58f prediction) | Negative: noise floor and genuine signal overlap; 0.10 is Pareto-optimal; the cliff is structural | S58f |

## 3. Verified theorem list (English)

1. **Mirror-impossibility**: vote(−w)=1−vote(w) ⟹ r(−w)=−r(w) blockwise ⟹
   E[r](−w)=−E[r](w); no suppression is learnable inside the two-arm mirror
   set (S40/S42, proof + experiment).
2. **D1 sign-ambiguity dissolution**: every meta-level learns from the
   reward-rate *change* Δr; the "improvement" convention is irrelevant, so
   meta-learning is well-defined (S41).
3. **D2 timescale separation (verified)**: a system can only correct
   changes slower than ~2× its observation timescale; sharp threshold at
   τ_env/τ₂ ≈ 1–2, quantitative form τ_env ≳ 2τ₂ (S58e; previously
   "theory" since S45).
4. **D3 regression boundary**: finite hierarchy + fixed top-level learning
   rule; infinite bootstrapping violates the self-reference boundary
   (theory).
5. **D4 signal trust**: a self-evolving system optimizes the signal it
   measures; a polluted signal drives self-deception and cannot be
   self-detected (S42).
6. **1-D completeness**: the mirror-flip is complete for inversion-class
   regimes; translation/compression regimes collapse (E[r]≈0, no action)
   (S43 theory + S44).
7. **±1=label identity**: r=+1 ⟹ y=vote, r=−1 ⟹ y=1−vote; the ±1 stream is
   equivalent to label-after-prediction (proof + S46).
8. **Rule-selection criterion**: credit-assignment failure is rule
   selection, not an information-theoretic limit (S46).
9. **Structural-correction criterion**: corrections must carry per-dimension
   gain; scalar gain in the E[r]≈0 dead zone is a noise amplifier (S47).
10. **L2 decodability**: smooth rotation is linearly decodable from the
    coupled substrate and trackable online; speed is absorbed (S48/S49).
11. **L3 probe-conditioning criterion**: capacity probes on high-dim
    block-mean features must be well-conditioned (PCA/regularized);
    underdetermined ridge misreads true capacity (S53).
12. **L3 = boundary complexity, not dimension**: pure-quadratic ball
    degrades 2-D→3-D but 4-D balance clipping breaks the dimension axis;
    the annulus declines monotonically with dimension (S55→S57).
13. **L3 asymptotic degradation, never collapse**: complexity saturates
    above chance; the coupled substrate keeps an irreducible ~+4–6 pp edge;
    full collapse exists only on the uncoupled substrate (S58a).
14. **Rich-kernel saturation (corrected)**: the margin is a monotone
    function of kernel width but an inverted-U of coupling; capacity peaks
    at κ*≈25–30 (S58d).
15. **L4 dynamic memory**: frozen hypothesis memory + reward-rate selection
    beats continuous re-adaptation under revisits; forgetting is
    eviction-driven (S52/S54).
16. **Autonomous capacity sense**: the well-conditioned relative sense
    correctly idles on sufficient substrates and triggers on insufficient
    ones (S53/S56).
17. **Capacity-sense applicability boundary**: reliability = window/segment
    ratio × insufficient-exposure fraction; misses are margin/hold-gated;
    denser exposure rescues (S58b).
18. **Multi-level self-evolution chain**: sense–expand–sense-again works;
    the L2 upgrade is correctly refused when the substrate cannot support
    it; zero-padding preserves memory across upgrades (S58c).
19. **Memory = divergence protection**: memoryless RLS diverges on
    unfittable boundaries; frozen-snapshot selection masks it (S58c).
20. **Margin calibration cannot fix the cliff**: the noise floor and the
    genuine signal overlap; REL_MARGIN=0.10 is Pareto-optimal; the s58b
    reliability cliff is structural (window mixing), not a calibration
    artifact (S58f).
21. **Self-tuned snapshot gate (G1)**: the memory layer learns WHEN to keep
    experience (value-learning generalised from S41) — the learned gate
    stays low in dynamic environments (memory useful) and rises to suppress
    snapshotting in static ones (memory redundant), matching the hand-set
    threshold's performance with no harm; the last hand-set memory threshold
    is removed (S59).
22. **Structural repair is bounded (G2)**: the capacity-sense window cannot
    fix the s58b cliff either — short clean windows strengthen the
    single-refit delta but their probe variance blocks the hold requirement
    (SEG=250 unchanged, SEG=500 worse); probe conditioning (sample count) is
    the binding constraint; the cliff is unfixable by sense parameters
    (margin S58f, window S60), only by exposure count (S58b, experience-side)
    (S60).
23. **Cross-family transfer is partial and structure-matched (G3)**: frozen
    hypotheses transfer to boundaries sharing their structure (radial disk →
    annulus, +8.9 pp on the uncoupled substrate), not to arbitrary similar
    tasks; structurally different families show no transfer (selection
    discards them), and foreign content does not poison prior hypotheses
    (S61).
24. **The D2 threshold is pipeline-bound (G4)**: adapting τ₂ gives a
    working-end speed gain (ratio-2 mean +13.4 pp) but cannot soften the
    threshold — the flip pipeline latency (check interval + value window) is
    the true floor; "τ_env ≫ τ₂" is really "τ_env ≳ pipeline latency" (S62).

## 4. Key-number table (committed data)

| Result | Numbers | Files |
|---|---|---|
| Direction in reward-rate sign | probe_directed 0.898/0.939 | `s39_prediction_reward_v2.*` |
| Flip mechanism, no probe | passive_flip 0.898/0.939, detect 39 vs 99 blocks | `s40_meta_flip_v1.*` |
| Meta learns its own policy | envA 0.899/0.940, v_final +0.504 | `s41_meta_reward_v1.*` |
| Pollution self-deception (D4) | corrupted arms 0.106/0.064 | `s42_corrupt_signal_v1.*` |
| ±1=label; rule selection | delta-derived R1/R4 0.989/0.999, R2/R3 0.536–0.752 | `s46_rule_adjudication_v1.*` |
| ~~Structural correction~~ **retracted** | s47 script-level P2 **refuted** (boosted arm at/below base on R2/R3); numbers **not adopted** in the manuscript — not a negative result | `s47_self_correction_gain_v1.*` |
| L2 rotation decodable | random probe 0.91; rls tracks | `s48_rotation2d_v1.*` |
| Speed not L3 | rls steady ≥ ridge at every speed | `s49_capacity_probe_v1.*` |
| L3 probe correction | random raw-feature circle 0.80–0.88; parallel 0.48 | `s53_auto_basis_v1.*` |
| Memory wins revisits | pop_mem 0.888 vs rls 0.838; circle revisit +14.9 pp | `s52_dynamic_memory_v1.*` |
| Composition (S56) | parallel pop_auto 0.769 > pop_mem 0.758 > rls_auto 0.737 ≈ oracle 0.762 | `s56_joint_self_evolution_v1.*` |
| Complexity saturation (S58a) | 4-D shell→double→triple rls 0.544→0.547→0.559 | `s58a_complexity_trigger_v1.*` |
| Sense reliability cliff (S58b) | ABAB 0.90→0.27→0.00; seed-4 = hold flicker 2<3, cq 0.733 | `s58b_relative_sense_stress_v1.*` |
| Multi-level + divergence (S58c) | L1 10/10, L2 5/10 (circle revisits), ring refused 10/10; 0.801 vs 0.504 | `s58c_multi_level_expansion_v1.*` |
| Rich-kernel margin (S58d) | 4-D shell 0.100→0.022→0.021 (κ↓), 0.034→0.100→0.107 (N↑), κ=40 → 0.008 | `s58d_kernel_capacity_sweep_v1.*` |
| D2 threshold (S58e) | post 0.486→0.561→0.938 at ratio 0.25/1/2; succ 1.0 from ratio 2 | `s58e_d2_timescale_v1.*` |
| Margin cannot fix the cliff (S58f) | reliability 0.900→0.600→0.100 (SEG=500) as margin 0.10→0.15→0.20; d_best noise 0.129 vs signal 0.187 | `s58f_margin_sensitivity_v1.*` |
| Self-tuned snapshot gate (G1) | v_snap dyn 0.252 vs static 0.922; mean 0.876≈0.885 (dyn), 0.955=0.946 (static) | `s59_self_tuned_gate_v1.*` |
| Window adaptation cannot fix cliff (G2) | SEG=250 reliability 0.270=0.270; SEG=500 0.900→0.500; d_best ↑0.198 vs 0.173 | `s60_adaptive_window_v1.*` |
| Cross-family transfer (G3) | ring +8.9pp (parallel), mem_acc 0.576–0.619; revisit +1.1–3.1pp | `s61_cross_family_v1.*` |
| Adaptive τ₂ (G4) | ratio-2 mean +13.4pp; ratio ≤1 post ≈ 0.49–0.54 | `s62_adaptive_meta_v1.*` |

## 5. Positioning statement (draft for the paper)

The contribution is the *autonomous* correction stack on top of a standard
error-driven base: **(i)** the capacity sense (well-conditioned, relative,
window-bounded) that lets the system detect its own representation
insufficiency and decide *when* and *to what* to expand; **(ii)** frozen
hypothesis memory that preserves past solutions across revisits and
upgrades, and doubles as divergence protection; **(iii)** the multi-level
chain that refuses upgrades the substrate cannot support; **(iv)** the
verified timescale limit (D2) that bounds when the meta layer can act at
all. We honestly note that a plain error-driven learner (RLS) is simpler
and achieves the same single-task performance; the value is in the
autonomous, hierarchical, self-monitoring structure — and in the negative
results that bound it (complexity saturation, the reliability cliff, the
over-coupling collapse, the timescale threshold).
