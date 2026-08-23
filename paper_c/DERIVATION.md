# Paper C - Derivation Notes: The Metadata Equalizer

**Working title**: The Metadata Equalizer: Substrate-Agnostic Slow-Trace
Transfer and Disturbance Robustness in Online Reservoir Computing

**Status**: derivation stage, s14 DONE. Anchors from committed full-seed
data: `data/s10_esn_metadata_v1.*`, `data/s11_disturbance_chain_v1.*`,
and the new `data/s14_esn_disturbance_chain_v1.*` (Section 8). The s14
result falsifies the original "sequential robustness transfers" thesis and
reframes the paper as a three-mechanism disentanglement (see Sections 5, 8,
9). Experiments s15-s17 are still pending.

---

## 0. Scope and non-overlap with Papers A and B

| Dimension | Paper A (substrate theory) | Paper B (integrated system) | Paper C (this work) |
|---|---|---|---|
| Protagonist | substrate physics | REDEM (RLS + M3 + M4 + M5) | a single mechanism: the M3 slow trace |
| Substrate | Si3N4 relaxation array | Si3N4; ESN as honest baseline | any reservoir: ESN primary, Si3N4 secondary |
| Core claim | material forgetting kernel M(t), phase diagram, homeostat theory | training-inference-unified online learning, robust to disturbance | slow trace = synthesized forgetting kernel, substrate-agnostic; the metadata equalizer |
| New experiments needed | none | none | s14-s17 (Section 8) |

Hard non-overlap rules:

1. Paper C never re-derives substrate dynamics, the homeostat, or plasticity.
   M5/M4 appear at most as controlled ablations for honesty.
2. Paper C does not claim REDEM superiority. Honest data position: in S10 the
   best overall accuracy belongs to `esn_dual`, not `redem`. The claim is that
   the *mechanism* transfers, and that the equalization is about timescale
   coverage, not substrate physics.
3. Paper C cites Paper A Section 5 (material forgetting kernel) and Paper B
   Section 4.5 (metadata-transfer experiment) as its background, closing the
   citation loop A -> B -> C.

## 1. Setup and notation

Let a reservoir (ESN or Si3N4 relaxation array) produce a fast state vector
`x(t) in R^N` at pulse index `t`. Task: `regime_switch` (S5/S8 protocol,
`streaming_tasks.py`), where the inter-pulse interval `dt` is drawn from a
regime-dependent distribution and the regime switches every

    L = RS_REGIME_LEN = 1500 pulses.

The readout must output the current regime index from the recent input
history, i.e. it must represent the empirical statistics of `dt` over a
window of order L.

Slow trace (M3 metadata, `integrated_benchmark.slow_ema`):

    m(t) = (1 - 1/tau_m) * m(t-1) + (1/tau_m) * x(t),        (1)

an exponential moving average with algorithmic timescale `tau_m` (tau_m = 500
pulses in S10). Augmented feature:

    phi(t) = [ x_norm(t), m_norm(t), 1 ],                    (2)

where `_norm` is per-unit standardization computed on the first 30% of the
stream. Readout: OnlineRLS with forgetting `lambda_f = 0.999`, i.e. weight
integration time `tau_f = 1/(1 - lambda_f) = 1000` pulses, predict-before-
update (Paper B Section 2).

## 2. Proposition 1 - The slow trace is a synthesized forgetting kernel

**Claim.** The channel input -> slow trace implements a forgetting kernel
`h_m(Delta_t)` with (i) a zero at `Delta_t = 0` (recent transients are
suppressed), (ii) a peak at a finite lag, and (iii) an exponential tail whose
1/e horizon is exactly `tau_m` - an externally controllable, substrate-
independent parameter.

**Derivation.** The EMA (1) has impulse response `(1/tau_m) * e^{-k/tau_m}`
on the state channel. If the fast state is itself a linear filter of the
input history with kernel `h_x` (for the substrate: the material forgetting
kernel `M(Delta_t) = integral p(tau) e^{-Delta_t/tau} d tau` of Paper A; for
an ESN: approximately `e^{-Delta_t/tau_eff}`), the input->slow-trace kernel
is the convolution

    h_m(Delta_t) = (1/tau_m) * integral_0^inf e^{-s/tau_m} h_x(Delta_t - s) ds.

For a single-exponential fast kernel `h_x(u) = e^{-u/tau_x}`:

    h_m(Delta_t) = ( e^{-Delta_t/tau_m} - e^{-Delta_t/tau_x} ) / (tau_m - tau_x).  (3)

Properties of (3):

- `h_m(0) = 0`: the slow channel ignores the instantaneous transient.
- Peak lag `Delta_t* = (tau_m * tau_x / (tau_m - tau_x)) * ln(tau_m / tau_x)`.
- Tail `e^{-Delta_t/tau_m}`: the 1/e horizon is `tau_m`, set purely by the
  algorithm, independent of the substrate.

**Contrast with Paper A.** The material kernel M(t) (log-normal, median
tau0 ~ 174 us, CV 0.20, 1/e horizon ~16 pulses in pulse units) is fixed by
the physics; `h_m` is synthesized by the algorithm and its horizon is a free
knob. The augmented system's effective kernel is the convolution
`M_eff = M (x) h_m`, whose 1/e horizon extends to roughly `tau_m + tau0`.
This is the precise sense in which the metadata is "the macroscopic
equivalent of the forgetting kernel, not a physical property" (S10 design
decision): Paper A gives the material's kernel, Paper C gives the
algorithmic kernel that any substrate can wear.

**Testable prediction.** The held-out memory curve MC(k) (Paper A protocol,
Jaeger MC(k) ridge readout) of the *augmented* system should have its 1/e
horizon moved from ~16 pulses (fast only) to ~tau_m + 16 pulses. Not yet
measured - listed as experiment s16.

## 3. Proposition 2 - Timescale coverage, not physics, sets task performance; metadata equalizes substrates

**Claim.** On the long-horizon statistical task the bottleneck is missing
slow modes, not substrate physics. Both fast channels (ESN, Si3N4) decorrelate
at `tau_eff << L = 1500` pulses, so neither can represent the regime
statistic directly; the slow trace supplies the missing timescale to either
substrate, and performance converges to a common band set by the task noise
and `tau_m`.

**Argument.** The optimal readout on regime_switch is a function of the
empirical distribution of dt over the recent ~L pulses. The fast state only
reflects the last `tau_eff` pulses of input; the readout therefore has to
*integrate* many fast snapshots through its weight memory (`tau_f = 1000`
pulses) after every switch. The slow trace (3) is a direct estimate of the
recent statistics: it re-centers on the current regime within ~`tau_m + tau_eff`
pulses. Because the slow-trace construction (1)-(2) never touches the
substrate equations, connectivity, or timescales, portability is by
construction: the same `tau_m`, the same standardization, the same RLS
readout run on whichever fast signal the substrate emits.

**Data (S10, 10 seeds per arm, committed).**

| arm | overall acc | steady acc | adapt time (window units) |
|---|---|---|---|
| esn_fast (ESN fast state only) | 0.9955 +- 0.0015 | 1.000 | 11.22 +- 10.67 |
| esn_dual (ESN + slow trace)   | 0.9979 +- 0.0001 | 1.000 | 0.24  +- 0.17  |
| redem (Si3N4 + dual metadata) | 0.9942 +- 0.0011 | 1.000 | 11.26 +- 5.91  |

Honest reading, to be locked into the paper text:

- The metadata raises ESN accuracy 0.9955 -> 0.9979 (+0.24 pp, ~5x the
  standard error) and collapses adaptation ~47x (11.22 -> 0.24). All three
  arms fall inside [0.994, 0.998] with steady accuracy 1.000 - the
  "equalization band".
- `esn_dual`, not REDEM, is the best overall accuracy. The equalizer claim is
  about the mechanism, not about beating the ESN; REDEM's differentiation
  (material-set memory design, local sparse structure, robustness mechanisms)
  is Paper B's story, and is untouched by Paper C.
- Scope of the equalizer: *statistical* tasks (accuracy and adaptation on
  regime_switch). It does NOT equalize raw memory capacity - the S14 MC
  probe shows the substrate at 10.19 vs ~6.17 for both ESN arms, and the
  slow trace adds no MC on the ESN (r0 paired diff -0.0035). "Timescale
  coverage" and "episodic memory capacity" are different quantities.

## 4. Proposition 3 - Why adaptation collapses: feature stationarity vs weight integration

**Claim.** Post-switch adaptation time is bounded by the slower of (i) the
time for the readout's *features* to become informative about the new regime
and (ii) the time for the RLS *weights* to re-estimate the mapping. Fast-only
features force (ii) to dominate (~tau_f ~ 1000 pulses of re-integration); the
slow trace makes (i) dominate instead, and (i) is ~`tau_m + tau_eff` ~ 500
pulses - but in practice much less, because a 500-pulse window of the new
regime's statistics is already highly informative (regime statistics are
almost surely separated).

**Sketch.** Let the readout error e(t) = y(t) - y_hat(t) be driven by a
mixture of weight error and feature mismatch. After a switch at t0, fast-only
features are uninformative about the new regime for t - t0 < tau_eff (the
state still reflects the old regime), and remain *statistically biased* until
the RLS has integrated ~tau_f new samples; accuracy recovers on the timescale
of the weight memory. With the slow trace, the feature vector itself
re-centers: m(t) tracks the new regime's state distribution within
~`tau_m + tau_eff`, so the readout input is near-stationary and the existing
weights (which were trained on the same feature mapping) transfer; recovery
is limited by the feature-side timescale.

**Caveat (honesty).** The S10 adaptation metric is a 200-pulse-windowed
rolling statistic measured on segment tails with unknown exact switch
instants; the ratio 11.22/0.24 is a headline number but the protocol should
be made exact (known switch instants, reported distribution) before the paper
is written - experiment s15.

## 5. Proposition 4 - Disturbance attenuation and sequential robustness

**Claim.** The EMA is a first-order low-pass filter; short fast-channel
disturbances are attenuated in the metadata channel, and persistent
disturbances cause the slow trace to *re-anchor* on the new statistics
rather than corrupting it. This explains the transferable readout-noise
attenuation observed in S14 (r3 NMSE -9.5%). It does NOT confer
memory-capacity recovery: S14 shows the slow trace does not protect (and
slightly degrades) MC under disturbance on the ESN; sequential recovery is
the homeostat's role (S11).

**Derivation.** The EMA transfer function has gain

    |H(omega)|^2 = 1 / (1 + (omega * tau_m)^2).

A fast-channel disturbance of amplitude sigma and duration T_d << tau_m
moves the slow trace by at most sigma * (1 - e^{-T_d/tau_m}) ~ sigma * T_d /
tau_m, i.e. it is strongly suppressed for short transients (edge pruning,
pulse noise). A persistent disturbance (e.g. tau drift of the substrate) is
*tracked*: the slow trace re-anchors on the post-disturbance statistics
within ~tau_m, which is exactly what an online tracker should do - the slow
channel degrades gracefully instead of failing.

**Data anchor (S11, 10 seeds, REDEM regulated arm; the homeostat system).**
After three sequential disturbances (tau x1.5 drift at t=7000, 40% edge
pruning at t=14000, readout noise sigma=0.1 at t=21000), the regulated arm
retains held-out memory MC = 8.47 +- 0.39 vs 6.41 +- 0.41 for the
fixed-coupling arm (+32%), with the homeostat drifting kappa
25.3 -> 26.2 -> 27.4 -> 28.5 to compensate. Note that the S11 features are
fast-state only (no slow trace): the +32% recovery is produced by the
homeostat alone.

**Transfer test (S14, 10 seeds, DONE - the decisive falsification).** The
same disturbance chain was run with substrate-agnostic disturbance
definitions on ESN arms with and without the slow trace (esn_fast vs
esn_dual, see Section 8). Result: the metadata does NOT transfer sequential
robustness in memory-capacity terms. Paired per-seed differences
(esn_dual - esn_fast) on the MC probe are -0.78 +- 0.14 (r1), -0.76 +- 0.20
(r2), -0.69 +- 0.14 (r3): consistently negative in 10/10 seeds (~5x the
paired std) - the slow trace slightly *degrades* MC under disturbance on
the ESN. The r0 paired difference is -0.0035 +- 0.0020 (no effect at
nominal physics, as Prop. 1 predicts for i.i.d. input: smoothing destroys
per-lag identifiability). Meanwhile redem_reg reproduces the S11 anchor
exactly (r0 10.19, r1 7.17, r2 7.51, r3 8.47) and recovers +18% r1->r3,
while neither ESN arm recovers (dual 1.04->1.05, fast 1.82->1.74).

**What the metadata DOES transfer (S14, task NMSE).** On the online
Mackey-Glass prediction task, the dual arm improves NMSE at the readout-
noise round (r3: 0.0251 vs 0.0277, -9.5%, 10/10 seeds) and slightly at r1
(0.0020 vs 0.0022, 10/10), consistent with the EMA low-pass attenuation of
Prop. 4; it is slightly worse at r2 (0.0011 vs 0.0009, 10/10).

**Attribution corrected.** The S11 +32% sequential recovery is homeostat-
driven (kappa adaptation), not metadata-driven. The metadata's transferable
role is (i) statistical/timescale coverage on regime tasks (S10) and
(ii) readout-noise attenuation on the online task (S14 r3). The raw-memory
advantage (r0 MC 10.19 substrate vs 6.17 ESN) is substrate physics, not
algorithm. These three facts disentangle the three mechanisms - metadata,
homeostat, substrate physics - which is the reframed thesis of Paper C
(Section 9).

## 6. Related theory (anchor points, not derivations)

- **Slow Feature Analysis** (Wiskott & Sejnowski 2002): the EMA is the
  linear operator that minimizes temporal variation of the readout input,
  i.e. a one-tap SFA on reservoir states. SFA is substrate-independent by
  construction; the equalizer thesis is the statement that slow features
  transfer.
- **Complementary learning systems** (McClelland, McNaughton & O'Reilly 1995)
  and **synaptic consolidation** (Benna & Fusi 2016): fast state + slow trace
  is a two-timescale memory hierarchy; Paper C isolates and characterizes the
  slow component as a forgetting-kernel synthesizer.
- **Fading memory** (Boyd & Chua 1985): the readout computes a fading-memory
  functional; the metadata extends the memory horizon without touching the
  substrate - the algorithmic complement of Paper A's material kernel.

## 7. Data anchors (all committed, full-seed)

| Claim | Data file | Numbers |
|---|---|---|
| Metadata raises ESN accuracy and collapses adaptation | `data/s10_esn_metadata_v1.*` | 0.9955 -> 0.9979; adapt 11.22 -> 0.24 (10 seeds) |
| Equalization band | same | all arms in [0.994, 0.998], steady 1.000 |
| Sequential robustness = homeostat (S11, fast-state features only) | `data/s11_disturbance_chain_v1.*` | MC 8.47 vs 6.41 (+32%), kappa 25.3->28.5 (10 seeds) |
| Metadata does NOT transfer MC robustness to ESN | `data/s14_esn_disturbance_chain_v1.*` | paired diffs -0.78/-0.76/-0.69 at r1/r2/r3, 0/10 positive (10 seeds) |
| Metadata attenuates readout noise on the online task | same | r3 NMSE 0.0251 vs 0.0277 (-9.5%), 10/10 seeds |
| redem_reg reproduces the S11 anchor exactly | same | r0 10.19, r1 7.17, r2 7.51, r3 8.47 |
| Substrate raw memory > ESN raw memory | same | r0 MC 10.19 vs ~6.17 (both ESN arms) |
| Regime length | `scripts/streaming_tasks.py` | RS_REGIME_LEN = 1500 |
| Slow trace definition | `scripts/integrated_benchmark.py` | slow_ema, tau_m = 500 |

## 8. Experiments: status and results

### s14 - ESN+metadata under the disturbance chain (DONE, 10 seeds)

Script `scripts/s14_esn_disturbance_chain.py` (Type: PAPER). Three arms:
esn_fast (ESN fast states + RLS), esn_dual (ESN + slow trace tau_m=500 +
RLS), redem_reg (S11 regulated arm, re-run via the s11 function for exact
reproducibility). Disturbance chain cumulative, substrate-agnostic:
timescale drift (substrate tau*=1.5; ESN leaking rate /=1.5) at t=7k,
structure prune (substrate 40% edges; ESN 40% weights) at t=14k, readout
noise sigma=0.1 at t=21k. Metrics per round: NMSE (Mackey-Glass online
task) and Jaeger MC heldout probe at the settled physics (features [fast]
or [fast, slow]).

Mean over 10 seeds:

| arm | r0_nmse | r3_nmse | r0_mc | r1_mc | r2_mc | r3_mc |
|---|---|---|---|---|---|---|
| esn_fast | 0.0069 | 0.0277 | 6.17 | 1.82 | 1.76 | 1.74 |
| esn_dual | 0.0069 | 0.0251 | 6.16 | 1.04 | 1.00 | 1.05 |
| redem_reg | 0.0657 | 0.1063 | 10.19 | 7.17 | 7.51 | 8.47 |

Readings (all paired, 10 seeds):

- MC r0: esn_dual = esn_fast (paired diff -0.0035) - no episodic-memory
  effect of the slow trace, as Prop. 1 predicts for i.i.d. input.
- MC r1-r3: esn_dual consistently BELOW esn_fast (paired diffs
  -0.78/-0.76/-0.69, 0/10 seeds positive) - the metadata does not transfer
  MC robustness to the ESN.
- NMSE r3 (readout noise): esn_dual BETTER (0.0251 vs 0.0277, -9.5%,
  10/10 seeds) - the EMA low-pass attenuation of Prop. 4 transfers.
- redem_reg reproduces the S11 anchor exactly (10.19/7.17/7.51/8.47) and
  recovers +18% r1->r3 via the homeostat; neither ESN arm recovers.

Interpretation: three mechanisms, three roles - metadata = timescale
coverage + noise attenuation (statistical memory); homeostat = disturbance
recovery; substrate physics = raw memory capacity (r0 MC 10.19 vs 6.17).

### Remaining experiments (not yet run)

1. **s15 - Controlled adaptation protocol.** Known switch instants; report
   T_adapt distribution per arm (fixes the windowed-statistic ambiguity of
   S10's adapt_time). Needed for the Prop. 3 claim.
2. **s16 - tau_m sweep on ESN and REDEM.** tau_m in {50, 200, 500, 1000,
   2000}; verify (a) the S10 ceiling is controlled by tau_m, (b) the
   Prop. 1 kernel prediction on a correlated-input task, and (c) whether
   the S14 MC finding (metadata does not add episodic memory) is robust to
   feature standardization and tau_m - the falsification should be stress-
   tested before it enters the paper.
3. **s17 - substrate stress (optional).** Vary ESN spectral radius
   {0.7, 0.9, 0.99} and heterogeneity to show the S10 equalizer gain is
   substrate-insensitive.

## 9. Headline claims (honest wording; s14 locked, s15-s16 pending)

1. A single algorithmic component - a slow exponential trace of reservoir
   states - constitutes a synthesized forgetting kernel with a controllable
   horizon (Proposition 1).
2. On long-horizon statistical tasks the performance bottleneck is timescale
   coverage, not substrate physics; the same slow trace equalizes ESN and
   Si3N4 reservoirs on accuracy and collapses post-switch adaptation ~47x
   (Proposition 2-3; S10 data).
3. The slow trace is a *statistical memory*, not an episodic one: it does
   not add raw memory capacity (S14: MC unchanged at nominal physics, and
   slightly degraded under disturbance on the ESN, 0/10 seeds positive) but
   it attenuates readout noise on the online task (S14 r3 NMSE -9.5%,
   10/10 seeds; Proposition 4).
4. Three mechanisms, three roles (the disentanglement thesis, S10 + S11 +
   S14): metadata = timescale coverage and noise attenuation (statistical
   memory); homeostat = sequential disturbance recovery (+32% in S11,
   +18% r1->r3 in S14, kappa 25.3->28.5); substrate physics = raw memory
   capacity (r0 MC 10.19 vs ~6.17 for the ESN).

## 10. Open decisions (user gates)

- **Thesis reframing (decided by s14's falsification; user confirmation
  required).** The original "sequential robustness transfers" claim is
  falsified: the +32% recovery is homeostat-driven. Recommended thesis:
  the three-mechanism disentanglement (Section 9, claim 4) - metadata is a
  substrate-agnostic *statistical* memory; disturbance recovery is the
  homeostat's job; raw memory is the substrate's physics.
- **Title.** If the disentanglement thesis is adopted, "Disturbance
  Robustness" should move out of the title (it is not the metadata's
  property); suggested working title becomes: "The Metadata Equalizer:
  a substrate-agnostic statistical memory for online reservoirs".
- Venue: Neurocomputing vs Neural Networks (short/communication) vs similar.
- Run s15-s16 before drafting? (Recommended: yes - s15 fixes the Prop. 3
  adaptation claim, s16 stress-tests the falsification before it is printed.)
- Citation closure: cite Paper B Section 4.5 as the seed of Paper C once B is
  public (preprint or acceptance).
