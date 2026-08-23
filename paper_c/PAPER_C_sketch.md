# Paper C - Sketch: Three Mechanisms, Three Roles

Working title (user proposal, adopted): Dissecting Online Learning
Mechanisms: Statistical Memory, Homeostatic Recovery, and Substrate Physics
are Non-Transferable

Status: post-s14, post-s16, post-s15, post-s16b, post-s5b (all 10 seeds).
The three-mechanism disentanglement thesis is adopted (user gate,
2026-02-17). The decisive experiments are DONE: s14 falsifies "metadata
transfers disturbance robustness" (the +32% recovery is homeostat-driven);
s16 confirms the strong claim at every tau_m in {200, 500, 1000, 2000};
s16b shows the falsification is robust in sign to the probe protocol but
protocol-dependent in magnitude (readout-noise -0.69 vs state-noise -0.01);
s15/s5b re-measure adaptation with known switch instants (ESN arms ~10
pulses + variance collapse; substrate arms factor 1.3-2.4; the old
"47x"/"9-20x" ratios were metric artifacts, and Paper B was revised).
Only s17 (optional substrate stress) remains before drafting. PAPER_C.tex
drafted (target ~6 pages; venue decision pending).

## Positioning (one paragraph, honest)

A single mechanism - the slow exponential trace of reservoir states (M3) - is
a synthesized forgetting kernel: controllable horizon, exponential tail,
substrate-independent by construction (Prop. 1). On long-horizon statistical
tasks the bottleneck is timescale coverage, not physics; the same slow trace
equalizes an ESN and the Si3N4 substrate on accuracy and speeds post-switch
adaptation (s15 controlled protocol: ~10 pulses of ~200, plus variance
collapse from p90 76.5 to 42; the S10 "47x" ratio was a metric artifact).
It is a *statistical* memory, not an episodic one: it adds no raw memory
capacity and does not transfer disturbance robustness to an ESN at any
metadata timescale (s14 + s16: MC paired diffs -0.68 to -0.70, 0/10 seeds
positive; s16b: robust in sign to probe protocol). Its transferable value is
denoising STATE-level corruption (s16b V2: gap -0.69 -> -0.01 when noise
enters before the slow trace) plus readout-noise attenuation on the online
task (r3 NMSE -9.5% to -15%). Sequential disturbance recovery (+32% in S11,
+18% r1->r3 in S14) is produced by the homeostat, not the metadata. Three
mechanisms, three roles: metadata = timescale coverage and state-noise
denoising; homeostat = disturbance recovery; substrate physics = raw memory
capacity (r0 MC 10.19 vs ~6.17 for the ESN). We never claim REDEM beats the
ESN: in S10 the ESN+metadata arm has the best overall accuracy.

## Section outline

1. **Introduction** (~0.7 p)
   - Hook: frozen batch models (incl. large foundation models) cannot adapt
     post-deployment without expensive retraining; online reservoirs can,
     but their readouts face a timescale gap on long-horizon statistical
     tasks.
   - The question: which mechanism in an online learning system does what?
     Three candidates: slow-trace metadata, homeostatic regulation, and
     substrate physics.
   - Claim: a falsifying transfer experiment dissects them - each role is
     functionally non-transferable. Cite Paper B Section 4.5 and Paper A
     Section 5.
2. **Setup** (~0.8 p)
   - regime_switch task (L = 1500 pulses), reservoir family (ESN, Si3N4
     substrate), OnlineRLS readout (lambda_f = 0.999), slow trace
     (EMA, tau_m). Equations (1)-(2) of `DERIVATION.md`.
   - Disturbance chain protocol with substrate-agnostic disturbance
     definitions (S14): timescale drift, structure prune, readout noise.
3. **Theory: the slow trace as a synthesized forgetting kernel** (~1.2 p)
   - Prop. 1 derivation (convolution of EMA kernel with fast kernel; peak
     lag; controllable tail). Contrast with Paper A material kernel M(t);
     M_eff = M (x) h_m; 1/e horizon ~ tau_m + tau0.
   - Prop. 2 argument (timescale coverage, not physics).
   - Prop. 3 argument (feature stationarity vs weight integration).
   - Prop. 4 argument (EMA low-pass; readout-noise attenuation; NOT a
     memory-capacity protection - stated as such, confirmed by s14/s16).
4. **Experiments** (~2.2 p)
   - S10 equalization: 3 arms x 10 seeds (esn_fast / esn_dual / redem);
     Table 1: overall acc, steady acc, adapt time.
   - S14 disturbance chain: 3 arms x 10 seeds; Table 2: per-round MC and
     r3 NMSE. Headline: redem_reg reproduces the S11 anchor exactly
     (10.19/7.17/7.51/8.47) and recovers via the homeostat; esn_dual shows
     no MC benefit (paired diffs -0.78/-0.76/-0.69) but better r3 NMSE
     (-9.5%).
   - S16 tau_m pressure test: 4 tau_m x 10 seeds; Table 3 (or Fig. 2):
     r3 MC vs tau_m. Strong claim: no tau_m closes the gap (0/10 seeds
     positive; paired diffs -0.68 to -0.70); noise attenuation transfers
     at all tau_m (r3 NMSE 0.0235-0.0254 vs 0.0277).
   - S16b probe stress test: 2 tau_m x 3 variants x 10 seeds; falsification
     robust in sign (0/10 positive everywhere); magnitude protocol-
     dependent: V0 (readout noise) -0.69, V1 (std-slow) -0.66, V2
     (state noise) -0.01 - the metadata denoises state-level corruption.
   - S15 controlled adaptation (esn arms): known switch instants; T40 40.6
     (dual) vs 49.9 (fast) vs 52.7 (redem), T40 p90 42 vs 76.5.
   - S5b controlled re-measurement (substrate arms): T200 fast 264.8-303.9
     vs dual/slow 202-211 pulses (factor 1.3-2.4); the "47x"/"9-20x"
     adaptation ratios are window-position artifacts - report the honest
     factors + variance collapse.
5. **Discussion** (~0.8 p)
   - The disentanglement thesis; honesty paragraph (esn_dual best overall
     accuracy; the +32% is homeostat, not metadata; substrate MC advantage
     is physics).
   - Why non-transferability is informative: it delineates the boundary
     conditions of each mechanism (statistical memory vs episodic memory vs
     recovery).
   - Relation to SFA (Wiskott & Sejnowski 2002), complementary learning
     systems (McClelland 1995), consolidation (Benna & Fusi 2016), fading
     memory (Boyd & Chua 1985).
   - Limitations: slow-trace attenuation covers fast-channel readout noise;
     direct corruption of the metadata memory is out of scope; tau_m is a
     hyperparameter.
   - Future work: consolidation of stable knowledge into structure (cite
     Paper B Conclusion future-work paragraph).
6. **Conclusion** (~0.3 p) - four claims of Section 9 in
   `DERIVATION.md`.

## Figures (vector PDF only, per repo rule)

- Fig. 1 (DONE): slow-trace kernel h_m(k) vs substrate kernel M(k)
  (Eq. 3 and Paper A M(t)); log-log tails, tau_m in {200,500,1000}.
  Output `figures/paperC_fig1_kernel.pdf` via
  `scripts/gen_paperC_fig1_kernel.py`.
- Fig. 2 (DONE): post-disturbance MC recovery vs tau_m, esn_dual vs
  esn_fast baseline and redem_reg anchor; right panel r3 NMSE vs tau_m.
  Output `figures/paperC_fig2_recovery.pdf` via
  `scripts/gen_paperC_fig2_recovery.py`.

## Tables

- Table 1: S10 arms x {overall acc, steady acc, adapt time} (committed
  data).
- Table 2: S14 arms x {r0/r1/r2/r3 MC, r3 NMSE} + paired diff column
  (committed data, `data/s14_esn_disturbance_chain_v1.*`).
- Table 3: S16 tau_m sweep x {r0/r3 MC, r3 NMSE, paired diff} (committed
  data, `data/s16_tau_m_pressure_test_v1.*`).

## Key numbers (from committed data, do not re-derive)

- S10: esn_fast 0.9955 +- 0.0015 / adapt 11.22; esn_dual 0.9979 +- 0.0001 /
  adapt 0.24; redem 0.9942 +- 0.0011 / adapt 11.26. Steady acc 1.000 all.
- S11: regulated r3 MC 8.47 +- 0.39 vs fixed 6.41 +- 0.41 (+32%); kappa
  25.3 -> 28.5.
- S14 (10 seeds): r0 MC - esn_fast 6.17, esn_dual 6.16, redem_reg 10.19;
  r3 MC - 1.74 / 1.05 / 8.47; paired (dual-fast) MC diffs
  -0.78 / -0.76 / -0.69 at r1/r2/r3 (0/10 positive); r3 NMSE 0.0277 vs
  0.0251 (-9.5%, 10/10 positive for dual).
- S16 (10 seeds): r3 MC esn_dual - 1.04 (tau_m=200), 1.05 (500),
  1.06 (1000), 1.06 (2000); esn_fast 1.74. Paired diffs
  -0.701/-0.693/-0.682/-0.678 (0/10 positive each). r3 NMSE dual
  0.0254/0.0251/0.0235/0.0235 vs fast 0.0277 (9-10/10 better).
- S16b (10 seeds): r3 MC dual V0 1.05/1.06, V1 1.08/1.07, V2 1.72/1.73
  (tau_m 500/2000); fast 1.74. Paired diffs: V0 -0.69/-0.68, V1 -0.66,
  V2 -0.015/-0.009 (0/10 positive all).
- S15 (10 seeds, 5 switches): T40 mean esn_dual 40.6 / esn_fast 49.9 /
  redem 52.7; T40 p90 42.0 / 76.5 / 67.0; T200 mean 199.2 / 210.2 / 210.3;
  overall acc 0.9979 / 0.9955 / 0.9942 (replicates S10).
- RS_REGIME_LEN = 1500; tau_m = 500 (S10/S14); tau_f = 1000 (lambda_f =
  0.999); substrate 1/e horizon ~16 pulses (Paper A).

## Submission checklist (Paper C)

- [ ] Replace author placeholder; affiliation: Independent Researcher,
      Guiyang, China.
- [ ] Compile with target journal class (Neurocomputing: elsarticle-style;
      Neural Networks short: elsarticle.cls).
- [ ] Honest wording locked: equalizer is about statistical memory; +32% is
      homeostat; esn_dual has the best S10 accuracy - state all three.
- [ ] Strong claim wording locked on s16 (no tau_m closes the MC gap);
      report the 0/10-seed sign consistency and ~5-sigma paired diffs.
- [ ] Cite Paper A Section 5 and Paper B Section 4.5 (citation loop).
- [ ] Code/data availability: REDEM repo (private during review).
