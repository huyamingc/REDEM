# Paper C - Sketch: The Metadata Equalizer

Working title (proposed after s14): The Metadata Equalizer: a
Substrate-Agnostic Statistical Memory for Online Reservoirs
(previous title carried "Disturbance Robustness", which s14 assigned to the
homeostat - see `DERIVATION.md` Section 10).

Status: post-s14. The decisive transfer experiment is DONE and falsifies the
original "sequential robustness transfers" thesis; the paper is reframed as
a three-mechanism disentanglement (`DERIVATION.md` Sections 5, 8, 9). s15
(controlled adaptation) and s16 (tau_m sweep + falsification stress test)
still pending. Target: short paper ~6 pages (Neurocomputing or Neural
Networks short/communication - decide before submission).

## Positioning (one paragraph, honest)

A single mechanism - the slow exponential trace of reservoir states (M3) - is
a synthesized forgetting kernel: controllable horizon, exponential tail,
substrate-independent by construction (Prop. 1). On long-horizon statistical
tasks the bottleneck is timescale coverage, not physics; the same slow trace
equalizes an ESN and the Si3N4 substrate on accuracy and collapses post-
switch adaptation ~47x (Props. 2-3, S10). It is a *statistical* memory, not
an episodic one: it adds no raw memory capacity (S14 MC unchanged, 0/10
seeds positive for a benefit) but attenuates readout noise on the online
task (S14 r3 NMSE -9.5%, 10/10 seeds; Prop. 4). Sequential disturbance
recovery (+32% in S11, +18% r1->r3 in S14) is produced by the homeostat,
not the metadata. Three mechanisms, three roles: metadata = timescale
coverage and noise attenuation; homeostat = disturbance recovery; substrate
physics = raw memory capacity (r0 MC 10.19 vs ~6.17 for the ESN). We never
claim REDEM beats the ESN: in S10 the ESN+metadata arm has the best overall
accuracy.

## Section outline

1. **Introduction** (~0.7 p)
   - Hook: frozen batch models (incl. large foundation models) cannot adapt
     post-deployment without expensive retraining; online reservoirs can,
     but their readouts face a timescale gap on long-horizon statistical
     tasks.
   - The gap: fast reservoir states decorrelate faster than the task
     statistics; the readout must re-integrate after every change.
   - Claim: one algorithmic component closes the gap on any reservoir, and
     three distinct mechanisms - metadata, homeostat, substrate physics -
     play three distinct roles. Cite Paper B Section 4.5 and Paper A
     Section 5.
2. **Setup** (~0.8 p)
   - regime_switch task (L = 1500 pulses), reservoir family (ESN, Si3N4
     substrate), OnlineRLS readout (lambda_f = 0.999), slow trace
     (EMA, tau_m = 500). Equations (1)-(2) of `DERIVATION.md`.
   - Disturbance chain protocol with substrate-agnostic disturbance
     definitions (S14): timescale drift, structure prune, readout noise.
3. **Theory: the slow trace as a synthesized forgetting kernel** (~1.2 p)
   - Prop. 1 derivation (convolution of EMA kernel with fast kernel; peak
     lag; controllable tail). Contrast with Paper A material kernel M(t);
     M_eff = M (x) h_m; 1/e horizon ~ tau_m + tau0.
   - Prop. 2 argument (timescale coverage, not physics).
   - Prop. 3 argument (feature stationarity vs weight integration).
   - Prop. 4 argument (EMA low-pass; readout-noise attenuation; NOT a
     memory-capacity protection - stated as such).
4. **Experiments** (~2.2 p)
   - S10 equalization: 3 arms x 10 seeds (esn_fast / esn_dual / redem);
     Table 1: overall acc, steady acc, adapt time.
   - S14 disturbance chain: 3 arms x 10 seeds; Table 2: per-round MC and
     r3 NMSE. Headline: redem_reg reproduces the S11 anchor exactly
     (10.19/7.17/7.51/8.47) and recovers via the homeostat; esn_dual shows
     no MC benefit (paired diffs -0.78/-0.76/-0.69) but better r3 NMSE
     (-9.5%).
   - s16 tau_m sweep (pending): ceiling vs tau_m; MC falsification stress
     test (standardization, tau_m range).
   - s15 controlled adaptation protocol (pending): T_adapt distribution.
5. **Discussion** (~0.8 p)
   - The disentanglement thesis; honesty paragraph (esn_dual best overall
     accuracy; the +32% is homeostat, not metadata; substrate MC advantage
     is physics).
   - Relation to SFA (Wiskott & Sejnowski 2002), complementary learning
     systems (McClelland 1995), consolidation (Benna & Fusi 2016), fading
     memory (Boyd & Chua 1985).
   - Limitations: slow-trace attenuation covers fast-channel readout noise;
     direct corruption of the metadata memory is out of scope; tau_m is a
     hyperparameter (task-timescale tuning).
   - Future work: consolidation of stable knowledge into structure (cite
     Paper B Conclusion future-work paragraph).
6. **Conclusion** (~0.3 p) - four claims of Section 9 in
   `DERIVATION.md`.

## Figures (vector PDF only, per repo rule)

- Fig. 1: slow-trace kernel h_m(t) vs substrate kernel M(t) (Eq. 3 and
  Paper A M(t)); log-log tails. (FIG script: `gen_paperC_fig1_kernel.py`)
- Fig. 2: S14 disturbance chain, per-round MC for the three arms (bar or
  line, 10 seeds); highlights homeostat recovery vs metadata no-benefit.
- Fig. 3 (if s16 done): tau_m sweep; and/or MC(k) horizon shift.

## Tables

- Table 1: S10 arms x {overall acc, steady acc, adapt time} (committed
  data).
- Table 2: S14 arms x {r0/r1/r2/r3 MC, r3 NMSE} + paired diff column
  (committed data, `data/s14_esn_disturbance_chain_v1.*`).
- Table 3 (optional, s16): tau_m sweep.

## Key numbers (from committed data, do not re-derive)

- S10: esn_fast 0.9955 +- 0.0015 / adapt 11.22; esn_dual 0.9979 +- 0.0001 /
  adapt 0.24; redem 0.9942 +- 0.0011 / adapt 11.26. Steady acc 1.000 all.
- S11: regulated r3 MC 8.47 +- 0.39 vs fixed 6.41 +- 0.41 (+32%); kappa
  25.3 -> 28.5.
- S14 (10 seeds): r0 MC - esn_fast 6.17, esn_dual 6.16, redem_reg 10.19;
  r3 MC - 1.74 / 1.05 / 8.47; paired (dual-fast) MC diffs
  -0.78 / -0.76 / -0.69 at r1/r2/r3 (0/10 positive); r3 NMSE 0.0277 vs
  0.0251 (-9.5%, 10/10 positive for dual).
- RS_REGIME_LEN = 1500; tau_m = 500; tau_f = 1000 (lambda_f = 0.999);
  substrate 1/e horizon ~16 pulses (Paper A).

## Submission checklist (Paper C)

- [ ] Replace author placeholder; affiliation: Independent Researcher,
      Guiyang, China.
- [ ] Compile with target journal class (Neurocomputing: elsarticle-style;
      Neural Networks short: elsarticle.cls).
- [ ] Honest wording locked: equalizer is about statistical memory; +32% is
      homeostat; esn_dual has the best S10 accuracy - state all three.
- [ ] Cite Paper A Section 5 and Paper B Section 4.5 (citation loop).
- [ ] Code/data availability: REDEM repo (private during review).
