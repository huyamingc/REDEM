# Paper C - Sketch: The Metadata Equalizer

Working title: The Metadata Equalizer: Substrate-Agnostic Slow-Trace Transfer
and Disturbance Robustness in Online Reservoir Computing

Status: pre-draft. Derivation in `DERIVATION.md`. No new data run yet
(s14-s17 pending, see `DERIVATION.md` Section 8). Target length: short paper
~6 pages (Neurocomputing or Neural Networks short/communication - decide
before submission).

## Positioning (one paragraph, honest)

A single mechanism - the slow exponential trace of reservoir states (M3) - is
a synthesized forgetting kernel: controllable horizon, exponential tail,
substrate-independent by construction (Prop. 1). On long-horizon statistical
tasks the bottleneck is timescale coverage, not physics; the same slow trace
equalizes an ESN and the Si3N4 substrate to a common performance band
(Prop. 2). It collapses post-switch adaptation ~47x (Prop. 3) and is robust
under sequential disturbance (Prop. 4). We never claim REDEM beats the ESN:
in S10 the ESN+metadata arm has the best overall accuracy; the paper is about
the mechanism, not the substrate.

## Section outline

1. **Introduction** (~0.7 p)
   - Hook: frozen batch models (incl. large foundation models) cannot adapt
     post-deployment without expensive retraining; online reservoirs can, but
     their readouts face a timescale gap on long-horizon statistical tasks.
   - The gap: fast reservoir states decorrelate faster than the task
     statistics; the readout must re-integrate after every change.
   - Claim: one algorithmic component closes the gap on any reservoir.
   - Cite Paper B Section 4.5 (metadata-transfer seed) and Paper A Section 5
     (material forgetting kernel).
2. **Setup** (~0.8 p)
   - regime_switch task (L = 1500 pulses), reservoir family (ESN,
     Si3N4 substrate), OnlineRLS readout (lambda_f = 0.999), slow trace
     (EMA, tau_m = 500).
   - Equations (1)-(2) of `DERIVATION.md`.
3. **Theory: the slow trace as a synthesized forgetting kernel** (~1.2 p)
   - Prop. 1 derivation (convolution of EMA kernel with fast kernel; peak
     lag; controllable tail). Contrast with Paper A material kernel M(t);
     M_eff = M (x) h_m; 1/e horizon ~ tau_m + tau0.
   - Prop. 2 argument (timescale coverage, not physics).
   - Prop. 3 argument (feature stationarity vs weight integration).
   - Prop. 4 argument (EMA low-pass; short-transient attenuation; re-anchoring
     on persistent change).
4. **Experiments** (~2 p)
   - S10 equalization: 3 arms x 10 seeds (esn_fast / esn_dual / redem);
     Table 1: overall acc, steady acc, adapt time.
   - s16 tau_m sweep (pending): ceiling vs tau_m; MC(k) 1/e horizon shift
     (Prop. 1 prediction).
   - s14 disturbance chain on ESN+metadata (pending): sequential robustness
     transfer; Table 2 (or figure): MC per round, arms esn_fast / esn_dual /
     redem-regulated.
   - s15 controlled adaptation protocol (pending): T_adapt distribution.
5. **Discussion** (~0.8 p)
   - Equalizer thesis; honesty paragraph (esn_dual best overall; REDEM's
     differentiation is Paper B's story).
   - Relation to SFA (Wiskott & Sejnowski 2002), complementary learning
     systems (McClelland 1995), consolidation (Benna & Fusi 2016), fading
     memory (Boyd & Chua 1985).
   - Limitations: slow-trace disturbance attenuation covers fast-channel
     transients; direct corruption of the metadata memory is out of scope;
     tau_m is a hyperparameter (task-timescale tuning).
   - Future work: consolidation of stable knowledge into structure (cite
     Paper B Conclusion future-work paragraph).
6. **Conclusion** (~0.3 p) - three claims of Section 9 in
   `DERIVATION.md`.

## Figures (vector PDF only, per repo rule)

- Fig. 1: slow-trace kernel h_m(t) vs substrate kernel M(t) (Eq. 3 and
  Paper A M(t)); log-log tails. (FIG script: `gen_paperC_fig1_kernel.py`)
- Fig. 2 (if s16 done): MC(k) 1/e horizon vs tau_m; augmented vs fast-only.
- Fig. 3 (if s14 done): disturbance chain MC per round, esn_fast / esn_dual /
  redem-regulated.

## Tables

- Table 1: S10 arms x {overall acc, steady acc, adapt time} (committed data).
- Table 2: s14 rounds x arms x MC (pending data).
- Table 3 (optional): s16 tau_m sweep (pending data).

## Key numbers (from committed data, do not re-derive)

- esn_fast: overall 0.9955 +- 0.0015; steady 1.000; adapt 11.22 +- 10.67.
- esn_dual: overall 0.9979 +- 0.0001; steady 1.000; adapt 0.24 +- 0.17.
- redem: overall 0.9942 +- 0.0011; steady 1.000; adapt 11.26 +- 5.91.
- S11 regulated r3: MC 8.47 +- 0.39 vs fixed 6.41 +- 0.41 (+32%); kappa
  25.3 -> 26.2 -> 27.4 -> 28.5.
- RS_REGIME_LEN = 1500; tau_m = 500; tau_f = 1000 (lambda_f = 0.999);
  substrate 1/e horizon ~16 pulses (Paper A).

## Submission checklist (Paper C)

- [ ] Replace author placeholder; affiliation: Independent Researcher,
      Guiyang, China.
- [ ] Compile with target journal class (Neurocomputing: elsarticle-style;
      Neural Networks short: elsarticle.cls).
- [ ] Honest-ESN wording: the equalizer is about the mechanism, not about
      beating the ESN; esn_dual has the best overall accuracy - state it.
- [ ] Cite Paper A Section 5 and Paper B Section 4.5 (citation loop).
- [ ] Code/data availability: REDEM repo (private during review).
