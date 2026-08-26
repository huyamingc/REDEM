# Paper D - REDEM-SSM

**Title (user proposal, final 2026-02-19)**: REDEM-SSM: A
State-Space Architecture with Native Online Learning, Meta-Adaptation, and
Structural Plasticity (was "A Foundation Model Architecture ..." — dropped
as overclaiming for a toy-scale, CPU-only PoC).

**Status**: P1–P5 DONE 2026-02-18. P1/P3a falsified with mechanisms
isolated (exact token recovery is a deconvolution impossibility — Prop. 1;
the pooled state readout fails out of sample (ppl 58–116), 0/10; M1 works
on the additive input path, 10/10). **Revision pass 3 (review audit)**: the
P1 corollary "ridge finds no useful linear map because none exists" was
**retracted and replaced** by a scoped boundary analysis (s35): the current
token IS linearly decodable from the state's fast channels (τ≤8,
88.9–99.7% out-of-sample, chance 3.1%), the in-sample ridge oracle is
window-dependent (31.2 full vs 17.3 half) and nested-violating (skip 18.0
> proj 7.25 on 10/10 seeds), and fast-channel-only direct/two-stage
next-token readouts still fail out-of-sample (ppl 68–104 vs static table
13.9–17.4) — the measured 0/10 failure is a property of the pooled
squared-loss readout under the CE metric (no calibrated next-token
probabilities from any closed-form linear state readout), not missing
linear information.
Related-work section added (activates all 13 bibliography entries);
code/data availability statement added. P2 partially replicated (M3
routing transfers, forgetting −2.05…−1.20 at τ_m≤1000, 10/10; gating-only
inverts on RLS readouts — readout-dynamics-dependent, with the
what-is-paused qualifier). P3 both
supported (soft routing beats abrupt −1.81, 10/10; dormant covariance
refresh flips the soft-routing stream result to −3.54; M5 homeostat
bounds the state 11.3 vs 50.2
and restores the full-state EMA detector 5/5). P4 benchmark (4-domain
irregular switches) both hypotheses 10/10 (REDEM vs bare −2.25/−4.47, vs
TF-A1 −9.28/−10.08) + real-text transfer (S23, two Gutenberg books:
REDEM vs bare −1.27 stream 10/10, vs TF-A1 −5.23 10/10; corpus in
`../data/corpora/`). **P5: PAPER_D.tex draft (14 pp, zero warnings; author
info filled; revision pass 1: A2-claim qualifier + real-text results;
revision pass 2 (audit follow-ups): P1 deconvolution theorem
(Proposition 1 + Krylov proof), tuned-TF reference grid + 4-adapter TF-A3
routing (s26), char-bigram oracle (s31), M5-in-P4 honest negative (s33);
revision pass 3 (review audit): P1 corollary rewritten with s35 boundary
probes, related-work section, availability statement).**
Data: `data/s19_ssm_rls_readout_v1.csv`, `data/s20_ssm_m3_routing_v1.csv`,
`data/s21_ssm_m4_m5_v1.csv`, `data/s22_ssm_p4_benchmark_v1.csv`,
`data/s23_ssm_p4_realtext_v1.csv`, plus follow-ups `s26_ssm_p4_fair_tf_v1.*`,
`s31_char_bigram_oracle_v1.*`, `s33_ssm_p4_m5_v1.*`,
`s35_readout_boundary_probe_v1.*`. Next: submission
checklist or arXiv format pass.

## Origin and motivation

Paper C §7 conclusion point (6) (added 2026-02-18) states the host-boundary
result: the s18 negative result (A2 gating 0/10 seeds) together with the
narrow A3 routing window (stream-ppl benefit only at tau_m <= 500) show that
the *full* REDEM framework does not instantiate on a frozen-feedforward
host. The failure is attributed to the update policy (M1 removed, M4
switching made abrupt), not to the Transformer per se --- routing transferred
on most of the operating range. Paper D takes the constructive consequence:
design a foundation model whose host natively provides the three
requirements (a continuously active online readout, a slowly integrating
statistical memory, and gradually applied structural switches) instead of
retrofitting them.

## Non-overlap with Papers A, B, C (hard rules)

| Paper | Owns |
|---|---|
| A | substrate physics: log-normal tau spectrum, FTLE, material forgetting kernel M(t) |
| B | the REDEM algorithm as such (RLS readout, M3 EMA metadata, M4 plasticity, M5 homeostat) on the physical substrate |
| C | the three-mechanism disentanglement thesis + the Transformer transfer boundary (s18) |
| **D (this work)** | the **SSM instantiation** of the mechanisms (A-diagonal tau-spectrum mapping, per-token RLS readout, EMA second state, dynamic state-dimension/spectrum plasticity, eigenvalue-monitored dt regulation) and its empirical validation on streaming benchmarks |

Paper D cites A/B/C as foundations; Paper C §7 point (6) is its roadmap
entry. Paper D never re-derives substrate physics (A), the REDEM algorithm
(B), or the disentanglement thesis (C).

## Honesty / falsification discipline (carried from C)

1. **D2 boundary**: M5-on-SSM is *new research* (stability regulation), NOT
   validation of the substrate homeostat. Paper A's FTLE edge-of-chaos
   results do not port literally: a linear SSM is not chaotic. The M5
   analogue regulates the linear stability edge (|lambda| -> 1 / state-norm
   growth rate), which is an analogy, not an identity.
2. **"Training = inference" is not claimed**: SSM parallel training is
   offline gradient training; M1 is the per-token online RLS at inference.
   These are distinct and both are stated.
3. **Related-work honesty**: online learning inside SSMs already exists
   (Longhorn ICLR 2025, Titans NeurIPS 2025, TTT ICLR 2024, test-time SSM
   alignment for user-interest tracking). Paper D's claimed novelty is the
   three-mechanism *division of labor* + the *second-order (RLS)* update +
   *mechanism-level* stability regulation --- not "first online learning in
   an SSM".
4. **Scale honesty**: the PoC is tiny by design (state dim 64-256, ~1M
   params, torch CPU, s18 pattern). No SOTA or scaling claims. Benchmarks
   are synthetic multi-domain streams (s16 protocol generalized) + small
   char-level corpora (s18 style); WikiText-103 streaming is NOT feasible on
   CPU and is not promised.
5. **10-seed falsification discipline**: every claim is paired per-seed with
   sign-consistency counts (n_pos/10), exactly as Papers A-C.

## Contents

| File | Purpose |
|---|---|
| `README.md` | This file: status, non-overlap rules, honesty discipline |
| `PAPER_D_sketch.md` | The design doc: positioning, REDEM-SSM mapping, novelty, falsifiable predictions, roadmap P1-P5 |
| `PAPER_D.tex` | First draft (elsarticle preprint, 14 pp, zero warnings; P1-P4 evidence + honest falsification narrative + P1 theorem / boundary probes / reference fairness / M5 negative / oracle / related work); figures in `../figures/paperD_fig*.pdf` |

## Roadmap (corrected estimates, part-time 3-4 months)

| Phase | Task | Deliverable | Estimate |
|---|---|---|---|
| P1 | Hand-rolled diagonal SSM (state dim 64-256) + per-token RLS on output projection; streaming tracking task | Working REDEM-SSM prototype (torch CPU, no mamba-ssm dependency) | 1-2 w |
| P2 | M3 EMA second state; drift detection on two-domain stream | M3-on-SSM drift detection + routing evidence | 1 w |
| P3 | M4 (state-dim activation / A-spectrum prune-grow, "gentle wins") + M5 (log|lambda| / state-norm monitor -> adjust dt) | Complete REDEM-SSM system | 2-3 w |
| P4 | Benchmarks: synthetic multi-domain streams + small char-level corpora vs bare SSM vs Transformer+LoRA reference (s18 reuse) | Full comparison data (10 seeds) | 1-2 w |
| P5 | Paper writing; arXiv + workshop first | Submission-ready draft | 2-3 w |

Total ~8-12 weeks full-time (optimistic), 3-4 months part-time. Phase
gates: each phase's exit criteria are defined in `PAPER_D_sketch.md`
Section 6; a falsified exit criterion stops the phase and is reported, not
silently worked around.

## Scripts, data, figures (P1–P5, all committed)

All Paper D scripts are ML/FIG class per `CLAUDE.md` (torch CPU, no `@njit`);
the shared substrate/generators come from `s19_ssm_rls_readout.py` (imported
by s20–s23). 10-seed paired discipline, `torch.manual_seed` per trial,
CSV+JSON dual output.

| Script | Type | Experiment | Key result |
|---|---|---|---|
| `../scripts/s19_ssm_rls_readout.py` | ML | P1/P3a: per-token RLS readout on a diagonal SSM (7 arms × 10 seeds) | state arms 58.5–115.9 ppl (0/10) vs B-proj input path 11.75 (d −3.26, 10/10); oracles 7.25 (mean; 7.15–7.36) |
| `../scripts/s20_ssm_m3_routing.py` | ML | P2: M3 fast-channel EMA + drift detection + routing (A1/A2/A3 × τ_m × 10) | A2 stream −1.52…−2.45 (10/10); A3 forget −2.05/−1.87/−1.20 at τ_m≤1000 (10/10) |
| `../scripts/s21_ssm_m4_m5.py` | ML | P3: M4 soft vs abrupt routing + M5 state-norm homeostat (E1/E2 × 10) | soft 8.22 vs abrupt 10.03 (−1.81, 10/10); M5 flips detector 5/5, norm 11.3 vs 50.2 |
| `../scripts/s22_ssm_p4_benchmark.py` | ML | P4: 4-domain irregular-switch benchmark (3 arms × 10) | REDEM-SSM 13.18/8.93 vs bare 15.43/13.40 vs TF 22.46/19.01 (10/10) |
| `../scripts/s23_ssm_p4_realtext.py` | ML | real-text transfer: Alice vs Dickens, 32-symbol chars (3 arms × 10) | REDEM-SSM 12.07/11.85 vs bare 13.34/12.31 vs TF 17.31/13.86 (−1.27/−5.23, 10/10) |
| `../scripts/s26_ssm_p4_fair_tf.py` | ML | fair TF references for P4: tuned A1 grid (4 lrs × 2 ranks) + 4-adapter A3 routing (9 arms × 10) | tuning cuts the stream gap to −1.68 (0/10) but collapses retention to 62.2; TF-A3 transfers −8.17 forgetting (10/10) without fixing stream (+8.59) — mechanisms host-agnostic, stream performance is not |
| `../scripts/s31_char_bigram_oracle.py` | PAPER | char-bigram oracle on the real-text protocol (full-book vs ref-window fits, 10 seeds) | true first-order ceiling ppl 10.97 ± 0.18; REDEM-SSM 12.07 within ~1.1 ppl of it — first-order scope now quantitative |
| `../scripts/s33_ssm_p4_m5.py` | ML | M5 state-norm homeostat added to the P4 stack (2 arms × 10) | honest negative: REDEM-SSM+M5 worse 10/10 (stream +1.53, forgetting +2.90, t=−22.8/−26.5) — Δt modulation breaks Δt=1 whitening; S22's exclusion of M5 validated |
| `../scripts/s35_readout_boundary_probe.py` | PAPER | P1 readout boundary probes (10 seeds, s19 host verbatim): full/half-window oracle, skip-vs-proj nested check, fast/slow token decoding, fast-channel next-token readouts vs static-table reference | full-window oracle 31.2 (matches s19, max diff 0.00) vs half-window 17.3 — the oracle is window-dependent; skip 18.0 > proj 7.25 (nested violation, 10/10); current token linearly decodable from fast channels (τ≤8) at 88.9–99.7% out-of-sample (chance 3.1%), slow channels at chance; fast-channel-only direct/two-stage next-token readouts still fail out-of-sample (ppl 68–104 vs static table 13.9–17.4) — "no useful linear map" is false, but no closed-form squared-loss state readout yields calibrated next-token probabilities; the P1 failure is a pooled-readout/metric property, not missing linear information |
| `../scripts/gen_paperD_fig1_p1_arms.py` | FIG | Paper D Fig. 1 | `../figures/paperD_fig1_p1_arms.pdf` |
| `../scripts/gen_paperD_fig2_routing.py` | FIG | Paper D Fig. 2 | `../figures/paperD_fig2_routing.pdf` |
| `../scripts/gen_paperD_fig3_benchmark.py` | FIG | Paper D Fig. 3 | `../figures/paperD_fig3_benchmark.pdf` |

Data: `../data/s19_ssm_rls_readout_v1.{csv,json}`,
`../data/s20_ssm_m3_routing_v1.{csv,json}`,
`../data/s21_ssm_m4_m5_v1.{csv,json}`,
`../data/s22_ssm_p4_benchmark_v1.{csv,json}`,
`../data/s23_ssm_p4_realtext_v1.{csv,json}`,
`../data/s26_ssm_p4_fair_tf_v1.{csv,json}`,
`../data/s31_char_bigram_oracle_v1.{csv,json}`,
`../data/s33_ssm_p4_m5_v1.{csv,json}`,
`../data/s35_readout_boundary_probe_v1.{csv,json}`;
corpus `../data/corpora/` (Gutenberg #11 Alice, #98 Dickens; public domain,
see `../data/corpora/README.md`).

## Reproduction (CPU-only, project venv)

```powershell
& ..\.venv\Scripts\python.exe ..\scripts\s19_ssm_rls_readout.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s20_ssm_m3_routing.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s21_ssm_m4_m5.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s22_ssm_p4_benchmark.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s23_ssm_p4_realtext.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s26_ssm_p4_fair_tf.py
& ..\.venv\Scripts\python.exe ..\scripts\s31_char_bigram_oracle.py
& ..\.venv\Scripts\python.exe ..\scripts\s33_ssm_p4_m5.py
& ..\.venv\Scripts\python.exe ..\scripts\s35_readout_boundary_probe.py --workers 4
& ..\.venv\Scripts\python.exe ..\scripts\gen_paperD_fig1_p1_arms.py
& ..\.venv\Scripts\python.exe ..\scripts\gen_paperD_fig2_routing.py
& ..\.venv\Scripts\python.exe ..\scripts\gen_paperD_fig3_benchmark.py
```

All results are committed under `../data/` (CSV per run + JSON with params
and aggregates); figures are vector PDF under `../figures/`. The paper
includes the extension-less basename so `pdflatex` picks the vector file.

## Submission checklist (Paper D — arXiv + workshop first, NeurIPS/ICML stretch)

- [x] Title revised (2026-02-19): "A State-Space Architecture ..." —
      "Foundation Model" dropped as overclaiming for the toy-scale PoC.
- [x] Author filled (2026-02-19): Yaming Hu, ORCID 0009-0003-1406-0485,
      Independent Researcher, Guiyang, Guizhou Province, China;
      64687555@qq.com. Abstract note / cover letter: `COVER_LETTER.md`.
- [x] `PAPER_D.tex` compiles: elsarticle preprint, 14 pp, zero warnings.
- [x] P1–P5 evidence committed: s19–s23 + Fig 1–3 + data CSVs/JSONs (10-seed
      paired discipline, n_pos/10 sign consistency throughout).
- [x] Honesty scoping in text: CPU-only, toy-scale, no scaling claims;
      deconvolution impossibility scoped to linear readouts (formalized as
      Proposition 1); TF-A1 reference fairness closed by the tuned grid and
      TF-A3 routing baseline (s26 — best tuned stream 14.86 vs 13.18, 10/10,
      at the cost of catastrophic retention); real-text first-order ceiling
      quantified by the bigram oracle (s31, ppl 10.97 ± 0.18); M5-in-P4
      reported as a negative (s33).
- [x] P1 corollary revised (review audit): "ridge finds no useful linear map
      because none exists" retracted; replaced by the s35 boundary analysis
      (fast-channel token decodability 88.9–99.7% out-of-sample;
      window-dependent and nested-violating oracle) scoping the failure to
      the pooled readout under the CE metric.
- [x] Related work section added (activates all 13 bibliography entries:
      Mamba/S4/Longhorn/Titans/TTT/DeltaNet/LoRA/Transformer/ESN/continual
      learning).
- [x] Code and data availability statement added (matches Papers A–C
      wording).
- [ ] Optional before arXiv: per-mechanism ablation table.
- [ ] Optional: arXiv abstract/format check; workshop CFP match.
