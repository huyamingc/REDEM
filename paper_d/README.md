# Paper D — REDEM-SSM: A State-Space Architecture with Native Online Learning, Meta-Adaptation, and Structural Plasticity

Paper D of the four-paper REDEM series (**role: architecture**). It
instantiates the REDEM mechanisms — an online readout (M1), a slow-trace
statistical memory (M3), structural plasticity (M4), and a stability
homeostat (M5) — natively on a state-space host (a diagonal linear
recurrence with a log-uniform timescale spectrum), instead of retrofitting
them onto a frozen-feedforward Transformer.

- **Current submission version (Neurocomputing):**
  PDF: [`PAPER_D.pdf`](PAPER_D.pdf) | LaTeX:
  [`PAPER_D.tex`](PAPER_D.tex) — `elsarticle` class.
  Highlights: [`Highlights.docx`](Highlights.docx) (single source of
  truth; the older `.txt`/`_notes.md` working files were removed on
  2026-09-17).
  Cover letter: [`COVER_LETTER.docx`](COVER_LETTER.docx) (single source
  of truth; the older `COVER_LETTER_D_NC.txt` was removed on
  2026-09-17).
- Target history: ~~*PRX Intelligence* (APS)~~ **desk-rejected (2026-09): scope mismatch**
  — the paper is CS/ML, not physics-advancing. Intermediate retarget:
  *Applied Intelligence* (Springer).
- Historical sources, AI-line files, change log, and internal notes
  were removed on 2026-09-17 (git history retains the last committed
  state; the superseded `PAPER_D_NC.*` manuscript was deleted as part
  of the same cleanup).
  (not part of the submission folder).
- Preprint: [doi:10.5281/zenodo.22110623](https://doi.org/10.5281/zenodo.22110623)
- Code and data: <https://github.com/huyamingc/REDEM>
- Series overview and reading order (A → B → C → D):
  [`../README.md`](../README.md)

## Key results

A falsifying development sequence (10-seed paired discipline throughout)
isolates three host requirements:

1. **The readout must use a calibrated input path.** A linear readout on a
   linearly-decayed state mixture cannot recover the current token exactly
   (a deconvolution impossibility), and the pooled state readout fails out
   of sample under the clipped-linear stream metric (0/10 seeds). That
   ranking is a *calibration* property: coverage-spectrum state arms achieve
   lower unclipped CE while parking 10–12% of tokens on the clip floor
   (the log-normal arms park 7.5–8.3%),
   whereas the input-path control clips only ~1.6%. Fixed Mamba-style
   multiplicative gates do not repair calibration (0/10 in the tested
   γ ∈ {1, 5} settings), so input selectivity must be learned.
2. **The state's role is statistical metadata, not the readout.** A
   fast-channel state EMA supplies the domain statistic the gate reads
   (it fires on only 2.80/5 known switches at τ_m = 200 and none at
   τ_m ≥ 1000, so it marks where routing helps rather than being causal),
   and routing over domain
   specialists retains them: forgetting improves at τ_m ≤ 1000 under both
   the reported and the unclipped metric. Pause-learning (A2) does **not**
   pass the retention rubric once the clip floor is accounted for — its
   reported stream gain reverses, and forgetting is never better at small
   τ_m.
3. **Specialist retention is robust; stream rankings against a higher-floor
   arm are not.** Soft routing retains forgetting vs. the bare readout;
   abrupt hard switching retains purer specialists than soft on forgetting.
   The reported soft-over-abrupt *stream* margin reverses under the
   unclipped metric and is not claimed. A state-norm homeostat bounds the
   state and restores the full-state EMA as a valid domain statistic
   **on the two-domain task only**: on the four-domain benchmark it
   degrades both metrics on all 10 seeds and is excluded from the full
   stack.

On a four-domain irregular-switch benchmark the full stack beats the bare
host robustly on forgetting (−4.47 reported, −4.53 unclipped, 10/10); the
reported stream margin (−2.25) is floor-sensitive and reverses under the
unclipped metric. Floor sensitivity is scoped to **this squared-loss RLS
readout family**; a companion study localizes the artifact in the training
objective (Paper F). The full stack also beats the untuned Transformer+LoRA
reference on both reported axes (−9.28, −10.08, 10/10; unclipped
undefined for the softmax reference). A P4 factor ablation separates M3
assignment from M4 soft mixtures: hard M3 buys forgetting (−5.24, 10/10)
at a stream cost (+2.90, 0/10); soft M4 recovers stream (−5.14 vs hard)
while giving up a small forgetting margin (+0.77); uniform specialists
alone barely move either axis. A classical ESN+RLS reservoir fails
out of sample under the same protocol (stream ~98 vs bare 15.4 /
one-hot control 12.4); the bare-vs-lin gap is the clip floor
(unclipped bare ~11.5), not a reservoir advantage. All
results are CPU-scale proofs of concept; no scaling claims are made.
Metric-sensitivity table: `PAPER_D.tex` §Metric sensitivity
(`tab:sens`).

## Contents

| File | Purpose |
|---|---|
| `PAPER_D.tex` / `PAPER_D.pdf` | **Current submission version** — Neurocomputing, `elsarticle` |
| `COVER_LETTER.docx` | Cover letter for the NC version (single source of truth) |
| `Highlights.docx` | Elsevier Highlights (bullets only; single source of truth) |
| `Supplementary_Material_PaperD.zip` | Supplementary package |
| `README.md` | This file |
| (historical sources, AI-line files, change log, sketches and build logs removed 2026-09-17; git history retains the last committed state) |

## Compile

From this directory (figures resolve via `../figures/`):

```powershell
pdflatex PAPER_D.tex   # current version; run twice for cross-references
```

## Figures and data anchors

| Item | File | Data |
|---|---|---|
| Fig. 1 P1/P3a readout arms | `../figures/paperD_fig1_p1_arms.pdf` | `../data/s19_ssm_rls_readout_v1.*` |
| Fig. 2 routing retention | `../figures/paperD_fig2_routing.pdf` | `../data/s20_ssm_m3_routing_v1.*`, `../data/s21_ssm_m4_m5_v1.*` |
| Fig. 3 P4 benchmark | `../figures/paperD_fig3_benchmark.pdf` | `../data/s22_ssm_p4_benchmark_v1.*`, `../data/s23_ssm_p4_realtext_v1.*` |
| fair Transformer references | — | `../data/s26_ssm_p4_fair_tf_v1.*` |
| char-bigram oracle | — | `../data/s31_char_bigram_oracle_v1.*` |
| M5 in P4 (honest negative) | — | `../data/s33_ssm_p4_m5_v1.*` |
| readout boundary probes | — | `../data/s35_readout_boundary_probe_v1.*` |
| dormant-covariance-refresh isolation | — | `../data/s37_dormant_p_probe_v1.json` |
| P4 M3/M4 factor ablation | — | `../data/s38_ssm_p4_m3m4_ablation_v1.*` |
| classical ESN+RLS P4 baseline | — | `../data/s40_esn_rls_p4_baseline_v1.*` |
| real-text corpus | — | `../data/corpora/` (Gutenberg #11 Alice, #98 Dickens; public domain) |

Two claims in the manuscript are carried by measured columns rather than by
prose: the M3 detector's per-switch counts (`switches_detected` / `n_switches`
in `../data/s20_ssm_m3_routing_v1.csv`, produced by the instrumentation in
`s20_ssm_m3_routing.py`), and the dormant-covariance-refresh isolation arm
(`../data/s37_dormant_p_probe_v1.json`).

Both are regenerated by their own commands — the probe lives behind a flag on
`s21_ssm_m4_m5.py` rather than in a separate script, so it is not part of the
default run:

```bash
python ../scripts/s20_ssm_m3_routing.py --sequential
python ../scripts/s21_ssm_m4_m5.py --frozen-p-probe
```

`--frozen-p-probe` re-derives the two numbers quoted in §4.3 from the committed
data before exiting and prints a drift report; `s21_ssm_m4_m5.py` also carries
those values as `S37_*` / `S20_DETECTED` fixtures so a re-run flags any gap
between the manuscript and the data instead of failing silently.

## Reproduce

CPU-only (torch CPU). **All experiment paths are relative to the repository
root** (preferred), or equivalently paper-relative as below. `python` may be
`.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (POSIX):

```bash
# from repository root
python scripts/s19_ssm_rls_readout.py --sequential
python scripts/s20_ssm_m3_routing.py --sequential
# ...
# from this paper folder (paper-relative equivalent)
python ../scripts/s20_ssm_m3_routing.py --sequential
python ../scripts/s21_ssm_m4_m5.py --frozen-p-probe
```

Host runs also dump `../data/per_token/*.npz` via `../scripts/per_token_io.py`.
Experiment scripts accept `--quick` (reduced smoke run); `--sequential`
(where listed) disables the multiprocessing `Pool`, `--workers N` caps it.
Each run regenerates the committed `../data/` files. The table below is the
experiment index: what each script does, what it writes, and the finding it
establishes (with the paper location).

**Note on the reported metric.** Perplexity here is the clipped-linear
cross-entropy of Section "Metric sensitivity" in the manuscript. Every
comparison marked floor-sensitive there also carries a clip-free
(unclipped) value; entries below state the reported metric unless noted.
The Key results section above is the authoritative summary.

| Script | What it does | Writes | Finding (paper anchor) |
|---|---|---|---|
| `s19_ssm_rls_readout.py --sequential` | P1/P3a: per-token RLS readout on a hand-rolled diagonal SSM — state-mixture readouts vs. the additive input path (7 arms, 10 seeds) | `../data/s19_ssm_rls_readout_v1.{csv,json}` | State-mixture arms 58.5–115.9 ppl (0/10) vs. input-path 11.75 (d −3.26, 10/10); oracle 7.25 — under the reported metric the readout must use the input path; the ranking is a *calibration* property, since the coverage-spectrum arms score lower on unclipped CE while parking 10–12% of tokens on the floor (Fig. 1) |
| `s20_ssm_m3_routing.py --sequential` | P2: M3 fast-channel EMA + drift detection + routing policies (A1/A2/A3 × τ_m, 90 runs); also reports per-switch M3 detections (`switches_detected`) | `../data/s20_ssm_m3_routing_v1.{csv,json}` | A3 forgetting −2.05/−1.87/−1.20 at τ_m ≤ 1000 (10/10; −2.98/−2.68/−2.16 unclipped) — the gain is strongest where the M3 detector fires (2.80/5 switches at τ_m=200, none at τ_m ≥ 1000) but survives where it never fires, so the detector is not causal (Fig. 2); the A2 pause arm's reported stream gain (−1.52…−2.45) reverses unclipped and it fails the retention rubric |
| `s21_ssm_m4_m5.py --sequential` | P3: M4 soft vs. abrupt routing + M5 state-norm homeostat (E1/E2, 70 runs) | `../data/s21_ssm_m4_m5_v1.{csv,json}` | Soft routing 8.22 vs. abrupt 10.03 reported (−1.81 ppl, 10/10) but +0.76 (0/10) unclipped, while abrupt retains purer specialists on forgetting (+0.38, 1/10) — the soft-over-abrupt *stream* margin is floor-driven and is **not** claimed; M5 restores the EMA detector 5/5, state norm 11.3 vs. 50.2 |
| `s21_ssm_m4_m5.py --frozen-p-probe` | P3 follow-up: dormant-covariance-refresh isolation (weights frozen, covariances refreshed, 10 runs) | `../data/s37_dormant_p_probe_v1.json` | With the readout weights frozen and only the inverse covariances refreshed, stream and forgetting perplexity are both exactly 32.0 (uniform chance) on all 10 seeds — refresh alone learns nothing, so the routing gain belongs to the training, not the refresh |
| `s22_ssm_p4_benchmark.py --sequential` | P4: four-domain irregular-switch benchmark — bare SSM vs. REDEM-SSM vs. TF+LoRA (3 arms, 10 seeds) | `../data/s22_ssm_p4_benchmark_v1.{csv,json}` | REDEM-SSM vs. bare: forgetting −4.47 reported (−4.53 unclipped, 10/10 both) — the primary claim; the stream margin (−2.25) is floor-driven and reverses unclipped (+1.45, 0/10). vs. TF+LoRA −9.28 / −10.08 reported (10/10), but the unclipped metric is undefined for the TF arm (no clip floor) so that axis has no clip-free counterpart (Fig. 3, Table 2) |
| `s23_ssm_p4_realtext.py --sequential` | Real-text transfer: Alice vs. Dickens, 32-symbol char vocab (3 arms, 10 seeds) | `../data/s23_ssm_p4_realtext_v1.{csv,json}` | vs. bare: forgetting −0.46 reported / −0.33 unclipped (9/10 both); the stream margin (−1.26, 10/10) nearly vanishes unclipped (−0.05, 7/10). vs. TF −5.23 reported (10/10) — the forgetting result generalizes to real text |
| `s26_ssm_p4_fair_tf.py` | Fair Transformer references: tuned A1 grid (4 lr × 2 ranks) + 4-adapter A3 routing (9 arms, 10 seeds) | `../data/s26_ssm_p4_fair_tf_v1.{csv,json}` | Tuning cuts the stream gap to −1.68 (0/10) but collapses retention to 62.2; TF-A3 forgetting −8.17 (10/10) without fixing stream — the mechanisms transfer across the two hosts tested (with unequally tuned references), and the stream performance is not host-agnostic. The TF-A1 baseline for these deltas is read from `s22` (this script has no TF-A1 arm) |
| `s31_char_bigram_oracle.py` | Char-bigram oracle on the real-text protocol (full-book vs. ref-window fits, 10 seeds) | `../data/s31_char_bigram_oracle_v1.{csv,json}` | First-order ceiling 10.97 ± 0.18 ppl; REDEM-SSM 12.07 within ~1.1 ppl — the "first-order regime" boundary is quantitative |
| `s33_ssm_p4_m5.py` | M5 state-norm homeostat added to the P4 stack (2 arms, 10 seeds) | `../data/s33_ssm_p4_m5_v1.{csv,json}` | M5 is significantly worse in P4: stream +1.53 / forgetting +2.90 (10/10) — honest negative; Δt modulation breaks the Δt=1 whitening, validating S22's exclusion of M5 |
| `s35_readout_boundary_probe.py --workers 4` | P1 boundary probes on the s19 host (10 seeds): full/half-window oracle, skip-vs-proj nested check, token decoding, fast-channel next-token readouts | `../data/s35_readout_boundary_probe_v1.{csv,json}` | Full-window oracle 31.2 vs. half-window 17.3 (window-dependent); skip 18.0 > proj 7.25 (nested violation, 10/10); current token decodable from fast channels at 88.9–99.7%, yet fast-channel-only next-token readouts fail (68–104 vs. 13.9–17.4) — the P1 failure is a pooled-readout/metric property, not missing linear information (§3) |
| `s38_ssm_p4_m3m4_ablation.py` | P4 factor ablation: bare / uniform (M4 without M3) / hard (M3 without soft M4) / soft full stack (4 arms, 10 seeds) | `../data/s38_ssm_p4_m3m4_ablation_v1.{csv,json}` | Uniform: +0.46/−0.43 (2/10, 9/10). Hard: +2.90/−5.24 (0/10, 10/10). Soft vs hard: −5.14/+0.77 (10/10, 2/10). M3 assignment carries forgetting; soft M4 recovers stream (§P4 factor ablation) |
| `s40_esn_rls_p4_baseline.py` | Classical ESN + online RLS on the P4 protocol (4 arms incl. one-hot control, 10 seeds) | `../data/s40_esn_rls_p4_baseline_v1.{csv,json}` | ESN-128 stream 98.8 vs bare 15.4 / lin 12.4 (0/10 vs both); ρ∈{0.9,1.0} and N=256 do not close the gap — classical reservoir state-mixture readout fails like P1. Bare-vs-lin is a clip-floor effect (unclipped bare ~11.5); SSM-bare forgetting is 13.40 (matches `tab:p4` / s38 mean), not ESN-lin's own 13.41 (§Classical ESN baseline) |
| `gen_paperD_fig1_p1_arms.py` | FIG: P1/P3a state-readout falsification + input-path control | `../figures/paperD_fig1_p1_arms.pdf` | Fig. 1 |
| `gen_paperD_fig2_routing.py` | FIG: P2 routing retention + P3 soft vs. abrupt | `../figures/paperD_fig2_routing.pdf` | Fig. 2 |
| `gen_paperD_fig3_benchmark.py` | FIG: P4 benchmark bars | `../figures/paperD_fig3_benchmark.pdf` | Fig. 3 |

Seed discipline and the shared S1–s36 A–D pipeline:
[`../README_REDEM.md`](../README_REDEM.md).
Paper D follow-ups beyond that registry (documented here):
`s21 --frozen-p-probe` → `s37_dormant_p_probe_v1.json`,
`s38_ssm_p4_m3m4_ablation.py`, `s40_esn_rls_p4_baseline.py`.

## Companion papers

- Paper A (physics): [`../paper_a/`](../paper_a/) — substrate characterization
- Paper B (algorithm): [`../paper_b/`](../paper_b/) — REDEM online learning
- Paper C (dissection): [`../paper_c/`](../paper_c/) — mechanism disentanglement
- Paper F (calibrated readout): [`../paper_f/`](../paper_f/) — objective-level calibration and sparse selectivity

## Data provenance

<!-- BEGIN PROVENANCE (generated by scripts/provenance_map.py) -->

Claims registered for Paper D (includes shared anchors where relevant). Every registered headline quantity maps to a committed artefact expression. Generated by `scripts/provenance_map.py` from `scripts/claims_registry.json` — do not hand-edit it.

- claim call sites: **16**

| paper | section | manuscript quantity | where | produced by |
|---|---|---|---|---|
| D | P1 | B-proj / B-lin-clip stream ppl (input path) | abstract / Paper D/F | `11.7541` |
| D | P1 | oracle / pooled-table ceiling stream ppl | Paper D | `7.24781` |
| D | P2 | A3-A1 forgetting diff at tau_m=200 | Paper D | `-2.05268` |
| D | P2 | A3-A1 forgetting diff at tau_m=500 | Paper D | `-1.86947` |
| D | P2 | A3-A1 forgetting diff at tau_m=1000 | Paper D | `-1.20153` |
| D | P3 | A3-soft stream ppl (E1) | Paper D | `8.21824` |
| D | P3 | A3-abrupt stream ppl (E1) | Paper D | `10.0328` |
| D | P3 | M5 regulated whitened-state norm mean | Paper D | `11.3137` |
| D | P3 | bare host whitened-state norm mean | Paper D | `50.1882` |
| D | P4 | SSM-REDEM stream ppl | Paper D tables | `13.1796` |
| D | P4 | SSM-REDEM forgetting ppl | Paper D tables | `8.93112` |
| D | P4-text | REDEM-SSM real-text stream ppl | Paper D | `12.071` |
| D | P4-text | char-bigram full-book ceiling ppl | Paper D | `10.9737` |
| D | P4 | SSM-REDEM vs bare stream diff (ppl, negative=better) | Paper D tables | `-2.24672` |
| D | P4 | SSM-REDEM vs bare forgetting diff | Paper D tables | `-4.47347` |
| D | P4 | SSM-REDEM vs TF-A1 stream diff | Paper D tables | `-9.27882` |

Register new numbers in `scripts/verify_claims.py`, re-run `run_all_audits.py`, then let this script rewrite the block.
<!-- END PROVENANCE -->
