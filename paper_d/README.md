# Paper D — REDEM-SSM: A State-Space Architecture with Native Online Learning, Meta-Adaptation, and Structural Plasticity

Paper D of the four-paper REDEM series (**role: architecture**). It
instantiates the REDEM mechanisms — an online readout (M1), a slow-trace
statistical memory (M3), structural plasticity (M4), and a stability
homeostat (M5) — natively on a state-space host (a diagonal linear
recurrence with a log-uniform timescale spectrum), instead of retrofitting
them onto a frozen-feedforward Transformer.

- PDF: [`PAPER_D.pdf`](PAPER_D.pdf) | LaTeX: [`PAPER_D.tex`](PAPER_D.tex)
- Preprint: [doi:10.5281/zenodo.22110624](https://doi.org/10.5281/zenodo.22110624)
- Series overview and reading order (A → B → C → D):
  [`../README.md`](../README.md)

## Key results

A falsifying development sequence (10-seed paired discipline throughout)
isolates three host requirements:

1. **The readout must use the additive input path.** A linear readout on a
   linearly-decayed state mixture cannot recover the current token exactly
   (a deconvolution impossibility), and the pooled state readout fails out
   of sample (0/10 seeds; stream perplexity 58–116 vs. 15.0 for a
   Transformer+LoRA reference); the input-path readout beats the reference
   10/10. Fixed Mamba-style multiplicative gates do not repair the state
   readout (0/10 at either tested sharpness), so input selectivity must be
   learned.
2. **The state's role is statistical metadata, not the readout.** A
   fast-channel state EMA tracks domain switches and soft routing retains
   per-domain specialists (forgetting −2.05 to −1.20 ppl, 10/10 seeds at
   τ_m ≤ 1000); the pause-learning (gating-only) policy improves stream
   perplexity on the RLS readout (10/10) — the policy lesson is
   readout-dynamics-dependent.
3. **Gentle beats abrupt.** Soft routing beats abrupt switching (−1.81 ppl,
   10/10) and a state-norm homeostat bounds the state and restores the
   full-state EMA as a valid domain statistic (5/5 switch detections).

On a four-domain irregular-switch benchmark the full stack beats the bare
host (−2.25 stream, −4.47 forgetting, 10/10) and the Transformer+LoRA
reference (−9.28, −10.08, 10/10). All results are CPU-scale proofs of
concept; no scaling claims are made.

## Contents

| File | Purpose |
|---|---|
| `PAPER_D.tex` | LaTeX source (elsarticle preprint class) |
| `PAPER_D.pdf` | Compiled PDF (14 pages) |

## Compile

From this directory (figures resolve via `../figures/`):

```powershell
pdflatex PAPER_D.tex    # run three times for cross-references (elsarticle)
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
| real-text corpus | — | `../data/corpora/` (Gutenberg #11 Alice, #98 Dickens; public domain) |

## Reproduce

CPU-only (torch CPU); uses the project virtual environment (`../.venv`).
Run from this directory:

```powershell
& ..\.venv\Scripts\python.exe ..\scripts\<script> [flags]
```

Experiment scripts accept `--quick` (reduced smoke run); `--sequential`
(where listed) disables the multiprocessing `Pool`, `--workers N` caps it.
Each run regenerates the committed `../data/` files. The table below is the
experiment index: what each script does, what it writes, and the finding it
establishes (with the paper location).

| Script | What it does | Writes | Finding (paper anchor) |
|---|---|---|---|
| `s19_ssm_rls_readout.py --sequential` | P1/P3a: per-token RLS readout on a hand-rolled diagonal SSM — state-mixture readouts vs. the additive input path (7 arms, 10 seeds) | `../data/s19_ssm_rls_readout_v1.{csv,json}` | State-mixture arms 58.5–115.9 ppl (0/10) vs. input-path 11.75 (d −3.26, 10/10); oracle 7.25 — the readout must use the input path; a squared-loss state readout cannot deconvolve the decay (Fig. 1) |
| `s20_ssm_m3_routing.py --sequential` | P2: M3 fast-channel EMA + drift detection + routing policies (A1/A2/A3 × τ_m, 90 runs) | `../data/s20_ssm_m3_routing_v1.{csv,json}` | A2 stream −1.52…−2.45 (10/10); A3 forgetting −2.05/−1.87/−1.20 at τ_m ≤ 1000 (10/10) — the state's role is statistical metadata for routing, not the readout (Fig. 2) |
| `s21_ssm_m4_m5.py --sequential` | P3: M4 soft vs. abrupt routing + M5 state-norm homeostat (E1/E2, 70 runs) | `../data/s21_ssm_m4_m5_v1.{csv,json}` | Soft routing 8.22 vs. abrupt 10.03 (−1.81 ppl, 10/10); M5 restores the EMA detector 5/5, state norm 11.3 vs. 50.2 — gentle beats abrupt; the homeostat makes the full-state EMA a valid statistic |
| `s22_ssm_p4_benchmark.py --sequential` | P4: four-domain irregular-switch benchmark — bare SSM vs. REDEM-SSM vs. TF+LoRA (3 arms, 10 seeds) | `../data/s22_ssm_p4_benchmark_v1.{csv,json}` | REDEM-SSM vs. bare −2.25 stream / −4.47 forgetting; vs. TF+LoRA −9.28 / −10.08 (10/10) — the full stack wins on both axes (Fig. 3, Table 2) |
| `s23_ssm_p4_realtext.py --sequential` | Real-text transfer: Alice vs. Dickens, 32-symbol char vocab (3 arms, 10 seeds) | `../data/s23_ssm_p4_realtext_v1.{csv,json}` | vs. bare −1.27 (10/10); vs. TF −5.23 (10/10) — the result generalizes from synthetic streams to real text |
| `s26_ssm_p4_fair_tf.py` | Fair Transformer references: tuned A1 grid (4 lr × 2 ranks) + 4-adapter A3 routing (9 arms, 10 seeds) | `../data/s26_ssm_p4_fair_tf_v1.{csv,json}` | Tuning cuts the stream gap to −1.68 (0/10) but collapses retention to 62.2; TF-A3 forgetting −8.17 (10/10) without fixing stream — mechanisms are host-agnostic, stream performance is not |
| `s31_char_bigram_oracle.py` | Char-bigram oracle on the real-text protocol (full-book vs. ref-window fits, 10 seeds) | `../data/s31_char_bigram_oracle_v1.{csv,json}` | First-order ceiling 10.97 ± 0.18 ppl; REDEM-SSM 12.07 within ~1.1 ppl — the "first-order regime" boundary is quantitative |
| `s33_ssm_p4_m5.py` | M5 state-norm homeostat added to the P4 stack (2 arms, 10 seeds) | `../data/s33_ssm_p4_m5_v1.{csv,json}` | M5 is significantly worse in P4: stream +1.53 / forgetting +2.90 (10/10) — honest negative; Δt modulation breaks the Δt=1 whitening, validating S22's exclusion of M5 |
| `s35_readout_boundary_probe.py --workers 4` | P1 boundary probes on the s19 host (10 seeds): full/half-window oracle, skip-vs-proj nested check, token decoding, fast-channel next-token readouts | `../data/s35_readout_boundary_probe_v1.{csv,json}` | Full-window oracle 31.2 vs. half-window 17.3 (window-dependent); skip 18.0 > proj 7.25 (nested violation, 10/10); current token decodable from fast channels at 88.9–99.7%, yet fast-channel-only next-token readouts fail (68–104 vs. 13.9–17.4) — the P1 failure is a pooled-readout/metric property, not missing linear information (§3) |
| `gen_paperD_fig1_p1_arms.py` | FIG: P1/P3a state-readout falsification + input-path control | `../figures/paperD_fig1_p1_arms.pdf` | Fig. 1 |
| `gen_paperD_fig2_routing.py` | FIG: P2 routing retention + P3 soft vs. abrupt | `../figures/paperD_fig2_routing.pdf` | Fig. 2 |
| `gen_paperD_fig3_benchmark.py` | FIG: P4 benchmark bars | `../figures/paperD_fig3_benchmark.pdf` | Fig. 3 |

Seed discipline and the full S1–s35 pipeline:
[`../README_REDEM.md`](../README_REDEM.md).

## Companion papers

- Paper A (physics): [`../paper_a/`](../paper_a/) — substrate characterization
- Paper B (algorithm): [`../paper_b/`](../paper_b/) — REDEM online learning
- Paper C (dissection): [`../paper_c/`](../paper_c/) — mechanism disentanglement
