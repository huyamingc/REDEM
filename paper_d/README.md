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
Each run regenerates the committed `../data/` files; the key values below
are what the committed data and the paper report.

| Script | Writes | Expected key values (paper anchor) |
|---|---|---|
| `s19_ssm_rls_readout.py --sequential` | `../data/s19_ssm_rls_readout_v1.{csv,json}` | state-mixture arms 58.5–115.9 ppl (0/10) vs input-path 11.75 (d −3.26, 10/10); oracle 7.25 (Fig. 1) |
| `s20_ssm_m3_routing.py --sequential` | `../data/s20_ssm_m3_routing_v1.{csv,json}` | A2 stream −1.52…−2.45 (10/10); A3 forgetting −2.05/−1.87/−1.20 at τ_m ≤ 1000 (10/10) (Fig. 2) |
| `s21_ssm_m4_m5.py --sequential` | `../data/s21_ssm_m4_m5_v1.{csv,json}` | soft routing 8.22 vs abrupt 10.03 (−1.81 ppl, 10/10); M5 restores the EMA detector 5/5, state norm 11.3 vs 50.2 |
| `s22_ssm_p4_benchmark.py --sequential` | `../data/s22_ssm_p4_benchmark_v1.{csv,json}` | REDEM-SSM vs bare host −2.25 stream / −4.47 forgetting; vs TF+LoRA −9.28 / −10.08 (10/10) (Fig. 3, Table 2) |
| `s23_ssm_p4_realtext.py --sequential` | `../data/s23_ssm_p4_realtext_v1.{csv,json}` | vs bare −1.27 (10/10); vs TF −5.23 (10/10) on the two-book real-text protocol |
| `s26_ssm_p4_fair_tf.py` | `../data/s26_ssm_p4_fair_tf_v1.{csv,json}` | tuned TF: stream gap −1.68 (0/10) with retention collapse to 62.2; TF-A3 forgetting −8.17 (10/10) without fixing stream |
| `s31_char_bigram_oracle.py` | `../data/s31_char_bigram_oracle_v1.{csv,json}` | first-order ceiling 10.97 ± 0.18 ppl; REDEM-SSM 12.07 (within ~1.1 ppl) |
| `s33_ssm_p4_m5.py` | `../data/s33_ssm_p4_m5_v1.{csv,json}` | M5 in the P4 stack is worse: stream +1.53 / forgetting +2.90 (10/10) — honest negative |
| `s35_readout_boundary_probe.py --workers 4` | `../data/s35_readout_boundary_probe_v1.{csv,json}` | full-window oracle 31.2 vs half-window 17.3 (window-dependent); skip 18.0 > proj 7.25 (10/10); current-token decode 88.9–99.7% from fast channels; fast-channel next-token readouts 68–104 vs static-table 13.9–17.4 |
| `gen_paperD_fig1_p1_arms.py` | `../figures/paperD_fig1_p1_arms.pdf` | Fig. 1 |
| `gen_paperD_fig2_routing.py` | `../figures/paperD_fig2_routing.pdf` | Fig. 2 |
| `gen_paperD_fig3_benchmark.py` | `../figures/paperD_fig3_benchmark.pdf` | Fig. 3 |

Seed discipline and the full S1–s35 pipeline:
[`../README_REDEM.md`](../README_REDEM.md).

## Companion papers

- Paper A (physics): [`../paper_a/`](../paper_a/) — substrate characterization
- Paper B (algorithm): [`../paper_b/`](../paper_b/) — REDEM online learning
- Paper C (dissection): [`../paper_c/`](../paper_c/) — mechanism disentanglement
