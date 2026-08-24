# Paper B — REDEM: Training-Inference Unified Learning with Meta-Adaptation and Structural Plasticity

Online learning architecture paper (single author). Target: *Neural Networks*
(Elsevier). Companion substrate-theory paper: [`../paper_a/`](../paper_a/)
(target: IJBC / Chaos).

## Contents

| File | Purpose |
|---|---|
| `PAPER_B.tex` | Submission-ready LaTeX (compiles standalone with `article`) |
| `PAPER_B.pdf` | Compiled PDF (11 pages, MiKTeX / pdflatex ×2, zero warnings) |
| `PAPER_B_draft.md` | Markdown draft (source prose; revision history in `../NEW_ALGORITHM_PLAN.md`) |
| `PAPER_B_sketch.md` | Outline, figure/table inventory, key numbers |
| `README.md` | This file |

## Compile

From this directory (figures resolve via `../figures/`):

```powershell
pdflatex PAPER_B.tex    # run twice for cross-references
# or, if Perl is installed:
latexmk -pdf PAPER_B.tex
```

## Submission checklist (Paper B)

- [x] Author filled (2026-02-19): Yaming Hu, ORCID 0009-0003-1406-0485,
      Independent Researcher, Guiyang, Guizhou Province, China;
      64687555@qq.com. Cover letter: `COVER_LETTER.md`.
- [ ] Swap the document class: `elsarticle.cls` (Elsevier template).
- [ ] Title deliberately contains no "physics" (S1 review decision); the
      physical substrate appears only as background motivation in §1.
- [ ] Abstract leads with the differentiated mechanism set (S2); the honest
      ESN parity comparison sits in the final paragraph.
- [ ] Negative results compressed to Table 1 + three sentences (S3); full
      data in `../data/s3_three_factor_v1.*` and
      `../data/s4_intrinsic_reward_v1.*`.
- [ ] Fig. 1 shows the M4 ↔ M5 coupling loop (T2): κ gates the rewiring rate,
      rewiring feeds the Lyapunov estimate.
- [ ] Metadata-transfer conclusion honest (T1): the slow-state mechanism is
      substrate-agnostic and transfers to a matched ESN; REDEM's
      differentiation is the mechanism set around it (self-regulation,
      disturbance robustness, local sparse coupling).
- [x] §2 cross-references the companion Paper A for the substrate model (T3);
      `\cite{companionA}` added at the abstract substrate sentence and §2
      (2026-08-24 review audit — the bibitem was previously orphaned).
- [ ] Figures 1–6 (vector `.pdf` in `../figures/`):
      `../figures/paperB_fig1_redem.pdf`,
      `../figures/s2_online_readout_v1.pdf`,
      `../figures/paperB_fig3_metadata.pdf`,
      `../figures/paperA_fig4_robustness.pdf` (shared with Paper A),
      `../figures/paperB_fig5_ablation.pdf`,
      `../figures/paperB_fig6_showdown.pdf`.
- [ ] Tables: Table 1 (reward-only learners fail at the inversion; RLS
      selected), Table 2 (standard-task benchmarks vs ESN / GRU / transformer).
- [ ] Code availability statement:
      <https://github.com/huyamingc/REDEM> (private during review; public on
      acceptance).
- [x] Follow-up evidence integrated (2026-08-24): §4.3 causal audit re-run on
      the S11 disturbance chain (s28 — no leak arm improves recovery; the
      no-plasticity arm reproduces the S11 homeostat anchor 8.47 exactly;
      leak-sensitivity scan s34 scopes the claim: 10× FTLE leaks NS, 30%
      plasticity correlation significant +0.57 — audit resolution bounded);
      N=1024 at 10 seeds (s30 — full 0.9970 vs baseline 0.9753, paired
      t=15.3); §4.3 novelty-gated rewiring (s25, 14.59 vs 12.43) and the
      M4-M5 coupling negative (s24).
- [ ] Cover letter: the online / local / no-BPTT corner; self-regulation under
      disturbance; standard-task parity with a well-tuned ESN reported
      honestly.
