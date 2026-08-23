# Paper B — REDEM: Training-Inference Unified Learning with Meta-Adaptation and Structural Plasticity

Online learning architecture paper (single author). Target: *Neural Networks*
(Elsevier). Companion substrate-theory paper: [`../paper_a/`](../paper_a/)
(target: IJBC / Chaos).

## Contents

| File | Purpose |
|---|---|
| `PAPER_B.tex` | Submission-ready LaTeX (compiles standalone with `article`) |
| `PAPER_B.pdf` | Compiled PDF (10 pages, MiKTeX / pdflatex ×2, zero warnings) |
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

- [ ] Replace `[Author Name]` / affiliation / email in `\author`.
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
- [ ] §2 cross-references the companion Paper A for the substrate model (T3).
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
- [ ] Cover letter: the online / local / no-BPTT corner; self-regulation under
      disturbance; standard-task parity with a well-tuned ESN reported
      honestly.
