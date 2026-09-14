# Paper F — Learning the Readout Form (calibration + sparse selectivity)

Companion to Paper D: Paper D diagnosed that the squared-loss / clip-floor
metric is load-bearing; Paper F *changes the instrument* (trained softmax
readout) and then asks which selectivity pressure the clean metric still needs.

- Manuscript: [`PAPER_F.tex`](PAPER_F.tex) / [`PAPER_F.pdf`](PAPER_F.pdf)
  (elsarticle draft, **13 pp**; Figures 1–2 embedded)
- Full title: *Learning the Readout Form: Objective-Level Calibration, Sparse
  Selectivity, and Expert Routing on a Diagonal State-Space Host*
- Target: Neurocomputing (or TMLR)
- Protocol: same diagonal SSM host and paired stream as Paper D (s18/s19
  family), 10 seeds
- Primary data: `../data/s50_*`, `s51_*`, `s52_*`; scaling/transfer
  `s53_*`, `s53b_*`, `s54_*`; external baseline
  `../data/s66_external_ssm_baseline_v1.*`
- Claim IDs in scripts/JSON: **C1/C2/C3** (s50), **C4/C4b** (s51),
  **C5** (s52); s53/s53b/s54 are supporting scaling/transfer labels

## Key results (10-seed paired unless noted)

1. **Objective, not range (C1/C2).** Replacing clip-floor CE with a trained
   softmax readout removes the floor and lowers stream CE by 0.307 nats/token
   (ppl 11.75 → 8.65, 10/10); a post-hoc clip-negatives + renormalize step
   (Π_Δ) does **not** repair the metric. Full-state under the clean metric
   does not revive Skip (C3).
2. **Sparsity is the missing gate pressure (C4b).** Unconstrained CE gating
   does not sparsify and hurts retention; hard top-k (k=32) is the best
   stream arm **among the variants that keep retention near baseline**
   (7.03). A weakly L1-regularised gate edges it on stream alone (6.994,
   6% of gates open) but pays 2.37 ppl more retention (11.574 vs 9.201).
3. **Expert routing (C5).** Over two online softmax experts it cuts
   forgetting further (9.20 → 7.79, 10/10) at stream parity; hard-routing
   ablation is slightly better on retention (6.94) at a small stream tax.
4. **External selective-SSM baseline (s66, completed 2026-09-12).** A
   Mamba-style arm on the *same* host, stream, seeds and metric: the
   **frozen-host** external arm's 24,864 host parameters are held at a
   per-seed **random** init, so it *trains* only the shared readout
   (8,224) — i.e. F's Gate-C-topk arm trains **3.0× more** parameters
   (24,736), not fewer. In *total* parameters the external arm is the
   larger (33,088 vs 24,736), so the two budgets are not comparable in
   either direction and the paper does not claim a capacity-matched win.
   Frozen-host external stream 9.03 vs F's Gate-C-topk 7.03 (**0/10**
   external better). Jointly trained host adaptation is destructive
   (stream 10.19 at the matched learning rate; a 4× larger host learning
   rate makes joint training diverge — the committed s66 rows pool both
   rates under one arm name).
   Frozen is the stronger *external* arm but still loses to F's
   B-softmax-sgd anchor (stream 9.03 vs 8.65; forget 10.38 vs 8.89).
   TTT fast-weight arm removed as non-viable at this scale
   (stream ppl worse than uniform 32). Against F's ungated
   B-softmax-sgd anchor (4,128 trained) the external arm trains 2.0× more
   and still loses on both axes. Pairing of the re-implemented F arms is
   bit-exact, 40/40 (4 arms × 10 seeds) against the committed s50/s51
   anchors at worst relative difference 0.0 — re-verified 2026-09-14 by
   re-running s66's own `validate()` against the committed rows
   (`review_workspace/def_audit/s66_revalidate.py`); the `pairs: 0`
   previously stored in `s66_external_ssm_baseline_v1.json`
   `anchor_validation` was a stale field and has been corrected to 40.
5. **Scaling / transfer (s53, s53b, s54).** C1 improves with N
   (B-softmax stream 8.63 → 7.25 at N=128→512). Gate-C-topk fails at
   N=512 when k is kept at N/4; a fixed small k (16–32) restores and
   beats the baseline (k-sweep, 3 seeds). Sparse-gate advantage transfers
   to Mackey–Glass (bin-32, 5 seeds): Gate-C-topk holdout 16.27 vs
   B-softmax 29.07.
6. **Host freeze policy (s67, completed 2026-09-13).** Same protocol,
   10 seeds; anchors bit-exact to s66/s50/s51 (80/80). Random-frozen
   selective host 9.03/10.38; **any** host CE (1-segment pretrained-
   then-freeze 9.58/84.71, or online 10.19/115.24) is retention-
   destructive (Q1 0/10 vs random-frozen; Q2 10/10 better than online).
   F top-k on frozen selective host recovers stream (7.38/7.18) but
   does not beat diagonal Gate-C-topk (7.03). Paper H must default to
   a frozen host.
7. **Form-learning program role (`F_FORM_LEARNING_ROLE.md`).** Formal
   design rule (learnable instrument L1–L3), self+causality seeds
   (C1 verify / C4b select / C5 organize), counterfactual and
   world-model readings of the top-k gate, width/k capability
   boundary, and the path to Paper H (form generation) gated on s67.

## Contents

| File | Purpose |
|---|---|
| `PAPER_F.tex` / `PAPER_F.pdf` | Current manuscript (13 pp, Figures 1–2 embedded) |
| `PAPER_F_PLAN.md` / `PAPER_F_sketch.md` | Planning notes (not submission) |
| `PARADIGM_AND_RULES.md` | Scope rules (mechanism study, no SoTA claim) |
| `F_FORM_LEARNING_ROLE.md` | Program-level form-learning derivation (design rule → Paper H) |
| `COVER_LETTER.txt` / `COVER_LETTER.docx` | Submission cover letter (docx regenerated 2026-09-13; no “capacity-matched”) |
| `Highlights.txt` / `Highlights.docx` | Elsevier Highlights — bullets only, 5 bullets, 62/62/61/55/65 chars (≤85). Rationale in `Highlights_notes.md` |
| `../review_workspace/modification_log_paper_f.md` | Full experiment / manuscript change log |
| `../review_workspace/modification_log_EF_clean_p0.md` | 2026-09-13 clean-P0 + r2/r3 consistency rounds |
| `../review_workspace/modification_log_EF_meta_20260913.md` | 2026-09-13 P1 fixes (bib, abstract 266 words with LaTeX and math stripped, Π_Δ wording, …) |
| `../review_workspace/S66_EXTERNAL_BASELINE_FINDINGS.md` | External-baseline audit |
| `../figures/paperF_fig1_arms.{pdf,png}` | Figure 1: headline arms C1–C5 (stream/forget, 10-seed mean ± std) |
| `../figures/paperF_fig2_floor.{pdf,png}` | Figure 2: clip-floor artifact (C1/C2) |
| `../figures/paperF_graphical_abstract.{png,pdf}` | Graphical abstract for submission (300 dpi PNG) |

## Reproduce

```powershell
# from repo root
& .\.venv\Scripts\python.exe scripts\s50_paper_f_pilot_nl_readout.py --sequential
& .\.venv\Scripts\python.exe scripts\s51_paper_f_learned_gate.py --sequential
& .\.venv\Scripts\python.exe scripts\s52_paper_f_soft_route_experts.py --sequential
& .\.venv\Scripts\python.exe scripts\s53_paper_f_scaling.py --sequential
& .\.venv\Scripts\python.exe scripts\s53b_paper_f_ksweep.py --sequential
& .\.venv\Scripts\python.exe scripts\s54_paper_f_mackey_glass.py --sequential
& .\.venv\Scripts\python.exe scripts\s66_external_ssm_baseline.py
& .\.venv\Scripts\python.exe scripts\s67_host_freeze.py
# optional read-only report
& .\.venv\Scripts\python.exe scripts\s66_report.py
# figures (reads committed s50/s51/s52 CSVs only)
& .\.venv\Scripts\python.exe scripts\gen_paperF_figs.py
```

Regenerate submission docs after editing the txt sources:

```powershell
& $env:MIMO_PYTHON scripts\gen_cover_letter_docx.py paper_f COVER_LETTER
& $env:MIMO_PYTHON scripts\gen_cover_letter_docx.py paper_f Highlights
```

## Status

- [x] Primary experiments s50–s52
- [x] Scaling / k-sweep / MG transfer s53, s53b, s54
- [x] External selective-SSM baseline s66 (Limitations + Discussion updated)
- [x] Host freeze follow-up s67 (random/pretrained/online; bit-exact anchors)
- [x] Cover letter draft (`COVER_LETTER.txt` + regenerated `.docx`)
- [x] Highlights draft (`Highlights.txt` bullets only + regenerated
      `.docx`; ≤85 chars each)
- [x] 2026-09-13 clean-P0 / meta / r2–r6 (C-ID alignment, capacity-budget
      accounting vs Gate-C-topk, claim JSON patch, abstract tightened,
      TTT bibitem → ICML
      2025 + arXiv:2407.04620, duplicated-sentence & overfull fixes)
- [x] 2026-09-14 compliance pass: declarations moved before the reference
      list, `Highlights.txt` reduced to bullets only (+ `.docx`
      regenerated), Paper E given a `\bibitem`, D given its Zenodo DOI,
      Lukoševičius name/venue corrected, bibliography re-ordered into
      first-citation order, capacity sentence corrected, s66 anchor field
      refreshed
- [x] 2026-09-13 r6 figures: `gen_paperF_figs.py` → Figs 1–2 embedded
      (13 pp) + graphical abstract; s66 lr-pooling disclosure in
      Limitations (10.19 = matched-lr value; 4× lr diverges)
- [x] Form-learning derivation note (`F_FORM_LEARNING_ROLE.md`)
- [x] Git track + commit of F manuscript, data, scripts, figures
- [ ] Supplementary zip if the venue requires it
- [ ] Author pass on Highlights wording (txt is the source of truth)
