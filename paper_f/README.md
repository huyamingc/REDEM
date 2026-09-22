# Paper F — Learning the Readout Form (calibration + sparse selectivity)

- Preprint (Zenodo concept DOI, all versions): [doi:10.5281/zenodo.22888049](https://doi.org/10.5281/zenodo.22888049)

Companion to Paper D: Paper D diagnosed that the squared-loss / clip-floor
metric is load-bearing; Paper F *changes the instrument* (trained softmax
readout) and then asks which selectivity pressure the clean metric still needs.

- Manuscript: [`PAPER_F.tex`](PAPER_F.tex) / [`PAPER_F.pdf`](PAPER_F.pdf)
  (elsarticle draft, **15 pp**; Figures 1–2 embedded)
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
7. **Form-learning program role** (internal note removed on 2026-09-17;
   the role summary is retained in this README and in the manuscript's
   Discussion).
   Formal design rule (learnable instrument L1–L3), self+causality seeds
   (C1 verify / C4b select / C5 organize), counterfactual and
   world-model readings of the top-k gate, width/k capability
   boundary, and the path to Paper H (form generation) gated on s67.

## Contents (submission folder only)

Paths below are relative to the **repository root** unless marked `../`
(paper-folder-relative). Cover letters / Highlights `.docx` and supplementary
zips are **local-only** (git-ignored); the public repo keeps tex/pdf/README
plus `scripts/` + `data/` + `figures/`.

| File | Purpose |
|---|---|
| `PAPER_F.tex` / `PAPER_F.pdf` | Current manuscript |
| `COVER_LETTER.docx` | Submission cover letter (local-only; single source of truth) |
| `Highlights.docx` | Elsevier Highlights (local-only; bullets only) |
| `README.md` | This file |
| `../scripts/s66_report.py` | Read-only s66 audit (paired external vs F anchors; lr 1.0 vs 4.0) |
| `../figures/paperF_fig1_arms.{pdf,png}` | Figure 1: headline arms C1–C5 |
| `../figures/paperF_fig2_floor.{pdf,png}` | Figure 2: clip-floor artifact |
| `../figures/paperF_graphical_abstract.{png,pdf}` | Graphical abstract |
| (internal notes removed 2026-09-17; git history retains `PAPER_F_PLAN.md`, `PARADIGM_AND_RULES.md`, `F_FORM_LEARNING_ROLE.md`, `Highlights_notes.md`) |
| `review_workspace/` (root; git-ignored) | Local change logs only — not in the public clone |

## Reproduce

Commands are **repository-relative** (run from the repo root; forward slashes).
`python` may be replaced by `.venv/Scripts/python.exe` (Windows) or
`.venv/bin/python` (POSIX).

```bash
# from repo root
python scripts/s50_paper_f_pilot_nl_readout.py --sequential
python scripts/s51_paper_f_learned_gate.py --sequential
# v2 re-run adds channel-utilization histograms; writes data/s51_paper_f_learned_gate_v2.{csv,json}
# and refreshes data/per_token/s51__*.npz via scripts/per_token_io.py
python scripts/s52_paper_f_soft_route_experts.py --sequential
python scripts/s53_paper_f_scaling.py --sequential
python scripts/s53b_paper_f_ksweep.py --sequential
python scripts/s54_paper_f_mackey_glass.py --sequential
python scripts/s66_external_ssm_baseline.py
python scripts/s67_host_freeze.py
# optional read-only audit of s66 (supports Discussion/Limitations quotes)
python scripts/s66_report.py
# figures (reads committed s50/s51/s52 CSVs only)
python scripts/gen_paperF_figs.py
```

`data/per_token/*.npz` for this paper are produced by the s50–s52 / s66–s67
runs above (helper: `scripts/per_token_io.py`). Committed dumps are the
archives used for same-token paired tests.

Regenerate submission docs after editing the txt sources (local-only docx):

```bash
python scripts/gen_cover_letter_docx.py paper_f COVER_LETTER
python scripts/gen_cover_letter_docx.py paper_f Highlights
```

## Status

- [x] Primary experiments s50–s52
- [x] Scaling / k-sweep / MG transfer s53, s53b, s54
- [x] External selective-SSM baseline s66 (Limitations + Discussion updated)
- [x] Host freeze follow-up s67 (random/pretrained/online; bit-exact anchors)
- [x] Cover letter draft (`COVER_LETTER.docx`; single source of truth)
- [x] Highlights draft (`Highlights.docx`; bullets only, ≤85 chars each;
      single source of truth)
- [x] 2026-09-13 clean-P0 / meta / r2–r6 (C-ID alignment, capacity-budget
      accounting vs Gate-C-topk, claim JSON patch, abstract tightened,
      TTT bibitem → ICML
      2025 + arXiv:2407.04620, duplicated-sentence & overfull fixes)
- [x] 2026-09-14 compliance pass: declarations moved before the reference
      list, `Highlights.txt` reduced to bullets only (+ `.docx`
      regenerated), Paper E given a `\bibitem` + Zenodo concept DOI
      `10.5281/zenodo.22888172`, D given its Zenodo DOI,
      Lukoševičius name/venue corrected, bibliography re-ordered into
      first-citation order, capacity sentence corrected, s66 anchor field
      refreshed
- [x] 2026-09-13 r6 figures: `gen_paperF_figs.py` → Figs 1–2 embedded
      (15 pp) + graphical abstract; s66 lr-pooling disclosure in
      Limitations (10.19 = matched-lr value; 4× lr diverges)
- [x] Form-learning derivation note (removed 2026-09-17 with the other
      internal notes; git history retains it)
- [x] Git track + commit of F manuscript, data, scripts, figures
- [x] Supplementary zip (`Supplementary_Material_PaperF.zip`, local-only;
      rebuild with `python scripts/rebuild_paperF_supp_zip.py`)
- [x] VITAE + Declaration of interests (2026-09-22)
- [ ] Author pass on Highlights wording (`Highlights.docx` is the source of truth)

## Data provenance

<!-- BEGIN PROVENANCE (generated by scripts/provenance_map.py) -->

Claims registered for Paper F (includes shared anchors where relevant). Every registered headline quantity maps to a committed artefact expression. Generated by `scripts/provenance_map.py` from `scripts/claims_registry.json` — do not hand-edit it.

- claim call sites: **10**

| paper | section | manuscript quantity | where | produced by |
|---|---|---|---|---|
| F | C1 | B-softmax stream ppl | abstract / Paper F | `8.64834` |
| F | C1 | stream CE improvement magnitude (nats/token) | abstract / Paper F | `0.306837` |
| F | C1 | clip-floor park rate low (~1.6%) | Paper F | `1.58175` |
| F | C1 | clip-floor park rate high (~10.2%) | Paper F | `10.2289` |
| F | C1 | simplex does not repair (stream ppl delta vs lin-clip) | Paper F | `0.0283896` |
| F | C4 | Gate-C-topk stream ppl | abstract / Paper F | `7.03266` |
| F | C5 | soft routing forgetting ppl | Paper F | `7.78883` |
| F | C5 | hard routing forgetting ppl | Paper F | `6.93944` |
| F | external | X-SelSSM-frozen stream ppl | Paper F / Discussion | `9.02967` |
| F | external | F Gate-C-topk stream anchor (same host/stream) | Paper F | `7.03266` |

Register new numbers in `scripts/verify_claims.py`, re-run `run_all_audits.py`, then let this script rewrite the block.
<!-- END PROVENANCE -->
