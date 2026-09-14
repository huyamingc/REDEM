# paper_e — Self-Evolution Main Line (S39–S65)

This folder holds the REDEM "self-evolution / self-correction /
self-iteration" paper for **Neurocomputing**. It is **self-contained**
(frozen `deps/`, scripts `s39`–`s65` import those modules and can run
without the A–D pipeline). The Chinese derivation chapter (`PAPER_E.tex`)
is the historical claim-driven draft; the current manuscript is
`PAPER_E_v3.tex`.

Current title (v3, 2026-09-13 clean-P0): *Self-evolution under ±1 reward:
sign-flip correction, reconstructed-label capacity sensing, and
snapshot-gated memory in a recurrent relaxation reservoir*.

## Contents

- [`PAPER_E.tex`](PAPER_E.tex) / [`PAPER_E.pdf`](PAPER_E.pdf) — historical
  claim-driven assembly chapter (route R58F): Abstract, Introduction with six claims
  (R1–R6), Methods, Results organized by claim, Discussion (positioning,
  methodological notes, limitations), Conclusion, and appendices (theorem
  index, experiment reproducibility index, key numbers). The
  derivation-chain labels (s39–s65) appear only in the reproducibility
  index, not in the narrative. Standalone `article` class (Elsevier "Your
  Paper Your Way" convention, same as paper_a–d). Compiles cleanly: 11 pp,
  exit 0, zero overfull. **Superseded by the v1/v2/v3 elsarticle line.**
- [`PAPER_E_v1.tex`](PAPER_E_v1.tex) / [`PAPER_E_v1.pdf`](PAPER_E_v1.pdf) —  **reviewed + submission-ready version** (2026-09-09): converted to the
  Elsevier `elsarticle` class (`\documentclass[preprint,12pt]{elsarticle}`,
  `\journal{Neurocomputing}`), with 26 references, submission declarations
  (competing interests / funding / generative-AI / acknowledgements), and
  all P0/P1 review fixes (see
  [`../review_workspace/`](../review_workspace/), suffix `_E_v1`). 23 pp,
  compiles clean. The pre-conversion `article` version is backed up at
  `../review_workspace/PAPER_E_v1_article_backup.tex`.
- [`PAPER_E_v2.tex`](PAPER_E_v2.tex) / [`PAPER_E_v2.pdf`](PAPER_E_v2.pdf) —
  **deep second-round review version** (2026-09-10):
  a six-agent re-review (Agent A–F, including a full audit of all 37 chain
  scripts) plus a meta-level pass, cross-validation and two-stage
  adjudication produced **132 findings (28 P0 + 104 P1)**; all of them are
  closed in this file, including the C-tier items that required editing and
  re-running the scripts and back-filling the resulting numbers
  (`../review_workspace/c_tier_rerun_report_E_v2.md`,
  `../review_workspace/modification_log_E_v2.md`). 60 pp, compiles with
  0 errors / 0 undefined references / 0 `Float too large`. Substantive
  changes vs v1: the substrate is described as what it is (a *recurrent
  relaxation reservoir*, a physics-inspired numerical model, not a spiking
  device), a `Statistics and reporting conventions` paragraph fixes the
  statistical contract (n=10, sample SD with the n−1 denominator, paired t
  with df=9), s63/s64 were promoted from Discussion into Results as R7/R8,
  the appendix theorem index became a classified *findings and empirical
  claims index*, and the numbers that changed after the C-tier reruns were
  back-filled (κ peak at 20 vs deployed 25, the 11-point D2 ratio sweep,
  +15.0 pp for the adaptive τ₂, the s60 window comparison, and the s53
  raw-feature capacity band). `PAPER_E_v1.tex` is kept unchanged as the
  reviewed baseline.
- [`PAPER_E_v3.tex`](PAPER_E_v3.tex) / [`PAPER_E_v3.pdf`](PAPER_E_v3.pdf) —
  **current manuscript** (2026-09-10 stats/positioning repair, then
  2026-09-13 meta / clean-P0 wording, then 2026-09-13 autonomy/self
  expansion). v3 starts from the v2 text and makes
  the following changes, each verified against the committed per-run data:
  (i) two **false significance statements** are corrected. The deployed
  cross-family ring gain (`+1.4±1.8` pp) was reported as falling *below* the
  95% critical value `2.262`, but the committed `s61_cross_family_v1.csv`
  runs give `t=2.45` (`df=9`, `p≈0.037`), which is *above* it; the claim is
  now withheld on effect size and seed consistency (`5/10` wins, `11.3%`
  mean selection fraction) rather than on a failure to reach significance.
  The appendix claim that only the last per-segment memory gain is
  significant is also wrong: three of the four exceed the critical value
  (`t=2.72`, `2.73`, `11.46`) and only the `+2.6` pp linear revisit
  (`t=2.02`) does not. (ii) The Introduction sentence whose
  `not accuracy … but autonomy` contrast had been split across two `but`s is
  rewritten, and the "not engaged properly" positioning paragraph now places
  the paper against the four upstream literatures, with three references
  added (`dambre2012`, `gama2014`, `truong2020`; bibliography 27 → 30).
  (iii) The Abstract is cut from 379 words to **238** (MethodA count; ≤250),
  with a boundary map instead of a negative-results laundry list.
  (iv) An **external baseline** is added, closing the "no external method
  comparison" gap: `scripts/s65_ewc_baseline.py` runs Elastic Weight
  Consolidation (Kirkpatrick et al. 2017) on the *same* A-B-A-B revisit
  protocol, task instances and seeds as s52, at a matched optimiser (the EWC
  penalty injected into the same RLS readout as exact least-squares
  pseudo-observations), with a second arm for EWC as published trained by
  AdaGrad and a no-penalty AdaGrad control. Tuning is a 3-seed grid reported
  in `data/s65_tuning_v1.json`; λ is selected on revisit retention because
  selecting on whole-run mean drives it to zero. The reproduction check is
  exact: the reference arm matches s52's committed per-seed values in 50/50
  comparisons (max difference 0.0), so the arms are paired on identical task
  realizations. Result: EWC does not improve the run (whole-run
  −0.85±0.70 pp, t=−3.87, 0/10 seeds) and does not change total revisit
  retention (−0.60±2.02 pp, n.s.); it only redistributes retention between the
  two revisits (circle +4.95±4.28 pp, linear −6.15±3.72 pp), whereas the
  frozen-snapshot arm gains +8.70±2.92 pp of retention with no linear loss.
  The AdaGrad arm is optimiser-limited (−22.3 pp, t=−12.03, 0/10) and is
  reported as such rather than as evidence about EWC. All "no external
  method" statements in the text (Abstract, Introduction, Discussion,
  Limitations, Conclusion) were updated accordingly.
  (v) **2026-09-13 meta pass** (`modification_log_EF_meta_20260913.md`):
  sign-test claim removed / win counts descriptive; capacity-sense wording
  decision-level; D2 as empirical regularity; figure SEM captions; several
  conclusion/R-metric alignments.
  (vi) **2026-09-13 clean-P0** (`modification_log_EF_clean_p0.md`):
  title retargeted to the retracted-claim-safe vocabulary (**sign-flip
  correction**, **reconstructed-label capacity sensing**, **snapshot-gated
  memory**); cover letter and new `Highlights.txt` aligned; abstract
  compressed further; Discussion no longer claims a measured "genuine
  trade-off" for the too-fast EMA path.
  (vii) **2026-09-13 autonomy/self expansion**
  (`modification_log_E_v3.md` §6): Discussion gains formal autonomy-vs-self
  distinction, self-attribution/verification/bounding, graded negatives
  (s42/s58c controlled vs s47 retracted, not a negative), host portability
  (E=reservoir, D/F=SSM, G out of scope), and Limitations conditions (single
  substrate, single ±1 feedback budget, no embodied closed loop).
  68 pp after the expansion; 0 LaTeX errors; the new display equation was
  compacted after an initial 298 pt overfull. Compile with
  `pdflatex PAPER_E_v3.tex` ×3 (`latexmk` is unavailable on this machine —
  MiKTeX cannot find its `perl` script engine).
  `Supplementary_Material_PaperE.zip` was rebuilt for v3 (manuscript =
  `PAPER_E_v3.tex`, scripts s39–s65, 179 entries).
  Change logs: [`../review_workspace/modification_log_E_v3.md`](../review_workspace/modification_log_E_v3.md),
  [`../review_workspace/modification_log_EF_meta_20260913.md`](../review_workspace/modification_log_EF_meta_20260913.md),
  [`../review_workspace/modification_log_EF_clean_p0.md`](../review_workspace/modification_log_EF_clean_p0.md).
- [`COVER_LETTER.txt`](COVER_LETTER.txt) / [`COVER_LETTER.docx`](COVER_LETTER.docx)
  — cover letter for Neurocomputing (plain-text source of truth + Word
  render; regenerate with `python ../scripts/gen_cover_letter_docx.py paper_e COVER_LETTER`).
- [`Highlights.txt`](Highlights.txt) / [`Highlights.docx`](Highlights.docx) —
  Elsevier Highlights (5 bullets, lengths **74/80/76/84/77**, all ≤85),
  regenerated 2026-09-13 against the corrected title/abstract (the previous
  docx still said “spiking substrate” / per-dimension correction). Regenerate
  with `python ../scripts/gen_cover_letter_docx.py paper_e Highlights`.
- [`DECLARATION_OF_INTERESTS.docx`](DECLARATION_OF_INTERESTS.docx) —  Elsevier declaration-of-interests form for Neurocomputing (no competing
  interests; date 2026-09-09), same layout as paper_a/b/c.
- [`Supplementary_Material_PaperE.zip`](Supplementary_Material_PaperE.zip) —  self-contained reproduction package (manuscript, scripts s39–s65, frozen
  `deps/`, committed data, figures, README) mirroring the paper_a–d
  supplementary zips; staging lives at `../review_workspace/_zip_build_E/`.
  Rebuilt for v3 (manuscript = `PAPER_E_v3.tex`; scripts 34 incl. `s65`;
  post-rerun committed full-run files plus the
  `s58b_trace_seed4.json` trace and refreshed s60 products).
- [`VITAE.txt`](VITAE.txt) / [`VITAE.docx`](VITAE.docx) — author biography
  (82 words ≤100), Times New Roman; regenerate with
  `node ../review_workspace/gen_vitae_E.js`.
- [`../submission/Suggested_Reviewers_E.txt`](../submission/Suggested_Reviewers_E.txt)
  — five real, field-matched candidate reviewers (reservoir computing /
  reward-modulated plasticity / memory consolidation / continual learning),
  each a cited author in the manuscript, with affiliation + e-mail + match
  rationale.
- [`deps/`](deps/) — Paper E's **frozen, self-contained dependencies** (the
  four core modules; extended versions of `online_readout.py`/`streaming_tasks.py`
  used by the chain). The shared `scripts/` core was reverted to its
  original state (Papers A–D canonical) on 2026-09-09; see `deps/README.md`.
  Two audited fixes landed here on 2026-09-10: `shallow_trap_array_simulator.py`
  now uses `np.broadcast_to(arr, shape).copy()` on its noiseless branch (the
  previous `np.broadcast(...)` raised `ValueError` for `col_noise_rms == 0`;
  a `--self-test` noiseless regression check was added), and
  `recurrent_substrate.py` documents the `alpha_eff` clip equivalence and the
  deployed saturation fraction (0.393). Both are recorded in
  `../review_workspace/modification_log_scripts.md`; neither changes the
  noisy main path or any committed number.
- Fixes applied to the shared chain scripts for v2 (details in
  `../review_workspace/edit_log_A7_E_v2.md`, `edit_log_A8_E_v2.md` and
  `modification_log_scripts.md`): `s48_rotation2d.py` (probe over all 10
  seeds + a post-rotation accuracy window), `s52_dynamic_memory.py`
  (per-specialist seeds for the `pop_plain` control), `s53_auto_basis.py`
  (new `raw_capacity_probe` block that sources the raw-feature capacity
  band), `s54_memory_boundaries.py` (collision-free cycle sub-seeds),
  `s58b_relative_sense_stress.py` / `s58f_margin_sensitivity.py` (real
  per-arm branching instead of the duplicated-arm pseudo-replication),
  `s58d_kernel_capacity_sweep.py` (8-point κ grid), `s58e_d2_timescale.py`
  (11-point ratio sweep), `s58e`/`s62` (explicit `w_max=None`, no norm cap),
  `s62_adaptive_meta_timescale.py` (the physical `[0.001, 0.10]` injection
  clip is no longer shadowed by local meta-layer bounds),
  `s59_self_tuned_gate.py` / `s60_adaptive_window.py` (docstrings aligned to
  the implementation and to the re-run data), `s60_adaptive_window.py`
  (memory-gated snapshot block so the two adaptive arms are independent
  runs), `s63_content_rendering.py` (`AGREE_THRESH` as a recorded constant),
  `s43`/`s43b`/`s44`/`s45`/`s46`/`s47` (refuted-prediction annotations,
  hyper-parameters written to JSON, dead constant removed), `s55`/`s57`/
  `s58a`/`s58d` (CORE constants imported instead of hard-coded), and
  `gen_fig_self_evolution.py` (per-file data map, error bars and sample
  sizes, f6 tick 0.24, f1 shading 0–120).
- [`self_evolution_mainline.md`](self_evolution_mainline.md) — the
  research-narrative / provenance companion (how the claims were derived);
  the paper itself is `PAPER_E_v3.tex`.
- Figures (in `../figures/`, generated by `../scripts/gen_fig_self_evolution.py`):
  `self_evo_f1_reliability_cliff.pdf` (R4/s58b), `self_evo_f2_margin_kappa_N.pdf`
  (R2/s58d), `self_evo_f3_d2_threshold.pdf` (R6/s58e),
  `self_evo_f4_vsnap_gate.pdf` (R5/s59), `self_evo_f5_cross_family.pdf`
  (R5/s61), `self_evo_f6_meta_timescale.pdf` (R6/s62). All six were redrawn
  for v2 (error bars / Wilson intervals / sample-size labels, the 8-point κ
  grid, the 5 interior D2 ratio points, the f6 tick at 0.24) and audited
  again after the C-tier reruns: re-rendering them from the current
  committed data reproduces the shipped PDFs byte-for-byte once the PDF
  creation timestamp is removed (see
  `../review_workspace/edit_log_A8_E_v2.md` §二).


## Reproduction

Every chain script lives in `../scripts/` (`s39_*` – `s64_*`) and imports
the frozen `deps/` modules here (they shadow the shared core). Run from the
repo root, e.g.:

```powershell
$env:PYTHONUNBUFFERED=1; python scripts\s58e_d2_timescale.py
$env:PYTHONUNBUFFERED=1; python scripts\s63_content_rendering.py --quick
$env:PYTHONUNBUFFERED=1; python scripts\s64_multi_source_validation.py --quick
```

The chain scripts accept `--quick` (a 2-seed screening run) and nothing else;
the `--sequential` flag used in earlier drafts of this file is **not
implemented** and is silently ignored (the scripts always run their Monte
Carlo sweep through `multiprocessing.Pool`). Full 10-seed runs of the long
scripts (s58b ~17 min, s58f ~17 min, s62 ~2.5 min) must therefore be
launched **one script at a time** — two concurrent full runs exhausted the
available memory during the v2 audit.

`s63_content_rendering.py` (PAPER-type, 2026-09-09) tests content–rendering
separation: the same semantic content (fixed u, fixed labels) rendered by
two speakers (affine render A_s) — single linear readout 0.938 → 0.900
(mixed, paired t=-18.2, 10/10), quadratic expansion does not repair
(0.889, t=-4.7), frozen-hypothesis memory recovers (0.913, t=+9.3), and
the observational meta layer is render-blind (0 flips at every switch
rate). Full 10-seed (320 runs) results are in
`data/s63_content_rendering_v1.{csv,json}` (2-seed screening in
`..._v1_quick.*`); the Discussion paragraph and appendix index cite it as
s63.

`s64_multi_source_validation.py` (PAPER-type, 2026-09-10) tests
multi-source validation: the same content served by five information
sources that give conflicting suggestions — can a majority rule find the
correct one, or must selection be validation-driven (the S52
argmax-r_ema device applied to sources)? Key results (10 seeds):
majority-good (3 good + 2 noise) all arms at ceiling (vote 1.000);
minority-good (1 good + 4 noise) the majority rule collapses to 0.684
(exactly 11/16, the Binomial(4,0.5) majority chance) while the
validation-selected arms stay at the oracle (emasel 1.000, worker 0.938 =
worker_true, t=100.5, 10/10) — the majority fallacy is demonstrated;
all-bad (5 noise) every arm at chance ~0.49 and the EMA selector finds no
signal (242 switches) — the D4 self-detection boundary; drifting (good
source migrates) the majority rule stays stuck at 0.681 while validation
re-selects the new good source (0.991, one switch, t=107.4, 10/10). Full
10-seed (500 rows) results are in
`data/s64_multi_source_validation_v1.{csv,json}` (2-seed screening in
`..._v1_quick.*`); the Discussion paragraph, Table S64 (tab:multisource)
and appendix index cite it as s64.

## Source of truth

- Chinese traceability (per-script records, negative-result detail,
  modification log): `../review_workspace/derivation_record_s39-s44.md`
- Route map (executed / remaining candidate routes):
  `../review_workspace/derivation_route_map.md`
- CORE modification log (2026-09-08, four authorized non-destructive
  changes): `../review_workspace/modification_log_scripts.md`
- Headline registry and reproduction commands: `../README_REDEM.md`

## Retracted claims and honest negatives (must match PAPER_E_v3)

These are **not** headline results. Do not re-advertise them in series
README tables or cover letters.

| Item | Status in v3 | Evidence |
|---|---|---|
| **Structural (per-dimension) gain correction required** | **RETRACTED — not a negative result.** The only correction the stack demonstrates is a uniform sign flip $w\to-w$, which is *not* a per-dimension gain (`PAPER_E_v3.tex` ~L805–811, ~L171–173; Discussion names s47 a retracted design attempt). | `s47_self_correction_gain` script-level **P2 [REFUTED by own data]**, P3 **[NOT VERIFIABLE]**; **no number from s47 is cited in the manuscript**. Do not mix with controlled negatives s42/s58c. |
| **s42 corrupt reward (D4)** | **Honest negative (controlled), reported.** Corrupted-window accuracy $0.106\pm0.014$ / $0.064\pm0.008$; 3.0 vs 1.0 flips. Self-deception: polluted reward is not self-detectable. | `s42_corrupt_signal`, Discussion + reproducibility index |
| **s58c multi-level expansion** | **Scoped positive + honest negative.** L1 10/10, L2 5/10, ring/cubic rung **refused**; ladder capped at level 2. Memory *masks* divergence (does not prevent it): both arms measure weight norm; worker divergence identical; 0.801 vs 0.504. | `s58c_multi_level_expansion`, Discussion |
| **Series README `README.md:112`** | Must not list per-dimension correction as a core result. | Fixed 2026-09-13 in the root table |

## Status

- [x] S39–S45 Hebbian learner + meta compensation
- [x] S46–S47 credit assignment = rule selection; ~~structural correction~~ (retracted; only sign flip remains)
- [x] S48–S49 L2 decodability base
- [x] S50–S58a L3 corrected trigger picture (incl. honest negatives)
- [x] S52–S56 the self-evolution loop (memory × relative capacity sense)
- [x] S58b capacity-sense applicability boundary
- [x] S58c multi-level chain + memory = divergence protection
- [x] S58d rich-kernel saturation (κ inverted-U, N monotone)
- [x] S58e D2 timescale separation verified
- [x] S58f margin calibration negative (cliff is structural, not tunable)
- [x] G1 self-tuned snapshot gate (S59, positive: memory layer learns when to keep experience)
- [x] G2 adaptive window (S60, honest negative: window can't fix the cliff either)
- [x] G3 cross-family transfer (S61, partial & structure-matched: +8.9pp on the ring)
- [x] G4 adaptive meta timescale (S62, τ₂ adaptable but D2 threshold is pipeline-bound)
- [x] s63 content–rendering separation (R5: memory, not expansion, separates renders)
- [x] s64 multi-source validation (majority fallacy vs validation-driven selection; D4 boundary)
- [x] s65 external EWC baseline (does not improve the run; only redistributes revisit retention)
- [x] Six figures generated (figures/self_evo_f1–f6.pdf)
- [x] Claim-driven paper compiled (PAPER_E.tex/.pdf, 11 pp, zero overfull; R1–R6 structure, S-labels only in the appendix; historical)
- [x] Frozen self-contained deps/ (shared core reverted to A–D canonical)
- [x] v1 review round closed (PAPER_E_v1.tex/.pdf, 23 pp; suffix `_E_v1` in `../review_workspace/`)
- [x] **v2 deep review round closed** (PAPER_E_v2.tex/.pdf, 60 pp): six-agent + meta-level review, 132 findings (28 P0 + 104 P1) all closed, C-tier scripts edited and re-run, numbers back-filled, six figures re-audited
- [x] **v3 closed** (PAPER_E_v3.tex/.pdf, 63 pp): significance repairs, EWC baseline, Abstract ≤250, supplementary rebuilt for v3
- [x] **2026-09-13 meta + clean-P0**: title/abstract/cover letter/Highlights aligned to retracted-claim-safe vocabulary; abstract now 238 words
- [x] **2026-09-13 autonomy/self expansion**: Discussion gains operational-self formalization, graded negatives (s42/s58c controlled; s47 retracted), host scoping; cover letter/Highlights re-aligned (s47 no longer "honest negative"); supplementary zip manuscript entry refreshed
- [x] journal targeting / final copy-edit

## Open items

From the v2 audit (see `../review_workspace/modification_log_E_v2.md`
section "D 档清单"): `s52` `paired_stats()` still drops its per-arm block,
the `s58e`/`s62` docstrings still claim a `w_max` cap that the code
deliberately does not apply, the pooled Wilson intervals in `s58b` are
computed at n=20 while the effective sample size is ~10 seeds (the paper
states this explicitly), the `--sequential` flag is not implemented,
16 JSON files outside this paper's citation set still contain bare `NaN`,
and the public repository push with a commit/tag is still pending.

From the v3 residual notes (`modification_log_E_v3.md`): `_index/papers.json`
still points at a nonexistent `PAPER_E_v4.tex`; external comparison remains
one method deep (EWC only — no reservoir-specific online learner, no
matched-budget supervised baseline; that boundary is in Limitations).

Author review: `Highlights.txt` was newly drafted on 2026-09-13 (no prior
txt source); adjust the five bullets there, then regenerate the docx.
