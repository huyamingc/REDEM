# paper_e — Self-Evolution Main Line (S39–S65)

- Preprint (Zenodo concept DOI, all versions): [doi:10.5281/zenodo.22888172](https://doi.org/10.5281/zenodo.22888172)

This folder holds the REDEM "self-evolution / self-correction /
self-iteration" paper for **Neurocomputing**. It is **self-contained**
(frozen `deps/`, scripts `s39`–`s65` import those modules and can run
without the A–D pipeline). The current manuscript is `PAPER_E.tex`
(renamed from `PAPER_E_v3.tex`; the earlier assembly chapter and the
v1/v2 review rounds were removed on 2026-09-17 with the other internal
working files — git history retains the last committed state).

Current title (v3, 2026-09-13 clean-P0): *Self-evolution under ±1 reward:
sign-flip correction, reconstructed-label capacity sensing, and
snapshot-gated memory in a recurrent relaxation reservoir*.

> **Repository vs. local material.** This folder keeps only the current
> submission package: `PAPER_E.tex` / `.pdf`, cover letter, Highlights,
> VITAE/DECLARATION, README, Supplementary zip. Historical rounds
> (pre-rename assembly chapter, `PAPER_E_v1.*`, `PAPER_E_v2.*`), internal
> notes (`self_evolution_mainline.md`, Highlights_notes), and LaTeX build
> logs were removed on 2026-09-17; git history retains the last committed
> state of the tracked ones.
> Shared `scripts/` + `data/` + `deps/` remain the reproducibility path.

## Contents

- **Current manuscript:** `PAPER_E.tex` / `PAPER_E.pdf`
- Submission forms: `COVER_LETTER.docx`, `Highlights.docx`,
  `DECLARATION_OF_INTERESTS.docx`, `VITAE.docx`,
  `Supplementary_Material_PaperE.zip`
- Historical sources (pre-rename `PAPER_E`, `PAPER_E_v1.*`, `PAPER_E_v2.*`)
  and internal notes were removed on 2026-09-17 (git history retains
  the tracked ones)
- `deps/` — frozen substrate used by s39–s65
- [`PAPER_E.tex`](PAPER_E.tex) / [`PAPER_E.pdf`](PAPER_E.pdf) —
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
  memory**); cover letter and new Highlights aligned; abstract
  compressed further; Discussion no longer claims a measured "genuine
  trade-off" for the too-fast EMA path.
  (vii) **2026-09-13 autonomy/self expansion**
  (`modification_log_E_v3.md` §6): Discussion gains formal autonomy-vs-self
  distinction, self-attribution/verification/bounding, graded negatives
  (s42/s58c controlled vs s47 retracted, not a negative), host portability
  (E=reservoir, D/F=SSM, G out of scope), and Limitations conditions (single
  substrate, single ±1 feedback budget, no embodied closed loop).
  70 pp in the current PDF; 0 LaTeX errors; the new display equation was
  compacted after an initial 298 pt overfull. Compile with
  `pdflatex PAPER_E.tex` ×3 (`latexmk` is unavailable on this machine —
  MiKTeX cannot find its `perl` script engine).
  `Supplementary_Material_PaperE.zip` was rebuilt for v3 (manuscript =
  `PAPER_E.tex`, scripts s39–s65, 179 entries).
  Change logs (local only): `../review_workspace/modification_log_E_v3.md`,
  `../review_workspace/modification_log_EF_meta_20260913.md`,
  `../review_workspace/modification_log_EF_clean_p0.md`.
- `COVER_LETTER.docx` (submission artifact; single source of truth)
  — cover letter for Neurocomputing (plain-text source of truth + Word
  render; regenerate with `python ../scripts/gen_cover_letter_docx.py paper_e COVER_LETTER`).
- `Highlights.docx` (submission artifact; single source of truth) —
  Elsevier Highlights, **bullets only** (5 bullets, lengths
  **75/77/67/82/78**, all ≤85). Regenerate the docx with
  `node ../review_workspace/gen_highlights_docx.js paper_e`.
- `DECLARATION_OF_INTERESTS.docx` (local only) —  Elsevier declaration-of-interests form for Neurocomputing (no competing
  interests; date 2026-09-09), same layout as paper_a/b/c.
- `Supplementary_Material_PaperE.zip` (local only) —  self-contained reproduction package (manuscript, scripts s39–s65, frozen
  `deps/`, committed data, figures, README) mirroring the paper_a–d
  supplementary zips; staging lives at `../review_workspace/_zip_build_E/`.
  Rebuilt for v3.1 (manuscript = `PAPER_E.tex`, R6 direction fix;
  scripts 34 incl. `s65`; post-rerun committed full-run files plus
  refreshed s60 products; internal s58b trace excluded).
- `VITAE.docx` (submission artifact) — author biography
  (82 words ≤100), Times New Roman; regenerate with
  `node ../review_workspace/gen_vitae_E.js`.
- `../submission/Suggested_Reviewers_E.txt` (local only)
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
- Research-narrative companion `self_evolution_mainline.md` was removed
  on 2026-09-17 (git history retains it); the paper itself is
  `PAPER_E.tex`.
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

Every chain script lives in `../scripts/` (`s39_*` – `s65_*`, plus the
2026-09-18 follow-up `s64b_weak_source_reliability.py`) and imports
the frozen `deps/` modules here (they shadow the shared core). Run from the
**repo root** with **repository-relative paths** (forward slashes). `python`
may be `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (POSIX):

```bash
# from repo root
PYTHONUNBUFFERED=1 python scripts/s58e_d2_timescale.py
PYTHONUNBUFFERED=1 python scripts/s63_content_rendering.py --quick
PYTHONUNBUFFERED=1 python scripts/s64_multi_source_validation.py --quick
# s64 follow-up: intermediate-reliability cells → data/s64b_weak_source_reliability_v1.{csv,json}
PYTHONUNBUFFERED=1 python scripts/s64b_weak_source_reliability.py --quick
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

## Retracted claims and honest negatives (must match PAPER_E)

These are **not** headline results. Do not re-advertise them in series
README tables or cover letters.

| Item | Status in v3 | Evidence |
|---|---|---|
| **Structural (per-dimension) gain correction required** | **RETRACTED — not a negative result.** The only correction the stack demonstrates is a uniform sign flip $w\to-w$, which is *not* a per-dimension gain (`PAPER_E.tex` ~L805–811, ~L171–173; Discussion names s47 a retracted design attempt). | `s47_self_correction_gain` script-level **P2 [REFUTED by own data]**, P3 **[NOT VERIFIABLE]**; **no number from s47 is cited in the manuscript**. Do not mix with controlled negatives s42/s58c. |
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
- [x] **v3 closed** (PAPER_E.tex/.pdf, 70 pp current): significance repairs, EWC baseline, supplementary rebuilt for v3
- [x] **2026-09-13 meta + clean-P0**: title/abstract/cover letter/Highlights aligned to retracted-claim-safe vocabulary; abstract 283 words
      (LaTeX and math stripped) — Neurocomputing states no abstract word limit
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

Author review: the Highlights were newly drafted on 2026-09-13 (no prior
txt source); adjust the five bullets there, then regenerate the docx.

## Data provenance

<!-- BEGIN PROVENANCE (generated by scripts/provenance_map.py) -->

Claims registered for Paper E (includes shared anchors where relevant). Every registered headline quantity maps to a committed artefact expression. Generated by `scripts/provenance_map.py` from `scripts/claims_registry.json` — do not hand-edit it.

- claim call sites: **67**

| paper | section | manuscript quantity | where | produced by |
|---|---|---|---|---|
| E | R5 | frozen-hypothesis memory mean acc | abstract / Paper E | `0.888019` |
| E | R5 | re-adaptation / rls_single mean acc | Paper E | `0.837625` |
| E | R5 | frozen-snapshot retention gain (pp) | abstract / Paper E / s65 | `8.7` |
| E | EWC | EWC whole-run accuracy change (pp) | abstract / Paper E | `-0.851951` |
| E | EWC | EWC whole-run paired t | Paper E | `-3.87059` |
| E | EWC | s65 reference arm bit-matches s52 (max abs diff) | reproduction_check | `0` |
| E | R5 | cross-family ring gain paired t (corrected) | Paper E / MAINTENANCE | `2.44949` |
| E | R1 | post-inversion flip acc uncoupled | abstract / Paper E | `0.898` |
| E | R1 | post-inversion flip acc coupled | abstract / Paper E | `0.93875` |
| E | R1 | oracle post-inversion acc uncoupled | Paper E | `0.975` |
| E | R1 | oracle post-inversion acc coupled | Paper E | `1` |
| E | R1 | first-flip latency (blocks after inversion) | Paper E | `39` |
| E | R1 | learned flip value v (post-flip contrast) | Paper E | `0.503762` |
| E | R1 | stable-stream whole-run acc uncoupled (bit-identical no-meta) | Paper E | `0.897925` |
| E | R1 | stable-stream whole-run acc coupled (bit-identical no-meta) | Paper E | `0.938241` |
| E | R1 | R1 zoo coupled full-swap rmhl post | Paper E / Table tab:zoo | `0.058` |
| E | R1 | R1 zoo coupled full-swap flip post | Paper E / Table tab:zoo | `0.942` |
| E | R1 | R1 zoo coupled near-inversion rmhl post | Paper E / Table tab:zoo | `0.053` |
| E | R1 | R1 zoo coupled near-inversion flip post | Paper E / Table tab:zoo | `0.947` |
| E | R1 | R1 zoo partial-shift flip post-window | Paper E / Table tab:zoo | `0.48475` |
| E | R1 | R1 zoo partial-shift flip steady | Paper E / Table tab:zoo | `0.508` |
| E | R1 | R1 zoo partial-shift flip mean flips | Paper E / Table tab:zoo | `2.7` |
| E | R1 | corrupt-window acc uncoupled (honest negative) | Paper E | `0.106` |
| E | R1 | corrupt-window acc coupled (honest negative) | Paper E | `0.064` |
| E | R1 | rule-adjudication delta pre-swap uncoupled | Paper E | `0.99` |
| E | R1 | rule-adjudication delta steady uncoupled | Paper E | `0.989` |
| E | R1 | rule-adjudication RMHL post-swap uncoupled | Paper E | `0.0965` |
| E | R2 | raw capacity probe coupled (circle) | Paper E | `0.841441` |
| E | R2 | raw capacity probe uncoupled (circle) | Paper E | `0.546843` |
| E | R2 | RLS acc 4D shell2 (two-shell) | Paper E | `0.546798` |
| E | R2 | RLS acc 4D shell3 (three-shell) | Paper E | `0.559027` |
| E | R2 | 4D-shell margin at N=64 | Paper E | `0.034` |
| E | R2 | 4D-shell margin at N=256 | Paper E | `0.0995` |
| E | R2 | 4D-shell margin at N=512 | Paper E | `0.107` |
| E | R2 | 4D-shell margin at kappa=20 (sampled peak) | Paper E | `0.1185` |
| E | R2 | 4D-shell margin at deployed kappa=25 | Paper E | `0.0995` |
| E | R2 | 2D-circle margin flat across N (N=64) | Paper E | `0.401` |
| E | R3 | per-segment memory gain seg1_acc (pp) | Paper E | `4.15` |
| E | R3 | per-segment memory gain paired t seg1_acc | Paper E | `2.72314` |
| E | R3 | per-segment memory gain seg2_acc (pp) | Paper E | `3.6` |
| E | R3 | per-segment memory gain paired t seg2_acc | Paper E | `2.7309` |
| E | R3 | per-segment memory gain seg3_acc (pp) | Paper E | `2.55` |
| E | R3 | per-segment memory gain paired t seg3_acc | Paper E | `2.0214` |
| E | R3 | per-segment memory gain seg4_acc (pp) | Paper E | `14.85` |
| E | R3 | per-segment memory gain paired t seg4_acc | Paper E | `11.4637` |
| E | R3 | population control (no frozen snapshot) mean acc | Paper E | `0.838739` |
| E | R4 | sense reliability ABAB SEG=500 | Paper E | `0.9` |
| E | R4 | sense reliability ABAB SEG=250 | Paper E | `0.27` |
| E | R4 | sense reliability ABAB SEG=125 (cliff floor) | Paper E | `0` |
| E | R5 | learned snapshot gate v_snap final (dynamic) | Paper E | `0.252` |
| E | R5 | memory selection fraction (dynamic) | Paper E | `0.22695` |
| E | R6 | D2 post-swap at ratio 0.25 (fail band) | Paper E | `0.486161` |
| E | R6 | D2 post-swap at ratio 1.0 (intermediate) | Paper E | `0.561438` |
| E | R6 | D2 post-swap at ratio 1.25 (band interior) | Paper E | `0.791466` |
| E | R6 | D2 post-swap at ratio 1.5 (band interior) | Paper E | `0.849878` |
| E | R6 | D2 post-swap at ratio 1.75 (band interior) | Paper E | `0.909436` |
| E | R6 | D2 post-swap at ratio 2.0 (complete recovery) | Paper E | `0.937509` |
| E | R7 | single-speaker linear acc | Paper E | `0.937713` |
| E | R7 | A-B-A-B linear acc (switch_every=200) | Paper E | `0.899586` |
| E | R7 | A-B-A-B memory-recovered acc | Paper E | `0.912895` |
| E | R7 | capacity probe single-speaker | Paper E | `0.9493` |
| E | R7 | capacity probe A-B-A-B | Paper E | `0.9003` |
| E | R8 | minority-good majority-rule acc | Paper E / Table tab:multisource | `0.683686` |
| E | R8 | minority-good validation-selector acc | Paper E / Table tab:multisource | `1` |
| E | R8 | drifting majority-rule acc | Paper E / Table tab:multisource | `0.680815` |
| E | R8 | drifting validation-selector acc | Paper E / Table tab:multisource | `0.991307` |
| E | R8 | all-bad majority-rule acc (no source to trust) | Paper E | `0.493634` |

Register new numbers in `scripts/verify_claims.py`, re-run `run_all_audits.py`, then let this script rewrite the block.
<!-- END PROVENANCE -->
