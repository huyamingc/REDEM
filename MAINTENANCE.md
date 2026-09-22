# MAINTENANCE.md — maintainer notes (REDEM A–F)

This file is the ops history for `D:\work\papers\testpynew` (REDEM). `README.md`
is the front door for readers; this file keeps the historical narrative that is
useful when editing papers or scripts but noisy for readers. Fine-grained
round logs stay under `review_workspace/` (local / git-ignored).

---

## B-layer provenance suite (2026-09, first landing)

Goal: make important data, derivations, and logic chains **machine-traceable**
in the same sense as the author's induced-gravity paper (`llun`), adapted to a
six-paper ML program.

### What was added

| Role | File | Notes |
|---|---|---|
| CLAIM | `scripts/verify_claims.py` | **121** headline claims (A–F, incl. Paper E R1–R8 logic chain), reads committed `data/*` |
| TABLE | `scripts/check_tables.py` | claim doc-values matched to **tex table cells** (22 tabulars), with prose fallback |
| STALE/reverse | `scripts/audit_tex_numbers.py` | forbidden wording + light reverse count |
| PMAP | `scripts/provenance_map.py` | rewrites root + `paper_*/README.md` provenance blocks only |
| RNUM | `scripts/audit_readme.py` | README prose anchors must match registered claims |
| ORCH | `scripts/run_all_audits.py` | order above; `RUN_ALL_AUDITS_EXIT` |

`CLAUDE.md` App.1 Project map updated from the `llun` demo table to this
repo's real roles. Multi-paper convention: `claim(..., paper="E")` etc.

### Design choices (vs physics-level llun)

- **No `order_estimates.py` closed-form library** — almost all quoted numbers
  are experiment aggregates, not analytic magnitudes.
- **CLAIM recomputes from committed artefacts**, not from a 350 CPU-hour
  full re-run. Full re-runs stay sampled (`SCALE_AUDIT_REDEM.md`).
- **Reverse scan is light**: forbid known stale wordings; do not require every
  auxiliary coefficient to be registered (MVP).
- **Paper E s65 `reproduction_check_vs_s52.max_abs_difference` must stay 0**
  (identity claim) — this is the bit-exact pairing guarantee.

### Spot-check proven before the suite

| Quoted | Artefact | Status |
|---|---|---|
| 0.996 / 0.973 | `s8_integrated_v1.json` | match |
| 0.9970 / 0.9753 | `s30_integrated_1024_v1.json` | match |
| 8.47 / 6.41 | `s11_disturbance_chain_v1.json` | match |
| +8.70 pp retention | `s52_dynamic_memory_v1.csv` (seg3+seg4) | match |
| −0.85 pp, t=−3.87 | `s65_ewc_baseline_v1.json` | match |
| 11.75 → 8.65, 7.03 | `s50`/`s51` Paper F | match |
| cross-family t=2.45 | `s61_cross_family_v1.csv` seg4 coupled paired t | match (was falsely read as t<2.262) |

### Failures fixed in the same round

1. Cross-family claim initially had no executable source — now derived from
   `s61` CSV as paired `seg4_acc` (`pop_mem - rls_single`) on `random_graph_k25`
   → `t=2.449`. Manuscript/README wording that `2.45 < 2.262` is blacklisted.
2. F simplex Δppl quoted as `+0.03` measures `0.0284`; tolerance set to 10%
   relative and noted in the claim.
3. RNUM required `8.65` in README prose; Paper F bullet now carries the
   headline `11.75 → 8.65`.

---

## Paper E significance corrections (pre-existing, 2026-09-10)

Recorded here so the reverse audit and future editors share one story:

- Cross-family ring gain `+1.4±1.8` pp on the deployed substrate has paired
  `t=2.45` (`df=9`, `p≈0.037`), **above** the 95% critical `2.262`. Earlier
  text claimed it fell below critical and withheld the claim for the wrong
  reason. Corrected withholding rests on effect size / seed consistency
  (`5/10` wins), not on non-significance.
- Per-segment memory appendix: three of four gains are significant
  (`t=2.72`, `2.73`, `11.46`); only the `+2.6` pp linear revisit (`t=2.02`)
  is not. The old “only the last is significant” sentence was false.

## Paper E retract boundary

`s47_self_correction_gain` is a **retracted design attempt**, not a negative
result. Title/abstract use sign-flip / reconstructed-label / snapshot-gated
vocabulary only. Do not reintroduce “per-dimension correction required”.

## Paper F instrument change

Squared-loss clip-floor readout parks ~1.6–10.2% of tokens on a floor; trained
softmax is the load-bearing instrument change (stream 11.75→8.65). Same-token-set
CE pairing is supported by `data/per_token/*.npz` + `per_token_io.py`.
External SelSSM host params are frozen at random init (s66/s67).

## Scale ceiling

Largest simulated substrate **N=1024** (`s30`); largest host state 512
(`s53`/`s53b`). Vocabulary 32. No scaling claims. Details:
`SCALE_AUDIT_REDEM.md`.

## Red lines still in force (CLAUDE.md A11 / App.1)

Do not change: physics/device constants inherited from Paper 0 (γ, τ₀, …),
CORE formulas in `recurrent_substrate.py` / `online_readout.py`, CSV/JSON
output column names, RNG seed rules, existing `claim(...)` registrations
without re-running `run_all_audits.py`, or output filenames the manuscripts
already cite.

`MAINTENANCE.md` may be public; `review_workspace/` and `CLAUDE.md` stay
local-only if the repo is published without process notes.

## A–C claim expansion + per-paper PMAP (same session, second landing)

- `verify_claims` 31 → **50** claims: Paper A κ* (25.3/27.4/27.9), S6 restore
  (7.9/18.3/11.8%), E4 λ_target=0 +25% at CV=0.1; Paper B S30 paired t=15.3,
  S25 novelty 14.59 vs corr 12.43, S24 8.47/5.27, S2 band mean; Paper C S10
  equalization 0.998/0.996/0.994, S14 falsification diffs −0.78/−0.76/−0.69.
- `provenance_map` now writes **root + `paper_a..f/README.md`** blocks.
- RNUM extended (15 checks). Fixed `s30` CSV stem (`*_v1`) that left the
  paired-t claim uncomputed on the first pass.
- Gates: `run_all_audits` exit 0 (50/0, 50/0, FLAG=0, PMAP OK, readme 15/0).

## Paper D headline expansion (same session)

Added 11 Paper D claims from `s20`/`s21`/`s22`/`s23`/`s31`:
A3−A1 forgetting diffs (−2.05/−1.87/−1.20 at τ_m=200/500/1000), soft vs
abrupt stream (8.22 vs 10.03), M5 norms (11.3 vs 50.2), P4 table
(13.18/8.93), real-text (12.07) and bigram ceiling (10.97). Fixed `_j()`
to accept stems without `.json`. Gate: **61/0**, table 61/0, FLAG=0,
PMAP OK, RNUM 15/0.

## Paper E R1–R8 logic-chain claims + table-cell audit (same session)

- `verify_claims` 61 → **121** claims: full Paper E R1–R8 chain from committed
  artefacts (s39/s40/s41/s42/s44/s46 rule selection and zoo; s53/s58a/s58d
  representation and margins; s52 per-segment memory gains + t; s58b
  reliability cliff 0.90/0.27/0.00; s59 learned snapshot gate; s58e D2 band
  0.486→0.938 across ratio 0.25–2.0; s63 content–rendering 0.938→0.900 and
  memory recovery 0.913; s64 majority fallacy 0.684 vs 1.000).
- `check_tables` upgraded from corpus-string presence to **tex table-cell**
  matching: parses 22 `tabular` blocks, matches claim `doc_value` to numeric
  cells, records `table_hit` vs `prose`, and only *forces* a cell hit for
  explicit table anchors (`tab:zoo`, `tab:multisource`).
- Paper A κ* claims (`25.3`/`27.4`/`27.9`) live in prose (fine-sweep text),
  not table cells; `where` no longer claims `Table`.
- Gate: verify_claims **121/0**, check_tables **121 / 22 tables / 0 fail**.

## Pre-submission package check (2026-09-22)

- **E**: package complete (PAPER_E.pdf 71 pp, COVER_LETTER, Highlights, VITAE,
  DECLARATION_OF_INTERESTS, Supplementary_Material_PaperE.zip). Key R1–R8
  table/abstract numbers present in tex; claim gate 121/0.
- **D**: added missing `VITAE.docx` + `DECLARATION_OF_INTERESTS.docx` (same
  author boilerplate as E, manuscript title swapped). PDF 32 pp. Headline
  numbers (11.75→8.65 family is F; D quotes −1.81 soft−abrupt margin rather
  than absolute 8.22/10.03 — those absolutes stay in CLAIM/README only).
- **F**: added `VITAE.docx`, `DECLARATION_OF_INTERESTS.docx`, and
  `Supplementary_Material_PaperF.zip` via new
  `scripts/rebuild_paperF_supp_zip.py` (manuscript + s50–s54/s66–s67 +
  data + figures). PDF 15 pp.
- Still human-only: suggested reviewers (`../submission/Suggested_Reviewers_E.txt`
  referenced but not in repo), author final read, journal portal fields.

## Backups

Prefer `git stash` / branches over `.bak` copies. Temporary claim probes
(`review_workspace/_tmp_claim_check.py`) are superseded by
`scripts/verify_claims.py`.
