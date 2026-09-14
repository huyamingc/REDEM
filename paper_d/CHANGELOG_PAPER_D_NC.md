# CHANGELOG — Paper D Neurocomputing retarget (`PAPER_D_NC.tex`)

## 2026-09-13 — epistemic scope (P1 nonlinear extension, calibration vs information, floor formalization, Conclusion/Limitations)

- **Backup:** `PAPER_D_NC.tex.bak.pre_epi_scope` (and PDF if present).
- **New section `sec:epi`** after `sec:sens`, before Discussion:
  - `sec:p1nl` — Extension of Proposition 1 to nonlinear readouts:
    P1 is affine-linear exact recovery only; post-hoc simplex (0/10) vs trained
    softmax (ΔCE −0.307 nats); full state stream-neutral/retention-harmful under
    softmax; 32-d slice stream-positive → **selection** statement, not nonlinear
    impossibility. Nonlinear/SiN-style measurements sit outside P1's linear
    measurement assumption; pooled head still fails.
  - `sec:calinfo` — Calibration vs information: recoverability under a function
    class vs support-validity/proper-scoring calibration; s35 (fast-channel
    linear decode) coexists with 0/10 pooled-head failure; forbidden inference
    named explicitly.
  - `sec:floor` — Floor sensitivity formalization:
    `E[S]=(1−α)μ+αc_ε`, floor-tax differential, floor-dominated ranking,
    same-token-set as partial repair, floor-bearing vs floor-free metric
    families; host requirements as knowability vs verifiability boundaries.
- **Discussion:** wires `sec:epi` into the three host requirements; closes with
  the knowability/verifiability split for the self-evaluation program.
- **Limitations:** scope of P1, nonlinear extension, floor decomposition, and
  calibration terminology.
- **Conclusion:** F's prescription is no longer open — simplex falsified,
  trained softmax executed, hard top-k repairs retention; remaining next steps
  restated; D positioned as the diagnosis layer.
- **Unchanged:** all experimental numbers, Prop 1 statement/proof, tables,
  figures, abstract F pointer.

## 2026-09-13 — post-review fixes (SSM-bare cell, intro ESN pointer, ESN-lin note)

Follow-up to the ESN/ablation round, from the review evaluation:
- `tab:m3m4` / `tab:esn`: SSM-bare forgetting cell corrected `13.41` -> `13.40`
  (matches `tab:p4` and the s38/s22 JSON mean 13.4046; ESN-lin's own 13.41 is
  its data and is unchanged).
- Introduction: "represented by the bare M1 arm rather than a separately wired
  random reservoir" replaced by a pointer to the new `sec:esn` ESN+RLS control.
- `sec:esn`: added one sentence explaining ESN-lin (one-hot control, 12.4) vs
  reported SSM-bare (15.4): the gap is the clip floor (unclipped bare ~11.5).
- No experiment rerun; no other numeric claim changed.

## 2026-09-13 — ESN baseline + P4 M3/M4 ablation + F bib

New experiments (no change to prior numeric claims):

| Artifact | Role |
|---|---|
| `scripts/s38_ssm_p4_m3m4_ablation.py` | P4 factor: bare / uniform / hard / soft (10 seeds) |
| `data/s38_ssm_p4_m3m4_ablation_v1.*` | Factor-ablation results |
| `scripts/s40_esn_rls_p4_baseline.py` | Classical ESN+RLS on P4 (10 seeds) |
| `data/s40_esn_rls_p4_baseline_v1.*` | ESN baseline results |

Paper changes:
- New subsections `sec:m3m4` and `sec:esn` with Tables `tab:m3m4` / `tab:esn`.
- Limitations no longer claim "no ESN baseline" / "M3 and M4 not separated".
- Conclusion updated with the two new findings.
- Formal bibitem `redemf` for Paper F; abstract cites `\cite{redemf}`.
- `s21 --frozen-p-probe` fixture check: 0 drift (frozenW = 32.0 exactly).

## 2026 — F-wording (scope floor to readout family + companion note)

- **Backup:** `PAPER_D_NC.tex.bak.pre_F_wording` (and PDF if present).
- **Abstract:** keep floor sensitivity scoped to *this squared-loss RLS readout family*; add one sentence that a companion study localizes the artifact in the **training objective** (post-hoc simplex removes negatives without repairing stream scores; trained softmax does).
- **Conclusion (already present, retained):** next steps name a *calibrated probabilistic readout in the training objective* and cite the companion execution (simplex negative control vs trained softmax). The old "non-negative readout or simplex projection" range-fix framing is **not** used as the primary prescription.
- **Unchanged:** all experimental numbers, Prop 1, M5 negative, four-domain stack claims, E abstract.
- **Not done here:** full Paper F cross-ref bib entry (companion cited in prose only until bibliography update).

Date: 2026-09-10  
Source baseline: `PAPER_D_AI.tex` (Applied Intelligence draft)  
New target: **Neurocomputing** (Elsevier, `elsarticle`)  
Related new files: `Highlights_D_NC.txt`, `COVER_LETTER_D_NC.txt`, this changelog.  
Unchanged originals: `PAPER_D.tex`, `PAPER_D_AI.tex`, `Highlights_D_AI.txt`, `COVER_LETTER_D_AI.txt`.

This log is for traceability only. Experimental numbers, equations, data anchors, and figure files were **not** regenerated; every numeric claim is carried over from the AI draft.

---

## 1. Journal retarget (structural)

| Item | AI draft | NC draft | Why |
|---|---|---|---|
| Document class | `article` 11pt + `geometry` | `elsarticle` preprint 12pt | Elsevier/Neurocomputing template |
| Front matter | manual `\maketitle` | `\begin{frontmatter}` with `\ead`, `\address`, abstract, keywords | Elsevier style |
| Proposition numbering | Roman section (`IV.1`) | Arabic section (`4.1`) | matches `article`/`elsarticle` section counters; avoids mixed I/1 styles |
| Journal field | comment only | `\journal{Neurocomputing}` | explicit target |
| Keywords | online learning; SSM; continual learning; timescale coverage; homeostatic regulation | online learning; state-space models; continual learning; **reservoir computing**; homeostatic regulation | align with Neurocomputing readership |
| Compile entry | `PAPER_D_AI.tex` | `PAPER_D_NC.tex` | new file; originals kept |

---

## 2. Content fixes carried from the pre-submission review

### 2.1 Abstract — compressed (330 → ~210 words)
> **Corrected after audit (2026-09-11):** measured counts are 330 (AI) → 207
> (NC, post-audit revision; 203 before it). The original "~210" was an estimate,
> not a measurement.
- **Where:** `\begin{abstract}...\end{abstract}`
- **What:** Removed dense per-seed numeric laundry list; kept the three host requirements, the four-domain headline win, M5 two-domain/four-domain split, and the CPU-scale disclaimer.
- **Numbers removed from abstract but retained in body:** 58–116, 15.0, −2.05/−1.20, −1.81, 5/5 vs 0.6/5, −2.25/−4.47, −9.28/−10.08, −1.68.
- **Why:** Springer/Elsevier abstracts of this length risk desk pushback; Neurocomputing abstracts are typically ~150–250 words.

### 2.2 Introduction
1. **Opening reframed for Neurocomputing** (new first paragraph): online adaptation under domain shift; SSM + reservoir lineage; “which mechanisms must be native.”
2. **M2 gap closed** (after REDEM mechanism list):
   - Added: M2 is the reward-gated plasticity of `redemb`, not required by the RLS-only constraint, not instantiated here.
   - Why: reviewers reading Paper D alone were confused by the M1/M3/M4/M5 numbering.
3. **Contribution bullet 3 tightened:**
   - Was: homeostat “restores the full-state EMA as a valid domain statistic” (no four-domain caveat in the bullet).
   - Now: explicitly *two-domain task only*; four-domain degradation + exclusion cross-referenced to §P4.
4. **Closing contribution claim:** added sentence that this is a **host-design study in the reservoir/online-readout lineage**, not a new sequence architecture.
   - Why: Neurocomputing reviewers otherwise read “architecture” as a new network claim.

### 2.3 Related work — restructured and expanded
- **Was:** one long paragraph.
- **Now:** three labeled paragraphs:
  1. State-space sequence models and online adaptation
  2. Reservoir computing and fixed-substrate readouts
  3. Continual and online learning practice
- **Added positioning:** bare M1 arm plays the fixed-substrate+online-readout role relative to classical ESN; no separately wired random reservoir is claimed.
- **No new citations invented** — only regrouped existing ones (`jaeger*`, `maass`, `verstraeten`, `lukosevicius`, `parisi2019`, etc.).

### 2.4 Methods (mechanism-level clarifications)
- **M5 subsection:** added closing sentence that M5 is a two-domain mechanism result and is **not** part of the benchmark stack.
- **Tasks/discipline:** unchanged scientifically.
- All equations `eq:recur`–`eq:m5` byte-identical in meaning.

### 2.5 Results
| Location | Fix |
|---|---|
| Fig. 2 caption | `Sec.~IV.B` / `Sec.~IV.A` → `Section~\ref{sec:p2}` / `Section~\ref{sec:p1}` |
| Fig. 2 caption wording | “gating-only (A2)” → “pause-learning (A2)” for consistency with body |
| Table 1 caption | “gating-only (A2)” → “pause-learning (A2)” |
| Table 1 body | **no numeric changes**. The claimed "write-time typo `A3@500 (0/0)` → `(0/10)`" was **wrong**: `(0/0)` occurs in no file in the repository (the string exists only in this changelog), and the A3@500 row is byte-identical between the two drafts. The printed `(0/10)` is nonetheless correct — `s20` gives `stream_improved_n = 0` of 10 seeds |
| §P3 closing | added sentence: two-domain M5 results do not transfer to four-domain benchmark |
| Body numbers (all sections) | carried over verbatim from `PAPER_D_AI.tex` |

### 2.6 Discussion
- `gentle structural change beat abrupt` → **`beats`** (subject–verb agreement).
- Requirement (ii): M5 described as *conditional on the task*.
- Negative-results list: added four-domain homeostat; **removed `state regulation` from the “working design” list** so it no longer contradicts P4.

### 2.7 Limitations
- Added explicit limitation: **no separately wired random reservoir (ESN) baseline**; bare M1 occupies that role.
- M5 scope wording kept (stability-edge analogy, not validation of companion homeostat).
- Other limitations unchanged.

### 2.8 Conclusion
- **Was:** working design included “a state-norm homeostat”; next steps had redundant “larger-scale … and scaling”.
- **Now:** working design = input-path RLS + fast-channel metadata + soft routing + dormant-covariance refresh; M5 is a **conditional safeguard** (two-domain yes, four-domain excluded).
- Next steps: larger-scale / higher-order text, learned selectivity, **classical reservoir comparison** (replaces redundant “and scaling”).

### 2.9 Bibliography
- Companion items `redema`/`redemb`/`redemc` reformatted from “Author's companion Paper X: …” to
  `Y. Hu, <title>, Zenodo (2026). Companion preprint (Paper X), <DOI>`.
- All other bibitems unchanged.
- Body “Paper~C” phrase retained (in-text narrative); formal refs are now complete.

---

## 3. New companion files

| File | Purpose |
|---|---|
| `Highlights_D_NC.txt` | 5 bullets aligned with NC manuscript (fixes AI Highlights #3 “gating-only is falsified” and #4 missing four-domain exclusion); lengths 75/57/76/64/74 |
| `COVER_LETTER_D_NC.txt` | Elsevier/Neurocomputing cover letter (reservoir + host-requirements framing) |
| `CHANGELOG_PAPER_D_NC.md` | this file |

### Highlights semantic fixes (vs `Highlights_D_AI.txt`)
| # | AI Highlights (problematic) | NC Highlights (fixed) |
|---|---|---|
| 3 | “gating-only is falsified” — **contradicts Paper D body** (A2 improves 10/10); that claim belongs to Paper C | “pause-learning is readout-dependent” |
| 4 | homeostat “restores the domain statistic” with no scope | “helps two domains but is excluded on four” |
| notes | lengths 74/53/63/63/76 **wrong** | verified 75/57/76/64/74 |

---

## 4. What was intentionally NOT changed

1. **No new experiments** (no ESN baseline run, no extra γ grid, no P4 M3/M4 split). Those remain optional rebuttal/future work.
2. **No fabricated numbers, citations, or dataset claims.**
3. **Original AI/REVTeX sources left intact** for rollback.
4. Figures still point to `../figures/paperD_fig{1,2,3}_*.pdf`.
5. Data-csv anchors in prose unchanged.

---

## 5. How to compile

```powershell
cd D:\work\papers\testpynew\paper_d
pdflatex PAPER_D_NC.tex
pdflatex PAPER_D_NC.tex
```

Figures resolve via `\graphicspath{{../figures/}}`.

---

## 6. Residual risks (not fixed here; need new experiments or editor judgment)

1. Still toy-scale; absolute ppl is first-order bigram regime.
2. Still no classic ESN+RLS wired-reservoir baseline (only bare M1).
3. Clipped-ppl diagnostic still not tabulated per arm.
4. P4 still does not ablate M3 vs M4 separately.
5. Companion coupling via s18 protocol remains (self-containment improved but not fully standalone reproduction of Paper C metrics).

---

## 7. Post-audit revision (2026-09-11)

An independent verification pass (full diff, per-seed recomputation of every
printed number, compile-log and citation audits) found the retarget sound but
flagged four issues that this section fixes. **No experimental number, table
cell, equation, label or bibliography entry was changed** — verified by
comparing the numeric-token multiset of the source before and after (the only
new tokens are `1` from `\setlength{\emergencystretch}{1em}` and a line-break
`6` in the CRediT section).

| # | Issue found by the audit | Fix applied |
|---|---|---|
| P0-1 | Root `README.md` contribution table still asserted "gating-only falsified" for **Paper D** — the very claim removed from Highlights #3 as belonging to Paper C and being contradicted by the A2 result | The Paper D cell now states that pause-learning *improves* stream on the RLS readout (10/10) and that the companion gating-only falsification does not carry over |
| P0-2 | Compressing the abstract dropped the AI draft's "untuned reference" qualifier, leaving two unqualified "beats a Transformer+LoRA reference on every seed" claims | Abstract (ii) now reads "the **untuned** Transformer+LoRA reference"; the fixed-gate falsification is retained alongside |
| P1-1 | §4.3 heading still read "the state must be regulated" — the only unconditional state-regulation claim left, contradicting the Introduction/Conclusion rewording and the P4 homeostat negative | Heading is now "P3: gentle wins at the tested point, and state regulation is conditional" |
| P1-2 | Table 1 rows read `A2 gate` while captions/body read "pause-learning (A2)", re-creating the gate/pause conflation the earlier review had flagged | Rows renamed `A2 pause` and the caption now says "the paused-readout policy, not the multiplicative feature gates of Section 4.1" |
| P1-3 | Keywords dropped `timescale coverage` although it remains one of the paper's stated host-design contributions | Restored (6 keywords, within the Elsevier limit) |
| P2-1 | Could not keep `microtype` (elsarticle's default CM is a bitmap set), so the retarget introduced two Overfull hboxes where the AI draft had none | `lmodern` (scalable Type 1) + `microtype` + `\emergencystretch=1em`; compile is now **0 Overfull / 0 Underfull / 0 errors**, 19 pages |
| P2-2 | No CRediT statement (Elsevier requirement) | `\section*{CRediT authorship contribution statement}` added |

**Deliberately not changed in this pass** (recorded so they are not mistaken for oversights):

- `figures/paperD_fig2_routing.pdf`, whose legend still prints `A2 gate-only`.
  Regenerating it requires a FIG-script run; the table caption now disambiguates
  the arm in text, so the figure can be regenerated with the other figure work.

---

## 8. Second revision round (2026-09-11, evening): two claims tested against new data

The audit left three items open: two inherited double-rounding cells and two
sentences with no supporting data. Both of the unsupported sentences were
tested by **measuring the missing quantity** rather than by rewording. In both
cases the measurement came out against the original sentence, so the sentences
were rewritten around the new numbers. Every legacy metric value is unchanged
(verified cell-by-cell, `max|diff| = 0.000e+00`).

### 8.1 Numeric corrections (inherited double-rounding)

| Location | Was | Now | Data |
|---|---|---|---|
| Table 1, A2 @ τ_m=500, Δ_ppl | `−2.23` | **`−2.22`** | `s20` `paired_vs_a1["A2-500"].stream_diff_mean` = −2.224663 |
| §4.4 real text, stream | `−1.27` | **`−1.26`** | `s23` `SSM-REDEM_vs_SSM-bare.stream_diff_mean` = −1.264624 |
| §4.4 real text, forgetting | `−0.47` | **`−0.46`** | `s23` `forgetting_diff_mean` = −0.464648 |

These three were present identically in `PAPER_D_AI.tex` (and `PAPER_D.tex`)
and share one cause: the experiment scripts print 3-decimal values that were
then rounded a second time to 2 decimals.

### 8.2 "M3 tracks all five known switches" — falsified by measurement, sentence rewritten

The sentence had no supporting data. `scripts/s20_ssm_m3_routing.py` was given
additive per-switch instrumentation (`switches_detected`, `n_switches` columns;
`DETECT_WINDOW = 200` tokens after each known switch) and re-run over the full
90-run grid. Regeneration fidelity was checked against the committed revision
(`git show HEAD:`): **90/90 rows and all 360 metric cells are identical**, only
the two new columns are added. Measured mean detections of the 5 known switches
(10 seeds; the detector is shared by A2 and A3, identical counts):

| τ_m | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| detected / 5 | **2.80** | 0.30 | 0.00 | 0.00 |

The detector does **not** track the switches. The claim is replaced by the
measured counts, plus the finding that emerged from them: the routing benefit
is bounded by the detector. Seeds with ≥1 detection gain **−1.99** ppl on
forgetting versus **−0.98** for seeds with none, and the forgetting gain is
absent at exactly the two τ_m values where the detector never fires. The
"routing transfers" lesson of Paper C therefore holds only as far as the
detector reaches.

### 8.3 "dormant-covariance refresh alone" — misattribution, and the first isolation attempt was itself wrong

No committed arm isolated dormant-covariance refresh: the `+1.55 → −3.54` change
compared s20's hard routing with a *fully frozen* inactive readout against s21's
soft routing with *both specialists trained*. An isolation arm was added
(`A3-soft-frozenW`, same softmax routing, weights frozen, inverse covariances
refreshed each token) behind `s21_ssm_m4_m5.py --frozen-p-probe`, writing
`data/s37_dormant_p_probe_v1.json`.

**Correction to the first attempt.** The first version of that arm was buggy: it
called `rls_update` unconditionally and then applied an extra `rls_refresh_P`, so
it *trained* the weights and merely added a redundant covariance update. It
reported −3.16 vs the bare host and +0.38 worse than A3-soft, and those two
numbers were briefly written into §3.4/§4.2/§4.3. A referee re-read caught it
(two independent tells: no code path froze `W`, and a genuinely frozen arm must
score the uniform prior 32.0, not 8.6). The arm now branches before the weight
update, and `assert_W_still_frozen` compares `W` against its initial value every
run — it fired immediately on the second latent bug this exposed (`W0`/`W1` were
initialised in a branch that did not include the new arm).

**Corrected measurement** (10 seeds, `s37` `stream_vs_a3soft`/`stream_vs_a1`):

| Arm | stream ppl | forgetting ppl | vs bare host |
|---|---|---|---|
| A3-soft (both specialists trained, committed) | **8.22** | 6.40 | −3.54 (10/10) |
| A3-soft-frozenW (covariances refreshed, weights frozen) | **32.00** (every seed) | 32.00 (every seed) | **+20.25** (0/10) |

32.0 is exactly the uniform-prior cross-entropy for a 32-symbol vocabulary:
with the weights frozen the readout never leaves its initialisation, so
**refreshing the inverse covariances alone learns nothing**. The entire stream
gain belongs to training the specialists. Consequences now applied: the
isolation paragraph states the chance-level result, "dormant-covariance refresh"
is **removed** from the working-design list in Discussion and Conclusion (it is
not a component that carries any result), §3.4 no longer claims it separates a
bundled effect, and the §3.4 subsection title drops the term.

### 8.4 Figure legend

`scripts/gen_paperD_fig2_routing.py` label `A2 gate-only` → `A2 pause`, and
`figures/paperD_fig2_routing.pdf` regenerated so the figure now uses the same
arm name as the text, Table 1 and the other caption.

### 8.5 New/changed artifacts from this round

| Artifact | Change |
|---|---|
| `scripts/s20_ssm_m3_routing.py` | additive per-switch detection instrumentation (`DETECT_WINDOW`, `det_events`, `switches_detected`, `n_switches`); two CSV columns and one `params` note added |
| `data/s20_ssm_m3_routing_v1.{csv,json}` | regenerated; all pre-existing metric values bit-identical, two columns added |
| `scripts/s21_ssm_m4_m5.py` | new `A3-soft-frozenW` arm, `rls_refresh_P`, `--frozen-p-probe` mode, computed-text fixtures (`S37_*`, `S20_DETECTED`) |
| `data/s37_dormant_p_probe_v1.json` | **new** — the dormant-covariance-refresh isolation probe |
| `scripts/gen_paperD_fig2_routing.py`, `figures/paperD_fig2_routing.pdf` | legend label aligned with the manuscript |

### 8.6 Independent re-read of the revised manuscript (same day)

A fresh-eyes audit of the revised version (every number recomputed per seed;
all three figure PDFs re-extracted) confirmed Tables 1–2 are numerically perfect
(all cells, all `N/10` counts, all paired differences, both captions), that all
five numbers revised in §8.1–8.3 are correct at the values now printed, and that
no superseded value survives in the `.tex`, the figures or the PDF. It also
found five further defects, all now fixed:

| # | Defect | Fix |
|---|---|---|
| A | Fig. 2 caption claimed pause-learning "does not" retain specialists below the baseline, but A2 is below it at $\tau_m$=1000 (−0.23, 10/10) and 2000 (−0.37, 9/10) — contradicting Table 1 | Caption now states it fails to at $\tau_m\le500$ and crosses below only at the two slowest timescales |
| B | §3.6 and Limitations stated "the figures show per-arm means without error bars"; Figs. 1 and 3 both pass `yerr=np.std(...)` | Both statements now say Figs. 1 and 3 carry $\pm1$ s.d. error bars and Fig. 2 does not |
| C | Both quoted paired-$t$ values had the sign opposite to the adjacent margin (`t=8.8` beside −1.68; `t=−22.8/−26.5` beside +1.53/+2.90) | Now `t=−8.8` and `t=+22.8/+26.5`, matching the margins and the generator's own ordering |
| D | "the gain is absent at the two $\tau_m$ values where the detector never fires" — at $\tau_m$=1000 the detector never fires yet the gain is −1.20 (10/10) | Rewritten: the gain is *strongest* where the detector fires, survives at 1000 without it, and is nil only at 2000 — i.e. the detector marks where routing helps most, it is not what makes routing work |
| E | §4.2 still opened "The gating-only arm (A2 …)" after the arm was renamed everywhere else | Now "The pause-learning arm (A2 … the paused-readout policy …)" |

### 8.7 Still open after this round (known, not fixed)

- The two paired `t` values (`t=8.8`, `t=−22.8/−26.5`): reproducible by hand
  from the committed CSVs with `ddof=1`, but no script records them and their
  sign is opposite to the adjacent Δ convention.
- The P4 paragraph's flip from `−1.68` (REDEM − arm) to `+8.59` (arm − REDEM).
- Neither affects any reported value; both are presentation/convention issues.
