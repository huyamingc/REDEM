# REDEM — Physics-Grounded Online Learning Architecture

**Last full verification** (fast-path audits, not a full experiment re-run):
`scripts/run_all_audits.py` exit 0; `verify_claims` 121/0; `check_tables`
121 cells / 22 tables / 0 failures; `audit_tex_numbers` FLAG=0;
`provenance_map` OK (root + paper_a..f); `audit_readme` 15/0. Full experiment
re-runs are sampled separately (see `SCALE_AUDIT_REDEM.md`).

This repository contains the complete code, data, and figures for
**six companion preprints** that form one research program on online
learning in physics-constrained systems. The papers share a single
simulation pipeline and data set, cite each other as companions, and are
designed to be read in order (**A → B → C → D**, then the standalone
self-evolution extension **E** and the readout-form companion **F**):

## Paper status

| Paper | Target journal | Status |
| :--- | :--- | :--- |
| A | Chaos, Solitons & Fractals (transferred from *Nonlinear Science*) | **Submitted, under review** — manuscript frozen |
| B | Neural Networks | **Submitted, under review** — manuscript frozen |
| C | Neural Networks | **Submitted, under review** — manuscript frozen |
| D | Neurocomputing | Draft, submission-ready (2026-09-17 revision: companion-quote fix, M4/M5 host-instantiation naming, D/F contribution boundary) |
| E | Neurocomputing | Submission materials prepared (2026-09-17 revision: Theorem 1 conditioned, pipeline-latency wording fixed, flip margin disclosed) |
| F | Neurocomputing | Draft, submission-ready (2026-09-17 revision: abstract novelty boundary, sign-count convention, table references) |

| Paper | Role | Core question | Preprint |
| :--- | :--- | :--- | :--- |
| **A** | Physics | What does the substrate compute? (memory–chaos phase diagram, forgetting kernel, homeostat) | Zenodo [10.5281/zenodo.22109664](https://doi.org/10.5281/zenodo.22109664) |
| **B** | Algorithm | How do you learn on top of it? (REDEM: RLS readout, meta-adaptation, structural plasticity) | Zenodo [10.5281/zenodo.22110606](https://doi.org/10.5281/zenodo.22110606) |
| **C** | Dissection | Which mechanism does which job? (statistical memory ≠ robustness recovery) | Zenodo [10.5281/zenodo.22110618](https://doi.org/10.5281/zenodo.22110618) |
| **D** | Architecture | What host makes these mechanisms native? (state-space-native REDEM) | Zenodo [10.5281/zenodo.22110623](https://doi.org/10.5281/zenodo.22110623) |
| **E** | Self-evolution | Can the substrate detect and repair its own failures under ±1 reward? (sign-flip correction, reconstructed-label capacity sensing, snapshot-gated memory) | Zenodo [10.5281/zenodo.22888172](https://doi.org/10.5281/zenodo.22888172) |
| **F** | Readout form | Which selectivity pressure does a calibrated (clip-free) readout still need? (trained softmax, top-k, expert routing) | Zenodo [10.5281/zenodo.22888049](https://doi.org/10.5281/zenodo.22888049) |

In one sentence each: **Paper A** establishes the physics (what the
substrate computes); **Paper B** builds the learning architecture on that
physics; **Paper C** dissects the mechanisms by falsifying transfers and
locates the host boundary; **Paper D** redesigns the host so the mechanisms
are native rather than retrofitted, motivated by C's boundary result;
**Paper E** extends the program to autonomous self-correction — the
substrate detecting, repairing, and remembering its own failures from a
sign-only reward, self-contained in its own directory with frozen
dependencies; **Paper F** is D's companion on the readout form —
replacing the squared-loss / clip-floor instrument with a trained
softmax, then asking which sparse-selectivity pressure the clean metric
still needs.

## Papers

- **Paper A — Physics**: *"Memory and chaos in a physics-constrained
  relaxation substrate: phase diagram, multi-timescale forgetting, and
  disturbance robustness"* — substrate characterization (target:
  *Chaos, Solitons & Fractals*)
  → [`paper_a/PAPER_A.pdf`](paper_a/PAPER_A.pdf) |
  [`paper_a/PAPER_A.tex`](paper_a/PAPER_A.tex) |
  [`paper_a/README.md`](paper_a/README.md) |
  Zenodo [10.5281/zenodo.22109664](https://doi.org/10.5281/zenodo.22109664)

- **Paper B — Algorithm**: *"REDEM: Online Learning with Meta-Adaptation and
  Structural Plasticity for Non-Stationary Environments"*
  — the online learning architecture (target: *Neural Networks*)
  → [`paper_b/PAPER_B.pdf`](paper_b/PAPER_B.pdf) |
  [`paper_b/PAPER_B.tex`](paper_b/PAPER_B.tex) |
  [`paper_b/README.md`](paper_b/README.md) |
  Zenodo [10.5281/zenodo.22110606](https://doi.org/10.5281/zenodo.22110606)

- **Paper C — Dissection**: *"Dissecting Online Learning Mechanisms:
  Statistical Memory, Homeostatic Recovery, and Substrate Physics are
  Non-Substitutable in the Directions and Conditions Tested"* — three-mechanism disentanglement built on a
  falsifying transfer experiment (target: *Neural Networks*, same journal as Paper B)
  → [`paper_c/PAPER_C.pdf`](paper_c/PAPER_C.pdf) |
  [`paper_c/PAPER_C.tex`](paper_c/PAPER_C.tex) |
  [`paper_c/README.md`](paper_c/README.md) |
  Zenodo [10.5281/zenodo.22110618](https://doi.org/10.5281/zenodo.22110618)

- **Paper D — Architecture**: *"REDEM-SSM: A State-Space Architecture with
  Native Online Learning, Meta-Adaptation, and Structural Plasticity"* —
  a native SSM-hosted architecture instantiating M1/M3/M4/M5 from the
  ground up, motivated by Paper C's host-boundary result (desk-rejected by
  *PRX Intelligence*: scope mismatch — the paper is CS/ML, not
  physics-advancing). Intermediate AI draft kept for rollback; **current
  submission = Neurocomputing** (`elsarticle`), with P4 M3/M4
  factor ablation and a classical ESN+RLS baseline. The 2026-09-17
  revision fixes the companion quote (Paper C explicitly did not test a
  frozen-feedforward host), states the M4/M5 host-instantiation naming
  convention, adds the D/F contribution-boundary attribution sentence,
  conditions the Proposition-1 wording, and corrects the clip-fraction
  range to the data ([0.07%, 11.3%])
  → [`paper_d/PAPER_D.pdf`](paper_d/PAPER_D.pdf) |
  [`paper_d/PAPER_D.tex`](paper_d/PAPER_D.tex) |
  [`paper_d/README.md`](paper_d/README.md)
  Zenodo [10.5281/zenodo.22110623](https://doi.org/10.5281/zenodo.22110623)

- **Paper E — Self-evolution**: *"Self-evolution under ±1 reward:
  sign-flip correction, reconstructed-label capacity sensing, and
  snapshot-gated memory in a recurrent relaxation reservoir"* — the
  autonomous correction stack: direction detection and sign-flip
  correction, a reconstructed-label capacity sense, snapshot-gated
  frozen-hypothesis memory with divergence protection, the multi-level
  expansion chain, the measured D2 timescale bound, content–rendering
  separation, multi-source validation, and an external EWC baseline
  (target: *Neurocomputing*; submission-ready elsarticle
  version). **Current manuscript = v3** (2026-09-10 stats/positioning
  repair + 2026-09-13 meta/clean-P0 wording): v3 corrects two false
  significance statements against the committed per-run data (cross-family
  ring gain `t=2.45 > 2.262`, `p≈0.037`; three of four per-segment memory
  gains significant), rewrites the Introduction positioning, tightens the
  Abstract (283 words on one counting convention; Neurocomputing states no
  abstract word limit), adds **EWC**
  (`scripts/s65_ewc_baseline.py`, bit-exact pairing to s52: whole-run
  −0.85±0.70 pp, t=−3.87, 0/10; only redistributes revisit retention,
  while frozen-snapshot memory gains +8.70±2.92 pp with no linear loss),
  and aligns title/abstract/cover letter with the retracted
  per-dimension-correction claim (uniform sign flip only); 70 pp
  → [`paper_e/PAPER_E.pdf`](paper_e/PAPER_E.pdf) |
  [`paper_e/PAPER_E.tex`](paper_e/PAPER_E.tex) |
  [`paper_e/README.md`](paper_e/README.md) |
  submission materials (**local only, not committed**):
  `paper_e/COVER_LETTER.docx`, `paper_e/VITAE.docx`,
  `paper_e/Highlights.docx`,
  `paper_e/DECLARATION_OF_INTERESTS.docx`,
  `paper_e/Supplementary_Material_PaperE.zip` (**rebuilt for v3**: manuscript =
  `PAPER_E.tex` (the zip's manuscript copy; rebuilt after the 2026-09-17 revision),
  scripts s39–s65, post-rerun committed data, six figures)
  (Paper E is self-contained: its scripts `s39`–`s65` import the frozen
  `paper_e/deps/` modules, so it is reproducible from its own folder
  without the A–D pipeline.)

- **Paper F — Readout form** (companion to D):
  *"Learning the Readout Form: Objective-Level Calibration, Sparse
  Selectivity, and Expert Routing on a Diagonal State-Space Host"* —
  Paper D shows the squared-loss / clip-floor metric is load-bearing;
  Paper F *changes the instrument* (trained softmax readout) and then
  asks which selectivity pressure the clean metric still needs (top-k
  gates, expert routing). External Mamba-style selective-SSM baseline
  (s66) loses to Gate-C-topk on the same host/stream/seeds
  (stream 9.03 vs 7.03, 0/10 external better). The external arm's 24,864
  host parameters are **frozen at a random init**, so it trains *fewer*
  parameters than Gate-C-topk (8,224 vs 24,736), not more. Target:
  *Neurocomputing* / TMLR
  (submission-ready draft, 15 pp; 2026-09-17 revision adds the abstract
  novelty boundary, the sign-count convention, table references, and the
  med-column note). Headline instrument change: stream ppl **11.75 → 8.65**
  (ΔCE 0.307 nats/token); Gate-C-topk 7.03; external frozen SSM 9.03.
  → [`paper_f/PAPER_F.pdf`](paper_f/PAPER_F.pdf) |
  [`paper_f/PAPER_F.tex`](paper_f/PAPER_F.tex) |
  [`paper_f/README.md`](paper_f/README.md)

## What each paper contributes to the series

| | Paper A | Paper B | Paper C | Paper D | Paper E | Paper F |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Question** | What can this substrate compute? | How do you learn on it? | Which mechanism does what? | What host makes it native? | How does it fix itself under ±1 reward? | Which readout form / selectivity pressure is still required under a clean metric? |
| **Core result** | κ*∈(25,30); held-out MC +24–53% just before it; forgetting kernel r=0.97; λ-homeostat +7.9–18% | 0.996 vs 0.973 (p<0.0001; N=1024: 0.9970 vs 0.9753); tracks drift where frozen learners never recover | the +32% sequential-recovery gain is the homeostat's, not the metadata's; metadata robustness transfer falsified (0/10 seeds at every τ_m); routing transfers, gating-only falsified | calibrated input path (state-mixture ranking is a clip-floor/calibration property); routing retains specialists; pause-learning does **not** pass the retention rubric once the floor is accounted for; full stack beats bare host on forgetting (−4.47/−4.53, 10/10) and TF+LoRA on reported axes; P4 factor ablation: hard M3 carries forgetting, soft M4 recovers stream; classical ESN+RLS fails out of sample (~98 vs bare 15.4 / lin 12.4) | reward-rate sign hides the direction; credit assignment = rule selection; ~~structural (per-dimension) gain correction required~~ **retracted** (only a uniform sign flip is demonstrated; s47 is a retracted design attempt, **not** a negative result); capacity sense idles/triggers by reconstruction; frozen memory beats re-adaptation; D2 timescale bound verified; decision-level autonomy + minimal operational self (persistence / functional identity / causal use); memory, not expansion, separates content–rendering (s63); majority rule collapses under a wrong majority but validation-driven selection finds the correct source (s64); s42 corrupt-reward is an honest negative (self-deception, not self-detectable); s58c multi-level expansion refuses the cubic rung and memory masks (not prevents) divergence; EWC does not improve the run and only redistributes revisit retention; hosts: E on relaxation reservoir, D/F on SSM, G is a port design (not in E) | trained softmax removes the clip-floor artifact (simplex projection does not); hard top-k is required for selective gates to retain; expert routing cuts forgetting further at stream parity; external selective-SSM baseline loses on stream (0/10); its 24,864 host parameters are frozen at a random init, so it trains fewer parameters than Gate-C-topk (8,224 vs 24,736) — see `paper_f/README.md` |
| **Target venue** | Chaos, Solitons & Fractals | Neural Networks | Neural Networks | Neurocomputing (NC retarget; PRX desk-reject) | Neurocomputing | Neurocomputing |
| **Preprint DOI** | [10.5281/zenodo.22109664](https://doi.org/10.5281/zenodo.22109664) | [10.5281/zenodo.22110606](https://doi.org/10.5281/zenodo.22110606) | [10.5281/zenodo.22110618](https://doi.org/10.5281/zenodo.22110618) | [10.5281/zenodo.22110623](https://doi.org/10.5281/zenodo.22110623) | [10.5281/zenodo.22888172](https://doi.org/10.5281/zenodo.22888172) | [10.5281/zenodo.22888049](https://doi.org/10.5281/zenodo.22888049) |

## Provenance: prior Si₃N₄ pulse-encoding paper

This repository continues the author's prior standalone paper, which serves
as the device-physics foundation ("Paper 0") of the A–D series:

> **Si₃N₄ Shallow-Trap Relaxation for Temporal Pattern Encoding: A
> Systematic Design-Space Analysis** — preprint, Zenodo
> [DOI 10.5281/zenodo.21753791](https://doi.org/10.5281/zenodo.21753791)
> (2026); target venue: *Neuromorphic Computing and Engineering*.

What the prior paper contributes to this repository:

- **Device calibration** — γ = ln 100, τ₀ = 174 µs (Eₐ = 0.55 eV,
  ν = 10¹³ s⁻¹, T = 300 K), α = 0.02, τ-spread CV = 0.20: the parameter
  set every Paper A–D simulation inherits, bit-for-bit.
- **Two legacy scripts** — `shallow_trap_array_simulator.py` and
  `fair_esn_comparison.py` are frozen from the prior project
  ([github.com/huyamingc/Si3N4-Pulse-Encoding](https://github.com/huyamingc/Si3N4-Pulse-Encoding)),
  imported by the current code, and never modified.
- **Scope split** — the prior paper studies a *parallel array of
  independent devices* (quasi-static block coupling, offline Ridge/MLP
  readout); Papers A–D study the *per-pulse coupled recurrent* substrate
  (online RLS readout, chaos homeostat). Paper A §1 states this gap
  explicitly and cites the prior work as `\cite{prior}`; no experimental
  results are shared between the two lines beyond the device model itself.

If the prior paper is accepted at *Neuromorphic Computing and Engineering*,
update the `\cite{prior}` entry in `paper_a/PAPER_A.tex` to the journal
version.

**Two papers, one pipeline — two different stories.** Paper A is a physics /
nonlinear-dynamics theory paper about what the substrate *computes*; Paper B
is a machine-learning paper about how to *learn on* it. They share the same
simulation code and data but ask different questions:

| | Paper A — substrate characterization | Paper B — REDEM online learning |
|---|---|---|
| **Question** | What can this physical substrate compute? | How do you learn on top of it? |
| **Content** | Dynamics theory: memory–chaos phase diagram, forgetting kernel, λ-homeostat robustness (full derivations in Appendix A) | Learning algorithm + benchmarks: online RLS readout, dual-timescale metadata, chaos homeostat, structure plasticity, ablations |
| **Key results** | Order–chaos transition at κ*∈(25,30); held-out memory +24–53% just before it; forgetting kernel M(t)=∫p(τ)e^{−t/τ}dτ matches the measured memory curve (r=0.97); λ-homeostat restores 7.9–18% after disturbances; edge of chaos (λ_target=0) identified as the best-supported target (+25%; main results run at the conservative −0.02) | Tracks drift where frozen batch learners (GRU, transformer) fail permanently; integrated system beats the bare baseline (0.996 vs 0.973, p<0.0001) and matches every ablation (homeostat removal ties; N=1024, 10 seeds: 0.9970 vs 0.9753); metadata is substrate-agnostic — its statistical-memory benefit transfers to a matched ESN (robustness transfer falsified in Paper C); +32% memory after three sequential disturbances; causal audit confirms all mechanisms are causally clean at the operational leak magnitudes tested |
| **Target journal** | Chaos, Solitons & Fractals | Neural Networks |
| **Relationship** | Substrate theory; cites the prior Si₃N₄ pulse-encoding paper for device calibration | Builds on Paper A's substrate theory (cited as the companion in §2) |

All experiments are CPU-only and fully reproducible via the scripts in
`scripts/` (a shared CORE substrate and readout library supports both papers).
Data and figures live in `data/` and `figures/`.

- `README_REDEM.md` — detailed technical README: architecture, headline
  results (S1–S10), script inventory, reproduction commands.

## Repository layout

All paths in this repository are **relative to the repository root**. Run
scripts from the root; do not hard-code machine-specific absolute paths.

```
├── paper_a/     Paper A: substrate characterization (PDF, LaTeX, README)
├── paper_b/     Paper B: REDEM online learning architecture (PDF, LaTeX, README)
├── paper_c/     Paper C: three-mechanism disentanglement (PDF, LaTeX, README)
├── paper_d/     Paper D: native REDEM-SSM architecture (PDF, LaTeX, README; current = Neurocomputing)
├── paper_e/     Paper E: self-evolution (submission-ready elsarticle, frozen deps/, submission materials)
├── paper_f/     Paper F: readout form / calibrated objective + sparse selectivity (draft, companion to D)
├── scripts/     Shared simulation code (CORE substrate, tasks, readouts, figure scripts; s39–s67 = Paper E/F chains; F uses s50–s54, s66–s67)
├── data/        Experiment results (CSV + JSON, 10-seed means)
│   ├── corpora/     Real-text inputs (Gutenberg public domain; see data/corpora/README.md)
│   ├── per_token/   Per-token CE dumps (.npz) for Paper D/F same-token rescoring
│   └── *.csv|json   Script outputs named data/<script_tag>_<experiment>_<version>.*
├── figures/     Publication figures (vector PDFs; a few PNGs duplicate same-basename PDFs)
├── scripts/run_all_audits.py   B-layer audit ORCH (CLAIM/TABLE/STALE/PMAP/RNUM)
├── requirements.txt        Direct dependency list
├── requirements-lock.txt   Pinned versions used for the committed runs
├── MAINTENANCE.md          Ops history (not reader-facing results)
└── *.md         Overview and technical README
```

### Paths and data generation

Scripts locate outputs via `Path(__file__)` / `os.path.abspath(__file__)`
relative to `scripts/`, then write under `data/` and `figures/`. After clone:

```bash
# from repository root
python -m pip install -r requirements.txt
# pin the exact environment used for the committed results
python -m pip install -r requirements-lock.txt
```

| Output | How it is generated (all commands from repo root) |
| --- | --- |
| `data/*.csv` + `data/*.json` | Re-run the matching `scripts/<experiment>.py`; each script writes its own `_v1` (or `_v2`) stem. Full command registry: `README_REDEM.md` (A–D) and the Paper E/F tables above. |
| `data/corpora/*` | Static public-domain inputs; not regenerated by scripts (provenance in `data/corpora/README.md`). |
| `data/per_token/*.npz` | Written as a side effect by host experiments that import `scripts/per_token_io.py` (`save_stream_and_holdout`). Writers include Paper D: `s19_ssm_rls_readout.py`, `s20_ssm_m3_routing.py`, `s21_ssm_m4_m5.py`, `s22_ssm_p4_benchmark.py`, `s23_ssm_p4_realtext.py`, `s33_ssm_p4_m5.py`; Paper F: `s50_paper_f_pilot_nl_readout.py`, `s51_paper_f_learned_gate.py`, `s52_paper_f_soft_route_experts.py`, `s66_external_ssm_baseline.py`, `s67_host_freeze.py`. File pattern: `data/per_token/<script_tag>__<arm>__<extra>__s<seed>.npz`. Committed dumps support F’s same-token paired tests without a full re-run. |
| `figures/*.pdf` | `scripts/gen_*.py` figure scripts (read committed CSVs/JSON only). |
| Paper PDFs | `pdflatex` / `latexmk -pdf` on `paper_*/PAPER_*.tex` from the paper folder (LaTeX intermediates are git-ignored). |

Local-only (git-ignored, never part of the public clone): `.venv/`,
`review_workspace/`, `rerun_backup/`, `rerun_logs/`, journal cover letters /
Highlights / Vitae / declaration forms / supplementary zips, and LaTeX build
artifacts (`*.aux`, `*.log`, `*.spl`, …).

`review_workspace/` is **not part of the repository** (git-ignored): it holds
the local review records, change logs and submission materials (cover letters,
Vitae, `Highlights.docx`, declaration forms, supplementary zips, and superseded
manuscript drafts). Wherever this README names a `review_workspace/` path or a
`.docx`/`.zip`/superseded-draft file, the reference is to local material that a
reader of the public repository will not find; the committed equivalents are
the per-paper `*.tex`/`*.pdf` and the scripts and data under `scripts/` and
`data/`.

Each paper folder contains its own `README.md` (key results, figures and
data anchors, compile instructions, and the reproduction commands scoped
to that paper), so a paper can be read and reproduced independently of the
series. Paper E additionally carries its **frozen dependency set** in
`paper_e/deps/` (the four core modules its scripts import), so it is fully
self-contained and reproducible from its own folder without the A–D
pipeline; its submission materials (cover letter, Vitae, Highlights,
declaration of interests, supplementary zip) live beside the manuscript.

## Scripts

All committed scripts in `scripts/`, typed
(ML > CORE > PAPER > FIG > EXPLORE). The **Paper** column marks which paper
each script serves (A / B / C / D; shared = library or figure used by more
than one paper). Two legacy scripts from the prior Si₃N₄-pulse-encoding
project are kept as shared dependencies (imported by the new code) and are
never modified. `README_REDEM.md` holds the full headline-results registry
(S1–s36) and reproduction commands for Papers A–D.

**Paper E chain (s39–s65).** The self-evolution experiments live in the
same `scripts/` directory but import the **frozen `paper_e/deps/` modules**
(they shadow the shared core), so they are reproducible independently of
Papers A–D. They are enumerated script-by-script in the first table below;
see [`paper_e/README.md`](paper_e/README.md) for the claim→script→data index.

**Paper F chain (s50–s54, s53b, s66).** Readout-form experiments on the
Paper D diagonal-SSM host / stream protocol, enumerated script-by-script
in the second table below.
See [`paper_f/README.md`](paper_f/README.md) for the claim→script→data index.

Paper E script-by-script index (each row: one committed experiment; the
claim-to-script-to-data mapping also lives in `paper_e/README.md`):

| Script | Type | Purpose | Key result |
|---|---|---|---|
| `s39_prediction_reward_test.py` | PAPER | R1 direction test: multi-view prediction as a repair candidate vs the reward-rate sign; oracle upper bound (gold per-pulse label) | probe_directed post-swap 0.898/0.939; oracle 0.975/1.000; RMHL family fails |
| `s40_meta_flip_ablation.py` | PAPER | R1 flip-mechanism ablation: passive mirror-flip vs v-gated probe across margins 0.1/0.2/0.3 | passive_flip first flip block 1039 (latency 39, 10/10, SD 0); probe_directed 1099 |
| `s41_meta_reward_self_tuning.py` | PAPER | R1 meta-reward self-tuning: learned flip value v (init 0.15, beta 0.5) vs fixed margin | v converges to +0.504; stable stream: no flip, bit-identical to no-meta (0.897925/0.938241) |
| `s42_corrupt_signal_test.py` | PAPER | R1/D4 corrupt-reward boundary: reward wrong for 500 blocks | accuracy inside corruption 0.106/0.064; 3.0 vs 1.0 flips — self-deception not self-detected |
| `s43_hypothesis_population.py` | PAPER | R1 hypothesis population: prune/seed tracking vs single mirror-flip | population tracks continuous drift better than a single flip |
| `s43b_synthetic_rotation.py` | EXPLORE | R1 synthetic 2D rotation: hypothesis population on a continuously rotating boundary | population beats single-flip on rotation tracking |
| `s44_regime_zoo.py` | PAPER | R1 regime zoo: mirror-flip sufficiency across 4 regimes x 2 substrates | inversion-class solvable (0.942-0.947); shift/compression collapse (0.485 post / 0.508 steady) |
| `s45_population_r2r3.py` | PAPER | R2/R3 population x capacity sense: does a hypothesis population recover from boundary shifts | population + R2/R3 composition readings |
| `s46_rule_adjudication.py` | PAPER | R1 rule adjudication: is the +/-1-reward credit limit information-theoretic or rule selection? | delta-derived rules 0.989/0.999 (full swap) vs RMHL 0.097 — rule selection, not an information limit |
| `s47_self_correction_gain.py` | PAPER | R1 retracted arm: reward-driven per-dimension gain modulation on the error-driven base | pre-declared prediction not supported; **retracted**, no number adopted |
| `s48_rotation2d.py` | PAPER | R2 smooth rotation: who tracks a continuously rotating boundary | ridge probe 0.93 coupled / 0.82 uncoupled — L2 decodable |
| `s49_capacity_probe.py` | PAPER | R2 capacity probe: single online readout vs supervised ceiling on a quadratic boundary | readout recovers toward the supervised ceiling; expansion needed beyond it |
| `s50_nonlinear_capacity.py` | PAPER | R2/L3 capacity trigger: is a nonlinear 2D boundary linearly decodable | raw circle coupled 0.80-0.88; uncoupled parallel 0.48 linear — the earlier "undecodable" was probe conditioning |
| `s51_quadratic_circle.py` | PAPER | R2 hypothesis generation on the circle: quadratic basis expansion vs linear | expansion recovers the circle; the single basis refuses |
| `s53_auto_basis.py` | PAPER | R2/R3 autonomous basis generation with a well-conditioned capacity sense (top-50 PCA probe) | sense attributes "insufficient" correctly; refuses unsupportable rungs |
| `s55_sphere_capacity.py` | PAPER | R2 dimensional escalation: 3D sphere capacity probe | escalation readings for the L3 curve |
| `s57_dimension_stress.py` | PAPER | R2 dimension stress: completes the corrected L3 stress-response curve (4D) | ball 2D->3D 0.800->0.630; annulus 0.650->0.600->0.540 |
| `s58a_complexity_trigger.py` | PAPER | R3/L3 complexity trigger: boundary complexity (quadratic-surface count) | no collapse, not saturation: 0.544->0.559 (t=2.98, 7/10) |
| `s58d_kernel_capacity_sweep.py` | PAPER | R3 rich-kernel saturation: is the irreducible +4-6pp edge monotone in kernel width | kappa peak 0.1185 (kappa=20) vs 0.0995 (kappa=25); kappa>=30 ~0; N up: 0.034->0.107 |
| `s52_dynamic_memory.py` | PAPER | R5 memory: frozen-hypothesis memory + reward-rate selection vs re-adaptation | memory 0.888±0.014 vs 0.838±0.010; circle revisit +14.9pp (10/10) |
| `s54_memory_boundaries.py` | PAPER | R5 memory boundaries: slot capacity K, regime count M, wall behavior | capacity wall fixed at K; multi-regime readings |
| `s58c_multi_level_expansion.py` | PAPER | R3 multi-level chain: does expansion continue after a first expansion | stage-1 10/10; stage-2 5/10; ladder capped at level 2 — open failure, not certified refusal |
| `s56_joint_self_evolution.py` | PAPER | R3 composition: memory x autonomous basis expansion jointly | joint 0.769 ~ oracle 0.762; orthogonal axes compose |
| `s58b_relative_sense_stress.py` | PAPER | R4 sense reliability: relative capacity sense under regime-switch stress | reliability 0.90 (SEG=500) -> 0.27 (250) -> 0.00 (125/60) — the cliff |
| `s58f_margin_sensitivity.py` | PAPER | R4 margin sensitivity: can a larger REL_MARGIN repair the cliff | cannot: 0.900->0.100 at SEG=500; noise 0.129 vs signal 0.188 |
| `s60_adaptive_window.py` | PAPER | R4 adaptive window: can the sense self-scale its integration window | cannot: SEG=250 0.270 unchanged; window candidates {100,240} |
| `s59_self_tuned_gate.py` | PAPER | R5 self-tuned snapshot gate: learned v_snap vs fixed 0.55 | v_snap 0.252 dynamic vs 0.922 static — the gate learns when to keep experience |
| `s61_cross_family_transfer.py` | PAPER | R5 cross-family transfer: does a frozen hypothesis help a different task family? | ring segment +8.9±2.5pp uncoupled (10/10) but +1.4±1.8pp coupled (5/10) — weak, absent on the deployed substrate |
| `s58e_d2_timescale.py` | PAPER | R6/D2 timescale: meta layer vs environment change speed | D2 band: post 0.486->0.938 over ratio 0.25->2; success 1.0 |
| `s62_adaptive_meta_timescale.py` | PAPER | R6 adaptive tau_2: self-shortened observation scale vs the D2 band | +15.0pp coupled / +13.8pp uncoupled at ratio 2; the band is not softened (pipeline floor) |
| `s63_content_rendering.py` | PAPER | R7 content-rendering separation: same content, different renderers | probe 0.949->0.900 (t=8.38, 10/10); meta n_flips=0 — memory separates content from rendering |
| `s64_multi_source_validation.py` | PAPER | R8 multi-source validation: reward-driven selection of the CORRECT source | minority-good vote 0.684 vs validation 1.000 (t=100.5); drifting 0.681->0.991 (selection fraction 0.991) |
| `s64b_weak_source_reliability.py` | PAPER | R8 follow-up: intermediate-reliability (p=0.70) cells absent from the committed s64 v1 data | minority-weak: selection beats majority +0.128 (10/10); all-weak 0/10 — validation pays only when a reliability ordering exists |
| `s65_ewc_baseline.py` | PAPER | R3 external baseline: EWC on the s52 revisit protocol (bit-exact pairing) | whole-run -0.85±0.70pp (0/10); redistributes revisit retention; memory +8.70pp with no linear loss |
| `s67_host_freeze.py` | ML | F host-freeze policy: random / pretrained / online host (feeds the Paper F README Paper-H gate) | random-frozen host wins: Q1 0/10 vs online; F top-k on frozen host 7.38/7.18 |

Paper F script-by-script index (all on the Paper D host / stream protocol;
claim-to-script-to-data mapping in `paper_f/README.md`):

| Script | Type | Purpose | Key result |
|---|---|---|---|
| `s50_paper_f_pilot_nl_readout.py` | ML | C1-C3 pilot: trained softmax vs post-hoc simplex vs clipped-linear RLS (B-*/Skip-* arms) | CE diff -0.3068 (t=-92.3, 10/10); simplex +0.028 (10/10 worse); clip fractions 0-10.2% |
| `s51_paper_f_learned_gate.py` | ML | C4/C4b: learned feature gate under softmax CE (Gate-C / top-k / L1 variants) | top-k 7.033 stream / 9.201 forget; Gate-C 7.741 but forget 31.075; L1 sparse (6%) but pays 2.37pp retention |
| `s52_paper_f_soft_route_experts.py` | ML | C5: soft vs hard expert routing over online softmax experts on the Gate-C-topk base | soft forget 7.789 (-1.41, 10/10) at stream parity; hard 6.939 with a small stream tax |
| `s53_paper_f_scaling.py` | ML | width sweep N in {128,256,512}, 5 seeds (preliminary) | input path 8.63->7.25 with N; top-k/soft fail at 512 (sparsity, not width) |
| `s53b_paper_f_ksweep.py` | ML | k-sweep at N=512, 3 seeds (preliminary) | k in {16,32} restores the ordering (6.71/6.79 stream); k=128 collapses (16.14) |
| `s54_paper_f_mackey_glass.py` | ML | Mackey-Glass bin-prediction transfer, 5 seeds (preliminary) | top-k/soft 19.9/16.3 vs input path 29.5/29.1 train/holdout |
| `s66_external_ssm_baseline.py` | ML | external selective-SSM / TTT baseline on the identical host, stream, seeds | frozen 9.03/10.38 vs F top-k 7.03 (0/10 external better); joint training destructive |
| `s66_report.py` | EXPLORE | read-only s66 audit helper: arm summary + paired comparisons from the committed CSV | prints the numbers F quotes |
| `s67_host_freeze.py` | ML | host-policy comparison quoted in F's Discussion: random / pretrained / online host x gate modes, 10 seeds, pre-registered Q1-Q4 | pretrained is NOT better than random-frozen (9.58/84.71 vs 9.03/10.38); top-k on frozen host recovers stream (7.38/7.18) but not retention |
| `s51_paper_f_learned_gate.py` (v2) | ML | C4/C4b v2: adds per-channel top-k selection histograms (chan_util_min / gini / never-selected) for the channel-utilization evidence; v1 numbers bit-identical | see F Limitations channel-utilization sentence |

| Script | Type | Paper | Purpose | Key result |
|---|---|---|---|---|
| `recurrent_substrate.py` | CORE | shared | per-pulse contrast-coupled relaxation substrate (numba core); self-test 4/4 | — |
| `shallow_trap_array_simulator.py` | CORE (legacy) | shared | Si₃N₄ shallow-trap device simulator; constants γ/τ₀/gen_tau_vec/preprogram_vec imported by 12 scripts | — |
| `fair_esn_comparison.py` | ML (legacy) | B | matched ESN reservoir class; imported by `baseline_showdown.py` and `esn_metadata_comparison.py` | — |
| `streaming_tasks.py` | CORE | shared | task generators: drift_binary, narma10, mackey_glass, context_switch, regime_switch | — |
| `online_readout.py` | CORE | shared | OnlineRLS, ThreeFactorReadout, ridge_fit, MC/accuracy metrics | — |
| `substrate_recurrence_characterization.py` | PAPER | A | S1: FTLE / held-out MC / separation vs κ sweep (610 runs) | κ* ∈ (25,30); held-out MC +24–53% at the edge |
| `online_readout_streaming.py` | PAPER | B | S2: online RLS vs offline ridge on streaming tasks | acc 0.974–0.982, recover 225–616; MG NMSE 0.0018 |
| `three_factor_online_readout.py` | PAPER | B | S3: reward-modulated Hebbian vs error-gated vs RLS (negative result) | reward-only post-inversion 0.06–0.10 |
| `intrinsic_reward_experiment.py` | PAPER | B | S4: novelty intrinsic reward ablation (negative result) | stream mean ≤ 0.514; never rescues |
| `dual_timescale_metadata.py` | PAPER | B | S5: fast/dual/slow metadata on regime-switch | +1.3–2.1 pp (p < 0.0001) |
| `chaos_regulator.py` | PAPER | B | S6: λ-homeostat under disturbances | held-out MC +7.9–18% |
| `structure_plasticity.py` | PAPER | B | S7: correlation-guided rewiring | gentle (5%) +8–11%; aggressive (20%) −23% |
| `integrated_benchmark.py` | PAPER | B | S8: full system vs ablations; N=1024 confirmation | 0.996 vs 0.973 (p<0.0001); N=1024 (s30, 10 seeds) 0.9970 vs 0.9753 |
| `baseline_showdown.py` | ML | B | S9: vs matched ESN / GRU / tiny transformer (torch CPU); fairness protocol: per-trial torch seeding precedes weight init | REDEM 0.991 vs GRU 0.371 / TF 0.353 (ESN 1.000, honest; z-scored features); MG NMSE GRU 1.80 / TF 0.77 |
| `forgetting_curve_theory.py` | EXPLORE | A | S10: forgetting-kernel theory M(t), Gauss–Hermite validation | measured MC follows M(t), r = 0.97 |
| `esn_metadata_comparison.py` | PAPER | B | S10: metadata transfer to a matched ESN | equalizes: ESN+meta 0.998 ≈ ESN 0.996 ≈ REDEM 0.994 |
| `cv_sweep.py` | PAPER | A | S10: task-level CV sweep (Paper A Supp. Note 1) | uncoupled MC +33% with CV; coupled near-critical falls |
| `s11_disturbance_chain.py` | PAPER | B, C | E3: sequential disturbance chain (3 rounds, 10 seeds; anchor for Paper C's ESN falsification) | regulated MC 8.47 vs fixed 6.41 (+32%) |
| `s12_lambda_target_sweep.py` | PAPER | A | E4: λ_target × CV optimization sweep (5 seeds) | λ_target=0 optimal: +25%/+19%/+5% MC |
| `s13_causal_audit.py` | PAPER | B | O4: causal leakage audit (7 arms, 10 seeds) | all mechanisms causally clean (≤0.02 pp; plasticity-correlation leak implemented, −0.01 pp) |
| `s14_esn_disturbance_chain.py` | PAPER | C | ESN+metadata under the disturbance chain — falsifies metadata robustness transfer (3 arms, 10 seeds) | paired diffs −0.78/−0.76/−0.69, 0/10 seeds positive |
| `s16_tau_m_pressure_test.py` | PAPER | C | τ_m ∈ {200,500,1000,2000} stress test — falsification robust across metadata timescales (10 seeds) | 0/10 seeds positive at every τ_m (~5σ) |
| `s15_controlled_adaptation.py` | PAPER | C | controlled adaptation protocol with known switch instants (10 seeds) | true effect ~10 pulses + variance collapse (p90 76.5→42) |
| `s16b_falsification_stress_test.py` | PAPER | C | probe-protocol stress test (V0/V1/V2, 4 τ_m, 10 seeds) | sign robust ≤1/10 everywhere; magnitude V0 −0.69 → V2 −0.04/−0.01 across τ_m |
| `s5b_controlled_adaptation.py` | PAPER | B | S5 three-arm controlled adaptation re-measurement (2 substrates, 10 seeds) | table-endpoint factor 1.25–2.7 (T200 1.25–1.50, T40 1.54–2.73; Paper B §4.3) |
| `s17_substrate_stress.py` | PAPER | C | ESN substrate stress — equalizer gain at all configs (120 runs) | gain positive at all 6 ESN configs (+0.1 to +1.0 pp) |
| `s18_llm_drift_gate.py` | ML | C | §7 PoC: LLM drift gate (tiny transformer + LoRA, 90 runs, torch CPU) | A3 routing transfers −2.5 ppl (10/10); A2 gate falsified (0/10) |
| `s19_ssm_rls_readout.py` | ML | D | P1/P3a: per-token RLS readout on a hand-rolled diagonal SSM (7 arms, 70 runs, torch CPU) | state arms 58.5–115.9 (0/10) vs B-proj 11.75 (d −3.26, 10/10); oracle 7.25 |
| `s20_ssm_m3_routing.py` | ML | D | P2: M3 EMA metadata + drift detection + routing on the SSM host (A1/A2/A3, 90 runs, torch CPU) | A2 stream −1.52…−2.45 (10/10); A3 forget −2.05/−1.87/−1.20 (10/10, τ_m≤1000) |
| `s21_ssm_m4_m5.py` | ML | D | P3: M4 soft vs abrupt routing + M5 state-norm homeostat (E1/E2, 70 runs, torch CPU) | soft 8.22 vs abrupt 10.03 (−1.81, 10/10); M5 restores EMA detector 5/5, norm 11.3 vs 50.2 |
| `s22_ssm_p4_benchmark.py` | ML | D | P4: 4-domain irregular-switch benchmark (SSM-bare / SSM-REDEM / TF-A1, 30 runs, torch CPU) | REDEM-SSM vs bare −2.25/−4.47, vs TF −9.28/−10.08 (10/10) |
| `s23_ssm_p4_realtext.py` | ML | D | real-text benchmark: two Gutenberg books (Alice vs Dickens), 32-symbol char vocab (30 runs, torch CPU) | REDEM-SSM vs bare −1.26 (10/10), vs TF −5.23 (10/10) |
| `s24_homeo_plasticity_coupling.py` | PAPER | B | M4-M5 coupling loop under the S11 disturbance chain (4 arms, 10 seeds) | coupling does not help: homeostat alone r3 MC 8.47 vs +fixed-churn 6.45 (t=−9.6, 0/10) / +coupled 5.27 (0/10) — rewiring during disturbance is harmful |
| `s25_reward_gated_plasticity.py` | PAPER | B | novelty-reward-gated vs correlation-guided rewiring (4 arms, 10 seeds, S7 protocol) | novelty-guided MC 14.59 vs corr 12.43 (+2.15, t=4.5, 10/10) — intrinsic signals are structure-level tools |
| `s26_ssm_p4_fair_tf.py` | ML | D | fair Transformer references for P4: tuned A1 grid (lr×rank) + 4-adapter A3 routing (9 arms, 10 seeds) | tuning cuts the stream gap to −1.68 (0/10) but collapses forgetting to 62.2; TF-A3 routing retains specialists (−8.17 forgetting, 10/10) yet stream stays 21.77 — mechanisms transfer, the host does not |
| `s27_clip_kappa_fine.py` | PAPER | A | clip-range ablation × fine κ grid (3 topologies × 3 α_max × 13 κ × 10 seeds) | κ* = 25.3/27.4/27.9; invariant to 5× clip widening for lateral_ring/random_graph (coupling-driven chaos), ring_bidir shifts +3.4 (clip contributes) |
| `kernel_coupling_shape.py` | PAPER | A | kernel-coupling shape stability: physical-kernel Pearson r against √MC(k) per operating point (from `substrate_phase_diagram_v2.json`) | r_phys = 0.91–0.99 over κ 15–30 of the three mode-1 topologies (0.98 at random_graph κ=25); effective-median dilation reported qualitatively |
| `s28_causal_audit_chain.py` | PAPER | B | causal leak audit re-run on the S11 disturbance chain (6 arms, 10 seeds) | no leak arm improves recovery (r3 MC deltas +0.00/+0.00/+0.28/+0.24, NS); no-plasticity r3 MC 8.47 = S11 anchor exactly — causal cleanliness survives a non-ceilinged protocol |
| `s30_integrated_1024.py` | PAPER | B | N=1024 integrated replication at 10 seeds (baseline vs full, paired) | full 0.9970 vs baseline 0.9753 (+2.2 pp, paired t=15.3, 10/10) — the scale-up claim now supports a paired test |
| `s31_char_bigram_oracle.py` | PAPER | D | char-bigram oracle on the real-text protocol (full-book vs ref-window fits, 10 seeds) | true first-order ceiling ppl 10.97±0.18; REDEM-SSM 12.07 sits within ~1.1 ppl of it — the "first-order regime" boundary is now quantitative |
| `s32_ftle_noise_robustness.py` | PAPER | A | homeostat with Gaussian noise on every FTLE estimate (5 levels × 10 seeds, S11 chain) | no significant MC degradation up to σ=0.10 (≈100% of estimate scale); settled κ 28.5–28.7 — clipped proportional feedback integrates out estimation noise |
| `s33_ssm_p4_m5.py` | ML | D | M5 state-norm homeostat added to the P4 stack (2 arms, 10 seeds) | M5 is significantly worse (stream +1.53, forgetting +2.90, t=−22.8/−26.5, 10/10) — Δt modulation breaks the Δt=1-calibrated whitening; S22's exclusion of M5 validated |
| `s21_ssm_m4_m5.py --frozen-p-probe` | ML | D | dormant-covariance-refresh isolation (weights frozen, inverse P refreshed; writes s37) | stream/forget both exactly 32.0 (uniform) on all 10 seeds — refresh alone learns nothing |
| `s38_ssm_p4_m3m4_ablation.py` | ML | D | P4 factor ablation: bare / uniform / hard M3 / soft full stack (4 arms, 10 seeds) | Hard M3: +2.90/−5.24 (0/10, 10/10); soft vs hard: −5.14/+0.77 — M3 carries forgetting, soft M4 recovers stream |
| `s40_esn_rls_p4_baseline.py` | ML | D | classical ESN + online RLS on P4 (4 arms incl. one-hot control, 10 seeds) | ESN-128 stream 98.8 vs bare 15.4 / lin 12.4 (0/10 vs both); bare-vs-lin gap is the clip floor (unclipped bare ~11.5) |
| `s34_leak_sensitivity.py` | PAPER | B | leak sensitivity scan on the S28 audit (7 leak configs, 10 seeds); leak_k drives the FTLE leak horizon (s28 hard-codes 400) | 10× FTLE leak still NS at every tested horizon (+0.13/+0.39/+0.21 at horizons 50/200/400, CIs include 0); plasticity 30% future correlation +0.57 mean (7/10; paired t=2.04, 95% CI [−0.06,+1.20]) — audit resolution is bounded, "causally clean" scoped to operational leaks |
| `s35_readout_boundary_probe.py` | PAPER | D | P1 readout boundary probes (10 seeds, s19 host verbatim): full/half-window oracle, skip-vs-proj nested check, fast/slow token decoding, fast-channel next-token readouts vs static-table reference | full-window oracle 31.2 (matches s19, max diff 0.00) vs half-window 17.3 — oracle is window-dependent; skip 18.0 > proj 7.25 (nested violation, 10/10); token linearly decodable from fast channels (τ≤8) at 88.9–99.7% out-of-sample (chance 3.1%), slow channels at chance; yet fast-channel-only direct and two-stage next-token readouts still fail out-of-sample (ppl 68–104 vs static table 13.9–17.4) — "no useful linear map" is false, and no closed-form squared-loss state readout yields calibrated next-token probabilities; the P1 failure is a pooled-readout/metric property, not missing linear information |
| `s36_random_graph_instance_variability.py` | PAPER | A | random-graph instance variability: S1 random_graph sweep repeated on 9 Erdős–Rényi instances (8 fresh + original 777), 10 substrate seeds each (990 runs); reuses substrate_recurrence_characterization.run_single with per-task TOPO_SEED | peak held-out MC 13.47±0.57 (range 12.80–14.35), +48.5%±6.3% over uncoupled; κ*=25 (7/9) or 30 (2/9); original instance 13.91 (+53%) inside the envelope |
| `gen_paperD_fig1_p1_arms.py` | FIG | D | Paper D Fig. 1: P1/P3a state-readout falsification + input-path control; pooled-table ceiling computed from s19 `oracle_ppl` 10-seed mean 7.25 | `figures/paperD_fig1_p1_arms.pdf` |
| `gen_paperD_fig2_routing.py` | FIG | D | Paper D Fig. 2: P2 routing retention + P3 soft vs abrupt | `figures/paperD_fig2_routing.pdf` |
| `gen_paperD_fig3_benchmark.py` | FIG | D | Paper D Fig. 3: P4 benchmark bars | `figures/paperD_fig3_benchmark.pdf` |
| `gen_architecture_schematic.py` | FIG | shared | Paper Fig. 1 schematics (substrate / REDEM, M4↔M5 loop) | `figures/paperA_fig1_substrate.pdf`, `figures/paperB_fig1_redem.pdf` |
| `gen_substrate_phase_diagram.py` | FIG | A | S1 phase-diagram figure | `figures/substrate_phase_diagram_v2.pdf` |
| `gen_s2_curves.py` | FIG | B | S2 figure: drift-tracking curves + final-30% NMSE bars (reads the curves npz and the s2 JSON aggregates) | `figures/s2_online_readout_v1.pdf` |
| `gen_paper_figures.py` | FIG | shared | batch: robustness / metadata / ablation / showdown | `figures/paperA_fig4_robustness.pdf`, `figures/paperB_fig3_metadata.pdf`, `figures/paperB_fig5_ablation.pdf`, `figures/paperB_fig6_showdown.pdf` |
| `gen_paperA_supp_figures.py` | FIG | A | Paper A Supplementary Fig. S1 (CV sweep) | `figures/paperA_figS1_cv_sweep.pdf` |
| `gen_paperC_fig1_kernel.py` | FIG | C | Paper C Fig. 1: slow-trace kernel vs material forgetting kernel | `figures/paperC_fig1_kernel.pdf` |
| `gen_paperC_fig2_recovery.py` | FIG | C | Paper C Fig. 2: post-disturbance MC recovery vs τ_m (s16) | `figures/paperC_fig2_recovery.pdf` |
| `gen_paperC_fig3_llm.py` | FIG | C | Paper C Fig. 3: LLM drift-gate results (s18) | `figures/paperC_fig3_llm.pdf` |
| `gen_paperF_figs.py` | FIG | F | Paper F Figs. 1-2 + graphical abstract (headline arms across C1-C5; clip-floor story) | `figures/paperF_fig1_arms.pdf`, `figures/paperF_fig2_floor.pdf`, `figures/paperF_graphical_abstract.png` |
| `gen_fig_self_evolution.py` | FIG | E | Paper E Figs. 1-6 from the committed CSVs (reliability cliff, margin/kappa/N, D2 threshold, v-snap gate, cross-family, meta timescale) | `figures/self_evo_f1..f6*.pdf` |
| `gen_cover_letter_docx.py` | FIG | submission tooling | docx single-source-of-truth updater: `--print paper_x BASE` dumps the body, `--text-file f paper_x BASE` rewrites it in place (std-lib only) | paper_{d,e,f}/COVER_LETTER.docx, Highlights.docx |
| `per_token_io.py` | EXPLORE | D, F | per-token CE dump for same-token-set rescoring (writes `data/per_token/*.npz`; F's paired t on same-token CE reads these arrays) | `data/per_token/*.npz` (590 files; regenerate via host scripts listed under *Paths and data generation*) |

Audit / provenance scripts (B-layer; run via `run_all_audits.py`, not mixed into experiment chains):

| Script | Type | Role | Output |
|---|---|---|---|
| `run_all_audits.py` | AUDIT | ORCH — runs the chain below | console `RUN_ALL_AUDITS_EXIT` |
| `verify_claims.py` | AUDIT | CLAIM — paper/README numbers vs `data/*` | `scripts/claims_registry.json`, `verification_report.md` |
| `check_tables.py` | AUDIT | TABLE — claim doc-values matched to tex table cells | `scripts/table_check_report.md` |
| `audit_tex_numbers.py` | AUDIT | STALE/reverse — forbidden wording + light scan | `scripts/tex_number_audit.md` |
| `provenance_map.py` | AUDIT | PMAP — regenerates provenance blocks only | `README.md` + `paper_*/README.md` blocks |
| `audit_readme.py` | AUDIT | RNUM — README prose anchors vs claims | `scripts/readme_audit.md` |

Each FIG script emits vector output (journal submission); the papers
include the extension-less basename so `pdflatex` picks the vector file
automatically. Reproduction commands (S1–s36 shared registry) are in
`README_REDEM.md`; the Paper E and F chains carry per-script tables above,
and every script stem in `scripts/` is named somewhere in this README or a
per-paper README — there are no orphan scripts.

## Provenance rules

Forward: every headline number quoted from the model/program is produced by a
script under `scripts/`, stored in `data/*`, and re-checked by
`scripts/verify_claims.py` (`claim(...)` registrations). Table cells go through
`scripts/check_tables.py`. The generated block below maps each claim to the
expression the audit evaluates.

Reverse: `scripts/audit_tex_numbers.py` scans manuscripts for forbidden stale
wording and counts headline-like tokens; registered claims must appear in at
least one `.tex` string (enforced by `check_tables`; table-anchored claims must
hit a numeric cell inside a `tabular`). Auxiliary coefficients are out
of scope in the MVP.

Exemptions: sourceless or derived-only quantities must be listed in
`verify_claims.py` with a `note=`, not silently omitted from the manuscript.

New-number workflow:
1. Write the number into `paper_*/PAPER_*.tex` (and README only if it is a
   SSOT status/summary token).
2. Produce it: artefact JSON/CSV from the matching experiment script.
3. Register it: `claim(paper=..., section=..., quantity=..., ...)` in
   `scripts/verify_claims.py`.
4. Run `python scripts/run_all_audits.py` — all must exit 0.
5. `provenance_map.py` rewrites the block below; record the round in
   `MAINTENANCE.md`.

Historical narrative lives in [`MAINTENANCE.md`](MAINTENANCE.md) (not in this
file). Do not hand-edit the generated provenance block.

## Code availability

All simulation code and data required to reproduce the results are openly
available at <https://github.com/huyamingc/REDEM>.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The manuscripts in `paper_a/` and
`paper_b/` are the author's preprints: copyright is retained by the author
until journal publication, after which the journals' copyright terms apply.

## Data provenance

<!-- BEGIN PROVENANCE (generated by scripts/provenance_map.py) -->

All papers (A–F). Failures in last run: **0**. Every registered headline quantity maps to a committed artefact expression. Generated by `scripts/provenance_map.py` from `scripts/claims_registry.json` — do not hand-edit it.

- claim call sites: **121**

| paper | section | manuscript quantity | where | produced by |
|---|---|---|---|---|
| A | S10 | forgetting-kernel Pearson r | Paper A | `0.970686` |
| A | S27 | kappa* lateral_ring (amax 0.1) | Paper A | `25.3` |
| A | S27 | kappa* ring_bidir (amax 0.1) | Paper A | `27.4477` |
| A | S27 | kappa* random_graph (amax 0.1) | Paper A | `27.9429` |
| A | S6 | homeostat restore after edge_prune (%) | Paper A / tables | `7.93954` |
| A | S6 | homeostat restore after tau_drift (%) | Paper A / tables | `18.2665` |
| A | S6 | homeostat restore after noise (%) | Paper A / tables | `11.8045` |
| A | E4 | lambda_target=0 MC gain at CV=0.1 (%) | Paper A / Paper B table | `24.5514` |
| B | S8 | full-system overall acc @ N=256 | abstract / README / tables | `0.996211` |
| B | S8 | bare-baseline overall acc @ N=256 | abstract / README / tables | `0.973411` |
| B | S8 | no-homeostat ties full (acc) | Paper B ablations | `0.996222` |
| B | S30 | full-system acc @ N=1024 | abstract / README | `0.997022` |
| B | S30 | baseline acc @ N=1024 | abstract / README | `0.975344` |
| B | S11 | regulated r3 MC after 3 disturbances | abstract / Paper B/C | `8.47495` |
| B | S11 | fixed-kappa r3 MC | abstract / Paper B/C | `6.41455` |
| B | S11 | relative sequential recovery +32% | abstract / Paper B/C | `0.321206` |
| B | S30 | N=1024 full-vs-baseline paired t | Paper B / README | `15.256` |
| B | S25 | novelty-guided rewiring MC_final | Paper B | `14.5854` |
| B | S25 | correlation-guided rewiring MC_final | Paper B | `12.434` |
| B | S24 | homeostat-only r3 MC (no plasticity) | Paper B / S11 anchor | `8.47495` |
| B | S24 | coupled rewiring r3 MC (harmful) | Paper B | `5.27153` |
| B | S2 | online RLS drift-binary mean acc | Paper B / README_REDEM | `0.977939` |
| C | S10 | ESN+meta overall acc | Paper B/C table | `0.997911` |
| C | S10 | ESN-fast overall acc | Paper B/C table | `0.995511` |
| C | S10 | REDEM-full overall acc | Paper B/C table | `0.994233` |
| C | E3-transfer | s14 paired MC diff at r1 (dual-fast) | Paper C | `-0.778506` |
| C | E3-transfer | s14 paired MC diff at r2 (dual-fast) | Paper C | `-0.759189` |
| C | E3-transfer | s14 paired MC diff at r3 (dual-fast) | Paper C | `-0.693147` |
| D | P1 | B-proj / B-lin-clip stream ppl (input path) | abstract / Paper D/F | `11.7541` |
| D | P1 | oracle / pooled-table ceiling stream ppl | Paper D | `7.24781` |
| D | P2 | A3-A1 forgetting diff at tau_m=200 | Paper D | `-2.05268` |
| D | P2 | A3-A1 forgetting diff at tau_m=500 | Paper D | `-1.86947` |
| D | P2 | A3-A1 forgetting diff at tau_m=1000 | Paper D | `-1.20153` |
| D | P3 | A3-soft stream ppl (E1) | Paper D | `8.21824` |
| D | P3 | A3-abrupt stream ppl (E1) | Paper D | `10.0328` |
| D | P3 | M5 regulated whitened-state norm mean | Paper D | `11.3137` |
| D | P3 | bare host whitened-state norm mean | Paper D | `50.1882` |
| D | P4 | SSM-REDEM stream ppl | Paper D tables | `13.1796` |
| D | P4 | SSM-REDEM forgetting ppl | Paper D tables | `8.93112` |
| D | P4-text | REDEM-SSM real-text stream ppl | Paper D | `12.071` |
| D | P4-text | char-bigram full-book ceiling ppl | Paper D | `10.9737` |
| D | P4 | SSM-REDEM vs bare stream diff (ppl, negative=better) | Paper D tables | `-2.24672` |
| D | P4 | SSM-REDEM vs bare forgetting diff | Paper D tables | `-4.47347` |
| D | P4 | SSM-REDEM vs TF-A1 stream diff | Paper D tables | `-9.27882` |
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




















