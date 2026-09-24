# REDEM research program — technical summary (A–F) and honest gap analysis


> **Provenance note (2026-09-17).** The internal working documents cited in some evidence cells below — `self_evolution_mainline.md`, `paper_f/PARADIGM_AND_RULES.md`, `paper_f/PAPER_F_PLAN.md`, and the pre-rename script name `s53b_paper_f_k_sweep.py` — were removed or renamed in the 2026-09-17 cleanup. The claims they supported stand on the committed data files; the removed documents remain retrievable from git history.


Scope: `D:\work\papers\testpynew` as of the working tree (README.md, README_REDEM.md, paper_a–f READMEs/TeX, `scripts/`, `data/`). Every claim below carries a `file:line` anchor; numbers were read from the READMEs/TeX and spot-checked against the committed CSVs/JSON in `data/`. Where the documentation and the data disagree, the data wins and the disagreement is flagged.

One structural fact to hold in mind throughout: **the A–D pipeline is a simulation-only study of a synthetic relaxation substrate at N≈256, and E/F extend it at the same scale.** There is no fabricated device, no GPU, and no model larger than ~1.3×10⁵ parameters anywhere in the program.

---

## A. Per-paper cards

### A — Physics

| | |
|---|---|
| **One-line question** | What does the substrate compute? (memory–chaos phase diagram, forgetting kernel, homeostat) — `README.md:12` |
| **Core mechanism** | Recurrent relaxation substrate with per-pulse, topology-dependent *contrast* modulation of the injection coefficient. Measured with a Benettin paired-trajectory FTLE and Jaeger memory capacity (MC). Analytic multi-timescale forgetting kernel over the log-normal trap spectrum. A λ-homeostat steps κ toward a target FTLE. |

Equations (as implemented in `scripts/recurrent_substrate.py:222-283` and stated in `paper_e/PAPER_E.tex:278-294`):

```
i_j       = exp(gamma * x_j)                       # current ratio, scale-free
g_i       = (mean_{j in nb(i)} i_j - i_i) / i_i    # CONTRAST_SELF (mode 1)
alpha_eff = clip(alpha0 * (1 + kappa * g_i), 0.001, 0.10)
x_i       <- clip( x_i + alpha_eff * (1 - x_i) ) * exp(-pw/tau_i) * exp(-dt_t/tau_i)
```

```latex
M(t) = \int p(\tau) e^{-t/\tau} d\tau              % forgetting kernel (Paper A §3)
```

| | |
|---|---|
| **Headline results** | Order–chaos transition at κ\* ∈ (25,30); held-out MC peaks **+24–53 %** over the uncoupled baseline just before it (`paper_a/README.md:21-23`). Forgetting kernel matches the measured MC curve at **Pearson r = 0.97**; 1/e horizon pinned at ≈16 pulses (`paper_a/README.md:24-27`). λ-homeostat restores **+7.9–18 %** post-disturbance held-out MC (`paper_a/README.md:28-29`). κ\* = 25.3/27.4/27.9 across three topologies in s27 (1170 runs) (`paper_a/README.md:81`). s36: 9 Erdős–Rényi instances × 11 κ × 10 seeds = **990 runs**, peak held-out MC 13.47 ± 0.57, **+48.5 % ± 6.3 %** over uncoupled (`paper_a/README.md:83`). |
| **Statistical strength** | 10 seeds for S1/s27/s32/s36; **5 seeds** for the E4 λ_target sweep (`README_REDEM.md:51`). Mostly paired/descriptive with seed SD; S1 is a 610-run sweep, s36 990 runs. No multiple-comparison correction is mentioned for the κ/CV/topology grids. |
| **Substrate / host** | Numerical Si₃N₄-style shallow-trap relaxation array, N = 256, log-normal τ (τ₀ = 174 µs, CV = 0.20, Eₐ = 0.55 eV, ν = 10¹³ s⁻¹, T = 300 K), γ = ln 100, α₀ = 0.02 (`README.md:128-130`). CPU only. |
| **Claimed vs shown** | The paper's framing is "physics-constrained". What is *shown* is a numerical model whose constants are inherited from a prior device-calibration paper but are not re-measured here. Paper E states this in the plainest terms for the shared substrate: *"The model is physics-inspired rather than device-calibrated: its constants are model parameters of the simulator (representative values for a shallow-trap resistive-switching array), not values measured on a specific fabricated device."* (`PAPER_E.tex:262-265`). Also: the +7.9–18 % homeostat gain is a **substrate-probe** (memory capacity) gain, not a task gain — S6 records that task-level Mackey–Glass NMSE ties with fixed κ (`NEW_ALGORITHM_PLAN.md:300-306`). |
| **Venue / status** | Originally *Chaos, Solitons & Fractals*; **transferred via the Elsevier Article Transfer Service on 2026-09-08** to *Nonlinear Science* (`paper_a/README.md:11`). Zenodo preprint `10.5281/zenodo.22109664` (`README.md:12`). Cover letter, Highlights, declaration and supplementary zip are prepared (`paper_a/` listing). **Under review** (Ms. Ref. NLS-D-26-00781, confirmed 2026-09-23). Not accepted. |

### B — Algorithm (REDEM)

| | |
|---|---|
| **One-line question** | How do you learn on top of it? — `README.md:13` |
| **Core mechanism** | Four mechanisms on the substrate: **M1** online per-pulse RLS readout (λ=0.999, predict-before-update), **M3** per-unit slow EMA metadata, **M4** correlation-guided structural rewiring, **M5** FTLE-based chaos homeostat (`README_REDEM.md:23-27`). |

Equations (`paper_b/PAPER_B.tex:180-234`):

```latex
K_t = \frac{P_{t-1}\phi_t}{\lambda_f + \phi_t^\top P_{t-1}\phi_t},\quad
W \leftarrow W + K_t e_t,\quad
P_t = \frac{1}{\lambda_f}(P_{t-1} - K_t \phi_t^\top P_{t-1})          % lambda_f = 0.999
m_i(t) = (1 - 1/\tau_m) m_i(t-1) + (1/\tau_m) f_i(t),  \tau_m = 200..1000 pulses
\kappa \leftarrow clip(\kappa + \eta\,clip(\lambda_{target} - \hat\lambda, \pm 1), [1,60]),\ \lambda_{target}=-0.02,\ \eta=3
C_{ij} = corr(f_i, f_j); prune 5% lowest |C|, grow 5% highest |C|   % every 2000 pulses
```

| | |
|---|---|
| **Headline results** | Full system 0.996 vs bare baseline 0.973 (**p < 0.0001**); N = 1024 replication s30: **0.9970 ± 0.0007 vs 0.9753 ± 0.0044, paired t = 15.3, 10/10 seeds** (`README_REDEM.md:70`, `paper_b/README.md:18-20`). Drift tracking: abrupt class-interval inversion recovered within **225–616 pulses** while frozen GRU/transformer never recover (`paper_b/README.md:21-23`). M3 metadata +1.3–2.1 pp (p<0.0001); controlled re-measurement (s5b) factor **1.25–2.7** (`README_REDEM.md:42,57`). M5 homeostat +7.9–18 % MC. M4 gentle 5 % churn +7.8 %/+11.3 %, aggressive 20 % **−23 %** (`paper_b/README.md:86`). S9: REDEM 0.991 vs **ESN 1.000**, untuned GRU 0.371, transformer 0.353; MG NMSE REDEM 0.0018 vs **ESN 3.6×10⁻⁵** (`paper_b/README.md:88`). |
| **Statistical strength** | 10 seeds throughout; paired t with df = 9; the N=1024 claim originally rested on **3 seeds** in S8 and was upgraded to a 10-seed paired test by s30 (`README_REDEM.md:70`). Negative results with hard numbers: s24 M4–M5 coupling **0/10** (t = −9.6, t = −5.9), s33 M5 in the P4 stack **10/10 worse** (t = −22.8/−26.5), s34 plasticity leak dose–response (30 % future correlation +0.57 mean, 7/10, paired t = 2.04, 95 % CI [−0.06, +1.20]) (`README_REDEM.md:65,73,74`). |
| **Substrate / host** | Same recurrent relaxation substrate (N = 256; N = 1024 in s30). Baselines: matched ESN-256-hetero (`fair_esn_comparison.py`), GRU-64, tiny transformer d=64 (`baseline_showdown.py:64-81`). CPU/torch. |
| **Claimed vs shown** | "Training == inference" online learning is genuinely demonstrated. But the *comparative* claim is narrow: the win over neural baselines is against an **untuned** frozen GRU/transformer, and a matched ESN beats REDEM on both standard tasks (`paper_b/README.md:28-31`: *"competitive with a well-tuned ESN on standard tasks (drift accuracy 0.991 vs. 1.000; Mackey–Glass NMSE 0.0018 vs. 3.6×10⁻⁵); the differentiated value is the mechanism set, not raw benchmark supremacy"*). Also, in the S8 ablation the homeostat removal **ties exactly** (no_homeostat 0.996/3p vs full 0.996/3p, `NEW_ALGORITHM_PLAN.md:324-329`), so the headline 0.996-vs-0.973 gap is carried by metadata + plasticity, not by M5. |
| **Venue / status** | *Neural Networks* (Elsevier) (`paper_b/README.md:12`); **submitted, with the editor, not yet sent for peer review** (2026-09-23). Zenodo `10.5281/zenodo.22110606`. Submission materials present. Not accepted. |

### C — Dissection

| | |
|---|---|
| **One-line question** | Which mechanism does which job? (statistical memory ≠ robustness recovery) — `README.md:14` |
| **Core mechanism** | Falsifying *transfer* experiments: transplant each mechanism onto a host that lacks it, and see which capability fails to come along. M3 metadata into a matched ESN; τ_m pressure test; probe-protocol stress test (noise injected before vs after the slow trace); controlled-adaptation protocol with known switch instants; a tiny-transformer + LoRA proof of concept (s18). |
| **Headline results** | The substrate's **+32 %** sequential-recovery gain is the **homeostat's**, not the metadata's: ESN+metadata under the same disturbance chain gives paired diffs **−0.78/−0.76/−0.69, 0/10 seeds positive** (`paper_c/README.md:18-21,85`). τ_m ∈ {200,500,1000,2000}: **−0.701/−0.693/−0.682/−0.678, 0/10 positive at every τ_m** ("~5σ", `README_REDEM.md:54`). Metadata's transferable value is narrower: it denoises state-level corruption (**V0 −0.69 → V2 −0.04/−0.01**, s16b) and accelerates boundary adaptation by **1.25–2.7×** on the substrate, ~10 pulses on the ESN (s15: T40 40.6 vs 49.9/52.7; p90 76.5→42) (`paper_c/README.md:22-29,86-91`). Raw capacity stays a substrate property: **MC 10.19 vs 6.17**. s17: equalizer gain positive at **all 6** ESN configs (+0.1 to +1.0 pp). |
| **Statistical strength** | 10 seeds, paired, sign-consistency reported per cell (0/10, 9/10, 10/10). s17 = 120 runs. Effect sizes are small in absolute terms (≈0.7 pp on an MC number ≈6–10) and the paper says so: the title itself is scoped — *"…are Non-Substitutable **in the Directions and Conditions Tested**"* (`paper_c/README.md:1`). |
| **Substrate / host** | Recurrent relaxation substrate + matched ESN + tiny transformer (s18). |
| **Claimed vs shown** | The falsification is real and pre-registered, and the paper *scopes* it (τ_m sweep, protocol variants). The one inferential leap is attributing the substrate's +32 % to the homeostat: the direct evidence is that metadata does **not** transfer it to the ESN, supported by s28's no-plasticity arm reproducing the 8.47 anchor exactly (`README_REDEM.md:69`). That is strong circumstantial evidence, not a substrate-side M5 knockout. |
| **Venue / status** | Target *Neural Networks* — the most recent commit in the repo is literally `408f9cd Retarget Paper C journal references to Neural Networks`; `paper_c/README.md:12` agrees. **Submitted, with the editor, not yet sent for peer review** (2026-09-23). Zenodo `10.5281/zenodo.22110618`. Not accepted. |

### D — Architecture (REDEM-SSM)

| | |
|---|---|
| **One-line question** | What host makes these mechanisms native rather than retrofitted? — `README.md:15` |
| **Core mechanism** | Native state-space instantiation: a diagonal linear recurrence with a log-uniform timescale spectrum replaces the physical substrate; the readout is per-token RLS on a **calibrated additive input path**; M3 becomes a fast-channel state EMA used for domain routing; M4 becomes soft vs abrupt expert routing; M5 becomes a state-norm homeostat. |

```latex
h_t = A \odot h_{t-1} + B e_{t-1},\quad A = diag(\exp(-1/\tau_i)),\ \tau_i \sim \text{log-uniform}[1,3000]
h^w_t = h_t \odot \sqrt{N(1 - A_i^2)}                 % spectrum whitening, N = 128
x_t   = B e_{t-1}                                      % clean additive input path
```

| | |
|---|---|
| **Headline results** | **P1 (falsification):** pooled state-mixture readouts 58.5–115.9 ppl (**0/10**) vs input-path readout **11.75** (d = −3.26, 10/10); in-sample oracle 7.25 (`paper_d/README.md:150`). **P2:** A3 routing forgetting −2.05/−1.87/−1.20 at τ_m ≤ 1000 (10/10); the A2 pause arm's stream gain *reverses* under the unclipped metric (`paper_d/README.md:151`). **P3:** soft 8.22 vs abrupt 10.03 (−1.81, 10/10) **but +0.76 (0/10) unclipped** — the soft-over-abrupt stream margin "is floor-driven and is **not** claimed" (`paper_d/README.md:152`). **P4:** 4-domain benchmark, forgetting −4.47 reported / −4.53 unclipped (10/10); stream −2.25 is floor-driven and reverses unclipped (+1.45, 0/10); vs untuned TF+LoRA −9.28/−10.08 (10/10) (`paper_d/README.md:154`). **Real text (s23):** Alice vs Dickens, 32-symbol char vocab — forgetting −0.46 reported / −0.33 unclipped (9/10); stream −1.26 → −0.05 unclipped (`paper_d/README.md:155`). s31 char-bigram ceiling **10.97 ± 0.18 ppl**; REDEM-SSM 12.07 sits ~1.1 ppl above it (`README_REDEM.md:71`). |
| **Statistical strength** | 10 seeds; 70/90/70/30/30 runs for s19–s23; paired sign counts everywhere. **s35 is the most important methodological result**: the in-sample oracle is *window-dependent* (full-window 31.2 vs half-window 17.3), `skip` (18.0) beats `proj` (7.25) — a nested violation, 10/10 — and while the current token is linearly decodable from fast channels at **88.9–99.7 %** (chance 3.1 %), fast-channel next-token readouts still fail out-of-sample (**68–104 ppl vs a static table at 13.9–17.4**) (`README_REDEM.md:75`). |
| **Substrate / host** | Hand-rolled diagonal SSM, N_STATE = 128, VOCAB = 32 (`s19_ssm_rls_readout.py:98-99`); the Transformer reference is s18's tiny char LM + LoRA; real text = two Gutenberg books. CPU/torch. |
| **Claimed vs shown** | This paper is the program's most self-critical: it introduced the *metric-sensitivity* disclosure (`PAPER_D.tex` §Metric sensitivity, `tab:sens`; `paper_d/README.md:65-72`). Explicitly: *"All results are CPU-scale proofs of concept; no scaling claims are made."* The Transformer comparison is against an **untuned** reference; the fair-reference study (s26) shows tuning closes the stream gap to **−1.68 (0/10)** while collapsing retention to 62.2 (`paper_d/README.md:156`). |
| **Venue / status** | **PRX Intelligence → desk-rejected (2026-09): scope mismatch** → *Applied Intelligence* (`PAPER_D_AI.tex`, kept for rollback) → current submission version **`PAPER_D.tex` / `.pdf` (31 pp, elsarticle) for *Neurocomputing*** (`paper_d/README.md:13-21`). Zenodo `10.5281/zenodo.22110623`. |

### E — Self-evolution

| | |
|---|---|
| **One-line question** | Can the substrate detect and repair its own failures under ±1 reward? — `README.md:16` |
| **Core mechanism** | A plain error-driven RLS base plus an **autonomous correction stack**: (i) reward-rate *sign* detection and readout-weight flip; (ii) credit assignment as **rule selection**; (iii) design criterion only (not a demonstrated mechanism): an operator whose magnitude **scales with the noisy reward** is a positive-feedback trap in the E[r]≈0 dead zone — the **per-dimension / structural-gain claim is retracted**; (iv) a well-conditioned **relative capacity sense** that decides from its own reconstructed labels whether the representation is insufficient; (v) autonomous **basis expansion**; (vi) **frozen-hypothesis memory** with reward-rate selection, doubling as divergence protection; (vii) a **self-tuned snapshot gate**; (viii) a measured **timescale band (D2)** for the reward-rate-EMA meta layer. Abstract: `PAPER_E.tex:44-69`. |
| **Headline results** | See the full inventory in §D below. Extremes: R3 memory beats continuous re-adaptation **0.888 ± 0.014 vs 0.838 ± 0.010 (+5.0 ± 0.9 pp, t = 16.9, 10/10)**; R4 reliability cliff **0.90 → 0.27 → 0.00**; R5 divergence masking 0.801 vs 0.504; R6 recovery band **[1,2]** with a 60-block pipeline floor; R8 majority fallacy 0.684 vs validation 1.000. |
| **Statistical strength** | n = 10 seeds, paired t with df = 9, 95 % critical value 2.262, sign tests for win counts, sample SD (n−1). Quoted verbatim in `PAPER_E.tex:671-693`. Multiple-comparison exposure is acknowledged for the four per-segment gains (`:933-938`). Two **false significance statements** in v2 were corrected in v3 (`paper_e/README.md:52-61`). |
| **Substrate / host** | N = 256 recurrent relaxation reservoir (`paper_e/deps/recurrent_substrate.py`), 2000 blocks × 20 pulses = **40,000 pulses per run**; readout RLS on a 257/513/769-column block-mean feature ladder (`PAPER_E.tex:547`). CPU/numba. |
| **Claimed vs shown** | The claim is deliberately narrow and is stated in the abstract: *"The contribution is autonomy at the decision level, not accuracy"*; the Discussion formalizes an **operational self** (persistent, functionally identified, causally used state) distinct from autonomy; *"no accuracy advantage over a supervised, label-fed readout is claimed, EWC at matched optimiser does not improve the revisit run, the capacity sense idles on the deployed substrate"* (`PAPER_E.tex:63-68`). **One headline mechanism from the v1 line is retracted in v3**: the "structural (per-dimension) gain correction required" claim is replaced by the weaker statement that the only demonstrated correction is a uniform sign flip `w→−w`, *"which is not a per-dimension gain"* (`PAPER_E.tex:805-811`). **s47 status (authoritative)**: script-level P2 `[REFUTED by own data]`, P3 `[NOT VERIFIABLE]` (`s47_self_correction_gain.py:39-64`); **no number from s47 is cited in the manuscript**; treat as *"retracted design attempt, not adopted"* — **not** as a negative scientific result (no trustworthy measurement entered the paper). Two further v2 significance statements were false and were corrected in v3 (`paper_e/README.md:52-61`). Also: s54 carries committed data whose numbers appear **nowhere** in the manuscript. |
| **Venue / status** | *Neurocomputing*; current manuscript `PAPER_E.tex`/`.pdf` (**68 pp** after autonomy/self expansion, 2026-09-13). Submission materials exist and are re-aligned (cover letter no longer calls s47 an honest negative; Highlights 5×≤85 chars; Vitae; declaration; `Supplementary_Material_PaperE.zip` manuscript entry refreshed, 179 entries). **No Zenodo DOI, no submission record.** |

### F — Readout form (draft)

| | |
|---|---|
| **One-line question** | Must the evaluation instrument itself be learnable rather than prescribed? (`PAPER_F.tex:79-86`) |
| **Core mechanism** | Keep Paper D's host/task/seeds, change what is allowed to be learned: (C1) a **trained softmax readout** replaces squared-loss RLS + clip; (C2) a **post-hoc simplex projection** is shipped as the falsifying control; (C3) measure what the clean metric does to the state path; (C4b) a **hard top-k sparse gate**; (C5) **soft routing over two online softmax experts**. |

```latex
g_t = TopK_k(\sigma(C x_t + b)),\quad \phi_t = [\,h^w_t \odot g_t;\ x_t;\ 1\,],\quad k = 32
\pi_{t,k} = softmax_k(-\|m_t - ref_k\|/T_r),\quad p_t = \sum_k \pi_{t,k} softmax(W_k \phi_t)
```

| | |
|---|---|
| **Headline results** | C1: **−0.307 nats/token** stream CE (ppl 11.754 → 8.648, 10/10, t = −92.3), price **+0.32 ppl retention (1/10)**. C2: post-hoc simplex removes every negative (`neg_frac` 1.582 % → 0) and moves stream by **+0.03 ppl (0/10)**. C3: full state stream-neutral (+0.0029, t = 0.22, 5/10) and retention-harmful (−1.348, 10/10, t = −6.73); 128-d noise control stream 10.271/forget 9.554; 32-d substate stream-positive (+0.061, 9/10). C4: unconstrained gate stream **7.741** (10/10) but forget **31.075** (0/10). C4b: top-k=32 stream **7.033** (10/10) with forget **9.201** (7/10 ≤ Skip-sub). C5: soft-topk forget **7.789** vs single 9.201 (−1.41, 10/10, t = −10.4) at stream **parity** (7.065 vs 7.033, t = 0.68). Full stack vs B-softmax: stream 7.065 vs 8.648, forget 7.789 vs 8.890 (`PAPER_F.tex:261-353`). |
| **Statistical strength** | 10 seeds, paired, sign counts; 30–80 rows per experiment. Two of the six pre-registered propositions **fail or partly fail and are reported as such** (C4 FAIL: retention 0/10; C5 stream is parity, not the pre-registered 7/10 win) (`PAPER_F_sketch.md:136-145`). |
| **Substrate / host** | Paper D's diagonal SSM, **N = 128**, `~4k readout parameters`, one unswept host configuration (`PARADIGM_AND_RULES.md` (removed; git history)). |
| **Claimed vs shown** | The sketch's "Not claimed" list is explicit: *"Prop 1 is beaten; universal continual learning; SOTA vs Mamba/TTT/DeltaNet; large-scale language modeling; E-machinery ported onto this host"* (`PAPER_F_sketch.md:31`). The N=512 failure is reported and diagnosed (k ∝ N was the bug, not width) (`PAPER_F.tex:401-419`). |
| **Venue / status** | `\journal{Neurocomputing}` in the TeX; sketch says *"Neurocomputing or TMLR … Stretch: IEEE TNNLS"* (`PAPER_F_sketch.md:195`). **Locally compiled 13 pp** (`paper_f/PAPER_F.log:481`: *"Output written on PAPER_F.pdf (13 pages, 387886 bytes)"*). README, cover letter and Highlights are present, and `paper_f/` together with the F scripts/data (s50–s67) **are tracked** (commits `3b4fc1c`, `0724e01`, `13c3b03`). Not submitted. |

---

## B. The shared substrate — what `recurrent_substrate.py` actually simulates

It is **both, in a specific and defensible proportion**: a *device-shaped numerical model* whose constants are inherited from a device-calibration paper, wrapped around a *designed* per-pulse nonlinear coupling that was never measured on a device.

### Module docstring (verbatim, `scripts/recurrent_substrate.py:1-39`)

> ```
> Recurrent relaxation substrate with per-pulse topology coupling (REDEM S1).
> =============================================================================
> Type:           CORE
> Paper Section:  New-algorithm project Step S1
> Experiment:     REDEM recurrent substrate core dynamics
>
> Purpose:
>   Generalizes the parallel shallow-trap array (shallow_trap_array_simulator)
>   into a recurrently coupled substrate: the injection coefficient of each
>   unit is modulated every pulse by a topology-dependent contrast against its
>   neighbors' current ratios. This is the per-pulse temporal generalization of
>   the quasi-static alpha_eff coupling in topology_comparison.py.
>
> Physics (per pulse step t with interval dt_t after the pulse):
>   1. Current ratios  i_j = exp(gamma * x_j)   (scale-free observable, in [1, 100])
>   2. Coupling contrast g_i from CSR topology neighbors:
>        CONTRAST_SELF (mode 1): g_i = (wmean(i_nbrs) - i_i) / i_i        (v4 style)
>        CONTRAST_NBR  (mode 2): g_i = (i_i - wmean(i_nbrs)) / wmean(i_nbrs)
>                                 (Phase3 ring_unidir / hub_star style)
>        ADDITIVE      (mode 3): control condition, non-physical ESN-style
>                                 drive: x_i += kappa * wmean(x_nbrs) applied
>                                 after the standard uncoupled update, clipped.
>   3. alpha_eff_i = clip(alpha0 * (1 + kappa * g_i), alpha_min, alpha_max)
>   4. Injection:           x_i = x_i + alpha_eff_i * (1 - x_i)
>   5. Pulse-width relax.:  x_i *= exp(-pw / tau_i)
>   6. Interval relaxation: x_i *= exp(-dt_t / tau_i)
>   7. Clip to [0, 1]
>
>   The two-pass structure (compute all contrasts from the pre-update state,
>   then update all units) makes the step order-independent.
>
> Downstream importers (do not break this API without notice):
>   scripts/substrate_recurrence_characterization.py (PAPER, S1)
>
> Dependencies: numpy, numba (optional), shallow_trap_array_simulator (CORE).
> NOTE: shallow_trap_array_simulator.py is NEVER modified; only imported.
> ```

### The actual update, from the numba core (`scripts/recurrent_substrate.py:252-283`)

```python
# --- contrast-coupled alpha_eff ---
alpha_eff = np.empty(n)
for i in range(n):
    a = alpha0
    if indptr[i + 1] > indptr[i]:                      # coupled unit
        s = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            s += wts[e] * np.exp(gamma_ * x[indices[e]])   # wmean of neighbor current ratios
        i_self = np.exp(gamma_ * x[i])
        if mode == 1:
            g = (s - i_self) / i_self                  # CONTRAST_SELF
        else:
            g = (i_self - s) / s                       # CONTRAST_NBR
        a = alpha0 * (1.0 + kappa * g)
        if a < alpha_min:  a = alpha_min; diag[0] += 1.0
        elif a > alpha_max: a = alpha_max; diag[0] += 1.0
    alpha_eff[i] = a
for i in range(n):
    xi = x[i] + alpha_eff[i] * (1.0 - x[i])            # injection
    xi = xi * decay_pw[i] * np.exp(-dt / tau[i])       # pulse-width + interval relaxation
    if xi < 0.0: xi = 0.0
    elif xi > 1.0: xi = 1.0
    x[i] = xi
```

Physical constants come from the frozen legacy module, not from a measurement in this program (`scripts/shallow_trap_array_simulator.py:31-38`):

```python
kB = 8.617333262145e-5 ; T0 = 300.0 ; Ea = 0.55 ; nu = 1e13
tau0 = 1.0 / nu * np.exp(Ea / (kB * T0))     # = 174 us
I_HRS = 50e-15 ; ON_OFF_RATIO = 100 ; gamma = np.log(ON_OFF_RATIO)
```

### Two facts that materially qualify the "physics" framing

1. **The clip-equivalence and — more importantly — the saturation level.** The Paper E frozen copy records (`paper_e/deps/recurrent_substrate.py:49-57, 113`):

   > ```
   > Measured saturation at the deployed coupling:
   >   CLIP_FRAC_DEPLOYED below records the fraction of unit-pulse alpha_eff
   >   values that sit on a clip boundary at kappa=25 on the deployed random
   >   graph (N=256, alpha_max=0.10). It is high enough that the deployed
   >   coupling is a saturating regime rather than a weakly perturbed one:
   >   g_i >= +0.16 pins alpha_eff at alpha_max and g_i <= -0.038 pins it at
   >   alpha_min. ...
   > ```
   > `CLIP_FRAC_DEPLOYED = 0.39293`

   **At the deployed operating point, ~39 % of unit-pulse injection coefficients sit on a clip bound.** So the "physics-constrained" nonlinearity is, for two fifths of the unit-pulses, a hard saturation rather than a smooth contrast response. The manuscript's own Methods write the update with the *outer* clip applied before relaxation; the code applies the bound to the coefficient before it multiplies `(1 − x_i)`. The frozen copy argues exact equivalence (both relaxation factors ≤ 1, states in [0,1], 50-pulse comparison max abs diff = 0.0).

2. **What the substrate is *not*: fabricated or measured.** `paper_e/PAPER_E.tex:258-265`:

   > \paragraph{Substrate (numerical model).} Every result reported in this paper is obtained from CPU numerical simulation of the model described here: **no device was fabricated for this work, no physical measurement is reported, and every "verification" below is a numerical verification against committed simulation output.** The model is physics-inspired rather than device-calibrated: its constants are model parameters of the simulator (representative values for a shallow-trap resistive-switching array), not values measured on a specific fabricated device.

   The originating plan says the same in Chinese (`NEW_ALGORITHM_PLAN.md:13-15`):

   > 无实验室、无流片条件；**全部权重/状态均为数字模拟**（Digital Model 级）。

**Verdict on B:** it is an *abstract reservoir with a physical parameterization*. The relaxation spectrum (log-normal τ, γ = ln 100, per-pulse injection-then-decay) is device-motivated and inherited bit-for-bit from the prior calibration paper; the *recurrence* (contrast coupling with κ) is a designed control knob, is largely saturating at the deployed κ = 25, and its "chaos" is a property of that designed nonlinearity, not of Si₃N₄.

---

## C. Scale

### C.1 Largest network size N actually simulated

| Host | Largest N | Where | Note |
|---|---|---|---|
| Recurrent relaxation substrate | **1024** | `scripts/s30_integrated_1024.py:51` (`N_1024 = 1024`), 20 runs, 10 seeds, 91.8 s/run | Also S8's 3-seed N=1024 confirmation (`integrated_benchmark.py:257-263`) |
| Recurrent substrate, all other work | **256** | `N_UNITS = 256` in ~50 scripts — *every* `N_UNITS = <n>` assignment in the repo is literally 256 (`substrate_recurrence_characterization.py:73` … `s65_ewc_baseline.py:119`) | Every s39–s65 Paper-E script is deployed at N=256 |
| Paper E's largest N | **512** | `s58d_kernel_capacity_sweep.py:115,126` (N ∈ {64, 256, 512}, "kernel width") | Paper E never runs N=1024 |
| Diagonal SSM host (Papers D/F) | **512** | `scripts/s53_paper_f_scaling.py:57` (`N_LIST = [128, 256, 512]`), 5 seeds; `s53b_paper_f_ksweep.py:51` (`N = 512`), 3 seeds | Primary tables are at **N = 128** |
| Transformer / LM | **d_model 64**, 2 layers, 2 heads, ctx 128, vocab 32, LoRA rank ≤ 32 | `s18_llm_drift_gate.py:56-62`; `baseline_showdown.py:81-84` | The only neural LM in the program |
| Legacy demo (dead code) | 16,384 | `shallow_trap_array_simulator.py:264` | `__main__`-only parallel-array demo; its output `data/square_array_results.csv` is absent from the repo |
| Tiny char LM (Papers C/D) | vocab 32, d_model 64, 2 layers, **ctx 128** | `s18_llm_drift_gate.py:56-61` | The only neural LM in the program |

**Do not be misled by one grep hit.** `scripts/shallow_trap_array_simulator.py:264` loops `for N in [64, 256, 1024, 4096, 16384]` — but that is the frozen *legacy* demo of the prior project: a **parallel array of independent devices** with an **offline sklearn `RidgeClassifier`** readout (`shallow_trap_array_simulator.py:11-13`), i.e. "Paper 0", not the REDEM recurrent substrate. No paper A–F claim rests on it.

### C.2 Largest model parameter count

Measured, not estimated. **The largest trainable object in the programme is not a network at all — it is the hand-rolled gate + expert readout of Paper F at its largest swept width.**

| Object | Where | Trainable params |
|---|---|---|
| **Gate `C(512,512)+b` + two 32×1025 experts (s53/s53b, N=512)** | `s53_paper_f_scaling.py:144-145`, `s53b_paper_f_ksweep.py:80-81` | **328,256 trainable** (+16,896 frozen substrate = 345,152 total) |
| **TinyCharLM with 4 LoRA adapters — the largest *torch* model** | `s26_ssm_p4_fair_tf.py:74,235` (`extend_adapters(model, N_DOMAINS=4, rank=16)`) | **177,696 total** (100,384 trainable) |
| TinyCharLM, 2 adapters (s18/s22/s23) | `s18_llm_drift_gate.py:56-62` | **128,544 total** / 51,232 trainable (A3) / 14,368 (A1–A2) |
| TinyTransformer (Paper B baseline) | `baseline_showdown.py:81-84,109-119` | **83,521** |
| ESN-256 (matched reservoir) | `fair_esn_comparison.py:31,48-58` | **66,048** random + sklearn ridge readout |
| GRU-64 (Paper B baseline) | `baseline_showdown.py:80,98-102` | **12,929** |
| Paper E's entire learnable object | RLS readout over the **257 / 513 / 769-column** block-mean ladder (`PAPER_E.tex:547`) | ~10²–10³ readout weights |

The one large *array* in Paper E is the RLS inverse covariance `P`, not a model: at N=1024 the metadata arm needs `F = 2049`, so `P = 2049² = 4,198,401` float64 ≈ 33.6 MB — which the script itself flags as the runtime bottleneck (`integrated_benchmark.py:257-258`).

**Largest model parameter count anywhere in the program ≈ 3.3 × 10⁵ trainable / 1.8 × 10⁵ for the largest network.** That is roughly **4–5 orders of magnitude below "billions"**, and ~10³–10⁴× below GPT-2-124M.

### C.3 Wall-clock cost

Aggregated from the `runtime_s` column of every committed full-run CSV under `data/` (74 files; quick screening runs and `.bak` excluded):

| | |
|---|---|
| **Total recorded per-run compute** | **≈ 356,940 s ≈ 99 CPU-hours ≈ 4.1 days serial single-core**, over **11,550 runs** in 74 committed full-run CSVs (plus 40 `_quick` screening files, 1,267 runs / 11,528 s) |
| **Longest single run anywhere** | **3,216 s = 53.6 min** — `s9_baseline_showdown` `task=mackey_glass, system=gru, seed_idx=1`. Not the N=1024 substrate (s8's N=1024 full run peaks at 2,244 s = 37.4 min) |
| Cheapest per-run | **1.0 s** — s64 multi-source validation (500 runs, 518 s total) |
| Largest single experiment by total | **s9 baseline showdown: 80 runs, 56,935 s ≈ 15.8 CPU-h** (711.7 s/run average) |
| Paper E heaviest | s40 meta-flip 100 runs / 26,680 s; s58f 240 runs / 8,150 s; s58b 240 runs / 7,836 s; s58c 80 runs / 7,204 s |
| Paper D | s19 70 runs / 424 s; s20 90 / 410 s; s21 70 / 249 s; s22 30 / 262 s; s23 30 / 129 s; s18 90 / 894 s |
| Paper F **wall-clock** (`elapsed_s`, the only scripts that record it) | s50 63.1 s / 80 runs; s51 235.0 s / 80; s52 124.4 s / 30; s53 409.1 s / 45; **s53b 870.3 s / 24**; s54 43.3 s / 15 |
| Documented wall clock, long Paper E scripts | *"Full 10-seed runs of the long scripts (s58b ~17 min, s58f ~17 min, s62 ~2.5 min) must therefore be launched one script at a time — two concurrent full runs exhausted the available memory"* (`paper_e/README.md:186-191`) |
| S1 phase diagram | *"610 runs, ~1 min with multiprocessing"* (`README_REDEM.md:135`) |

Recording caveat: there is **no** `wall`/`duration`/`total_runtime` key anywhere in the repo, and ~55 of the result CSVs carry no runtime column at all. The 99 CPU-hours figure is the **sum of per-run `runtime_s`**, i.e. serial-equivalent compute, not wall clock.

Longest single trajectory: **Paper E s59 `static` = 4000 blocks × 20 pulses = 80,000 substrate pulses per run** (`s59_self_tuned_gate.py:142` × `streaming_tasks.py:66`). Runners-up: 60,000 (s58b/s58c), 48,000 (s57/s58a/s58d), 40,000 (≈30 scripts). Paper A S1 = 1200 pulses/trajectory (`substrate_recurrence_characterization.py:75`). Largest real-text run = 18,000 characters (s23 and every s18-protocol script).

### C.4 Real text

**No — only Paper D's two Gutenberg books.** Exhaustive check:

- `data/corpora/` contains exactly two files and a README: `alice.txt` (154,482 B / 144,604 chars) and `dickens.txt` (787,412 B / 757,610 chars). `data/corpora/README.md`:
  > | `alice.txt` | Lewis Carroll, *Alice's Adventures in Wonderland* (ebook #11) | ~148k |
  > | `dickens.txt` | Charles Dickens, *A Tale of Two Cities* (ebook #98) | ~774k |
  > Usage: `scripts/s23_ssm_p4_realtext.py` builds a two-source character-level streaming domain-drift task (alternating windows, known switch instants, s18 protocol) with a **32-symbol vocabulary (31 most frequent chars + UNK)**.
  (The README's sizes are rounded up: 144,604 and 757,610 chars.)
- Tokenisation is `text.lower()` + `Counter.most_common(31)` + UNK (`s23_ssm_p4_realtext.py:88-103`). Total corpus **902,214 characters**, 57 distinct lowercased character types, **7,282 characters (0.81 %) mapped to UNK**; the three non-ASCII survivors are curly quotes. **No BPE, no SentencePiece, no `transformers`, no WikiText anywhere in the repo** (grep for `tokenizer|AutoTokenizer|transformers|torchtext|nltk|load_dataset` over `scripts/*.py` returns zero hits).
- Only two scripts ever open a corpus: `s23_ssm_p4_realtext.py:90-93` and `s31_char_bigram_oracle.py:109-114`. Both are Paper D.
- **Paper C's "LLM" experiment (s18) does not use real text.** Its "two domains" are two synthetic biased-bigram Markov chains (`s18_llm_drift_gate.py:97-140`: `gen_transition(seed, shift, bias)` / `gen_stream`), alternating every 3000 tokens over an 18,000-token stream with vocab 32. The same generator is reused verbatim by s19/s20/s21/s33/s50/s51/s52/s53/s53b.
- Paper F's stream is a "two-domain biased-bigram drift" stream (`PAPER_F.tex:249-250`), also synthetic; s54 is Mackey–Glass binned into 32 integer bins (`s54_paper_f_mackey_glass.py:12`).
- The repo states the scope itself (`s22_ssm_p4_benchmark.py:38-40`): *"Honest scope: no WikiText-103, no scaling claims; the real-text corpus benchmark is deferred (no external text in the repo — user decision needed)."*

So: **sub-word tokenization never appears anywhere; the total real-text corpus is ~0.9 MB of two 19th-century novels rendered as a 32-character alphabet.**

---

## D. Self-evolution claim inventory (Paper E, s39–s65; plus F)

Legend for **Evidence**: **[C]** = controlled experiment with baseline/ablation/control arm and paired seed statistics; **[P]** = pre-registered falsification (criterion fixed before the run, outcome reported either way); **[D]** = demonstration / illustration without a matched control; **[T]** = theory with no new experiment.

### D.1 R1 — direction, rule selection, retracted structural-gain attempt

| # | Mechanism claimed | Experiment | Metric | Number | Evidence |
|---|---|---|---|---|---|
| 1 | Direction information is present but hidden in the **sign of the aggregated reward rate**; a meta-layer that detects the sign and inverts the readout weight recovers the task | s39 `prediction_reward`, s40 `meta_flip` | post-inversion block-vote accuracy | **0.898 ± 0.013** (uncoupled) / **0.939 ± 0.011** (coupled) vs oracle 0.975 ± 0.005 / 1.000 ± 0.000; detection 10/10 seeds; first flip at block 1039 in every seed (39 blocks after inversion at 1000), zero seed SD (`PAPER_E.tex:699-712`) | **[C]** vs an oracle arm — but the paper explicitly warns the s39 `probe_directed` and s40 `passive_flip` rows are *"the same measurement and not two independent confirmations"* (`PAPER_E.tex:728-737`) |
| 2 | The complementary-hypothesis probe is **not load-bearing**; `passive_flip` matches it and detects faster | s40 | blocks to detect | **39 vs 99 blocks** (mainline note (removed; git history)) | **[C]** |
| 3 | The meta layer **learns its own flip policy** online; no hand-set margin; **zero self-harm** on a stable stream | s41 | accuracy / bit-identity | envA 0.899/0.940; learned value **v → +0.504**; stable-stream arm *bit-identical* to the no-meta arm (0.897925 / 0.938241, six decimals) (`PAPER_E.tex:713-717`) | **[C]** |
| 4 | Credit assignment under ±1 reward is a **rule-selection** problem, not an information-theoretic limit: `r=+1 ⟹ y=vote`, `r=−1 ⟹ y=1−vote` | s46 `rule_adjudication` | accuracy of delta-derived rules | R1/R4 **0.989/0.999**; R2/R3 0.536–0.752 (mainline note (removed; git history)) | **[C]** + **[T]** (proof) |
| 5 | ~~Corrections must be **structural** (per-dimension gain)~~ **RETRACTED IN v3 — not a negative result.** Surviving criterion: an operator whose magnitude **scales with the noisy reward** is a positive-feedback trap in the E[r]≈0 dead zone | s47 `self_correction_gain` (script-level P2 **[REFUTED by own data]**, P3 **[NOT VERIFIABLE]**; **no number from s47 appears in the manuscript** — reproducibility index only) | design criterion + retracted arm | v1 asserted "per-dimension gain correction required"; v3 states instead: *"the operative distinction between correction operators is whether their magnitude scales with the noisy reward signal, **not** whether their gain is per-dimension"* (`PAPER_E.tex:171-173`) and *"the only correction our stack demonstrates is the uniform sign flip $w\to-w$, **which is not a per-dimension gain**"* (`:805-811`), labelled *"a design motivation rather than a proven necessity"*; Discussion now names s47 a **retracted design attempt** and forbids citing it as a field-level negative | **[T]** + retracted arm (evidence grade: **not a [C] negative**) — *do not mix with s42/s58c honest negatives* |
| 6 | **D4 signal trust**: a polluted reward drives self-deception and cannot be self-detected | s42 `corrupt_signal` | accuracy | corrupted arms **0.106 / 0.064** (mainline note (removed; git history)) | **[C]** (negative) |
| 7 | 1-D regime family completeness: inversion-class regimes are flip-solvable; translation/compression collapse for the whole RMHL family even though the task is learnable | s43/s44 | post-inversion accuracy | coupled substrate no-meta RMHL at **0.058 / 0.053** (anti-learned) and **0.505** (chance) — span **0.053–0.505** (`PAPER_E.tex:739-748`) | **[C]** (negative) |

### D.2 R2 — representation sufficiency / capacity sensing

| # | Mechanism claimed | Experiment | Metric | Number | Evidence |
|---|---|---|---|---|---|
| 8 | *"Circle is undecodable, basis expansion necessary"* was a **probe-conditioning artifact**; a well-conditioned probe (ridge on top-50 PCA) shows raw features suffice on the coupled substrate | s50/s51 → **s53** `auto_basis` | held-out accuracy | coupled raw-feature circle **0.80–0.88**; uncoupled parallel **0.48** linear / 0.57 quadratic (mainline note (removed; git history)) | **[C]** — self-correction of an earlier claim (a strength) |
| 9 | The **capacity sense idles** on a sufficient substrate and **triggers** on an insufficient one | s53 | trigger counts / repair | **9/10** seeds no false trigger; **10/10** trigger on insufficient; weak repair **0.52 → 0.60** | **[C]** |
| 10 | Dimension is **secondary**; boundary *complexity* (annulus) is the monotone stress axis | s55 → **s57** `dimension_stress` | probe accuracy | ball 2-D→3-D 0.800→0.630, but 4-D balanced ball reads *higher* 0.700; annulus declines monotonically 0.650 → 0.600 → 0.540 | **[C]** — corrects an earlier claim |
| 11 | **L3 is asymptotic degradation, never collapse** on the coupled substrate: complexity saturates above chance, leaving an irreducible ~+4–6 pp edge | s58a `complexity_trigger` | rls accuracy | 4-D shell → double → triple **0.544 → 0.547 → 0.559** (probe within noise) | **[C]** (negative/prediction refuted) |
| 12 | **Rich-kernel saturation**: the margin is monotone in kernel *width* but an **inverted-U of coupling**; capacity peaks at κ\* ≈ 25–30 | s58d `kernel_capacity_sweep` | 4-D-shell margin | κ↓ 0.100→0.022→0.021; N↑ 0.034→0.100→0.107; **κ=40 → 0.008** | **[C]** |

### D.3 R3 — the self-evolution loop (memory × capacity sense × expansion)

| # | Mechanism claimed | Experiment | Metric | Number | Evidence |
|---|---|---|---|---|---|
| 13 | **Frozen-hypothesis memory + reward-rate selection beats continuous re-adaptation** across A-B-A-B revisits | s52 `dynamic_memory` (re-verified by s65's reference arm, 50/50 bit-identical) | whole-run accuracy | **0.888 ± 0.014 vs 0.838 ± 0.010**, paired **+5.0 ± 0.9 pp, t = 16.9, 10/10** (`PAPER_E.tex:922-925`; JSON `mean_acc_all` 0.88802 vs 0.83762) | **[C]** — plus a **population control** (freshly initialized specialists, no snapshot) at 0.839 ± 0.010, indistinguishable from re-adaptation (`:938-942`) |
| 14 | The gain is **segment-specific**, and quoting only the largest is a selection over four segments | s52 | per-segment paired gains | **+4.2 ± 4.8 pp** (t=2.72, 5/10), **+3.6 ± 4.2** (t=2.73, 6/10), **+2.6 ± 4.0** (t=2.02 — *below* the 2.262 critical value, 4/10), **+14.9 ± 4.1** (t=11.5, 10/10, worst seed +10 pp) (`PAPER_E.tex:926-938`) | **[C]** with explicit multiple-comparison disclosure |
| 15 | Hypotheses are identified **functionally** (prediction agreement) not geometrically; forgetting is **capacity-driven**, not time-driven | s52/s54 | policy | weight cosine of the *same* boundary only **~0.1–0.2** in 513-d; but the paper states this is a **development observation, not an ablation** — *"no matched-budget geometric-versus-functional control arm was run"* (`PAPER_E.tex:1257-1272`) | **[D]** — honestly labelled as such |
| 16 | Multi-level **sense → expand → sense-again** chain works; the upgrade is **correctly refused** when the substrate cannot support it; zero-padding preserves memory across upgrades | s58c `multi_level_expansion` | trigger/refusal counts, revisit accuracy | parallel L1 **10/10**, L2 **5/10**, ring **refused 10/10**; linear revisit **0.95 vs 0.90 oracle** | **[C]** |
| 17 | **Memory = divergence protection**: a memoryless RLS *blows up* (weight norm ~2800, max 3265) chasing an unfittable boundary and collapses to chance; frozen-snapshot selection **masks** the identical divergence | s58c | accuracy | **0.504 → 0.801**; the paper stresses *"what this comparison measures is the selection step, not the learning step"* and that memory removes the *consequence*, not the blow-up (`PAPER_E.tex:1276-1292`) | **[C]**, scoped by the authors themselves |
| 18 | Memory and basis expansion **compose** on orthogonal axes | s56 `joint_self_evolution` | accuracy | parallel: pop_auto **0.769** > pop_mem 0.758 > rls_auto 0.737 ≈ oracle 0.762 (mainline note (removed; git history)) | **[C]** |
| 19 | In dynamic environments the capacity sense must be **relative** (candidate vs current basis); an absolute threshold false-triggers at regime boundaries | s56 | false triggers | absolute sense false-triggers at boundaries; relative sense adopted | **[C]** |

### D.4 R4 — bounds of the capacity sense

| # | Mechanism claimed | Experiment | Metric | Number | Evidence |
|---|---|---|---|---|---|
| 20 | **Reliability cliff** governed by window/segment ratio × insufficient-exposure fraction; reliability = (1−P_false)·P_correct | s58b `relative_sense_stress` | reliability | **0.90 (SEG=500) → 0.27 (250) → 0.00 (125/60)**; counts 18/20 correct, 0/20 false at SEG=500 (`PAPER_E.tex:1126-1130`) | **[C]** — measured on the **uncoupled** testbed, stated as such (`:1086-1095`) |
| 21 | Denser exposure **rescues** the cliff (a 3-regime sequence does not degrade reliability, it improves it) | s58b | reliability | ABCABC sequence improves reliability | **[C]** |
| 22 | The cliff is **not tunable by the sense's own margin**: the probe-noise floor overlaps the diluted genuine signal | s58f `margin_sensitivity` | reliability vs margin | SEG=500 reliability **0.900 → 0.600 → 0.100** as REL_MARGIN 0.10 → 0.15 → 0.20; d_best noise 0.129 vs signal 0.187; 0.10 is Pareto-optimal | **[C]** (negative) |
| 23 | The cliff is **not tunable by an adaptive window** either: short clean windows strengthen the delta but their variance blocks the hold requirement | s60 `adaptive_window` | reliability | SEG=250 **0.270 = 0.270**; SEG=500 **0.900 → 0.500** (worse) | **[C]** (negative) |
| 24 | Reliability numbers are **rates over autocorrelated per-window decisions** (trend-level, not sharp), and the product form **assumes independence without verifying it** | — | — | stated in Limitations (`PAPER_E.tex:1872-1875`) | **[T]/caveat** |

### D.5 R5 — the memory layer

| # | Mechanism claimed | Experiment | Metric | Number | Evidence |
|---|---|---|---|---|---|
| 25 | **Self-tuned snapshot gate (G1)**: the last hand-set threshold (SNAP_EMA_MIN = 0.55) is removed; the system learns *when* to keep experience | s59 `self_tuned_gate` | learned gate value; accuracy | v_snap **0.252** (dynamic A-B-A-B; memory useful) vs **0.922** (static; snapshots self-suppress); accuracy 0.876 ≈ 0.885 (dyn) and 0.955 vs 0.946 (static) | **[C]** (two environments) |
| 26 | **Cross-family transfer (G3)** is partial and **structure-matched**; absent on the deployed coupled substrate | s61 `cross_family` | accuracy gain | ring **+8.9 pp** on the *uncoupled* substrate (mem_acc 0.576–0.619); deployed cross-family ring gain **+1.4 ± 1.8 pp**, and here the paper *withholds* the claim: t = 2.45 (p ≈ 0.037, above the 2.262 critical value) but only **5/10 seed wins** and **11.3 %** mean selection fraction (`paper_e/README.md:52-57`) | **[C]** — note this is a **v3 correction of a v2 false statement** |
| 27 | **Content–rendering separation** (s63, R7): a speaker change is a real extra dimension; a single linear readout degrades, quadratic expansion does **not** repair it, frozen-hypothesis memory partially recovers it | s63 `content_rendering` (320 runs) | accuracy | **0.938 → 0.900** (paired t = −18.2, 10/10); quadratic **0.889** (t = −4.7, **0/10**); memory **0.913** (t = +9.3, 10/10) (`PAPER_E.tex:1588-1604`) | **[C]** — and the paper *withdraws* an earlier geometric explanation and replaces it (`:1590-1598`) |
| 28 | The capacity probe **registers but does not trigger** on render-induced insufficiency | s63 | probe accuracy; trigger | held-out 0.9493 ± 0.0145 → 0.9003 ± 0.0250 (paired drop 0.049 ± 0.018, t = 8.38, 10/10); largest per-seed drop 0.084 < the 0.10 trigger margin, so **no trigger** — reported as "detecting the degradation while staying below its trigger threshold", *not* as a correct flag (`PAPER_E.tex:1606-1622`) | **[C]**, explicitly scoped |
| 29 | The observational meta layer is **render-blind** (n_flips = 0 at every switch rate) — scoping D2 to semantic flips, not render identity | s40/s41/s58e machinery in s63 | flip count | **0 flips** | **[C]** |
| 30 | **Multi-source validation** (s64, R8): the majority rule is a fallacy when the majority is wrong; validation-driven selection finds the correct source | s64 `multi_source_validation` (500 rows, 5 scenarios × 5 arms) | accuracy / selection fraction | majority-good: vote 1.000, wm/we 0.938, oracle wt 0.938; **minority-good: vote collapses to 0.684 vs the analytic majority-vote value 11/16 = 0.6875**, validation **1.000** (t = 100.5, 10/10; through the worker t = 61.4); drifting: vote 0.681 vs validation 0.991 (t = 107.4); all-bad: source-consuming arms ≈ **0.49**, EMA selector 242 switches | **[C]** — with two self-imposed caveats: the large t's reflect a **tiny seed SD (0.0099)** because sources are fixed at 1.0 or 0.5, and no intermediate reliability (0.7 vs 0.6) was ever sampled (`PAPER_E.tex:1688-1702`) |
| 31 | **External baseline (s65, EWC)**: the canonical continual-learning competitor does not improve the run at a matched optimiser | s65 `ewc_baseline` (tuned on a 3-seed grid; pairing verified bit-exact vs s52, 50/50, max diff 0.0) | accuracy / retention | whole-run **−0.85 ± 0.70 pp (t = −3.87, 0/10)**; total revisit retention **−0.60 ± 2.02 pp (t = −0.94, n.s., 3/10)**; it only **redistributes** retention: circle **+4.95 ± 4.28 pp (t = 3.65, 8/10)**, linear **−6.15 ± 3.72 pp (t = −5.23, 0/10)**; the frozen-snapshot arm gains **+8.70 ± 2.92 pp of total retention (t = 9.42, 10/10)** with no linear loss; EWC-by-AdaGrad (as published) is optimiser-limited at **−22.3 pp (t = −12.03, 0/10)** and is explicitly *not* read as evidence about EWC (`PAPER_E.tex:944-978`) | **[C]** — verified against `data/s65_ewc_baseline_v1.json:332-398` |

### D.6 R6 — the meta timescale (D2)

| # | Mechanism claimed | Experiment | Metric | Number | Evidence |
|---|---|---|---|---|---|
| 32 | **D2 timescale bound**: a reward-rate-EMA meta layer can only correct changes slower than its own observation scale | s58e `d2_timescale`, **11-point** ratio sweep τ_env/τ₂ ∈ {0.25 … 10} | post-swap accuracy | failing range τ_env/τ₂ ≤ 1 spans **0.486 ± 0.004 → 0.561 ± 0.042**; recovery complete at ratio ≥ 2 (**0.938 ± 0.002**, 0.939, 0.940) (`PAPER_E.tex:1403-1416`) | **[C]** — but reported as a **monotone transition band, not a sharp threshold**, correcting the earlier "sharp threshold at 1–2" |
| 33 | D2 is a **bandwidth statement about the estimator**, not a theorem about the system; the contribution is the **60-block pipeline floor** that binds before the bandwidth does | s58e/s62 + Haykin's tracking-bandwidth limit | — | quoted verbatim (`PAPER_E.tex:1391-1401`) | **[T]** + **[C]** |
| 34 | **The D2 threshold is pipeline-bound (G4)**: τ₂ *can* be adapted (working-end gain) but cannot soften the threshold | s62 `adaptive_meta_timescale` | mean accuracy | ratio-2 mean **+13.4 pp**; ratio ≤ 1 post ≈ 0.49–0.54 | **[C]** (partial negative) |
| 35 | The binding constraint is the **detection timescale**, not value learning: `passive_fixed` tracks `meta_self` exactly | s58e | — | identical trajectories (mainline note (removed; git history)) | **[C]** |

### D.7 Paper F claims (v1.1)

| # | Mechanism claimed | Experiment | Metric | Number | Evidence |
|---|---|---|---|---|---|
| 36 | The clip-floor artifact is a property of the **objective**, not the numeric range | s50 (80 rows, 8 arms × 10 seeds); B-lin-clip ≡ s19 B-proj exactly, max\|diff\| = 0 | stream CE / ppl | **−0.307 nats/token** (11.754 → 8.648), 10/10, t = −92.3; price **+0.32 ppl retention (1/10)** | **[C]** + **[P]** |
| 37 | **Post-hoc calibration does not repair it** | s50 B-simplex / Skip-simplex | ppl | neg_frac 1.582 % → 0 and stream **+0.03 (0/10)** | **[C]**, pre-registered negative control |
| 38 | Under a clean metric the state-path boundary **moves but does not open**; dimension/optimizer excluded | s50 + P-F0b noise/substate arms | ppl | full state stream **+0.0029 (t = 0.22, 5/10)**, retention **−1.348 (10/10, t = −6.73)**; 128-d noise 10.271/9.554; 32-d substate stream **+0.061 (9/10)** | **[C]** |
| 39 | Unconstrained CE does **not** sparsify → retention collapses | s51 (80 rows) | ppl | Gate-C stream **7.741 (10/10)** but forget **31.075 (0/10)** | **[C]**, reported as a **FAIL** of the pre-registered C4 |
| 40 | **Hard top-k sparsity is the missing pressure** | s51 | ppl | Gate-C-topk **7.033 stream (10/10)** / **9.201 forget (7/10 ≤ Skip-sub)**; C4b **PASS**. Caveat added 2026-09-14: the weakly $L_1$-regularised arm is marginally *better* on stream (**6.994**, 6 % of gates open vs 25 % for top-k) at a retention cost (**11.574** mean / **11.136** median), so the claim holds only as "best stream **among the retention-safe** variants" | **[C]** + **[P]** |
| 41 | **Soft expert routing buys retention without giving up stream** | s52 F (30 rows) | ppl | soft-topk forget **7.789 vs 9.201 (−1.41, 10/10, t = −10.4)**; stream **parity** (7.065 vs 7.033, +0.03, t = 0.68) — *"Report as retention-supported, stream-neutral"*, not the pre-registered 7/10 stream win (PAPER_F_PLAN, removed; git history) | **[C]** + **[P]** partial |
| 42 | Scaling: the claim needs a **fixed modest k**, not k ∝ N | s53 (N ∈ {128,256,512}, 5 seeds) and s53b (k-sweep at N=512, 3 seeds) | ppl | at N=512 with k=max(8,N/4): Gate-C-topk stream 9.16, forget 27.3 (**fail**); k=32 restores **6.79 / 9.89**; soft-topk k=16 **6.59 / 6.27**; k=128 collapses | **[C]** |
| 43 | The sparse-gate advantage transfers off the bigram stream | s54 Mackey–Glass bin prediction (5 seeds) | ppl | train/holdout ≈ **19.9 / 16.3** vs B-softmax **29.5 / 29.1** | **[C]**, 5 seeds only |

**Inventory verdict.** Of the 43 entries above, ~39 are controlled experiments with baselines, ablations, controls or pre-registered criteria; only #15 (functional-vs-geometric identity) is an un-controlled development observation, and it is labelled as such in the manuscript. Several entries are *corrections of the program's own earlier claims* — #5 (per-dimension gain **retracted**, not a negative), #8 (circle-undecodable artifact), #10 (dimension axis), #14 (segment selection), #26 (cross-family claim withheld), #27 (geometric explanation withdrawn), #32 (threshold → transition band), #39 (C4 FAIL) — and that self-correction is the program's strongest methodological feature. The weakness is **not** missing controls; it is **external validity**: 1 substrate model, 1 protocol family, 2–4-dimensional synthetic decision boundaries, n = 10 seeds, and exactly one external competitor method (EWC). Two further caveats: (i) entry #5 is a **retracted design attempt** (s47 script-level P2 refuted) and must never be mixed with the controlled negatives s42/s58c; the series README no longer advertises the retracted headline, and (ii) the manuscript now carries an explicit Discussion taxonomy of negative-result grades, while several additional script-level refutations (s43/s43b/s45 population-memory) still never reach the text.

---

## E. Honest gap analysis — how far the evidence is from "self-evolving large model"

### E.1 Token and parameter scale

| Quantity | Program maximum | LLM reference | Gap |
|---|---|---|---|
| Model parameters (largest network) | **177,696** (TinyCharLM + 4 LoRA adapters, s26) | 10⁹–10¹² | **~4–7 orders of magnitude** |
| Trainable parameters (largest object of any kind) | **328,256** (Paper F gate + 2 experts at N=512) | — | — |
| Trainable parameters in the "self-evolving" system (Paper E) | **~10²–10³** readout weights over a 257/513/769-column RLS ladder | — | — |
| Sequence length | 80,000 substrate pulses (s59); 18,000 tokens of text (s18) | 10⁵–10⁶+ | 1–2 orders |
| Vocabulary | **32 characters** | ~10⁵ sub-word tokens | ~3–4 orders; no tokenizer exists in the repo |
| Real text | 2 Gutenberg novels, **902,214 characters** (0.81 % UNK) | 10¹²+ tokens | ~6+ orders |
| Task | two-class block-vote on synthetic 2–4-D boundaries | open-ended generation | not comparable |
| Seeds | 10 (5 in the two scaling studies, 3 in one k-sweep) | — | low power; the paper concedes *"with df = 9 these tests have low power"* (`PAPER_E.tex:689-691`) |

The program's own documents state the ceiling. `paper_c/LLM_EXTENSION.md:23-27`:

> Practical constraint: this project is CPU-only (torch CPU, OpenBLAS 24 threads, no GPU). The repo already runs a tiny transformer baseline on CPU (Paper B S9, `scripts/baseline_showdown.py`), so a small-scale LoRA-style PoC is feasible; **a GPT-2-124M-scale study is not** (CPU fine-tuning of 124M parameters is hours-to-days per run).

### E.2 Real-world embodiment

None. There is no fabricated device, no measured waveform, no hardware loop. Verbatim (`paper_e/PAPER_E.tex:258-262`):

> \paragraph{Substrate (numerical model).} Every result reported in this paper is obtained from CPU numerical simulation of the model described here: **no device was fabricated for this work, no physical measurement is reported, and every "verification" below is a numerical verification against committed simulation output.**

And (`PAPER_E.tex:1859-1863`):

> \paragraph{Limitations.} The demonstration is a CPU numerical study of one physics-inspired substrate model and one protocol family; **no device was fabricated and no measurement is reported, so nothing here is evidence about hardware behaviour**, and the "adaptive experience system" is established in this setting, not as a universal theorem.

The substrate's device constants are the frozen legacy pair `γ = ln 100`, `τ₀ = 174 µs` from the prior paper; the *recurrence* that makes the substrate interesting is a designed per-pulse contrast coupling that was never fabricated. Worse for the physics narrative: at the deployed κ = 25, **39.3 % of unit-pulse injection coefficients saturate against the α clip** (`paper_e/deps/recurrent_substrate.py:49-57,113`), i.e. the "physical nonlinearity" is, for two fifths of unit-pulses, a hard bound.

Paper E's "speakers" in s63 are **affine renders** `A_s` applied to a 2-D semantic point stream — not audio, not people:

> A per-speaker affine render $A_s$ changes only the physical encoding the readout observes (`PAPER_E.tex:1582-1585`).

### E.3 Does "self-evolution" change weights/structure at LLM-comparable scale?

No, and the mechanisms are categorically smaller than "changing a model":

1. **It never touches the substrate/backbone.** `paper_c/LLM_EXTENSION.md:144-152`:
   > The resolution is that REDEM's RLS **never touches the substrate/backbone weights** - it adapts only the READOUT W over the feature vector phi (the reservoir state in this paper; the frozen backbone's activations on a Transformer).
2. **"Structural self-correction" is, in the current manuscript, a *uniform sign flip* only.** The "per-dimension gain" version was retracted in v3: *"the only correction our stack demonstrates is the uniform sign flip $w\to-w$, which is not a per-dimension gain"* (`PAPER_E.tex:805-811`). s47 is a retracted design attempt (script-level P2 refuted), **not** a negative result. The series README no longer lists per-dimension correction as a core result.
3. **"Multi-level expansion" (s58c) is a basis-ladder step** on the readout features: **257 → 513 → 769 columns** (linear → quadratic → cubic), with the whole chain refusing the cubic rung on the coupled substrate (`PAPER_E.tex:547`, `s58c`).
4. **"Memory" (s52) is frozen weight snapshots** of that same small readout, selected by reward-rate argmax.
5. **Capacity sensing** idles on the deployed substrate (**9/10 seeds never trigger**, `PAPER_E.tex:983-985`), so on the deployed system the "self-evolution" loop mostly does nothing.
6. **The autonomy ceiling is a human-supplied rule set.** `PARADIGM_AND_RULES.md` (removed; git history), lines 71-82::
   > > Autonomy in the decision layer is bounded by the breadth of the candidate set a human supplied. With a ±1 channel the gradient to a parameter is not available, so the only latitude left is rule selection. Every additional capability costs another mechanism: E needs a direction detector, a capacity probe, a frozen-hypothesis memory, a self-tuned gate, and a source-selection device.
7. **Paper F's boundary statement** (`PAPER_F.tex:394-399`): *"Rules that make evaluation trustworthy do not generate new candidate families. F learns form *inside* supplied families … The next ceiling is candidate-space *breadth*: a trainable generator of hypotheses, not only a scorer of human-supplied ones."*

### E.4 Is there any LLM-level (billions of parameters) experiment?

**No — and the largest transformer in the program is not even the largest model.** The largest *torch* model is **177,696 parameters** (`s26`, TinyCharLM with 4 LoRA adapters); the largest trainable object anywhere is **328,256** (Paper F's gate + 2 experts at N = 512). The single experiment branded as an "LLM extension" is `scripts/s18_llm_drift_gate.py` — a char-level 2-layer/2-head/d=64 transformer (128,544 params) with LoRA rank 16 on two **synthetic biased-bigram Markov domains**, vocab 32, 18k tokens. Paper C's own §6 headline is careful (`paper_c/README.md:30-33`):

> Transformer proof of concept: slow-trace domain routing transfers (forgetting up to −28%, 10/10 seeds; its stream-perplexity benefit holds at τ_m ≤ 500 and reverses sign at τ_m ≥ 1000) while a gating-only variant is falsified at every τ_m (0/10 seeds).

and the scope is locked in the extension document (`paper_c/LLM_EXTENSION.md:244-246`):

> Scope claim (must be locked in the paper): "the principles instantiate on a Transformer substrate" — **NOT** "this beats Online-LoRA/SLoRA on benchmarks". No SOTA claims.

### E.5 Is the substrate physically fabricated or purely simulated?

Purely simulated, by the authors' own repeated statement. The originating plan (`NEW_ALGORITHM_PLAN.md:13-17`) says:

> 前提条件（用户已确认）：
> - **无实验室、无流片条件**；全部权重/状态均为数字模拟（Digital Model 级）。
> - 计算环境 CPU-only（numba + 多进程，禁止 GPU）。

`paper_e/PAPER_E.tex:258-262` (quoted in full in §E.2) and `:1860-1862` agree. **No result in A–F is evidence about hardware.** The prior "Paper 0" Si₃N₄ pulse-encoding work supplies the parameter set only (`README.md:126-140`), and explicitly covers a different object — *"a parallel array of independent devices (quasi-static block coupling, offline Ridge/MLP readout)"* vs *"the per-pulse coupled recurrent substrate"* (`README.md:136-140`).

### E.6 Additional blunt weaknesses not usually on the list

- **Repository-level reproducibility for E and F is committed but not yet published (re-checked 2026-09-14).** The earlier `git status` gap — `?? paper_e/`, `?? paper_f/`, `?? data/s39_…`–`?? data/s65_…`, `?? scripts/s39_…`–`?? scripts/s65_…` — is closed in the working tree: Paper D (NC manuscript + its evidence data), Paper E (manuscript, figures, per-token data, s39–s65 scripts+data) and Paper F (form-learning package, s66/s67) are all committed (`8e3098b`, `06df9e2`, `85237d2`, `c45b075`, `2a3e823`, `3b4fc1c`, `0724e01`, `13c3b03`). **But the branch is 9 commits ahead of `origin/main`, so this is not yet true for a reader of the public repository.** The root README's *"All simulation code and data required to reproduce the results are openly available"* (`README.md:276-277`) therefore still overstates the published state until the push happens, and `paper_e/README.md:269-270` still says *"the public repository push with a commit/tag is still pending."*
- **Paper E's README quotes population SDs where the paper's stated convention is sample SD** in one place (the s65 retention numbers ±4.28 / ±3.72 = the JSON's stored `sd`). Verification against `data/s65_ewc_baseline_v1.csv` shows the *paper* is internally consistent (sample SD = 0.696 pp for the −0.85 pp figure, 4.28 pp for the +4.95 pp figure), so this is a documentation nit, not a numerical error — but the convention paragraph's claim that "the archived JSON files record the population SD in their `*_std` fields" holds for per-arm std fields and **not** for the paired-stats `sd` fields, which already store the sample SD.
- **The repo README and the current manuscript have drifted apart on a headline mechanism (cleared 2026-09-13).** `README.md` no longer lists *"structural (per-dimension) correction required"* among Paper E's core results; `PAPER_E.tex:171-173,805-811` and the Discussion retract that claim and name s47 a retracted design attempt.
- **Several committed experiments contribute no numbers to Paper E**: `s47_self_correction_gain_v1` (32 cells / 320 rows) and `s54_memory_boundaries_v1` appear in the reproducibility index but are never cited numerically in the text; `s49_capacity_probe_v1` and `s55_sphere_capacity_v1` are cited by name only. **s47 is deliberately not cited numerically**: script-level P2 **[REFUTED by own data]**, P3 **[NOT VERIFIABLE]** (`s47:39-64`); the arm is a retracted design attempt, **not** an honest negative and **not** a field-level finding. Do not promote s47 numbers into the manuscript or into negative-result tables that mix s42/s58c (controlled negatives) with s47.
- **The falsification discipline lives in the scripts, not the manuscript.** `PAPER_E.tex` contains **zero** occurrences of "pre-registration" or "falsification"; the scripts carry a repeated header *"Predictions (status checked against the committed data, 10 seeds; the refuted entries keep their original claim plus the measured outcome)"* with explicit verdicts — `s43:36-41` **[REFUTED]**, `s43b:41` **[REFUTED]**, `s44:49-75` **P2/P3 [REFUTED]**, `s45:49-70` **P1/P2 [REFUTED]**, `s46:32-48` **P1 [REFUTED]**, `s47:39-64` **P2 [REFUTED] / P3 [NOT VERIFIABLE]**, `s58f:25-40` **P1 [PARTIALLY REFUTED]**, `s60:50-76` **P1/P3 [REFUTED]**. Of these, the s44/s45/s46/s47/s58f/s60 refutations surface (partly) in the text; **the s43/s43b/s45 population-memory refutations — which falsify the "hypothesis population beats a single flip" prediction — do not appear in the manuscript at all.**
- **Documented internal inconsistencies in the manuscripts** (all minor, none changing a conclusion): the `fig:margin` caption says error bars are the SEM while the body calls the same ± the seed SD (`PAPER_E.tex:1505-1506`); the same physical quantity is printed as 0.096 in one row and 0.097 in another (s44 vs s46 reruns 1 ULP apart); v3's ±-convention paragraph asserts that JSON `*_std` fields store the population SD, which holds for per-arm std fields but **not** for paired-stats `sd` fields (those already store the sample SD — verified against `data/s65_ewc_baseline_v1.csv`); and the removed `PAPER_F_PLAN.md` claimed 60 rows for `s51_paper_f_learned_gate_v1.csv` when the file has 80.
- **The metric problem is systemic**, not confined to D. Paper D discloses it; Paper F exists because of it; Paper E's deployed cross-family ring gain was a **false significance statement** in v2 that v3 had to retract (`paper_e/README.md:52-61`). Three of four per-segment memory gains were also mis-stated as non-significant. The v2 review found **132 findings (28 P0 + 104 P1)** in a 60-page manuscript (`paper_e/README.md:28-35`) — evidence of how much correction the small-scale statistics needed.
- **Two known open defects carried forward** (`paper_e/README.md:259-270`): pooled Wilson intervals in s58b computed at n = 20 while the effective sample size is ~10 seeds; the `--sequential` flag documented in the reproduction section is **not implemented and silently ignored**; 16 JSON files outside Paper E's citation set still contain bare `NaN`.
- **The internal "chaos" claim is a designed-knob claim.** The order–chaos transition moves with the coupling *clip range* for one topology (`ring_bidir` shifts +3.4 in κ\* under a 5× clip widening, `README_REDEM.md:68`) and 39 % of the deployed α_eff values are clipped — so part of the "edge of chaos" is an artifact of the saturating knob, not of the material.
- **Corrections applied 2026-09-14 to D/E/F** (details in `review_workspace/def_audit/` and `review_workspace/modification_log_scripts.md`). (i) *Capacity accounting in F was inverted*: the frozen external selective-SSM arm holds its 24,864 host parameters at a per-seed **random** init, so it trains only the 8,224-parameter readout, while F's Gate-C-topk arm trains 24,736 — i.e. F trains **3.0× more**, not 1.34× fewer; the "not a capacity fluke" reading is withdrawn and the paper/READMEs now state that neither budget comparison is like-for-like. (ii) F's claims that top-$k$ is "the best stream arm" were contradicted by F's own Table (`Gate-C-L1 = 6.994 < 7.033`); the claims are now scoped to retention-safe variants and the $L_1$ arm is discussed. (iii) D called A3 "soft routing" while `s20` implements a discrete hysteresis nearest-reference selector (`GATE_MARGIN = 1.15`); D now calls it hysteresis-gated hard routing and keeps M4's softmax routing distinct. (iv) A *vacuous* validation field was found and fixed: `data/s66_external_ssm_baseline_v1.json` stored `anchor_validation = {pairs: 0}`; re-running s66's own `validate()` against the committed rows reproduces the s50/s51 anchors **40/40 at relative difference 0.0**, so the field is corrected to `pairs: 40` (metadata only — no result row changed). (v) Highlights files for D/E/F carried a title block and a "Notes" section around the bullets and are now bullets-only (`Highlights.txt`, `.docx` regenerated); F's bullet 5 repeated the inverted parameter claim and was rewritten. (vi) The declaration sections of F sat **after** the reference list; the guide requires them before it, and they have been moved, with a Code-and-data-availability statement and repository URL added. (vii) All three reference lists were re-ordered into first-citation order (Elsevier numbered style) and F's Lukoševičius entry corrected.
- **The "abstract ≤ 250 words" constraint previously assumed for D/E/F does not exist for this journal.** The current Neurocomputing Guide for Authors (retrieved 2026-09-14; the ScienceDirect page returns HTTP 403, so a full mirror was used) states only that "a concise and factual abstract is required" and that references should be avoided — it sets **no** word limit. Measured lengths are D 261, E 283–287, F 266–289 words depending on whether math is counted; README/plan figures of "238" and "~241 words" were not reproducible and have been replaced. Highlights (3–5 bullets, ≤85 chars) are "optional yet highly encouraged" for this journal.

---

## F. `paper_c/LLM_EXTENSION.md` — what it proposes, and what it actually showed

### F.1 What it proposes

`paper_c/LLM_EXTENSION.md` is explicitly strategic, not empirical: *"Status: strategic analysis (no experiments run for this extension yet)"* (`:3`). It maps Paper C's three mechanisms onto a Transformer/LoRA system and **pre-commits to a falsification criterion** before running anything.

Mechanism mapping (`:29-36`): M3 slow trace → **strong candidate** (EMA of hidden states = cheap drift detector + domain statistic); M5 homeostat → **weak / must be redefined** (*"no FTLE analog defined"*); M4 plasticity → **partial, principle only**; substrate physics → **not transferable by design**.

It also supplies a scalability argument to defuse the obvious objection (`:142-171`):

> The resolution is that REDEM's RLS never touches the substrate/backbone weights - it adapts only the READOUT W over the feature vector phi … The covariance P is F x F where F is the readout's FEATURE dimension …
> - **Output projection head**: F = model width d. For 7B-class (d=4096) the covariance is d^2 ~= 16.7M entries (~67 MB fp32); for 70B-class (d=8192), ~67M entries (~268 MB). …
> - **What you must NOT do**: full-rank RLS over the concatenated adapter parameter vector (that IS O(params^2) and infeasible).

**Note the status of that argument:** it is an *arithmetic feasibility estimate*, not a measurement. No 7B or 70B model is run; the estimate is anchored on `"the numba RLS kernel at F=513 is ~1-2 ms/step"` (`:168`) — i.e. on the Paper-E-scale readout.

The pre-registered falsification (`:136-140`):

> **Falsification pre-commitment** (mirrors S14): if the drift gate does not improve stream perplexity or adaptation over bare online LoRA with 0/10-seed-sign-consistency across the tau_m sweep, the extension claim is reported as falsified — and that itself is a Paper C-consistent result (mechanisms do not transfer beyond their job).

And the locked scope claim (`:244-246`, quoted in §E.4 above): instantiation of principles, **not** SOTA.

### F.2 What it actually showed — s18, the "tiny transformer + LoRA, 90 runs" PoC

Model and protocol (`:248-253`):

> Script `scripts/s18_llm_drift_gate.py` (Type: ML; torch CPU; char-level tiny transformer **d=64/L2/H2/ctx=128, vocab 32, LoRA rank 16 on QKV+out+FFN**; two biased-bigram domains alternating every **3000 tokens** with known switches; stream **18k tokens**). Data: `data/s18_llm_drift_gate_v1.*`.

**Measured result table** (`paper_c/LLM_EXTENSION.md:255-265`; independently consistent with `data/s18_llm_drift_gate_v1.csv`, 90 runs, 894 s total):

| arm | τ_m | stream ppl | paired diff vs A1 | forgetting ppl | paired diff vs A1 |
|---|---|---|---|---|---|
| A1 bare | – | **15.01 ± 0.33** | – | **8.94 ± 0.33** | – |
| A2 gate | 200 | 17.31 | +2.29 (**0/10**) | 9.66 | +0.73 (0/10) |
| A2 gate | 500 | 17.06 | +2.05 (0/10) | 9.48 | +0.54 (0/10) |
| A2 gate | 1000 | 16.58 | +1.57 (0/10) | 9.24 | +0.30 (0/10) |
| A2 gate | 2000 | 16.26 | +1.25 (0/10) | 9.03 | +0.09 (0/10) |
| **A3 route** | 200 | **13.94** | **−1.07 (10/10)** | **6.42** | **−2.51 (10/10)** |
| A3 route | 500 | 14.58 | −0.43 (10/10) | 6.75 | −2.18 (10/10) |
| A3 route | 1000 | 15.47 | +0.46 (0/10) | 7.43 | −1.50 (10/10) |
| A3 route | 2000 | 16.07 | +1.06 (0/10) | 8.41 | −0.53 (**9/10**) |

**The stated falsification fired.** The extension document reports it plainly (`:267-290`):

> 1. **A3 domain routing TRANSFERS**: it reduces forgetting at every tau_m (9-10/10 seeds, up to -2.5 ppl, 28% at tau_m=200) … It also improves stream perplexity when the gate is fast (tau_m in {200,500}: 10/10 seeds).
> 2. **A2 drift gate is FALSIFIED** (0/10 seeds at every tau_m): aggressive update suppression ("adapt only near switches") loses because the model needs continuous in-domain learning on this task. Reported as a negative - Paper C-consistent (not every metadata instantiation transfers).
> 3. **Timescale sensitivity confirmed** (the s16 lesson): the ppl benefit requires small tau_m (<=500); tau_m >= 1000 flips the ppl sign, and the forgetting benefit narrows. The sensitive interval is reported, not hidden.
> 4. **T_adapt floors at the 20-token window** for all arms: the task's transition is learnable within one chunk, so adaptation is too fast to resolve; reported at the floor rather than as ratios (the s15 lesson).
> 5. The gate estimation itself tracks the domains (5 flips per stream at the 5 known switches, ~500-token lag at tau_m=500, 10% wrong-est chunks) - the metadata channel is functional; the negative A2 result is about the update policy, not the detector.

### F.3 How to read F.2

| Element | Status |
|---|---|
| A3 routing reduces forgetting | **Measured**, paired, 10/10 at τ_m = 200 with Δ = −2.51 ppl (≈28 %), and at every τ_m at 9–10/10 |
| A3 improves *stream* perplexity | **Measured but τ_m-bounded**: −1.07 / −0.43 at τ_m ≤ 500; the sign **reverses** at τ_m ≥ 1000 |
| A2 drift gate | **Falsified**, 0/10 at every τ_m, magnitude +1.25 … +2.29 ppl *worse* than bare online LoRA |
| Adaptation speed (T_adapt) | **Not resolvable** — every arm sits on the 20-token window floor, so claim #4 of the prediction table could not be tested |
| "LLM" | It is a **128,544-parameter** char-level model on **synthetic bigram** text with a **32-symbol** vocabulary. Nothing in the result says anything about billion-parameter behaviour; the paper says so (`:292-294`) |
| Generalization | Single domain pair (2 Markov chains), one model configuration, 10 seeds. The prediction that "gain peaks near tau_m ~ L/4..L/2" (`:231`) was not what happened: gain *decreased monotonically* with τ_m |

---

## Cross-cutting summary table (what is measured vs asserted)

| Item | Measured | Asserted |
|---|---|---|
| Substrate memory–chaos phase diagram | ✅ simulation, 610 + 990 runs, 10 seeds | that it describes a *physical* Si₃N₄ device ❌ (constants inherited, recurrence designed, 39 % α saturation) |
| Online RLS learning on the substrate | ✅ 10 seeds, paired, replicated at N=256/1024 | "beats frozen batch learners" ✅ *only* against **untuned** GRU/TF; a matched ESN wins |
| Metadata / homeostat / plasticity division of labour | ✅ falsifying transfers, τ_m and protocol sweeps | "non-substitutable" ✅ *scoped in the title* to the directions/conditions tested |
| SSM-hosted REDEM | ✅ 10 seeds, N=128, reported *and* unclipped metrics | the stream advantage ❌ explicitly withheld (floor-driven) |
| Self-evolution under ±1 reward | ✅ 30+ controlled experiments, s39–s65, n=10, one external baseline (EWC); every quoted number reproduces from the committed CSVs/JSON | "self-evolving" at model scale ❌ — readout-only, 257→769 features, capacity sense idles on the deployed substrate; decision-level autonomy + minimal operational self (not accuracy); s47 per-dimension gain **retracted** (not a negative); single substrate, single ±1 feedback budget, no embodied closed loop |
| Instrument-must-be-learnable rule (F) | ✅ 6 pre-registered propositions, 2 of which fail as reported | SOTA / scaling ❌ explicitly disclaimed |
| LLM extension | ✅ 90 runs on a 128 k-parameter char model | that it says anything about LLMs ❌ disclaimed in the same document |
| Reproducibility | ✅ A–D committed and paired bit-exact checks (s50 ≡ s19; s65 ≡ s52 50/50) | "all code and data openly available" ❌ E and F are git-untracked |

**Bottom line.** This is a careful, unusually self-critical *small-scale* program: the controls, script-level falsification discipline, paired 10-seed statistics and self-corrections (a dozen retracted or withheld claims across D/E/F) are better than typical. But nothing in it is at, or extrapolates safely to, LLM scale. The distance from the evidence to "self-evolving large model" is: **~4 orders of magnitude in parameters (largest network 1.8 × 10⁵), ~3–4 in vocabulary (32 characters), ~6 in text (0.9 M characters), zero hardware, and a mechanism set that modifies a few hundred to a few thousand readout weights rather than a model.**
