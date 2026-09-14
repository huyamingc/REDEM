# SCALE AUDIT — REDEM program (papers A–F), repo `D:\work\papers\testpynew`

Method: read/grep/glob over `scripts/`, `data/`, `paper_*/`, `review_workspace/`; plus a
read-only Python pass over `data/*.json` and `data/*.csv` to aggregate the recorded
`runtime_s` / `elapsed_s` / `n_runs` fields and the corpus files. Every number below is
either quoted verbatim from a file with a `file:line` anchor, or is arithmetic shown in full.

**Blunt headline.** The largest network *actually simulated* in this program is **N = 1024
units** (`s30_integrated_1024.py`, plus a 3-seed confirmation inside `integrated_benchmark.py`).
The largest language-model vocabulary is **32 symbols**; the largest transformer is
**d_model = 64, 2 layers, 2 heads**; the largest neural-network parameter count anywhere is
**177,696 parameters** (`s26` 4-adapter TinyCharLM), and the largest hand-rolled learned model
is **328,256 trainable parameters** (`s53`/`s53b` Soft-topk at N = 512). Total recorded
experimental compute over the 74 committed result CSVs is **11,550 runs / 356,940 s ≈ 99.2 h
≈ 4.13 days of single-core CPU time** — before multiprocessing. The program's own documentation
concedes the scope: *"All results are CPU-scale proofs of concept; no scaling claims are made."*
(`paper_d/README.md:70-71`).

---

## 1. Largest network size (N) actually simulated

### 1.1 Exhaustive constant sweep

Grepped every `*.py` under `scripts/` for `N_UNITS`, `N_DEV`, `n_units`, `N=`, `N_STATE`,
`SUBSTATE_DIM`, `d_model`, `vocab`, `hidden`, `EMB`, `N_TAUS`.

| Constant | Max value | Where | Notes |
|---|---|---|---|
| `N_UNITS` | **256** | ~50 scripts, e.g. `scripts/s11_disturbance_chain.py:56`, `s19…s64` chain, `substrate_recurrence_characterization.py:73`, `baseline_showdown.py:64`, `s58d_kernel_capacity_sweep.py` (imports) | This is the universal substrate width. Every single `N_UNITS = <n>` assignment in the repo is literally `256` — 50 of them, no exceptions. |
| `N_1024` | **1024** | `scripts/s30_integrated_1024.py:51` → `N_1024 = 1024` | The one true scale-up. |
| literal `1024` | **1024** | `scripts/integrated_benchmark.py:261` → `all_args.append((arm, s, 1024))` | S8's original 3-seed N=1024 confirmation. |
| `N_STATE` | **128** base, scaled to **512** | `scripts/s19_ssm_rls_readout.py:99` → `N_STATE = 128`; `scripts/s53_paper_f_scaling.py:57` → `N_LIST = [128, 256, 512]`; `scripts/s53b_paper_f_ksweep.py:51` → `N = 512` | Paper D/F diagonal-SSM host width. |
| `N_UNITS` in `s58d` configs | **512** | `scripts/s58d_kernel_capacity_sweep.py:115` `('4Dshell', K4D, 25.0, 512),` and `:126` `('2Dcircle', DB_K_PULSES, 25.0, 512),` | Confirmed in data: `data/s58d_kernel_capacity_sweep_v1.csv` has `n_units ∈ {64, 256, 512}`. |
| `n_units` in `s43b` | **3** | `scripts/s43b_synthetic_rotation.py:268` → `'n_units': d` (d = 3, a 3-D synthetic state) | Smallest substrate in the program. |
| `VOCAB` | **32** | `scripts/s18_llm_drift_gate.py:56` → `VOCAB = 32`; `scripts/s19_ssm_rls_readout.py:98` → `VOCAB = 32`; `scripts/s23_ssm_p4_realtext.py:78` → `V = 32   # 31 frequent chars + UNK (matches VOCAB=32)` | Never larger. |
| `D_MODEL` | **64** | `scripts/s18_llm_drift_gate.py:57` → `D_MODEL = 64`; `scripts/baseline_showdown.py:81` → `TRANS_D_MODEL = 64` | Never larger. |
| `N_LAYERS` | **2** | `scripts/s18_llm_drift_gate.py:58`; `scripts/baseline_showdown.py:82` → `TRANS_LAYERS = 2` | Never larger. |
| `N_HEADS` | **2** | `scripts/s18_llm_drift_gate.py:59`; `scripts/baseline_showdown.py:83` | Never larger. |
| `CTX` / `TRANS_CONTEXT` | **128** / **256** | `s18_llm_drift_gate.py:60` → `CTX = 128`; `baseline_showdown.py:84` → `TRANS_CONTEXT = 256` | 256 is a 1-D-signal window, not a token vocabulary. |
| `RANK` (LoRA) | **16** default, **32** max | `s18_llm_drift_gate.py:61` → `RANK = 16`; `s26_ssm_p4_fair_tf.py:71-72` → `A1_GRID = [(3e-4, 16), …, (1e-2, 32)]` | Rank 32 is the ceiling. |
| `GRU_HIDDEN` | **64** | `scripts/baseline_showdown.py:80` | |
| `SUBSTATE_DIM` | **32** | `scripts/s50_paper_f_pilot_nl_readout.py:89` | |
| `NOISE_DIM` | **128** | `scripts/s50_paper_f_pilot_nl_readout.py:88` | |
| `N_TAUS` | **does not exist** | grep over all `*.py`: zero hits for `N_TAUS`, `N_TAU`, `n_taus` | There is no "number of taus" constant; τ is drawn per unit via `gen_tau_vec(n_units, CV_TAU, tau0, seed)`. |
| `N_DEV`, `SUBSTATE_DIM` (elsewhere) | absent / 32 | `N_DEV` has zero hits | |
| Largest `N` anywhere in code | **16384** — but NOT simulated | `scripts/shallow_trap_array_simulator.py:264` → `for N in [64, 256, 1024, 4096, 16384]:` | This is the `__main__` self-test of the CORE simulator. Its output `data/square_array_results.csv` (line 287) **does not exist in the repo**, and `review_workspace/agent_f_result_E_v2_part7.md:3` records *"输出 data/square_array_results.csv 未随仓库提交、论文 E 正文亦无 square_array 引用（无溯源链）"*. Treat it as dead code, not a simulated size. |

### 1.2 The six files the task asked about, specifically

| Script | Size constant | Value |
|---|---|---|
| `scripts/s30_integrated_1024.py` | `N_1024 = 1024` (line 51); `N_SEEDS = 10` (52) | **1024 units, 2 arms × 10 seeds = 20 runs** |
| `scripts/substrate_recurrence_characterization.py` | `N_UNITS = 256` (line 73); `T_TOTAL = 1200` (75); `K_MAX = 50` (77); `N_SEEDS = 10` (78) | 256 units, 610 runs |
| `scripts/s36_random_graph_instance_variability.py` | imports `src.N_UNITS` (=256), line 67; `TOPO_SEEDS` 9 entries (49) | 256 units, 9 graphs × 11 κ × 10 seeds = **990 runs** |
| `scripts/s19_ssm_rls_readout.py` | `N_STATE = 128` (99), `VOCAB = 32` (98) | 128-dim SSM, 70 runs |
| `scripts/s18_llm_drift_gate.py` | `VOCAB=32, D_MODEL=64, N_LAYERS=2, N_HEADS=2, CTX=128, RANK=16, N_ADAPTERS=2` (56-62) | tiny char LM, 90 runs |
| `scripts/s22_ssm_p4_benchmark.py` | imports `N_STATE`(=128), `VOCAB`(=32), `D_MODEL`(=64) from s18/s19; `N_DOMAINS=4, N_SEGMENTS=8` (79-80) | 30 runs |
| `scripts/s23_ssm_p4_realtext.py` | imports `N_STATE`(=128), `VOCAB`(=32), `CTX`; `V = 32` (78) | 30 runs |
| `scripts/s50_paper_f_pilot_nl_readout.py` | `NOISE_DIM = 128` (88), `SUBSTATE_DIM = 32` (89); `N_STATE`=128 imported | 80 runs |
| `scripts/s51_paper_f_learned_gate.py` | `TOPK_K = SUBSTATE_DIM`=32 (81); gate `C` is `N_STATE × N_STATE` (110) | 80 runs |
| `scripts/s53_paper_f_scaling.py` | `N_LIST = [128, 256, 512]` (57), `N_SEEDS = 5` (58) | **max 512**, 45 runs |
| `scripts/s58d_kernel_capacity_sweep.py` | CONFIGS include `n_units` 64/256/**512** (103-127) | max 512, 200 runs |
| `scripts/s65_ewc_baseline.py` | `N_UNITS = 256` (119); `"quadratic basis (d=513)"` (docstring line 21); `"d = 2*256 + 1 = 513"` (line 186) | 256 units, 40 runs |
| `scripts/baseline_showdown.py` | `N_UNITS = 256` (64), `GRU_HIDDEN=64` (80), `TRANS_D_MODEL=64, TRANS_LAYERS=2, TRANS_HEADS=2, TRANS_CONTEXT=256` (81-84) | 80 runs |
| `scripts/fair_esn_comparison.py` | `N = 256` (31); ESN sizes from docstring: `ESN-40`, `ESN-256` (lines 15-18) | max 256 |

### 1.3 Stated N in the paper documents

| Document | Anchor | Verbatim |
|---|---|---|
| `paper_f/PAPER_F_PLAN.md` | line 160 | `Evidence gaps (declare, no exceptions): two-domain biased-bigram; N=128 CPU; no Mamba/TTT baseline;` |
| `paper_f/PAPER_F_PLAN.md` | line 189 | `\| M4 \| scale (s53 N=128/256/512, 5 seeds) \| **done** — C1 scales; C4b/C5 fail to hold at N=512 \|` |
| `paper_f/PAPER_F_sketch.md` | line 83 | `\| Host \| N=128, log-uniform τ∈[1,3000], whitened (s19 CV) \|` |
| `paper_f/PAPER_F_sketch.md` | lines 168-169 | `- Primary tables at N=128. **s53 + s53b:** with k=max(8,N/4), C4b/C5 fail at N=512` |
| `paper_d/README.md` | line 155 | `…Real-text transfer: Alice vs. Dickens, 32-symbol char vocab (3 arms, 10 seeds)…` |
| `paper_d/PAPER_D_sketch.md` | line 133 | `SSM (N=128), per-token RLS on the output projection (O(F^2), F=N+1),` |
| `README_REDEM.md` | line 70 | `\| s30 \| N=1024 integrated replication at 10 seeds (baseline vs full, paired per seed): full 0.9970 ± 0.0007 vs baseline 0.9753 ± 0.0044 (+2.2 pp, paired t=15.3, 10/10) …` |
| `paper_b/PAPER_B_draft.md` | line 29 | `…the substrate is a per-pulse contrast-coupled relaxation array: $N=256$ units, log-normal $\tau$ (median 174 $\mu$s, CV 0.20)…` |
| `paper_a/PAPER_A_draft.md` | line 47 | `Five topologies over the $N=256$ units are used: …` |

**Verdict for Q1:** the largest network size actually simulated is **N = 1024** (s30/S8).
Everything else in the substrate line is N = 256 (or 512 in exactly three scripts: `s53`,
`s53b`, `s58d`), and the LLM/SSM line is N = 128 (scaled to 512 only in `s53`/`s53b`).

---

## 2. Largest model parameter count

Only **three** torch `nn.Module` model families exist in the whole repo. Grep for
`nn.Linear|nn.Embedding|nn.GRU|nn.LSTM|TransformerEncoderLayer|nn.MultiheadAttention` over all
non-`.venv` `*.py` returns only `scripts/baseline_showdown.py` (lines 101, 102, 113, 115, 119)
and `scripts/s18_llm_drift_gate.py` (lines 151, 203, 210). Everything else is hand-rolled
tensor arithmetic.

### 2.1 `baseline_showdown.py` — GRUModel (lines 80, 98-102)

`nn.GRU(1, 64, batch_first=True)` + `nn.Linear(64, 1)`

```
GRU  : weight_ih_l0 (192,1)  =    192
       weight_hh_l0 (192,64) = 12,288
       bias_ih_l0   (192,)   =    192
       bias_hh_l0   (192,)   =    192
                             --------
                               12,864
head : Linear(64,1)          =     65   (64 + 1)
                             --------
TOTAL  =  12,929 trainable parameters
```

### 2.2 `baseline_showdown.py` — TinyTransformer (lines 81-84, 109-119)

`embed = nn.Linear(1, 64)`; `pos` = `(1, 256, 64)`; `nn.TransformerEncoderLayer(d_model=64,
nhead=2, dim_feedforward=2*64=128)` × 2; `fc = nn.Linear(64, 1)`

```
embed Linear(1,64)                    =    64 + 64   =    128
pos  (1,256,64)                       = 16,384
per encoder layer:
  self_attn.in_proj_weight (192,64)   = 12,288
  self_attn.in_proj_bias   (192,)     =    192
  self_attn.out_proj (64,64) + bias   =  4,160
  linear1 (64→128) + bias             =  8,320
  linear2 (128→64) + bias             =  8,256
  norm1 LayerNorm(64)                 =    128
  norm2 LayerNorm(64)                 =    128
                                      --------
                                       33,472  per layer
  × 2 layers                          = 66,944
fc Linear(64,1)                       =     65
                                      --------
TOTAL = 83,521 trainable parameters  (all trained: TRAIN_EPOCHS=20, BATCH=128, 30% of stream)
```

### 2.3 `s18_llm_drift_gate.py` — TinyCharLM with hand-rolled LoRA (lines 56-62, 145-217)

`VOCAB=32, D_MODEL=64, N_LAYERS=2, N_HEADS=2, CTX=128, RANK=16, N_ADAPTERS=2`.
Frozen base: `embed=nn.Embedding(32,64)` (line 203, `requires_grad_(False)`), `pos (128,64)`
frozen (205-206), `ln` frozen (208-209). Trainable: LoRA `A`/`B` per adapter + `head=Linear(64,32)`.

```
qkv  LoRALinear(64→192): A (16,64)=1,024 ×2 adapters = 2,048
                         B (192,16)=3,072 ×2        = 6,144   → LoRA 8,192 ; base 64·192+192 = 12,480
out  LoRALinear(64→64) : A 1,024×2 = 2,048 ; B (64,16)=1,024×2 = 2,048 → LoRA 4,096 ; base 4,160
ff1  LoRALinear(64→128): A 1,024×2 = 2,048 ; B (128,16)=2,048×2 = 4,096 → LoRA 6,144 ; base 8,320
ff2  LoRALinear(128→64): A (16,128)=2,048×2 = 4,096 ; B 1,024×2 = 2,048 → LoRA 6,144 ; base 8,256
                                                          per-block LoRA = 24,576
                                                          per-block base = 33,216 ; LayerNorms 256
× 2 blocks:  LoRA 49,152   base 66,432   LN 512
+ embed 32·64 = 2,048 (frozen) ; pos 128·64 = 8,192 (frozen) ; final LN 128 (frozen)
+ head Linear(64,32) = 64·32 + 32 = 2,080 (trainable)
                                                     ---------
TOTAL = 128,544 parameters
        frozen base      = 77,312
        trainable  A1/A2 = 12,288 (one adapter) + 2,080 =  14,368
        trainable  A3    = 49,152 (both adapters) + 2,080 = 51,232
```

### 2.4 `s26_ssm_p4_fair_tf.py` — the largest torch model (lines 71-75, 86-97, 235)

Same `TinyCharLM`, but `extend_adapters(model, N_DOMAINS, rank=rank)` with `N_DOMAINS = 4`:

```
4 adapters × 12,288 per block × 2 blocks = 98,304 LoRA params
+ head 2,080
TOTAL trainable (A3) = 100,384
TOTAL model          = 77,312 (frozen) + 98,304 + 2,080 = 177,696 parameters   ← largest torch model
A1 rank-32 grid runs : LoRA = 32-rank adapter 0 only = 12,288 × 2 blocks? no:
                       per-block rank-32 single adapter = 24,576 → ×2 = 49,152 + head 2,080 = 51,232 trainable
                       model total still 177,696 (two rank-32 adapters instantiated)
```

### 2.5 `fair_esn_comparison.py` — ESN reservoir (lines 31, 48-58)

```
ESN-256 : W_in (256,1) = 256 ; W_res (256,256) = 65,536 ; bias (256,) = 256
          TOTAL = 66,048 untrained random parameters (readout = sklearn RidgeClassifier on 257 features)
ESN-40  : W_in 40 ; W_res (40,40) = 1,600 ; bias 40  → TOTAL = 1,680
```
Docstring (lines 15-18) enumerates the four configs actually compared:
`ESN-40-uniform`, `ESN-256-uniform`, `ESN-256-hetero`, `ESN-256-hyperopt`. Maximum ESN = 66,048.

### 2.6 Diagonal-SSM readouts (hand-rolled torch, float64)

Host: `A` diagonal (`N_STATE`), `B` (`N_STATE × VOCAB`). Readout `W` (`VOCAB × F`).

| Script | N | F | W params | Extra learned module | Trainable total |
|---|---|---|---|---|---|
| `s19_ssm_rls_readout.py` (`F = N_STATE + 1 = 129`, line 280; JSON `data/s19_ssm_rls_readout_v1.json:12` `"feature_dim": 129`) | 128 | 129 | 32×129 = **4,128** | `B-proj` arm: `C (128,128)=16,384` + `b_gate 128` (`s19:255-257`) → **20,736** | ≤ 24,864 |
| `s20/s21/s22/s23/s33_ssm_*` (`F = N_STATE + 1 = 129`, e.g. `s22:159`) | 128 | 129 | 4,128 | — | 4,128 |
| `s50_paper_f_pilot_nl_readout.py` (`F = N_STATE+1=129` / `+NOISE_DIM`=257, line 162) | 128 | 257 | 32×257 = **8,224** | — | 8,224 |
| `s51_paper_f_learned_gate.py` (`F = 2*N_STATE+1 = 257`, line 127) | 128 | 257 | 8,224 | gate `C (128,128)+b(128)` = **16,512** — recorded verbatim as `"n_params_gate": 16512` in `data/s51_paper_f_learned_gate_v1.json:350-500` | **24,736** |
| `s52_paper_f_soft_route_experts.py` (`F = 2*N_STATE+1 = 257`, line 101) | 128 | 257 | 2 experts × 8,224 = 16,448 | gate 16,512 | **32,960** |
| `s53_paper_f_scaling.py` at N=256 (`F = 2n+1 = 513`, line 148) | 256 | 513 | 2×32×513 = 32,832 | `C (256,256)+b(256)` = 65,792 (line 144-145) | **98,624** |
| **`s53_paper_f_scaling.py` / `s53b_paper_f_ksweep.py` at N=512** | 512 | 1025 | Soft-topk 2 experts × 32×1025 = **65,600** | `C (512,512)+b(512)` = **262,656** (`s53:144-145`; `s53b:80-81`) | **328,256 trainable** (+16,896 frozen substrate A(512)+B(512×32)) = **345,152 total** |
| `s54_paper_f_mackey_glass.py` (`F = 2*N_STATE+1 = 257`, line 136) | 128 | 257 | 16,448 | gate 16,512 | 32,960 |

**The single largest parameter count in the whole program is therefore 345,152**
(`s53`/`s53b` Soft-topk at N = 512: 262,656 gate + 65,600 readout + 16,896 frozen substrate),
of which **328,256 are trainable**. If you count only parameters a gradient/learning rule ever
touches, it is **328,256**. The largest *torch* `nn.Module` is **177,696** (`s26` 4-adapter
TinyCharLM).

### 2.7 Paper E substrate readout (s39–s64 chain)

`N_UNITS = 256`; `OnlineRLS(F, n_outputs)` with `F = 256 + 1 = 257` features (e.g.
`s59_self_tuned_gate.py:217` uses `build_features`, which is
`np.hstack([(obs_raw - mu)/sd, np.full((T,1),BIAS)])`).

```
Linear basis  : W = 257 × n_outputs  ∈ {257, 514, 771} weights
                + RLS inverse covariance P = 257² = 66,049 float64 = 528 KB
Quadratic     : d = 513  ("d = 2*256 + 1 = 513", s65_ewc_baseline.py:186)
                W = 513 × n_outputs ; P = 513² = 263,169 float64 = 2.1 MB
Cubic (s58c)  : MAX_LEVEL = 2 (s58c:134) → F = 771 ; P = 771² = 594,441 float64 ≈ 4.8 MB
s65 EWC       : AdaGrad w (513,) + accumulator G (513,) + up to 4 Fisher diagonals + 4 anchors
                ≈ 5 × 513 = 2,565 numbers per arm
```
These are *tiny* — hundreds of weights. The only large object is the RLS covariance.

### 2.8 The single largest matrix in the program

`integrated_benchmark.py` with the metadata arm at N = 1024 builds `F = 2*1024 + 1 = 2,049`
features (line 164: `F = np.hstack([fast_s, slow_s])`, plus bias at line 167). The RLS inverse
covariance is then `P = 2049² = 4,198,401 float64 ≈ 33.6 MB`. The script itself names it:

> `scripts/integrated_benchmark.py:257-258` — `# N=1024 scale confirmation (3 seeds; the 2049-dim RLS is the`
> `# runtime bottleneck ~2min/run, so keep the confirmation lean)`

---

## 3. Wall-clock cost per experiment

### 3.1 What is actually recorded

Grepped all `data/*.json` for `runtime|elapsed|seconds|duration|wall|n_runs|seeds`:

* A top-level `"elapsed_s"` key exists **only in the Paper F scripts**: `s50`, `s51`, `s52`,
  `s53`, `s53b`, `s54` (10 files incl. `_quick`).
* A per-run `"runtime_s"` inside the JSON/CSV rows exists in **19 committed CSVs**
  (s2, s3, s5, s8, s9, s10–s15, s17–s28, s30, s32–s37, s39–s65 subsets).
* **No** `wall`, `duration`, `total_runtime`, `compute`, or `seconds` keys exist anywhere.
* The remaining ~55 CSVs carry no runtime column at all — their cost is only recoverable from
  prose logs (below).

### 3.2 Table — runs and wall-clock per experiment (committed data)

`Σrt` = sum of the per-run `runtime_s` values in the file = **serial single-core CPU-seconds**.
`max run` = longest single run in that file.

| Experiment / script | Data file | Runs | Σ serial time | max single run | Wall (if recorded) | Anchor |
|---|---|---|---|---|---|---|
| S1 phase diagram | `substrate_phase_diagram_v2.csv` | **610** | 956.3 s | 4.73 s | *"~1 min with multiprocessing"* | `paper_a/README.md:76`; `README_REDEM.md:135` → `# S1 phase diagram (610 runs, ~1 min with multiprocessing)` |
| S2 online readout | `s2_online_readout_v1.csv` | 120 | 3,564.5 s | 124.29 s | — | |
| S3 three-factor | `s3_three_factor_v1.csv` | 160 | 13,774.3 s | 431.21 s | — | |
| S8 integrated + N=1024 | `s8_integrated_v1.csv` | 56 | **43,210.7 s (12.00 h)** | **2,243.86 s (37.4 min)** | — | `NEW_ALGORITHM_PLAN.md:321` → `S8 完成（56 runs；N=256 10 seeds 消融 + N=1024 3 seeds 终确认；` |
| **S9 baseline showdown** | `s9_baseline_showdown_v1.csv` | 80 | **56,934.7 s (15.82 h)** | **3,216.24 s (53.6 min)** | **436.7 s** | `NEW_ALGORITHM_PLAN.md:506` → `**重跑结果**（80 runs，436.7s）` |
| s10 CV sweep | `s10_cv_sweep_v1.csv` | 90 | 984.7 s | 43.98 s | — | |
| s10 ESN metadata | `s10_esn_metadata_v1.csv` | 30 | 3,834.5 s | 212.89 s | — | |
| s11 disturbance chain | `s11_disturbance_chain_v1.csv` | 20 | 819.2 s | 54.81 s | 13.8 s (`--quick`) | `review_workspace/experiment_summary_report.md:13` |
| s12 λ_target sweep | `s12_lambda_target_sweep_v1.csv` | **120** | 8,189.6 s | 183.61 s | 992.4 s (`--quick`, 48 runs) | `review_workspace/experiment_summary_report.md:14`; `NEW_ALGORITHM_PLAN.md:477` → `5 seeds × 4λ × 3CV × 2 arms = 120 runs` |
| s13 causal audit | `s13_causal_audit_v1.csv` | 70 | 3,286.8 s | 52.61 s | **217.8 s** | `NEW_ALGORITHM_PLAN.md:493` → `**重跑结果**（7 臂 × 10 seeds，217.8s）`; `review_workspace/modification_log_scripts.md:434` → `全量重跑：7 臂 × 10 seeds = 70 runs，exit 0，217.8s；` |
| s13 single O4 run (before/after optimisation) | — | 1 | — | — | **604 s → 107.5 s** | `NEW_ALGORITHM_PLAN.md:483` → `单次 O4 运行 604s→107.5s（5.6×）`; `review_workspace/modification_log_scripts.md:130` → `单次 O4 运行时间从 604s 降至 107.5s（5.6x 加速）。` |
| s14 ESN chain | `s14_esn_disturbance_chain_v1.csv` | 30 | 1,171.3 s | 90.91 s | 107 s | `review_workspace/modification_log_scripts.md:207` → `全量运行：3 臂 × 10 seeds = 30 runs，107s，exit 0。` |
| s15 controlled adaptation | `s15_controlled_adaptation_v1.csv` | 30 | 1,622.2 s | 109.57 s | 132 s | `review_workspace/modification_log_scripts.md:285` → `全量（s15: 30 runs, 132s …）` |
| s16 τ_m pressure | `s16_tau_m_pressure_test_v1.csv` | 50 | — | — | 423 s | `review_workspace/modification_log_scripts.md:241` → `全量运行：50 runs（fast 10 + dual 40），423s，exit 0。` |
| s16b falsification stress | `s16b_falsification_stress_test_v2.csv` | 130 | — | — | 286 s | `review_workspace/modification_log_scripts.md:285` → `（s16b: 30 任务循环, 286s）` |
| s17 substrate stress | `s17_substrate_stress_v1.csv` | **120** | 2,564.1 s | 47.81 s | 175 s | `review_workspace/modification_log_scripts.md:379` → `全量（120 runs, 175s）exit 0` |
| s18 LLM drift gate | `s18_llm_drift_gate_v1.csv` | 90 | 894.2 s | 13.25 s | — | |
| s19 SSM P1/P3a | `s19_ssm_rls_readout_v1.csv` | 70 | 424.3 s | 19.53 s | — | |
| s20 M3 routing | `s20_ssm_m3_routing_v1.csv` | 90 | 410.1 s | 6.82 s | — | `review_workspace/modification_log_scripts.md:562` → `### 关键结果（10 seeds，90 runs，65s）` |
| s21 M4/M5 | `s21_ssm_m4_m5_v1.csv` | 70 | 249.0 s | 8.57 s | — | |
| s22 P4 benchmark | `s22_ssm_p4_benchmark_v1.csv` | 30 | 262.4 s | 20.85 s | — | |
| s23 real text | `s23_ssm_p4_realtext_v1.csv` | 30 | 128.7 s | 8.43 s | — | |
| s24 homeo×plasticity | `s24_homeo_plasticity_coupling_v1.csv` | 40 | 443.1 s | 15.04 s | — | |
| s25 reward-gated plasticity | `s25_reward_gated_plasticity_v1.csv` | 40 | 231.6 s | 9.69 s | — | |
| s26 fair TF | `s26_ssm_p4_fair_tf_v1.csv` | 90 | 1,049.6 s | 18.38 s | — | |
| s27 clip×κ fine grid | `s27_clip_kappa_fine_v1.csv` | **1,170** | 2,129.7 s | 4.11 s | — | |
| s28 causal audit chain | `s28_causal_audit_chain_v1.csv` | 60 | 9,711.8 s | 180.38 s | — | |
| s30 N=1024 replication | `s30_integrated_1024_v1.csv` | 20 | 2,386.8 s | 239.43 s | — | `README_REDEM.md:275` → `\| s30 N=1024 replication \| done (20 runs; 10 seeds, paired t=15.3) \|` |
| s31 bigram oracle | `s31_char_bigram_oracle_v1.csv` | 10 | — | — | — | |
| s32 FTLE noise | `s32_ftle_noise_v1.csv` | 50 | 8,135.0 s | 506.17 s | — | |
| s33 M5 in P4 | `s33_ssm_p4_m5_v1.csv` | 20 | 397.6 s | 21.09 s | — | |
| **s34 leak sensitivity** | `s34_leak_sensitivity_v1.csv` | 70 | **23,458.5 s (6.52 h)** | **935.65 s (15.6 min)** | — | |
| s35 boundary probe | `s35_readout_boundary_probe_v1.csv` | 10 | — | — | — | |
| **s36 graph variability** | `s36_topo_instance_variability_v1.csv` | **990** | **28,307.3 s (7.86 h)** | 130.71 s | — | `paper_a/README.md:83` → `…10 substrate seeds each (990 runs)` |
| s37 dormant-P probe | `s37_dormant_p_probe_v1.json` | 10 | 245.2 s | 27.11 s | — | |
| s39 prediction reward | `s39_prediction_reward_v1.csv` | 120 | 18,670.9 s | 1,608.51 s | — | |
| s40 meta flip | `s40_meta_flip_v1.csv` | 100 | 26,680.1 s | 1,360.75 s | — | |
| s41 meta reward | `s41_meta_reward_v1.csv` | 100 | 389.1 s | 6.40 s | — | |
| s42 corrupt signal | `s42_corrupt_signal_v1.csv` | 80 | 271.1 s | 6.24 s | — | |
| s43 population | `s43_population_v1.csv` | 120 | 597.3 s | 8.90 s | — | |
| s43b synthetic rotation | `s43b_rotation_v1.csv` | 30 | 74.5 s | 4.22 s | 11.3 s | `review_workspace/c_tier_rerun_report_E_v2.md:48` |
| s44 regime zoo | `s44_regime_zoo_v1.csv` | 240 | 1,678.0 s | 14.87 s | — | |
| s45 population R2R3 | `s45_population_r2r3_v1.csv` | **400** | 9,976.7 s | 50.38 s | — | |
| s46 rule adjudication | `s46_rule_adjudication_v1.csv` | 240 | 5,037.2 s | 48.00 s | — | |
| s47 self-correction gain | `s47_self_correction_gain_v1.csv` | 320 | 1,034.8 s | 7.55 s | 136.1 s | `review_workspace/c_tier_rerun_report_E_v2.md:60` |
| s48 rotation2d | `s48_rotation2d_v1.csv` | 160 | 858.6 s | 8.84 s | 156.7 s | `review_workspace/c_tier_rerun_report_E_v2.md:49` |
| s49 capacity probe | `s49_capacity_probe_v1.csv` | 160 | 312.1 s | 3.98 s | — | |
| s50 nonlinear capacity | `s50_nonlinear_capacity_v1.csv` | 120 | 471.4 s | 6.52 s | — | |
| **s50 Paper F P-F0** | `s50_paper_f_pilot_nl_readout_v1.csv` | 80 | 4,531.4 s | 164.38 s | **`elapsed_s = 63.06`** | `data/s50_paper_f_pilot_nl_readout_v1.json:56` → `"elapsed_s": 63.057164907455444,` |
| **s51 Paper F C4/C4b** | `s51_paper_f_learned_gate_v1.csv` | 80 | 4,982.8 s | 195.71 s | **`elapsed_s = 234.98`** | `data/s51_paper_f_learned_gate_v1.json:36` → `"elapsed_s": 234.97958207130432,` |
| **s52 Paper F C5** | `s52_paper_f_soft_route_experts_v1.csv` | 30 | 1,699.3 s | 60.11 s | **`elapsed_s = 124.42`** | `data/s52_paper_f_soft_route_experts_v1.json:15` |
| s51 quadratic circle | `s51_quadratic_circle_v1.csv` | 80 | 399.1 s | 7.97 s | — | |
| s52 dynamic memory | `s52_dynamic_memory_v1.csv` | 30 | 679.0 s | 45.96 s | 115.8 s | `review_workspace/c_tier_rerun_report_E_v2.md:59` |
| **s53 Paper F scaling** | `s53_paper_f_scaling_v1.csv` | 45 | 1,269.0 s | 84.18 s | **`elapsed_s = 409.06`** | `data/s53_paper_f_scaling_v1.json:15` |
| **s53b k-sweep** | `s53b_paper_f_ksweep_v1.csv` | 24 | 1,734.5 s | 78.90 s | **`elapsed_s = 870.25`** | `data/s53b_paper_f_ksweep_v1.json:10` → `"elapsed_s": 870.2541151046753,` |
| s53 auto basis | `s53_auto_basis_v1.csv` | 80 | 849.2 s | 17.88 s | 119.2 s | `review_workspace/c_tier_rerun_report_E_v2.md:50` |
| **s54 Paper F Mackey-Glass** | `s54_paper_f_mackey_glass_v1.csv` | 15 | 113.2 s | 12.52 s | **`elapsed_s = 43.31`** | `data/s54_paper_f_mackey_glass_v1.json:6` |
| s54 memory boundaries | `s54_memory_boundaries_v1.csv` | 180 | 1,677.2 s | 12.41 s | 220.7 s | `review_workspace/c_tier_rerun_report_E_v2.md:51` |
| s55 sphere capacity | `s55_sphere_capacity_v1.csv` | 100 | 409.9 s | 7.61 s | — | |
| s56 joint self-evolution | `s56_joint_self_evolution_v1.csv` | 100 | 811.7 s | 17.38 s | — | |
| s57 dimension stress | `s57_dimension_stress_v1.csv` | 260 | 1,103.8 s | 7.19 s | — | |
| s58a complexity trigger | `s58a_complexity_trigger_v1.csv` | 240 | 1,318.7 s | 8.21 s | — | |
| **s58b relative sense** | `s58b_relative_sense_stress_v1.csv` | 240 | 7,836.2 s | 655.88 s | **1000.9 s (16.7 min)** | `review_workspace/c_tier_rerun_report_E_v2.md:43` → `最长单脚本 1035.6 s（s58f，17.3 min）、次长 1000.9 s（s58b，16.7 min）` |
| s58c multi-level | `s58c_multi_level_expansion_v1.csv` | 80 | 7,203.8 s | 326.92 s | 918.5 s | `review_workspace/c_tier_rerun_report_E_v2.md:62` |
| s58d kernel capacity | `s58d_kernel_capacity_sweep_v1.csv` | 200 | 1,452.9 s | 18.69 s | 200.7 s | `review_workspace/c_tier_rerun_report_E_v2.md:63` |
| s58e D2 timescale | `s58e_d2_timescale_v1.csv` | 660 | 2,491.5 s | 9.60 s | 322.8 s | `review_workspace/c_tier_rerun_report_E_v2.md:64` |
| **s58f margin sensitivity** | `s58f_margin_sensitivity_v1.csv` | 240 | 8,150.5 s | 431.06 s | **1035.6 s (17.3 min)** | `review_workspace/c_tier_rerun_report_E_v2.md:43`; `paper_e/README.md:189` → `scripts (s58b ~17 min, s58f ~17 min, s62 ~2.5 min) must therefore be` |
| s59 self-tuned gate | `s59_self_tuned_gate_v1.csv` | 60 | 426.8 s | 10.55 s | — | |
| s5 dual timescale | `s5_dual_timescale_v1.csv` | 100 | 9,039.7 s | 251.32 s | — | |
| s5b controlled adaptation | `s5b_controlled_adaptation_v1.csv` | 100 | 1,671.3 s | 51.51 s | 116 s | `review_workspace/modification_log_scripts.md:321` → `全量（100 runs, 116s）exit 0` |
| s60 adaptive window | `s60_adaptive_window_v1.csv` | 160 | 1,519.9 s | 18.85 s | **198.7 s** | `review_workspace/modification_log_scripts.md:2527` → `**全量**（10 seeds，160 runs，8 进程 Pool）：**198.7 s / exit 0**（16:41:34 → 16:44:53）` |
| s61 cross-family | `s61_cross_family_v1.csv` | 40 | 215.1 s | 9.68 s | — | |
| s62 adaptive meta | `s62_adaptive_meta_v1.csv` | 240 | 1,101.2 s | 9.75 s | 149.1 s (~2.5 min) | `review_workspace/modification_log_E_v2.md:198`; `paper_e/README.md:189` |
| s63 content rendering | `s63_content_rendering_v1.csv` | **320** | 750.3 s | 18.57 s | — | `paper_e/README.md:199` → `Full 10-seed (320 runs) results are in` |
| s64 multi-source | `s64_multi_source_validation_v1.csv` | **500** | 518.0 s | 1.88 s | — | `paper_e/README.md:218` → `Full 10-seed (500 rows) results are in` |
| s65 EWC baseline | `s65_ewc_baseline_v1.csv` | 40 | 226.5 s | 10.28 s | — | |
| s6 chaos regulator | `s6_chaos_regulator_v1.csv` | 200 | 5,448.8 s | 79.46 s | — | |
| s7 structure plasticity | `s7_structure_plasticity_v1.csv` | 70 | 367.3 s | 12.13 s | — | |
| figures (Paper E) | — | 1 | — | — | **3.7 s** (6 PDFs) | `review_workspace/modification_log_E_v2.md:207` |
| `gen_paper_figures.py` etc. | — | — | — | — | not recorded | |

### 3.3 Total experimental compute (derivable)

```
74 committed (non-_quick, non-.bak) CSVs with a runtime_s column:
   runs  = 11,550
   Σruntime_s = 356,940.2 s = 99.15 h = 4.13 days   (single-core serial)

40 _quick screening CSVs (not part of the reported results):
   runs = 1,267 ; Σ = 11,527.5 s = 3.20 h
```
This is **serial** time. With `Pool(min(cpu_count(), n_runs))` the wall clock is far smaller —
e.g. S9's 15.8 h of serial GRU/transformer training finished in **436.7 s** on a pool, and
S1's 610 runs finished in *"~1 min"*.

**Longest single run in the entire program (one run, wall clock):**
`s9_baseline_showdown_v1.csv`, `task=mackey_glass, system=gru, seed_idx=1` → **3,216.24 s
(53.6 min)**, and the same config at `seed_idx=0` → 3,204.8 s. The next-longest are
`drift_binary/gru` runs at ~2,400–2,670 s. This is the offline-trained torch GRU
(`TRAIN_EPOCHS = 20`, `BATCH = 128`, `t_total = 21000/40000`), **not** the N=1024 substrate —
the N=1024 substrate's longest single run is `s8_integrated_v1.csv arm=full seed_idx=0
n_units=1024` at **2,243.86 s (37.4 min)**, which the code itself calls out:

> `scripts/integrated_benchmark.py:257-258` — `# N=1024 scale confirmation (3 seeds; the 2049-dim RLS is the` / `# runtime bottleneck ~2min/run, so keep the confirmation lean)`

---

## 4. Real natural-language text

### 4.1 Answer: exactly two Gutenberg books, nothing else. Ever.

A greedy search over the whole repo returns **zero** hits for `wikitext|WikiText|tokenizer|
AutoTokenizer|transformers|torchtext|nltk|load_dataset|datasets.|urlopen|requests.get|wget`
in any `*.py` under `scripts/`. The only `*.txt` files in the project tree are
`requirements.txt` and files inside `.venv/Lib/site-packages/`.

**`data/corpora/` contains exactly three files:**

| File | Bytes | Characters | Identity |
|---|---|---|---|
| `data/corpora/alice.txt` | 154,482 | **144,604** | Lewis Carroll, *Alice's Adventures in Wonderland*, Project Gutenberg ebook **#11** |
| `data/corpora/dickens.txt` | 787,412 | **757,610** | Charles Dickens, *A Tale of Two Cities*, Project Gutenberg ebook **#98** |
| `data/corpora/README.md` | 863 | — | the corpus README |

`data/corpora/README.md` (whole file is 18 lines) states verbatim:

> line 1: `# Real-text corpus for the Paper D real-text benchmark (S23)`
> lines 3-6: `Two public-domain English texts, downloaded from Project Gutenberg` / `(https://www.gutenberg.org), header/footer license blocks stripped, body` / `only. Both are public domain in the United States (Project Gutenberg` / `License applies to the electronic versions).`
> lines 10-11: `| `alice.txt` | Lewis Carroll, *Alice's Adventures in Wonderland* (ebook #11) | ~148k |` / `| `dickens.txt` | Charles Dickens, *A Tale of Two Cities* (ebook #98) | ~774k |`
> lines 13-15: `Usage: `scripts/s23_ssm_p4_realtext.py` builds a two-source character-level` / `streaming domain-drift task (alternating windows, known switch instants,` / `s18 protocol) with a 32-symbol vocabulary (31 most frequent chars + UNK).`

The README's size figures are approximate/rounded up (~148k vs actual 144,604; ~774k vs
actual 757,610).

### 4.2 Tokenisation (verbatim from `scripts/s23_ssm_p4_realtext.py`)

> line 78: `V = 32                       # 31 frequent chars + UNK (matches VOCAB=32)`
> lines 88-103:
> ```python
> def load_corpus():
>     """Returns (alice_encoded, dickens_encoded, mapping, unk_idx)."""
>     with open(os.path.join(CORPUS_DIR, 'alice.txt'), encoding='utf-8') as f:
>         alice = f.read()
>     with open(os.path.join(CORPUS_DIR, 'dickens.txt'), encoding='utf-8') as f:
>         dickens = f.read()
>     counts = Counter((alice + dickens).lower())
>     top = [ch for ch, _ in counts.most_common(V - 1)]
>     mapping = {ch: i for i, ch in enumerate(top)}
>     unk = V - 1
> 
>     def enc(text):
>         return np.array([mapping.get(ch, unk) for ch in text.lower()],
>                         dtype=np.int64)
> ```

**Exact tokenisation facts (computed directly from the files):**

* Method: **character-level**, `text.lower()`, **no BPE, no SentencePiece, no HF tokenizer, no
  word segmentation**. Vocabulary built from the *concatenation of both books* lowercased.
* **Vocabulary size = 32** (31 most-frequent characters + `UNK` index 31).
* Distinct lowercased characters in the two books combined: **57** → 26 character types are
  mapped to `UNK`.
* The 31 most-frequent characters, in rank order, are:
  `[' ', 'e', 't', 'a', 'o', 'n', 'i', 'h', 's', 'r', 'd', 'l', 'u', '\n', 'm', 'w', 'c', ',', 'f', 'g', 'y', 'p', 'b', '.', 'v', 'k', U+201C, U+201D, '-', U+2019, '!']`
  — the three non-ASCII symbols are the curly quotes `“` `”` `’`.
* Total corpus = **902,214 characters** (144,604 + 757,610), of which **7,282 (0.81 %) map to
  `UNK`** → 99.19 % character coverage. (The other non-ASCII types — `è`, `é`, `ù`, `—`, `‘` —
  fall below the top-31 cutoff and become `UNK`.)
* Window usage: `"6 alternating segments of 3000 chars each … 18k chars"`
  (`s23_ssm_p4_realtext.py:15-16`), so **each run sees exactly 18,000 tokens**, drawn from
  per-seed random start positions (`gen_real_stream`, lines 106-119).

Confirmed by the committed data:
`data/s23_ssm_p4_realtext_v1.json:3-5` →
`"desc": "two public-domain books (Alice vs Dickens), char-level, case-folded, 32-symbol vocab (31 + UNK), 6 x 3000 alternating segments, known switch instants, per-seed window positions", "corpus": "data/corpora/{alice,dickens}.txt (Gutenberg, public domain)"`;
and by the stream length in `data/per_token/s23__SSM-bare__rt__s0.npz` → `stream_ce` shape `(18000,)`.

### 4.3 Everywhere else: synthetic, not real text

* `s18_llm_drift_gate.py:97-113` (`gen_transition`) — a **two-state biased-bigram Markov chain**
  over `VOCAB=32` symbols: `"""P(next | prev) = p_shift * onehot((prev+shift) mod VOCAB) + (1-p_shift) * q …"""`. Not text.
* `s19`/`s20`/`s21`/`s33`/`s50`/`s51`/`s52`/`s53`/`s53b` — all reuse the same synthetic bigram
  chain (`s19:137` header: `# ========================== Task (verbatim from s18, Paper C Sec 6) ==========`).
* `s22_ssm_p4_benchmark.py:14-16` — four biased-bigram Markov domains, `~24k tokens`.
* `s31_char_bigram_oracle.py` — a count-based bigram **oracle** over the same two real books
  (`fit_bigram`, lines 68-76; `ALPHA = 1.0` Laplace smoothing, line 60); uniform baseline
  `ppl = VOCAB = 32` (line 156). This is the only other script that reads real text, and it
  reads the same two files.
* `s54_paper_f_mackey_glass.py` — real **numerical** data (Mackey–Glass ODE, `mackey_glass()`
  lines 71-84), binned into 32 integer bins (`bin_series`, 87-93). Not language.

The repo says so itself:

> `scripts/s22_ssm_p4_benchmark.py:38-40` — `Honest scope: no WikiText-103, no scaling claims; the real-text corpus` / `benchmark is deferred (no external text in the repo - user decision` / `needed); the transformer routing reference (TF-A3) is limited to 2`

---

## 5. Largest single simulation run

Ranked by number of pulses / tokens / steps in **one run** (max recorded `t_total`):

| Rank | Pulses / tokens / steps | Script | Environment | Anchor |
|---|---|---|---|---|
| **1** | **80,000 pulses** | `scripts/s59_self_tuned_gate.py` | `env='static'`: `STATIC_BLOCKS = 4000` blocks × `DB_K_PULSES = 20` | `s59:142` `STATIC_BLOCKS = 4000`; `paper_e/deps/streaming_tasks.py:66` `DB_K_PULSES = 20            # pulses per block (constant interval within block)`; confirmed `max_t_total = 80000` in `data/s59_self_tuned_gate_v1.csv` |
| 2 | **60,000 pulses** | `scripts/s58b_relative_sense_stress.py`, `scripts/s58c_multi_level_expansion.py` | 3000 blocks × 20 pulses | `max_t_total = 60000` in `data/s58b_relative_sense_stress_v1.csv`, `data/s58c_multi_level_expansion_v1.csv` |
| 3 | **50,000 pulses** | `scripts/s61_cross_family_transfer.py` | 2500 blocks × 20 | `data/s61_cross_family_v1.csv` |
| 4 | **48,000 pulses** | `scripts/s57_dimension_stress.py`, `s58a_complexity_trigger.py`, `s58d_kernel_capacity_sweep.py` | `ROT4D_K_PULSES = 24` per block (`paper_e/deps/streaming_tasks.py:243`) | `data/s57…csv`, `s58a…csv`, `s58d…csv` |
| 5 | **42,000 pulses** | `scripts/s55_sphere_capacity.py` | | `data/s55_sphere_capacity_v1.csv` |
| 6 | **40,000 pulses** | ~30 scripts (s3, s39–s54, s56, s60, s62, s63, s64, s65, s9, s2) | 2000 blocks × 20 (`K=20 pulses per block so T=40000`, `s65_ewc_baseline.py:21`) | `max_t_total = 40000` across those CSVs |
| 7 | **22,790 tokens** | `scripts/s22_ssm_p4_benchmark.py` / `s26` / `s33` | `N_SEGMENTS = 8` × U[2500,3500] tokens | `data/per_token/s22__SSM-bare__p4__s0.npz` → `stream_ce` shape `(22790,)`; header `s22:13-15` `8 segments, ~24k tokens` |
| 8 | **21,000 samples** | `scripts/baseline_showdown.py` GRU, Mackey–Glass | `t_total=21000` | `data/s9_baseline_showdown_v1.csv` row `task=mackey_glass, system=gru` |
| 9 | **20,000 tokens** | `scripts/s54_paper_f_mackey_glass.py` | `N_TRAIN = 18000` + `N_HOLD = 2000` | `s54:60-61` |
| 10 | **18,000 tokens** | `s18`, `s19`, `s20`, `s21`, `s23` (**incl. the real-text run**), `s50`, `s51`, `s52`, `s53`, `s53b`, `s54_paper_f` | 6 segments × 3000 | `s18:68-70` `SEG_LEN = 3000` / `N_SEGMENTS = 6` / `T_TOTAL = SEG_LEN * N_SEGMENTS`; `s23:75-76`; `data/per_token/s23__SSM-bare__rt__s0.npz` → `(18000,)` |
| — | **1,200 pulses** | `scripts/substrate_recurrence_characterization.py` (S1) | `T_TOTAL = 1200        # pulses per trajectory` | `substrate_recurrence_characterization.py:75`; `data/substrate_phase_diagram_v2.csv` `max_t_total = 1200` |
| — | **9,000 pulses** | `scripts/s5_dual_timescale_metadata.py` | | `data/s5_dual_timescale_v1.csv` |

**Verdict for Q5: the largest single simulation run is 80,000 pulses**
(`s59_self_tuned_gate.py`, static single-regime environment, N = 256 units, 4,000 blocks ×
20 pulses/block, `data/s59_self_tuned_gate_v1.csv`). The largest **real-text** run is
**18,000 characters** (`s23`, and identically in every s18-protocol script).

---

## 6. Things that could NOT be found / caveats

1. **No `N_TAUS`, `N_DEV` constant exists.** Zero grep hits anywhere. τ is always drawn as a
   vector (`gen_tau_vec(n_units, CV_TAU, tau0, seed)`).
2. **Wall-clock (`elapsed_s`) is recorded for only 6 Paper F scripts.** For the other ~55
   result CSVs there is no wall-clock field at all; Σ`runtime_s` is the only quantitative
   proxy, and it is serial CPU time, not wall clock.
3. **`--quick` runs are documented inconsistently.** `paper_f/PAPER_F_PLAN.md:172-173` claims
   `data/s50_paper_f_pilot_nl_readout_v1.csv | 80 rows` (matches) but
   `data/s51_paper_f_learned_gate_v1.csv | 60 rows` — the actual file has **80 rows** (8 arms ×
   10 seeds), so the plan's row count is wrong.
4. **`docs`/paper texts do not state N for several papers.** `paper_d/README.md` never states
   N = 128 numerically; the value appears only in `paper_d/PAPER_D_sketch.md:133` and in the
   data (`data/s19_ssm_rls_readout_v1.json:5` `"state_dim": 128`).
5. **A 16,384-unit configuration exists as dead code**, not as a simulated result:
   `scripts/shallow_trap_array_simulator.py:264` — `for N in [64, 256, 1024, 4096, 16384]:`
   inside `if __name__ == "__main__":`, writing `data/square_array_results.csv`, **which is not
   in the repo**, and which `review_workspace/agent_f_result_E_v2_part7.md:3` explicitly records
   as having no provenance chain.
6. **Only torch NN models in the entire program:** `baseline_showdown.py` (`GRUModel`,
   `TinyTransformer`) and `s18_llm_drift_gate.py` (`TinyCharLM` + `LoRALinear`, reused by
   `s22`, `s23`, `s26`, `s31`). Grep for every `nn.*` layer type confirms this.
