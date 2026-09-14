# Paper F — Plan under Paradigm Rules R1–R4

> Authority: `paper_f/PARADIGM_AND_RULES.md`. Status: v1.1 — C1–C5 + s53 + **s53b k-sweep** + **s54 MG**.
> N=512 failure is **k∝N**, not width: k=16/32 restores C4b/C5. MG transfers sparse-gate win.
> D abstract/conclusion updated (F companion note). Git commit still pending.
> Language: planning in Chinese+English terms; all code artifacts English.

---

## 0. Role of F (fixed by the paradigm)

| | |
|---|---|
| **Not** | another mechanism demo; "Prop 1 beaten"; "state resurrects under softmax"; architecture SOTA |
| **Is** | first paper written *as a rule*: the evaluation instrument must be **learned**, not prescribed |
| Paradigm | Self-improvement requires trustworthy self-evaluation; the instrument must be learnable |
| Layer | Readout / measurement (Trap 1 from D); decision-layer E machinery is **not** ported (R3/R4) |
| Feedback budget (R4) | **per-token label** (strictly richer than E's ±1); price measured in retention |

F sits between D (prescribed linear+clip readout) and E (prescribed correction laws). It crosses
the boundary D only *flagged* (learned selectivity) **after** implementing D's own calibrated-output
prescription, and it reports the post-hoc control that makes the prescription a mechanism claim.

---

## 1. Propositions C1–C4 (pre-registered; numbers from committed data)

| ID | Proposition | Evidence | Status |
|----|-------------|----------|--------|
| **C1** | Artifact is in the **objective**, not the numeric range: trained probabilistic readout removes floor and lowers stream CE by **0.307 nats/token** (ppl −26%), no clip / no floor / no post-process. **Price in the same sentence**: retention same-set dppl **+0.32 (1/10)**; mean forget 8.89 vs 8.39 | s50 B-softmax vs B-lin-clip; 10/10, t=−92.3 | **DONE** |
| **C2** | **Post-hoc calibration does not repair** — the falsifying control that turns C1 into a mechanism claim. Simplex removes every negative (`neg_frac` 1.582%→0) and stream dppl **+0.03 (0/10)** | s50 B-simplex / Skip-simplex | **DONE** |
| **C3** | Under a clean metric the boundary **moves but does not open**: full state stream-neutral (+0.0029, 5/10, t=0.22) and retention-harmful (−1.348, 10/10, t=−6.73); **dimension/optimizer excluded** (128-d noise: stream −0.172 worse, forget −0.075 near-zero; 32-d substate: stream **+0.061 better 9/10**, forget −0.176 mild) | s50 + P-F0b | **DONE** |
| **C4** | Unconstrained CE gate improves stream form but does not sparsify | s51 Gate-C: stream 7.741 (10/10) forget 31.1 (0/10) | **DONE partial** |
| **C4b** | Sparsity pressure (top-k=32) keeps stream win **and** buys retention | Gate-C-topk stream **7.033**; forget **9.201** | **SUPPORTED** |

### C4b / M2b result (s51, 10 seeds)

| Arm | stream | forget mean/med | vs B stream | vs Skip-sub stream | vs B forget |
|-----|--------|-----------------|-------------|--------------------|-------------|
| B-softmax | 8.648 | 8.890 / 8.870 | — | worse | — |
| Skip-sub | 8.142 | 11.231 / 9.908 | better | — | worse |
| Gate-C (no sparse) | 7.741 | 31.075 / 26.155 | better 10/10 | better 10/10 | worse 0/10 |
| **Gate-C-topk** | **7.033** | **9.201 / 9.330** | **better 10/10** (−1.62) | **better 10/10** (−1.11) | 3/10 better; mean **+0.31 ≈ B** |
| Gate-C-L1 | 6.994 | 11.574 / 11.136 | better 10/10 | better 10/10 | worse 0/10 (+2.68) |

Pre-registered C4b gate: sparse arm beats Skip-sub on stream **and** forget ≤ Skip-sub on ≥7/10.
**Gate-C-topk: 10/10 stream, 7/10 forget ≤ Skip-sub → PASS.** L1 is stream-best but retention stays near Skip-sub, not B.

**Price sentence (rule 3):** Gate-C-topk buys −1.62 ppl stream vs B-softmax at near-parity retention (+0.31 mean forget, 3/10); unconstrained Gate-C bought the same stream direction at +22 forget.

| **C5** | Soft routing over online softmax experts on the Gate-C-topk base improves retention without giving up stream | s52 | **SUPPORTED (retention 10/10; stream parity)** |

### C5 / M3 result (s52, 10 seeds, τ_m=500)

| Arm | stream | forget mean/med | vs Single stream | vs Single forget |
|-----|--------|-----------------|------------------|------------------|
| Single-topk (= C4b control) | 7.033 | 9.201 / 9.330 | — | — |
| **Soft-topk** | 7.065 | **7.789 / 7.717** | +0.03 (6/10, t=+0.68, **parity**) | **−1.41, 10/10, t=−10.4** |
| Hard-topk | 7.111 | **6.939 / 6.913** | +0.08 (2/10, small stream cost) | **−2.26, 10/10, t=−10.3** |

- Pre-reg: Soft improves forget ≥7/10 **and** stream not worse ≥7/10 → forget **10/10 PASS**; stream is statistical **parity** (mean +0.03, t=0.68), not a 7/10 win. Report as **retention-supported, stream-neutral**.
- **Full stack vs B-softmax**: Soft-topk stream 7.065 vs 8.648; forget 7.789 vs 8.890 — **both axes better** (recompute paired vs s51/s50 for the paper table).
- Hard routing is the retention specialist (D s20 lesson survives under softmax) but pays a small stream tax; Soft is the balanced default.
- Data: `data/s52_paper_f_soft_route_experts_v1.csv` (30 rows).

### C4 result so far (s51, 10 seeds) — must stay in every summary

| Arm | stream | forget mean/med | vs B stream | vs Skip-sub stream | vs B forget |
|-----|--------|-----------------|-------------|--------------------|-------------|
| B-softmax | 8.648 | 8.890 / 8.870 | — | worse | — |
| Skip-sub (32-d) | 8.142 | 11.231 / 9.908 | better | — | worse |
| B-noise | 10.271 | 9.554 / 9.480 | worse | worse | worse |
| **Gate-C** | **7.741** | 31.075 / 26.155 | **better 10/10** | **better 10/10** | **worse 0/10** |
| Gate-static | 8.152 | 26.515 / 22.996 | better 9/10 | ≈ | worse 0/10 |
| Gate-elem | 8.693 | 59.533 / 36.985 | ≈ | worse | worse 0/10 |
| **Gate-C-topk** | **7.033** | 9.201 / 9.330 | better 10/10 | better 10/10 | ≈ B (mean +0.31) |
| Gate-C-L1 | 6.994 | 11.574 / 11.136 | better 10/10 | better 10/10 | worse 0/10 |

**Interpretation under R1–R3:** the objective discovers a better stream form through the gate (R3).
Pure CE does **not** close the gate (`gate_mean`≈0.72). **Hard top-k sparsity is the missing
pressure**: it keeps the stream win and removes the wide-state retention toxin (C4b).

---

## 2. Reporting rules F must obey (non-negotiable)

1. **Never** quote softmax `floor_frac=0` as a result — true by construction.
2. Keep `clip_frac` / `neg_frac` / `floor_frac` as **separate columns**.
3. **Every** C1-style claim ships with the **post-hoc simplex arm** in the same table (C2).
4. Every upside number gets its **price in the same sentence** (stream ↔ retention).
5. **Autonomy language** only where a *form* is learned (gate, readout map), never where a rule
   among supplied rules is selected, and never without stating the **feedback budget** (R4:
   per-token labels here; ±1 is E's budget).
6. Do **not** port E's five-mechanism stack onto this host (R3: gradient exists; that stack's
   reason for existing is absent). Contrast E vs F as a **feedback-budget trade**, not a port.

---

## 3. Why routing after M2b (decision, executed)

| Option | Fits paradigm? | Note |
|--------|----------------|------|
| **M2b sparse gate (C4b)** | **Yes** — tests whether R3 + sparsity closes C4's retention hole | Direct falsification of "gate is useless for retention" |
| C5 soft routing experts | **Yes, after C4b** | Specialize a *sparse* state slice; executed in s52 |
| Port E autonomy stack | **No** (R3/R4) | Predicted to underperform; re-run not contribution |

**Order:** M2b → re-evaluate C4 → only then C3 routing / scaling / std-benchmarks.

---

## 4. M2b design (implement in `s51`, no new protocol)

Same host, task, seeds, AdaGrad CE as s51. Two new arms only:

| Arm | Gate | Sparsity pressure |
|-----|------|-------------------|
| `Gate-C-topk` | `g = σ(C x + b)` then keep **top-k=32** entries, zero the rest (k = SUBSTATE_DIM) | hard cardinality |
| `Gate-C-L1` | same `g`; add λ‖g‖₁ to loss → extra grad `λ · sign(g) · g(1-g)` on pre-activation | soft L1 |

**Falsification (C4b):** fail if no sparse arm beats Skip-sub on stream **and** stays within
Skip-sub retention (forget ≤ Skip-sub on ≥7/10), or if both are worse than Gate-C on stream and
still worse than B on forget.

Baselines in the same run file: B-softmax, Skip-sub, Gate-C (recomputed for pairing).

---

## 5. Host / mechanism map (unchanged from v0.4, scoped by C1–C4)

```
h_t = A ⊙ h_{t-1} + B e_{t-1}
h^w_t = h_t ⊙ sqrt(N(1-A_i²))
x_t = B e_{t-1}                         # clean input path (D: required)
g_t = gate(x_t or h^w_t or static)      # C4: learned; C4b: + sparsity
φ_t = [ h^w_t ⊙ g_t ; x_t ; 1 ]
p_t = softmax(W φ_t)                    # C1: trained, not post-hoc
```

M3 fast-EMA routing (s20) remains the metadata form; M5 regulator deferred until C4b clears.

---

## 6. Paper form (updated after C4 partial)

| After C4b | Form |
|-----------|------|
| **C4b success (current)** | **Methodological + architecture paper**: R1–R3 instantiated; D prescription completed with C2 control; sparse learned selectivity + soft experts both improve over B-softmax. Strong Neurocomputing/TMLR candidate. |
| C3 routing later succeeds | Upgrade with multi-domain retention; keep E contrast as feedback-budget table |

**C2 alone is never the paper form** — it is D's prescription executed and controlled.

---

## 7. Discussion must state (F+1 boundary)

From `PARADIGM_AND_RULES.md` §5: after trustworthy self-evaluation, the ceiling is **candidate-space
breadth**. F learns form *inside* a supplied family (softmax readout + gate family). Generating new
candidate families is out of scope and is the next paper's problem.

Evidence gaps (declare, no exceptions): two-domain biased-bigram; N=128 CPU; no Mamba/TTT baseline;
no scaling until s53; no multi-step RL.

---

## 8. Script / data map

| Artifact | Role | Status |
|----------|------|--------|
| `PARADIGM_AND_RULES.md` | Rules R1–R4, traps D/E, C1–C4 | authority |
| `scripts/s50_paper_f_pilot_nl_readout.py` | C1, C2, C3, P-F0b controls | done |
| `scripts/s51_paper_f_learned_gate.py` | C4 + **C4b (M2b)** | C4 done; C4b next |
| `data/s50_paper_f_pilot_nl_readout_v1.csv` | 80 rows | untracked → commit |
| `data/s51_paper_f_learned_gate_v1.csv` | 60 rows | untracked → commit |
| `review_workspace/modification_log_paper_f.md` | Chinese log | append M2b |
| s52 routing / s53 scale / s54 benchmarks | after C4b | planned |
| D.tex L76–79, L1043–1046 reword | scope floor to squared-loss+clip family | **needs user confirm** |

---

## 9. Milestones

| ID | Deliverable | Exit |
|----|-------------|------|
| M1/M1b | s50 C1–C3 + controls | done |
| M2 | s51 C4 unconstrained gates | done partial |
| **M2b** | Gate-C-topk + Gate-C-L1 | **done — C4b SUPPORTED** |
| M2c | C4 verdict rewrite in PARADIGM §4 | **done** |
| M3 | C3 `s52` soft/hard routing on top-k | **done — retention 10/10; stream parity** |
| M4 | scale (s53 N=128/256/512, 5 seeds) | **done** — C1 scales; C4b/C5 fail to hold at N=512 |
| M5 | abstract + tex skeleton | **done** — `PAPER_F.tex` full draft from sketch |
| — | git commit of F artifacts | **user confirm** |
| — | D.tex reword (L76–79, L1043–1046) | **user confirm** |
| — | pdflatex compile + polish | after user env / Overleaf |
| — | k-sweep / scaling / MG-NARMA | optional strengtheners |

---

## 10. Open defaults

| # | Default | Alt |
|---|---------|-----|
| F-D1 | softmax T=1.0 | learnable |
| F-D2 | AdaGrad 0.05 + clip 5.0 | — |
| F-D6 | noise seed off +9 | — |
| **F-D7** | **top-k = 32** | k ∈ {16, 64} sweep |
| **F-D8** | **L1 λ = 0.01** on gate pre-grad | λ ∈ {0.001, 0.05} |
