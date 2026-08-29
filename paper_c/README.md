# Paper C — Dissecting Online Learning Mechanisms: Statistical Memory, Homeostatic Recovery, and Substrate Physics are Non-Substitutable

Paper C of the four-paper REDEM series (**role: dissection**). It dissects
the online learner into three mechanisms — the slow-trace statistical memory
(M3), the homeostatic chaos regulator (M5), and the raw substrate physics —
and uses falsifying transfer experiments to establish which mechanism is
responsible for which capability, and where each one stops.

- PDF: [`PAPER_C.pdf`](PAPER_C.pdf) | LaTeX: [`PAPER_C.tex`](PAPER_C.tex)
- Preprint: [doi:10.5281/zenodo.22110618](https://doi.org/10.5281/zenodo.22110618)
- Target: *Neurocomputing* (Elsevier)
- Series overview and reading order (A → B → C → D):
  [`../README.md`](../README.md)

## Key results

- The +32% sequential-recovery gain of the substrate system is produced by
  the homeostat, not by the metadata (falsifying transfer experiment, ESN
  with and without the metadata under a sequential disturbance chain,
  10 seeds).
- The metadata does not transfer memory-capacity robustness to an ESN at any
  tested timescale (τ_m ∈ {200, 500, 1000, 2000}; paired differences −0.68
  to −0.70, 0/10 seeds positive).
- The metadata's transferable value is narrower and precisely located: it
  denoises state-level corruption (gap −0.69 → −0.04…−0.01 at every tested
  τ_m) and accelerates boundary adaptation by a controlled-measured factor
  of 1.3–2.4 on the substrate and ~10 pulses on the ESN.
- Raw memory capacity remains a substrate property (MC 10.19 vs. 6.17).
- Transformer proof of concept: slow-trace domain routing transfers
  (forgetting up to −28%, 10/10 seeds) while a gating-only variant is
  falsified at every τ_m (0/10 seeds).

## Contents

| File | Purpose |
|---|---|
| `PAPER_C.tex` | LaTeX source (elsarticle preprint class) |
| `PAPER_C.pdf` | Compiled PDF (14 pages) |

## Compile

From this directory (figures resolve via `../figures/`):

```powershell
pdflatex PAPER_C.tex    # run three times for cross-references (elsarticle)
```

## Figures and data anchors

| Item | File | Data |
|---|---|---|
| Fig. 1 slow-trace vs. material kernel | `../figures/paperC_fig1_kernel.pdf` | `../data/forgetting_curve_theory_overlay_v1.json` |
| Fig. 2 post-disturbance recovery vs. τ_m | `../figures/paperC_fig2_recovery.pdf` | `../data/s16_tau_m_pressure_test_v1.*` |
| Fig. 3 LLM drift-gate PoC | `../figures/paperC_fig3_llm.pdf` | `../data/s18_llm_drift_gate_v1.*` |
| S10 equalization | — | `../data/s10_esn_metadata_v1.*` |
| S11 substrate chain anchor | — | `../data/s11_disturbance_chain_v1.*` |
| S14 transfer test | — | `../data/s14_esn_disturbance_chain_v1.*` |
| S15 controlled adaptation | — | `../data/s15_controlled_adaptation_v1.*` |
| S5b substrate arms controlled | — | `../data/s5b_controlled_adaptation_v1.*` |
| S16 τ_m pressure test | — | `../data/s16_tau_m_pressure_test_v1.*` |
| S16b falsification stress test | — | `../data/s16b_falsification_stress_test_v2.*` |
| S17 substrate stress | — | `../data/s17_substrate_stress_v1.*` |
| S18 LLM drift gate | — | `../data/s18_llm_drift_gate_v1.*` |

## Reproduce

CPU-only; uses the project virtual environment (`../.venv`). Run from this
directory:

```powershell
& ..\.venv\Scripts\python.exe ..\scripts\<script> [flags]
```

Experiment scripts accept `--quick` (reduced smoke run); `--sequential`
(where listed) disables the multiprocessing `Pool`. Each run regenerates the committed `../data/` files. The table below is the
experiment index: what each script does, what it writes, and the finding it
establishes (with the paper location).

| Script | What it does | Writes | Finding (paper anchor) |
|---|---|---|---|
| `esn_metadata_comparison.py` | S10: the slow metadata trace transplanted into a matched ESN on the statistical task | `../data/s10_esn_metadata_v1.{csv,json}` | ESN+meta 0.998 ≈ ESN 0.996, closing the gap to REDEM 0.994 — statistical memory is substrate-agnostic (Table 1) |
| `s11_disturbance_chain.py` | E3: substrate sequential-disturbance chain (τ-drift → edge prune → readout noise), regulated vs. fixed κ | `../data/s11_disturbance_chain_v1.{csv,json}` | Substrate anchor: regulated r3 MC 8.47 ± 0.39 vs. fixed 6.41 ± 0.41 — the +32% the transfer tests must explain |
| `s14_esn_disturbance_chain.py --sequential` | The falsifying transfer test: ESN with and without metadata under the same disturbance chain (3 arms, 10 seeds) | `../data/s14_esn_disturbance_chain_v1.{csv,json}` | Paired differences −0.78/−0.76/−0.69 at r1/r2/r3, 0/10 seeds positive — metadata does NOT transfer robustness; the +32% belongs to the homeostat |
| `s15_controlled_adaptation.py --sequential` | Controlled adaptation protocol with known switch instants (10 seeds) | `../data/s15_controlled_adaptation_v1.{csv,json}` | True boundary-adaptation effect ~10 pulses; variance collapse p90 76.5 → 42 — the metadata accelerates and stabilizes adaptation |
| `s16_tau_m_pressure_test.py --sequential` | τ_m ∈ {200, 500, 1000, 2000} pressure test of the falsification (10 seeds) | `../data/s16_tau_m_pressure_test_v1.{csv,json}` | Paired differences −0.701/−0.693/−0.682/−0.678, 0/10 seeds positive at every τ_m — the falsification is not a timescale artifact (Fig. 2) |
| `s16b_falsification_stress_test.py --sequential` | Probe-protocol stress test of the falsification (V0/V1/V2 noise placements × 4 τ_m, 10 seeds) | `../data/s16b_falsification_stress_test_v2.{csv,json}` | Gap −0.69 (V0) → −0.04/−0.01 (V2) across τ_m — the metadata DOES denoise state-level corruption entering before the slow trace (Table 4): narrower, precisely located value |
| `s17_substrate_stress.py --sequential` | ESN substrate stress: 6 reservoir configs, metadata on/off (120 runs) | `../data/s17_substrate_stress_v1.{csv,json}` | Equalizer gain positive at all 6 configs (+0.1 to +1.0 pp) — the bottleneck is timescale coverage, not substrate physics |
| `s18_llm_drift_gate.py --sequential` | Transformer PoC: tiny transformer + LoRA, routing (A3) vs. gating-only (A2) slow-trace policies (90 runs, torch CPU) | `../data/s18_llm_drift_gate_v1.{csv,json}` | A3 routing transfers: forgetting −2.51 ppl (−28%) at τ_m = 200, stream −1.07 (10/10); A2 gating falsified at every τ_m (0/10) — detection without continuous correction is insufficient (Fig. 3, Table 6) |
| `s5b_controlled_adaptation.py --sequential` | Substrate arms of the controlled adaptation re-measurement (10 seeds) | `../data/s5b_controlled_adaptation_v1.{csv,json}` | Substrate adaptation factor 1.3–2.4 — matches the ESN boundary-acceleration scale |
| `gen_paperC_fig1_kernel.py` | FIG: slow-trace kernel vs. material forgetting kernel | `../figures/paperC_fig1_kernel.pdf` | Fig. 1 |
| `gen_paperC_fig2_recovery.py` | FIG: post-disturbance MC recovery vs. τ_m | `../figures/paperC_fig2_recovery.pdf` | Fig. 2 |
| `gen_paperC_fig3_llm.py` | FIG: LLM drift-gate results | `../figures/paperC_fig3_llm.pdf` | Fig. 3 |

Seed discipline and the full S1–s35 pipeline:
[`../README_REDEM.md`](../README_REDEM.md).

## Companion papers

- Paper A (physics): [`../paper_a/`](../paper_a/) — substrate characterization
- Paper B (algorithm): [`../paper_b/`](../paper_b/) — REDEM online learning
- Paper D (architecture): [`../paper_d/`](../paper_d/) — state-space-native REDEM
