# Paper C — Dissecting Online Learning Mechanisms: Statistical Memory, Homeostatic Recovery, and Substrate Physics are Non-Substitutable

Paper C of the four-paper REDEM series (**role: dissection**). It dissects
the online learner into three mechanisms — the slow-trace statistical memory
(M3), the homeostatic chaos regulator (M5), and the raw substrate physics —
and uses falsifying transfer experiments to establish which mechanism is
responsible for which capability, and where each one stops.

- PDF: [`PAPER_C.pdf`](PAPER_C.pdf) | LaTeX: [`PAPER_C.tex`](PAPER_C.tex)
- Preprint: [doi:10.5281/zenodo.22110619](https://doi.org/10.5281/zenodo.22110619)
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
| S16b falsification stress test | — | `../data/s16b_falsification_stress_test_v1.*` |
| S17 substrate stress | — | `../data/s17_substrate_stress_v1.*` |
| S18 LLM drift gate | — | `../data/s18_llm_drift_gate_v1.*` |

## Reproduce

CPU-only; uses the project virtual environment (`../.venv`).

```powershell
& ..\.venv\Scripts\python.exe ..\scripts\esn_metadata_comparison.py
& ..\.venv\Scripts\python.exe ..\scripts\s14_esn_disturbance_chain.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s15_controlled_adaptation.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s16_tau_m_pressure_test.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s16b_falsification_stress_test.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s17_substrate_stress.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s18_llm_drift_gate.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\s5b_controlled_adaptation.py --sequential
& ..\.venv\Scripts\python.exe ..\scripts\gen_paperC_fig1_kernel.py
& ..\.venv\Scripts\python.exe ..\scripts\gen_paperC_fig2_recovery.py
& ..\.venv\Scripts\python.exe ..\scripts\gen_paperC_fig3_llm.py
```

Every script accepts `--quick` for a smoke run. Full pipeline and seed
discipline: [`../README_REDEM.md`](../README_REDEM.md).

## Companion papers

- Paper A (physics): [`../paper_a/`](../paper_a/) — substrate characterization
- Paper B (algorithm): [`../paper_b/`](../paper_b/) — REDEM online learning
- Paper D (architecture): [`../paper_d/`](../paper_d/) — state-space-native REDEM
