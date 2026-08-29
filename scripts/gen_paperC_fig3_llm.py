#!/usr/bin/env python3
"""
Paper C Fig. 3: LLM drift-gate PoC results (s18).
=============================================================================
Type:           FIG
Paper:          Paper C Section 7 (extension to LLMs via LoRA)
Data:           data/s18_llm_drift_gate_v1.csv (10 seeds)
Figure:         Left: stream perplexity vs metadata timescale tau_m for
                A1 (bare online LoRA, horizontal line), A2 (drift gate),
                A3 (domain routing). Right: forgetting perplexity
                (held-out previous-domain ppl) vs tau_m. Error bars =
                std over seeds. Shows: A2 never beats A1 on ppl (gate
                falsified); A3 beats A1 on ppl at tau_m <= 500 (10/10
                seeds) and on forgetting at every tau_m (9-10/10 seeds).

Output:         figures/paperC_fig3_llm.pdf (vector only, no PNG)
Usage:          python gen_paperC_fig3_llm.py
"""
import os
import sys
import csv

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')

S18_CSV = os.path.join(DATA_DIR, 's18_llm_drift_gate_v1.csv')
OUT_PDF = os.path.join(FIG_DIR, 'paperC_fig3_llm.pdf')

TAUS = [200.0, 500.0, 1000.0, 2000.0]


def load():
    with open(S18_CSV, newline='') as f:
        return list(csv.DictReader(f))


def mean_std(rows, arm, tm, field):
    v = np.array([float(r[field]) for r in rows
                  if r['arm'] == arm and float(r['tau_m']) == tm])
    return float(np.mean(v)), float(np.std(v))


def main():
    rows = load()
    a1 = {f: mean_std(rows, 'A1', 0.0, f) for f in
          ['stream_ppl', 'forgetting_ppl']}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # arm labels must match the paper caption: A2 = gating-only policy,
    # A3 = domain routing (a shared generic label misdescribes both arms)
    ARM_LABELS = {'A2': 'A2 (gating-only)', 'A3': 'A3 (domain routing)'}

    for ax, field, ylab, title in [
            (axes[0], 'stream_ppl', 'stream perplexity',
             'Online PPL (lower better)'),
            (axes[1], 'forgetting_ppl', 'forgetting perplexity',
             'Old-domain PPL (lower better)')]:
        base, base_std = a1[field]
        for arm, color, mk in [('A2', '#d62728', 's'), ('A3', '#1f77b4', 'o')]:
            ys = [mean_std(rows, arm, tm, field)[0] for tm in TAUS]
            err = [mean_std(rows, arm, tm, field)[1] for tm in TAUS]
            ax.errorbar(TAUS, ys, yerr=err, fmt=mk + '-', capsize=3,
                        color=color, label=ARM_LABELS[arm])
        ax.axhline(base, color='#ff7f0e', linestyle='--', linewidth=1.3,
                   label=f'A1 bare LoRA ({base:.2f})')
        ax.set_xscale('log')
        ax.set_xticks(TAUS)
        ax.set_xticklabels([f'{int(t)}' for t in TAUS])
        # log axis spans one decade, so matplotlib would auto-label minor
        # ticks (3e2, 4e2, ...) on top of the explicit major labels
        ax.minorticks_off()
        ax.set_xlabel('metadata timescale tau_m (tokens)')
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc='best')

    fig.suptitle('Paper C Fig. 3 - LLM drift-gate PoC (s18, tiny '
                 'transformer + LoRA, 10 seeds)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT_PDF, format='pdf')
    print(f"Wrote {OUT_PDF}")
    print(f"  A1: stream {a1['stream_ppl'][0]:.2f} +- "
          f"{a1['stream_ppl'][1]:.2f}, forget {a1['forgetting_ppl'][0]:.2f}")
    for tm in TAUS:
        for arm in ['A2', 'A3']:
            p, ps = mean_std(rows, arm, tm, 'stream_ppl')
            f, fs = mean_std(rows, arm, tm, 'forgetting_ppl')
            print(f"  {arm} tau_m={tm:>5.0f}: stream {p:>6.2f} +- {ps:.2f}, "
                  f"forget {f:>6.2f} +- {fs:.2f}")


if __name__ == '__main__':
    main()
