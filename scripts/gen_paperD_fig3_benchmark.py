#!/usr/bin/env python3
"""
Paper D Fig 3 - P4 benchmark: 4-domain irregular-switch protocol.
=============================================================================
Type:           FIG (matplotlib only, reads CSV, no @njit, no Pool)
Paper §:        Paper D Section 3 (Results: P4)
Reads:          ../data/s22_ssm_p4_benchmark_v1.csv
Output:         ../figures/paperD_fig3_benchmark.pdf
=============================================================================
"""
import os
import sys
import csv

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, '..', 'data', 's22_ssm_p4_benchmark_v1.csv')
OUT = os.path.join(HERE, '..', 'figures', 'paperD_fig3_benchmark.pdf')

ARMS = ['SSM-bare', 'SSM-REDEM', 'TF-A1']
LABELS = ['SSM-bare\n(M1 only)', 'SSM-REDEM\n(M1+M3+M4)', 'TF-A1\n(transformer+LoRA)']


def main():
    rows = list(csv.DictReader(open(CSV)))
    stream = [np.mean([float(r['stream_ppl']) for r in rows
                       if r['arm'] == a]) for a in ARMS]
    stream_std = [np.std([float(r['stream_ppl']) for r in rows
                          if r['arm'] == a]) for a in ARMS]
    forget = [np.mean([float(r['forgetting_ppl']) for r in rows
                       if r['arm'] == a]) for a in ARMS]
    forget_std = [np.std([float(r['forgetting_ppl']) for r in rows
                          if r['arm'] == a]) for a in ARMS]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    x = np.arange(len(ARMS))
    for ax, vals, stds, ylab, title in [
            (axes[0], stream, stream_std, 'stream perplexity',
             'Stream (4-domain, irregular switches)'),
            (axes[1], forget, forget_std, 'forgetting perplexity',
             'Forgetting (held-out previous domain)')]:
        bars = ax.bar(x, vals, yerr=stds, capsize=3, width=0.55,
                      color=['#8c8c8c', '#c44e52', '#4c72b0'],
                      edgecolor='black', linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}",
                    ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=7)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT} (stream: {stream}, forget: {forget})")


if __name__ == '__main__':
    main()
