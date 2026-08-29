#!/usr/bin/env python3
"""
Paper D Fig 1 - P1/P3a falsification and the input-path control.
=============================================================================
Type:           FIG (matplotlib only, reads CSV, no @njit, no Pool)
Paper §:        Paper D Section 3 (Results: P1/P3a)
Reads:          ../data/s19_ssm_rls_readout_v1.csv
Output:         ../figures/paperD_fig1_p1_arms.pdf
=============================================================================
"""
import os
import sys

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, '..', 'data', 's19_ssm_rls_readout_v1.csv')
OUT = os.path.join(HERE, '..', 'figures', 'paperD_fig1_p1_arms.pdf')

ORDER = ['LN-raw', 'LN-whiten', 'CV-whiten', 'CV-gate', 'CV-gate-g5',
         'CV-skip', 'B-proj']
LABELS = ['LN-raw', 'LN-whit', 'CV-whit', 'CV-gate', 'CV-gate-g5',
          'CV-skip', 'B-proj\n(input path)']


def main():
    data = {}
    with open(CSV, 'r') as f:
        import csv
        for row in csv.DictReader(f):
            data.setdefault(row['arm'], []).append(float(row['stream_ppl']))

    means = [float(np.mean(data[a])) for a in ORDER]
    stds = [float(np.std(data[a])) for a in ORDER]

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(ORDER))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color='#4c72b0',
                  edgecolor='black', linewidth=0.6, width=0.62)
    bars[-1].set_color('#c44e52')          # B-proj control highlighted

    ax.axhline(15.01, color='black', linestyle='--', linewidth=1.0)
    ax.axhline(7.34, color='gray', linestyle=':', linewidth=1.0)
    # white backing boxes: both labels sit on top of tall bars and would
    # otherwise be illegible where they cross the bar bodies
    ann_bbox = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85)
    ax.text(len(ORDER) - 0.4, 16.4, 'A1 (transformer+LoRA) = 15.01',
            ha='right', fontsize=8, zorder=5, bbox=ann_bbox)
    ax.text(len(ORDER) - 0.4, 8.0, 'pooled-table ceiling 7.34',
            ha='right', fontsize=8, color='gray', zorder=5, bbox=ann_bbox)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel('stream perplexity', fontsize=9)
    ax.set_ylim(0, 130)
    ax.set_title('P1/P3a: state-based readouts fail; the additive input '
                 'path works', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT} (means: "
          f"{', '.join(f'{a}={m:.2f}' for a, m in zip(ORDER, means))})")


if __name__ == '__main__':
    main()
