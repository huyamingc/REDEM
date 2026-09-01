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
    oracle = {}
    with open(CSV, 'r') as f:
        import csv
        for row in csv.DictReader(f):
            data.setdefault(row['arm'], []).append(float(row['stream_ppl']))
            if row.get('oracle_ppl'):
                oracle.setdefault(row['arm'], []).append(
                    float(row['oracle_ppl']))

    means = [float(np.mean(data[a])) for a in ORDER]
    stds = [float(np.std(data[a])) for a in ORDER]

    # pooled-table ceiling: 10-seed mean of the oracle projection readout
    # on the B-proj (pooled-table) arm
    ceiling = float(np.mean(oracle['B-proj']))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(ORDER))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color='#4c72b0',
                  edgecolor='black', linewidth=0.6, width=0.62)
    bars[-1].set_color('#c44e52')          # B-proj control highlighted

    ax.axhline(15.01, color='black', linestyle='--', linewidth=1.0,
               label='TF-A1 (transformer+LoRA) = 15.01')
    ax.axhline(ceiling, color='gray', linestyle=':', linewidth=1.0,
               label=f'pooled-table ceiling = {ceiling:.2f}')
    # upper-right region is bar-free (CV-skip tops out at 58): a legend
    # there keeps the line labels clear of the bars and the dashed lines
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

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
