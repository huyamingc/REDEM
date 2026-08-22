#!/usr/bin/env python3
"""
Paper A Supplementary Fig. S1 - task-level CV sweep.
=============================================================================
Type:           FIG
Paper Section:  Paper A, Supplementary Note 1
Experiment:     Plot held-out MC vs spectrum width CV for the S10 CV sweep arms.

Reads:  data/s10_cv_sweep_v1.json (aggregates with mc_mean/mc_std per config)
Output: figures/paperA_figS1_cv_sweep.png

Usage: python gen_paperA_supp_figures.py
=============================================================================
"""
import sys
import os
import json

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
DATA_PATH = os.path.join(ROOT, 'data', 's10_cv_sweep_v1.json')
OUT_PATH = os.path.join(ROOT, 'figures', 'paperA_figS1_cv_sweep.png')

LABELS = {
    'parallel': 'uncoupled (parallel)',
    'random_graph_k25': 'random graph, kappa = 25',
    'ring_bidir_k20': 'ring bidir, kappa = 20',
}
COLORS = {
    'parallel': '#3b6ea5',
    'random_graph_k25': '#c0392b',
    'ring_bidir_k20': '#2e7d32',
}


def main():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for agg in data['aggregates']:
        cfg = agg['config']
        ax.errorbar(agg['cv'], agg['mc_mean'], yerr=agg['mc_std'],
                    marker='o', capsize=4, lw=1.6, color=COLORS[cfg],
                    label=LABELS[cfg])

    ax.set_xlabel('spectrum width CV')
    ax.set_ylabel('held-out MC total')
    ax.set_title('Fig. S1: task-level CV sweep (10 seeds)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print('saved figures/paperA_figS1_cv_sweep.png')


if __name__ == '__main__':
    main()
