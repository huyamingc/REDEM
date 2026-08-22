#!/usr/bin/env python3
"""
S2 online-readout curves figure (drift adaptation + regression learning).
=============================================================================
Type:           FIG
Paper Section:  New-algorithm project Step S2 (see NEW_ALGORITHM_PLAN.md)
Experiment:     Visualize data/s2_online_readout_v1_curves.npz

Panels:
  (a) drift_binary: running accuracy vs pulse index (log-x after warmup);
      vertical line at the first class-interval swap; shows online RLS
      tracking vs frozen offline collapse.
  (b) narma10 / (c) mackey_glass: running MSE (log-y) vs pulse index.

Usage: python gen_s2_curves.py [--npz PATH] [--out PATH]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
NPZ_PATH = os.path.join(DATA_DIR, 's2_online_readout_v1_curves.npz')
OUT_PATH = os.path.join(FIG_DIR, 's2_online_readout_v1.png')

STYLE = {
    ('drift_binary', 'parallel', 'rls'): dict(color='tab:blue', ls='-', label='parallel + RLS'),
    ('drift_binary', 'parallel', 'offridge'): dict(color='tab:blue', ls='--', label='parallel + offline ridge'),
    ('drift_binary', 'random_graph_k25', 'rls'): dict(color='tab:red', ls='-', label='random_graph k=25 + RLS'),
    ('drift_binary', 'random_graph_k25', 'offridge'): dict(color='tab:red', ls='--', label='random_graph k=25 + offline ridge'),
}


def main():
    args = sys.argv[1:]
    npz_path = NPZ_PATH
    out = OUT_PATH
    for i, a in enumerate(args):
        if a == '--npz' and i + 1 < len(args):
            npz_path = args[i + 1]
        elif a == '--out' and i + 1 < len(args):
            out = args[i + 1]

    z = np.load(npz_path)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # ---- drift_binary accuracy ----
    ax = axes[0]
    swap1 = 20000  # block 1000 * 20 pulses (stream parameter, documented)
    for (task, sub, read), st in STYLE.items():
        if task != 'drift_binary':
            continue
        x = z[f'{task}_{sub}_{read}_x']
        y = z[f'{task}_{sub}_{read}_y']
        sem = z[f'{task}_{sub}_{read}_sem']
        ax.plot(x, y, **st)
        ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, color=st['color'],
                        alpha=0.15)
    ax.axvline(swap1, color='k', ls=':', lw=1.2)
    ax.text(swap1, 0.02, 'class swap', rotation=90, fontsize=8, va='bottom')
    ax.set_xlabel('pulse index')
    ax.set_ylabel('running accuracy (window 200)')
    ax.set_ylim(0.0, 1.02)
    ax.set_title('(a) drift_binary: online tracking vs frozen collapse')
    ax.grid(True, alpha=0.3)

    # ---- narma10 / mackey_glass MSE ----
    for j, task in enumerate(['narma10', 'mackey_glass']):
        ax = axes[j + 1]
        for (t, sub, read), st in STYLE.items():
            if t != task:
                continue
            x = z[f'{task}_{sub}_{read}_x']
            y = z[f'{task}_{sub}_{read}_y']
            ax.semilogy(x, np.maximum(y, 1e-8), **st)
        ax.set_xlabel('pulse index')
        ax.set_ylabel('running MSE (window 200, log)')
        ax.set_title(f'({chr(98 + j)}) {task}: online vs offline learning')
        ax.grid(True, which='both', alpha=0.3)

    handles, labels = [], []
    for key, st in STYLE.items():
        t, s, r = key
        if t == 'drift_binary':
            handles.append(plt.Line2D([], [], color=st['color'], ls=st['ls'],
                                      label=st['label']))
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=9,
               frameon=False)
    fig.suptitle('S2 online readout: RLS (lambda=0.999) vs frozen offline ridge '
                 '(10 seeds, mean +/- 1.96 SEM)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out[:-4] + '.pdf')  # vector PDF for journal submission
    print(f"saved: {out}")


if __name__ == '__main__':
    main()
