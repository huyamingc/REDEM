#!/usr/bin/env python3
"""
S2 online-readout figure (drift tracking + final NMSE comparison).
=============================================================================
Type:           FIG
Paper Section:  New-algorithm project Step S2
Experiment:     Panel (a) visualizes data/s2_online_readout_v1_curves.npz
                (running accuracy vs pulse index). Panels (b)/(c) visualize
                data/s2_online_readout_v1.json aggregates (final-30% NMSE,
                10 seeds): the running-MSE curves in the npz overflow to NaN
                mid-run on narma10, while the final-30% NMSE aggregates are
                finite, so the comparison is drawn from the aggregates.

Panels:
  (a) drift_binary: running accuracy vs pulse index (log-x after warmup);
      vertical line at the first class-interval swap; shows online RLS
      tracking vs frozen offline collapse.
  (b) narma10 / (c) mackey_glass: final-30% NMSE (log-y) for the frozen
      offline ridge vs online RLS on the uncoupled and near-critically
      coupled (kappa=25) substrate.

Usage: python gen_s2_curves.py [--npz PATH] [--json PATH] [--out PATH]
"""
import json
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
JSON_PATH = os.path.join(DATA_DIR, 's2_online_readout_v1.json')
OUT_PATH = os.path.join(FIG_DIR, 's2_online_readout_v1.pdf')

STYLE = {
    ('drift_binary', 'parallel', 'rls'): dict(color='tab:blue', ls='-', label='parallel + RLS'),
    ('drift_binary', 'parallel', 'offridge'): dict(color='tab:blue', ls='--', label='parallel + offline ridge'),
    ('drift_binary', 'random_graph_k25', 'rls'): dict(color='tab:red', ls='-', label='random_graph k=25 + RLS'),
    ('drift_binary', 'random_graph_k25', 'offridge'): dict(color='tab:red', ls='--', label='random_graph k=25 + offline ridge'),
}

SUBSTRATES = [('parallel', 'parallel'), ('random_graph_k25', 'random graph k=25')]
READOUTS = [('offridge', 'offline ridge (frozen)', '#b0b0b0', ''),
            ('rls', 'online RLS', '#4c72b0', '')]


def main():
    args = sys.argv[1:]
    npz_path = NPZ_PATH
    json_path = JSON_PATH
    out = OUT_PATH
    for i, a in enumerate(args):
        if a == '--npz' and i + 1 < len(args):
            npz_path = args[i + 1]
        elif a == '--json' and i + 1 < len(args):
            json_path = args[i + 1]
        elif a == '--out' and i + 1 < len(args):
            out = args[i + 1]

    z = np.load(npz_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        agg = json.load(f)['aggregates']
    nmse = {(a['task'], a['substrate'], a['readout']):
            (a['nmse_final30_mean'], a.get('nmse_final30_std', 0.0))
            for a in agg if 'nmse_final30_mean' in a}

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

    # ---- narma10 / mackey_glass final NMSE bars ----
    for j, task in enumerate(['narma10', 'mackey_glass']):
        ax = axes[j + 1]
        width = 0.34
        top = 0.0
        for gi, (sub, sub_lbl) in enumerate(SUBSTRATES):
            for ri, (read, read_lbl, color, _) in enumerate(READOUTS):
                mean, std = nmse[(task, sub, read)]
                pos = gi + (ri - 0.5) * width
                # legend entries only once (first substrate), otherwise the
                # two substrate groups duplicate every label
                ax.bar(pos, mean, width=width, yerr=std, capsize=3,
                       color=color, edgecolor='black', linewidth=0.6,
                       label=read_lbl if (j == 0 and gi == 0) else None)
                ax.text(pos, mean * 1.18, f'{mean:.4g}', ha='center',
                        va='bottom', fontsize=8)
                top = max(top, mean + std)
        ax.set_yscale('log')
        ax.set_ylim(top=top * 6)  # headroom so bar value labels are not clipped
        ax.set_xticks(range(len(SUBSTRATES)))
        ax.set_xticklabels([lbl for _, lbl in SUBSTRATES], fontsize=9)
        ax.set_xlim(-0.55, len(SUBSTRATES) - 0.45)
        ax.set_xlabel('substrate')
        ax.set_ylabel('NMSE, final 30% (log)')
        ax.set_title(f'({chr(98 + j)}) {task}: final-30% NMSE (10 seeds)')
        ax.grid(True, axis='y', which='both', alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc='upper left', frameon=False)

    handles, labels = [], []
    for key, st in STYLE.items():
        t, s, r = key
        if t == 'drift_binary':
            handles.append(plt.Line2D([], [], color=st['color'], ls=st['ls'],
                                      label=st['label']))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.96),
               ncol=2, fontsize=9, frameon=False)
    fig.suptitle('S2 online readout: RLS (lambda=0.999) vs frozen offline '
                 'ridge (10 seeds)', fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)  # vector PDF for journal submission
    print(f"saved: {out}")


if __name__ == '__main__':
    main()
