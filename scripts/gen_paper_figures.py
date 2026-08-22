#!/usr/bin/env python3
"""
Batch figure generation for the Paper A/B writing pass (REDEM S10).
=============================================================================
Type:           FIG
Paper Section:  Paper A Fig 4, Paper B Figs 3/5/6 (see PAPER_*_sketch.md)
Experiment:     Render remaining result figures from the experiment JSONs.

Figures:
  paperA_fig4_robustness.png  : S6 homeostat post-disturbance MC gains
  paperB_fig3_metadata.png    : S5 dual-timescale regime results
  paperB_fig5_ablation.png    : S8 integrated ablation matrix (N=256, 1024)
  paperB_fig6_showdown.png    : S9 online-vs-frozen showdown (drift + MG)

Usage: python gen_paper_figures.py
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')


def load(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ============ Paper A Fig 4 / Paper B Fig 4: homeostat robustness ============
    s6 = load('s6_chaos_regulator_v1.json')['aggregates']
    p2 = [a for a in s6 if a['part'] == 'p2']
    dists = sorted({a['disturb'] for a in p2})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    x = np.arange(len(dists))
    w = 0.36
    fixed_mc = [next((a['mc_at_settled_mean'] for a in p2
                      if a['disturb'] == d and a['arm'] == 'fixed'), np.nan)
                for d in dists]
    reg_mc = [next((a['mc_at_settled_mean'] for a in p2
                    if a['disturb'] == d and a['arm'] == 'regulated'), np.nan)
              for d in dists]
    ax.bar(x - w / 2, fixed_mc, w, label='fixed kappa=25')
    ax.bar(x + w / 2, reg_mc, w, label='lambda-homeostat')
    ax.set_xticks(x, dists)
    ax.set_ylabel('post-disturbance held-out MC')
    ax.set_title('S6: chaos homeostat restores substrate memory')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax = axes[1]
    gains = [(d, 100 * (reg - fixed) / fixed) for d, reg, fixed in
             zip(dists, reg_mc, fixed_mc) if fixed == fixed]
    ax.bar([g[0] for g in gains], [g[1] for g in gains],
           color='tab:orange')
    ax.set_ylabel('MC gain (%)')
    ax.set_title('relative gain over fixed coupling')
    ax.axhline(0, color='k', lw=0.6)
    ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('S6: lambda-homeostat (M5)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG_DIR, 'paperA_fig4_robustness.pdf'))
    print('saved figures/paperA_fig4_robustness.pdf')

    # ============ Paper B Fig 3: dual-timescale metadata (S5) ============
    s5 = load('s5_dual_timescale_v1.json')['aggregates']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ai, sub in enumerate(['parallel', 'random_graph_k25']):
        ax = axes[ai]
        rows = [a for a in s5 if a['substrate'] == sub]
        labels = [f"{a['arm']}" + (f" t={a['tau_slow']:.0f}"
                                   if a['tau_slow'] > 0 else '')
                  for a in rows]
        accs = [a['overall_acc_mean'] for a in rows]
        colors = ['tab:blue' if 'fast' in a['arm'] else
                  ('tab:orange' if a['arm'] == 'dual' else 'tab:green')
                  for a in rows]
        ax.bar(labels, accs, color=colors)
        ax.axhline(1 / 3, color='r', ls=':', lw=1.2)
        ax.text(0.99, 0.06, 'chance 1/3', transform=ax.transAxes, ha='right',
                fontsize=8, color='r')
        ax.set_ylim(0.3, 1.01)
        ax.set_title(f'S5: {sub} — regime task accuracy')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('S5: dual-timescale metadata (M3), p<0.0001', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG_DIR, 'paperB_fig3_metadata.pdf'))
    print('saved figures/paperB_fig3_metadata.pdf')

    # ============ Paper B Fig 5: integrated ablation (S8) ============
    s8 = load('s8_integrated_v1.json')['aggregates']
    fig, ax = plt.subplots(figsize=(9, 5))
    order = ['baseline', 'no_plasticity', 'no_homeostat', 'no_metadata', 'full']
    for n in [256, 1024]:
        rows = [a for a in s8 if a['n_units'] == n]
        if not rows:
            continue
        vals = []
        for arm in order:
            a = next((x for x in rows if x['arm'] == arm), None)
            vals.append(a['overall_acc_mean'] if a else np.nan)
        ax.plot(order, vals, 'o-', label=f'N={n}')
    ax.set_ylim(0.95, 1.005)
    ax.set_ylabel('overall accuracy (regime task)')
    ax.set_title('S8: integrated system vs ablations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'paperB_fig5_ablation.pdf'))
    print('saved figures/paperB_fig5_ablation.pdf')

    # ============ Paper B Fig 6: showdown (S9) ============
    s9 = load('s9_baseline_showdown_v1.json')['aggregates']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    rows = [a for a in s9 if a['task'] == 'drift_binary']
    systems = sorted({a['system'] for a in rows})
    x = np.arange(len(systems))
    w = 0.25
    for j, field, lab in [(0, 'pre_swap_acc', 'pre-swap'),
                          (1, 'post_swap_acc', 'post-swap'),
                          (2, 'mean_acc', 'stream mean')]:
        vals = [next((a[f'{field}_mean'] for a in rows if a['system'] == s), np.nan)
                for s in systems]
        ax.bar(x + (j - 1) * w, vals, w, label=lab)
    ax.set_xticks(x, systems)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('accuracy')
    ax.set_title('S9: drift task — online vs frozen batch')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax = axes[1]
    rows = [a for a in s9 if a['task'] == 'mackey_glass']
    systems = sorted({a['system'] for a in rows})
    vals = [next((a['nmse_final30_mean'] for a in rows if a['system'] == s), np.nan)
            for s in systems]
    ax.bar(systems, vals)
    ax.set_yscale('log')
    ax.set_ylabel('NMSE (last 30%, log)')
    ax.set_title('S9: Mackey-Glass forecasting')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'paperB_fig6_showdown.pdf'))
    print('saved figures/paperB_fig6_showdown.pdf')


if __name__ == '__main__':
    main()
