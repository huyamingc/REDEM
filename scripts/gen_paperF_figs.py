#!/usr/bin/env python3
"""
Paper F figures - headline arms, clip-floor artifact, graphical abstract.
=============================================================================
Type:           FIG (matplotlib only, reads CSV, no @njit, no Pool)
Paper §:        Paper F Section 5 (Experiments, C1-C5) / Section 7 (Discussion)
Reads:          ../data/s50_paper_f_pilot_nl_readout_v1.csv
                ../data/s51_paper_f_learned_gate_v1.csv
                ../data/s52_paper_f_soft_route_experts_v1.csv
Output:         ../figures/paperF_fig1_arms.{pdf,png}
                ../figures/paperF_fig2_floor.{pdf,png}
                ../figures/paperF_graphical_abstract.{png,pdf}
Notes:          X-SelSSM-full is deliberately NOT plotted: the committed s66
                rows pool lr_scale=1.0 (sane, stream 10.19) and lr_scale=4.0
                (diverged, stream ~1e3-1e4) under one arm name; only the
                lr-independent frozen-host arm (9.03) is used as a reference.
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
DATA = os.path.join(HERE, '..', 'data')
FIG = os.path.join(HERE, '..', 'figures')

S50 = os.path.join(DATA, 's50_paper_f_pilot_nl_readout_v1.csv')
S51 = os.path.join(DATA, 's51_paper_f_learned_gate_v1.csv')
S52 = os.path.join(DATA, 's52_paper_f_soft_route_experts_v1.csv')

# (source file key, arm name in CSV, display label)
ARMS = [
    ('s50', 'B-lin-clip',       'B-lin-clip\n(RLS, clip)'),
    ('s50', 'B-simplex',        'B-simplex\n(post-hoc)'),
    ('s50', 'B-softmax-sgd',    'B-softmax\n(C1)'),
    ('s51', 'Gate-C-softmax',   'Gate-C\n(C4)'),
    ('s51', 'Gate-C-topk-softmax', 'Gate-C-topk\n(C4b)'),
    ('s52', 'Soft-topk',        'Soft-topk\n(C5)'),
    ('s52', 'Hard-topk',        'Hard-topk\n(abl.)'),
]
ARM_COLORS = ['#4c72b0', '#4c72b0', '#7293cf', '#c44e52',
              '#55a868', '#8172b2', '#ccb974']

FLOOR_ARMS = [
    ('s50', 'B-lin-clip',      'B-lin-clip'),
    ('s50', 'B-simplex',       'B-simplex'),
    ('s50', 'B-softmax-sgd',   'B-softmax'),
    ('s50', 'Skip-lin-clip',   'Skip-lin-clip'),
    ('s50', 'Skip-simplex',    'Skip-simplex'),
    ('s50', 'Skip-softmax-sgd', 'Skip-softmax'),
]

EXT_FROZEN_MEAN = 9.03   # X-SelSSM-frozen, 10 seeds (lr-independent)


def load(fn):
    """arm -> list of row dicts."""
    import csv
    out = {}
    with open(fn, 'r') as f:
        for r in csv.DictReader(f):
            out.setdefault(r['arm'], []).append(r)
    return out


def stats(src, arm, col):
    """(mean, std) of one metric over the arm's seeds."""
    v = np.array([float(r[col]) for r in src[arm]])
    return float(v.mean()), float(v.std())


def make_fig1():
    """Headline C1-C5 arms: stream and held-out forgetting ppl."""
    src = {'s50': load(S50), 's51': load(S51), 's52': load(S52)}
    s_mean, s_std = [], []
    f_mean, f_std = [], []
    for key, arm, _ in ARMS:
        m, d = stats(src[key], arm, 'stream_ppl')
        s_mean.append(m); s_std.append(d)
        m, d = stats(src[key], arm, 'forgetting_ppl')
        f_mean.append(m); f_std.append(d)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    x = np.arange(len(ARMS))

    ax = axes[0]
    ax.bar(x, s_mean, yerr=s_std, capsize=3, color=ARM_COLORS,
           edgecolor='black', linewidth=0.6, width=0.66)
    ax.axhline(EXT_FROZEN_MEAN, color='black', linestyle='--', linewidth=1.0,
               label='X-SelSSM-frozen = 9.03 (s66)')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([l for _, _, l in ARMS], fontsize=7)
    ax.set_ylabel('stream perplexity', fontsize=9)
    ax.set_ylim(0, 13.5)
    ax.set_title('(a) Stream (lower is better)', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    ax.bar(x, f_mean, yerr=f_std, capsize=3, color=ARM_COLORS,
           edgecolor='black', linewidth=0.6, width=0.66)
    ax.set_xticks(x); ax.set_xticklabels([l for _, _, l in ARMS], fontsize=7)
    ax.set_ylabel('held-out forgetting ppl', fontsize=9)
    ax.set_ylim(0, 36)
    ax.set_title('(b) Retention (lower is better)', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    pdf = os.path.join(FIG, 'paperF_fig1_arms.pdf')
    png = os.path.join(FIG, 'paperF_fig1_arms.png')
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"fig1: {pdf}, {png}")
    print(f"  stream: {', '.join(f'{l.splitlines()[0]}={m:.3f}' for (_, _, l), m in zip(ARMS, s_mean))}")
    print(f"  forget: {', '.join(f'{l.splitlines()[0]}={m:.3f}' for (_, _, l), m in zip(ARMS, f_mean))}")


def make_fig2():
    """Clip-floor artifact: negatives, clip mass and floor mass per arm."""
    src = load(S50)
    names = []
    clip_m, floor_m, neg_m = [], [], []
    for _, arm, label in FLOOR_ARMS:
        rows = src[arm]
        n = len(rows)
        names.append(label)
        clip_m.append(100.0 * sum(float(r['clip_frac']) for r in rows) / n)
        floor_m.append(100.0 * sum(float(r['floor_frac']) for r in rows) / n)
        neg_m.append(100.0 * sum(float(r['neg_frac']) for r in rows) / n)

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.bar(x - w, clip_m, w, color='#4c72b0', edgecolor='black',
           linewidth=0.6, label='clip$\\%$ (evaluated)')
    ax.bar(x, floor_m, w, color='#dd8452', edgecolor='black',
           linewidth=0.6, label='floor$\\%$ (evaluated)')
    ax.bar(x + w, neg_m, w, color='#c44e52', edgecolor='black',
           linewidth=0.6, label='neg$\\%$ (pre-eval)')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('token mass (%)', fontsize=9)
    ax.set_ylim(0, 12)
    ax.set_title('The artifact lives in the objective, not the range (C1/C2)',
                 fontsize=9)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    pdf = os.path.join(FIG, 'paperF_fig2_floor.pdf')
    png = os.path.join(FIG, 'paperF_fig2_floor.png')
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"fig2: {pdf}, {png}")
    for nm, c, fl, ng in zip(names, clip_m, floor_m, neg_m):
        print(f"  {nm:16s} clip={c:.2f}% floor={fl:.2f}% neg={ng:.2f}%")


def make_graphical():
    """Tall graphical abstract: key-arm stream ppl + floor artifact + takeaway."""
    src = {'s50': load(S50), 's52': load(S52)}
    key = [
        ('s50', 'B-lin-clip',    'B-lin-clip\n(RLS, clip)'),
        ('s50', 'B-softmax-sgd', 'B-softmax\n(C1)'),
        ('s52', 'Soft-topk',     'Soft-topk\n(C5)'),
        ('s52', 'Hard-topk',     'Hard-topk'),
    ]
    s_mean, s_std = [], []
    for kk, arm, _ in key:
        m, d = stats(src[kk], arm, 'stream_ppl')
        s_mean.append(m); s_std.append(d)

    floor_arms = ['B-lin-clip', 'B-simplex', 'B-softmax-sgd']
    floor_m = [1.58, 1.58, 0.0]

    fig = plt.figure(figsize=(6.0, 7.6))
    fig.suptitle('Learning the Readout Form\n'
                 'calibration, sparse selectivity, expert routing '
                 'on a diagonal SSM host',
                 fontsize=13, fontweight='bold')

    ax1 = fig.add_axes([0.13, 0.52, 0.74, 0.33])
    x1 = np.arange(len(key))
    ax1.bar(x1, s_mean, yerr=s_std, capsize=3,
            color=['#4c72b0', '#7293cf', '#8172b2', '#ccb974'],
            edgecolor='black', linewidth=0.7, width=0.6)
    ax1.axhline(EXT_FROZEN_MEAN, color='black', linestyle='--', linewidth=1.0,
                label='X-SelSSM-frozen = 9.03')
    ax1.set_xticks(x1)
    ax1.set_xticklabels([l for _, _, l in key], fontsize=9)
    ax1.set_ylabel('stream ppl', fontsize=10)
    ax1.set_ylim(0, 13.5)
    ax1.set_title('Stream perplexity (10 seeds, mean $\\pm$ std)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3)

    ax2 = fig.add_axes([0.13, 0.14, 0.74, 0.28])
    x2 = np.arange(len(floor_arms))
    ax2.bar(x2, floor_m, color=['#4c72b0', '#dd8452', '#7293cf'],
            edgecolor='black', linewidth=0.7, width=0.55)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(floor_arms, fontsize=9)
    ax2.set_ylabel('floor token mass (%)', fontsize=10)
    ax2.set_ylim(0, 2.2)
    ax2.set_title('Range fixes relocate the artifact; the objective removes it '
                  '(C2 vs C1)', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    fig.text(0.5, 0.055,
             'The evaluation instrument must be part of the learned system.',
             ha='center', fontsize=11, style='italic')

    png = os.path.join(FIG, 'paperF_graphical_abstract.png')
    pdf = os.path.join(FIG, 'paperF_graphical_abstract.pdf')
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"graphical abstract: {png}, {pdf}")


def main():
    os.makedirs(FIG, exist_ok=True)
    make_fig1()
    make_fig2()
    make_graphical()


if __name__ == '__main__':
    main()
