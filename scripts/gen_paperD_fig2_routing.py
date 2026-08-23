#!/usr/bin/env python3
"""
Paper D Fig 2 - P2/P3 routing: M3 retains domain specialists; gentle
(soft) routing beats abrupt on stream perplexity.
=============================================================================
Type:           FIG (matplotlib only, reads CSV, no @njit, no Pool)
Paper §:        Paper D Section 3 (Results: P2/P3)
Reads:          ../data/s20_ssm_m3_routing_v1.csv
                ../data/s21_ssm_m4_m5_v1.csv
Output:         ../figures/paperD_fig2_routing.pdf
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
S20 = os.path.join(HERE, '..', 'data', 's20_ssm_m3_routing_v1.csv')
S21 = os.path.join(HERE, '..', 'data', 's21_ssm_m4_m5_v1.csv')
OUT = os.path.join(HERE, '..', 'figures', 'paperD_fig2_routing.pdf')

TAUS = [200.0, 500.0, 1000.0, 2000.0]


def load_s20():
    rows = list(csv.DictReader(open(S20)))
    a1f = np.mean([float(r['forgetting_ppl']) for r in rows
                   if r['arm'] == 'A1'])
    a1s = np.mean([float(r['stream_ppl']) for r in rows
                   if r['arm'] == 'A1'])
    out = {'a1_forget': a1f, 'a1_stream': a1s,
           'a2_forget': [], 'a3_forget': [], 'a2_stream': [],
           'a3_stream': []}
    for tm in TAUS:
        for arm, k in [('A2', 'a2'), ('A3', 'a3')]:
            out[k + '_forget'].append(np.mean(
                [float(r['forgetting_ppl']) for r in rows
                 if r['arm'] == arm and float(r['tau_m']) == tm]))
            out[k + '_stream'].append(np.mean(
                [float(r['stream_ppl']) for r in rows
                 if r['arm'] == arm and float(r['tau_m']) == tm]))
    return out


def load_s21():
    rows = list(csv.DictReader(open(S21)))
    e1 = [r for r in rows if r['exp'] == 'E1']
    return {a: np.mean([float(r['stream_ppl']) for r in e1
                        if r['arm'] == a]) for a in
            ['A1', 'A3-abrupt', 'A3-soft']}


def main():
    s20 = load_s20()
    s21 = load_s21()

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))

    # Left: forgetting ppl vs tau_m (P2: routing retains specialists)
    ax = axes[0]
    ax.axhline(s20['a1_forget'], color='black', linestyle='--', linewidth=1.0,
               label='A1 bare')
    ax.plot(TAUS, s20['a2_forget'], 'o-', color='#dd8452', label='A2 gate-only')
    ax.plot(TAUS, s20['a3_forget'], 's-', color='#4c72b0',
            label='A3 routing')
    ax.set_xscale('log')
    ax.set_xticks(TAUS)
    ax.set_xticklabels([f"{int(t)}" for t in TAUS], fontsize=8)
    ax.set_xlabel(r'$\tau_m$', fontsize=9)
    ax.set_ylabel('forgetting perplexity', fontsize=9)
    ax.set_title('P2: routing retains domain specialists', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)

    # Right: soft vs abrupt routing stream ppl (P3: gentle wins)
    ax = axes[1]
    names = ['A1 bare', 'A3 abrupt', 'A3 soft']
    vals = [s21['A1'], s21['A3-abrupt'], s21['A3-soft']]
    cols = ['#8c8c8c', '#4c72b0', '#c44e52']
    bars = ax.bar(names, vals, color=cols, edgecolor='black', linewidth=0.6,
                  width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                ha='center', fontsize=8)
    ax.set_ylabel('stream perplexity', fontsize=9)
    ax.set_ylim(0, 15)
    ax.set_title('P3: soft routing (gentle) beats abrupt, '
                 r'$\tau_m=500$', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
