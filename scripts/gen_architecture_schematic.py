#!/usr/bin/env python3
"""
Architecture schematics for Paper A (substrate) and Paper B (REDEM).
=============================================================================
Type:           FIG
Paper Section:  Paper A Fig 1 / Paper B Fig 1
Experiment:     Programmatic block diagrams (no simulation).

Figures:
  figures/paperA_fig1_substrate.pdf : substrate schematic (units, coupling,
                                      log-normal tau histogram, readout)
  figures/paperB_fig1_redem.pdf     : REDEM system schematic (substrate,
                                      metadata, readout, homeostat, plasticity)

Usage: python gen_architecture_schematic.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')


def box(ax, x, y, w, h, text, fc='#e8f0fe', ec='#3b6ea5', fs=9, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal')


def arrow(ax, x1, y1, x2, y2, color='#555', lw=1.4, style='-|>', ls='-'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                        color=color, lw=lw, linestyle=ls)
    ax.add_patch(a)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ================= Paper A Fig 1: substrate =================
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.axis('off')

    # input pulses
    box(ax, 0.2, 2.2, 1.2, 1.0, 'pulse stream\n(dt_t)', fc='#fff3e0', ec='#b5651d')
    arrow(ax, 1.4, 2.7, 2.0, 2.7)

    # substrate: N units in a ring with coupling
    box(ax, 2.0, 1.3, 4.2, 3.0, '', fc='#e8f0fe', ec='#3b6ea5')
    ax.text(4.1, 4.15, 'relaxation substrate  (N = 256)', ha='center',
            fontsize=10, fontweight='bold', color='#2c4a6e')
    # log-normal tau histogram inset
    rng = np.random.RandomState(0)
    tau = rng.lognormal(np.log(174e-6) - 0.5 * np.log(1.04), np.sqrt(np.log(1.04)), 5000)
    hist_ax = ax.inset_axes([2.25, 3.35, 1.2, 0.7])
    hist_ax.hist(tau * 1e6, bins=40, color='#3b6ea5', alpha=0.8)
    hist_ax.set_xlim(0, 400)
    hist_ax.set_xticks([])
    hist_ax.set_yticks([])
    hist_ax.set_xlabel('tau (us)', fontsize=7)
    hist_ax.set_title('log-normal tau', fontsize=8, color='#2c4a6e')
    # units
    xs = np.linspace(2.5, 5.7, 6)
    for x in xs:
        c = Circle((x, 2.4), 0.16, fc='white', ec='#3b6ea5', lw=1.4)
        ax.add_patch(c)
    ax.text(5.9, 2.4, '...', ha='center', va='center', fontsize=12)
    # coupling arrows (kappa)
    for i in range(5):
        arrow(ax, xs[i] + 0.17, 2.62, xs[i + 1] - 0.17, 2.62, color='#c0392b', lw=1.1)
    ax.text(4.1, 2.95, 'per-pulse contrast coupling  kappa', ha='center',
            fontsize=8, color='#c0392b')
    ax.text(2.15, 1.55, 'x_i(t+1) = [x_i + alpha_eff(1-x_i)] exp(-dt/tau_i)',
            fontsize=7.5, color='#2c4a6e', style='italic')
    ax.text(2.15, 1.12, 'alpha_eff,i = clip(alpha0(1 + kappa g_i), 0.001, 0.10)',
            fontsize=7.5, color='#2c4a6e', style='italic')
    arrow(ax, 6.2, 2.7, 6.8, 2.7)

    # readout
    box(ax, 6.8, 2.0, 1.6, 1.4, 'current ratios\ni = exp(gamma x)', fc='#e8f5e9',
        ec='#2e7d32')
    arrow(ax, 8.4, 2.7, 9.0, 2.7)
    box(ax, 9.0, 2.0, 1.7, 1.4, 'readout\n(MC probe / RLS)',
        fc='#fff3e0', ec='#b5651d')
    ax.text(0.2, 5.1, 'Fig 1 (Paper A): physics-constrained relaxation substrate',
            fontsize=11, fontweight='bold', color='#333')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'paperA_fig1_substrate.pdf'))
    print('saved figures/paperA_fig1_substrate.pdf')

    # ================= Paper B Fig 1: REDEM =================
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis('off')

    box(ax, 0.3, 3.6, 2.6, 1.5, 'input pulses\n(dt_t)', fc='#fff3e0', ec='#b5651d')
    arrow(ax, 2.9, 4.3, 3.6, 4.3)

    # substrate
    box(ax, 3.6, 2.6, 3.4, 3.6, '', fc='#e8f0fe', ec='#3b6ea5')
    ax.text(5.3, 5.95, 'relaxation substrate', ha='center', fontsize=10,
            fontweight='bold', color='#2c4a6e')
    ax.text(5.3, 5.55, 'log-normal tau traps + per-pulse contrast coupling (kappa)',
            ha='center', fontsize=7.5, color='#2c4a6e')
    ax.text(5.3, 5.05, 'fast state x(t) -> i = exp(gamma x)', ha='center',
            fontsize=7.5, color='#2c4a6e')
    # unit row
    uxs = np.linspace(4.05, 6.55, 6)
    for x in uxs:
        c = Circle((x, 4.55), 0.14, fc='white', ec='#3b6ea5', lw=1.2)
        ax.add_patch(c)
    ax.text(6.8, 4.55, '...', ha='center', va='center', fontsize=11)

    # metadata (M3)
    box(ax, 7.4, 5.0, 2.2, 1.0, 'M3 metadata\nslow EMA m(t)', fc='#e8f5e9',
        ec='#2e7d32')
    arrow(ax, 7.0, 5.5, 7.4, 5.5)
    ax.text(7.35, 5.62, 'fast features', fontsize=6.5, color='#2e7d32',
            ha='right', va='bottom')

    # homeostat (M5) loop
    box(ax, 7.4, 3.4, 2.2, 1.0, 'M5 chaos homeostat\nFTLE -> kappa',
        fc='#fce4ec', ec='#c62828')
    arrow(ax, 7.0, 4.1, 7.4, 4.1)  # FTLE estimate into M5
    ax.text(7.35, 4.22, 'FTLE estimate', fontsize=6.5, color='#c62828',
            ha='right', va='bottom')
    # kappa control: M5 -> substrate (dashed red loop)
    a = FancyArrowPatch((7.4, 3.55), (7.0, 3.05), connectionstyle='arc3,rad=0.3',
                        arrowstyle='-|>', mutation_scale=13, color='#c62828',
                        lw=1.2, linestyle='--')
    ax.add_patch(a)
    ax.text(7.42, 3.1, 'kappa', fontsize=6.5, color='#c62828')

    # plasticity (M4) inside substrate bottom
    box(ax, 3.9, 2.7, 2.8, 0.85, 'M4 structure plasticity\n(corr-guided rewiring)',
        fc='#f3e5f5', ec='#6a1b9a', fs=7)
    arrow(ax, 4.85, 4.35, 4.85, 3.55, color='#6a1b9a', ls='--')  # corr in
    ax.text(4.72, 4.0, 'corr', fontsize=6.5, color='#6a1b9a', ha='right')
    arrow(ax, 5.9, 3.55, 5.9, 4.35, color='#6a1b9a', ls='--')    # rewire out
    ax.text(6.02, 4.0, 'rewire', fontsize=6.5, color='#6a1b9a', ha='left')

    # M4 <-> M5 coupling loop (T2): kappa gates rewiring rate; structure feeds lambda
    a = FancyArrowPatch((7.4, 3.45), (6.7, 3.05), connectionstyle='arc3,rad=-0.35',
                        arrowstyle='<|-|>', mutation_scale=12, color='#6a1b9a',
                        lw=1.1, linestyle=':')
    ax.add_patch(a)
    ax.text(7.42, 2.62, 'M4 <-> M5:\nkappa gates rewiring;\nstructure feeds lambda',
            fontsize=6, color='#6a1b9a', ha='left', va='top')

    # readout
    arrow(ax, 7.0, 4.85, 9.6, 4.85, color='#555', ls='-')
    arrow(ax, 9.6, 4.85, 9.6, 3.3, color='#555', ls='-')
    ax.text(8.35, 4.93, 'features [f, m]', fontsize=6.5, color='#555',
            ha='center', va='bottom')
    box(ax, 9.6, 1.9, 2.1, 1.4, 'RLS readout\n(predict-before-update)',
        fc='#fff3e0', ec='#b5651d', bold=True)
    ax.text(10.0, 3.05, 'error e = y - yhat\nupdates W online', ha='center',
            fontsize=6.5, color='#b5651d')
    # output
    arrow(ax, 11.2, 3.3, 11.2, 4.9, color='#555')
    box(ax, 10.6, 4.9, 1.2, 1.0, 'output\nyhat(t)', fc='#e8f0fe', ec='#3b6ea5')
    ax.text(0.3, 6.15, 'Fig 1 (Paper B): REDEM — training == inference',
            fontsize=11, fontweight='bold', color='#333')
    ax.text(0.3, 5.85, 'inference activity feeds every learning signal; no BPTT, no separate training phase',
            fontsize=8, color='#666', style='italic')
    ax.text(0.3, 5.55, 'dashed loops: M5 regulates kappa; M4 rewires edges (slow); dotted loop: M4 <-> M5 coupling',
            fontsize=7.5, color='#666', style='italic')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'paperB_fig1_redem.pdf'))
    print('saved figures/paperB_fig1_redem.pdf')


if __name__ == '__main__':
    main()
