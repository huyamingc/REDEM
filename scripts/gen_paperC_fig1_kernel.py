#!/usr/bin/env python3
"""
Paper C Fig. 1: the slow-trace kernel vs the material forgetting kernel.
=============================================================================
Type:           FIG
Paper:          Paper C (Proposition 1: the slow trace is a synthesized
                forgetting kernel)
Data:           analytic curves only:
                  M(k)  = material forgetting kernel (Paper A: log-normal
                          tau, CV 0.20, median tau0 ~174 us), evaluated at
                          k pulses via dt_bar = 11 us
                  h_m(k) = (e^{-k/tau_m} - e^{-k/tau_x}) / (tau_m - tau_x),
                          the input->slow-trace kernel (the Paper C
                          M3 slow-trace formula) with tau_x = 16 pulses (substrate
                          fast timescale), tau_m in {200, 500, 1000}
Figure:         log-log plot of the normalized kernels: the material kernel
                has a log-normal tail fixed by physics; the slow-trace
                kernel has a band-pass envelope (zero at k=0, peak at a
                finite lag) and an exponential tail whose 1/e horizon is
                exactly tau_m - controllable, substrate-independent.

Output:         figures/paperC_fig1_kernel.pdf (vector only, no PNG)
Usage:          python gen_paperC_fig1_kernel.py
"""
import os
import sys

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forgetting_curve_theory import forgetting_kernel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
OUT_PDF = os.path.join(FIG_DIR, 'paperC_fig1_kernel.pdf')

DT_BAR = 11e-6            # mean inter-pulse interval (s)
TAU_X = 16.0              # substrate fast timescale (pulses, Paper A 1/e
                          # horizon of the fast channel)
TAU_M_LIST = [200.0, 500.0, 1000.0]
CV = 0.20
K_GRID = np.logspace(0, 3.7, 400)   # lag in pulses, 1..~5000


def slow_trace_kernel(k, tau_m, tau_x=TAU_X):
    """Input->slow-trace kernel h_m(k) (Paper C, M3 slow-trace formula)."""
    return (np.exp(-k / tau_m) - np.exp(-k / tau_x)) / (tau_m - tau_x)


def normalize(v):
    m = v.max()
    return v / m if m > 0 else v


def main():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # Material kernel M(k): evaluate the log-normal kernel at k * DT_BAR
    M = np.array([forgetting_kernel(k * DT_BAR, CV) for k in K_GRID])

    ax.loglog(K_GRID, normalize(M), '-', color='#2ca02c', linewidth=2,
              label='material kernel M(k) (log-normal, CV=0.20)')
    for tm in TAU_M_LIST:
        h = slow_trace_kernel(K_GRID, tm)
        ax.loglog(K_GRID, normalize(np.maximum(h, 1e-6)), '-',
                  color='#1f77b4', alpha=0.55 + 0.2 * (tm / 1000.0),
                  label=f'slow-trace kernel h_m(k), tau_m={int(tm)}')

    # 1/e horizon markers
    for tm in [200.0, 1000.0]:
        ax.axvline(tm, color='#1f77b4', linestyle=':', linewidth=0.8,
                   alpha=0.5)
    ax.axvline(16, color='#2ca02c', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(16, 0.9, '~16 (material 1/e)', fontsize=7, rotation=90,
            va='top', ha='right', color='#2ca02c')
    ax.text(200, 0.9, 'tau_m=200', fontsize=7, rotation=90,
            va='top', ha='right', color='#1f77b4')
    ax.text(1000, 0.9, 'tau_m=1000', fontsize=7, rotation=90,
            va='top', ha='right', color='#1f77b4')

    ax.set_xlabel('lag k (pulses)')
    ax.set_ylabel('normalized kernel amplitude')
    ax.set_title('Paper C Fig. 1 - synthesized vs material forgetting '
                 'kernel', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT_PDF, format='pdf')
    print(f"Wrote {OUT_PDF}")


if __name__ == '__main__':
    main()
