#!/usr/bin/env python3
"""
Paper C Fig. 2: post-disturbance MC recovery vs metadata timescale.
=============================================================================
Type:           FIG
Paper:          Paper C (three-mechanism disentanglement)
Data:           data/s16_tau_m_pressure_test_v1.csv (esn_dual x tau_m,
                esn_fast baseline), data/s14_esn_disturbance_chain_v1.csv
                (redem_reg homeostat anchor)
Figure:         Left panel: r3_mc (mean +- std over seeds) vs tau_m for
                esn_dual, with flat reference lines for esn_fast (no
                metadata) and redem_reg (homeostat, no metadata). Right
                panel: r3_nmse vs tau_m (dual) with esn_fast reference -
                shows the noise-attenuation transfer that DOES hold.

Output:         figures/paperC_fig2_recovery.pdf (vector only, no PNG)
Usage:          python gen_paperC_fig2_recovery.py
"""
import os
import sys
import csv

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')

S16_CSV = os.path.join(DATA_DIR, 's16_tau_m_pressure_test_v1.csv')
S14_CSV = os.path.join(DATA_DIR, 's14_esn_disturbance_chain_v1.csv')
OUT_PDF = os.path.join(FIG_DIR, 'paperC_fig2_recovery.pdf')

TAU_M_LIST = [200.0, 500.0, 1000.0, 2000.0]


def load_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def mean_std(rows, field):
    v = np.array([float(r[field]) for r in rows], dtype=float)
    v = v[~np.isnan(v)]
    return float(np.mean(v)), float(np.std(v))


def main():
    rows16 = load_rows(S16_CSV)
    rows14 = load_rows(S14_CSV)

    dual = {tm: [r for r in rows16
                 if r['arm'] == 'esn_dual' and float(r['tau_m']) == tm]
            for tm in TAU_M_LIST}
    fast = [r for r in rows16 if r['arm'] == 'esn_fast']
    redem = [r for r in rows14 if r['arm'] == 'redem_reg']

    fast_r3, fast_r3_std = mean_std(fast, 'r3_mc')
    fast_r3n, _ = mean_std(fast, 'r3_nmse')
    redem_r3, redem_r3_std = mean_std(redem, 'r3_mc')

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # Left: MC recovery after the full disturbance chain vs tau_m
    ax = axes[0]
    xs = [0.0] + TAU_M_LIST
    xlabels = ['no meta'] + [f'{int(tm)}' for tm in TAU_M_LIST]
    dual_r3 = [fast_r3] + [mean_std(dual[tm], 'r3_mc')[0] for tm in TAU_M_LIST]
    dual_std = [fast_r3_std] + [mean_std(dual[tm], 'r3_mc')[1]
                                for tm in TAU_M_LIST]
    ax.errorbar(xs, dual_r3, yerr=dual_std, fmt='o-', capsize=3,
                color='#1f77b4', label='ESN + slow trace (tau_m)')
    ax.axhline(fast_r3, color='#ff7f0e', linestyle='--', linewidth=1.2,
               label=f'ESN fast only (r3 MC = {fast_r3:.2f})')
    ax.axhline(redem_r3, color='#2ca02c', linestyle=':', linewidth=1.4,
               label=f'REDEM + homeostat (r3 MC = {redem_r3:.2f})')
    ax.fill_between([0, TAU_M_LIST[-1]], fast_r3 - fast_r3_std,
                    fast_r3 + fast_r3_std, color='#ff7f0e', alpha=0.12)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel('metadata timescale tau_m (pulses)')
    ax.set_ylabel('MC after 3 disturbances (r3)')
    ax.set_title('Memory capacity: no transfer', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc='lower right')

    # Right: task NMSE at the readout-noise round vs tau_m
    ax = axes[1]
    dual_r3n = [fast_r3n] + [mean_std(dual[tm], 'r3_nmse')[0]
                             for tm in TAU_M_LIST]
    dual_nstd = [0.0] + [mean_std(dual[tm], 'r3_nmse')[1] for tm in TAU_M_LIST]
    ax.errorbar(xs, dual_r3n, yerr=dual_nstd, fmt='s-', capsize=3,
                color='#1f77b4', label='ESN + slow trace (tau_m)')
    ax.axhline(fast_r3n, color='#ff7f0e', linestyle='--', linewidth=1.2,
               label=f'ESN fast only (r3 NMSE = {fast_r3n:.4f})')
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel('metadata timescale tau_m (pulses)')
    ax.set_ylabel('NMSE at noise round (r3)')
    ax.set_title('Task noise: attenuation transfers', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc='upper right')

    fig.suptitle('Paper C Fig. 2 - three mechanisms, three roles '
                 '(s16, 10 seeds)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT_PDF, format='pdf')
    print(f"Wrote {OUT_PDF}")
    print(f"  r3 MC: fast {fast_r3:.3f} +- {fast_r3_std:.3f}; "
          f"redem_reg {redem_r3:.3f} +- {redem_r3_std:.3f}")
    for tm in TAU_M_LIST:
        m, s = mean_std(dual[tm], 'r3_mc')
        print(f"  tau_m={tm:>5.0f}: dual r3 MC {m:.3f} +- {s:.3f}, "
              f"r3 NMSE {mean_std(dual[tm], 'r3_nmse')[0]:.4f}")


if __name__ == '__main__':
    main()
