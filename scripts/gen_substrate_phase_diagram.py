#!/usr/bin/env python3
"""
Phase diagram figure for the recurrent substrate (S1): kappa-MC-FTLE.
=============================================================================
Type:           FIG
Paper Section:  New-algorithm project Step S1
Experiment:     Visualize the substrate_recurrence_characterization output.

Reads data/substrate_phase_diagram_v2.json (aggregates) and produces:
  figures/substrate_phase_diagram_v2.pdf (vector PDF for journal submission)
    2x2 panels:
      (a) FTLE vs kappa per topology          (log-x)
      (b) held-out MC_total vs kappa          (log-x, zero-capped)
      (c) train MC_total vs kappa             (log-x)
      (d) alpha clip fraction vs kappa        (log-x)
    3-sigma error bars (mean +- 1.96*std/sqrt(n_runs)).

Usage: python gen_substrate_phase_diagram.py [--json PATH] [--out PATH]
"""
import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')

JSON_PATH = os.path.join(DATA_DIR, 'substrate_phase_diagram_v2.json')
OUT_PATH = os.path.join(FIG_DIR, 'substrate_phase_diagram_v2.pdf')

# Panel order and styling
TOPO_STYLE = {
    'parallel':      dict(marker='s', color='k', ls='--', label='parallel (uncoupled)'),
    'ring_bidir':    dict(marker='o', color='tab:blue', label='ring_bidir (mode 1)'),
    'lateral_ring':  dict(marker='v', color='tab:orange', label='lateral_ring (mode 1)'),
    'random_graph':  dict(marker='^', color='tab:green', label='random_graph (mode 1)'),
    'ring_unidir':   dict(marker='D', color='tab:red', label='ring_unidir (mode 2)'),
    'hub_star':      dict(marker='P', color='tab:purple', label='hub_star (mode 2)'),
    'add_ring_bidir': dict(marker='x', color='gray', label='add_ring_bidir (mode 3, control)'),
    'add_random_graph': dict(marker='+', color='darkgray', label='add_random_graph (mode 3, control)'),
}


def main():
    args = sys.argv[1:]
    jpath = JSON_PATH
    out = OUT_PATH
    for i, a in enumerate(args):
        if a == '--json' and i + 1 < len(args):
            jpath = args[i + 1]
        elif a == '--out' and i + 1 < len(args):
            out = args[i + 1]

    with open(jpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    agg = data['aggregates']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_ftle, ax_mcte), (ax_mctr, ax_clip) = axes

    zero_cap = 1e-4  # avoid log(0) for zero-valued MC

    for a in agg:
        name = a['config_name']
        if name not in TOPO_STYLE:
            continue
        st = TOPO_STYLE[name]
        k = a['kappa']
        err = 1.96 * a['ftle_std'] / np.sqrt(max(a['n_runs'], 1))
        ax_ftle.errorbar(k, a['ftle_mean'], yerr=err, **st)

        err_te = 1.96 * a['mc_total_test_std'] / np.sqrt(max(a['n_runs'], 1))
        ax_mcte.errorbar(k, max(a['mc_total_test_mean'], zero_cap), yerr=err_te, **st)

        err_tr = 1.96 * a['mc_total_std'] / np.sqrt(max(a['n_runs'], 1))
        ax_mctr.errorbar(k, max(a['mc_total_mean'], zero_cap), yerr=err_tr, **st)

        ax_clip.plot(k, a['alpha_clip_frac_mean'], **st)

    for ax in (ax_ftle, ax_mcte, ax_mctr, ax_clip):
        ax.set_xscale('log')
        ax.set_xlabel(r'coupling strength $\kappa$')
        ax.axvline(0.0, color='k', lw=0.5)
        ax.grid(True, which='both', alpha=0.3)

    ax_ftle.axhline(0.0, color='r', lw=1.0, ls=':')
    ax_ftle.set_ylabel(r'FTLE per pulse $\lambda$')
    ax_ftle.set_title('(a) Dynamics regime: edge of chaos at $\\lambda \\approx 0$')
    ax_ftle.text(0.99, 0.02, 'ordered: $\\lambda<0$  |  CHAOTIC: $\\lambda>0$',
                 transform=ax_ftle.transAxes, ha='right', va='bottom', fontsize=8,
                 style='italic')

    ax_mcte.set_ylabel('held-out MC (lags 1..50)')
    ax_mcte.set_title('(b) Held-out memory capacity (honest estimate)')
    ax_mcte.set_ylim(bottom=zero_cap)

    ax_mctr.set_ylabel('train MC (Jaeger)')
    ax_mctr.set_title('(c) Train-segment memory capacity (overfitting inflated in chaos)')
    ax_mctr.set_ylim(bottom=zero_cap)

    ax_clip.set_ylabel(r'fraction of units at $\alpha_{eff}$ clip bounds')
    ax_clip.set_title('(d) Saturation: clip fraction (self-limiting dynamics)')
    ax_clip.set_ylim(-0.02, 1.05)

    handles, labels = [], []
    for name, st in TOPO_STYLE.items():
        exists = any(a['config_name'] == name for a in agg)
        if exists:
            handles.append(plt.Line2D([], [], marker=st.get('marker', ''),
                                      color=st['color'], ls=st.get('ls', '-'),
                                      label=st['label']))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=4, fontsize=9, frameon=False)
    fig.suptitle('Substrate phase diagram: N=256, CV=0.20, '
                 'dt~U[2us,20us], 10 seeds per config',
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.895])

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)  # vector PDF for journal submission
    print(f"saved: {out}")


if __name__ == '__main__':
    main()
