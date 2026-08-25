#!/usr/bin/env python3
"""
Forgetting-curve theory for the log-normal tau spectrum (Paper A theory).
=============================================================================
Type:           EXPLORE  (analysis + figure; no simulation loops, no njit)
Paper Section:  Paper A, multi-timescale forgetting section
Experiment:     M(t) = E[exp(-t/tau)] for log-normal tau -- the physical
                forgetting kernel of the Si3N4 substrate -- compared with
                single-exponential, power-law (Benna-Fusi style) and
                stretched-exponential decays, and validated against the
                S1 parallel-substrate memory-capacity curve.

Physics: each trap relaxes as exp(-t/tau); the substrate's population
kernel is the average over the log-normal tau spectrum:
    M(t) = int p(tau) exp(-t/tau) dtau,  p = lognormal(median tau0, CV).
Gauss-Hermite quadrature over z = (ln tau - mu)/sigma gives M(t) exactly
and cheaply. As t -> inf the tail is dominated by the log-normal upper
tail: ln M(t) ~ -(ln t - mu)^2 / (2 sigma^2) (log-Gaussian decay), which
is much slower than a single exponential but faster than a power law.

Outputs:
  data/forgetting_curve_theory.csv   (M(t) per CV on the pulse grid)
  data/forgetting_curve_theory_overlay_v1.json
                                    (Pearson r of sqrt(MC(k)) vs M(k) at
                                     CV=0.2 -- the Paper A r = 0.97 anchor)
  figures/forgetting_curve_theory.pdf (vector PDF for journal submission)
  (a) M(t) vs t in pulses, log-log, for CV in {0.02, 0.2, 0.5, 1.0}
      with single-exp, power-law and stretched-exp references
  (b) local log-log slope (decay regime characterization)
  (c) Weibull plot (ln(-ln M) vs ln t): stretched-exp would be linear
  (d) S1 parallel-substrate MC curve overlay: sqrt(MC(k)) vs M(k*dt_bar)
"""
import os
import sys
import json

import numpy as np
from numpy.polynomial.hermite import hermgauss

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
CSV_PATH = os.path.join(DATA_DIR, 'forgetting_curve_theory.csv')
FIG_PATH = os.path.join(FIG_DIR, 'forgetting_curve_theory.pdf')

TAU0 = 174e-6          # median trap time constant [s] (paper anchor)
DT_BAR = 11e-6         # mean pulse interval of the U[2,20]us drive [s]
PULSES_PER_SEC = 1.0 / DT_BAR

CVS = [0.02, 0.2, 0.5, 1.0]


def lognormal_params(cv, median=TAU0):
    """Return (mu, sigma) of the log-normal with given CV and median."""
    sigma = np.sqrt(np.log(1.0 + cv ** 2))
    mu = np.log(median) - 0.5 * sigma ** 2
    return mu, sigma


def forgetting_kernel(t_sec, cv, median=TAU0):
    """M(t) = E[exp(-t/tau)] for log-normal tau, Gauss-Hermite quadrature."""
    mu, sigma = lognormal_params(cv, median)
    x, w = hermgauss(160)
    tau = np.exp(mu + sigma * np.sqrt(2.0) * x)
    return float(np.sum(w / np.sqrt(np.pi) * np.exp(-t_sec / tau)))


def forgetting_curve(cvs, t_sec_grid):
    """M(t) for each cv on the time grid. Returns (t_sec, M (len(cvs), T))."""
    out = np.empty((len(cvs), len(t_sec_grid)))
    for i, cv in enumerate(cvs):
        for j, t in enumerate(t_sec_grid):
            out[i, j] = forgetting_kernel(t, cv)
    return out


def effective_horizon(t_sec, m):
    """Pulse index where M(t) = e^-1 (the 1/e memory horizon)."""
    i = int(np.argmin(np.abs(m - np.exp(-1.0))))
    return t_sec[i] * PULSES_PER_SEC


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # pulse grid: 0.02 .. 20000 pulses
    t_pulse = np.logspace(np.log10(0.02), np.log10(20000.0), 400)
    t_sec = t_pulse / PULSES_PER_SEC
    M = forgetting_curve(CVS, t_sec)

    # references (cv -> 0 limit is the single exponential)
    single_exp = np.exp(-t_sec / TAU0)
    # power law with slope 0.5 matched at t=1 pulse
    pow_ref = (1.0 + t_pulse) ** -0.5
    pow_ref *= np.exp(-1.0) / pow_ref[np.argmin(np.abs(t_pulse - 1.0))]
    # stretched exponential with k = 0.5
    strexp = np.exp(-np.sqrt(t_pulse))
    strexp *= np.exp(-1.0) / strexp[np.argmin(np.abs(t_pulse - 1.0))]

    # CSV: pulses | M(cv=0.02) | M(0.2) | M(0.5) | M(1.0) | single_exp
    with open(CSV_PATH, 'w', newline='') as f:
        f.write('pulses,')
        f.write(','.join(f'M_cv{cv}' for cv in CVS))
        f.write(',single_exp\n')
        for j in range(len(t_pulse)):
            f.write(f'{t_pulse[j]:.6g},')
            f.write(','.join(f'{M[i, j]:.6e}' for i in range(len(CVS))))
            f.write(f',{single_exp[j]:.6e}\n')

    # decay-regime analysis: local log-log slope at large t
    print('Forgetting kernel: 1/e horizons (pulses) and tail slope d ln M / d ln t')
    tail_slopes = []
    for i, cv in enumerate(CVS):
        h = effective_horizon(t_sec, M[i])
        lo, hi = 5000, 20000
        jlo = int(np.argmin(np.abs(t_pulse - lo)))
        jhi = int(np.argmin(np.abs(t_pulse - hi)))
        slope = (np.log(M[i, jhi]) - np.log(M[i, jlo])) / \
                (np.log(t_pulse[jhi]) - np.log(t_pulse[jlo]))
        tail_slopes.append(slope)
        print(f'  CV={cv:<5} horizon={h:>10.1f} pulses  tail_slope={slope:+.4f}')
    print('  references: single-exp slope -> -inf (log-log), power-law slope = -0.5,')
    print('             stretched-exp k=0.5 slope -> -0.5 (log-log)')
    # Benna-Fusi ideal: power-law slope -0.5 (their exponent regime)

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    ax = axes[0, 0]
    for i, cv in enumerate(CVS):
        ax.loglog(t_pulse, M[i], color=colors[i],
                  label=f'log-normal CV={cv}')
    ax.loglog(t_pulse, single_exp, 'k--', lw=1.2, label='single exp (CV->0)')
    ax.loglog(t_pulse, pow_ref, 'k:', lw=1.2, label='power law t^-0.5')
    ax.loglog(t_pulse, strexp, 'k-.', lw=1.2, label='stretched exp k=0.5')
    ax.set_xlabel('pulses t')
    ax.set_ylabel('M(t)')
    ax.set_title('(a) forgetting kernel M(t) = E[exp(-t/tau)]')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[0, 1]
    for i, cv in enumerate(CVS):
        slope = np.gradient(np.log(M[i]), np.log(t_pulse))
        ax.semilogx(t_pulse, slope, color=colors[i], label=f'CV={cv}')
    ax.axhline(-0.5, color='k', ls=':', lw=1.2, label='power-law slope')
    ax.set_xlabel('pulses t')
    ax.set_ylabel('d ln M / d ln t')
    ax.set_title('(b) local decay slope (log-log)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for i, cv in enumerate(CVS):
        ax.semilogx(t_pulse, np.log(-np.log(M[i])), color=colors[i],
                    label=f'CV={cv}')
    ax.axhline(0.0, color='k', lw=0.5)
    ax.set_xlabel('pulses t (log)')
    ax.set_ylabel('ln(-ln M)')
    ax.set_title('(c) Weibull plot: stretched-exp would be linear')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    # S1 parallel-substrate memory capacity overlay
    try:
        with open(os.path.join(DATA_DIR, 'substrate_phase_diagram_v2.json')) as f:
            s1 = json.load(f)
        for a in s1['aggregates']:
            if a['config_name'] == 'parallel':
                mc_curve = np.array(a['mc_curve_mean'])
                k = np.arange(len(mc_curve))
                r = np.sqrt(np.clip(mc_curve, 0, None))
                ax.semilogx(k[1:], r[1:], 'ks', ms=3,
                            label='S1 parallel sqrt(MC(k))')
                t_ref = k  # lag k in pulses (DT_BAR * PULSES_PER_SEC = 1)
                # theory kernel evaluated AT the lag times t = k pulses
                # (interpolated on the log pulse grid)
                M_at_k = np.interp(t_ref[1:], t_pulse, M[1])
                ax.semilogx(t_ref[1:], M_at_k,
                            color=colors[1], lw=1.5,
                            label='theory M(t), CV=0.2')
                # Pearson correlation over the plotted lags (Paper A r = 0.97)
                corr_r = float(np.corrcoef(r[1:], M_at_k)[0, 1])
                m_at_10 = float(np.interp(10.0, t_pulse, M[1]))
                print(f'  overlay: Pearson r(sqrt(MC(k)), M(k)) k=1..{len(k) - 1}'
                      f' = {corr_r:.4f}  (lag-10: sqrt(MC) = {r[10]:.4f} vs '
                      f'M(10) = {m_at_10:.4f})')
                overlay = {
                    'params': {
                        'note': 'Paper A: sqrt(MC(k)) vs M(k) at CV=0.2, '
                                'lag k in pulses; S1 parallel substrate '
                                '(held-out protocol)',
                        'cv': 0.2,
                        'k_range': [1, int(len(k) - 1)],
                        'data': 'data/substrate_phase_diagram_v2.json '
                                '(parallel mc_curve_mean) + '
                                'data/forgetting_curve_theory.csv (M_cv0.2)',
                    },
                    'r_pearson': corr_r,
                    'lag10': {'sqrt_mc': float(r[10]), 'm_theory': m_at_10},
                }
                overlay_path = os.path.join(
                    DATA_DIR, 'forgetting_curve_theory_overlay_v1.json')
                with open(overlay_path, 'w') as f:
                    json.dump(overlay, f, indent=2)
                print(f'saved: {overlay_path}')
                break
        ax.set_xlabel('lag k (pulses)')
        ax.set_ylabel('corr r / M(t)')
        ax.set_title('(d) S1 memory capacity vs theory kernel')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    except Exception as exc:
        ax.text(0.5, 0.5, f'S1 overlay unavailable: {exc}', ha='center')
        ax.set_title('(d) S1 overlay (missing data)')

    fig.suptitle('Log-normal tau forgetting kernel: physics-designed decay curve',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_PATH)  # vector PDF for journal submission
    print(f'\nsaved: {FIG_PATH}')
    print(f'saved: {CSV_PATH}')


if __name__ == '__main__':
    main()
