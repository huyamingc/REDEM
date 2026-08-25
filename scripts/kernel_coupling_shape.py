#!/usr/bin/env python3
"""
Kernel-coupling shape analysis (Paper A "the kernel survives coupling").
=============================================================================
Type:           PAPER
Paper Section:  Paper A, "The kernel survives coupling (shape stability)"
Experiment:     For every operating point of the committed phase diagram, fit
                an effective log-normal spectrum to the per-lag held-out
                memory curve sqrt(MC(k)); report the Pearson correlation of
                the fitted curve and of the physical spectrum (174 us, CV
                0.20) against sqrt(MC(k)), plus the fitted median stretch.

Physics:        single-unit theory M(t) = E[exp(-t/tau)] over log-normal tau
                (Gauss-Hermite quadrature, as in the paper's Appendix). Coupled
                operation may dilate the effective median (near-critical
                slowdown) while the kernel shape survives.

Outputs:
  data/kernel_coupling_shape_v1.json   (per-config fit + r values)
  data/kernel_coupling_shape_v1.csv    (same, tabular)
=============================================================================
"""
import os
import json
import csv

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
SRC_PATH = os.path.join(DATA_DIR, 'substrate_phase_diagram_v2.json')
OUT_JSON = os.path.join(DATA_DIR, 'kernel_coupling_shape_v1.json')
OUT_CSV = os.path.join(DATA_DIR, 'kernel_coupling_shape_v1.csv')

DT_BAR = 11e-6          # mean pulse interval of the U[2,20]us drive [s]
TAU0 = 174e-6           # physical median trap time constant [s]
CV0 = 0.20              # physical spectrum width
KMAX = 50               # lags used (leakage buffer protocol)

_HERM = None


def _herm():
    """Gauss-Hermite nodes/weights (160 nodes, exact to machine precision)."""
    global _HERM
    if _HERM is None:
        _HERM = hermgauss(160)
    return _HERM


def m_lognormal(t_sec, median, cv):
    """M(t) = E[exp(-t/tau)] for log-normal tau with given median and CV."""
    x, w = _herm()
    sigma = np.sqrt(np.log(1.0 + cv ** 2))
    mu = np.log(median) - 0.5 * sigma ** 2
    tau = np.exp(mu + sigma * np.sqrt(2.0) * x)
    return np.sum(w / np.sqrt(np.pi) * np.exp(-t_sec[:, None] / tau[None, :]),
                  axis=1)


def fit_effective(r_mc, t_sec):
    """Fit (median, CV) by least squares on ln sqrt(MC) vs ln M(t), k=1..KMAX.

    Multi-start Nelder-Mead over log-parameters. The effective width is
    bounded to the physical range CV in [0.02, 3]: unconstrained, the fit
    degenerates to arbitrarily wide spectra (the paper's "fit saturates at
    wide CV" identifiability note), so the fitted median is reported only
    as a diagnostic, not as a precise renormalization map.
    """
    y = np.log(np.clip(r_mc[1:KMAX + 1], 1e-300, None))

    def loss(p):
        med = np.exp(p[0])
        cv = np.exp(p[1])
        m = m_lognormal(t_sec[1:KMAX + 1], med, cv)
        return float(np.sum((np.log(np.clip(m, 1e-300, None)) - y) ** 2))

    best = None
    for med0 in (30e-6, 174e-6, 600e-6, 2000e-6):
        for cv0 in (0.2, 0.5, 1.0, 2.0):
            res = minimize(loss, [np.log(med0), np.log(cv0)],
                           method='Nelder-Mead',
                           bounds=[(np.log(5e-6), np.log(5e-3)),
                                   (np.log(0.02), np.log(3.0))],
                           options={'maxiter': 4000, 'xatol': 1e-7,
                                    'fatol': 1e-14})
            if best is None or res.fun < best.fun:
                best = res
    med = float(np.exp(best.x[0]))
    cv = float(np.exp(best.x[1]))
    m = m_lognormal(t_sec[1:KMAX + 1], med, cv)
    r_fit = float(np.corrcoef(r_mc[1:KMAX + 1], m)[0, 1])
    return med, cv, r_fit, float(best.fun)


def main():
    d = json.load(open(SRC_PATH))
    rows = []
    for a in d['aggregates']:
        cfg = a['config_name']
        kappa = a['kappa']
        mc = np.array(a['mc_curve_mean'])
        if len(mc) < KMAX + 1:
            continue
        r_mc = np.sqrt(np.clip(mc, 0, None))
        k = np.arange(len(mc))
        t_sec = k * DT_BAR  # lag k in time (lag steps of dt_bar)
        med, cv, r_fit, sse = fit_effective(r_mc, t_sec)
        m_phys = m_lognormal(t_sec[1:KMAX + 1], TAU0, CV0)
        r_phys = float(np.corrcoef(r_mc[1:KMAX + 1], m_phys)[0, 1])
        rows.append({
            'config_name': cfg,
            'kappa': kappa,
            'r_fit': r_fit,
            'r_phys': r_phys,
            'med_eff_us': med * 1e6,
            'cv_eff': cv,
            'stretch': med / TAU0,
            'sse': sse,
        })
        print(f'{cfg:14s} k={kappa:7.3f}  r_fit={r_fit:.3f}  '
              f'r_phys={r_phys:.3f}  med_eff={med * 1e6:7.1f}us  '
              f'cv_eff={cv:.2f}  stretch={med / TAU0:.2f}x')

    out = {
        'params': {
            'note': 'Paper A kernel-coupling shape analysis: effective '
                    'log-normal fit to sqrt(MC(k)) (held-out), k=1..50; '
                    'physical spectrum 174us/CV0.20 for r_phys',
            'source': 'data/substrate_phase_diagram_v2.json '
                      '(mc_curve_mean per config)',
            'dt_bar_s': DT_BAR,
            'tau0_s': TAU0,
            'cv0': CV0,
            'kmax': KMAX,
        },
        'rows': rows,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'saved: {OUT_JSON}')
    print(f'saved: {OUT_CSV}')


if __name__ == '__main__':
    main()
