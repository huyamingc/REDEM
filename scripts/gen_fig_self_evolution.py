#!/usr/bin/env python3
"""
Self-evolution mainline figures (Paper E): key curves from the committed CSVs.
=============================================================================
Type:           FIG
Paper Section:  Paper E (self-evolution mainline, S58b-S62)
Experiment:     Reads exactly these committed CSVs and renders the six
                headline curves as PDFs into ../figures/:
  f1 reliability_cliff  : data/s58b_relative_sense_stress_v1.csv
                          (trigger reliability vs SEG_BLOCKS, ABAB/ABCABC)
  f2 margin_kappa_N     : data/s58d_kernel_capacity_sweep_v1.csv
                          (margin vs kappa + N axis, 4D shell / 2D circle)
  f3 d2_threshold       : data/s58e_d2_timescale_v1.csv
                          (post-swap recovery vs tau_env/tau_2, 3 arms)
  f4 vsnap_gate         : data/s59_self_tuned_gate_v1.csv
                          (learned snapshot gate v_snap, dyn vs static)
  f5 cross_family       : data/s61_cross_family_v1.csv
                          (per-segment accuracy pop_mem vs rls_single)
  f6 meta_timescale     : data/s62_adaptive_meta_v1.csv
                          (post vs ratio, meta_self vs meta_adapt)
                No other data file is read (the earlier docstring listed a
                bracketed range that included s56/s60, which this script
                never opens; corrected 2026-09-09, dedup P1-99).

Uncertainty (dedup P1-22): every curve carries error bars and every axis
states the sample size. Per-point error is the standard error of the mean
over the seeds in that cell (SEM = SD/sqrt(n)); the f1 reliability factors are
binomial trigger rates, so their error bar is the Wilson 95% interval of each
factor propagated into the product. Curves are therefore seed-mean +/- SEM
(n = number of seeds per cell, = N_SEEDS unless a cell has fewer runs).
=============================================================================
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

# Data files actually read by this script (kept in one place so the docstring
# and the code cannot drift apart again).
DATA_FILES = {
    'f1': 's58b_relative_sense_stress_v1.csv',
    'f2': 's58d_kernel_capacity_sweep_v1.csv',
    'f3': 's58e_d2_timescale_v1.csv',
    'f4': 's59_self_tuned_gate_v1.csv',
    'f5': 's61_cross_family_v1.csv',
    'f6': 's62_adaptive_meta_v1.csv',
}


def load(path, filt=None):
    rows = list(csv.DictReader(open(path)))
    if filt:
        rows = [r for r in rows if all(r[k] == str(v)
                                       for k, v in filt.items())]
    return rows


def mean(rows, field):
    vals = np.array([float(r[field]) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float('nan')


def mean_sem(rows, field):
    """(mean, SEM) over the rows (one row per seed); SEM = 0 when n < 2."""
    vals = np.array([float(r[field]) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return float('nan'), 0.0
    if vals.size < 2:
        return float(vals[0]), 0.0
    return (float(np.mean(vals)),
            float(np.std(vals, ddof=1) / np.sqrt(vals.size)))


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (k of n)."""
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability_point(rows_rnd, rows_par):
    """(reliability, lo, hi, false_n, false_tot, corr_n, corr_tot).

    reliability = (1 - false_rate_random) * correct_rate_parallel, the same
    definition as s58b's reliability_table; the interval propagates the Wilson
    95% intervals of the two factors (disjoint run sets).
    """
    fn = sum(1 for r in rows_rnd if float(r['switch_block']) >= 0)
    ft = len(rows_rnd)
    cn = sum(1 for r in rows_par if float(r['switch_block']) >= 0)
    ct = len(rows_par)
    if ft == 0 or ct == 0:
        return (float('nan'), float('nan'), float('nan'), fn, ft, cn, ct)
    rel = (1.0 - fn / ft) * (cn / ct)
    f_lo, f_hi = wilson_ci(fn, ft)
    c_lo, c_hi = wilson_ci(cn, ct)
    return (rel, (1.0 - f_hi) * c_lo, (1.0 - f_lo) * c_hi, fn, ft, cn, ct)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---------- f1: s58b reliability cliff ----------
    b58 = load(os.path.join(DATA_DIR, DATA_FILES['f1']))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for seq in ['ABAB', 'ABCABC']:
        segs = sorted({int(r['seg_len']) for r in b58
                       if r['seq_kind'] == seq})
        rel, lo, hi = [], [], []
        for s in segs:
            rnd = [r for r in b58 if r['seq_kind'] == seq
                   and int(r['seg_len']) == s
                   and r['substrate'] == 'random_graph_k25']
            par = [r for r in b58 if r['seq_kind'] == seq
                   and int(r['seg_len']) == s
                   and r['substrate'] == 'parallel']
            r0, l0, h0, fn, ft, cn, ct = reliability_point(rnd, par)
            rel.append(r0)
            lo.append(max(0.0, r0 - l0))
            hi.append(max(0.0, h0 - r0))
        ax.errorbar(segs, rel, yerr=[lo, hi], marker='o', capsize=2.5,
                    lw=1.0, label=seq)
    # shading marks the collapse region only (ratio window/SEG >= 2 => SEG <=
    # 120 for the 240-block window); the earlier 0-260 span overshot it.
    ax.axvspan(0, 120, color='red', alpha=0.08)
    ax.set_xscale('log', base=2)
    ax.set_xticks([60, 125, 250, 500])
    ax.set_xticklabels(['60', '125', '250', '500'])
    ax.set_xlabel('SEG_BLOCKS (regime length)')
    ax.set_ylabel('trigger reliability\n(mean; Wilson 95% CI)')
    ax.set_title('S58b: capacity-sense reliability cliff\n'
                 '(WIN/SEG >= ~2 collapses; n=10 seeds x 2 arms)',
                 fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, title='n=20 runs/point', title_fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'self_evo_f1_reliability_cliff.pdf'))
    plt.close(fig)

    # ---------- f2: s58d margin vs kappa / N ----------
    b58d = load(os.path.join(DATA_DIR, DATA_FILES['f2']))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for env, ax in zip(['4Dshell', '2Dcircle'], axes):
        ks = sorted({float(r['kappa']) for r in b58d
                     if r['env'] == env and r['n_units'] == '256'})
        mg, mg_e = [], []
        for k in ks:
            rs = [r for r in b58d if r['env'] == env
                  and float(r['kappa']) == k and r['n_units'] == '256']
            m, e = mean_sem(rs, 'rls_steady')
            mg.append(m - 0.5)
            mg_e.append(e)
        ax.errorbar(ks, mg, yerr=mg_e, marker='o', capsize=2.5, lw=1.0,
                    label='margin vs kappa (N=256)')
        ns = sorted({int(r['n_units']) for r in b58d
                     if r['env'] == env and r['kappa'] == '25.0'})
        mg2, mg2_e = [], []
        for n in ns:
            rs = [r for r in b58d if r['env'] == env
                  and r['kappa'] == '25.0' and int(r['n_units']) == n]
            m, e = mean_sem(rs, 'rls_steady')
            mg2.append(m - 0.5)
            mg2_e.append(e)
        ax2 = ax.twiny()
        ax2.errorbar(ns, mg2, yerr=mg2_e, marker='s', linestyle='--',
                     color='C1', capsize=2.5, lw=1.0,
                     label='margin vs N (kappa=25)')
        ax2.set_xlabel('N (units)', color='C1')
        ax2.tick_params(axis='x', labelcolor='C1')
        ax.set_xlabel('kappa')
        ax.set_ylabel('margin (steady - 0.5), mean +/- SEM')
        ax.set_title(env, fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0.0, color='k', lw=0.5)
        ax.legend(fontsize=6, loc='best')
    # Wording corrected 2026-09-09 (dedup P1-99): the N axis is NOT monotone
    # in the 2D-circle panel (0.401 -> 0.390 -> 0.410), so the suptitle must
    # not claim monotonicity in N.
    fig.suptitle('S58d: rich-kernel margin — inverted-U in kappa; '
                 'N dependence is panel-specific (not monotone for 2D circle)\n'
                 'n=10 seeds/cell; error bars = SEM', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIG_DIR, 'self_evo_f2_margin_kappa_N.pdf'))
    plt.close(fig)

    # ---------- f3: s58e D2 threshold ----------
    b58e = load(os.path.join(DATA_DIR, DATA_FILES['f3']))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for arm in ['rmhl_oracle', 'passive_fixed', 'meta_self']:
        ratios = sorted({float(r['ratio']) for r in b58e})
        posts, errs = [], []
        for ratio in ratios:
            rs = [r for r in b58e if r['substrate'] == 'random_graph_k25'
                  and r['readout'] == arm and float(r['ratio']) == ratio]
            m, e = mean_sem(rs, 'post_mean')
            posts.append(m)
            errs.append(e)
        ax.errorbar(ratios, posts, yerr=errs, marker='o', capsize=2.5,
                    lw=1.0, label=arm)
    ax.axvspan(0.25, 1.0, color='red', alpha=0.08)
    ax.set_xscale('log', base=2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 10])
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '4', '10'])
    ax.set_xlabel('tau_env / tau_2')
    ax.set_ylabel('post-swap recovery\n(mean +/- SEM)')
    ax.set_title('S58e: D2 threshold in the interior points of the sweep\n'
                 '(n=10 seeds/point; threshold bracketed by data)',
                 fontsize=8)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'self_evo_f3_d2_threshold.pdf'))
    plt.close(fig)

    # ---------- f4: s59 self-tuned snapshot gate ----------
    b59 = load(os.path.join(DATA_DIR, DATA_FILES['f4']))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    envs = ['dyn', 'static']
    fs, fe = zip(*[mean_sem([r for r in b59 if r['env'] == e
                             and r['readout'] == 'mem_vlearn'], 'v_snap_final')
                   for e in envs])
    q1s, q1e = zip(*[mean_sem([r for r in b59 if r['env'] == e
                               and r['readout'] == 'mem_vlearn'], 'v_snap_q1')
                     for e in envs])
    x = np.arange(2)
    ax.bar(x - 0.15, q1s, 0.3, yerr=q1e, capsize=3,
           label='gate early (q1)', color='C2')
    ax.bar(x + 0.15, fs, 0.3, yerr=fe, capsize=3,
           label='gate final', color='C3')
    ax.axhline(0.55, color='k', ls='--', lw=1,
               label='hand-set threshold (0.55)')
    ax.set_xticks(x)
    ax.set_xticklabels(['dynamic\n(memory useful)', 'static\n(memory redundant)'])
    ax.set_ylabel('learned snapshot gate v_snap\n(mean +/- SEM)')
    ax.set_ylim(0, 1.0)
    ax.set_title('S59 (G1): the memory layer learns WHEN to keep\n'
                 'experience (n=10 seeds/env)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'self_evo_f4_vsnap_gate.pdf'))
    plt.close(fig)

    # ---------- f5: s61 cross-family ----------
    b61 = load(os.path.join(DATA_DIR, DATA_FILES['f5']))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    seg_labels = ['s1 lin', 's2 cir', 's3 lin', 's4 ring', 's5 lin']
    for ax, topo in zip(axes, ['random_graph_k25', 'parallel']):
        x = np.arange(5)
        pm, pme = zip(*[mean_sem([r for r in b61 if r['substrate'] == topo
                                  and r['readout'] == 'pop_mem'],
                                 f'seg{s}_acc') for s in range(1, 6)])
        rs, rse = zip(*[mean_sem([r for r in b61 if r['substrate'] == topo
                                  and r['readout'] == 'rls_single'],
                                 f'seg{s}_acc') for s in range(1, 6)])
        ax.errorbar(x, pm, yerr=pme, marker='o', capsize=2.5, lw=1.0,
                    label='pop_mem')
        ax.errorbar(x, rs, yerr=rse, marker='s', ls='--', capsize=2.5,
                    lw=1.0, label='rls_single')
        ax.axvspan(3, 4, color='red', alpha=0.1)
        ax.set_xticks(x)
        ax.set_xticklabels(seg_labels, fontsize=7)
        ax.set_ylabel('segment accuracy (mean +/- SEM)')
        ax.set_ylim(0.45, 1.02)
        ax.set_title(topo, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle('S61 (G3): cross-family transfer is weak and only '
                 'partly structure-matched (ring segment shaded)\n'
                 'n=10 seeds/segment; error bars = SEM', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(FIG_DIR, 'self_evo_f5_cross_family.pdf'))
    plt.close(fig)

    # ---------- f6: s62 adaptive meta timescale ----------
    b62 = load(os.path.join(DATA_DIR, DATA_FILES['f6']))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for arm in ['rmhl_oracle', 'meta_self', 'meta_adapt']:
        ratios = sorted({float(r['ratio']) for r in b62})
        posts, errs = [], []
        for ratio in ratios:
            rs = [r for r in b62 if r['substrate'] == 'random_graph_k25'
                  and r['readout'] == arm and float(r['ratio']) == ratio]
            m, e = mean_sem(rs, 'post_mean')
            posts.append(m)
            errs.append(e)
        ax.errorbar(ratios, posts, yerr=errs, marker='o', capsize=2.5,
                    lw=1.0, label=arm)
    ax.axvspan(0.24, 1.0, color='red', alpha=0.08)
    ax.set_xscale('log', base=2)
    # Tick values taken from the data (ratio = swap_every/50 = 0.24 for the
    # shortest regime; the earlier fixed tick 0.25 did not match any point).
    ax.set_xticks([0.24, 0.5, 1, 2])
    ax.set_xticklabels(['0.24', '0.5', '1', '2'])
    ax.set_xlabel('tau_env / tau_2')
    ax.set_ylabel('post-swap recovery (mean +/- SEM)')
    ax.set_title('S62 (G4): adaptive tau_2 cannot soften the D2\n'
                 'threshold (n=10 seeds/point)', fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'self_evo_f6_meta_timescale.pdf'))
    plt.close(fig)

    print(f"figures written to {FIG_DIR}:")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith('self_evo_'):
            print('  ', f)


if __name__ == '__main__':
    main()
