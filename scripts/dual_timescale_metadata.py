#!/usr/bin/env python3
"""
Dual-timescale metadata benchmark (M3, REDEM S5).
=============================================================================
Type:           PAPER
Paper Section:  New-algorithm project Step S5
Experiment:     Does the slow metadata state (per-unit EMA of fast features,
                natural forgetting) give the readout long-horizon statistical
                memory beyond the fast substrate?

Task: regime_switch (streaming_tasks.py): 3 regimes with OVERLAPPING pulse-
interval ranges, random regime order, 1500 pulses per segment. A single
pulse (or the ~17-pulse fast memory window) cannot identify the regime;
only long-run interval statistics can.

Readout arms (all online RLS, forgetting=0.999):
  fast      : features = fast current ratios only      (257 dims, single-scale)
  dual_tau  : features = fast + slow EMA (tau pulses)  (514 dims)
  slow_tau  : features = slow EMA only                 (257 dims, ablation)

Metrics: overall accuracy, per-regime steady accuracy (last 500 pulses of
each segment), boundary adaptation time (pulses to reach within 2pp of the
segment's steady accuracy after a regime switch).

Expected: dual >= slow > fast on this statistics-dominated task; the
metadata (M3) is what carries the long-horizon regime information.

Output files:
  data/s5_dual_timescale_v1.csv    (one row per run)
  data/s5_dual_timescale_v1.json   (params + per-cell aggregates)

Usage: python dual_timescale_metadata.py [--quick]
"""
import os
import sys
import time
import csv
import json

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shallow_trap_array_simulator import gamma, tau0, gen_tau_vec, preprogram_vec
from recurrent_substrate import (
    COUPLING_NONE, COUPLING_CONTRAST_SELF,
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    run_trajectory_nb)
from online_readout import OnlineRLS
from streaming_tasks import gen_regime_switch, RS_REGIME_LEN, RS_EVENT_RATES, RS_BASE_RANGE, RS_EVENT_RANGE

# ========================== Fixed parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4
TAU_SLOW_GRID = [200, 1000]   # slow EMA time constants (pulses)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's5_dual_timescale_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's5_dual_timescale_v1.json')

CURVE_WINDOW = 200


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def slow_ema(fast, tau_slow):
    """Per-unit exponential moving average of fast features (metadata).
    Returns (T, N) slow features with the same scale as fast.
    """
    lam = 1.0 / tau_slow
    out = np.empty_like(fast)
    m = fast[0].copy()
    for t in range(fast.shape[0]):
        m = (1.0 - lam) * m + lam * fast[t]
        out[t] = m
    return out


def run_single(args):
    """(topo_name, arm, tau_slow, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, tau_slow, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)
    dt_seq, regime_seq = gen_regime_switch(seed=seed_idx)
    T = dt_seq.shape[0]

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    fast = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = fast[:n_fit].mean(axis=0)
    sd = fast[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    fast_s = (fast - mu) / sd

    if arm == 'fast':
        F = fast_s
    elif arm == 'dual':
        slow = slow_ema(fast, tau_slow)
        slow_s = (slow - mu) / sd          # same stats as fast
        F = np.hstack([fast_s, slow_s])
    elif arm == 'slow':
        slow = slow_ema(fast, tau_slow)
        F = (slow - mu) / sd
    else:
        raise ValueError(arm)
    F = np.hstack([F, np.full((T, 1), BIAS)])

    rls3 = OnlineRLS(F.shape[1], 3, forgetting=RLS_FORGETTING,
                     init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP, reg=RLS_REG)
    Y3 = np.zeros((T, 3))
    Y3[np.arange(T), regime_seq] = 1.0
    _, preds3 = rls3.fit_stream(F, Y3, n_warmup=200)
    pred3 = preds3.argmax(axis=1)
    acc = np.mean(pred3 == regime_seq)
    seg_len = RS_REGIME_LEN
    n_segs = T // seg_len
    steady3 = []
    adapt = []
    for s in range(n_segs):
        seg = pred3[s * seg_len:s * seg_len + seg_len]
        trg = regime_seq[s * seg_len:s * seg_len + seg_len]
        steady_seg = float(np.mean(seg[-500:] == trg[-500:]))
        steady3.append(steady_seg)
        if s > 0:
            # adaptation: first pulse where running (window 200) accuracy
            # within 2pp of steady_seg
            hits = ((seg == trg).astype(float))
            cum = np.cumsum(np.concatenate([[0.0], hits]))
            run = (cum[200:] - cum[:-200]) / 200.0
            thr = steady_seg - 0.02
            first = np.where(run >= thr)[0]
            adapt.append(float(first[0]) if first.size else np.nan)
    return {'substrate': topo_name, 'arm': arm, 'tau_slow': float(tau_slow),
            'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T),
            'overall_acc': float(acc),
            'steady_acc': float(np.mean(steady3)),
            'adapt_time': float(np.nanmean(adapt)),
            'runtime_s': time.time() - t0}


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['substrate'], r['arm'], r['tau_slow']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        sub, arm, tau_s = key
        entry = {'substrate': sub, 'arm': arm, 'tau_slow': tau_s,
                 'n_runs': len(rs)}
        for f in ['overall_acc', 'steady_acc', 'adapt_time']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 100)
    print("S5 RESULTS (mean over seeds): regime_switch (3 overlapping regimes)")
    print("=" * 100)
    for sub in ['parallel', 'random_graph_k25']:
        print(f"\n--- {sub} ---")
        print(f"  {'arm':<14} {'tau_slow':>8} | {'overall':>8} | "
              f"{'steady':>7} | {'adapt(pulses)':>13}")
        for a in [x for x in agg if x['substrate'] == sub]:
            print(f"  {a['arm']:<14} {a['tau_slow']:>8.0f} | "
                  f"{a['overall_acc_mean']:>8.3f} | "
                  f"{a['steady_acc_mean']:>7.3f} | "
                  f"{a['adapt_time_mean']:>13.0f}")


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S5 dual-timescale benchmark "
          f"(quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16,
                                           seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_RANDOM,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = 2 if quick else N_SEEDS
    arms = ['fast', 'dual', 'dual', 'slow', 'slow'] if not quick else ['fast', 'dual', 'slow']
    taus = [0.0, 200.0, 1000.0, 200.0, 1000.0] if not quick else [0.0, 1000.0, 1000.0]
    all_args = []
    for topo in ['parallel', 'random_graph_k25']:
        for arm, tau_s in zip(arms, taus):
            for s in range(n_seeds):
                all_args.append((topo, arm, tau_s, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seeds={n_seeds})")

    results = []
    with Pool(min(cpu_count(), max(1, n_runs))) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=2):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['substrate', 'arm', 'tau_slow', 'seed_idx', 'n_units',
                  't_total', 'runtime_s', 'overall_acc', 'steady_acc',
                  'adapt_time']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'kappa_random': KAPPA_RANDOM,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'rls_reg': RLS_REG, 'tau_slow_grid': TAU_SLOW_GRID,
        'regime_ranges': RS_EVENT_RATES, 'regime_base_range': RS_BASE_RANGE,
        'regime_event_range': RS_EVENT_RANGE, 'regime_len': RS_REGIME_LEN,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
