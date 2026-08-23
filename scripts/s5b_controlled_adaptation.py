#!/usr/bin/env python3
"""
S5 three-arm controlled adaptation re-measurement (Paper B revision).
=============================================================================
Type:           PAPER
Experiment:     Re-measure the S5 dual-timescale metadata arms (fast / dual
                / slow readouts on the Si3N4 substrate) with the S15
                controlled protocol: known regime-switch instants, T_adapt
                reported switch-relative in pulses with a per-switch
                distribution. Resolves whether the S5 "9-20x faster
                adaptation" claim (computed from window positions with
                near-zero denominators) survives a controlled measurement.

Arms (S5 protocol, 10 seeds, substrates parallel + random_graph_k25):
  fast      : features = fast current ratios only (single-scale)
  dual_tau  : features = fast + slow EMA (tau_slow in {200, 1000})
  slow_tau  : features = slow EMA only (tau_slow in {200, 1000})

Metrics per switch (5 switches/stream, same as S15):
  T_adapt_200 : first t >= t_s + 199 with 200-pulse running accuracy
                >= 0.98 (continuity with the S5/S8 window metric, now
                switch-relative and exact)
  T_adapt_40  : first t >= t_s + 39 with 40-pulse running accuracy
                >= 0.95 (fine-grained)

Output files:
  data/s5b_controlled_adaptation_v1.csv
  data/s5b_controlled_adaptation_v1.json

Usage: python s5b_controlled_adaptation.py [--quick] [--sequential]
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
from streaming_tasks import gen_regime_switch, RS_REGIME_LEN, RS_N_SEGMENTS

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

W_200, TH_200 = 200, 0.98
W_40, TH_40 = 40, 0.95

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's5b_controlled_adaptation_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's5b_controlled_adaptation_v1.json')


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def slow_ema(fast, tau_slow):
    """Per-unit EMA of fast features (S5 metadata semantics)."""
    lam = 1.0 / tau_slow
    out = np.empty_like(fast)
    m = fast[0].copy()
    for t in range(fast.shape[0]):
        m = (1.0 - lam) * m + lam * fast[t]
        out[t] = m
    return out


def first_crossing(hits, t_start, w, thr):
    """First t >= t_start+w-1 with w-pulse running accuracy >= thr,
    returned as pulses after t_start (NaN if never reached)."""
    T = hits.shape[0]
    cum = np.cumsum(np.concatenate([[0.0], hits]))
    for t in range(t_start + w - 1, T):
        acc = (cum[t + 1] - cum[t + 1 - w]) / w
        if acc >= thr:
            return float(t - t_start)
    return float('nan')


def run_single(args):
    """(topo_name, arm, tau_slow, seed_idx) -> dict with per-switch T_adapt."""
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
        F = np.hstack([fast_s, (slow - mu) / sd])
    elif arm == 'slow':
        slow = slow_ema(fast, tau_slow)
        F = (slow - mu) / sd
    else:
        raise ValueError(arm)
    F = np.hstack([F, np.full((T, 1), BIAS)])

    rls3 = OnlineRLS(F.shape[1], 3, forgetting=RLS_FORGETTING,
                     init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                     reg=RLS_REG)
    Y3 = np.zeros((T, 3))
    Y3[np.arange(T), regime_seq] = 1.0
    _, preds3 = rls3.fit_stream(F, Y3, n_warmup=200)
    pred3 = preds3.argmax(axis=1)
    acc = float(np.mean(pred3 == regime_seq))

    hits = (pred3 == regime_seq).astype(np.float64)
    switch_times = [RS_REGIME_LEN * s for s in range(1, RS_N_SEGMENTS)]

    results = {'substrate': topo_name, 'arm': arm, 'tau_slow': float(tau_slow),
               'seed_idx': seed_idx, 'overall_acc': acc,
               'runtime_s': time.time() - t0}
    for si, t_s in enumerate(switch_times):
        results[f's{si}_t_adapt_200'] = first_crossing(hits, t_s, W_200, TH_200)
        results[f's{si}_t_adapt_40'] = first_crossing(hits, t_s, W_40, TH_40)
    return results


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['substrate'], r['arm'], r['tau_slow']), []).append(r)
    agg = []
    for (sub, arm, tau_s), rs in sorted(groups.items()):
        entry = {'substrate': sub, 'arm': arm, 'tau_slow': tau_s,
                 'n_runs': len(rs),
                 'overall_acc_mean': float(np.mean([r['overall_acc'] for r in rs]))}
        for metric in ['t_adapt_200', 't_adapt_40']:
            vals = []
            for si in range(RS_N_SEGMENTS - 1):
                for r in rs:
                    v = r[f's{si}_{metric}']
                    if not np.isnan(v):
                        vals.append(v)
            vals = np.array(vals)
            if vals.size:
                entry[f'{metric}_mean'] = float(np.mean(vals))
                entry[f'{metric}_std'] = float(np.std(vals))
                entry[f'{metric}_median'] = float(np.median(vals))
                entry[f'{metric}_p90'] = float(np.percentile(vals, 90))
                entry[f'{metric}_n'] = int(vals.size)
            else:
                for k in [f'{metric}_mean', f'{metric}_std',
                          f'{metric}_median', f'{metric}_p90', f'{metric}_n']:
                    entry[k] = float('nan') if 'n' not in k else 0
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 104)
    print("S5b RESULTS: S5 arms under the controlled adaptation protocol "
          "(10 seeds, 5 switches/stream)")
    print("=" * 104)
    for sub in ['parallel', 'random_graph_k25']:
        print(f"\n--- {sub} ---")
        print(f"  {'arm':<12} {'tau_s':>6} | {'overall':>8} | "
              f"{'T40 mean':>8} {'T40 med':>7} {'T40 p90':>7} | "
              f"{'T200 mean':>9} {'T200 med':>8}")
        for a in [x for x in agg if x['substrate'] == sub]:
            print(f"  {a['arm']:<12} {a['tau_slow']:>6.0f} | "
                  f"{a['overall_acc_mean']:>8.4f} | "
                  f"{a.get('t_adapt_40_mean', float('nan')):>8.1f} "
                  f"{a.get('t_adapt_40_median', float('nan')):>7.1f} "
                  f"{a.get('t_adapt_40_p90', float('nan')):>7.1f} | "
                  f"{a.get('t_adapt_200_mean', float('nan')):>9.1f} "
                  f"{a.get('t_adapt_200_median', float('nan')):>8.1f}")


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S5b controlled adaptation "
          f"(quick={quick}, sequential={sequential})")

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
    arms = ['fast', 'dual', 'dual', 'slow', 'slow']
    taus = [0.0, 200.0, 1000.0, 200.0, 1000.0]
    all_args = []
    for topo in ['parallel', 'random_graph_k25']:
        for arm, tau_s in zip(arms, taus):
            for s in range(n_seeds):
                all_args.append((topo, arm, tau_s, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (2 substrates x 5 arm configs x {n_seeds} seeds)")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_single(a))
            if (i + 1) % max(1, n_runs // 10) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=2):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 10) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['substrate', 'arm', 'tau_slow', 'seed_idx', 'overall_acc',
                  'runtime_s']
    for si in range(RS_N_SEGMENTS - 1):
        fieldnames += [f's{si}_t_adapt_200', f's{si}_t_adapt_40']
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
        'rls_reg': RLS_REG,
        'arms': {'fast': 'fast states only',
                 'dual': 'fast + slow EMA (tau_slow)',
                 'slow': 'slow EMA only'},
        'tau_slow_grid': [200.0, 1000.0],
        'regime_len': RS_REGIME_LEN, 'n_segments': RS_N_SEGMENTS,
        'switch_times': [RS_REGIME_LEN * s for s in range(1, RS_N_SEGMENTS)],
        'metrics': {'t_adapt_200': 'first t>=t_s+199 with 200-window acc>=0.98',
                    't_adapt_40': 'first t>=t_s+39 with 40-window acc>=0.95'},
        's5_original_adapt_time_note': 'S5 adapt_time is a window-position '
                                       'metric (near-zero denominators); '
                                       'this script reports switch-relative '
                                       'pulses',
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
