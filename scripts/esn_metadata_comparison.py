#!/usr/bin/env python3
"""
ESN-with-metadata fair comparison (Paper B closing gap claim, REDEM S10).
=============================================================================
Type:           PAPER
Paper Section:  Paper B Discussion: "an ESN with a slow metadata state would
                close the gap on the regime task" -- now tested directly.
Experiment:     regime_switch task (S5/S8 protocol). Arms:
                  esn_fast : ESN-256-hetero (fair_esn_comparison class) +
                             RLS readout on reservoir states only
                  esn_dual : ESN + the S5 metadata (per-unit slow EMA of the
                             reservoir states, tau_slow=500) + RLS
                  redem    : the integrated REDEM full arm (S8 protocol)
                The question: does adding the metadata close the gap between
                the ESN and REDEM on the long-horizon statistical task?

Output files:
  data/s10_esn_metadata_v1.csv    (one row per run)
  data/s10_esn_metadata_v1.json   (params + aggregates)

Usage: python esn_metadata_comparison.py [--quick]
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
    COUPLING_CONTRAST_SELF,
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    adjacency_to_csr, run_trajectory_nb)
from online_readout import OnlineRLS
from streaming_tasks import gen_regime_switch, RS_REGIME_LEN
from fair_esn_comparison import ESN
from integrated_benchmark import random_graph_mask, slow_ema

N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0
TAU_SLOW = 500.0
RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's10_esn_metadata_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's10_esn_metadata_v1.json')


def run_single(args):
    """(arm, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    arm, seed_idx = args
    t0 = time.time()

    dt_seq, regime_seq = gen_regime_switch(seed=seed_idx)
    T = dt_seq.shape[0]
    u_norm = (dt_seq - dt_seq.min()) / max(dt_seq.max() - dt_seq.min(), 1e-12)

    if arm == 'redem':
        # REDEM full arm (S8 protocol): coupled substrate + dual metadata
        tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
        x0 = preprogram_vec(ALPHA0, tau)
        mask = random_graph_mask(N_UNITS)
        ip, idx, wt = adjacency_to_csr(mask)
        states, _, _ = run_trajectory_nb(x0, tau, dt_seq, PW, ip, idx, wt,
                                         KAPPA_RANDOM, ALPHA0, ALPHA_MIN,
                                         ALPHA_MAX, gamma,
                                         COUPLING_CONTRAST_SELF, 0)
        fast = np.exp(gamma * states) / FEATURE_SCALE
        use_meta = True
    else:
        esn = ESN(n_input=1, n_reservoir=N_UNITS, spectral_radius=0.9,
                  input_scaling=0.5, leaking_rate=0.2, hetero_lr=True,
                  cv_lr=CV_TAU, seed=seed_idx + 999)
        states = esn.process(u_norm[:, None])
        fast = states
        use_meta = (arm == 'esn_dual')

    n_fit = int(0.3 * T)
    mu = fast[:n_fit].mean(axis=0)
    sd = fast[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    fast_s = (fast - mu) / sd
    if use_meta:
        slow = slow_ema(fast, TAU_SLOW)
        slow_s = (slow - mu) / sd
        F = np.hstack([fast_s, slow_s])
    else:
        F = fast_s
    F = np.hstack([F, np.full((T, 1), BIAS)])

    rls = OnlineRLS(F.shape[1], 3, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP, reg=RLS_REG)
    Y3 = np.zeros((T, 3))
    Y3[np.arange(T), regime_seq] = 1.0
    _, preds3 = rls.fit_stream(F, Y3, n_warmup=200)
    pred3 = preds3.argmax(axis=1)
    acc = float(np.mean(pred3 == regime_seq))

    seg_len = RS_REGIME_LEN
    n_segs = T // seg_len
    steady = []
    adapt = []
    for s in range(n_segs):
        seg = pred3[s * seg_len:s * seg_len + seg_len]
        trg = regime_seq[s * seg_len:s * seg_len + seg_len]
        steady.append(float(np.mean(seg[-500:] == trg[-500:])))
        if s > 0:
            hits = (seg == trg).astype(float)
            cum = np.cumsum(np.concatenate([[0.0], hits]))
            run = (cum[200:] - cum[:-200]) / 200.0
            thr = steady[-1] - 0.02
            first = np.where(run >= thr)[0]
            adapt.append(float(first[0]) if first.size else np.nan)

    return {'arm': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
            'overall_acc': acc, 'steady_acc': float(np.mean(steady)),
            'adapt_time': float(np.nanmean(adapt)),
            'runtime_s': time.time() - t0}


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(r['arm'], []).append(r)
    agg = []
    for arm, rs in sorted(groups.items()):
        entry = {'arm': arm, 'n_runs': len(rs)}
        for f in ['overall_acc', 'steady_acc', 'adapt_time']:
            v = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(v))
            entry[f + '_std'] = float(np.nanstd(v))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 90)
    print("S10 ESN-metadata comparison (regime_switch, mean over seeds)")
    print("=" * 90)
    print(f"  {'arm':<9} | {'overall':>8} | {'steady':>7} | {'adapt':>8}")
    for a in sorted(agg, key=lambda x: -x['overall_acc_mean']):
        print(f"  {a['arm']:<9} | {a['overall_acc_mean']:>8.3f} | "
              f"{a['steady_acc_mean']:>7.3f} | {a['adapt_time_mean']:>8.0f}")


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S10 ESN-metadata comparison "
          f"(quick={quick})")

    n_seeds = 2 if quick else N_SEEDS
    arms = ['esn_fast', 'esn_dual', 'redem']
    all_args = [(a, s) for a in arms for s in range(n_seeds)]
    print(f"total runs: {len(all_args)}")

    results = []
    with Pool(min(cpu_count(), max(1, len(all_args)))) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=1):
            results.append(res)
            done += 1
            if done % max(1, len(all_args) // 5) == 0 or done == len(all_args):
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{len(all_args)}",
                      flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed_idx', 'n_units', 'overall_acc', 'steady_acc',
                  'adapt_time', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'tau_slow': TAU_SLOW,
        'kappa_random': KAPPA_RANDOM, 'rls_forgetting': RLS_FORGETTING,
        'esn': 'hetero-lr matched to tau spectrum', 'n_seeds': n_seeds,
        'quick': bool(quick),
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
