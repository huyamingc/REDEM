#!/usr/bin/env python3
"""
Substrate stress test for the metadata equalizer (Paper C S17).
=============================================================================
Type:           PAPER
Experiment:     Is the S10 equalizer gain (the slow trace raises ESN
                accuracy on regime_switch) substrate-insensitive? Sweeps
                ESN spectral radius {0.7, 0.9, 0.99} and heterogeneity
                (hetero-lr matched to the log-normal spectrum vs uniform
                leaking rate), arms esn_fast / esn_dual (tau_m=500),
                10 seeds. Sharpens "equalizer" vs "ESN-specific fix":
                if dual - fast > 0 at every configuration, the gain is a
                property of the metadata, not of a particular ESN tuning.

Metrics per run: overall accuracy, steady accuracy (mean of last-500
accuracy over segments), on regime_switch (S10 protocol).

Output files:
  data/s17_substrate_stress_v1.csv
  data/s17_substrate_stress_v1.json

Usage: python s17_substrate_stress.py [--quick] [--sequential]
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
from online_readout import OnlineRLS
from streaming_tasks import gen_regime_switch, RS_REGIME_LEN
from fair_esn_comparison import ESN
from integrated_benchmark import slow_ema

N_UNITS = 256
CV_TAU = 0.20
N_SEEDS = 10
BIAS = 1.0
TAU_SLOW = 500.0
RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4
WARMUP = 200

SPECTRAL_RADII = [0.7, 0.9, 0.99]
HETERO_OPTIONS = [True, False]   # hetero-lr matched to log-normal CV vs uniform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's17_substrate_stress_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's17_substrate_stress_v1.json')


def run_single(args):
    """(spectral_radius, hetero, arm, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    sr, hetero, arm, seed_idx = args
    t0 = time.time()

    dt_seq, regime_seq = gen_regime_switch(seed=seed_idx)
    T = dt_seq.shape[0]
    u_norm = (dt_seq - dt_seq.min()) / max(dt_seq.max() - dt_seq.min(), 1e-12)

    esn = ESN(n_input=1, n_reservoir=N_UNITS, spectral_radius=sr,
              input_scaling=0.5, leaking_rate=0.2, hetero_lr=hetero,
              cv_lr=CV_TAU, seed=seed_idx + 999)
    states = esn.process(u_norm[:, None])
    fast = states
    use_slow = (arm == 'esn_dual')

    n_fit = int(0.3 * T)
    mu = fast[:n_fit].mean(axis=0)
    sd = fast[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    fast_s = (fast - mu) / sd
    if use_slow:
        slow = slow_ema(fast, TAU_SLOW)
        F = np.hstack([fast_s, (slow - mu) / sd])
    else:
        F = fast_s
    F = np.hstack([F, np.full((T, 1), BIAS)])

    rls = OnlineRLS(F.shape[1], 3, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                    reg=RLS_REG)
    Y3 = np.zeros((T, 3))
    Y3[np.arange(T), regime_seq] = 1.0
    _, preds3 = rls.fit_stream(F, Y3, n_warmup=WARMUP)
    pred3 = preds3.argmax(axis=1)
    acc = float(np.mean(pred3 == regime_seq))

    seg_len = RS_REGIME_LEN
    n_segs = T // seg_len
    steady = []
    for s in range(n_segs):
        seg = pred3[s * seg_len:s * seg_len + seg_len]
        trg = regime_seq[s * seg_len:s * seg_len + seg_len]
        steady.append(float(np.mean(seg[-500:] == trg[-500:])))

    return {'spectral_radius': float(sr), 'hetero': hetero, 'arm': arm,
            'seed_idx': seed_idx, 'overall_acc': acc,
            'steady_acc': float(np.mean(steady)),
            'runtime_s': time.time() - t0}


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['spectral_radius'], r['hetero'], r['arm']),
                          []).append(r)
    agg = []
    for (sr, het, arm), rs in sorted(groups.items()):
        entry = {'spectral_radius': sr, 'hetero': het, 'arm': arm,
                 'n_runs': len(rs)}
        for f in ['overall_acc', 'steady_acc']:
            v = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.mean(v))
            entry[f + '_std'] = float(np.std(v))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 88)
    print("S17 RESULTS: substrate stress test (mean over 10 seeds)")
    print("=" * 88)
    for het in [True, False]:
        print(f"\n--- hetero_lr = {het} ---")
        print(f"  {'sr':>5} {'arm':>9} | {'overall':>8} | "
              f"{'steady':>7} | {'gain (dual-fast)':>16}")
        duals = {a['spectral_radius']: a for a in agg
                 if a['hetero'] == het and a['arm'] == 'esn_dual'}
        for a in sorted([x for x in agg if x['hetero'] == het],
                        key=lambda x: (x['spectral_radius'], x['arm'])):
            g = ''
            if a['arm'] == 'esn_dual':
                f = next(x for x in agg if x['hetero'] == het
                         and x['arm'] == 'esn_fast'
                         and x['spectral_radius'] == a['spectral_radius'])
                g = f"{a['overall_acc_mean'] - f['overall_acc_mean']:>16.4f}"
            print(f"  {a['spectral_radius']:>5.2f} {a['arm']:>9} | "
                  f"{a['overall_acc_mean']:>8.4f} | "
                  f"{a['steady_acc_mean']:>7.4f} | {g}")


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S17 substrate stress test "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 2 if quick else N_SEEDS
    all_args = [(sr, het, arm, s)
                for sr in SPECTRAL_RADII for het in HETERO_OPTIONS
                for arm in ['esn_fast', 'esn_dual'] for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (3 sr x 2 hetero x 2 arms x {n_seeds} seeds)")

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
    fieldnames = ['spectral_radius', 'hetero', 'arm', 'seed_idx',
                  'overall_acc', 'steady_acc', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'tau_slow': TAU_SLOW,
        'spectral_radii': SPECTRAL_RADII, 'hetero_options': HETERO_OPTIONS,
        'esn': 'input_scaling=0.5, base leaking_rate=0.2',
        'rls_forgetting': RLS_FORGETTING, 'warmup': WARMUP,
        'task': 'regime_switch (S10 protocol)',
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
