#!/usr/bin/env python3
"""
Controlled adaptation protocol (Paper C S15).
=============================================================================
Type:           PAPER
Experiment:     Precise post-switch adaptation time on regime_switch with
                KNOWN switch instants (exact segment boundaries at
                t = 1500, 3000, ..., 7500; RS_REGIME_LEN = 1500). Fixes the
                windowed-statistic ambiguity of the S10 adapt_time metric:
                T_adapt is reported switch-relative in pulses, with a
                per-switch distribution, instead of a segment-internal
                window position.

Arms (S10 protocol, 10 seeds):
  esn_fast : ESN-256-hetero fast states + OnlineRLS (no metadata)
  esn_dual : ESN + M3 slow trace (tau_slow=500) + OnlineRLS
  redem    : Si3N4 substrate + dual metadata + OnlineRLS (S10 redem arm)

Metrics per switch (5 switches per stream; switch at t_s is guaranteed to
change the regime):
  T_adapt_200 : first t >= t_s + 199 with 200-pulse running accuracy
                >= 0.98 (continuity with the S10 window metric, now
                switch-relative and exact)
  T_adapt_40  : first t >= t_s + 39 with 40-pulse running accuracy
                >= 0.95 (fine-grained, resolves sub-200-pulse adaptation)
  Both in pulses after the switch; NaN if not reached by stream end.

Output files:
  data/s15_controlled_adaptation_v1.csv   (per arm, seed, switch)
  data/s15_controlled_adaptation_v1.json

Usage: python s15_controlled_adaptation.py [--quick] [--sequential]
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
from streaming_tasks import gen_regime_switch, RS_REGIME_LEN, RS_N_SEGMENTS
from fair_esn_comparison import ESN
from integrated_benchmark import random_graph_mask, slow_ema

N_UNITS = 256
CV_TAU = 0.20
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0
TAU_SLOW = 500.0
RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4
WARMUP = 200

W_200, TH_200 = 200, 0.98
W_40, TH_40 = 40, 0.95

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's15_controlled_adaptation_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's15_controlled_adaptation_v1.json')


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
    """(arm, seed_idx) -> dict with per-switch adaptation times."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed_idx = args
    t0 = time.time()

    dt_seq, regime_seq = gen_regime_switch(seed=seed_idx)
    T = dt_seq.shape[0]
    u_norm = (dt_seq - dt_seq.min()) / max(dt_seq.max() - dt_seq.min(), 1e-12)

    if arm == 'redem':
        tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
        x0 = preprogram_vec(ALPHA0, tau)
        mask = random_graph_mask(N_UNITS)
        ip, idx, wt = adjacency_to_csr(mask)
        states, _, _ = run_trajectory_nb(x0, tau, dt_seq, PW, ip, idx, wt,
                                         KAPPA_RANDOM, ALPHA0, ALPHA_MIN,
                                         ALPHA_MAX, gamma,
                                         COUPLING_CONTRAST_SELF, 0)
        fast = np.exp(gamma * states) / FEATURE_SCALE
        use_slow = True
    else:
        esn = ESN(n_input=1, n_reservoir=N_UNITS, spectral_radius=0.9,
                  input_scaling=0.5, leaking_rate=0.2, hetero_lr=True,
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
        slow_s = (slow - mu) / sd
        F = np.hstack([fast_s, slow_s])
    else:
        F = fast_s
    F = np.hstack([F, np.full((T, 1), BIAS)])

    rls = OnlineRLS(F.shape[1], 3, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                    reg=RLS_REG)
    Y3 = np.zeros((T, 3))
    Y3[np.arange(T), regime_seq] = 1.0
    _, preds3 = rls.fit_stream(F, Y3, n_warmup=WARMUP)
    preds = preds3.argmax(axis=1)
    acc_full = float(np.mean(preds == regime_seq))

    hits = (preds == regime_seq).astype(np.float64)
    switch_times = [RS_REGIME_LEN * s for s in range(1, RS_N_SEGMENTS)]

    results = {'arm': arm, 'seed_idx': seed_idx,
               'overall_acc': acc_full, 'runtime_s': time.time() - t0}
    for si, t_s in enumerate(switch_times):
        results[f's{si}_t_adapt_200'] = first_crossing(hits, t_s, W_200, TH_200)
        results[f's{si}_t_adapt_40'] = first_crossing(hits, t_s, W_40, TH_40)
    return results


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(r['arm'], []).append(r)
    agg = []
    for arm, rs in sorted(groups.items()):
        entry = {'arm': arm, 'n_runs': len(rs),
                 'overall_acc_mean': float(np.mean([r['overall_acc'] for r in rs]))}
        n_sw = 5
        for metric in ['t_adapt_200', 't_adapt_40']:
            vals = []
            for si in range(n_sw):
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
    print("\n" + "=" * 96)
    print("S15 RESULTS: controlled adaptation (regime_switch, 10 seeds, "
          "5 switches/stream)")
    print("=" * 96)
    print(f" {'arm':>10} | {'overall':>8} | "
          f"{'T40 mean':>8} {'T40 med':>7} {'T40 p90':>7} {'n':>4} | "
          f"{'T200 mean':>9} {'T200 med':>8} {'n':>4}")
    for a in sorted(agg, key=lambda x: -x.get('t_adapt_40_mean', 1e9)):
        print(f" {a['arm']:>10} | {a['overall_acc_mean']:>8.4f} | "
              f"{a.get('t_adapt_40_mean', float('nan')):>8.1f} "
              f"{a.get('t_adapt_40_median', float('nan')):>7.1f} "
              f"{a.get('t_adapt_40_p90', float('nan')):>7.1f} "
              f"{a.get('t_adapt_40_n', 0):>4d} | "
              f"{a.get('t_adapt_200_mean', float('nan')):>9.1f} "
              f"{a.get('t_adapt_200_median', float('nan')):>8.1f} "
              f"{a.get('t_adapt_200_n', 0):>4d}")


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S15 controlled adaptation "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 2 if quick else N_SEEDS
    all_args = [(arm, s) for arm in ['esn_fast', 'esn_dual', 'redem']
                for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (arms=3, seeds={n_seeds})")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_single(a))
            if (i + 1) % max(1, n_runs // 5) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 5) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed_idx', 'overall_acc', 'runtime_s']
    for si in range(1, RS_N_SEGMENTS):
        fieldnames += [f's{si - 1}_t_adapt_200', f's{si - 1}_t_adapt_40']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'tau_slow': TAU_SLOW,
        'regime_switch': {'regime_len': RS_REGIME_LEN,
                          'n_segments': RS_N_SEGMENTS,
                          'switch_times': [RS_REGIME_LEN * s
                                           for s in range(1, RS_N_SEGMENTS)]},
        'rls_forgetting': RLS_FORGETTING, 'warmup': WARMUP,
        'metrics': {'t_adapt_200': 'first t>=t_s+199 with 200-window acc>=0.98',
                    't_adapt_40': 'first t>=t_s+39 with 40-window acc>=0.95'},
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
