#!/usr/bin/env python3
"""
Integrated system + ablation matrix (REDEM S8).
=============================================================================
Type:           PAPER
Paper Section:  New-algorithm project Step S8 (see NEW_ALGORITHM_PLAN.md)
Experiment:     Full system (RLS readout + dual-timescale metadata +
                lambda-homeostat + gentle structure plasticity) vs ablations
                on the regime_switch task; N=1024 scale confirmation.

Task: regime_switch (S5): 3 regimes differing only in the rare-event rate;
requires long-horizon statistical memory (the metadata's domain).

Arms (each mechanism on/off, 10 seeds at N=256):
  full          : RLS + dual metadata (tau_slow=500) + homeostat + plasticity
  no_metadata   : full without the slow features (fast-only readout)
  no_homeostat  : full with kappa fixed at 25
  no_plasticity : full with the random_graph topology fixed
  baseline      : RLS fast-only, kappa=25, fixed random_graph (S2/S5 base)

N=1024 confirmation: baseline vs full, 3 seeds.

Metrics (S5 protocol): overall accuracy, per-segment steady accuracy,
regime-switch adaptation time.

Output files:
  data/s8_integrated_v1.csv    (one row per run)
  data/s8_integrated_v1.json   (params + aggregates)

Usage: python integrated_benchmark.py [--quick]
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
    adjacency_to_csr, run_trajectory_nb, run_pair_ftle_nb)
from online_readout import OnlineRLS
from streaming_tasks import gen_regime_switch, RS_REGIME_LEN
from structure_plasticity import evolve_mask

# ========================== Fixed parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_NOMINAL = 25.0
KAPPA_MIN, KAPPA_MAX = 1.0, 60.0

RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

TAU_SLOW = 500.0            # metadata EMA time constant (pulses)
LAMBDA_TARGET = -0.02
ETA_LAMBDA = 3.0
FTLE_EVERY = 1000
FTLE_WINDOW = 400
HOMEOSTAT_ON = True
PLASTICITY_EVERY = 2000     # rewiring round period (pulses)
PLASTICITY_CHURN = 0.05     # gentle churn fraction per round

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's8_integrated_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's8_integrated_v1.json')


def random_graph_mask(n_units, seed=TOPO_SEED, avg_degree=AVG_DEGREE):
    ip, idx, wt = build_topology_csr('random_graph', n_units, seed=seed,
                                     avg_degree=avg_degree)
    mask = np.zeros((n_units, n_units), dtype=bool)
    for i in range(n_units):
        for e in range(ip[i], ip[i + 1]):
            j = int(idx[e])
            mask[min(i, j), max(i, j)] = True
    return mask


def slow_ema(fast, tau_slow):
    lam = 1.0 / tau_slow
    out = np.empty_like(fast)
    m = fast[0].copy()
    for t in range(fast.shape[0]):
        m = (1.0 - lam) * m + lam * fast[t]
        out[t] = m
    return out


def run_single(args):
    """(arm, seed_idx, n_units) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    arm, seed_idx, n_units = args
    t0 = time.time()

    use_meta = arm in ('full', 'no_homeostat', 'no_plasticity')
    use_homeo = arm in ('full', 'no_metadata', 'no_plasticity')
    use_plastic = arm in ('full', 'no_metadata', 'no_homeostat')

    tau = gen_tau_vec(n_units, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    mask = random_graph_mask(n_units)
    dt_seq, regime_seq = gen_regime_switch(seed=seed_idx)
    T = dt_seq.shape[0]

    kappa = KAPPA_NOMINAL
    x_cur = x0.copy()
    ip, idx, wt = adjacency_to_csr(mask)

    # running feature stats for standardization (first 30% window)
    fast_all = np.empty((T, n_units))
    for blk_start in range(0, T, 200):
        blk_end = min(blk_start + 200, T)
        if use_homeo and blk_start % FTLE_EVERY == 0:
            w = min(FTLE_WINDOW, T - blk_start)
            lam, _ = run_pair_ftle_nb(x_cur, tau, dt_seq[blk_start:blk_start + w],
                                      PW, ip, idx, wt, kappa,
                                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                      COUPLING_CONTRAST_SELF, 1e-8, 10)
            err = float(np.clip(LAMBDA_TARGET - lam, -1.0, 1.0))
            kappa = float(np.clip(kappa + ETA_LAMBDA * err, KAPPA_MIN, KAPPA_MAX))
        st_b, _, _ = run_trajectory_nb(x_cur, tau, dt_seq[blk_start:blk_end],
                                       PW, ip, idx, wt, kappa,
                                       ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                       COUPLING_CONTRAST_SELF, 0)
        fast_all[blk_start:blk_end] = np.exp(gamma * st_b) / FEATURE_SCALE
        x_cur = st_b[-1].copy()
        if use_plastic and blk_end % PLASTICITY_EVERY == 0 and blk_end < T:
            obs = fast_all[blk_start:blk_end]
            z = (obs - obs.mean(axis=0)) / (obs.std(axis=0) + 1e-12)
            C = (z.T @ z) / obs.shape[0]
            n_grow = max(1, int(PLASTICITY_CHURN * np.triu(mask, 1).sum()))
            mask = evolve_mask(mask, C, n_grow)
            ip, idx, wt = adjacency_to_csr(mask)

    # feature build + standardization (stats from first 30%)
    n_fit = int(0.3 * T)
    mu = fast_all[:n_fit].mean(axis=0)
    sd = fast_all[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    fast_s = (fast_all - mu) / sd
    if use_meta:
        slow = slow_ema(fast_all, TAU_SLOW)
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

    return {'arm': arm, 'seed_idx': seed_idx, 'n_units': n_units,
            'overall_acc': acc, 'steady_acc': float(np.mean(steady)),
            'adapt_time': float(np.nanmean(adapt)),
            'kappa_settled': float(kappa),
            'runtime_s': time.time() - t0}


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['arm'], r['n_units']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        arm, n = key
        entry = {'arm': arm, 'n_units': n, 'n_runs': len(rs)}
        for f in ['overall_acc', 'steady_acc', 'adapt_time', 'kappa_settled']:
            v = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(v))
            entry[f + '_std'] = float(np.nanstd(v))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 110)
    print("S8 RESULTS (mean over seeds): regime_switch task")
    print("=" * 110)
    arms = ['baseline', 'no_plasticity', 'no_homeostat', 'no_metadata', 'full']
    for n in sorted({a['n_units'] for a in agg}):
        print(f"\n--- N={n} ---")
        print(f"  {'arm':<14} | {'overall':>8} | {'steady':>7} | "
              f"{'adapt(pulses)':>13} | {'kappa_end':>9}")
        for arm in arms:
            a = next((x for x in agg if x['arm'] == arm and x['n_units'] == n), None)
            if a is None:
                continue
            print(f"  {a['arm']:<14} | {a['overall_acc_mean']:>8.3f} | "
                  f"{a['steady_acc_mean']:>7.3f} | "
                  f"{a['adapt_time_mean']:>13.0f} | "
                  f"{a['kappa_settled_mean']:>9.1f}")


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S8 integrated benchmark "
          f"(quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16, seed=TOPO_SEED)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_NOMINAL,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    arms = ['baseline', 'no_plasticity', 'no_homeostat', 'no_metadata', 'full']
    n_seeds = 2 if quick else N_SEEDS
    all_args = []
    for arm in arms:
        for s in range(n_seeds):
            all_args.append((arm, s, N_UNITS))
    if not quick:
        # N=1024 scale confirmation (3 seeds; the 2049-dim RLS is the
        # runtime bottleneck ~2min/run, so keep the confirmation lean)
        for arm in ['baseline', 'full']:
            for s in range(3):
                all_args.append((arm, s, 1024))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seeds={n_seeds}, N=256 + N=1024 confirm)")

    results = []
    with Pool(min(cpu_count(), max(1, n_runs))) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=1):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed_idx', 'n_units', 'overall_acc', 'steady_acc',
                  'adapt_time', 'kappa_settled', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'kappa_nominal': KAPPA_NOMINAL,
        'rls_forgetting': RLS_FORGETTING, 'tau_slow': TAU_SLOW,
        'lambda_target': LAMBDA_TARGET, 'eta_lambda': ETA_LAMBDA,
        'ftle_every': FTLE_EVERY, 'ftle_window': FTLE_WINDOW,
        'plasticity_every': PLASTICITY_EVERY,
        'plasticity_churn': PLASTICITY_CHURN,
        'n_seeds': n_seeds, 'arms': arms, 'quick': bool(quick),
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
