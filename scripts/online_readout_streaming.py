#!/usr/bin/env python3
"""
Online readout streaming benchmark: RLS vs offline ridge (REDEM S2).
=============================================================================
Type:           PAPER
Paper Section:  New-algorithm project Step S2
Experiment:     Online-vs-offline readout comparison on streaming tasks,
                over the S1-characterized substrate operating points.

Protocol (fixed seeds, paired design):
  * Substrate configs:
      parallel        : uncoupled baseline (kappa=0)
      random_graph_k25: mode-1 contrast coupling, kappa=25 (S1 held-out
                        MC peak, just before the chaos transition)
  * Tasks (streaming_tasks.py):
      drift_binary   : two-class interval blocks, continuous interval walk
                       + abrupt class swaps every 1000 blocks (40k pulses)
      narma10        : stationary memory benchmark (21k pulses)
      mackey_glass   : chaotic forecasting (21k pulses)
  * Readouts:
      RLS    : OnlineRLS with forgetting in {0.99, 0.999}
      OFFRIDGE: offline ridge fit on the first 30% of the stream and frozen
               (static baseline that cannot adapt to drift)
  * Features: current-ratio observables i_t = exp(gamma * x_t) / 10 plus a
    bias column (same observable as the S1 memory-capacity fits).
  * 10 seeds per cell; tau/substrate draws paired across readout configs.
  * Metrics:
      drift_binary : pre-swap steady accuracy, post-swap recovered accuracy,
                     adaptation time (pulses to recover after swap 1),
                     overall mean accuracy, pre-swap stability (std)
      narma/mg     : NMSE on the last 30% of targets (held-out segment;
                     for OFFRIDGE fit on first 30% / eval on last 30%)

Output files:
  data/s2_online_readout_v1.csv    (one row per run)
  data/s2_online_readout_v1.json   (params + per-cell aggregates)
  data/s2_online_readout_v1_curves.npz (mean running curves, subsampled)

Usage: python online_readout_streaming.py [--quick]
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
from online_readout import OnlineRLS, ridge_fit, running_mean_accuracy, running_mean_mse
from streaming_tasks import (gen_drift_binary, gen_narma10, gen_mackey_glass,
                             DB_K_PULSES)

# ========================== Fixed parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0          # divide current ratios for readout conditioning
BIAS = 1.0
RIDGE_LAMBDA = 1.0
RLS_INIT_COV = 100.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

KAPPA_RANDOM = 25.0           # S1 held-out MC peak operating point

# Readout specs per task family
# NOTE: forgetting < 0.999 diverges on long streams for the near-edge
# substrate (slow long-horizon RLS instability: P accumulates energy in
# near-singular eigendirections over 1/(1-lambda) horizons). Validated:
# lambda=0.999 stable over 21k pulses on all tested seeds; lambda=0.995
# diverges by 21k pulses (see S2 gate notes). Adaptive forgetting is
# deferred to S5 (metadata/dual-timescale mechanisms).
RLS_FORGETTINGS = [0.999]
SUBS = {
    'drift_binary': ('parallel', 'random_graph_k25'),
    'narma10': ('parallel', 'random_graph_k25'),
    'mackey_glass': ('parallel', 'random_graph_k25'),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's2_online_readout_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's2_online_readout_v1.json')
NPZ_PATH = os.path.join(DATA_DIR, 's2_online_readout_v1_curves.npz')

CURVE_WINDOW = 200
CURVE_SUBSAMPLE = 100


def build_substrate(topo_name, n_units):
    """Return (indptr, indices, wts, mode, kappa) for a substrate config."""
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', n_units)
        return ip, idx, wt, COUPLING_NONE, 0.0
    if topo_name == 'random_graph_k25':
        ip, idx, wt = build_topology_csr('random_graph', n_units,
                                         seed=TOPO_SEED, avg_degree=AVG_DEGREE)
        return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM
    raise ValueError(topo_name)


def extract_features(states):
    """Legacy helper kept for reference; feature extraction is now inline in
    run_single with fixed standardization from the first 30% of the stream.
    """
    obs = np.exp(gamma * states) / FEATURE_SCALE
    return np.hstack([obs, np.full((states.shape[0], 1), BIAS)])


# ========================== Single run ==========================

def run_single(args):
    """One (task, substrate, readout, seed) run. args tuple documented below."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    (task, topo_name, readout, forgetting, seed_idx) = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name, N_UNITS)

    # ---- generate task stream (paired across configs via seed_idx) ----
    if task == 'drift_binary':
        dt_seq, target_seq, swap_blocks = gen_drift_binary(seed=seed_idx)
        T = dt_seq.shape[0]
        target = target_seq.astype(np.float64)
    elif task == 'narma10':
        dt_seq, target_seq = gen_narma10(seed=seed_idx)
        T = dt_seq.shape[0]
        target = target_seq
    elif task == 'mackey_glass':
        dt_seq, target_seq = gen_mackey_glass(seed=seed_idx)
        T = dt_seq.shape[0]
        target = target_seq
    else:
        raise ValueError(task)

    # ---- substrate trajectory ----
    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)

    # ---- fixed feature standardization from the first 30% only ----
    # (the same information window the offline baseline fits on; the
    # standardization stats are never taken from the held-out segment)
    n_fit = int(0.3 * T)
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    obs = (obs_raw - mu) / sd
    F = np.hstack([obs, np.full((T, 1), BIAS)])

    # ---- readout ----
    res = {}
    if readout == 'rls':
        rls = OnlineRLS(F.shape[1], 1, forgetting=forgetting,
                        init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                        reg=RLS_REG)
        errs, preds = rls.fit_stream(F, target[:, None], n_warmup=200)
        pred = preds[:, 0]
        res['forgetting'] = float(forgetting)
    elif readout == 'offridge':
        fit = ridge_fit(F[:n_fit], target[:n_fit, None],
                        Xte=F[n_fit:], Yte=target[n_fit:, None],
                        ridge_lambda=RIDGE_LAMBDA)
        pred = np.full(T, np.nan)
        pred[:n_fit] = fit['pred_tr'][:, 0]
        pred[n_fit:] = fit['pred_te'][:, 0]
        res['forgetting'] = np.nan
    else:
        raise ValueError(readout)

    # ---- metrics ----
    if task == 'drift_binary':
        acc_run = running_mean_accuracy(pred, target, CURVE_WINDOW)
        swap1_pulse = int(swap_blocks[0] * DB_K_PULSES) if swap_blocks else None
        if swap1_pulse is None:
            raise RuntimeError('drift stream produced no swaps')
        # pre-swap steady: last 2000 pulses before swap1
        pre = acc_run[swap1_pulse - 2000:swap1_pulse]
        pre_steady = float(np.nanmedian(pre))
        pre_std = float(np.nanstd(pre))
        # post-swap recovered: pulses [swap1+4000, swap1+6000]
        post = acc_run[swap1_pulse + 4000:swap1_pulse + 6000]
        post_acc = float(np.nanmedian(post))
        # adaptation time: first pulse AFTER the swap where the running
        # accuracy (window entirely post-swap) crosses the threshold
        thr = pre_steady - 0.02
        seg = acc_run[swap1_pulse + CURVE_WINDOW:]
        hits = np.where(np.isfinite(seg) & (seg >= thr))[0]
        adapt = float(hits[0]) if hits.size else np.nan
        res.update({
            'pre_steady_acc': pre_steady, 'pre_stability_std': pre_std,
            'post_swap_acc': post_acc, 'adapt_time_pulses': adapt,
            'mean_acc_all': float(np.nanmean(acc_run)),
        })
        curve = acc_run
        curve_key = 'acc'
    else:
        # held-out segment = last 30%
        n_eval = int(0.3 * T)
        y_eval = target[-n_eval:]
        p_eval = pred[-n_eval:]
        var_eval = float(y_eval.var())
        mse_eval = float(np.nanmean((p_eval - y_eval) ** 2))
        res.update({'nmse_final30': mse_eval / var_eval if var_eval > 0 else np.nan})
        curve = running_mean_mse(pred, target, CURVE_WINDOW)
        curve_key = 'mse'

    # subsampled curve for aggregate storage
    idxs = np.arange(0, T, CURVE_SUBSAMPLE)
    res['curve_x'] = idxs.astype(int)
    res['curve_y'] = curve[idxs]
    res['curve_key'] = curve_key
    res['task'] = task
    res['substrate'] = topo_name
    res['readout'] = readout
    res['seed_idx'] = seed_idx
    res['n_units'] = N_UNITS
    res['t_total'] = int(T)
    res['w_norm'] = float(np.linalg.norm(rls.W)) if readout == 'rls' else np.nan
    res['runtime_s'] = time.time() - t0
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    """Per (task, substrate, readout) aggregates of scalar metrics."""
    groups = {}
    for r in results:
        key = (r['task'], r['substrate'], r['readout'])
        groups.setdefault(key, []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        task, topo, read = key
        fields = (['pre_steady_acc', 'post_swap_acc', 'adapt_time_pulses',
                   'mean_acc_all', 'pre_stability_std']
                  if task == 'drift_binary' else ['nmse_final30'])
        entry = {'task': task, 'substrate': topo, 'readout': read,
                 'n_runs': len(rs)}
        for f in fields:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        if read == 'rls':
            entry['forgetting'] = float(rs[0]['forgetting'])
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 100)
    print("S2 RESULTS (mean over seeds)")
    print("=" * 100)
    for task in ['drift_binary', 'narma10', 'mackey_glass']:
        rows = [a for a in agg if a['task'] == task]
        print(f"\n--- {task} ---")
        if task == 'drift_binary':
            print(f"  {'substrate':<18} {'readout':<10} | {'pre_steady':>10} | "
                  f"{'post_swap':>9} | {'adapt(pulses)':>13} | {'mean_all':>8} | "
                  f"{'stab_std':>8}")
            for a in rows:
                print(f"  {a['substrate']:<18} {a['readout']:<10} | "
                      f"{a['pre_steady_acc_mean']:>10.3f} | "
                      f"{a['post_swap_acc_mean']:>9.3f} | "
                      f"{a['adapt_time_pulses_mean']:>13.0f} | "
                      f"{a['mean_acc_all_mean']:>8.3f} | "
                      f"{a['pre_stability_std_mean']:>8.4f}")
        else:
            print(f"  {'substrate':<18} {'readout':<10} | {'NMSE_final30':>12}")
            for a in rows:
                print(f"  {a['substrate']:<18} {a['readout']:<10} | "
                      f"{a['nmse_final30_mean']:>12.4f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S2 online readout benchmark "
          f"(quick={quick})")

    # numba warmup (also populates the disk cache for spawned workers)
    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16,
                                           seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_RANDOM,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    tasks = ['drift_binary', 'narma10', 'mackey_glass']
    n_seeds = N_SEEDS
    if quick:
        n_seeds = 2
        tasks = ['drift_binary', 'narma10']

    all_args = []
    for task in tasks:
        for topo in SUBS[task]:
            for fg in RLS_FORGETTINGS:
                for s in range(n_seeds):
                    all_args.append((task, topo, 'rls', fg, s))
            for s in range(n_seeds):
                all_args.append((task, topo, 'offridge', np.nan, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (tasks={tasks}, seeds={n_seeds})")

    results = []
    with Pool(min(cpu_count(), max(1, n_runs))) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=2):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    # ---- CSV ----
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'substrate', 'readout', 'forgetting', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'w_norm',
                  'pre_steady_acc', 'pre_stability_std', 'post_swap_acc',
                  'adapt_time_pulses', 'mean_acc_all', 'nmse_final30']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    # ---- JSON aggregates ----
    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'topo_seed': TOPO_SEED, 'avg_degree': AVG_DEGREE,
        'kappa_random': KAPPA_RANDOM,
        'feature_scale': FEATURE_SCALE, 'bias': BIAS,
        'ridge_lambda': RIDGE_LAMBDA,
        'rls_forgettings': RLS_FORGETTINGS,
        'rls_init_cov': RLS_INIT_COV, 'rls_trace_cap': RLS_TRACE_CAP,
        'rls_reg': RLS_REG,
        'n_seeds': n_seeds,
        'curve_window': CURVE_WINDOW, 'curve_subsample': CURVE_SUBSAMPLE,
        'offline_fit_frac': 0.3, 'tasks': tasks,
        'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    # ---- curves npz (mean over seeds per cell) ----
    out_npz = NPZ_PATH if not quick else NPZ_PATH.replace('.npz', '_quick.npz')
    groups = {}
    for r in results:
        key = (r['task'], r['substrate'], r['readout'])
        groups.setdefault(key, []).append(r)
    save = {}
    for key, rs in groups.items():
        tag = '_'.join(key)
        x = rs[0]['curve_x']
        y = np.array([r['curve_y'] for r in rs], dtype=float)
        save[f'{tag}_x'] = x
        save[f'{tag}_y'] = y.mean(axis=0)
        save[f'{tag}_sem'] = y.std(axis=0) / np.sqrt(len(rs))
    np.savez_compressed(out_npz, **save)

    print_table(agg)
    print(f"\nCSV  : {out_csv}")
    print(f"JSON : {out_json}")
    print(f"NPZ  : {out_npz}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
