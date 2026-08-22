#!/usr/bin/env python3
"""
Three-factor learning benchmark (M1 eligibility + M2 reward gating, REDEM S3).
=============================================================================
Type:           PAPER
Paper Section:  New-algorithm project Step S3 (see NEW_ALGORITHM_PLAN.md)
Experiment:     Reward-gated vs error-gated vs second-order readouts on the
                drift-binary task (sparse block-end reward/supervision) and
                regression verification on NARMA-10 / Mackey-Glass.

Readout arms (all online; drift task uses block-end information only):
  rmhl      : reward-modulated Hebbian (ThreeFactorReadout 'reward' mode):
              eligibility x*o accumulates during a block; at block end the
              +/-1 correctness reward gates consolidation. NO class label.
  lms       : supervised delta rule with input-trace eligibility
              (ThreeFactorReadout 'error' mode, elig_decay=0 = plain LMS).
              Dense per-pulse class targets (drift) / regression targets.
  rls_sparse: RLS updated ONLY at block ends with (mean block feature,
              true block class) -- sparse supervision baseline.
  rls_dense : RLS with per-pulse supervision (S2-style reference).

Expected findings (pre-validated on seed 0):
  * rmhl learns the initial mapping but CANNOT recover from the class-
    interval inversion after the swap (reward-only credit assignment has no
    error/class information). Documented negative result.
  * lms (eta ~ 1e-4) tracks the drift but slower/weaker than RLS; larger
    eta diverges (first-order fragility on the weakly separated features).
  * rls_sparse / rls_dense recover robustly (second-order wins).

Output files:
  data/s3_three_factor_v1.csv    (one row per run)
  data/s3_three_factor_v1.json   (params + per-cell aggregates)

Usage: python three_factor_online_readout.py [--quick]
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
from online_readout import OnlineRLS, ThreeFactorReadout, running_mean_accuracy
from streaming_tasks import gen_drift_binary, gen_narma10, gen_mackey_glass, DB_K_PULSES

# ========================== Fixed parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

# Readout hyperparameters (pre-tuned on seed 0; see S3 gate notes)
RMHL_ETA = 0.05
RMHL_ELIG_DECAY = 0.9
LMS_ETA = 1e-4          # larger eta diverges on the weakly separated features
RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's3_three_factor_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's3_three_factor_v1.json')

CURVE_WINDOW = 200

# (task, readout_list): readout arms per task
READOUTS = {
    'drift_binary': ['rmhl', 'lms', 'rls_sparse', 'rls_dense'],
    'narma10': ['lms', 'rls_dense'],
    'mackey_glass': ['lms', 'rls_dense'],
}


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def drift_metrics(preds, target, swap_pulse, window=CURVE_WINDOW):
    """(pre_steady, post_swap, mean_acc, running_acc) for drift streams."""
    acc_run = running_mean_accuracy(preds, target, window)
    pre = float(np.nanmedian(acc_run[swap_pulse - 2000:swap_pulse]))
    post = float(np.nanmedian(acc_run[swap_pulse + 4000:swap_pulse + 6000]))
    mean = float(np.nanmean(acc_run))
    return pre, post, mean, acc_run


# ========================== Single run ==========================

def run_single(args):
    """(task, topo_name, readout, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    task, topo_name, readout, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    if task == 'drift_binary':
        dt_seq, target_seq, swap_blocks = gen_drift_binary(seed=seed_idx)
    elif task == 'narma10':
        dt_seq, target_seq = gen_narma10(seed=seed_idx)
    elif task == 'mackey_glass':
        dt_seq, target_seq = gen_mackey_glass(seed=seed_idx)
    else:
        raise ValueError(task)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F = np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])

    res = {'task': task, 'substrate': topo_name, 'readout': readout,
           'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T)}

    if readout == 'rmhl':
        # sparse reward-modulated Hebbian: eligibility over each block,
        # +/-1 correctness reward at block end (no class label used)
        tf = ThreeFactorReadout(F.shape[1], mode='reward',
                                learning_rate=RMHL_ETA,
                                elig_decay=RMHL_ELIG_DECAY, seed=seed_idx)
        preds = np.empty(T)
        bi = 0
        for t in range(T):
            preds[t] = tf.predict(F[t])
            tf.e = tf.elig_decay * tf.e + F[t] * preds[t]
            if (t + 1) % DB_K_PULSES == 0:
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                rw = 1.0 if vote == target[t] else -1.0
                tf.consolidate(rw, reset=True)
                bi += 1
    elif readout == 'lms':
        tf = ThreeFactorReadout(F.shape[1], mode='error',
                                learning_rate=LMS_ETA, elig_decay=0.0,
                                seed=seed_idx)
        preds, _ = tf.fit_stream(F, mode='dense', targets=target, n_warmup=200)
    elif readout == 'rls_sparse':
        rls = OnlineRLS(F.shape[1], 1, forgetting=RLS_FORGETTING,
                        init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                        reg=RLS_REG)
        preds = np.empty(T)
        feat_acc = np.zeros(F.shape[1])
        bi = 0
        for t in range(T):
            preds[t] = rls.predict(F[t])[0]
            feat_acc += F[t]
            if (t + 1) % DB_K_PULSES == 0:
                rls.update(feat_acc / DB_K_PULSES, target[t])
                feat_acc[:] = 0.0
                bi += 1
    elif readout == 'rls_dense':
        rls = OnlineRLS(F.shape[1], 1, forgetting=RLS_FORGETTING,
                        init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                        reg=RLS_REG)
        _, preds = rls.fit_stream(F, target[:, None], n_warmup=200)
        preds = preds[:, 0]
    else:
        raise ValueError(readout)

    if task == 'drift_binary':
        swap_pulse = int(swap_blocks[0] * DB_K_PULSES)
        pre, post, mean, acc_run = drift_metrics(preds, target, swap_pulse)
        res.update({'pre_steady_acc': pre, 'post_swap_acc': post,
                    'mean_acc_all': mean})
        idxs = np.arange(0, T, 100)
        res['curve_x'] = idxs.astype(int)
        res['curve_y'] = acc_run[idxs]
    else:
        n_eval = int(0.3 * T)
        y_eval = target[-n_eval:]
        p_eval = preds[-n_eval:]
        var_eval = float(y_eval.var())
        mse_eval = float(np.nanmean((p_eval - y_eval) ** 2))
        res['nmse_final30'] = mse_eval / var_eval if var_eval > 0 else np.nan
    res['runtime_s'] = time.time() - t0
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['task'], r['substrate'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        task, topo, read = key
        entry = {'task': task, 'substrate': topo, 'readout': read,
                 'n_runs': len(rs)}
        fields = (['pre_steady_acc', 'post_swap_acc', 'mean_acc_all']
                  if task == 'drift_binary' else ['nmse_final30'])
        for f in fields:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 100)
    print("S3 RESULTS (mean over seeds)")
    print("=" * 100)
    for task in ['drift_binary', 'narma10', 'mackey_glass']:
        rows = [a for a in agg if a['task'] == task]
        print(f"\n--- {task} ---")
        if task == 'drift_binary':
            print(f"  {'substrate':<18} {'readout':<11} | {'pre_steady':>10} | "
                  f"{'post_swap':>9} | {'mean_all':>8}")
            for a in rows:
                print(f"  {a['substrate']:<18} {a['readout']:<11} | "
                      f"{a['pre_steady_acc_mean']:>10.3f} | "
                      f"{a['post_swap_acc_mean']:>9.3f} | "
                      f"{a['mean_acc_all_mean']:>8.3f}")
        else:
            print(f"  {'substrate':<18} {'readout':<11} | {'NMSE_final30':>12}")
            for a in rows:
                print(f"  {a['substrate']:<18} {a['readout']:<11} | "
                      f"{a['nmse_final30_mean']:>12.4f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S3 three-factor benchmark "
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

    tasks = list(READOUTS.keys())
    n_seeds = N_SEEDS
    if quick:
        n_seeds = 2
        tasks = ['drift_binary']
    all_args = []
    for task in tasks:
        for topo in ['parallel', 'random_graph_k25']:
            for read in READOUTS[task]:
                for s in range(n_seeds):
                    all_args.append((task, topo, read, s))
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

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'substrate', 'readout', 'seed_idx', 'n_units',
                  't_total', 'runtime_s', 'pre_steady_acc', 'post_swap_acc',
                  'mean_acc_all', 'nmse_final30']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'topo_seed': TOPO_SEED, 'avg_degree': AVG_DEGREE,
        'kappa_random': KAPPA_RANDOM,
        'rmhl_eta': RMHL_ETA, 'rmhl_elig_decay': RMHL_ELIG_DECAY,
        'lms_eta': LMS_ETA,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'rls_reg': RLS_REG,
        'n_seeds': n_seeds, 'tasks': tasks, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
