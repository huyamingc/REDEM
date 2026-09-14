#!/usr/bin/env python3
"""
Capacity probe: does a single online readout recover to the supervised
ceiling after fast 2D rotation, and when does the gap become structural
(representation-insufficient = L3 trigger)? (REDEM S49)
=============================================================================
Type:           PAPER
Paper Section:  S48 follow-up (L3 capacity probe on the 2D substrate)
Experiment:     s48 showed a single RLS readout tracks 90-deg smooth rotation
                on the coupled substrate to ~the ridge ceiling when the
                rotation is slow (steady 0.90-0.93 vs probe 0.91). This
                script is the L3 capacity probe: sweep the ROTATION SPEED
                (rot_span = blocks over which the boundary turns pi/2) and
                ask whether the online learner's STEADY-STATE accuracy after
                the rotation reaches the supervised ridge ceiling at the
                final angle:

                  gap = ridge_late - rls_steady

                * If rls_steady -> ridge_late at EVERY speed (gap -> 0 in
                  steady), the single-hypothesis limit is TEMPORAL (tracking
                  lag during the rotation, recoverable after): speed stress
                  is absorbed by dynamics, NO L3 trigger -- a population
                  would only help during fast transients, not steady state.
                * If a permanent gap appears at some speed (rls_steady <
                  ridge_late even after the rotation stops and the angle is
                  held), the single linear hypothesis is STRUCTURALLY
                  insufficient at that regime speed: representation-
                  insufficient fires, the L3 (hypothesis generation) trigger
                  is justified, and the correct next move is a population
                  (s50).

Arms (rotation2d, hold theta0=0.6 for [0,1000), rotate pi/2 over
[1000,1000+rot_span), hold final angle to 2000, K=20, T=40000):
  rls_online   : block-level RLS on reconstructed labels (best online
                 learner, s47/s48); scored by block vote.
  delta_base   : error-driven delta, fixed eta (reference: slower optimizer).

Envs (speed sweep): rot_span in {1000, 250, 100, 40} -> angular speed
{0.09, 0.36, 0.9, 2.25} deg/block, each x {parallel, random_graph_k25} x
10 seeds. Ridge probe at the early (theta0) and late (theta0+pi/2) fixed
angles on the random_graph substrate (the decodable one) is computed in
main as the supervised ceiling.

Predictions:
  P1 (temporal vs structural): rls_steady decreases monotonically with speed
      (post-rotation), BUT recovers toward ridge_late in steady; the gap
      after recovery stays small (< 0.05) up to a critical speed, then
      opens. The critical speed where the STEADY gap becomes permanent is
      the L3 capacity boundary.
  P2 (speed cost is transient): the rotation-window accuracy (post_swap)
      drops steeply with speed for both arms -- the cost of speed is paid
      DURING rotation, not in the recovered steady state (if P1 holds).
  P3 (delta tracks slower): delta_base recovers less / more slowly than
      rls_online at every speed -- consistent with s46/s47 (rule + gain
      structure).

Output files:
  data/s49_capacity_probe_v1.csv    (one row per run)
  data/s49_capacity_probe_v1.json   (params + per-cell aggregates + probes)

Usage: python s49_capacity_probe.py [--quick]
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper_e', 'deps'))
from shallow_trap_array_simulator import gamma, tau0, gen_tau_vec, preprogram_vec
from recurrent_substrate import (
    COUPLING_NONE, COUPLING_CONTRAST_SELF,
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    run_trajectory_nb)
from online_readout import OnlineRLS, ridge_fit, running_mean_accuracy
from streaming_tasks import gen_rotation2d, DB_K_PULSES

# ==================== Fixed parameters (S3/S39-S48-consistent) ====================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

DELTA_ETA = 0.05
W_MAX = 20.0

RLS_FORGETTING = 0.99
RLS_INIT_COV = 1.0

ROT_AFTER = 1000
N_BLOCKS = 2000
THETA0 = 0.6
DTHETA = 0.5 * np.pi

# speed sweep: rotation over [1000, 1000+span) blocks
ROT_SPANS = [1000, 250, 100, 40]

# Supervised ridge probe protocol (dedup P0-19: report lambda and the
# train/test split instead of leaving them implicit).
RIDGE_LAMBDA = 1.0
RIDGE_LAMBDA_GRID = [0.01, 1.0, 100.0]
PROBE_SEEDS = 10           # probes are computed for this many seeds

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's49_capacity_probe_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's49_capacity_probe_v1.json')

CURVE_WINDOW = 200

ARMS = ['rls_online', 'delta_base']


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def acc_median_in(acc_run, b_lo, b_hi):
    lo, hi = b_lo * DB_K_PULSES, b_hi * DB_K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _cap(w, w_max):
    nw = float(np.linalg.norm(w))
    if nw > w_max:
        w = w * (w_max / nw)
    return w


def build_features(states, T):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F = np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])
    return F


def ridge_probe(F, target, b_lo, b_hi, ridge_lambda=RIDGE_LAMBDA,
                return_detail=False):
    """Held-out supervised ridge probe on a block window.

    Protocol (now explicit, dedup P0-19): the window [b_lo, b_hi) in blocks is
    split CHRONOLOGICALLY into a first half (ridge + intercept fitted here)
    and a second half (accuracy scored here); ridge_lambda is reported in the
    output JSON along with the sample counts. With return_detail=True the
    dict also carries n_train/n_test/lambda so the ceiling is auditable.
    """
    lo = b_lo * DB_K_PULSES
    hi = b_hi * DB_K_PULSES
    X, y = F[lo:hi], target[lo:hi]
    n_train = X.shape[0] // 2
    out = ridge_fit(X[:n_train], y[:n_train], X[n_train:], y[n_train:],
                    ridge_lambda=ridge_lambda)
    pred = out['pred_te']
    acc = float(np.mean((pred > 0.5).astype(np.float64) == y[n_train:]))
    if return_detail:
        return {'acc': acc, 'n_train': int(n_train),
                'n_test': int(X.shape[0] - n_train),
                'n_features': int(X.shape[1]),
                'ridge_lambda': float(ridge_lambda),
                'window_blocks': [int(b_lo), int(b_hi)]}
    return acc


def probe_lambda_sensitivity(F, target, b_lo, b_hi,
                             lams=RIDGE_LAMBDA_GRID):
    """Probe accuracy across the ridge grid (dedup P0-19).

    The single committed lambda was never reported; this shows the ceiling is
    not an artifact of one regularization choice. Returns {lambda: acc}.
    """
    return {float(l): ridge_probe(F, target, b_lo, b_hi, ridge_lambda=l)
            for l in lams}


# ========================== Single run ==========================

def run_single(args):
    """(topo_name, arm, rot_span, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, rot_span, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq, theta_seq, _, _ = gen_rotation2d(
        seed=seed_idx, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
        theta0=THETA0, dtheta=DTHETA, rot_after=ROT_AFTER, rot_span=rot_span)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F = build_features(states, T)
    d = F.shape[1]

    preds = np.empty(T)

    if arm == 'rls_online':
        rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV)
        for t in range(T):
            preds[t] = float(rls.predict(F[t])[0])
            if (t + 1) % DB_K_PULSES == 0:
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                x_blk = F[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
                rls.update(x_blk, np.array([y_derived]))
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
    else:  # delta_base
        rng_w = np.random.RandomState(seed_idx * 97 + 3)
        w = rng_w.uniform(-0.5, 0.5, d)
        acc_sF = np.zeros(d)
        acc_oF = np.zeros(d)
        for t in range(T):
            o = _sigmoid(float(w @ F[t]))
            preds[t] = o
            acc_sF += F[t]
            acc_oF += F[t] * o
            if (t + 1) % DB_K_PULSES == 0:
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                w += DELTA_ETA * (y_derived * acc_sF - acc_oF)
                w = _cap(w, W_MAX)
                acc_sF[:] = 0.0
                acc_oF[:] = 0.0

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'rotation2d', 'rot_span': rot_span, 'substrate': topo_name,
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000),
           'post_swap_acc': acc_median_in(acc_run, 1000, 1050),
           'steady_acc': acc_median_in(acc_run, 1800, 2000),
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['rot_span'], r['substrate'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        span, topo, read = key
        entry = {'rot_span': span, 'substrate': topo, 'readout': read,
                 'n_runs': len(rs)}
        for f in ['pre_swap_acc', 'post_swap_acc', 'steady_acc',
                  'mean_acc_all']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        agg.append(entry)
    return agg


def print_table(agg, probes):
    print("\n" + "=" * 122)
    print("S49 CAPACITY PROBE (pi/2 rotation over [1000, 1000+span), "
          "steady window at final angle)")
    print("=" * 122)
    print(f"\n  ridge probes (supervised ceiling, random_graph): "
          f"early={probes.get('random_graph_k25_early', float('nan')):.3f}, "
          f"late={probes.get('random_graph_k25_late', float('nan')):.3f}")
    print(f"  {'substrate':<17} | {'rot_span':>7} | {'readout':<10} | "
          f"{'pre':>6} | {'post_rot':>7} | {'steady':>6} | {'mean':>6}")
    for span in sorted(set(a['rot_span'] for a in agg), reverse=True):
        for a in [x for x in agg if x['rot_span'] == span]:
            print(f"  {a['substrate']:<17} | {a['rot_span']:>7} | "
                  f"{a['readout']:<10} | {a['pre_swap_acc_mean']:>6.3f} | "
                  f"{a['post_swap_acc_mean']:>7.3f} | "
                  f"{a['steady_acc_mean']:>6.3f} | "
                  f"{a['mean_acc_all_mean']:>6.3f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S49 capacity probe "
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

    n_seeds = N_SEEDS if not quick else 2
    all_args = []
    for span in ROT_SPANS:
        for topo in ['parallel', 'random_graph_k25']:
            for arm in ARMS:
                for s in range(n_seeds):
                    all_args.append((topo, arm, span, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (spans={len(ROT_SPANS)}, arms={len(ARMS)}, "
          f"seeds={n_seeds})")

    results = []
    n_proc = min(cpu_count(), 8, max(1, n_runs))
    with Pool(n_proc) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=2):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    # supervised ridge probes on the decodable (coupled) substrate.
    # Reworked 2026-09-09 (dedup P0-19): probes are computed for PROBE_SEEDS
    # seeds (mean/sd/CI instead of a single seed-0 value), the ridge lambda and
    # the chronological held-out split are recorded, and a lambda grid is swept
    # so the ceiling cannot be a regularization artifact.
    probes = {}
    probe_detail = {}
    tau_p = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=0)
    x0_p = preprogram_vec(ALPHA0, tau_p)
    ip_p, idx_p, wt_p, mode_p, kap_p = build_substrate('random_graph_k25')
    for span in ROT_SPANS:
        accs = {'early': [], 'late': []}
        for ps in range(PROBE_SEEDS):
            tau_s = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=ps)
            x0_s = preprogram_vec(ALPHA0, tau_s)
            dt_s, tgt_s, _, _, _ = gen_rotation2d(
                seed=ps, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
                theta0=THETA0, dtheta=DTHETA, rot_after=ROT_AFTER,
                rot_span=span)
            st_s, _, _ = run_trajectory_nb(
                x0_s, tau_s, dt_s, PW, ip_p, idx_p, wt_p, kap_p,
                ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode_p, 0)
            F_s = build_features(st_s, dt_s.shape[0])
            for which, (b_lo, b_hi) in (('early', (800, 1000)),
                                        ('late', (1800, 2000))):
                d_s = ridge_probe(F_s, tgt_s, b_lo, b_hi, return_detail=True)
                accs[which].append(d_s['acc'])
                if ps == 0 and span == ROT_SPANS[0]:
                    detail = dict(d_s)
                    detail['lambda_grid'] = probe_lambda_sensitivity(
                        F_s, tgt_s, b_lo, b_hi)
                    probe_detail[f'span{span}_{which}'] = detail
        for which in ('early', 'late'):
            a = np.array(accs[which], dtype=float)
            key = f'span{span}_{which}'
            probes[key] = float(np.mean(a))
            probes[key + '_sd'] = float(np.std(a, ddof=1))
            se = float(np.std(a, ddof=1) / np.sqrt(a.size))
            probes[key + '_ci95'] = [float(np.mean(a) - 2.262 * se),
                                     float(np.mean(a) + 2.262 * se)]
            probes[key + '_per_seed'] = [float(x) for x in a]
    for k, v in probes.items():
        if isinstance(v, float):
            print(f"  ridge probe {k}: {v:.3f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'rot_span', 'substrate', 'readout', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'mean_acc_all',
                  'pre_swap_acc', 'post_swap_acc', 'steady_acc']
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
        'delta_eta': DELTA_ETA, 'w_max': W_MAX,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'rot_after': ROT_AFTER, 'n_blocks': N_BLOCKS, 'theta0': THETA0,
        'dtheta': float(DTHETA), 'rot_spans': ROT_SPANS,
        'probe_protocol': {
            'features': '257-dim pulse-level features (build_features)',
            'split': 'chronological half/half within the window '
                     '(first half fit, second half scored)',
            'ridge_lambda': RIDGE_LAMBDA,
            'ridge_lambda_grid': RIDGE_LAMBDA_GRID,
            'probe_seeds': PROBE_SEEDS,
            'ci': 'two-sided 95% Student-t over seeds (t=2.262, n=10)',
            'windows_blocks': {'early': [800, 1000], 'late': [1800, 2000]},
        },
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg, 'ridge_probe': probes,
                   'ridge_probe_detail': probe_detail}, f, indent=2)

    print_table(agg, probes)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
