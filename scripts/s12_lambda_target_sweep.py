#!/usr/bin/env python3
"""
Lambda-target sweep for the chaos homeostat (REDEM S12).
=============================================================================
Type:           PAPER
Experiment:     Sweep the homeostat target lambda_target across
                {-0.05, -0.02, -0.005, 0.0} x CV {0.1, 0.2, 0.4} to
                determine the optimal criticality target and its
                robustness to the trap spectrum width. Single
                tau_drift disturbance at t=10k; Mackey-Glass online
                task; 5 seeds per cell.

Arms:
  fixed      : kappa=25 throughout (no homeostat), per CV
  regulated  : homeostat with the given lambda_target, per CV

Metrics:
  mc_heldout   : post-disturbance held-out MC at settled kappa (probe)
  kappa_settled: mean kappa over last 4000 pulses
  pre_nmse     : MG NMSE before disturbance
  post_nmse    : MG NMSE after disturbance (3k-5k post)

Output files:
  data/s12_lambda_target_sweep_v1.csv
  data/s12_lambda_target_sweep_v1.json

Usage: python s12_lambda_target_sweep.py [--quick]
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
    run_trajectory_nb, run_pair_ftle_nb)
from online_readout import OnlineRLS, memory_capacity_heldout
from streaming_tasks import gen_mackey_glass

# ========================== Fixed parameters ==========================
N_UNITS = 256
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 5
FEATURE_SCALE = 10.0

RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

KAPPA_NOMINAL = 25.0
KAPPA_MIN, KAPPA_MAX = 1.0, 60.0
ETA_LAMBDA = 3.0
FTLE_EVERY = 1000
FTLE_WINDOW = 400
HOMEO_BLOCK = 200

TAU_DRIFT = 1.5

LAMBDA_TARGETS = [-0.05, -0.02, -0.005, 0.0]
CV_VALUES = [0.1, 0.2, 0.4]

T_TOTAL = 21000
T_DISTURB = 10000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's12_lambda_target_sweep_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's12_lambda_target_sweep_v1.json')


def build_csr(n_units=N_UNITS):
    return build_topology_csr('random_graph', n_units,
                             seed=TOPO_SEED, avg_degree=AVG_DEGREE)


def run_single(args):
    """(lambda_target, cv_tau, arm, seed_idx) -> dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    lam_target, cv_tau, arm, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, cv_tau, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    ip0, idx0, wt0 = build_csr()
    dt_seq, target_seq = gen_mackey_glass(seed=seed_idx)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    tau_d = tau * TAU_DRIFT
    ip_d, idx_d, wt_d = ip0, idx0, wt0

    rls = OnlineRLS(N_UNITS + 1, 1, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP, reg=RLS_REG)
    preds = np.empty(T)
    kappa = KAPPA_NOMINAL
    kappa_hist = np.empty(T)
    kappa_hist[:] = np.nan

    rng_noise = np.random.RandomState(seed_idx * 131 + 3)

    def feed_features(states_seg, rng, start):
        obs = np.exp(gamma * states_seg) / FEATURE_SCALE
        F = np.hstack([obs, np.ones((obs.shape[0], 1))])
        seg_pred = np.empty(obs.shape[0])
        for j in range(obs.shape[0]):
            seg_pred[j] = rls.predict(F[j])[0]
            rls.update(F[j], target[start + j])
        return seg_pred

    if arm == 'fixed':
        st1, _, _ = run_trajectory_nb(x0, tau, dt_seq[:T_DISTURB], PW,
                                      ip0, idx0, wt0, KAPPA_NOMINAL,
                                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                      COUPLING_CONTRAST_SELF, 0)
        p1 = feed_features(st1, rng_noise, 0)
        preds[:T_DISTURB] = p1
        st2, _, _ = run_trajectory_nb(st1[-1], tau_d, dt_seq[T_DISTURB:], PW,
                                      ip_d, idx_d, wt_d, KAPPA_NOMINAL,
                                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                      COUPLING_CONTRAST_SELF, 0)
        p2 = feed_features(st2, rng_noise, T_DISTURB)
        preds[T_DISTURB:] = p2
        kappa_hist[:] = KAPPA_NOMINAL
    else:
        x_cur = x0.copy()
        tau_cur = tau
        ip_c, idx_c, wt_c = ip0, idx0, wt0
        disturbed = False
        for blk_start in range(0, T, HOMEO_BLOCK):
            blk_end = min(blk_start + HOMEO_BLOCK, T)
            if blk_start >= T_DISTURB and not disturbed:
                tau_cur = tau * TAU_DRIFT
                disturbed = True
            if blk_start % FTLE_EVERY == 0:
                w = min(FTLE_WINDOW, T - blk_start)
                lam, _ = run_pair_ftle_nb(
                    x_cur, tau_cur, dt_seq[blk_start:blk_start + w], PW,
                    ip_c, idx_c, wt_c, kappa,
                    ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                    COUPLING_CONTRAST_SELF, 1e-8, 10)
                err = float(np.clip(lam_target - lam, -1.0, 1.0))
                kappa = float(np.clip(kappa + ETA_LAMBDA * err,
                                     KAPPA_MIN, KAPPA_MAX))
            st_b, _, _ = run_trajectory_nb(
                x_cur, tau_cur, dt_seq[blk_start:blk_end], PW,
                ip_c, idx_c, wt_c, kappa,
                ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                COUPLING_CONTRAST_SELF, 0)
            p_b = feed_features(st_b, rng_noise, blk_start)
            preds[blk_start:blk_end] = p_b
            kappa_hist[blk_start:blk_end] = kappa
            x_cur = st_b[-1].copy()

    var_full = float(target[T_DISTURB:].var())
    def nmse(seg_p, seg_t):
        return float(np.mean((seg_p - seg_t) ** 2)) / max(var_full, 1e-12)
    pre_nmse = nmse(preds[T_DISTURB - 2000:T_DISTURB],
                    target[T_DISTURB - 2000:T_DISTURB])
    post_nmse = nmse(preds[T_DISTURB + 3000:T_DISTURB + 5000],
                     target[T_DISTURB + 3000:T_DISTURB + 5000])
    kappa_settled = float(np.nanmean(kappa_hist[-4000:]))

    # MC probe at settled kappa on disturbed physics
    x0p = preprogram_vec(ALPHA0, tau_d)
    rng2 = np.random.RandomState(seed_idx * 999 + 5)
    dt_probe = rng2.uniform(2e-6, 20e-6, 3000)
    kp_probe = KAPPA_NOMINAL if arm == 'fixed' else kappa_settled
    st_p, _, _ = run_trajectory_nb(x0p, tau_d, dt_probe, PW,
                                   ip_d, idx_d, wt_d, kp_probe,
                                   ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                   COUPLING_CONTRAST_SELF, 500)
    obs_p = np.exp(gamma * st_p) / FEATURE_SCALE
    mc = memory_capacity_heldout(obs_p, dt_probe[500:])

    return {'lambda_target': float(lam_target), 'cv_tau': float(cv_tau),
            'arm': arm, 'seed_idx': seed_idx,
            'pre_nmse': pre_nmse, 'post_nmse': post_nmse,
            'kappa_settled': kappa_settled, 'mc_heldout': mc,
            'runtime_s': time.time() - t0}


def aggregate(results):
    groups = {}
    for r in results:
        key = (r['lambda_target'], r['cv_tau'], r['arm'])
        groups.setdefault(key, []).append(r)
    agg = []
    for (lam, cv, arm), rs in sorted(groups.items()):
        entry = {'lambda_target': lam, 'cv_tau': cv, 'arm': arm,
                 'n_runs': len(rs)}
        for f in ['pre_nmse', 'post_nmse', 'kappa_settled', 'mc_heldout']:
            v = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.mean(v))
            entry[f + '_std'] = float(np.std(v))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 100)
    print("S12 RESULTS (mean over seeds): lambda_target sweep")
    print("=" * 100)
    print(f" {'lam_t':>7} {'CV':>5} {'arm':>10} | {'MC':>7} {'kap_set':>8} "
          f"{'pre_nmse':>9} {'post_nmse':>10}")
    for a in agg:
        print(f" {a['lambda_target']:>7.3f} {a['cv_tau']:>5.2f} {a['arm']:>10} | "
              f"{a['mc_heldout_mean']:>7.2f} {a['kappa_settled_mean']:>8.1f} "
              f"{a['pre_nmse_mean']:>9.4f} {a['post_nmse_mean']:>10.4f}")


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S12 lambda-target sweep "
          f"(quick={quick})")

    # numba warmup
    tau_w = gen_tau_vec(16, 0.2, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_csr(16)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_NOMINAL,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = 2 if quick else N_SEEDS
    all_args = []
    for lam in LAMBDA_TARGETS:
        for cv in CV_VALUES:
            for arm in ['fixed', 'regulated']:
                for s in range(n_seeds):
                    all_args.append((lam, cv, arm, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seeds={n_seeds}, "
          f"grid={len(LAMBDA_TARGETS)}x{len(CV_VALUES)}x2)")

    sequential = '--sequential' in sys.argv
    results = []
    if sequential:
        for i, args in enumerate(all_args):
            res = run_single(args)
            results.append(res)
            done = i + 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(6, max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 10) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['lambda_target', 'cv_tau', 'arm', 'seed_idx',
                  'pre_nmse', 'post_nmse', 'kappa_settled', 'mc_heldout',
                  'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'alpha0': ALPHA0, 'gamma': float(gamma),
        'tau0': float(tau0), 'kappa_nominal': KAPPA_NOMINAL,
        'kappa_range': [KAPPA_MIN, KAPPA_MAX], 'eta_lambda': ETA_LAMBDA,
        'ftle_every': FTLE_EVERY, 'ftle_window': FTLE_WINDOW,
        'homeo_block': HOMEO_BLOCK, 'tau_drift': TAU_DRIFT,
        'lambda_targets': LAMBDA_TARGETS, 'cv_values': CV_VALUES,
        't_total': T_TOTAL, 't_disturb': T_DISTURB,
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
