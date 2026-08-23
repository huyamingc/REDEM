#!/usr/bin/env python3
"""
Causal leak audit for REDEM adaptive mechanisms (S13).
=============================================================================
Type:           PAPER
Experiment:     Systematically verify that all adaptive mechanisms
                (RLS readout, metadata EMA, homeostat FTLE, structure
                plasticity) use only past information. Inject a small
                amount of FUTURE information into each mechanism; if
                performance improves, the mechanism has a causal leak.
                Also tests a causal-split protocol for M4 plasticity.

Part 1 - Future injection tests (regime_switch task, 9k pulses):
  normal            : full system, no injection (baseline)
  leak_rls          : RLS update at t mixes 1% of y[t+50] into y[t]
  leak_metadata     : metadata EMA at t mixes 1% of f[t+50] into f[t]
  leak_ftle         : homeostat lambda at t mixes 1% of lambda[t+400]
  leak_plasticity   : C_ij uses 10% of block-end (future) correlation
  no_plasticity     : M4 off (lower bound for plasticity comparison)

Part 2 - Plasticity causal split:
  causal_split      : M4 uses first 50% of block for C_ij, second 50%
                      to apply/validate the rewiring (vs normal: full
                      block for C_ij, apply next block)

If leak arms show NO improvement over normal -> causally clean.
If causal_split accuracy <= normal by < 1pp -> acceptable protocol.

Output files:
  data/s13_causal_audit_v1.csv
  data/s13_causal_audit_v1.json

Usage: python s13_causal_audit.py [--quick]
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
N_SEEDS = 3
FEATURE_SCALE = 10.0

RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

KAPPA_NOMINAL = 25.0
KAPPA_MIN, KAPPA_MAX = 1.0, 60.0
LAMBDA_TARGET = -0.02
ETA_LAMBDA = 3.0
FTLE_EVERY = 1000
FTLE_WINDOW = 400
HOMEO_BLOCK = 200

TAU_SLOW = 500.0
PLASTICITY_EVERY = 2000
PLASTICITY_CHURN = 0.05
LEAK_FRAC = 0.01
LEAK_FRAC_PLASTICITY = 0.10
LEAK_K = 50
LEAK_FTLE_AHEAD = 400

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's13_causal_audit_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's13_causal_audit_v1.json')


def build_csr(n_units=N_UNITS):
    return build_topology_csr('random_graph', n_units,
                             seed=TOPO_SEED, avg_degree=AVG_DEGREE)


def run_single(args):
    """(arm, seed_idx) -> dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    ip0, idx0, wt0 = build_csr()
    dt_seq, regime_seq = gen_regime_switch(seed=seed_idx)
    T = dt_seq.shape[0]
    n_classes = 3

    rls = OnlineRLS(N_UNITS + 1 + N_UNITS, n_classes,
                    forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV,
                    trace_cap=RLS_TRACE_CAP, reg=RLS_REG)

    # Metadata state
    m_state = np.zeros(N_UNITS)
    tau_m = TAU_SLOW

    # Homeostat
    kappa = KAPPA_NOMINAL
    kappa_hist = np.empty(T)
    kappa_hist[:] = np.nan

    # Plasticity mask (start from the random_graph topology)
    mask = np.zeros((N_UNITS, N_UNITS), dtype=bool)
    for i in range(N_UNITS):
        for e in range(ip0[i], ip0[i + 1]):
            j = int(idx0[e])
            mask[min(i, j), max(i, j)] = True
    n_edges = int(mask.sum())

    use_plasticity = arm not in ('no_plasticity',)
    use_causal_split = arm == 'causal_split'
    use_homeostat = True

    x_cur = x0.copy()
    preds = np.empty((T, n_classes))
    all_acc = np.empty(T)

    rng_noise = np.random.RandomState(seed_idx * 131 + 3)

    for blk_start in range(0, T, HOMEO_BLOCK):
        blk_end = min(blk_start + HOMEO_BLOCK, T)

        # Homeostat: estimate FTLE and adjust kappa
        if use_homeostat and blk_start % FTLE_EVERY == 0:
            w = min(FTLE_WINDOW, T - blk_start)
            lam, _ = run_pair_ftle_nb(
                x_cur, tau, dt_seq[blk_start:blk_start + w], PW,
                ip0, idx0, wt0, kappa,
                ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                COUPLING_CONTRAST_SELF, 1e-8, 10)

            if arm == 'leak_ftle' and blk_start + LEAK_FTLE_AHEAD + w < T:
                lam_future, _ = run_pair_ftle_nb(
                    x_cur, tau,
                    dt_seq[blk_start + LEAK_FTLE_AHEAD:
                           blk_start + LEAK_FTLE_AHEAD + w], PW,
                    ip0, idx0, wt0, kappa,
                    ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                    COUPLING_CONTRAST_SELF, 1e-8, 10)
                lam = (1.0 - LEAK_FRAC) * lam + LEAK_FRAC * lam_future

            err = float(np.clip(LAMBDA_TARGET - lam, -1.0, 1.0))
            kappa = float(np.clip(kappa + ETA_LAMBDA * err,
                                 KAPPA_MIN, KAPPA_MAX))

        # Run trajectory block
        ip_c, idx_c, wt_c = (adjacency_to_csr(mask) if use_plasticity
                             else (ip0, idx0, wt0))
        st_b, _, _ = run_trajectory_nb(
            x_cur, tau, dt_seq[blk_start:blk_end], PW,
            ip_c, idx_c, wt_c, kappa,
            ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
            COUPLING_CONTRAST_SELF, 0)

        obs = np.exp(gamma * st_b) / FEATURE_SCALE
        n_blk = obs.shape[0]

        for j in range(n_blk):
            t = blk_start + j

            # Metadata update
            f = obs[j]
            if arm == 'leak_metadata' and t + LEAK_K < T:
                f_future = np.exp(gamma * st_b[j + LEAK_K]) / FEATURE_SCALE \
                    if j + LEAK_K < n_blk else f
                f = (1.0 - LEAK_FRAC) * f + LEAK_FRAC * f_future
            m_state = (1.0 - 1.0 / tau_m) * m_state + (1.0 / tau_m) * f

            feat = np.concatenate([f, [1.0], m_state])

            # RLS predict
            pred = rls.predict(feat)
            preds[t] = pred

            # RLS update
            y = np.zeros(n_classes)
            y[regime_seq[t]] = 1.0
            if arm == 'leak_rls' and t + LEAK_K < T:
                y_future = np.zeros(n_classes)
                y_future[regime_seq[t + LEAK_K]] = 1.0
                y = (1.0 - LEAK_FRAC) * y + LEAK_FRAC * y_future
            rls.update(feat, y)

            all_acc[t] = float(np.argmax(pred) == regime_seq[t])

        kappa_hist[blk_start:blk_end] = kappa
        x_cur = st_b[-1].copy()

        # Structure plasticity
        if use_plasticity and (blk_start + n_blk) % PLASTICITY_EVERY == 0:
            obs_block = np.exp(gamma * st_b) / FEATURE_SCALE
            z = (obs_block - obs_block.mean(axis=0)) / (
                obs_block.std(axis=0) + 1e-12)
            C = (z.T @ z) / obs_block.shape[0]

            if use_causal_split:
                # Use only first 50% of the block for C
                half = n_blk // 2
                obs_half = obs_block[:half]
                z_half = (obs_half - obs_half.mean(axis=0)) / (
                    obs_half.std(axis=0) + 1e-12)
                C = (z_half.T @ z_half) / half

            n_grow = max(1, int(PLASTICITY_CHURN * n_edges))
            mask = evolve_mask(mask, C, n_grow)
            n_edges = int(mask.sum())

    # Metrics: overall accuracy, per-segment accuracy, kappa settled
    overall_acc = float(np.mean(all_acc[500:]))
    seg_accs = []
    for seg in range(6):
        lo = seg * RS_REGIME_LEN
        hi = lo + RS_REGIME_LEN
        if hi <= T:
            seg_accs.append(float(np.mean(all_acc[lo + 200:hi])))
    kappa_settled = float(np.nanmean(kappa_hist[-2000:]))

    return {'arm': arm, 'seed_idx': seed_idx,
            'overall_acc': overall_acc,
            'mean_seg_acc': float(np.mean(seg_accs)) if seg_accs else 0.0,
            'min_seg_acc': float(np.min(seg_accs)) if seg_accs else 0.0,
            'kappa_settled': kappa_settled,
            'n_edges': int(mask.sum()),
            'runtime_s': time.time() - t0}


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(r['arm'], []).append(r)
    agg = []
    for arm, rs in sorted(groups.items()):
        entry = {'arm': arm, 'n_runs': len(rs)}
        for f in ['overall_acc', 'mean_seg_acc', 'min_seg_acc',
                  'kappa_settled']:
            v = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.mean(v))
            entry[f + '_std'] = float(np.std(v))
        entry['n_edges_mean'] = float(np.mean([r['n_edges'] for r in rs]))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 90)
    print("S13 RESULTS (mean over seeds): causal leak audit")
    print("=" * 90)
    print(f" {'arm':>18} | {'acc':>7} {'seg_acc':>8} {'min_seg':>8} | "
          f"{'kappa':>6} {'edges':>6}")
    normal_acc = None
    for a in agg:
        if a['arm'] == 'normal':
            normal_acc = a['overall_acc_mean']
    for a in agg:
        delta = ""
        if normal_acc is not None and a['arm'] != 'normal':
            d = a['overall_acc_mean'] - normal_acc
            delta = f" ({'+' if d >= 0 else ''}{d:.4f})"
        print(f" {a['arm']:>18} | {a['overall_acc_mean']:>7.4f} "
              f"{a['mean_seg_acc_mean']:>8.4f} {a['min_seg_acc_mean']:>8.4f} | "
              f"{a['kappa_settled_mean']:>6.1f} {a['n_edges_mean']:>6.0f}{delta}")

    print("\n  Verdict:")
    if normal_acc is not None:
        for a in agg:
            if a['arm'].startswith('leak_'):
                d = a['overall_acc_mean'] - normal_acc
                status = "LEAK DETECTED" if d > 0.005 else "causally clean"
                print(f"    {a['arm']:>18}: delta = {d:+.4f} -> {status}")


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S13 causal audit "
          f"(quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_csr(16)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_NOMINAL,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    run_pair_ftle_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_NOMINAL,
                     ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                     COUPLING_CONTRAST_SELF, 1e-8, 10)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    arms = ['normal', 'leak_rls', 'leak_metadata', 'leak_ftle',
            'leak_plasticity', 'no_plasticity', 'causal_split']
    if quick:
        arms = ['normal', 'leak_rls', 'leak_ftle', 'causal_split']
    n_seeds = 2 if quick else N_SEEDS
    all_args = [(a, s) for a in arms for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seeds={n_seeds}, arms={len(arms)})")

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
        with Pool(min(4, max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 10) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed_idx', 'overall_acc', 'mean_seg_acc',
                  'min_seg_acc', 'kappa_settled', 'n_edges', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'kappa_nominal': KAPPA_NOMINAL, 'lambda_target': LAMBDA_TARGET,
        'eta_lambda': ETA_LAMBDA, 'tau_slow': TAU_SLOW,
        'plasticity_every': PLASTICITY_EVERY, 'plasticity_churn': PLASTICITY_CHURN,
        'leak_frac': LEAK_FRAC, 'leak_k': LEAK_K,
        'leak_ftle_ahead': LEAK_FTLE_AHEAD,
        'arms': arms, 'n_seeds': n_seeds, 'quick': bool(quick),
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
