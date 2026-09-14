#!/usr/bin/env python3
"""
Rich-kernel saturation: is the irreducible ~+4-6pp edge of s58a a monotone
function of SUBSTRATE CAPACITY? (REDEM S58D)
=============================================================================
Type:           PAPER
Paper Section:  S58a follow-up (mechanistic confirmation of the negative)
Experiment:     s58a showed the coupled (random_graph) substrate retains an
                irreducible ~+4-6pp margin over chance even at 6-surface
                boundaries (L3 = asymptotic degradation, never collapse),
                and proposed the "rich-kernel machine" explanation: the
                substrate kernel's capacity smoothly bounds the readout
                margin. s58d tests that explanation POSITIVELY: lower the
                kernel capacity and the margin should compress toward 0
                (weak kernel -> near collapse); raise it and the margin
                should lift.

Capacity is swept along two axes:
  kappa (coupling contrast, the kernel's construction source):
        KAPPA_RANDOM in {10, 15, 25, 40} at N=256.
  n_units (kernel width): N in {64, 256, 512} at kappa=25.

Environment (static, single regime, both axes on the coupled substrate):
  stress : 4D shell (annulus, 2 surfaces, P=0.501, s57/s58a) -- the s58a
           stress point whose margin is small (~+4-6pp).
  easy   : 2D circle (P=0.50, s55) -- reference where the margin is large
           (the kernel is never the binding constraint there).

One run per (config, seed) computes BOTH readouts on the same trajectory:
  rls_raw : online RLS on raw features (the operative readout).
  probe   : well-conditioned capacity probe (ridge on top-50 PCA of
            block-mean features, s53 method).

Prediction (falsifiable):
  margin (steady accuracy - 0.5) is a MONOTONE function of capacity:
    margin(kappa=10) ~ 0 (rich-kernel explanation predicts near-collapse
    when the coupling that constructs the kernel is weakened),
    margin(kappa=40) > margin(kappa=25),
    margin(N=64) < margin(N=256) < margin(N=512).
  On the 2D circle the same monotonicity holds but with a LARGER baseline
  margin at every capacity (the easy boundary never becomes binding).
  If margin is FLAT in kappa/N, the rich-kernel explanation is refuted and
  the s58a edge must come from elsewhere (e.g., the encoding's residual
  structure) -- recorded as a negative.

Output files:
  data/s58d_kernel_capacity_sweep_v1.csv   (one row per run)
  data/s58d_kernel_capacity_sweep_v1.json  (params + per-cell aggregates)

Usage: python s58d_kernel_capacity_sweep.py [--quick]
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
from streaming_tasks import (
    gen_nonlinear2d, gen_nonlinear4d, DB_K_PULSES, ROT4D_BANDS,
    ROT4D_K_PULSES, ROT4D_SPHERE_R, ROT4D_SHELL_IN, ROT4D_SHELL_OUT)

# ==================== Fixed parameters (S3/S39-S58-consistent) ====================
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0

RLS_FORGETTING = 0.99
RLS_INIT_COV = 1.0

CAP_N_PC = 50          # PCA components for the well-conditioned probe

N_BLOCKS = 2000
# 4D pulses per block (4 channels x 6) and the 4D shell band come from the CORE
# task module; imported instead of re-declared (dedup P1-95). The band was the
# literal (0.45, 0.640).
K4D = ROT4D_K_PULSES
SHELL4D = ((ROT4D_SHELL_IN, ROT4D_SHELL_OUT),)   # 4D shell band, P=0.501 (s57)

# capacity axis: (env, k_pulses, kappa, n_units)
# kappa grid densified 2026-09-09 (dedup P0-20): the original 4-point grid
# (10/15/25/40) left the inverted-U peak bracketed only by (25, 40), which
# cannot be reconciled with the claim that the deployment value kappa=25 is
# the optimum. kappa in {20, 30, 35, 45} added at N=256 for both boundaries so
# the peak and the over-coupling collapse are bracketed by data.
CONFIGS = [
    # kappa axis at N=256
    ('4Dshell', K4D, 10.0, 256),
    ('4Dshell', K4D, 15.0, 256),
    ('4Dshell', K4D, 20.0, 256),
    ('4Dshell', K4D, 25.0, 256),
    ('4Dshell', K4D, 30.0, 256),
    ('4Dshell', K4D, 35.0, 256),
    ('4Dshell', K4D, 40.0, 256),
    ('4Dshell', K4D, 45.0, 256),
    # N axis at kappa=25
    ('4Dshell', K4D, 25.0, 64),
    ('4Dshell', K4D, 25.0, 512),
    # easy-boundary reference (same axes)
    ('2Dcircle', DB_K_PULSES, 10.0, 256),
    ('2Dcircle', DB_K_PULSES, 15.0, 256),
    ('2Dcircle', DB_K_PULSES, 20.0, 256),
    ('2Dcircle', DB_K_PULSES, 25.0, 256),
    ('2Dcircle', DB_K_PULSES, 30.0, 256),
    ('2Dcircle', DB_K_PULSES, 35.0, 256),
    ('2Dcircle', DB_K_PULSES, 40.0, 256),
    ('2Dcircle', DB_K_PULSES, 45.0, 256),
    ('2Dcircle', DB_K_PULSES, 25.0, 64),
    ('2Dcircle', DB_K_PULSES, 25.0, 512),
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's58d_kernel_capacity_sweep_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's58d_kernel_capacity_sweep_v1.json')

CURVE_WINDOW = 200
PROBE_LO, PROBE_HI = 800, 2000   # probe window (blocks)


def _annulus_labels(uv, bands):
    """y = 1[d2 in one of the disjoint radial bands], d2 = ||u-0.5||^2."""
    d2 = sum((u - 0.5) ** 2 for u in uv)
    y = np.zeros(d2.shape[0], dtype=np.int64)
    for lo, hi in bands:
        y |= ((d2 > lo * lo) & (d2 < hi * hi)).astype(np.int64)
    return y


def gen_env(env, seed, k):
    # Geometry left as CORE defaults on purpose (dedup P1-95): the 2D circle
    # uses gen_nonlinear2d(radius=0.40) and the 4D points use
    # gen_nonlinear4d(radius=ROT4D_SPHERE_R); the effective values are recorded
    # in the params JSON.
    if env == '2Dcircle':
        dt, tgt, _, _ = gen_nonlinear2d(seed=seed, n_blocks=N_BLOCKS,
                                        k_pulses=k, boundary='circle')
        return dt, tgt
    # 4D shell: uniform 4D points (sphere generator), annulus relabel
    dt, tgt, *uv = gen_nonlinear4d(seed=seed, n_blocks=N_BLOCKS,
                                   k_pulses=k, boundary='sphere',
                                   bands=ROT4D_BANDS)
    labels = _annulus_labels(uv, SHELL4D)
    return dt, np.repeat(labels, k).astype(np.int64)


def build_features(states, T):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    return np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])


def capacity_pca(F, tgt, b_lo, b_hi, k, n_pc=CAP_N_PC):
    """Well-conditioned capacity probe: held-out ridge on top-n_pc PCA of
    block-mean features (s55/s57)."""
    Fb = np.array([F[b * k:(b + 1) * k].mean(axis=0)
                   for b in range(b_lo, b_hi)])
    yb = tgt[b_lo * k:(b_hi * k):k].astype(np.float64)
    n = Fb.shape[0]
    n_tr = n // 2
    mu = Fb[:n_tr].mean(axis=0)
    sd = Fb[:n_tr].std(axis=0) + 1e-9
    Xs = (Fb - mu) / sd
    U, S, Vt = np.linalg.svd(Xs[:n_tr], full_matrices=False)
    kk = min(n_pc, Vt.shape[0])
    Xp = Xs @ Vt[:kk].T
    out = ridge_fit(Xp[:n_tr], yb[:n_tr], Xp[n_tr:], yb[n_tr:],
                    ridge_lambda=1.0)
    pred = out['pred_te']
    return float(np.mean((pred > 0.5).astype(np.float64) == yb[n_tr:]))


def acc_median_in(acc_run, b_lo, b_hi, k):
    lo, hi = b_lo * k, b_hi * k
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


# ========================== Single run ==========================

def run_single(args):
    """(config_idx, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    config_idx, seed_idx = args
    env, k, kappa, n_units = CONFIGS[config_idx]
    t0 = time.time()

    tau = gen_tau_vec(n_units, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts = build_topology_csr('random_graph', n_units,
                                              seed=TOPO_SEED,
                                              avg_degree=AVG_DEGREE)
    mode = COUPLING_CONTRAST_SELF

    dt_seq, target_seq = gen_env(env, seed_idx, k)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F = build_features(states, T)
    d = F.shape[1]

    rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV)
    preds = np.empty(T)
    for t in range(T):
        preds[t] = float(rls.predict(F[t])[0])
        if (t + 1) % k == 0:
            seg = F[t - k + 1:t + 1]
            x_blk = seg.mean(axis=0)
            blk = preds[t - k + 1:t + 1]
            vote = float(np.mean(blk) > 0.5)
            r = 1.0 if vote == target[t] else -1.0
            y_derived = vote if r > 0.0 else (1.0 - vote)
            rls.update(x_blk, np.array([y_derived]))
            preds[t - k + 1:t + 1] = np.mean(blk)

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    probe = capacity_pca(F, target_seq, PROBE_LO, PROBE_HI, k)

    res = {'task': 'kernel_capacity', 'env': env, 'k_pulses': int(k),
           'kappa': float(kappa), 'n_units': int(n_units),
           'seed_idx': seed_idx, 't_total': int(T),
           'runtime_s': time.time() - t0,
           'rls_mean': float(np.nanmean(acc_run)),
           'rls_steady': acc_median_in(acc_run, N_BLOCKS - 200, N_BLOCKS, k),
           'probe': probe}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        key = (r['env'], r['kappa'], r['n_units'])
        groups.setdefault(key, []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        env, kappa, n_units = key
        entry = {'env': env, 'kappa': kappa, 'n_units': n_units,
                 'n_runs': len(rs)}
        for f in ['rls_mean', 'rls_steady', 'probe']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        entry['margin_rls'] = entry['rls_steady_mean'] - 0.5
        entry['margin_probe'] = entry['probe_mean'] - 0.5
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 118)
    print("S58D KERNEL-CAPACITY SWEEP: margin vs (kappa, N)")
    print("=" * 118)
    for env in ['4Dshell', '2Dcircle']:
        print(f"\n--- {env} (trivial baseline 0.5) ---")
        print(f"  {'config':<14} | {'rls_mean':>8} | {'rls_steady':>10} | "
              f"{'probe':>6} | {'margin_rls':>10} | {'margin_probe':>12}")
        for e in [x for x in agg if x['env'] == env]:
            print(f"  k={e['kappa']:<6.0f} N={e['n_units']:<5d} | "
                  f"{e['rls_mean_mean']:>8.3f} | "
                  f"{e['rls_steady_mean']:>10.3f} | "
                  f"{e['probe_mean']:>6.3f} | "
                  f"{e['margin_rls']:>10.3f} | "
                  f"{e['margin_probe']:>12.3f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S58D kernel-capacity sweep "
          f"(quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16,
                                           seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, 25.0,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = N_SEEDS if not quick else 2
    all_args = []
    for ci in range(len(CONFIGS)):
        for s in range(n_seeds):
            all_args.append((ci, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (configs={len(CONFIGS)}, seeds={n_seeds})")

    results = []
    n_proc = min(cpu_count(), 8, max(1, n_runs))
    with Pool(n_proc) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=2):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress "
                      f"{done}/{n_runs}", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'env', 'k_pulses', 'kappa', 'n_units', 'seed_idx',
                  't_total', 'runtime_s', 'rls_mean', 'rls_steady', 'probe']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'cv_tau': CV_TAU, 'alpha0': ALPHA0, 'gamma': float(gamma),
        'tau0': float(tau0), 'topo_seed': TOPO_SEED,
        'avg_degree': AVG_DEGREE,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'cap_n_pc': CAP_N_PC, 'n_blocks': N_BLOCKS,
        'probe_lo': PROBE_LO, 'probe_hi': PROBE_HI,
        'k4d': K4D, 'shell4d': list(SHELL4D[0]),
        # geometry traceability (dedup P1-95): K4D/SHELL4D above are the CORE
        # exports; the radii are the CORE generator defaults this script relies
        # on by NOT passing them.
        'sphere4d_radius': float(ROT4D_SPHERE_R),
        'circle2d_radius': 0.40,
        'configs': [{'env': c[0], 'k_pulses': c[1], 'kappa': c[2],
                     'n_units': c[3]} for c in CONFIGS],
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total "
          f"{time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
