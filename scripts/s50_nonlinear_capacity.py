#!/usr/bin/env python3
"""
L3 capacity trigger: is a nonlinear 2D boundary linearly decodable from the
substrate's state, or is it representation-insufficient? (REDEM S50)
=============================================================================
CORRECTION (audit P1-102, 2026-09-10) -- SUPERSEDED CAPACITY READING
The capacity conclusion recorded by this script is OVERTURNED and is kept
below for provenance only. The probe readings written by THIS script (circle
block-mean ridge 0.420 on random_graph_k25, 0.430 on parallel) came from a
DIFFERENT, ILL-CONDITIONED probe protocol: a ridge fit on the FULL 257-dim
block-mean features (plus a pulse-level ridge) with a MUCH SMALLER number of
training samples than the corrected block-mean PCA ridge used in
scripts/s53_auto_basis.py (held-out top-50 PCA ridge on block-mean features,
240-block window, refit every 20 blocks from block 300). Under that corrected,
well-conditioned protocol the coupled random_graph_k25 substrate's RAW
(non-expanded) features DO carry the circle: cap_mean = 0.8414 with
cap_sd = 0.0570 over 2550 readings, versus cap_mean = 0.5468 with
cap_sd = 0.0511 on the uncoupled parallel substrate
(data/s53_auto_basis_v1.json, field raw_capacity_probe). The "circle is not
decodable / basis expansion is necessary / L3 fires" reading below is
therefore an artifact of the ill-conditioned probe and is SUPERSEDED by
scripts/s53_auto_basis.py; representation-insufficiency survives only on the
parallel (uncoupled) substrate.
=============================================================================
Type:           PAPER
Paper Section:  S49 follow-up (L3 capacity probe, nonlinear boundary)
Experiment:     s49 showed rotation SPEED is a temporal limit (single RLS
                recovers to the ridge ceiling at every speed) -- speed does
                NOT trigger L3 (hypothesis generation). s50 asks the
                complementary question: does the substrate's nonlinear state
                expansion already make NONLINEAR 2D boundaries linearly
                decodable, or is there a structural capacity wall where a
                single linear readout (even RLS) cannot express the optimal
                boundary no matter how much it optimizes?

                If the circle/XOR boundaries are linearly decodable (ridge
                probe high AND rls reaches it), the substrate is a universal
                enough feature map -- hypothesis generation is unnecessary
                for this input family, L3 never fires. If the ridge probe is
                low (near chance), the boundary is representation-
                insufficient: L3 (a richer hypothesis class / population)
                is the ONLY way to improve, and s50 gives the first
                substrate-grounded observation of that failure mode.
                [SUPERSEDED by s53_auto_basis: see data/s53_auto_basis_v1.json]

Arms (nonlinear2d, static boundary over all 2000 blocks, K=20, T=40000;
ALL arms scored by the protocol decision = block vote mean(block)>0.5):
  delta_base   : error-driven delta, fixed eta, norm cap (honest online
                 learner; the RLS arm is deliberately NOT used here -- on an
                 undecodable task its uncapped covariance overfits (|W| -> O(1e3),
                 in-sample 1.000 while held-out ridge says chance), and capping
                 W breaks the P/W consistency; the capacity verdict must come
                 from the supervised ridge probe, and delta shows what is
                 honestly reachable online).
  rmhl_norm    : Hebbian with norm cap (does the Hebbian family express a
                 static nonlinear boundary at all?).

The supervised capacity ceiling is the RIDGE probe on BLOCK-MEAN features
(the protocol decision space: mean_t(w.x_t) = w.mean_t(x_t)):
  linear       : expect >= 0.85 (reproduces s48 decodability).
  circle       : EXPECT ~ chance (0.42-0.5) on BOTH substrates -- the
                 substrate's nonlinear expansion does NOT make a circular
                 boundary linearly decodable -> representation-insufficient,
                 the FIRST substrate-grounded L3 trigger.
                 [SUPERSEDED by s53_auto_basis: see data/s53_auto_basis_v1.json]
  xor          : partial on the coupled substrate (~0.7, the nonlinear
                 expansion partially expresses XOR), chance on parallel.

Envs (boundary type): linear, circle (radius 0.4), xor, each x
{parallel, random_graph_k25} x 10 seeds. Ridge probe (block-level AND
pulse-level, seed 0) on each substrate is computed in main.

Predictions:
  P1 (linear baseline): block ridge ~0.9; delta_base/rmhl_norm reach a
      good fraction of it (delta > rmhl, rule choice, s46).
  P2 (L3 trigger): circle block-ridge ~ chance on BOTH substrates ->
      representation-insufficient is REAL and substrate-grounded: L3 is
      justified; online learners also sit near chance (they cannot reach
      what the features do not contain).
      [SUPERSEDED by s53_auto_basis: see data/s53_auto_basis_v1.json]
  P3 (partial express): xor block-ridge ~0.7 on random_graph (coupling
      partially expresses XOR), ~chance on parallel.

Output files:
  data/s50_nonlinear_capacity_v1.csv    (one row per run)
  data/s50_nonlinear_capacity_v1.json   (params + per-cell aggregates + probes)

Usage: python s50_nonlinear_capacity.py [--quick]
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
from online_readout import OnlineRLS, ThreeFactorReadout, ridge_fit, \
    running_mean_accuracy
from streaming_tasks import gen_nonlinear2d, DB_K_PULSES

# ==================== Fixed parameters (S3/S39-S49-consistent) ====================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

RMHL_ETA = 0.05
RMHL_ELIG_DECAY = 0.9
W_MAX = 20.0
DELTA_ETA = 0.05

RLS_FORGETTING = 0.99
RLS_INIT_COV = 1.0

N_BLOCKS = 2000
BOUNDARIES = ['linear', 'circle', 'xor']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's50_nonlinear_capacity_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's50_nonlinear_capacity_v1.json')

CURVE_WINDOW = 200

ARMS = ['delta_base', 'rmhl_norm']


def _block_vote_score(preds_block, target_block):
    """Protocol decision: vote = mean(preds over block) > 0.5."""
    return float(np.mean(preds_block) > 0.5)


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


def ridge_probe(F, target, b_lo, b_hi):
    lo = b_lo * DB_K_PULSES
    hi = b_hi * DB_K_PULSES
    X, y = F[lo:hi], target[lo:hi]
    n_train = X.shape[0] // 2
    out = ridge_fit(X[:n_train], y[:n_train], X[n_train:], y[n_train:],
                    ridge_lambda=1.0)
    pred = out['pred_te']
    return float(np.mean((pred > 0.5).astype(np.float64) == y[n_train:]))


def ridge_probe_block(F, target, b_lo, b_hi):
    """Ridge on BLOCK-MEAN features (protocol decision space:
    mean_t(w.x_t) = w . mean_t(x_t)); the honest supervised ceiling."""
    n_blk = b_hi - b_lo
    Fb = np.empty((n_blk, F.shape[1]))
    for i, b in enumerate(range(b_lo, b_hi)):
        Fb[i] = F[b * DB_K_PULSES:(b + 1) * DB_K_PULSES].mean(axis=0)
    yb = target[b_lo * DB_K_PULSES:(b_hi * DB_K_PULSES):DB_K_PULSES]
    n_train = n_blk // 2
    out = ridge_fit(Fb[:n_train], yb[:n_train], Fb[n_train:], yb[n_train:],
                    ridge_lambda=1.0)
    pred = out['pred_te']
    return float(np.mean((pred > 0.5).astype(np.float64) == yb[n_train:]))


# ========================== Single run ==========================

def run_single(args):
    """(topo_name, arm, boundary, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, boundary, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq, _, _ = gen_nonlinear2d(
        seed=seed_idx, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
        boundary=boundary)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F = build_features(states, T)
    d = F.shape[1]

    preds = np.empty(T)

    if arm == 'rmhl_norm':
        tf = ThreeFactorReadout(d, mode='reward', learning_rate=RMHL_ETA,
                                elig_decay=RMHL_ELIG_DECAY, seed=seed_idx,
                                w_max=W_MAX)
        for t in range(T):
            o = tf.observe(F[t])
            preds[t] = o
            if (t + 1) % DB_K_PULSES == 0:
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                tf.consolidate(r, reset=True)
                # protocol decision = block vote; record block-mean so the
                # accuracy curve reflects the actual decision
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
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'nonlinear2d', 'boundary': boundary, 'substrate': topo_name,
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000),
           'steady_acc': acc_median_in(acc_run, 1800, 2000),
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['boundary'], r['substrate'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        boundary, topo, read = key
        entry = {'boundary': boundary, 'substrate': topo, 'readout': read,
                 'n_runs': len(rs)}
        for f in ['pre_swap_acc', 'steady_acc', 'mean_acc_all']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        agg.append(entry)
    return agg


def print_table(agg, probes):
    print("\n" + "=" * 118)
    print("S50 L3 CAPACITY TRIGGER: NONLINEAR 2D BOUNDARY DECODABILITY")
    print("=" * 118)
    for boundary in BOUNDARIES:
        rb = probes.get(f"{boundary}|random_graph_k25|block", float('nan'))
        pb = probes.get(f"{boundary}|parallel|block", float('nan'))
        rp = probes.get(f"{boundary}|random_graph_k25|pulse", float('nan'))
        pp = probes.get(f"{boundary}|parallel|pulse", float('nan'))
        print(f"\n--- {boundary} (block ridge: random={rb:.3f} "
              f"parallel={pb:.3f}; pulse ridge: random={rp:.3f} "
              f"parallel={pp:.3f}) ---")
        print(f"  {'substrate':<17} | {'readout':<10} | {'pre':>6} | "
              f"{'steady':>6} | {'mean':>6}")
        for a in [x for x in agg if x['boundary'] == boundary]:
            print(f"  {a['substrate']:<17} | {a['readout']:<10} | "
                  f"{a['pre_swap_acc_mean']:>6.3f} | "
                  f"{a['steady_acc_mean']:>6.3f} | "
                  f"{a['mean_acc_all_mean']:>6.3f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S50 nonlinear capacity "
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
    for boundary in BOUNDARIES:
        for topo in ['parallel', 'random_graph_k25']:
            for arm in ARMS:
                for s in range(n_seeds):
                    all_args.append((topo, arm, boundary, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (boundaries={len(BOUNDARIES)}, "
          f"arms={len(ARMS)}, seeds={n_seeds})")

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

    # supervised ridge probes at the fixed boundary on BOTH substrates, seed 0
    probes = {}
    for topo in ['random_graph_k25', 'parallel']:
        tau_p = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=0)
        x0_p = preprogram_vec(ALPHA0, tau_p)
        ip_p, idx_p, wt_p, mode_p, kap_p = build_substrate(topo)
        for boundary in BOUNDARIES:
            dt_p, tgt_p, _, _ = gen_nonlinear2d(
                seed=0, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
                boundary=boundary)
            st_p, _, _ = run_trajectory_nb(
                x0_p, tau_p, dt_p, PW, ip_p, idx_p, wt_p, kap_p,
                ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode_p, 0)
            F_p = build_features(st_p, dt_p.shape[0])
            probes[f"{boundary}|{topo}|pulse"] = ridge_probe(F_p, tgt_p, 800, 1000)
            probes[f"{boundary}|{topo}|block"] = ridge_probe_block(
                F_p, tgt_p.astype(np.float64), 800, 1000)
    for k in sorted(probes.keys()):
        print(f"  ridge probe {k}: {probes[k]:.3f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'boundary', 'substrate', 'readout', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'mean_acc_all',
                  'pre_swap_acc', 'steady_acc']
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
        'w_max': W_MAX, 'delta_eta': DELTA_ETA,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'n_blocks': N_BLOCKS, 'boundaries': BOUNDARIES,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg, 'ridge_probe': probes},
                  f, indent=2)

    print_table(agg, probes)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
