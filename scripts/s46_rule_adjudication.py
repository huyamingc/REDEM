#!/usr/bin/env python3
"""
Rule adjudication: is the +/-1-reward credit-assignment limit information-
theoretic, or a credit-assignment RULE choice? (REDEM S46)
=============================================================================
Type:           PAPER
Paper Section:  S39-S45 follow-up (reinterpretation of the reward-credit
                chain)
Experiment:     In this block protocol the +/-1 reward plus the learner's
                own vote DETERMINES the label exactly: r=+1 => y = vote,
                r=-1 => y = 1-vote (binary correctness). So the reward
                stream is informationally equivalent to "label revealed
                after prediction". RMHL fails on boundary shifts (R2/R3,
                s44/s45) because it uses r only as a Hebbian sign gate,
                discarding the label information. A rule that reconstructs
                y from (r, vote) and updates error-driven (delta rule) uses
                the SAME +/-1 stream but a different credit-assignment
                choice -- and should recover every regime, including the
                shifts that collapse all RMHL-family arms.

Arms (drift-binary, regime change @block 1000, K=20, T=40000):
  rmhl          : RMHL (s44/s45 baseline; fails R2/R3).
  rmhl_norm     : RMHL + weight norm cap (s45; recovers R1/R4 only).
  delta_derived : per-pulse sigmoid readout; per block, reconstruct
                  y_derived = vote if r=+1 else 1-vote; delta update
                  w += eta * sum_t (y_derived - o_t) F_t. Uses ONLY the
                  same (r, vote) the RMHL arms see. Norm cap applied.

Envs: R1 full swap / R2 partial shift / R3 compression / R4 near-inversion
(regime zoo from s44).

Predictions (status checked against the committed data, 10 seeds; the refuted
entry keeps its original claim plus the measured outcome):
  P1 (adjudication): delta_derived recovers ALL regimes (~0.9+ incl.
      R2/R3), matching the s44 rls_fresh oracle (0.988-1.000) -- the
      failure is the RULE, not the information in the +/-1 stream.
      [REFUTED by own data: see data/s46_rule_adjudication_v1.csv] the
      error-driven rule recovers the INVERSION regimes but only partly
      recovers the boundary-shift regimes, so it does not match the oracle.
      Measured steady/post (mean over 10 seeds): R1 0.9890/0.9160
      (parallel) and 0.9990/0.9340 (random_graph_k25); R2 0.5355/0.5487 and
      0.5563/0.5590; R3 0.5817/0.6040 and 0.7520/0.7415; R4 0.9915/0.8745
      and 0.9765/0.9170. The s44 rls_fresh oracle is 0.988-1.000 in R2/R3,
      which delta_derived does not reach (0.5355-0.7520 steady there).
  P2: rmhl_norm still collapses on R2/R3 (reproduces s45), confirming the
      rule choice (Hebbian vs error-driven) is the differentiator.
      HOLDS: rmhl_norm steady is 0.5030/0.4990 in R2 and 0.4930/0.4990 in R3
      (parallel/random_graph_k25), i.e. chance.

Output files:
  data/s46_rule_adjudication_v1.csv    (one row per run)
  data/s46_rule_adjudication_v1.json   (params + per-cell aggregates)

Usage: python s46_rule_adjudication.py [--quick]
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
from online_readout import running_mean_accuracy
from streaming_tasks import gen_drift_binary, DB_K_PULSES

# ========================== Fixed parameters (S3/S39-S45-consistent) ==========================
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
DELTA_ETA = 0.05
W_MAX = 20.0          # weight norm cap (bias-runaway fix, s45)

SWAP_BLOCK = 1000
N_BLOCKS = 2000

DT0_INIT, DT1_INIT = 10e-6, 60e-6
DT_MIN, DT_MAX = 4e-6, 120e-6

REGIMES = {
    'R1_full_swap':      lambda a, b: (b, a),
    'R2_partial_shift':  lambda a, b: (30e-6, b),
    'R3_compression':    lambda a, b: (25e-6, 45e-6),
    'R4_near_inversion': lambda a, b: (55e-6, 15e-6),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's46_rule_adjudication_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's46_rule_adjudication_v1.json')

CURVE_WINDOW = 200

ARMS = ['rmhl', 'rmhl_norm', 'delta_derived']


def gen_regime_stream(seed, regime):
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, 2, N_BLOCKS)
    dt0, dt1 = DT0_INIT, DT1_INIT
    block_dt = np.empty(N_BLOCKS)
    for b in range(N_BLOCKS):
        block_dt[b] = dt0 if labels[b] == 0 else dt1
        if (b + 1) % 200 == 0:
            dt0 = np.clip(dt0 * (1.0 + rng.randn() * 0.03), DT_MIN, DT_MAX)
            dt1 = np.clip(dt1 * (1.0 + rng.randn() * 0.03), DT_MIN, DT_MAX)
        if b + 1 == SWAP_BLOCK:
            dt0, dt1 = REGIMES[regime](dt0, dt1)
    dt_seq = np.repeat(block_dt, DB_K_PULSES)
    target_seq = np.repeat(labels, DB_K_PULSES).astype(np.int64)
    return dt_seq, target_seq


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


# ========================== Single run ==========================

def run_single(args):
    """(topo_name, arm, regime, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, regime, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq = gen_regime_stream(seed_idx, regime)
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
    d = F.shape[1]

    rng_w = np.random.RandomState(seed_idx * 97 + 3)
    w = rng_w.uniform(-0.5, 0.5, d)
    e = np.zeros(d)
    preds = np.empty(T)

    if arm == 'rmhl' or arm == 'rmhl_norm':
        is_norm = (arm == 'rmhl_norm')
        for t in range(T):
            o = _sigmoid(float(w @ F[t]))
            preds[t] = o
            e = RMHL_ELIG_DECAY * e + F[t] * o
            if (t + 1) % DB_K_PULSES == 0:
                b = (t + 1) // DB_K_PULSES - 1
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                w += RMHL_ETA * r * e
                e[:] = 0.0
                if is_norm:
                    nw = float(np.linalg.norm(w))
                    if nw > W_MAX:
                        w *= (W_MAX / nw)
    else:  # delta_derived
        acc_sF = np.zeros(d)
        acc_oF = np.zeros(d)
        for t in range(T):
            o = _sigmoid(float(w @ F[t]))
            preds[t] = o
            acc_sF += F[t]
            acc_oF += F[t] * o
            if (t + 1) % DB_K_PULSES == 0:
                b = (t + 1) // DB_K_PULSES - 1
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                # label reconstruction from (r, vote): r=+1 => y=vote,
                # r=-1 => y=1-vote. delta update over the block:
                #   w += eta * sum_t (y_derived - o_t) F_t
                y_derived = vote if r > 0.0 else (1.0 - vote)
                w += DELTA_ETA * (y_derived * acc_sF - acc_oF)
                acc_sF[:] = 0.0
                acc_oF[:] = 0.0
                nw = float(np.linalg.norm(w))
                if nw > W_MAX:
                    w *= (W_MAX / nw)

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'drift_binary', 'regime': regime, 'substrate': topo_name,
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000),
           'post_swap_acc': acc_median_in(acc_run, 1100, 1200),
           'steady_acc': acc_median_in(acc_run, 1800, 2000),
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['regime'], r['substrate'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        regime, topo, read = key
        entry = {'regime': regime, 'substrate': topo, 'readout': read,
                 'n_runs': len(rs)}
        for f in ['pre_swap_acc', 'post_swap_acc', 'steady_acc',
                  'mean_acc_all']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 120)
    print("S46 RULE ADJUDICATION (drift_binary, regime @1000, mean over seeds)")
    print("=" * 120)
    for regime in sorted(REGIMES.keys()):
        print(f"\n--- {regime} ---")
        print(f"  {'substrate':<17} | {'readout':<15} | {'pre':>6} | "
              f"{'post':>6} | {'steady':>6} | {'mean':>6}")
        for a in [x for x in agg if x['regime'] == regime]:
            print(f"  {a['substrate']:<17} | {a['readout']:<15} | "
                  f"{a['pre_swap_acc_mean']:>6.3f} | "
                  f"{a['post_swap_acc_mean']:>6.3f} | "
                  f"{a['steady_acc_mean']:>6.3f} | "
                  f"{a['mean_acc_all_mean']:>6.3f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S46 rule adjudication "
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
    for regime in sorted(REGIMES.keys()):
        for topo in ['parallel', 'random_graph_k25']:
            for arm in ARMS:
                for s in range(n_seeds):
                    all_args.append((topo, arm, regime, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (regimes={len(REGIMES)}, arms={len(ARMS)}, "
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

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'regime', 'substrate', 'readout', 'seed_idx',
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
        'rmhl_eta': RMHL_ETA, 'rmhl_elig_decay': RMHL_ELIG_DECAY,
        'delta_eta': DELTA_ETA, 'w_max': W_MAX,
        'swap_block': SWAP_BLOCK, 'n_blocks': N_BLOCKS,
        'dt0_init': DT0_INIT, 'dt1_init': DT1_INIT,
        'n_seeds': n_seeds, 'quick': bool(quick),
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
