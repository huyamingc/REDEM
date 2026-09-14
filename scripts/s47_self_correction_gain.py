#!/usr/bin/env python3
"""
Self-correction on the error-driven base: does reward-driven gain modulation
close the R2/R3 shift gap that S46's delta_derived left at 0.54-0.75?
=============================================================================
Type:           PAPER
Paper Section:  S46 follow-up (self-evolution step on the 1D input family)
Experiment:     S46 adjudicated that the +/-1 credit-assignment limit is a
                RULE choice: the error-driven learner that reconstructs the
                label from (r, vote) -- r=+1 => y=vote, r=-1 => y=1-vote --
                recovers the inversion regimes R1/R4 fully (0.989/0.999) but
                leaves the boundary-shift regimes R2/R3 partial (0.536-0.752)
                because a fixed-eta delta rule tracks a moving boundary
                slowly. This script asks the self-evolution question: can the
                learner CORRECT ITSELF, using only its own reward stream, by
                boosting its learning rate when its own reward-rate EMA
                collapses (the convention-free meta-signal of S40-S42, D1)?
                No oracle, no hand-timed regime announcement.

Arms (drift-binary, regime change @block 1000, K=20, T=40000):
  delta_base   : S46 delta_derived replication (fixed eta, norm cap).
  delta_boost  : delta_derived + reward-driven gain modulation:
                 eta_b = DELTA_ETA * min(G_MAX, 1 + G_K*(C_THRESH - |r_ema|)_+)
                 -- low eta while confident, boosted while uncertain.
  delta_high   : delta_derived with eta = DELTA_ETA * G_MAX always. Control
                 that separates "adaptivity" from "just a higher eta".
  rls_online   : pure-online block-level RLS (forgetting 0.99) on the
                 reconstructed label y_derived -- the adaptive per-dimension
                 gain ceiling reachable with ONLY the (r, vote) stream.
                 NOTE: the protocol decision is the block vote
                 mean(block) > 0.5, so the RLS arm is scored on block-mean
                 predictions (its linear output has no natural per-pulse
                 0.5 threshold when |W| grows; pulse-level thresholding of
                 an uncapped linear output is an evaluation artifact).

Envs: R1 full swap / R2 partial shift / R3 compression / R4 near-inversion
(regime zoo from s44).

Predictions (status checked against the committed data, 10 seeds; the refuted
entry keeps its original claim plus the measured outcome):
  P1 (regression): delta_base reproduces s46 (R1/R4 ~0.99, R2/R3 ~0.54-0.75).
      HOLDS: delta_base steady/post R1 0.9890/0.9160 and 0.9990/0.9340, R2
      0.5355/0.5487 and 0.5563/0.5590, R3 0.5817/0.6040 and 0.7520/0.7415,
      R4 0.9915/0.8745 and 0.9765/0.9170 (parallel / random_graph_k25).
  P2 (self-corr.): delta_boost improves R2/R3 substantially over delta_base,
      approaching rls_online, WITHOUT hurting R1/R4 (gain only rises when
      confidence collapses).
      [REFUTED by own data: see data/s47_self_correction_gain_v1.csv] the
      boosted arm does NOT improve R2/R3 at all; it lands at or below
      delta_base while rls_online is 1.0000 there. steady/post:
      R2 parallel delta_base 0.5355/0.5487 vs delta_boost 0.5275/0.5215;
      R2 random 0.5563/0.5590 vs 0.5275/0.5215; R3 parallel 0.5817/0.6040
      vs 0.5275/0.5215; R3 random 0.7520/0.7415 vs 0.6133/0.6475. It also
      costs on a stable regime: R4 parallel steady falls from 0.9915
      (delta_base) to 0.9575 (delta_boost).
  P3 (adaptivity): delta_boost >= delta_high on R2/R3 AND beats it on the
      stable regimes (R1/R4 steady) -- the modulation, not the higher eta,
      is doing the work.
      [NOT VERIFIABLE from own data: see data/s47_self_correction_gain_v1.csv]
      in the R2/R3 dead zone the two arms are numerically IDENTICAL -- the
      post_swap_acc and steady_acc columns are equal seed-by-seed for R2
      parallel, R2 random_graph_k25 and R3 parallel (both 0.5275/0.5215) --
      because the boosted eta saturates at G_MAX and therefore equals the
      delta_high eta, so that comparison carries no information; the
      stable-regime half does hold (R1 steady 0.9870 vs 0.6890 parallel and
      1.0000 vs 0.8460 random; R4 steady 0.9575 vs 0.6470 and 0.9790 vs
      0.8265).
  P4 (no oracle): the gain schedule is driven purely by the reward-rate EMA
      of the same stream (gain_mean/gain_max recorded per run).
      HOLDS: gain_mean_mean is non-zero only for delta_boost (1.12-4.82,
      gain_max_mean 9.82-10.00) and the other arms record 0.00.

Output files:
  data/s47_self_correction_gain_v1.csv    (one row per run)
  data/s47_self_correction_gain_v1.json   (params + per-cell aggregates)

Usage: python s47_self_correction_gain.py [--quick]
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
from online_readout import OnlineRLS, running_mean_accuracy
from streaming_tasks import gen_drift_binary, DB_K_PULSES

# ==================== Fixed parameters (S3/S39-S46-consistent) ====================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

DELTA_ETA = 0.05
W_MAX = 20.0            # weight norm cap (s45/s46)
R_EMA_ALPHA = 0.02      # reward-rate EMA smoothing (S40-S42 convention)
C_THRESH = 0.60         # confidence threshold: boost when |r_ema| < this
G_MAX = 10.0            # max gain multiplier on eta
G_K = (G_MAX - 1.0) / C_THRESH   # gain slope: c=0 -> eta = DELTA_ETA*G_MAX

RLS_FORGETTING = 0.99
RLS_INIT_COV = 1.0

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
CSV_PATH = os.path.join(DATA_DIR, 's47_self_correction_gain_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's47_self_correction_gain_v1.json')

CURVE_WINDOW = 200

ARMS = ['delta_base', 'delta_boost', 'delta_high', 'rls_online']


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


def _cap(w, w_max):
    nw = float(np.linalg.norm(w))
    if nw > w_max:
        w = w * (w_max / nw)
    return w


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
    preds = np.empty(T)

    gain_sum = 0.0
    gain_max = 0.0
    n_blocks = 0

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
                # protocol decision = block vote; record the block-mean
                # prediction so the accuracy curve reflects the actual
                # decision (per-pulse thresholding of an uncapped linear
                # output is an artifact once |W| grows).
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
    else:
        boost = (arm == 'delta_boost')
        high = (arm == 'delta_high')
        acc_sF = np.zeros(d)
        acc_oF = np.zeros(d)
        r_ema = 0.0
        for t in range(T):
            o = _sigmoid(float(w @ F[t]))
            preds[t] = o
            acc_sF += F[t]
            acc_oF += F[t] * o
            if (t + 1) % DB_K_PULSES == 0:
                n_blocks += 1
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                # eta: fixed for base/high; reward-driven for boost
                if high:
                    eta_b = DELTA_ETA * G_MAX
                elif boost:
                    r_ema = (1.0 - R_EMA_ALPHA) * r_ema + R_EMA_ALPHA * r
                    c = abs(r_ema)
                    g = 1.0 + G_K * max(0.0, C_THRESH - c)
                    eta_b = DELTA_ETA * min(G_MAX, g)
                    gain_sum += eta_b / DELTA_ETA
                    gain_max = max(gain_max, eta_b / DELTA_ETA)
                else:
                    eta_b = DELTA_ETA
                w += eta_b * (y_derived * acc_sF - acc_oF)
                w = _cap(w, W_MAX)
                acc_sF[:] = 0.0
                acc_oF[:] = 0.0

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'drift_binary', 'regime': regime, 'substrate': topo_name,
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000),
           'post_swap_acc': acc_median_in(acc_run, 1100, 1200),
           'steady_acc': acc_median_in(acc_run, 1800, 2000),
           'gain_mean': float(gain_sum / n_blocks) if n_blocks else float('nan'),
           'gain_max': float(gain_max),
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
        entry['gain_mean_mean'] = float(np.nanmean(
            np.array([r['gain_mean'] for r in rs], dtype=float)))
        entry['gain_max_mean'] = float(np.nanmean(
            np.array([r['gain_max'] for r in rs], dtype=float)))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 128)
    print("S47 SELF-CORRECTION GAIN ON ERROR-DRIVEN BASE (regime @1000, "
          "mean over seeds)")
    print("=" * 128)
    for regime in sorted(REGIMES.keys()):
        print(f"\n--- {regime} ---")
        print(f"  {'substrate':<17} | {'readout':<12} | {'pre':>6} | "
              f"{'post':>6} | {'steady':>6} | {'mean':>6} | {'g_mean':>6} | "
              f"{'g_max':>5}")
        for a in [x for x in agg if x['regime'] == regime]:
            print(f"  {a['substrate']:<17} | {a['readout']:<12} | "
                  f"{a['pre_swap_acc_mean']:>6.3f} | "
                  f"{a['post_swap_acc_mean']:>6.3f} | "
                  f"{a['steady_acc_mean']:>6.3f} | "
                  f"{a['mean_acc_all_mean']:>6.3f} | "
                  f"{a['gain_mean_mean']:>6.2f} | "
                  f"{a['gain_max_mean']:>5.1f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S47 self-correction gain "
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
                  'pre_swap_acc', 'post_swap_acc', 'steady_acc',
                  'gain_mean', 'gain_max']
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
        'r_ema_alpha': R_EMA_ALPHA, 'c_thresh': C_THRESH,
        'g_max': G_MAX, 'g_k': G_K,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'swap_block': SWAP_BLOCK, 'n_blocks': N_BLOCKS,
        'dt0_init': DT0_INIT, 'dt1_init': DT1_INIT,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    payload = _nan_to_null({'params': params, 'aggregates': agg,
                            'null_reason': ('non-finite aggregate values are '
                                            'written as null (undefined, e.g. '
                                            'an arm with no qualifying '
                                            'measurement)')})
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def _nan_to_null(obj):
    """Recursively map non-finite floats to None (strict-JSON safe).

    Bug fix 2026-09-09 (dedup P1-90): json.dump wrote bare NaN literals for
    undefined aggregates (e.g. gain_mean of an arm with no reward blocks),
    which is not valid JSON and is rejected by strict parsers.
    """
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_null(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
