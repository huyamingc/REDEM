#!/usr/bin/env python3
"""
Meta-flip ablation: is the complementary-hypothesis exploration load-bearing?
=============================================================================
Type:           PAPER
Paper Section:  S3/S39 follow-up (reward credit assignment)
Experiment:     probe_directed (S39 v2) vs passive_flip: flip the readout
                weight when the EMA reward rate of the CURRENT hypothesis
                drops below -margin/2, with NO complementary-hypothesis
                exploration blocks. Tests whether the "multi-view" probing
                apparatus in probe_directed is necessary for recovery, or
                whether flip-on-reward-rate-sign alone suffices.

Rationale: in probe_directed the opposite hypothesis votes exactly opposite
to the main one (v_eff = 1 - vote), so r_opp == -r_main deterministically
for a hard 0/1 vote. The flip condition r_opp_ema > r_main_ema + margin is
therefore algebraically equivalent to E[r_main] < -margin/2. If the
complementary exploration is NOT load-bearing, passive_flip (which only
thresholds the main reward-rate EMA and consolidates every block) must
recover to the same post-swap accuracy.

Arms (drift-binary block protocol, K=20 pulses/block, swap at block 1000,
T=40000 pulses):
  rmhl_oracle      : S3 reproduction, no flip. Expected anti-learning.
  probe_directed   : S39 v2 reproduction (25% explore blocks, flip when
                     r_opp_ema > r_main_ema + margin).
  passive_flip_m01 : every block consolidated; r_opp_ema := -r_main_ema
                     (deterministic mirror); flip when r_opp_ema >
                     r_main_ema + 0.10.
  passive_flip_m02 : same, margin 0.20.
  passive_flip_m03 : same, margin 0.30.

Output files:
  data/s40_meta_flip_v1.csv    (one row per run)
  data/s40_meta_flip_v1.json   (params + per-cell aggregates)

Usage: python s40_meta_flip_ablation.py [--quick]
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
from online_readout import ThreeFactorReadout, running_mean_accuracy
from streaming_tasks import gen_drift_binary, DB_K_PULSES

# ========================== Fixed parameters (S3/S39-consistent) ==========================
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

# Active / meta-flip hyperparameters (S39-consistent)
EXPLORE_P = 0.25          # probe_directed explore fraction
PROBE_EMA_ALPHA = 0.02    # reward-rate EMA (tau ~ 50 blocks)
FLIP_CHECK = 20           # flip-decision cadence (blocks)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's40_meta_flip_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's40_meta_flip_v1.json')

CURVE_WINDOW = 200

# (arm, margin) -- margin None for rmhl_oracle (no flip logic)
ARMS = [
    ('rmhl_oracle', None),
    ('probe_directed', 0.20),
    ('passive_flip', 0.10),
    ('passive_flip', 0.20),
    ('passive_flip', 0.30),
]


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
    """(topo_name, arm, margin, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, margin, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq, swap_blocks = gen_drift_binary(seed=seed_idx)
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

    tf = ThreeFactorReadout(F.shape[1], mode='reward',
                            learning_rate=RMHL_ETA,
                            elig_decay=RMHL_ELIG_DECAY, seed=seed_idx)
    preds = np.empty(T)
    rng = np.random.RandomState(seed_idx * 131 + 7)
    n_blocks = 0
    n_flips = 0
    first_flip_block = -1
    r_main_ema = 0.5
    r_opp_ema = 0.5
    blocks_since_check = 0

    is_probe = (arm == 'probe_directed')
    is_passive = (arm == 'passive_flip')

    for t in range(T):
        o = tf.predict(F[t])
        preds[t] = o
        tf.e = tf.elig_decay * tf.e + F[t] * o
        if (t + 1) % DB_K_PULSES == 0:
            n_blocks += 1
            blocks_since_check += 1
            b = (t + 1) // DB_K_PULSES - 1
            blk = preds[t - DB_K_PULSES + 1:t + 1]
            vote = float(np.mean(blk) > 0.5)
            r = 1.0 if vote == target[t] else -1.0
            if arm == 'rmhl_oracle':
                tf.consolidate(r, reset=True)
            elif is_probe:
                use_explore = rng.rand() < EXPLORE_P
                if use_explore:
                    v_eff = 1.0 - vote
                    r_p = 1.0 if v_eff == target[t] else -1.0
                    r_opp_ema = ((1.0 - PROBE_EMA_ALPHA) * r_opp_ema
                                 + PROBE_EMA_ALPHA * r_p)
                    tf.e[:] = 0.0  # information only; no consolidation
                else:
                    r_p = 1.0 if vote == target[t] else -1.0
                    r_main_ema = ((1.0 - PROBE_EMA_ALPHA) * r_main_ema
                                  + PROBE_EMA_ALPHA * r_p)
                    tf.consolidate(r_p, reset=True)
                if blocks_since_check >= FLIP_CHECK:
                    blocks_since_check = 0
                    if r_opp_ema > r_main_ema + margin:
                        tf.w = -tf.w
                        n_flips += 1
                        if first_flip_block < 0:
                            first_flip_block = b
                        r_main_ema = 0.5
                        r_opp_ema = 0.5
                        tf.e[:] = 0.0
            elif is_passive:
                r_main_ema = ((1.0 - PROBE_EMA_ALPHA) * r_main_ema
                              + PROBE_EMA_ALPHA * r)
                r_opp_ema = -r_main_ema  # deterministic mirror of the vote
                tf.consolidate(r, reset=True)
                if blocks_since_check >= FLIP_CHECK:
                    blocks_since_check = 0
                    if r_opp_ema > r_main_ema + margin:
                        tf.w = -tf.w
                        n_flips += 1
                        if first_flip_block < 0:
                            first_flip_block = b
                        r_main_ema = 0.5
                        r_opp_ema = -0.5
                        tf.e[:] = 0.0

    swap_block = int(swap_blocks[0])
    swap_pulse = swap_block * DB_K_PULSES
    pre, post, mean, acc_run = drift_metrics(preds, target, swap_pulse)
    late = float(np.nanmedian(acc_run[swap_pulse + 8000:swap_pulse + 10000]))
    res = {'task': 'drift_binary', 'substrate': topo_name, 'readout': arm,
           'margin': margin, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T),
           'pre_steady_acc': pre, 'post_swap_acc': post,
           'post_swap_late_acc': late, 'mean_acc_all': mean,
           'n_flips': n_flips, 'first_flip_block': first_flip_block,
           'flip_latency_blocks': (first_flip_block - swap_block
                                   if first_flip_block >= 0 else -1),
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['task'], r['substrate'], r['readout'],
                           r['margin']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        task, topo, read, margin = key
        entry = {'task': task, 'substrate': topo, 'readout': read,
                 'margin': margin, 'n_runs': len(rs)}
        fields = ['pre_steady_acc', 'post_swap_acc', 'post_swap_late_acc',
                  'mean_acc_all']
        for f in fields:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        if read != 'rmhl_oracle':
            fv = np.array([r['n_flips'] for r in rs], dtype=float)
            entry['n_flips_mean'] = float(np.nanmean(fv))
            lat = np.array([r['flip_latency_blocks'] for r in rs], dtype=float)
            pos = lat[lat >= 0]
            entry['flip_latency_mean'] = float(np.nanmean(pos)) if pos.size else float('nan')
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 120)
    print("S40 META-FLIP ABLATION (drift_binary, mean over seeds)")
    print("=" * 120)
    for topo in ['parallel', 'random_graph_k25']:
        rows = [a for a in agg if a['substrate'] == topo]
        print(f"\n--- {topo} ---")
        print(f"  {'readout':<17} | {'margin':>6} | {'pre':>6} | "
              f"{'post':>6} | {'late':>6} | {'mean':>6} | {'flips':>5} | "
              f"{'latency_blk':>12}")
        for a in rows:
            mg = f"{a['margin']:.2f}" if a['margin'] is not None else '  -'
            nf = f"{a.get('n_flips_mean', 0.0):.2f}"
            lt = f"{a.get('flip_latency_mean', float('nan')):>12.1f}" if a.get(
                'flip_latency_mean') is not None and a.get('flip_latency_mean') == a.get('flip_latency_mean') else "          nan"
            print(f"  {a['readout']:<17} | {mg:>6} | "
                  f"{a['pre_steady_acc_mean']:>6.3f} | "
                  f"{a['post_swap_acc_mean']:>6.3f} | "
                  f"{a['post_swap_late_acc_mean']:>6.3f} | "
                  f"{a['mean_acc_all_mean']:>6.3f} | {nf:>5} | {lt}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S40 meta-flip ablation "
          f"(quick={quick})")

    # numba warmup (identical to s39)
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
    for topo in ['parallel', 'random_graph_k25']:
        for arm, margin in ARMS:
            for s in range(n_seeds):
                all_args.append((topo, arm, margin, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (arms={len(ARMS)}, seeds={n_seeds})")

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
    fieldnames = ['task', 'substrate', 'readout', 'margin', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'pre_steady_acc',
                  'post_swap_acc', 'post_swap_late_acc', 'mean_acc_all',
                  'n_flips', 'first_flip_block', 'flip_latency_blocks']
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
        'explore_p': EXPLORE_P, 'probe_ema_alpha': PROBE_EMA_ALPHA,
        'flip_check': FLIP_CHECK, 'arms': ARMS,
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
