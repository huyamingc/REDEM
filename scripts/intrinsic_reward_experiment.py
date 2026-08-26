#!/usr/bin/env python3
"""
Intrinsic-reward rescue experiment (M2 completion, REDEM S4).
=============================================================================
Type:           PAPER
Paper Section:  New-algorithm project Step S4
Experiment:     Does a task-agnostic intrinsic reward (feature novelty)
                rescue reward-modulated Hebbian learning (RMHL) from the
                class-inversion failure found in S3? Ablation over the
                intrinsic weight kappa_int and the task-reward frequency.

Design (drift_binary task, both substrates, 10 seeds):
  RMHL readout (S3 settings: eta=0.05, elig_decay=0.9) with block-end
  consolidation R = R_task (if reward due) + kappa_int * R_int, where
  R_int = feature novelty of the block against the running mean of past
  blocks (normalized to be comparable with the +/-1 task reward).
  reward_every = 1 : task reward every block (S3 setup)
  reward_every = 5 : task reward every 5th block (reward-free stretches;
                     the plan's "self-supervised sustenance" regime)

Hypothesis from the seed-0 pre-tune:
  * novelty intrinsic does NOT carry class-directional information, so it
    cannot rescue the inversion; at kappa_int >= 0.5 it distorts learning
    (the readout locks onto the post-swap mapping, inverting pre/post
    accuracy). Systematic negative result expected.
  * sparser task rewards reduce the wrong-direction update damage.

Output files:
  data/s4_intrinsic_reward_v1.csv    (one row per run)
  data/s4_intrinsic_reward_v1.json   (params + per-cell aggregates)

Usage: python intrinsic_reward_experiment.py [--quick]
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
from online_readout import running_mean_accuracy
from streaming_tasks import gen_drift_binary, DB_K_PULSES

# ========================== Fixed parameters ==========================
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
KAPPA_INT_GRID = [0.0, 0.1, 0.5, 2.0]
REWARD_EVERY_GRID = [1, 5]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's4_intrinsic_reward_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's4_intrinsic_reward_v1.json')


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def run_single(args):
    """(topo_name, kappa_int, reward_every, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, kappa_int, reward_every, seed_idx = args
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
    K = DB_K_PULSES

    # RMHL with block-end consolidation R = R_task + kappa_int * R_int
    rng = np.random.RandomState(seed_idx)
    w = rng.uniform(-0.5, 0.5, F.shape[1])
    e = np.zeros(F.shape[1])
    preds = np.empty(T)
    blk_sum = np.zeros(F.shape[1])
    n_seen = 0
    for t in range(T):
        z = float(F[t] @ w)
        o = 1.0 / (1.0 + np.exp(-z))
        preds[t] = o
        e = RMHL_ELIG_DECAY * e + F[t] * o
        if (t + 1) % K == 0:
            blk = preds[t - K + 1:t + 1]
            vote = float(np.mean(blk) > 0.5)
            r = 0.0
            if (n_seen + 1) % reward_every == 0:
                r = 1.0 if vote == target[t] else -1.0
            if kappa_int > 0.0 and n_seen > 0:
                bf = F[t - K + 1:t + 1].mean(axis=0)
                # RAW novelty (no normalization): the mean block feature
                # norm is ~sqrt(257) ~ 16, so kappa_int=0.1 gives an
                # intrinsic term comparable to the +/-1 task reward and
                # kappa_int >= 0.5 makes the intrinsic term dominant.
                # Both regimes are of interest (see S4 gate notes).
                nov = float(np.linalg.norm(bf - blk_sum / n_seen))
                r += kappa_int * nov
            w = w + RMHL_ETA * r * e
            e[:] = 0.0
            blk_sum += F[t - K + 1:t + 1].mean(axis=0)
            n_seen += 1

    acc_run = running_mean_accuracy(preds, target, 200)
    swap_pulse = int(swap_blocks[0] * K)
    pre = float(np.nanmedian(acc_run[swap_pulse - 2000:swap_pulse]))
    post = float(np.nanmedian(acc_run[swap_pulse + 4000:swap_pulse + 6000]))
    mean = float(np.nanmean(acc_run))

    return {'substrate': topo_name, 'kappa_int': float(kappa_int),
            'reward_every': int(reward_every), 'seed_idx': seed_idx,
            'n_units': N_UNITS, 't_total': int(T),
            'pre_steady_acc': pre, 'post_swap_acc': post,
            'mean_acc_all': mean, 'runtime_s': time.time() - t0}


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['substrate'], r['kappa_int'], r['reward_every']),
                          []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        sub, kint, rev = key
        entry = {'substrate': sub, 'kappa_int': kint, 'reward_every': rev,
                 'n_runs': len(rs)}
        for f in ['pre_steady_acc', 'post_swap_acc', 'mean_acc_all']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.mean(vals))
            entry[f + '_std'] = float(np.std(vals))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 100)
    print("S4 RESULTS (mean over seeds): RMHL + novelty intrinsic")
    print("=" * 100)
    for sub in ['parallel', 'random_graph_k25']:
        print(f"\n--- {sub} ---")
        print(f"  {'k_int':>6} | {'rew_every':>9} | {'pre_steady':>10} | "
              f"{'post_swap':>9} | {'mean_all':>8}")
        for a in [x for x in agg if x['substrate'] == sub]:
            print(f"  {a['kappa_int']:>6g} | {a['reward_every']:>9} | "
                  f"{a['pre_steady_acc_mean']:>10.3f} | "
                  f"{a['post_swap_acc_mean']:>9.3f} | "
                  f"{a['mean_acc_all_mean']:>8.3f}")


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S4 intrinsic-reward experiment "
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

    n_seeds = 2 if quick else N_SEEDS
    kints = [0.0, 0.5] if quick else KAPPA_INT_GRID
    revs = [1] if quick else REWARD_EVERY_GRID
    all_args = []
    for topo in ['parallel', 'random_graph_k25']:
        for kint in kints:
            for rev in revs:
                for s in range(n_seeds):
                    all_args.append((topo, kint, rev, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seeds={n_seeds})")

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
    fieldnames = ['substrate', 'kappa_int', 'reward_every', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'pre_steady_acc',
                  'post_swap_acc', 'mean_acc_all']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'kappa_random': KAPPA_RANDOM,
        'rmhl_eta': RMHL_ETA, 'rmhl_elig_decay': RMHL_ELIG_DECAY,
        'kappa_int_grid': KAPPA_INT_GRID, 'reward_every_grid': REWARD_EVERY_GRID,
        'intrinsic': 'novelty', 'n_seeds': n_seeds, 'quick': bool(quick),
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
