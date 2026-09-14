#!/usr/bin/env python3
"""
Prediction-based credit test: can multi-view prediction repair an unreliable
reward signal? (REDEM S39; S3 follow-up)
=============================================================================
Type:           PAPER
Paper Section:  S3 follow-up (reward credit assignment)
Experiment:     Reward-modulated Hebbian readouts under oracle / delayed /
                noisy reward, with and without a multi-view prediction gate
                (fast readout vs slow consensus EMA). S3 drift-binary
                protocol and metrics are reused verbatim.

Arms (all online, drift-binary block protocol, K=20 pulses/block):
  rmhl_oracle      : S3 reproduction. +/-1 correctness reward at block end
                     (no class label used). Expected negative: cannot
                     recover from the class-interval inversion.
  rmhl_noisy       : reward flipped with probability NOISE_Q ("random
                     reward", the reward channel does not know correctness).
  rmhl_delayed     : reward for block b arrives DELAY_BLOCKS later and is
                     consolidated against the current (decayed / mixed)
                     eligibility trace ("late reward").
  pred_gated_oracle: RMHL + multi-view gate. A slow EMA of the readout
                     output is the second "view"; when fast and slow views
                     disagree, the external reward is distrusted (blocked)
                     and a consistency correction pulls the fast readout
                     toward the slow consensus. External reward oracle.
  pred_gated_noisy : same gate, external reward flipped with prob NOISE_Q.
  probe_directed   : ACTIVE multi-view probing. The learner alternates
                     between the current hypothesis and its exact opposite
                     ("different viewpoints" on the hidden label); the EMA
                     reward rate of each hypothesis is tracked, and when
                     the opposite hypothesis is consistently more rewarded,
                     the polarity flips. This is the "move the camera"
                     analog: hypothesis variation creates the baseline that
                     determines direction. Consolidation only on the main
                     hypothesis; explore blocks gather information only.
  rls_oracle       : error-driven RLS with per-pulse targets (gold
                     reference; it sees the true label, i.e. the "correct
                     weights" bound).

Hypothesis under test: the multi-view prediction principle ("a
representation that fits one view but not the others is exposed") can
verify an unreliable external reward and repair credit assignment; it
cannot conjure class-label direction out of a sign-free reward (no free
lunch: the label mapping is unidentifiable from features + +/-1 alone).

Output files:
  data/s39_prediction_reward_v2.csv    (one row per run)
  data/s39_prediction_reward_v2.json   (params + per-cell aggregates)

Usage: python s39_prediction_reward_test.py [--quick]
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
from online_readout import OnlineRLS, ThreeFactorReadout, running_mean_accuracy
from streaming_tasks import gen_drift_binary, DB_K_PULSES

# ========================== Fixed parameters (S3-consistent) ==========================
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
RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

# Corrupted-reward and gate hyperparameters (S39)
NOISE_Q = 0.30            # reward flip probability ("random reward")
DELAY_BLOCKS = 50         # reward delay in blocks (50 * 20 = 1000 pulses)
GATE_TAU_PULSES = 500     # slow consensus EMA timescale (metadata-like)
GATE_ETA_C = 0.02         # consistency-correction gain (fast -> slow)

# Active multi-hypothesis probing (S39 v2)
EXPLORE_P = 0.25          # fraction of blocks probing the opposite hypothesis
PROBE_EMA_ALPHA = 0.02    # reward-rate EMA (tau ~ 50 blocks)
FLIP_MARGIN = 0.20        # flip when opp-rate > main-rate + margin
FLIP_CHECK = 20           # flip-decision cadence (blocks)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's39_prediction_reward_v2.csv')
JSON_PATH = os.path.join(DATA_DIR, 's39_prediction_reward_v2.json')

CURVE_WINDOW = 200

ARMS = ['rmhl_oracle', 'rmhl_noisy', 'rmhl_delayed',
        'pred_gated_oracle', 'pred_gated_noisy',
        'probe_directed', 'rls_oracle']


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
    """(topo_name, readout, seed_idx) -> metrics dict (drift_binary only)."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, readout, seed_idx = args
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

    res = {'task': 'drift_binary', 'substrate': topo_name, 'readout': readout,
           'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T)}

    gated_blocks = 0
    n_blocks = 0
    n_flips = 0

    if readout == 'rls_oracle':
        rls = OnlineRLS(F.shape[1], 1, forgetting=RLS_FORGETTING,
                        init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                        reg=RLS_REG)
        _, preds = rls.fit_stream(F, target[:, None], n_warmup=200)
        preds = preds[:, 0]
    else:
        tf = ThreeFactorReadout(F.shape[1], mode='reward',
                                learning_rate=RMHL_ETA,
                                elig_decay=RMHL_ELIG_DECAY, seed=seed_idx)
        preds = np.empty(T)
        is_gated = readout.startswith('pred_gated')
        is_noisy = readout.endswith('noisy')
        is_delayed = (readout == 'rmhl_delayed')
        is_probe = (readout == 'probe_directed')
        rng = np.random.RandomState(seed_idx * 131 + 7)
        beta = 1.0 / GATE_TAU_PULSES
        o_slow = 0.5
        pending = {}  # arrival block index -> reward (delayed arm)
        r_main_ema = 0.5
        r_opp_ema = 0.5
        blocks_since_check = 0
        for t in range(T):
            o = tf.predict(F[t])
            preds[t] = o
            tf.e = tf.elig_decay * tf.e + F[t] * o
            if is_gated:
                o_slow = (1.0 - beta) * o_slow + beta * o
            if (t + 1) % DB_K_PULSES == 0:
                n_blocks += 1
                blocks_since_check += 1
                b = (t + 1) // DB_K_PULSES - 1
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                if is_noisy and rng.rand() < NOISE_Q:
                    r = -r
                if is_delayed:
                    pending[b + DELAY_BLOCKS] = r
                    if b in pending:
                        tf.consolidate(pending.pop(b), reset=True)
                elif is_gated:
                    s_vote = float(o_slow > 0.5)
                    agree = (vote == s_vote)
                    if agree:
                        tf.consolidate(r, reset=True)
                    else:
                        gated_blocks += 1
                        xbar = F[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
                        o_fast_bar = float(np.mean(blk))
                        tf.w += GATE_ETA_C * (o_slow - o_fast_bar) * xbar
                        tf.e[:] = 0.0
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
                        if r_opp_ema > r_main_ema + FLIP_MARGIN:
                            tf.w = -tf.w  # flip the hypothesis itself
                            n_flips += 1
                            r_main_ema = 0.5
                            r_opp_ema = 0.5
                            tf.e[:] = 0.0
                else:
                    tf.consolidate(r, reset=True)

    swap_pulse = int(swap_blocks[0] * DB_K_PULSES)
    pre, post, mean, acc_run = drift_metrics(preds, target, swap_pulse)
    late = float(np.nanmedian(acc_run[swap_pulse + 8000:swap_pulse + 10000]))
    res.update({'pre_steady_acc': pre, 'post_swap_acc': post,
                'mean_acc_all': mean, 'post_swap_late_acc': late,
                'gated_frac': (gated_blocks / n_blocks) if gated_blocks > 0
                              else float('nan'),
                'n_flips': n_flips})
    idxs = np.arange(0, T, 100)
    res['curve_x'] = idxs.astype(int)
    res['curve_y'] = acc_run[idxs]
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
        fields = ['pre_steady_acc', 'post_swap_acc', 'post_swap_late_acc',
                  'mean_acc_all']
        for f in fields:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        if read.startswith('pred_gated'):
            gv = np.array([r['gated_frac'] for r in rs], dtype=float)
            entry['gated_frac_mean'] = float(np.nanmean(gv))
        if read == 'probe_directed':
            fv = np.array([r['n_flips'] for r in rs], dtype=float)
            entry['n_flips_mean'] = float(np.nanmean(fv))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 108)
    print("S39 RESULTS (drift_binary, mean over seeds)")
    print("=" * 108)
    for topo in ['parallel', 'random_graph_k25']:
        rows = [a for a in agg if a['substrate'] == topo]
        print(f"\n--- {topo} ---")
        print(f"  {'readout':<18} | {'pre_steady':>10} | {'post_swap':>9} | "
              f"{'late':>7} | {'mean_all':>8} | {'gated':>6} | {'flips':>5}")
        for a in rows:
            gf = f"{a.get('gated_frac_mean', float('nan')):.2f}"
            nf = f"{a.get('n_flips_mean', 0.0):.0f}"
            print(f"  {a['readout']:<18} | {a['pre_steady_acc_mean']:>10.3f} | "
                  f"{a['post_swap_acc_mean']:>9.3f} | "
                  f"{a['post_swap_late_acc_mean']:>7.3f} | "
                  f"{a['mean_acc_all_mean']:>8.3f} | {gf:>6} | {nf:>5}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S39 prediction-reward test "
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
    for topo in ['parallel', 'random_graph_k25']:
        for read in ARMS:
            for s in range(n_seeds):
                all_args.append((topo, read, s))
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
    fieldnames = ['task', 'substrate', 'readout', 'seed_idx', 'n_units',
                  't_total', 'runtime_s', 'pre_steady_acc', 'post_swap_acc',
                  'post_swap_late_acc', 'mean_acc_all', 'gated_frac',
                  'n_flips']
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
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'rls_reg': RLS_REG,
        'noise_q': NOISE_Q, 'delay_blocks': DELAY_BLOCKS,
        'gate_tau_pulses': GATE_TAU_PULSES, 'gate_eta_c': GATE_ETA_C,
        'explore_p': EXPLORE_P, 'probe_ema_alpha': PROBE_EMA_ALPHA,
        'flip_margin': FLIP_MARGIN, 'flip_check': FLIP_CHECK,
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
