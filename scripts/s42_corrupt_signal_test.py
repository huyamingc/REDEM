#!/usr/bin/env python3
"""
Self-evolution optimizes what it measures: the corrupted-reward boundary.
(REDEM S42; S39/S40/S41 follow-up)
=============================================================================
Type:           PAPER
Paper Section:  S3/S39/S40/S41 follow-up (self-evolution boundaries)
Experiment:     D4 under test -- a self-evolving meta-controller hill-climbs
                its own measured reward-rate; if that signal is corrupted
                (reward adversarially inverted over a window), the controller
                correctly learns to chase the corrupted signal, driving
                itself to the task-wrong but reward-"correct" hypothesis,
                then self-repairs when truthful reward resumes.

Mirror-impossibility preamble (proven algebraically, verified here):
  vote(-w) = 1 - vote(w) for hard votes, so r(-w) = -r(w) PER BLOCK and
  E[r](-w) = -E[r](w) exactly. Within the {w, -w} mirror hypothesis set a
  flip is ALWAYS beneficial whenever the current hypothesis is tempted
  (E[r] < 0). Hence a value-learning controller over {w, -w} can never
  learn "don't flip" (inhibition is impossible); it can only be misled by a
  corrupted measurement signal -- exactly the boundary tested here.

Environment (drift-binary, T=40000, single class-interval swap at block
1000, s39/s40 protocol):
  corruption window: blocks [1200, 1700) -- the reward r is inverted there
  (r_given = -r_truth) for the 'corrupt' arms. The swap recovery (block
  ~1040) is therefore clean, the corruption is a transient 500-block
  deception window, and blocks 1700+ recover again.

Arms (all on the same stream; is_corrupt toggles the reward inversion):
  meta_truth   : S41 meta_self, truthful reward (control).
  meta_corrupt : S41 meta_self, reward inverted in [1200,1700).
  passive_truth: S40 passive_flip (fixed margin), truthful reward (control).
  passive_corrupt: passive_flip, reward inverted in [1200,1700).

Predictions:
  P1: corrupt_acc (blocks 1250-1650, inside the deception window) collapses
      to ~anti-level (0.06-0.10) for both corrupt arms while staying ~0.9
      for the truthful arms -- the system chases the corrupted signal.
  P2: pre/post-swap and steady_final windows are unaffected (~0.9 all arms)
      -- the deception is transient and self-repaired when truth resumes.
  P3: meta v_final stays positive in ALL conditions (the value learning
      correctly reports "flipping improves the measured reward"); it cannot
      detect that the measurement itself is corrupted -- the D4 statement.

Output files:
  data/s42_corrupt_signal_v1.csv    (one row per run)
  data/s42_corrupt_signal_v1.json   (params + per-cell aggregates)

Usage: python s42_corrupt_signal_test.py [--quick]
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

# ========================== Fixed parameters (S3/S39/S40/S41-consistent) ==========================
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

SWAP_EVERY = 1000        # single swap at block 1000 (s39 protocol)
N_BLOCKS = 2000

# Corruption window (blocks, half-open [lo, hi))
CORRUPT_LO = 1200
CORRUPT_HI = 1700

# Fixed-policy arm (S40 passive_flip)
FIXED_MARGIN = 0.20
PROBE_EMA_ALPHA = 0.02
FLIP_CHECK = 20

# Meta-self arm (S41)
META_EMA_ALPHA = 0.02
META_V_INIT = 0.15
META_V_THRESH = 0.02
META_BETA = 0.5
META_W_EST = 40
META_WARMUP_BLOCKS = 100

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's42_corrupt_signal_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's42_corrupt_signal_v1.json')

CURVE_WINDOW = 200

ARMS = ['meta_truth', 'meta_corrupt', 'passive_truth', 'passive_corrupt']


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def acc_median_in(acc_run, b_lo, b_hi):
    """Median running accuracy over pulse range [b_lo*k, b_hi*k)."""
    lo, hi = b_lo * DB_K_PULSES, b_hi * DB_K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


# ========================== Single run ==========================

def run_single(args):
    """(topo_name, arm, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq, swap_blocks = gen_drift_binary(
        seed=seed_idx, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
        swap_every=SWAP_EVERY)
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
    n_blocks = 0
    n_flips = 0
    v_final = float('nan')

    is_corrupt = arm.endswith('corrupt')
    is_meta = arm.startswith('meta')

    # passive_fixed state (S40 mirror formulation)
    r_main_ema = 0.5
    r_opp_ema = 0.5
    blocks_since_check = 0

    # meta_self state
    r_ema = 0.0
    v_flip = META_V_INIT
    flip_pending = False
    blocks_since_flip = 0

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
            if is_corrupt and CORRUPT_LO <= b < CORRUPT_HI:
                r = -r

            if is_meta:
                r_ema = ((1.0 - META_EMA_ALPHA) * r_ema
                         + META_EMA_ALPHA * r)
                tf.consolidate(r, reset=True)
                if flip_pending:
                    blocks_since_flip += 1
                    if blocks_since_flip >= META_W_EST:
                        v_flip = ((1.0 - META_BETA) * v_flip
                                  + META_BETA * r_ema)
                        flip_pending = False
                if (not flip_pending and b >= META_WARMUP_BLOCKS
                        and blocks_since_check >= FLIP_CHECK):
                    blocks_since_check = 0
                    if r_ema < 0.0 and v_flip > META_V_THRESH:
                        tf.w = -tf.w
                        n_flips += 1
                        r_ema = 0.0
                        blocks_since_flip = 0
                        flip_pending = True
                        tf.e[:] = 0.0
            else:
                r_main_ema = ((1.0 - PROBE_EMA_ALPHA) * r_main_ema
                              + PROBE_EMA_ALPHA * r)
                r_opp_ema = -r_main_ema
                tf.consolidate(r, reset=True)
                if blocks_since_check >= FLIP_CHECK:
                    blocks_since_check = 0
                    if r_opp_ema > r_main_ema + FIXED_MARGIN:
                        tf.w = -tf.w
                        n_flips += 1
                        r_main_ema = 0.5
                        r_opp_ema = -0.5
                        tf.e[:] = 0.0
            v_final = v_flip if is_meta else float('nan')

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'drift_binary', 'substrate': topo_name, 'readout': arm,
           'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T),
           'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000),
           'post_swap_acc': acc_median_in(acc_run, 1100, 1200),
           'corrupt_acc': acc_median_in(acc_run, 1250, 1650),
           'steady_final_acc': acc_median_in(acc_run, 1800, 2000),
           'n_flips': n_flips, 'v_final': v_final,
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['substrate'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        topo, read = key
        entry = {'substrate': topo, 'readout': read, 'n_runs': len(rs)}
        fields = ['mean_acc_all', 'pre_swap_acc', 'post_swap_acc',
                  'corrupt_acc', 'steady_final_acc']
        for f in fields:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        fv = np.array([r['n_flips'] for r in rs], dtype=float)
        entry['n_flips_mean'] = float(np.nanmean(fv))
        if read.startswith('meta'):
            vv = np.array([r['v_final'] for r in rs], dtype=float)
            entry['v_final_mean'] = float(np.nanmean(vv))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 116)
    print("S42 CORRUPTED-SIGNAL BOUNDARY (drift_binary, swap@1000, "
          "corruption[1200,1700), mean over seeds)")
    print("=" * 116)
    print(f"  {'substrate':<17} | {'readout':<15} | {'pre':>6} | "
          f"{'post':>6} | {'CORRUPT':>7} | {'steady':>6} | "
          f"{'mean':>6} | {'flips':>5} | {'v':>6}")
    for a in agg:
        vf = f"{a.get('v_final_mean', float('nan')):.2f}"
        print(f"  {a['substrate']:<17} | {a['readout']:<15} | "
              f"{a['pre_swap_acc_mean']:>6.3f} | "
              f"{a['post_swap_acc_mean']:>6.3f} | "
              f"{a['corrupt_acc_mean']:>7.3f} | "
              f"{a['steady_final_acc_mean']:>6.3f} | "
              f"{a['mean_acc_all_mean']:>6.3f} | "
              f"{a['n_flips_mean']:>5.1f} | {vf:>6}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S42 corrupted-signal test "
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
        for arm in ARMS:
            for s in range(n_seeds):
                all_args.append((topo, arm, s))
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
                  't_total', 'runtime_s', 'mean_acc_all', 'pre_swap_acc',
                  'post_swap_acc', 'corrupt_acc', 'steady_final_acc',
                  'n_flips', 'v_final']
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
        'swap_every': SWAP_EVERY, 'n_blocks': N_BLOCKS,
        'corrupt_lo': CORRUPT_LO, 'corrupt_hi': CORRUPT_HI,
        'fixed_margin': FIXED_MARGIN, 'probe_ema_alpha': PROBE_EMA_ALPHA,
        'flip_check': FLIP_CHECK,
        'meta_ema_alpha': META_EMA_ALPHA, 'meta_v_init': META_V_INIT,
        'meta_v_thresh': META_V_THRESH, 'meta_beta': META_BETA,
        'meta_w_est': META_W_EST, 'meta_warmup_blocks': META_WARMUP_BLOCKS,
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
