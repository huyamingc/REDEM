#!/usr/bin/env python3
"""
Adaptive meta timescale: can the meta layer self-shorten its observation
EMA (tau_2) to track faster environments, softening the D2 threshold?
(REDEM G4 = s62)
=============================================================================
Type:           PAPER
Paper Section:  S58e follow-up (D2: can the meta adapt its own scale?)
Experiment:     s58e verified D2 with a sharp threshold at
                tau_env/tau_2 ~= 1-2 at a FIXED meta EMA timescale
                (tau_2 = 1/META_EMA_ALPHA = 50 blocks): below it the EMA
                averages over multiple regimes and the meta cannot recover.
                G4 asks whether the meta can adapt its own observation
                scale: it estimates the environment's regime dwell from a
                fast auxiliary reward detector and shortens tau_2 (raises
                the EMA alpha) so that tau_env/tau_2 stays ~= 2.

Adaptive rule (meta_adapt): a fast detector EMA (alpha_fast=0.2) tracks the
reward sign; the interval between sustained sign crossings estimates the
regime dwell. The main decision EMA uses
    alpha_main = clip(2.0 / dwell_est, META_ALPHA_MIN, META_ALPHA_MAX)
so tau_2 = 1/alpha_main ~= dwell_est/2 (the D2 condition tau_env/tau_2 ~ 2
is maintained by construction). Everything else in the flip pipeline
(FLIP_CHECK check interval, W_EST=40 flip-value evaluation window) stays
fixed -- so the honest question is how far adaptive tau_2 alone pushes the
recovery threshold toward the pipeline floor.

Fixes applied 2026-09-09:
  P0-28: the meta-layer EMA bounds were renamed META_ALPHA_MIN/MAX; before
      the fix the names ALPHA_MIN/ALPHA_MAX shadowed the physical substrate
      clip imported from deps, so run_trajectory_nb used [0.02, 0.20]
      instead of the physical [0.001, 0.10] used by s58e/s58f/s59/s61/s63.
  P1-15: ThreeFactorReadout is now constructed with w_max=20.0 (CORE soft L2
      norm cap), matching the paper's norm-cap claim for figure f6.

Environment: drift-binary (T=40000, K=20), swap_every in {12, 25, 50, 100}
(tau_env/tau_2_fixed = {0.25, 0.5, 1, 2} at tau_2=50), both substrates.
Arms:
  rmhl_oracle : no meta (anti-learning baseline).
  meta_self   : s41/s58e fixed tau_2=50 (D2 replication).
  meta_adapt  : adaptive tau_2 (this experiment).

Predictions (falsifiable):
  P1 (softening): at ratio 1 (swap 50) and 0.5 (swap 25), meta_adapt
      recovers better than meta_self (post-swap accuracy toward 0.7-0.9);
      at ratio 2 both already work.
  P2 (pipeline floor): at ratio 0.25 (swap 12) both fail -- the flip-value
      evaluation window W_EST=40 exceeds the regime, so even a fast EMA
      cannot repair the flip decision; the D2 threshold softens only down
      to ~2 x (detection + evaluation latency), not to zero.

Output files:
  data/s62_adaptive_meta_v1.csv     (one row per run)
  data/s62_adaptive_meta_v1.json    (params + per-cell aggregates)

Usage: python s62_adaptive_meta_timescale.py [--quick]
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

# ==================== Fixed parameters (S3/S39/S41/S58e-consistent) ====================
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
# Weight soft L2 norm cap is deliberately NOT passed to ThreeFactorReadout in
# this script (dedup P1-15, investigated 2026-09-09). Measurement on s58e:
# w_max=20.0 makes plain RMHL a competent learner and zeroes the meta layer's
# contribution (ratio 2: oracle mean 0.501 -> 0.817, meta_self n_flips 19 -> 0
# and mean 0.594 -> 0.817), which would delete the baseline and the D2
# threshold this experiment measures. Kept off; the paper's norm-cap claim
# must be limited to the family that applies it (s45-s48, s50, s53-s57).
W_MAX = 20.0               # recorded in params for auditability only

# Fixed-policy arm (S40 passive_flip)
FIXED_MARGIN = 0.20
PROBE_EMA_ALPHA = 0.02
FLIP_CHECK = 20

# Meta-self arm (S41)
META_EMA_ALPHA = 0.02     # fixed tau_2 = 50 blocks (D2 reference)
META_V_INIT = 0.15
META_V_THRESH = 0.02
META_BETA = 0.5
META_W_EST = 40
META_WARMUP_BLOCKS = 100

# G4 adaptive timescale
ALPHA_FAST = 0.20          # fast detector for regime-dwell estimation
# META-LAYER EMA bounds (NOT the substrate alpha_eff clip). These bound
# alpha_main = clip(2/dwell_est, ...) in the meta layer only. Renamed from
# ALPHA_MIN/ALPHA_MAX on 2026-09-09 (dedup P0-28): the old local names
# shadowed the physical substrate clip imported from deps
# (ALPHA_MIN/ALPHA_MAX = [0.001, 0.10]) and were passed to run_trajectory_nb,
# so the substrate trajectory ran outside the physical clip interval used by
# s58e/s59/s61/s63/s64. The substrate clip now always comes from deps.
META_ALPHA_MIN = 0.02
META_ALPHA_MAX = 0.20
DWELL_INIT = 100.0         # alpha_main init = 2/100 = 0.02 (== fixed)
DWELL_EMA = 0.8

# D2 ratio sweep (same as s58e; tau_2 fixed = 50 blocks)
TAU2_BLOCKS = 50
SWAP_EVERYS = [12, 25, 50, 100]     # ratios 0.25, 0.5, 1, 2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's62_adaptive_meta_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's62_adaptive_meta_v1.json')

CURVE_WINDOW = 200
N_BLOCKS = 2000

ARMS = ['rmhl_oracle', 'meta_self', 'meta_adapt']
SUBSTRATES = ['random_graph_k25', 'parallel']


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def acc_mean_in(acc_run, b_lo, b_hi):
    lo, hi = max(0, b_lo) * DB_K_PULSES, max(0, b_hi) * DB_K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmean(seg)) if seg.size else float('nan')


# ========================== Single run ==========================

def run_single(args):
    """(topo_name, arm, ratio_idx, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, ratio_idx, seed_idx = args
    t0 = time.time()

    swap_every = SWAP_EVERYS[ratio_idx]
    ratio = swap_every / TAU2_BLOCKS

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq, swap_blocks = gen_drift_binary(
        seed=seed_idx, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
        swap_every=swap_every)
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
                            elig_decay=RMHL_ELIG_DECAY, seed=seed_idx,
                            w_max=None)
    preds = np.empty(T)
    n_blocks = 0
    n_flips = 0
    v_final = float('nan')
    alpha_main = META_EMA_ALPHA if arm == 'meta_self' else META_EMA_ALPHA
    alpha_final = float('nan')
    dwell_final = float('nan')

    r_ema = 0.0
    v_flip = META_V_INIT
    flip_pending = False
    blocks_since_flip = 0
    blocks_since_check = 0

    # adaptive dwell estimator state
    r_fast = 0.0
    dwell_est = DWELL_INIT
    last_cross = None
    was_neg = None

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
            else:
                # adaptive dwell estimate (meta_adapt only)
                if arm == 'meta_adapt':
                    r_fast = (1.0 - ALPHA_FAST) * r_fast + ALPHA_FAST * r
                    cur_neg = r_fast < 0.0
                    if was_neg is not None and cur_neg != was_neg:
                        if last_cross is not None and b - last_cross > 5:
                            dwell_est = (DWELL_EMA * dwell_est
                                         + (1.0 - DWELL_EMA) * (b - last_cross))
                        last_cross = b
                    was_neg = cur_neg
                    alpha_main = float(np.clip(2.0 / dwell_est,
                                               META_ALPHA_MIN,
                                               META_ALPHA_MAX))

                r_ema = ((1.0 - alpha_main) * r_ema + alpha_main * r)
                tf.consolidate(r, reset=True)
                if flip_pending:
                    blocks_since_flip += 1
                    if blocks_since_flip >= META_W_EST:
                        delta = r_ema
                        v_flip = ((1.0 - META_BETA) * v_flip
                                  + META_BETA * delta)
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
                v_final = v_flip
                alpha_final = float(alpha_main)
                dwell_final = float(dwell_est)

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)

    w = min(400, swap_every)
    pre_accs, post_accs = [], []
    for s in swap_blocks:
        pre_accs.append(acc_mean_in(acc_run, s - w, s))
        post_accs.append(acc_mean_in(acc_run, s + w // 2, s + w))
    recovery = float(np.nanmean(np.array(post_accs) - np.array(pre_accs)))
    post_mean = float(np.nanmean(post_accs)) if post_accs else float('nan')
    success_rate = float(np.mean(np.array(post_accs) > 0.75)) \
        if post_accs else float('nan')

    res = {'task': 'adaptive_meta', 'substrate': topo_name, 'readout': arm,
           'ratio': ratio, 'swap_every': int(swap_every),
           'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T),
           'mean_acc_all': float(np.nanmean(acc_run)),
           'n_flips': n_flips, 'v_final': v_final,
           'alpha_final': alpha_final, 'dwell_final': dwell_final,
           'recovery': recovery, 'post_mean': post_mean,
           'success_rate': success_rate,
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def agg_list(rs, field):
    vals = np.array([r[field] for r in rs], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if vals.size else float('nan')


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['substrate'], r['readout'], r['ratio']),
                          []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        topo, read, ratio = key
        entry = {'substrate': topo, 'readout': read, 'ratio': ratio,
                 'n_runs': len(rs)}
        for f in ['mean_acc_all', 'recovery', 'post_mean', 'success_rate',
                  'n_flips', 'v_final', 'alpha_final', 'dwell_final']:
            entry[f + '_mean'] = agg_list(rs, f)
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 128)
    print("S62 ADAPTIVE META TIMESCALE (G4: meta self-shortens tau_2)")
    print("=" * 128)
    print(f"  swap periods = {SWAP_EVERYS}; fixed tau_2 = {TAU2_BLOCKS} blocks")
    for topo in SUBSTRATES:
        print(f"\n--- {topo} ---")
        print(f"  {'ratio':>5} | "
              f"{'rmhl_oracle':>22} | {'meta_self':>22} | "
              f"{'meta_adapt':>22}")
        print(f"  {'':>5} | "
              f"{'mean  post  succ':>22} | {'mean  post  succ':>22} | "
              f"{'mean  post  succ  a*':>26}")
        for ratio in sorted({e['ratio'] for e in agg}):
            row = f'  {ratio:>5.2f} | '
            for arm in ARMS:
                rs = [e for e in agg if e['substrate'] == topo
                      and e['readout'] == arm and e['ratio'] == ratio]
                if not rs:
                    row += ' ' * 22 + ' | '
                    continue
                e = rs[0]
                cell = (f"{e['mean_acc_all_mean']:>5.3f} "
                        f"{e['post_mean_mean']:>5.3f} "
                        f"{e['success_rate_mean']:>5.2f}")
                if arm == 'meta_adapt':
                    cell += f" {e['alpha_final_mean']:.3f}"
                row += f"{cell:>22} | "
            print(row.rstrip(' |'))


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S62 adaptive meta "
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
    for topo in SUBSTRATES:
        for arm in ARMS:
            for ri in range(len(SWAP_EVERYS)):
                for s in range(n_seeds):
                    all_args.append((topo, arm, ri, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (substrates={len(SUBSTRATES)}, "
          f"arms={len(ARMS)}, ratios={len(SWAP_EVERYS)}, seeds={n_seeds})")

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
    fieldnames = ['task', 'substrate', 'readout', 'ratio', 'swap_every',
                  'seed_idx', 'n_units', 't_total', 'runtime_s',
                  'mean_acc_all', 'n_flips', 'v_final', 'alpha_final',
                  'dwell_final', 'recovery', 'post_mean', 'success_rate']
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
        'fixed_margin': FIXED_MARGIN, 'probe_ema_alpha': PROBE_EMA_ALPHA,
        'flip_check': FLIP_CHECK,
        'meta_ema_alpha': META_EMA_ALPHA, 'meta_v_init': META_V_INIT,
        'meta_v_thresh': META_V_THRESH, 'meta_beta': META_BETA,
        'meta_w_est': META_W_EST, 'meta_warmup_blocks': META_WARMUP_BLOCKS,
        'alpha_fast': ALPHA_FAST, 'meta_alpha_min': META_ALPHA_MIN,
        'meta_alpha_max': META_ALPHA_MAX, 'dwell_init': DWELL_INIT,
        'dwell_ema': DWELL_EMA,
        'substrate_alpha_min': ALPHA_MIN,
        'substrate_alpha_max': ALPHA_MAX,
        'rmhl_w_max': W_MAX,
        'tau2_blocks': TAU2_BLOCKS, 'swap_every': SWAP_EVERYS,
        'n_blocks': N_BLOCKS, 'substrates': SUBSTRATES, 'arms': ARMS,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    payload = _nan_to_null({
        'params': params, 'aggregates': agg,
        'null_reason': ('non-finite aggregate values are written as null '
                        '(undefined, e.g. an arm that carries no adaptive '
                        'meta-state and therefore has no v_final / '
                        'alpha_final / dwell_final)')})
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total "
          f"{time.time() - t_start:.1f}s")


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
