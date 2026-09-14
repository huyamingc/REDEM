#!/usr/bin/env python3
"""
Meta-reward self-tuning: can a level-3 controller learn its own corrective
policy online, without a hand-set flip margin? (REDEM S41)
=============================================================================
Type:           PAPER
Paper Section:  S3/S39/S40 follow-up (self-evolution via hierarchical
                credit assignment)
Experiment:     The sign-ambiguity-dissolution theorem (D1): each meta-level
                learns from "reward-rate improvement" Delta-r, which is
                convention-free, so the meta-controller can learn when to
                invert the base hypothesis from its OWN experience instead
                of a fixed margin.

Arms (drift-binary block protocol, K=20 pulses/block):
  rmhl_oracle  : plain RMHL, no meta controller (anti-learning baseline).
  passive_fixed: S40 passive_flip with hand-set margin=0.20 (fixed policy).
  meta_self    : no hand-set margin. Tracks r_ema of the current hypothesis;
                 maintains a LEARNED value v of the flip action
                 (v <- (1-beta) v + beta * Delta-r measured over W_EST
                 blocks after each flip); flips only when r_ema < 0 AND
                 v > V_THRESH. v_init is optimistic (exploration), so the
                 controller tries the action and lets experience decide.

Environments (both drift-binary, T=40000):
  envA (swap_every=500) : swaps at blocks 500/1000/1500 -- flip helps.
  envB (swap_every=1e6) : no swap -- flips never help.

Claims under test:
  C1 (envA): meta_self recovers after every swap with NO hand-set margin,
             matching passive_fixed; its learned v converges positive.
  C2 (envB): meta_self performs no worse than rmhl_oracle (no self-sabotage
             in a stable stream).
  C3: learned v reflects the environment (positive in envA, neutral in
      envB) -- the policy is experience-driven, not hand-tuned.
      REVISED against the committed data (audit P1-103). envA SUPPORTS C3:
      n_flips = 3 in every seed and v_final = 0.5038. envB does NOT: the
      envB arm produced ZERO flips in ANY of the runs (n_flips = 0 for all
      20 meta_self envB runs, n_flips_mean = 0.0), so v_final for envB is
      frozen at its initial value 0.15 (v_final_mean = 0.1500 = META_V_INIT)
      and is NOT evidence that a neutral v was learned -- the value update
      v <- (1-beta)*v + beta*Delta-r runs ONLY after a flip, so with no flip
      the learned value is never evaluated. The envB arm is therefore
      UNINFORMATIVE for C3; the original "~neutral in envB" reading was an
      artifact of reporting the untouched optimistic initial value as a
      learned result. C3 is supported by envA only.

Output files:
  data/s41_meta_reward_v1.csv    (one row per run)
  data/s41_meta_reward_v1.json   (params + per-cell aggregates)

Usage: python s41_meta_reward_self_tuning.py [--quick]
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

# ========================== Fixed parameters (S3/S39/S40-consistent) ==========================
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

# Environments
SWAP_EVERY_A = 500     # envA: swaps at blocks 500, 1000, 1500
SWAP_EVERY_B = 10 ** 6  # envB: no swap within 2000 blocks

# Fixed-policy arm (S40 passive_flip)
FIXED_MARGIN = 0.20
PROBE_EMA_ALPHA = 0.02
FLIP_CHECK = 20

# Meta-self arm (level-3 value learning)
META_EMA_ALPHA = 0.02   # r_ema timescale (tau ~ 50 blocks)
META_V_INIT = 0.15      # optimistic flip-value init (exploration)
META_V_THRESH = 0.02    # act only if learned value is above this
META_BETA = 0.5         # value update gain
META_W_EST = 40         # improvement window (blocks) after a flip
META_WARMUP_BLOCKS = 100  # no flip decisions before this block index

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's41_meta_reward_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's41_meta_reward_v1.json')

CURVE_WINDOW = 200
N_BLOCKS = 2000


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
    """(topo_name, arm, env, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, env, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    swap_every = SWAP_EVERY_A if env == 'A' else SWAP_EVERY_B
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
                            elig_decay=RMHL_ELIG_DECAY, seed=seed_idx)
    preds = np.empty(T)
    rng = np.random.RandomState(seed_idx * 131 + 7)
    n_blocks = 0
    n_flips = 0
    v_final = float('nan')

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

            if arm == 'rmhl_oracle':
                tf.consolidate(r, reset=True)

            elif arm == 'passive_fixed':
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

            elif arm == 'meta_self':
                r_ema = ((1.0 - META_EMA_ALPHA) * r_ema
                         + META_EMA_ALPHA * r)
                tf.consolidate(r, reset=True)
                if flip_pending:
                    blocks_since_flip += 1
                    if blocks_since_flip >= META_W_EST:
                        delta = r_ema  # r_ema was reset to 0 at the flip
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

    # Post-loop readout of the meta_self state (audit P1-103): this statement
    # needs to run once, after the pulse loop. It was previously written inside
    # the loop body, where it re-assigned the same value on every iteration
    # (after the meta_self branch had already updated v_flip), which had no
    # numerical effect but was misleading.
    v_final = v_flip if arm == 'meta_self' else float('nan')

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'drift_binary', 'substrate': topo_name, 'readout': arm,
           'env': env, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'n_flips': n_flips, 'v_final': v_final,
           'runtime_s': time.time() - t0}

    if env == 'A':
        # per-swap windows (blocks): pre [s-400, s), post [s+100, s+400)
        pre_accs, post_accs = [], []
        for s in swap_blocks:
            pre_accs.append(acc_median_in(acc_run, max(0, s - 400), s))
            post_accs.append(acc_median_in(acc_run, s + 100, s + 400))
        res['pre_swap_mean'] = float(np.nanmean(pre_accs))
        res['post_swap_mean'] = float(np.nanmean(post_accs))
    else:
        res['pre_swap_mean'] = float('nan')
        res['steady_acc'] = acc_median_in(acc_run, N_BLOCKS - 400, N_BLOCKS)
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['substrate'], r['readout'], r['env']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        topo, read, env = key
        entry = {'substrate': topo, 'readout': read, 'env': env,
                 'n_runs': len(rs)}
        fields = ['mean_acc_all']
        if env == 'A':
            fields += ['pre_swap_mean', 'post_swap_mean']
        else:
            fields += ['steady_acc']
        for f in fields:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        fv = np.array([r['n_flips'] for r in rs], dtype=float)
        entry['n_flips_mean'] = float(np.nanmean(fv))
        if read == 'meta_self':
            vv = np.array([r['v_final'] for r in rs], dtype=float)
            entry['v_final_mean'] = float(np.nanmean(vv))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 110)
    print("S41 META-REWARD SELF-TUNING (drift_binary, mean over seeds)")
    print("=" * 110)
    for env in ['A', 'B']:
        print(f"\n--- env {env} "
              f"({'swaps at 500/1000/1500' if env == 'A' else 'no swap'}) ---")
        print(f"  {'substrate':<17} | {'readout':<14} | {'pre/post':>16} | "
              f"{'mean':>6} | {'flips':>5} | {'v_final':>7}")
        for a in [x for x in agg if x['env'] == env]:
            if env == 'A':
                pp = (f"{a['pre_swap_mean_mean']:.3f}/"
                      f"{a['post_swap_mean_mean']:.3f}")
            else:
                pp = f"{a['steady_acc_mean']:.3f} (steady)"
            vf = f"{a.get('v_final_mean', float('nan')):.3f}"
            print(f"  {a['substrate']:<17} | {a['readout']:<14} | {pp:>16} | "
                  f"{a['mean_acc_all_mean']:>6.3f} | "
                  f"{a['n_flips_mean']:>5.1f} | {vf:>7}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S41 meta-reward self-tuning "
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
    combos = []
    for topo in ['parallel', 'random_graph_k25']:
        combos.append((topo, 'rmhl_oracle', 'A'))
        combos.append((topo, 'passive_fixed', 'A'))
        combos.append((topo, 'meta_self', 'A'))
        combos.append((topo, 'rmhl_oracle', 'B'))
        combos.append((topo, 'meta_self', 'B'))
    all_args = [(t, arm, env, s) for t, arm, env in combos
                for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (combos={len(combos)}, seeds={n_seeds})")

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
    fieldnames = ['task', 'substrate', 'readout', 'env', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'mean_acc_all',
                  'pre_swap_mean', 'post_swap_mean', 'steady_acc',
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
        'swap_every_a': SWAP_EVERY_A, 'swap_every_b': SWAP_EVERY_B,
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
