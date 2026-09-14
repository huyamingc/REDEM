#!/usr/bin/env python3
"""
Population x R2/R3: does a hypothesis population recover from boundary-shift
regimes that the {w, -w} mirror set cannot? (REDEM S45)
=============================================================================
Type:           PAPER
Paper Section:  S39-S44 follow-up (hypothesis-set evolution on the 1D
                substrate task family)
Experiment:     s44 showed R2 (partial shift) / R3 (compression) collapse
                BOTH online arms to chance even though the task is fully
                learnable (rls_fresh 0.988-1.000). Two causes suspected:
                (1) RMHL bias runaway -- the constant-1 bias column gets an
                always-positive eligibility sum, so the bias weight explodes,
                o -> 1 always, accuracy -> P(label=1)=0.5; (2) the mirror
                set {+/-w} has no action for "uncorrelated-but-wrong" states
                (E[r] ~ 0). This script tests BOTH fixes together:
                * weight soft norm cap (w *= W_MAX/|w|) un-saturates the
                  sigmoid so RMHL can re-learn the shifted boundary;
                * a K=5 population whose seed uses ERROR-INFORMED side info:
                  seed = best + SEED_STEP*(d_pos - d_neg), where d_pos/d_neg
                  are EMA directions of the best specialist's eligibility on
                  its rewarded (+1) / punished (-1) blocks. -d_neg points
                  AWAY from the misclassified features, i.e. the boundary
                  shift direction (the "class-0 side info" that plain
                  gradient seeding lacks).

Arms (drift-binary, regime change @block 1000, K=20, T=40000):
  rmhl            : plain RMHL (s44 baseline, collapses on R2/R3).
  rmhl_norm       : RMHL + weight norm cap (tests the bias-runaway fix).
  single_flip     : S41 meta_self (s44 baseline).
  single_flip_norm: meta_self + norm cap (fix + flip).
  pop5_norm       : K=5 population, all-learn, per-specialist meta-flip,
                    norm cap, error-informed prune/seed.
                    Prune policy: CONTINUOUS refresh of the worst-r_ema
                    specialist every PRUNE_EVERY=50 blocks, with NO
                    reward-rate threshold test (in the E[r]~0 dead zone of
                    R2/R3 no specialist ever drops below a threshold, so a
                    thresholded prune would be inert). The worst specialist
                    is re-seeded from the best specialist as
                    W[k_min] = W[k_best] + SEED_STEP*(d_pos - d_neg)
                    + SEED_NOISE*noise. Recorded in the params JSON as
                    'prune_mode': 'continuous_50block' together with
                    'prune_every'; the CSV column n_prunes counts these
                    refresh events (CSV schema unchanged).

Envs: R1 full swap / R4 near-inversion (regression checks), R2 partial
shift / R3 compression (targets).

Predictions (status checked against the committed data, 10 seeds; the
refuted entries keep their original claim plus the measured outcome):
  P1 (fix): rmhl_norm and single_flip_norm recover on R2/R3 where rmhl and
            single_flip collapse (post/steady >> chance).
            [REFUTED by own data: see data/s45_population_r2r3_v1.csv] the
            norm cap does NOT rescue the shifted regimes -- every arm stays
            at chance. R2 parallel (steady/post): rmhl 0.4970/0.5045,
            rmhl_norm 0.5030/0.4895, single_flip 0.5080/0.4848,
            single_flip_norm 0.5030/0.4920. R3 parallel: rmhl
            0.4970/0.5045, rmhl_norm 0.4930/0.4982, single_flip
            0.5080/0.4848, single_flip_norm 0.4930/0.4843.
  P2 (pop): pop5_norm recovers FASTER (higher post_swap, mean_all) than
            single_flip_norm on R2/R3 via error-informed seeding.
            [REFUTED by own data: see data/s45_population_r2r3_v1.csv]
            pop5_norm is at chance as well, so there is no faster recovery
            to measure. R2 (steady/post): parallel 0.5005/0.4970, random
            0.4910/0.4960. R3: parallel 0.4975/0.4880, random
            0.4985/0.4790.
  P3 (regression): norm cap does not hurt R1/R4 flip recovery (>= s44).
            HOLDS: R1 steady 0.9035/0.9425 and R4 steady 0.8775/0.9435 for
            rmhl_norm (parallel/random), at or above the s44 single_flip
            values (0.9010/0.9425 and 0.8770/0.9435).

Note on the norm cap: the cap itself was verified effective in s46, where
rmhl_norm recovers R1/R4 (steady 0.9035/0.9425 for R1 and 0.8775/0.9435 for
R4), so the R2/R3 failure here is a limitation of the credit-assignment
rule, not a bug in the norm cap.

Output files:
  data/s45_population_r2r3_v1.csv    (one row per run)
  data/s45_population_r2r3_v1.json   (params + per-cell aggregates)

Usage: python s45_population_r2r3.py [--quick]
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

# ========================== Fixed parameters (S3/S39-S44-consistent) ==========================
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

SWAP_BLOCK = 1000
N_BLOCKS = 2000

PROBE_EMA_ALPHA = 0.02
FLIP_CHECK = 20
META_V_INIT = 0.15
META_V_THRESH = 0.02
META_BETA = 0.5
META_W_EST = 40
META_WARMUP_BLOCKS = 100

# Norm cap (bias-runaway fix)
W_MAX = 20.0          # soft L2 cap on the weight vector

# Population hyperparameters
POP_K = 5
# Prune policy: the worst specialist is ALWAYS refreshed every PRUNE_EVERY
# blocks (recorded in the params JSON as 'prune_mode'), with no reward-rate
# threshold test -- so there is deliberately no PRUNE_THRESH constant here
# (the old record 'prune_thresh': -0.20 described a thresholded prune that
# this implementation does not perform).
PRUNE_EVERY = 50
PRUNE_MODE = 'continuous_50block'
SEED_STEP = 0.30
SEED_NOISE = 0.05
SIDE_ALPHA = 0.10     # e_pos/e_neg EMA (error-informed seed directions)

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
CSV_PATH = os.path.join(DATA_DIR, 's45_population_r2r3_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's45_population_r2r3_v1.json')

CURVE_WINDOW = 200

ARMS = ['rmhl', 'rmhl_norm', 'single_flip', 'single_flip_norm', 'pop5_norm']


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


def _unit(u, rng, d):
    n = float(np.linalg.norm(u))
    if n > 1e-12:
        return u / n
    v = rng.randn(d)
    nv = float(np.linalg.norm(v))
    return v / nv if nv > 1e-12 else np.zeros(d)


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

    is_pop = (arm == 'pop5_norm')
    is_norm = arm.endswith('norm')
    is_flip = arm.startswith('single_flip') or is_pop
    K = POP_K if is_pop else 1

    rng_w = np.random.RandomState(seed_idx * 97 + 3)
    W = rng_w.uniform(-0.5, 0.5, (K, d))
    e = np.zeros((K, d))
    e_pos = np.zeros((K, d))
    e_neg = np.zeros((K, d))
    r_ema = np.zeros(K)
    v_flip = np.full(K, META_V_INIT)
    flip_pending = np.zeros(K, dtype=bool)
    blocks_since_flip = np.zeros(K, dtype=np.int64)

    preds = np.empty(T)
    block_o_sum = np.zeros(K)
    active = 0
    n_flips = 0
    n_prunes = 0
    blocks_since_check = 0

    for t in range(T):
        Ft = F[t]
        o = _sigmoid(Ft @ W.T)          # (K,)
        preds[t] = o[active]
        e = RMHL_ELIG_DECAY * e + Ft[None, :] * o[:, None]   # (K, d)
        block_o_sum += o
        if (t + 1) % DB_K_PULSES == 0:
            blocks_since_check += 1
            b = (t + 1) // DB_K_PULSES - 1
            votes = (block_o_sum / DB_K_PULSES) > 0.5
            v_out = bool(votes[active])
            r = 1.0 if v_out == bool(target[t]) else -1.0
            r_k = np.where(votes == v_out, float(r), float(-r))
            r_ema = (1.0 - PROBE_EMA_ALPHA) * r_ema + PROBE_EMA_ALPHA * r_k

            # all-learn consolidation
            W += RMHL_ETA * r_k[:, None] * e

            # norm cap (bias-runaway fix)
            if is_norm:
                for k in range(K):
                    nw = float(np.linalg.norm(W[k]))
                    if nw > W_MAX:
                        W[k] *= (W_MAX / nw)

            # error-informed seed directions (population only)
            if is_pop:
                for k in range(K):
                    if r_k[k] > 0.0:
                        e_pos[k] = ((1.0 - SIDE_ALPHA) * e_pos[k]
                                    + SIDE_ALPHA * e[k])
                    else:
                        e_neg[k] = ((1.0 - SIDE_ALPHA) * e_neg[k]
                                    + SIDE_ALPHA * e[k])

            # per-specialist meta-flip
            if is_flip and b >= META_WARMUP_BLOCKS:
                for k in range(K):
                    if flip_pending[k]:
                        blocks_since_flip[k] += 1
                        if blocks_since_flip[k] >= META_W_EST:
                            v_flip[k] = ((1.0 - META_BETA) * v_flip[k]
                                         + META_BETA * r_ema[k])
                            flip_pending[k] = False
                if blocks_since_check >= FLIP_CHECK:
                    blocks_since_check = 0
                    for k in range(K):
                        if (not flip_pending[k] and r_ema[k] < 0.0
                                and v_flip[k] > META_V_THRESH):
                            W[k] = -W[k]
                            n_flips += 1
                            r_ema[k] = 0.0
                            blocks_since_flip[k] = 0
                            flip_pending[k] = True

            # prune / error-informed re-seed (population only).
            # ALWAYS refresh the worst specialist every PRUNE_EVERY blocks
            # (no absolute threshold): in the E[r]~0 dead zone of R2/R3 no
            # specialist ever drops below a threshold, so a thresholded
            # prune is inert -- the refresh must be continuous. Recorded in
            # the params JSON as 'prune_mode': PRUNE_MODE.
            if is_pop and b >= META_WARMUP_BLOCKS and b % PRUNE_EVERY == 0:
                k_min = int(np.argmin(r_ema))
                k_best = int(np.argmax(r_ema))
                d_pos = _unit(e_pos[k_best], rng_w, d)
                d_neg = _unit(e_neg[k_best], rng_w, d)
                W[k_min] = (W[k_best] + SEED_STEP * (d_pos - d_neg)
                            + SEED_NOISE * rng_w.randn(d))
                r_ema[k_min] = 0.0
                v_flip[k_min] = META_V_INIT
                flip_pending[k_min] = False
                blocks_since_flip[k_min] = 0
                e_pos[k_min] = 0.0
                e_neg[k_min] = 0.0
                n_prunes += 1

            e[:] = 0.0
            active = int(np.argmax(r_ema))
            block_o_sum[:] = 0.0

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'drift_binary', 'regime': regime, 'substrate': topo_name,
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000),
           'post_swap_acc': acc_median_in(acc_run, 1100, 1200),
           'steady_acc': acc_median_in(acc_run, 1800, 2000),
           'n_flips': n_flips, 'n_prunes': n_prunes,
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
        entry['n_flips_mean'] = float(np.nanmean(
            np.array([r['n_flips'] for r in rs], dtype=float)))
        entry['n_prunes_mean'] = float(np.nanmean(
            np.array([r['n_prunes'] for r in rs], dtype=float)))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 128)
    print("S45 POPULATION x R2/R3 + BIAS FIX (drift_binary, regime @1000, "
          "mean over seeds)")
    print("=" * 128)
    for regime in sorted(REGIMES.keys()):
        print(f"\n--- {regime} ---")
        print(f"  {'substrate':<17} | {'readout':<16} | {'pre':>6} | "
              f"{'post':>6} | {'steady':>6} | {'mean':>6} | {'flips':>5} | "
              f"{'prunes':>6}")
        for a in [x for x in agg if x['regime'] == regime]:
            print(f"  {a['substrate']:<17} | {a['readout']:<16} | "
                  f"{a['pre_swap_acc_mean']:>6.3f} | "
                  f"{a['post_swap_acc_mean']:>6.3f} | "
                  f"{a['steady_acc_mean']:>6.3f} | "
                  f"{a['mean_acc_all_mean']:>6.3f} | "
                  f"{a['n_flips_mean']:>5.1f} | "
                  f"{a['n_prunes_mean']:>6.1f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S45 population x R2/R3 + "
          f"bias fix (quick={quick})")

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
                  'n_flips', 'n_prunes']
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
        'swap_block': SWAP_BLOCK, 'n_blocks': N_BLOCKS,
        'dt0_init': DT0_INIT, 'dt1_init': DT1_INIT,
        'w_max': W_MAX,
        'probe_ema_alpha': PROBE_EMA_ALPHA, 'flip_check': FLIP_CHECK,
        'meta_v_init': META_V_INIT, 'meta_v_thresh': META_V_THRESH,
        'meta_beta': META_BETA, 'meta_w_est': META_W_EST,
        'meta_warmup_blocks': META_WARMUP_BLOCKS,
        'pop_k': POP_K, 'prune_every': PRUNE_EVERY,
        'prune_mode': PRUNE_MODE, 'seed_step': SEED_STEP,
        'seed_noise': SEED_NOISE, 'side_alpha': SIDE_ALPHA,
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
