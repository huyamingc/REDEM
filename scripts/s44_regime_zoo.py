#!/usr/bin/env python3
"""
Regime zoo: scoping the mirror-flip sufficiency on the substrate task
family. (REDEM S44; closes the S3/S39-S43 generalization question)
=============================================================================
Type:           PAPER
Paper Section:  S3/S39-S43 follow-up (mirror-set sufficiency scope)
Experiment:     With the {w, -w} mirror set, E[r](-w) = -E[r](w) exactly:
                a flip recovers from ANY anti-correlated regime. This zoo
                maps the boundary empirically over four class-interval
                regime changes at block 1000 (all drift-binary, K=20,
                T=40000, s39 protocol):

  R1 full_swap      : dt0<->dt1 (10us<->60us). Inversion: flip recovers,
                      rmhl anti-learns (known result, reproduction).
  R2 partial_shift  : dt0: 10us -> 30us; dt1 stays 60us. Boundary moves
                      but does NOT invert: E[r](wA) ~ 0 (no temptation), so
                      no flip fires; recovery is slow re-learning for all.
  R3 compression    : dt0: 10us -> 25us, dt1: 60us -> 45us. Classes get
                      closer, still ordered: E[r](wA) stays positive; no
                      flip; mild accuracy dip.
  R4 near_inversion : dt0: 10us -> 55us, dt1: 60us -> 15us. Nearly a swap:
                      E[r](wA) < 0, flip fires, -wA is close to optimal.

Arms:
  rmhl        : plain RMHL, no meta layer.
  single_flip : S41 meta_self (learned-value flip) on one hypothesis.
  rls_fresh   : CONTROL -- a fresh supervised OnlineRLS retrained on the
                post-regime segment only (blocks 1000-2000). Probes whether
                the shifted task is learnable at all on this substrate (if
                yes, R2/R3 failure is the ONLINE LEARNER's collapse, not
                the task/substrate). Sees labels (capacity probe, not an
                online learner).
                Hyperparameters (RLS_FRESH_* module constants, recorded in
                the JSON under 'rls_fresh_hyperparams'):
                forgetting=0.999, init_cov=1.0, trace_cap=1e8, reg=1e-4.
                NOTE: the forgetting factor 0.999 differs from the 0.99
                used by the s47-and-later RLS family; this control arm
                belongs to the EARLIER RLS family, so the paper Methods
                statement "RLS forgetting factor 0.99" does NOT describe
                this arm.
                NOTE: because the arm is fitted on the post-1000 block
                segment only, its pre-swap window holds the constant-0.5
                placeholder rather than a measurement; the aggregate
                therefore reports pre_swap_acc_mean / pre_swap_acc_std as
                null and flags it with pre_swap_placeholder / pre_swap_note
                in the JSON (the CSV row still carries the raw column).

Predictions (status checked against the committed data, 10 seeds; the
refuted entries keep their original claim plus the measured outcome):
  P1 (R1, R4): single_flip recovers to ~0.9; rmhl anti-learns (R1) or
                suffers (R4) -- flip is the correct action.
                HOLDS: R1 steady 0.9010 (parallel) / 0.9425 (random) for
                single_flip vs 0.1105 / 0.0640 for rmhl; R4 steady 0.8770 /
                0.9435 vs 0.1280 / 0.0710.
  P2 (R2, R3): no flip fires (E[r] >= -margin/2); single_flip == rmhl --
                the mirror set has NO acceleration for boundary shifts
                (recovery is slow re-learning; this is the honest limit of
                the 2-arm set in 1D, not a failure of the flip).
                [REFUTED by own data: see data/s44_regime_zoo_v1.csv] a
                flip DOES fire in the boundary-shift regimes: n_flips_mean
                for single_flip is 2.7 in BOTH R2 and R3 (10 seeds, both
                substrates), so "no flip fires" is false; the arms still
                stay at chance there (steady 0.5080 for single_flip vs
                0.4970 for rmhl; post 0.4848 vs 0.5045), i.e. the mirror
                fires but does not accelerate a boundary shift.
  P3: flip count = 1 only in R1/R4; 0 in R2/R3.
                [REFUTED by own data: see data/s44_regime_zoo_v1.csv]
                n_flips_mean for single_flip = 1.0 in R1 and R4 but 2.7 in
                both R2 and R3 (10 seeds, both substrates).
  P4 (control): rls_fresh steady_acc >> both online arms in R2/R3 if the
                shifted task is substrate-learnable (learner collapse) --
                otherwise the task itself is the limit.
                HOLDS: rls_fresh steady 0.9880-1.0000 in R2/R3 versus
                0.4970-0.5080 for the two online arms.

Output files:
  data/s44_regime_zoo_v1.csv    (one row per run)
  data/s44_regime_zoo_v1.json   (params + per-cell aggregates)

Usage: python s44_regime_zoo.py [--quick]
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

# ========================== Fixed parameters (S3/S39-S43-consistent) ==========================
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

# rls_fresh control-arm hyperparameters (earlier RLS family; see docstring).
# The forgetting factor 0.999 is deliberately NOT the 0.99 of the s47-and-
# later RLS family -- this arm is the s44 capacity probe and stays at the
# value it was run with. Recorded in the params JSON as
# 'rls_fresh_hyperparams'.
RLS_FRESH_FORGETTING = 0.999
RLS_FRESH_INIT_COV = 1.0
RLS_FRESH_TRACE_CAP = 1e8
RLS_FRESH_REG = 1e-4

# Placeholder value written into the pre-swap window of the rls_fresh arm
# (the arm is fitted on blocks 1000-2000 only, so pulses before s0 are never
# predicted). It is NOT a measurement and is flagged as such in the output.
RLS_FRESH_PRE_PLACEHOLDER = 0.5

DT0_INIT, DT1_INIT = 10e-6, 60e-6
DT_MIN, DT_MAX = 4e-6, 120e-6

# Regime transforms applied to (dt0, dt1) at SWAP_BLOCK
REGIMES = {
    'R1_full_swap':      lambda a, b: (b, a),
    'R2_partial_shift':  lambda a, b: (30e-6, b),
    'R3_compression':    lambda a, b: (25e-6, 45e-6),
    'R4_near_inversion': lambda a, b: (55e-6, 15e-6),
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's44_regime_zoo_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's44_regime_zoo_v1.json')

CURVE_WINDOW = 200

ARMS = ['rmhl', 'single_flip', 'rls_fresh']


def gen_regime_stream(seed, regime):
    """drift-binary stream with the regime transform at SWAP_BLOCK.

    Same walk/label structure as gen_drift_binary (walk every 200 blocks,
    +/-3%), but the block-1000 change is the regime's transform instead of
    a plain interchange. Returns (dt_seq, target_seq).
    """
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

    n_flips = 0

    if arm == 'rls_fresh':
        from online_readout import OnlineRLS
        s0 = SWAP_BLOCK * DB_K_PULSES
        rls = OnlineRLS(F.shape[1], 1, forgetting=RLS_FRESH_FORGETTING,
                        init_cov=RLS_FRESH_INIT_COV,
                        trace_cap=RLS_FRESH_TRACE_CAP, reg=RLS_FRESH_REG)
        _, preds_post = rls.fit_stream(F[s0:], target[s0:, None],
                                       n_warmup=200)
        preds = np.empty(T)
        preds[:s0] = RLS_FRESH_PRE_PLACEHOLDER
        preds[s0:] = preds_post[:, 0]
    else:
        tf = ThreeFactorReadout(F.shape[1], mode='reward',
                                learning_rate=RMHL_ETA,
                                elig_decay=RMHL_ELIG_DECAY, seed=seed_idx)
        preds = np.empty(T)
        r_ema = 0.0
        v_flip = META_V_INIT
        flip_pending = False
        blocks_since_flip = 0
        blocks_since_check = 0
        for t in range(T):
            o = tf.predict(F[t])
            preds[t] = o
            tf.e = tf.elig_decay * tf.e + F[t] * o
            if (t + 1) % DB_K_PULSES == 0:
                blocks_since_check += 1
                b = (t + 1) // DB_K_PULSES - 1
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                if arm == 'rmhl':
                    tf.consolidate(r, reset=True)
                else:
                    r_ema = ((1.0 - PROBE_EMA_ALPHA) * r_ema
                             + PROBE_EMA_ALPHA * r)
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

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'drift_binary', 'regime': regime, 'substrate': topo_name,
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000),
           'post_swap_acc': acc_median_in(acc_run, 1100, 1200),
           'steady_acc': acc_median_in(acc_run, 1800, 2000),
           'n_flips': n_flips,
           # JSON-only flag (no CSV column): True when the recorded
           # pre_swap_acc is the constant placeholder of the rls_fresh arm.
           'pre_swap_placeholder': bool(arm == 'rls_fresh'),
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
        if read == 'rls_fresh':
            # The rls_fresh control arm is fitted on the post-swap segment
            # only, so its pre-swap window holds a constant-0.5 placeholder:
            # the aggregate there is undefined (null), not a measurement.
            entry['pre_swap_acc_mean'] = None
            entry['pre_swap_acc_std'] = None
            entry['pre_swap_placeholder'] = True
            entry['pre_swap_note'] = (
                'pre_swap window of rls_fresh is a constant-0.5 placeholder '
                '(the arm is fitted on blocks 1000-2000 only); it is not a '
                'measurement and must not be compared with or averaged '
                'against the measured arms.')
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 118)
    print("S44 REGIME ZOO (drift_binary, regime change @block 1000, "
          "mean over seeds)")
    print("=" * 118)
    for regime in sorted(REGIMES.keys()):
        print(f"\n--- {regime} ---")
        print(f"  {'substrate':<17} | {'readout':<12} | {'pre':>6} | "
              f"{'post':>6} | {'steady':>6} | {'mean':>6} | {'flips':>5}")
        for a in [x for x in agg if x['regime'] == regime]:
            pre = a['pre_swap_acc_mean']
            pre_txt = f"{pre:>6.3f}" if pre is not None else f"{'n/a':>6}"
            print(f"  {a['substrate']:<17} | {a['readout']:<12} | "
                  f"{pre_txt} | "
                  f"{a['post_swap_acc_mean']:>6.3f} | "
                  f"{a['steady_acc_mean']:>6.3f} | "
                  f"{a['mean_acc_all_mean']:>6.3f} | "
                  f"{a['n_flips_mean']:>5.1f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S44 regime zoo (quick={quick})")

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
    with Pool(min(cpu_count(), max(1, n_runs))) as pool:
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
                  'pre_swap_acc', 'post_swap_acc', 'steady_acc', 'n_flips']
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
        'probe_ema_alpha': PROBE_EMA_ALPHA, 'flip_check': FLIP_CHECK,
        'meta_v_init': META_V_INIT, 'meta_v_thresh': META_V_THRESH,
        'meta_beta': META_BETA, 'meta_w_est': META_W_EST,
        'meta_warmup_blocks': META_WARMUP_BLOCKS,
        'rls_fresh_hyperparams': {
            'forgetting': RLS_FRESH_FORGETTING,
            'init_cov': RLS_FRESH_INIT_COV,
            'trace_cap': RLS_FRESH_TRACE_CAP,
            'reg': RLS_FRESH_REG},
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
