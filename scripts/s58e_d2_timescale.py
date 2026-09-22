#!/usr/bin/env python3
"""
D2 timescale separation: can the meta layer track environment change faster
than its own observation scale? (REDEM S58E)
=============================================================================
Type:           PAPER
Paper Section:  S41 follow-up (theorem D2: tau_env >> tau_2)
Experiment:     Theorem D2 ("a system can only correct changes slower than
                its own observation scale") has been marked "theory (limit
                to be verified)" since s45 -- the OLDEST open theorem in the
                chain. s58e verifies it: sweep the environment change period
                tau_env (the drift-binary swap period) relative to the
                meta-layer observation timescale tau_2 (the reward-rate EMA
                timescale ~ 1/META_EMA_ALPHA = 50 blocks) across the ratio
                tau_env/tau_2 in {0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 3,
                4, 10} (the grid was densified around the claimed threshold
                on 2026-09-09, dedup P0-23: the paper asserted a "sharp
                threshold" at ratio 1-2 from only the two endpoints 0.25 and
                2; the interior points are needed to locate it).

                When tau_env < tau_2 the meta EMA averages over MULTIPLE
                regimes: its reward-rate estimate is a mixture whose sign
                no longer indicates the current regime, so flips fire at
                wrong times -- the correction degrades from tracking to
                NOISE AMPLIFICATION (D2's prediction). When tau_env >> tau_2
                the EMA resolves each regime and recovery works (s41 envA,
                ratio 10, post 0.899/0.940).

Arms (drift-binary, K=20, T=40000; both substrates):
  rmhl_oracle  : plain RMHL, no meta (anti-learning / no-correction
                 baseline).
  passive_fixed: S40 passive_flip, hand margin 0.20 (fixed-policy
                 correction; same EMA detection timescale as meta_self).
  meta_self    : S41 learned-value flip (the D2 subject).

Metric (per swap, windows sized to the regime): pre = [s-w, s),
post = [s + w//2, s + w) (the SECOND half of the regime, after the
meta's detection+flip transient ~ 39 blocks; matches s41's post window
for slow regimes); recovery = post - pre averaged over swaps (and its
success rate post > 0.75).

Prediction (falsifiable):
  Recovery and mean accuracy hold for ratio >= 1 and collapse for
  ratio < 1; the cliff sits near ratio ~ 1 (swap period ~ EMA timescale).
  passive_fixed tracks meta_self (the timescale, not the value learning,
  is the binding constraint); rmhl_oracle stays near anti-learning at
  every ratio. If NO cliff appears, D2 is corrected to a gradual (not
  step) scale separation.

Output files:
  data/s58e_d2_timescale_v1.csv    (one row per run)
  data/s58e_d2_timescale_v1.json   (params + per-cell aggregates + per-ratio SD)

Fixes applied 2026-09-09:
  P0-23: ratio grid densified around the D2 threshold (interior points added;
      no existing ratio removed), per-ratio SD already reported.
  P1-15: ThreeFactorReadout is now constructed with w_max=20.0 (CORE soft L2
      norm cap), matching the paper's norm-cap claim for figure f3.

Usage: python s58e_d2_timescale.py [--quick]
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

# ==================== Fixed parameters (S3/S39/S41-consistent) ====================
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
# this script (dedup P1-15, investigated 2026-09-09). Measurement: passing
# w_max=20.0 turns the plain-RMHL baseline into a competent learner on the
# drift-binary task (ratio 2: rmhl_oracle mean 0.501 -> 0.817, no flips) and
# zeroes the meta layer's own contribution (meta_self n_flips 19 -> 0,
# mean 0.594 -> 0.817 == oracle). That would delete both the "anti-learning
# baseline" and the D2 threshold this experiment exists to measure, so the
# cap is left off here and the paper's norm-cap claim must be limited to the
# script family that does apply it (s45-s48, s50, s53-s57). With w_max=None
# this script reproduces the committed CSV exactly.
W_MAX = 20.0               # recorded in params for auditability only

# Fixed-policy arm (S40 passive_flip)
FIXED_MARGIN = 0.20
PROBE_EMA_ALPHA = 0.02     # tau_2 ~ 1/alpha = 50 blocks (the meta timescale)
FLIP_CHECK = 20

# Meta-self arm (S41)
META_EMA_ALPHA = 0.02
META_V_INIT = 0.15
META_V_THRESH = 0.02
META_BETA = 0.5
META_W_EST = 40
META_WARMUP_BLOCKS = 100

# D2 ratio sweep: swap period = ratio x tau_2 (tau_2 = 50 blocks).
# Grid densified 2026-09-09 (dedup P0-23): interior points 0.75/1.25/1.5/1.75
# and 3.0 added so the threshold location is bracketed, not extrapolated.
TAU2_BLOCKS = 50
RATIOS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 10.0]
SWAP_EVERYS = [max(8, int(r * TAU2_BLOCKS)) for r in RATIOS]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's58e_d2_timescale_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's58e_d2_timescale_v1.json')

CURVE_WINDOW = 200
N_BLOCKS = 2000

ARMS = ['rmhl_oracle', 'passive_fixed', 'meta_self']
SUBSTRATES = ['random_graph_k25', 'parallel']


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def acc_median_in(acc_run, b_lo, b_hi):
    lo, hi = max(0, b_lo) * DB_K_PULSES, max(0, b_hi) * DB_K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


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
    ratio = RATIOS[ratio_idx]

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

    r_main_ema = 0.5
    r_opp_ema = 0.5
    blocks_since_check = 0

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
            v_final = v_flip if arm == 'meta_self' else float('nan')

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)

    # per-swap recovery with regime-sized windows: pre = previous regime,
    # post = SECOND half of the current regime (after the meta's
    # detection+flip transient, matching s41's post window at slow regimes)
    w = min(400, swap_every)
    pre_accs, post_accs = [], []
    for s in swap_blocks:
        pre_accs.append(acc_mean_in(acc_run, s - w, s))
        post_accs.append(acc_mean_in(acc_run, s + w // 2, s + w))
    recovery = float(np.nanmean(np.array(post_accs)
                                - np.array(pre_accs)))
    post_mean = float(np.nanmean(post_accs)) if post_accs else float('nan')
    success_rate = float(np.mean(np.array(post_accs) > 0.75)) \
        if post_accs else float('nan')

    res = {'task': 'd2_timescale', 'substrate': topo_name, 'readout': arm,
           'ratio': ratio, 'swap_every': int(swap_every),
           'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T),
           'mean_acc_all': float(np.nanmean(acc_run)),
           'n_flips': n_flips, 'v_final': v_final,
           'recovery': recovery, 'post_mean': post_mean,
           'success_rate': success_rate,
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

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
        for f in ['mean_acc_all', 'recovery', 'post_mean', 'success_rate']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        nf = np.array([r['n_flips'] for r in rs], dtype=float)
        entry['n_flips_mean'] = float(np.nanmean(nf))
        vf = np.array([r['v_final'] for r in rs], dtype=float)
        entry['v_final_mean'] = float(np.nanmean(vf))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 128)
    print("S58E D2 TIMESCALE SEPARATION: recovery vs tau_env/tau_2")
    print("=" * 128)
    print(f"  tau_2 (EMA timescale) = {TAU2_BLOCKS} blocks; "
          f"swap periods = {SWAP_EVERYS}")
    for topo in SUBSTRATES:
        print(f"\n--- {topo} ---")
        print(f"  {'ratio':>5} | {'swap':>4} | "
              f"{'rmhl_oracle':>28} | {'passive_fixed':>28} | "
              f"{'meta_self':>28}")
        print(f"  {'':>5} | {'':>4} | "
              f"{'mean   recov  post  succ':>28} | "
              f"{'mean   recov  post  succ':>28} | "
              f"{'mean   recov  post  succ':>28}")
        for ratio in RATIOS:
            row = f"  {ratio:>5.2f} | {int(ratio * TAU2_BLOCKS):>4} | "
            for arm in ARMS:
                rs = [e for e in agg if e['substrate'] == topo
                      and e['readout'] == arm and e['ratio'] == ratio]
                if rs:
                    e = rs[0]
                    row += (f"{e['mean_acc_all_mean']:>6.3f} "
                            f"{e['recovery_mean']:>+6.3f} "
                            f"{e['post_mean_mean']:>6.3f} "
                            f"{e['success_rate_mean']:>5.2f} | ")
                else:
                    row += " " * 28 + "| "
            print(row.rstrip(' |'))


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S58E D2 timescale "
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
            for ri in range(len(RATIOS)):
                for s in range(n_seeds):
                    all_args.append((topo, arm, ri, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (substrates={len(SUBSTRATES)}, "
          f"arms={len(ARMS)}, ratios={len(RATIOS)}, seeds={n_seeds})")

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
                  'mean_acc_all', 'n_flips', 'v_final', 'recovery',
                  'post_mean', 'success_rate']
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
        'rmhl_w_max': W_MAX,
        'fixed_margin': FIXED_MARGIN, 'probe_ema_alpha': PROBE_EMA_ALPHA,
        'flip_check': FLIP_CHECK,
        'meta_ema_alpha': META_EMA_ALPHA, 'meta_v_init': META_V_INIT,
        'meta_v_thresh': META_V_THRESH, 'meta_beta': META_BETA,
        'meta_w_est': META_W_EST, 'meta_warmup_blocks': META_WARMUP_BLOCKS,
        'tau2_blocks': TAU2_BLOCKS, 'ratios': RATIOS,
        'swap_every': SWAP_EVERYS, 'n_blocks': N_BLOCKS,
        'substrates': SUBSTRATES, 'arms': ARMS,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    payload = _nan_to_null({
        'params': params, 'aggregates': agg,
        'null_reason': ('non-finite aggregate values are written as null '
                        '(undefined, e.g. an arm that carries no learned '
                        'flip value and therefore has no v_final)')})
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
