#!/usr/bin/env python3
"""
Smooth rotation in 2D-encoded substrate input: who tracks a continuously
rotating boundary, and does the reward-driven self-correction loop (s47)
generalize out of the R2/R3 dead zone? (REDEM S48)
=============================================================================
Type:           PAPER
Paper Section:  S46/S47 follow-up (L2 smooth-rotation regime on the
                substrate, 2D input encoding)
Experiment:     s44/s45/s46 mapped the 1D interval task family completely
                (inversions R1/R4 recoverable, shifts R2/R3 collapse for the
                RMHL family, error-driven rule + structural gain recover
                everything). s47 added the self-correction criterion: scalar
                gain boosting is self-defeating in the E[r]~0 dead zone,
                per-dimension (RLS) gain tracks. This script makes the L2
                "smooth rotation" regime decidable on the REAL substrate via
                the new 2D input encoding (streaming_tasks.gen_rotation2d):
                a 2D input point u=(u0,u1)~U(0,1)^2 is interleaved into two
                channel sub-bands (even pulses ch0 -> [2,10]us, odd pulses
                ch1 -> [12,20]us) and the class boundary normal rotates
                smoothly from theta0 to theta0+dtheta after block
                ROT_AFTER=1000. The substrate memory (~17 pulses) makes every
                pulse's state carry BOTH channels, so a readout can extract
                a joint (u0,u1) decision.

                Questions:
                  Q1 (decidability): is the rotating 2D boundary linearly
                     separable in substrate features at every angle? (ridge
                     probe on a held-out window at fixed theta)
                  Q2 (who tracks): rls_online (structural gain) should track
                     the rotation best; delta_derived partially; rmhl_norm
                     poorly (rotation is not an inversion, so the mirror-flip
                     has no action -- the Hebbian eligibility self-aligns on
                     symmetric features, s43b).
                  Q3 (self-correction transfer): delta_boost keeps |r_ema|
                     high on rotation (accuracy stays high, unlike R2/R3's
                     dead zone), so the scalar boost stays OFF -- does it
                     match delta_base (no harm) while rls still wins?
                This is the substrate-decidable L2 test: rotation is neither
                R1/R4 (discrete inversion, flip works) nor R2/R3 (dead zone,
                1D), it is the continuous hypothesis-rotation case.

Arms (rotation2d, hold theta0 for blocks [0,1000), rotate linearly over
blocks [1000,1500), hold theta0+dtheta to block 2000, K=20, T=40000):
  rmhl_norm    : ThreeFactorReadout 'reward' + w_max=20 (CORE norm cap).
  delta_base   : error-driven, reconstruct y from (r, vote), fixed eta.
  delta_boost  : delta_base + reward-driven scalar gain (s47; C_THRESH=0.6).
  rls_online   : block-level RLS on reconstructed labels (s47; forgetting
                 0.99), scored by block vote (protocol decision).
  ridge_probe  : offline ridge on a fixed-theta window (supervised oracle,
                 decidability certification only -- reported, not a run arm).

Envs: R2D_rot90 (dtheta = pi/2) and R2D_rot180 (dtheta = pi), each x
{parallel, random_graph_k25} x 10 seeds.

Predictions:
  P1 (decidability): ridge_probe >= 0.9 at both early (theta0) and late
      (theta0+dtheta) angles on both substrates -- the rotating boundary is
      linearly separable in substrate features.
  P2 (tracking): rls_online post_swap/steady >> delta_base > rmhl_norm.
      rmhl_norm should stay near chance post-rotation start (flip-less
      Hebbian cannot rotate its boundary).
  P3 (self-corr.): delta_boost ~ delta_base everywhere (no dead zone here,
      boost off) -- the s47 negative result does NOT transfer harmfully.

Output files:
  data/s48_rotation2d_v1.csv    (one row per run)
  data/s48_rotation2d_v1.json   (params + per-cell aggregates + ridge probe)

Usage: python s48_rotation2d.py [--quick]
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
from online_readout import OnlineRLS, ThreeFactorReadout, ridge_fit, \
    running_mean_accuracy
from streaming_tasks import gen_rotation2d, DB_K_PULSES

# ==================== Fixed parameters (S3/S39-S47-consistent) ====================
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
W_MAX = 20.0            # weight norm cap (CORE w_max, s45/s46)
DELTA_ETA = 0.05
R_EMA_ALPHA = 0.02
C_THRESH = 0.60
G_MAX = 10.0
G_K = (G_MAX - 1.0) / C_THRESH

RLS_FORGETTING = 0.99
RLS_INIT_COV = 1.0

ROT_AFTER = 1000
ROT_SPAN = 500            # rotate over blocks [1000, 1500), then hold
N_BLOCKS = 2000
THETA0 = 0.6

ENVS = {
    'R2D_rot90': 0.5 * np.pi,
    'R2D_rot180': np.pi,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's48_rotation2d_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's48_rotation2d_v1.json')

CURVE_WINDOW = 200

ARMS = ['rmhl_norm', 'delta_base', 'delta_boost', 'rls_online']


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


def build_features(states, T):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F = np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])
    return F


# ========================== Ridge decidability probe ==========================

def ridge_probe(F, target, b_lo, b_hi):
    """Supervised ridge on blocks [b_lo, b_hi) (fixed theta): fit on the
    first half, evaluate held-out accuracy on the second half."""
    lo = b_lo * DB_K_PULSES
    hi = b_hi * DB_K_PULSES
    X, y = F[lo:hi], target[lo:hi]
    n_train = X.shape[0] // 2
    out = ridge_fit(X[:n_train], y[:n_train], X[n_train:], y[n_train:],
                    ridge_lambda=1.0)
    pred = out['pred_te']
    return float(np.mean((pred > 0.5).astype(np.float64) == y[n_train:]))


# ========================== Single run ==========================

def run_single(args):
    """(topo_name, arm, env, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, env, seed_idx = args
    t0 = time.time()
    dtheta = ENVS[env]

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq, theta_seq, _, _ = gen_rotation2d(
        seed=seed_idx, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
        theta0=THETA0, dtheta=dtheta, rot_after=ROT_AFTER, rot_span=ROT_SPAN)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F = build_features(states, T)
    d = F.shape[1]

    preds = np.empty(T)
    gain_sum = 0.0
    gain_max = 0.0
    n_blocks_run = 0

    if arm == 'rmhl_norm':
        tf = ThreeFactorReadout(d, mode='reward', learning_rate=RMHL_ETA,
                                elig_decay=RMHL_ELIG_DECAY, seed=seed_idx,
                                w_max=W_MAX)
        for t in range(T):
            o = tf.observe(F[t])
            preds[t] = o
            if (t + 1) % DB_K_PULSES == 0:
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                tf.consolidate(r, reset=True)
    elif arm == 'rls_online':
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
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
    else:
        boost = (arm == 'delta_boost')
        rng_w = np.random.RandomState(seed_idx * 97 + 3)
        w = rng_w.uniform(-0.5, 0.5, d)
        acc_sF = np.zeros(d)
        acc_oF = np.zeros(d)
        r_ema = 0.0
        for t in range(T):
            o = _sigmoid(float(w @ F[t]))
            preds[t] = o
            acc_sF += F[t]
            acc_oF += F[t] * o
            if (t + 1) % DB_K_PULSES == 0:
                n_blocks_run += 1
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                if boost:
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
    res = {'task': 'rotation2d', 'env': env, 'substrate': topo_name,
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000),
           'post_swap_acc': acc_median_in(acc_run, 1100, 1200),
           # post-rotation window (rotation ends at ROT_AFTER + ROT_SPAN):
           # the corrected "has the readout recovered" metric (dedup P1-102)
           'post_rot_acc': acc_median_in(acc_run, ROT_AFTER + ROT_SPAN + 200,
                                          ROT_AFTER + ROT_SPAN + 400),
           'steady_acc': acc_median_in(acc_run, 1800, 2000),
           'gain_mean': float(gain_sum / n_blocks_run) if n_blocks_run else float('nan'),
           'gain_max': float(gain_max),
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['env'], r['substrate'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        env, topo, read = key
        entry = {'env': env, 'substrate': topo, 'readout': read,
                 'n_runs': len(rs)}
        for f in ['pre_swap_acc', 'post_swap_acc', 'post_rot_acc', 'steady_acc',
                  'mean_acc_all']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        entry['gain_mean_mean'] = float(np.nanmean(
            np.array([r['gain_mean'] for r in rs], dtype=float)))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 122)
    print("S48 SMOOTH ROTATION IN 2D-ENCODED SUBSTRATE (rotation after "
          "block 1000, mean over seeds)")
    print("=" * 122)
    for env in sorted(ENVS.keys()):
        print(f"\n--- {env} (dtheta={ENVS[env]:.2f} rad) ---")
        print(f"  {'substrate':<17} | {'readout':<12} | {'pre':>6} | "
              f"{'post':>6} | {'steady':>6} | {'mean':>6} | {'g':>5}")
        for a in [x for x in agg if x['env'] == env]:
            print(f"  {a['substrate']:<17} | {a['readout']:<12} | "
                  f"{a['pre_swap_acc_mean']:>6.3f} | "
                  f"{a['post_swap_acc_mean']:>6.3f} | "
                  f"{a['steady_acc_mean']:>6.3f} | "
                  f"{a['mean_acc_all_mean']:>6.3f} | "
                  f"{a['gain_mean_mean']:>5.2f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S48 rotation2d (quick={quick})")

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
    for env in sorted(ENVS.keys()):
        for topo in ['parallel', 'random_graph_k25']:
            for arm in ARMS:
                for s in range(n_seeds):
                    all_args.append((topo, arm, env, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (envs={len(ENVS)}, arms={len(ARMS)}, "
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

    # decidability probe (supervised ridge at fixed theta)
    # Fixed 2026-09-09 (dedup P1-102): (a) the probe was computed for seed 0
    # only and the rot90/rot180 cells came out identical -- it is now computed
    # for all n_seeds seeds, and both the mean/sd and the per-seed values are
    # written; (b) post_swap_acc sampled blocks 1100-1200, which lies INSIDE
    # the rotation interval [1000, 1500) -- 'post_rot_acc' now samples after
    # the rotation ends, and the in-rotation window is kept under its old name
    # for traceability. The seed-0 probe values are retained explicitly.
    probe = {}
    for env in sorted(ENVS.keys()):
        for topo in ['parallel', 'random_graph_k25']:
            dtheta = ENVS[env]
            key = f"{env}|{topo}"
            early, late = [], []
            per_seed = {}
            for sd in range(n_seeds):
                tau_p = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=sd)
                x0_p = preprogram_vec(ALPHA0, tau_p)
                ip_p, idx_p, wt_p, mode_p, kap_p = build_substrate(topo)
                dt_p, tgt_p, _, _, _ = gen_rotation2d(
                    seed=sd, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
                    theta0=THETA0, dtheta=dtheta, rot_after=ROT_AFTER,
                    rot_span=ROT_SPAN)
                st_p, _, _ = run_trajectory_nb(
                    x0_p, tau_p, dt_p, PW, ip_p, idx_p, wt_p, kap_p,
                    ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode_p, 0)
                F_p = build_features(st_p, dt_p.shape[0])
                e_ = ridge_probe(F_p, tgt_p, 800, 1000)
                l_ = ridge_probe(F_p, tgt_p, 1800, 2000)
                early.append(e_)
                late.append(l_)
                per_seed[str(sd)] = {'early': e_, 'late': l_}
            probe[key + '_early'] = float(np.mean(early))
            probe[key + '_late'] = float(np.mean(late))
            probe[key + '_early_sd'] = float(np.std(early, ddof=1))
            probe[key + '_late_sd'] = float(np.std(late, ddof=1))
            probe[key + '_early_seed0'] = per_seed['0']['early']
            probe[key + '_late_seed0'] = per_seed['0']['late']
            probe[key + '_per_seed'] = per_seed
    for k, v in probe.items():
        if isinstance(v, float):
            print(f"  ridge probe {k}: {v:.3f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'env', 'substrate', 'readout', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'mean_acc_all',
                  'pre_swap_acc', 'post_swap_acc', 'post_rot_acc',
                  'steady_acc', 'gain_mean', 'gain_max']
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
        'w_max': W_MAX, 'delta_eta': DELTA_ETA,
        'r_ema_alpha': R_EMA_ALPHA, 'c_thresh': C_THRESH,
        'g_max': G_MAX, 'g_k': G_K,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'rot_after': ROT_AFTER, 'rot_span': ROT_SPAN, 'n_blocks': N_BLOCKS,
        'theta0': THETA0,
        'envs': {k: float(v) for k, v in ENVS.items()},
        'n_seeds': n_seeds, 'quick': bool(quick),
        'probe_seeds': f'0..{n_seeds - 1} (mean/sd + per-seed; _seed0 keys '
                       f'retain the historical seed-0 values)',
        'acc_windows': {'pre_swap_acc': [800, 1000],
                        'post_swap_acc': [1100, 1200],
                        'post_rot_acc': [ROT_AFTER + ROT_SPAN + 200,
                                         ROT_AFTER + ROT_SPAN + 400],
                        'steady_acc': [1800, 2000],
                        'note': ('post_swap_acc lies INSIDE the rotation '
                                 'interval [1000, 1500); post_rot_acc is the '
                                 'post-rotation recovery window, added '
                                 '2026-09-09 (dedup P1-102)')},
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    payload = _nan_to_null({
        'params': params, 'aggregates': agg, 'ridge_probe': probe,
        'null_reason': ('non-finite aggregate values are written as null '
                        '(undefined, e.g. an arm that never modulated its '
                        'gain and therefore has no gain_mean)')})
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
