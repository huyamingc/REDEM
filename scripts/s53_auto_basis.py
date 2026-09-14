#!/usr/bin/env python3
"""
Autonomous basis generation with a WELL-CONDITIONED capacity sense: does the
system correctly judge whether its representation is sufficient, and does
expansion help where it is not? (REDEM S53, corrected design)
=============================================================================
Type:           PAPER
Paper Section:  S50/S51 correction + self-evolution endpoint
Experiment:     The s50/s51 circle results used ridge on the FULL 257-dim
                block-mean features with 500 training samples -- an
                underdetermined fit whose held-out accuracy (0.42) MISREADS
                the true capacity. A well-conditioned probe (ridge on the
                top-50 PCA components) shows the random_graph substrate's
                RAW features already decode the circle (0.80-0.88): the
                substrate's nonlinear expansion is sufficient, and the
                earlier "L3 trigger" and "basis expansion necessary" claims
                were probe artifacts. Representation-insufficiency is REAL
                only on the parallel (uncoupled) substrate (0.48 even
                conditioned), where quadratic features barely help (0.57).

                s53 asks the corrected self-evolution question: with a
                WELL-CONDITIONED capacity sense (PCA-50 ridge on recent
                block-mean features vs the system's own (r,vote)-reconstructed
                labels), does the system (a) correctly report "sufficient" on
                random_graph (no false L3 trigger, no unnecessary expansion),
                and (b) correctly report "insufficient" on parallel and does
                autonomous expansion to the quadratic basis actually help
                there?

Arms (static circle, K=20, T=40000; block-vote scoring; BOTH substrates):
  rls_linear_only : RLS on raw features, never expands.
  rls_quad_start  : RLS on quadratic features from block 0.
  rls_auto_expand : RLS + well-conditioned capacity sense; expands to
                    quadratic when capacity stays below threshold.
  delta_linear_only : delta (s50 reproduction; honest first-order).

Predictions (corrected):
  P1 (random_graph): capacity sense reads ~0.8 (sufficient) -> auto_expand
      does NOT trigger; rls_linear_only already recovers ~0.88-0.90 -- the
      substrate was never insufficient, no expansion needed.
  P2 (parallel): capacity sense reads ~0.48 (insufficient) -> auto_expand
      triggers and expands to quadratic, recovering partially (~0.55-0.65
      vs 0.48) -- expansion is a WEAK fix because the uncoupled substrate
      lacks the nonlinearity; the mechanism works but the substrate is the
      real bottleneck.
  P3: delta_linear_only stays ~ chance on both substrates (first-order
      cannot exploit the substrate's nonlinearity even where it exists).

Output files:
  data/s53_auto_basis_v1.csv    (one row per run)
  data/s53_auto_basis_v1.json   (params + per-cell aggregates)

Usage: python s53_auto_basis.py [--quick]
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
from online_readout import OnlineRLS, ridge_fit, running_mean_accuracy
from streaming_tasks import gen_nonlinear2d, DB_K_PULSES

# ==================== Fixed parameters (S3/S39-S52-consistent) ====================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

DELTA_ETA = 0.05
W_MAX = 20.0
RLS_FORGETTING = 0.99
RLS_INIT_COV = 1.0

# well-conditioned capacity sense
WIN_BLOCKS = 240
REFIT_EVERY = 20
CAP_N_PC = 50          # PCA components for the capacity probe
CAP_THRESH = 0.55
CAP_HOLD = 3
WARMUP_BLOCKS = 300

N_BLOCKS = 2000
BOUNDARY = 'circle'
SUBSTRATES = ['random_graph_k25', 'parallel']

# Raw-feature capacity probe (backfill, dedup P0-26): the paper (App. L.807)
# cites "random raw circle 0.80--0.88" against this script, but no raw-feature
# probe was ever computed or written here. This diagnostic evaluates the SAME
# well-conditioned probe protocol (top-50 PCA ridge on block-mean features,
# window WIN_BLOCKS, refit every REFIT_EVERY blocks from WARMUP_BLOCKS) on the
# RAW (non-expanded) features for every non-delta arm, on both substrates, and
# is written to the JSON as a NEW section ('raw_capacity_probe'); no existing
# JSON/CSV field is changed. Raw features are used regardless of the arm's
# active basis so the reading is substrate-attributable.
RAW_PROBE = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's53_auto_basis_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's53_auto_basis_v1.json')

CURVE_WINDOW = 200

ARMS = ['rls_linear_only', 'rls_quad_start', 'rls_auto_expand',
        'delta_linear_only']


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


def build_features(states, T, quadratic):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F = np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])
    if quadratic:
        d = F.shape[1]
        F = np.hstack([F, F[:, :d - 1] ** 2])
    return F


def capacity_pca50(Xw, yw):
    """Well-conditioned capacity probe: held-out ridge accuracy on the
    top-50 PCA components of a block-mean feature window."""
    Xa = np.asarray(Xw, dtype=np.float64)
    ya = np.asarray(yw, dtype=np.float64)
    n = Xa.shape[0]
    n_tr = max(20, n // 2)
    mu = Xa[:n_tr].mean(axis=0)
    sd = Xa[:n_tr].std(axis=0) + 1e-9
    Xs = (Xa - mu) / sd
    U, S, Vt = np.linalg.svd(Xs[:n_tr], full_matrices=False)
    k = min(CAP_N_PC, Vt.shape[0])
    Xp = Xs @ Vt[:k].T
    out = ridge_fit(Xp[:n_tr], ya[:n_tr], Xp[n_tr:], ya[n_tr:],
                    ridge_lambda=1.0)
    pred = out['pred_te']
    return float(np.mean((pred > 0.5).astype(np.float64) == ya[n_tr:]))


# ========================== Single run ==========================

def run_single(args):
    """(substrate, arm, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq, _, _ = gen_nonlinear2d(
        seed=seed_idx, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
        boundary=BOUNDARY)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F_lin = build_features(states, T, quadratic=False)
    F_quad = build_features(states, T, quadratic=True)

    auto = (arm == 'rls_auto_expand')
    quad_start = (arm == 'rls_quad_start')
    F = F_quad if quad_start else F_lin
    d = F.shape[1]

    preds = np.empty(T)
    switch_block = -1
    detect_cap = float('nan')
    n_expansions = 0
    rp_readings = []           # raw-feature capacity probe readings (P0-26)

    if arm == 'delta_linear_only':
        rng_w = np.random.RandomState(seed_idx * 97 + 3)
        w = rng_w.uniform(-0.5, 0.5, d)
        acc_sF = np.zeros(d)
        acc_oF = np.zeros(d)
        for t in range(T):
            o = _sigmoid(float(w @ F[t]))
            preds[t] = o
            acc_sF += F[t]
            acc_oF += F[t] * o
            if (t + 1) % DB_K_PULSES == 0:
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                w += DELTA_ETA * (y_derived * acc_sF - acc_oF)
                w = _cap(w, W_MAX)
                acc_sF[:] = 0.0
                acc_oF[:] = 0.0
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
    else:
        rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                        init_cov=RLS_INIT_COV)
        Xw = []
        yw = []
        cap_hold = 0
        # raw-feature probe state (dedup P0-26; independent of the sense)
        rp_Xw = []
        rp_yw = []
        rp_readings = []
        for t in range(T):
            preds[t] = float(rls.predict(F[t])[0])
            if (t + 1) % DB_K_PULSES == 0:
                b = (t + 1) // DB_K_PULSES - 1
                seg = F[t - DB_K_PULSES + 1:t + 1]
                x_blk = seg.mean(axis=0)
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                rls.update(x_blk, np.array([y_derived]))
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
                # raw-feature capacity probe (same window/refit protocol)
                if RAW_PROBE:
                    rp_Xw.append(
                        F_lin[t - DB_K_PULSES + 1:t + 1].mean(axis=0))
                    rp_yw.append(y_derived)
                    if len(rp_Xw) > WIN_BLOCKS:
                        rp_Xw.pop(0)
                        rp_yw.pop(0)
                    if (len(rp_Xw) >= 40 and b >= WARMUP_BLOCKS
                            and (b + 1) % REFIT_EVERY == 0):
                        rp_readings.append(
                            (b, capacity_pca50(rp_Xw, rp_yw)))
                # well-conditioned capacity sense (auto_expand only)
                if auto:
                    Xw.append(x_blk)
                    yw.append(y_derived)
                    if len(Xw) > WIN_BLOCKS:
                        Xw.pop(0)
                        yw.pop(0)
                    if (len(Xw) >= 40 and b >= WARMUP_BLOCKS
                            and (b + 1) % REFIT_EVERY == 0
                            and switch_block < 0):
                        cap = capacity_pca50(Xw, yw)
                        if cap < CAP_THRESH:
                            cap_hold += 1
                            if cap_hold >= CAP_HOLD:
                                switch_block = b
                                detect_cap = float(cap)
                                F = F_quad
                                d = F.shape[1]
                                rls = OnlineRLS(d, 1,
                                                forgetting=RLS_FORGETTING,
                                                init_cov=RLS_INIT_COV)
                                Xw = []
                                yw = []
                                cap_hold = 0
                                n_expansions += 1
                        else:
                            cap_hold = 0

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    if switch_block >= 0:
        post_acc = acc_median_in(acc_run, switch_block,
                                 min(N_BLOCKS, switch_block + 200))
    else:
        post_acc = float('nan')
    res = {'task': 'auto_basis', 'boundary': BOUNDARY,
           'substrate': topo_name, 'readout': arm,
           'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000),
           'steady_acc': acc_median_in(acc_run, 1800, 2000),
           'switch_block': switch_block, 'detect_cap': detect_cap,
           'n_expansions': n_expansions, 'post_switch_acc': post_acc,
           'runtime_s': time.time() - t0}
    raw_caps = np.array([c for _, c in rp_readings], dtype=float)
    raw_caps = raw_caps[np.isfinite(raw_caps)]
    res['raw_probe'] = {
        'features': 'raw (non-expanded) block-mean features',
        'protocol': 'top-50 PCA ridge, same window/refit cadence as the sense',
        'window_blocks': WIN_BLOCKS, 'refit_every': REFIT_EVERY,
        'warmup_blocks': WARMUP_BLOCKS, 'cap_n_pc': CAP_N_PC,
        'ridge_lambda': 1.0, 'boundary': BOUNDARY,
        'n_readings': int(raw_caps.size),
        'cap_mean': float(np.mean(raw_caps)) if raw_caps.size else float('nan'),
        'cap_min': float(np.min(raw_caps)) if raw_caps.size else float('nan'),
        'cap_max': float(np.max(raw_caps)) if raw_caps.size else float('nan'),
        'cap_std': float(np.std(raw_caps)) if raw_caps.size else float('nan'),
        'readings': [[int(b), float(c)] for b, c in rp_readings]}
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
        for f in ['mean_acc_all', 'pre_swap_acc', 'steady_acc',
                  'post_switch_acc']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        sw = np.array([r['switch_block'] for r in rs], dtype=float)
        entry['switch_block_mean'] = float(np.nanmean(sw))
        entry['switch_block_min'] = float(np.nanmin(sw))
        entry['switch_block_max'] = float(np.nanmax(sw))
        entry['n_detected'] = int(np.sum(np.array(
            [r['switch_block'] for r in rs]) >= 0))
        entry['detect_cap_mean'] = float(np.nanmean(
            np.array([r['detect_cap'] for r in rs], dtype=float)))
        entry['n_expansions_mean'] = float(np.nanmean(
            np.array([r['n_expansions'] for r in rs], dtype=float)))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 122)
    print("S53 AUTONOMOUS BASIS, WELL-CONDITIONED CAPACITY SENSE (circle)")
    print("=" * 122)
    for topo in SUBSTRATES:
        print(f"\n--- {topo} ---")
        print(f"  {'readout':<18} | {'pre':>6} | {'steady':>6} | "
              f"{'mean':>6} | {'post_sw':>6} | {'switch':>7} | "
              f"{'det_cap':>6} | {'det':>3} | {'exp':>3}")
        for a in [x for x in agg if x['substrate'] == topo]:
            print(f"  {a['readout']:<18} | {a['pre_swap_acc_mean']:>6.3f} | "
                  f"{a['steady_acc_mean']:>6.3f} | "
                  f"{a['mean_acc_all_mean']:>6.3f} | "
                  f"{a['post_switch_acc_mean']:>6.3f} | "
                  f"{a['switch_block_mean']:>7.1f} | "
                  f"{a['detect_cap_mean']:>6.3f} | "
                  f"{a['n_detected']:>3} | {a['n_expansions_mean']:>3.1f}")


def raw_probe_summary(results):
    """Aggregate the raw-feature capacity probe (dedup P0-26 backfill).

    Reports, per substrate, the distribution of the top-50 PCA ridge reading
    on RAW block-mean features (all non-delta arms and seeds pooled), plus the
    per-(substrate, arm, seed) means. This is the quantity the paper cites as
    "random raw circle 0.80--0.88" (App. L.807); until now s53 wrote no such
    field, so the number had no computable source.
    """
    per_sub = {}
    cells = []
    for topo in SUBSTRATES:
        vals = []
        for r in results:
            if r['substrate'] != topo or 'raw_probe' not in r:
                continue
            rp = r['raw_probe']
            cells.append({'substrate': topo, 'readout': r['readout'],
                          'seed_idx': r['seed_idx'],
                          'n_readings': rp['n_readings'],
                          'cap_mean': rp['cap_mean'],
                          'cap_min': rp['cap_min'],
                          'cap_max': rp['cap_max']})
            vals += [c for _, c in rp['readings']]
        v = np.array(vals, dtype=float)
        v = v[np.isfinite(v)]
        per_sub[topo] = {
            'n_readings': int(v.size),
            'cap_mean': float(np.mean(v)) if v.size else float('nan'),
            'cap_sd': float(np.std(v)) if v.size else float('nan'),
            'cap_min': float(np.min(v)) if v.size else float('nan'),
            'cap_max': float(np.max(v)) if v.size else float('nan'),
            'cap_p05': float(np.percentile(v, 5)) if v.size else float('nan'),
            'cap_p95': float(np.percentile(v, 95)) if v.size else float('nan'),
        }
    return {'features': 'raw (non-expanded) block-mean features',
            'protocol': ('held-out top-50 PCA ridge on block-mean features, '
                         'window %d blocks, refit every %d blocks from block '
                         '%d, ridge lambda 1.0, labels = the system\'s own '
                         '(r, vote)-reconstructed labels'
                         % (WIN_BLOCKS, REFIT_EVERY, WARMUP_BLOCKS)),
            'boundary': BOUNDARY, 'n_blocks': N_BLOCKS,
            'n_seeds': N_SEEDS, 'arms': [a for a in ARMS
                                         if a != 'delta_linear_only'],
            'paper_claim': ('App. L.807 "random raw circle 0.80--0.88" '
                            '(attributed to this script)'),
            'per_substrate': per_sub, 'per_cell': cells}


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S53 auto basis (quick={quick})")

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
            for s in range(n_seeds):
                all_args.append((topo, arm, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (substrates={len(SUBSTRATES)}, "
          f"arms={len(ARMS)}, seeds={n_seeds})")

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
    fieldnames = ['task', 'boundary', 'substrate', 'readout', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'mean_acc_all',
                  'pre_swap_acc', 'steady_acc', 'switch_block', 'detect_cap',
                  'n_expansions', 'post_switch_acc']
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
        'delta_eta': DELTA_ETA, 'w_max': W_MAX,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'win_blocks': WIN_BLOCKS, 'refit_every': REFIT_EVERY,
        'cap_n_pc': CAP_N_PC, 'cap_thresh': CAP_THRESH,
        'cap_hold': CAP_HOLD, 'warmup_blocks': WARMUP_BLOCKS,
        'n_blocks': N_BLOCKS, 'boundary': BOUNDARY,
        'substrates': SUBSTRATES,
        'n_seeds': n_seeds, 'quick': bool(quick),
        'raw_probe_enabled': RAW_PROBE,
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    raw_probe = raw_probe_summary(results)
    payload = _nan_to_null({
        'params': params, 'aggregates': agg,
        'raw_capacity_probe': raw_probe,
        'null_reason': ('non-finite aggregate values are written as null '
                        '(undefined, e.g. a cell where no expansion was '
                        'triggered, so post_switch_acc and detect_cap have '
                        'no measurement)')})
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print_table(agg)
    print("\n--- raw-feature capacity probe (top-50 PCA ridge, P0-26) ---")
    for topo, rp in raw_probe['per_substrate'].items():
        print(f"  {topo:<18} n={rp['n_readings']:>4} "
              f"mean={rp['cap_mean']:.3f} sd={rp['cap_sd']:.3f} "
              f"min={rp['cap_min']:.3f} max={rp['cap_max']:.3f} "
              f"p05-p95={rp['cap_p05']:.3f}-{rp['cap_p95']:.3f}")
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
