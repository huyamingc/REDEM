#!/usr/bin/env python3
"""
Hypothesis generation on the circle: does basis expansion (quadratic state
features) recover what a single linear readout cannot, and does a population
with reward-rate selection add anything on top? (REDEM S51)
=============================================================================
CORRECTION (audit P1-102, 2026-09-10) -- SUPERSEDED CAPACITY READING
The capacity conclusion recorded by this script is OVERTURNED and is kept
below for provenance only. The probe readings written by THIS script (circle
block-mean ridge 0.420 on random_graph_k25, 0.430 on parallel) came from a
DIFFERENT, ILL-CONDITIONED probe protocol: a ridge fit on the FULL 257-dim
block-mean features (plus a pulse-level ridge) with a MUCH SMALLER number of
training samples than the corrected block-mean PCA ridge used in
scripts/s53_auto_basis.py (held-out top-50 PCA ridge on block-mean features,
240-block window, refit every 20 blocks from block 300). Under that corrected,
well-conditioned protocol the coupled random_graph_k25 substrate's RAW
(non-expanded) features DO carry the circle: cap_mean = 0.8414 with
cap_sd = 0.0570 over 2550 readings, versus cap_mean = 0.5468 with
cap_sd = 0.0511 on the uncoupled parallel substrate
(data/s53_auto_basis_v1.json, field raw_capacity_probe). The "circle is not
decodable / basis expansion is necessary / L3 fires" reading below is
therefore an artifact of the ill-conditioned probe and is SUPERSEDED by
scripts/s53_auto_basis.py; on the coupled substrate a linear readout on the
RAW features already decodes the circle, so basis expansion is a convenience
rather than a necessity there. Representation-insufficiency survives only on
the parallel (uncoupled) substrate.
=============================================================================
Type:           PAPER
Paper Section:  S50 follow-up (L3->L4 bridge: hypothesis generation)
Experiment:     s50 triggered L3 on the substrate: the circle boundary is NOT
                linearly decodable from substrate state (block-mean ridge
                0.420 random_graph / 0.430 parallel = chance; all online
                arms 0.47-0.51). The capacity probe (s49) already ruled out
                rotation speed; s50's circle is the first substrate-grounded
                representation-insufficiency.
                [SUPERSEDED by s53_auto_basis: see data/s53_auto_basis_v1.json]
                s51 asks the L3->L4 question:
                (a) does GENERATING a richer hypothesis class -- quadratic
                features x_i^2 appended to the readout input (a linear
                readout on a quadratic expansion is quadratic in the state,
                and the state maps (u0,u1) near-linearly, so x^2 carries
                u0^2,u1^2,u0*u1 -- the circle is a quadratic form) -- recover
                the circle? (b) does a POPULATION of such readouts with
                reward-rate selection (L4) beat a single enriched readout?

                Probe (seed 0, block-level ridge): circle random_graph
                0.420 -> 0.810 with quadratic features, parallel 0.430 ->
                0.570 -- basis expansion restores decodability on the
                coupled substrate.
                [SUPERSEDED by s53_auto_basis: see data/s53_auto_basis_v1.json]
                s51 tests whether ONLINE learners reach that ceiling and
                whether the population helps.

Arms (nonlinear2d circle, static over 2000 blocks, K=20, T=40000;
ALL arms scored by the protocol decision = block vote mean(block)>0.5):
  delta_linear : error-driven delta on linear features (s50 replication).
  delta_quad   : same rule on [x, x^2] quadratic features (L3 hypothesis
                 generation: single enriched readout, first-order optimizer).
  rls_quad     : block-level RLS on quadratic features -- hypothesis
                 generation (expanded basis) PLUS structural per-dimension
                 gain (the s47 criterion); the full fix.
  pop_quad     : K=5 delta readouts on quadratic features + per-specialist
                 reward-rate EMA, block vote from the argmax-r_ema
                 specialist, error-informed reseed of the worst specialist
                 (L4: hypothesis population + selection).

Envs: circle boundary x {parallel, random_graph_k25} x 10 seeds. Ridge
probes (block-level, seed 0): linear-feature and quadratic-feature ceilings.

Predictions:
  P1 (reproduce s50): delta_linear ~ chance on circle (both substrates).
  P2 (L3 = basis expansion, necessary): quadratic-feature ridge >> linear
      (0.81 vs 0.42 random; 0.57 vs 0.43 parallel) -- the circle becomes
      linearly decodable once the quadratic class is GENERATED.
      [SUPERSEDED by s53_auto_basis: see data/s53_auto_basis_v1.json]
  P3 (basis alone insufficient): delta_quad stays ~ chance -- a first-order
      optimizer with fixed eta and a tight norm cap cannot reach the
      quadratic solution within 2000 blocks; the expanded basis is
      necessary but NOT sufficient.
  P4 (representation x gain structure): rls_quad recovers the circle
      (toward / above the quadratic ridge ceiling on random_graph) --
      the s47 "structural gain" criterion is required on top of the new
      basis; population selection (pop_quad) cannot rescue the first-order
      failure, showing L4 selection is redundant on a static capacity wall
      (its value is under dynamic multi-regime stress, not here).

Output files:
  data/s51_quadratic_circle_v1.csv    (one row per run)
  data/s51_quadratic_circle_v1.json   (params + per-cell aggregates + probes)

Usage: python s51_quadratic_circle.py [--quick]
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
from online_readout import ridge_fit, running_mean_accuracy
from streaming_tasks import gen_nonlinear2d, DB_K_PULSES

# ==================== Fixed parameters (S3/S39-S50-consistent) ====================
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

PROBE_EMA_ALPHA = 0.02
POP_K = 5
PRUNE_EVERY = 100
SEED_STEP = 0.30
SEED_NOISE = 0.05

N_BLOCKS = 2000
BOUNDARY = 'circle'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's51_quadratic_circle_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's51_quadratic_circle_v1.json')

CURVE_WINDOW = 200

ARMS = ['delta_linear', 'delta_quad', 'rls_quad', 'pop_quad']


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


def build_features(states, T, quadratic=False):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F = np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])
    if quadratic:
        d = F.shape[1]
        F = np.hstack([F, F[:, :d - 1] ** 2])   # x^2, skip bias^2
    return F


def ridge_probe_block(F, target, b_lo, b_hi):
    n_blk = b_hi - b_lo
    Fb = np.empty((n_blk, F.shape[1]))
    for i, b in enumerate(range(b_lo, b_hi)):
        Fb[i] = F[b * DB_K_PULSES:(b + 1) * DB_K_PULSES].mean(axis=0)
    yb = target[b_lo * DB_K_PULSES:(b_hi * DB_K_PULSES):DB_K_PULSES]
    n_train = n_blk // 2
    out = ridge_fit(Fb[:n_train], yb[:n_train], Fb[n_train:], yb[n_train:],
                    ridge_lambda=1.0)
    pred = out['pred_te']
    return float(np.mean((pred > 0.5).astype(np.float64) == yb[n_train:]))


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

    dt_seq, target_seq, _, _ = gen_nonlinear2d(
        seed=seed_idx, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES,
        boundary=BOUNDARY)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)

    quad = (arm != 'delta_linear')
    F = build_features(states, T, quadratic=quad)
    d = F.shape[1]

    preds = np.empty(T)
    n_prunes = 0
    n_active_flips = 0

    if arm == 'rls_quad':
        from online_readout import OnlineRLS
        rls = OnlineRLS(d, 1, forgetting=0.99, init_cov=1.0)
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
    elif arm == 'pop_quad':
        K = POP_K
        rng_w = np.random.RandomState(seed_idx * 97 + 3)
        W = rng_w.uniform(-0.5, 0.5, (K, d))
        r_ema = np.zeros(K)
        e_pos = np.zeros((K, d))
        e_neg = np.zeros((K, d))
        acc_sF = np.zeros((K, d))
        acc_oF = np.zeros((K, d))
        active = 0
        for t in range(T):
            o = _sigmoid(F[t] @ W.T)          # (K,)
            preds[t] = o[active]
            acc_sF += F[t][None, :]
            acc_oF += F[t][None, :] * o[:, None]
            if (t + 1) % DB_K_PULSES == 0:
                b = (t + 1) // DB_K_PULSES - 1
                seg = F[t - DB_K_PULSES + 1:t + 1]
                blk_o = _sigmoid(seg @ W.T)          # (K_pulses, K) sigmoid outputs
                mean_o = blk_o.mean(axis=0)          # (K,) block-mean per specialist
                r_k = np.where((mean_o > 0.5) == target[t], 1.0, -1.0)
                r_ema = (1.0 - PROBE_EMA_ALPHA) * r_ema + PROBE_EMA_ALPHA * r_k
                # all-learn delta on reconstructed labels
                for k in range(K):
                    vk = float(mean_o[k] > 0.5)
                    rk = 1.0 if vk == target[t] else -1.0
                    y_derived = vk if rk > 0.0 else (1.0 - vk)
                    W[k] += DELTA_ETA * (y_derived * acc_sF[k] - acc_oF[k])
                    W[k] = _cap(W[k], W_MAX)
                    if rk > 0.0:
                        e_pos[k] = (1.0 - 0.1) * e_pos[k] + 0.1 * acc_sF[k]
                    else:
                        e_neg[k] = (1.0 - 0.1) * e_neg[k] + 0.1 * acc_sF[k]
                acc_sF[:] = 0.0
                acc_oF[:] = 0.0
                # prune / reseed worst specialist
                if b >= 100 and b % PRUNE_EVERY == 0:
                    k_min = int(np.argmin(r_ema))
                    k_best = int(np.argmax(r_ema))
                    dp = e_pos[k_best] / (np.linalg.norm(e_pos[k_best]) + 1e-12)
                    dn = e_neg[k_best] / (np.linalg.norm(e_neg[k_best]) + 1e-12)
                    W[k_min] = (W[k_best] + SEED_STEP * (dp - dn)
                                + SEED_NOISE * rng_w.randn(d))
                    r_ema[k_min] = 0.0
                    e_pos[k_min] = 0.0
                    e_neg[k_min] = 0.0
                    n_prunes += 1
                a_new = int(np.argmax(r_ema))
                if a_new != active:
                    n_active_flips += 1
                active = a_new
                preds[t - DB_K_PULSES + 1:t + 1] = mean_o[active]
    else:
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

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'nonlinear2d', 'boundary': BOUNDARY, 'substrate': topo_name,
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000),
           'steady_acc': acc_median_in(acc_run, 1800, 2000),
           'n_prunes': n_prunes, 'n_active_flips': n_active_flips,
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
        for f in ['pre_swap_acc', 'steady_acc', 'mean_acc_all']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        entry['n_prunes_mean'] = float(np.nanmean(
            np.array([r['n_prunes'] for r in rs], dtype=float)))
        entry['n_active_flips_mean'] = float(np.nanmean(
            np.array([r['n_active_flips'] for r in rs], dtype=float)))
        agg.append(entry)
    return agg


def print_table(agg, probes):
    print("\n" + "=" * 110)
    print("S51 HYPOTHESIS GENERATION ON THE CIRCLE (quadratic basis + pop)")
    print("=" * 110)
    print(f"\n  ridge ceilings (block-level, seed 0):")
    for k in sorted(probes.keys()):
        print(f"    {k}: {probes[k]:.3f}")
    print(f"\n  {'substrate':<17} | {'readout':<12} | {'pre':>6} | "
          f"{'steady':>6} | {'mean':>6} | {'prunes':>6} | {'flips':>5}")
    for a in agg:
        print(f"  {a['substrate']:<17} | {a['readout']:<12} | "
              f"{a['pre_swap_acc_mean']:>6.3f} | "
              f"{a['steady_acc_mean']:>6.3f} | "
              f"{a['mean_acc_all_mean']:>6.3f} | "
              f"{a['n_prunes_mean']:>6.1f} | "
              f"{a['n_active_flips_mean']:>5.1f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S51 quadratic circle "
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
    print(f"total runs: {n_runs} (substrates=2, arms={len(ARMS)}, "
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

    # ridge ceilings (block-level) on both feature spaces, seed 0
    probes = {}
    for topo in ['random_graph_k25', 'parallel']:
        tau_p = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=0)
        x0_p = preprogram_vec(ALPHA0, tau_p)
        ip_p, idx_p, wt_p, mode_p, kap_p = build_substrate(topo)
        dt_p, tgt_p, _, _ = gen_nonlinear2d(
            seed=0, n_blocks=N_BLOCKS, k_pulses=DB_K_PULSES, boundary=BOUNDARY)
        st_p, _, _ = run_trajectory_nb(
            x0_p, tau_p, dt_p, PW, ip_p, idx_p, wt_p, kap_p,
            ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode_p, 0)
        F_p = build_features(st_p, dt_p.shape[0], quadratic=False)
        probes[f"{topo}|linear"] = ridge_probe_block(F_p, tgt_p, 800, 1000)
        F_pq = build_features(st_p, dt_p.shape[0], quadratic=True)
        probes[f"{topo}|quad"] = ridge_probe_block(F_pq, tgt_p, 800, 1000)
    for k, v in probes.items():
        print(f"  ridge probe {k}: {v:.3f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'boundary', 'substrate', 'readout', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'mean_acc_all',
                  'pre_swap_acc', 'steady_acc', 'n_prunes', 'n_active_flips']
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
        'probe_ema_alpha': PROBE_EMA_ALPHA, 'pop_k': POP_K,
        'prune_every': PRUNE_EVERY, 'seed_step': SEED_STEP,
        'seed_noise': SEED_NOISE,
        'n_blocks': N_BLOCKS, 'boundary': BOUNDARY,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg, 'ridge_probe': probes},
                  f, indent=2)

    print_table(agg, probes)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
