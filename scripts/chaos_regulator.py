#!/usr/bin/env python3
"""
Chaos regulator homeostat (M5, REDEM S6).
=============================================================================
Type:           PAPER
Paper Section:  New-algorithm project Step S6 (see NEW_ALGORITHM_PLAN.md)
Experiment:     Does a homeostatic coupling regulator (kappa adjusted online
                to hold a cheap activity proxy near its nominal target) keep
                the substrate in the memory-rich regime under environmental
                disturbance, and recover faster than a fixed coupling?

Disturbances (applied at pulse index T_DISTURB for Part 2, or from the
start for Part 1):
  tau_drift   : all unit time constants scaled by 1.5 (temperature drift)
  edge_prune  : 40% of the random-graph coupling edges removed (damage)
  noise       : readout feature noise sigma = 0.10 * feature std
  none        : nominal environment (control)

Part 1 (proxy validation, disturbed world from the start):
  Sweep kappa in {15, 25, 40} x disturbance; measure held-out memory
  capacity MC_te (Jaeger, 70/30 split) and the CHEAP proxy
  S = running std of the population-mean current ratio (window 200).
  Verifies MC_te peaks near a consistent S* band across disturbances.

Part 2 (online homeostat, Mackey-Glass task, 21k pulses):
  arm 'fixed'    : kappa = 25 throughout
  arm 'regulated': every FTLE_EVERY pulses, estimate the finite-time
        Lyapunov exponent on a short Benettin pair window with the CURRENT
        kappa, then kappa += eta_lambda * clip(lambda_target - lam, -1, 1),
        clipped to [1, 60]. lambda_target = -0.02 (slightly ordered, the
        memory-rich side per S1: held-out MC peaks just before chaos).
        A lambda-homeostat is robust to disturbance type because it tracks
        the dynamical quantity itself (tau drift / pruning shift the
        effective lambda; the controller follows it), unlike a fixed
        activity target whose optimum moves with the disturbance.
  RLS readout (forgetting 0.999) learns throughout. Disturbance at 10k.
  Metrics: NMSE before/after disturbance, recovery time, final NMSE.

Output files:
  data/s6_chaos_regulator_v1.csv    (one row per run)
  data/s6_chaos_regulator_v1.json   (params + per-cell aggregates)

Usage: python chaos_regulator.py [--quick]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shallow_trap_array_simulator import gamma, tau0, gen_tau_vec, preprogram_vec
from recurrent_substrate import (
    COUPLING_CONTRAST_SELF,
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    run_trajectory_nb, run_pair_ftle_nb)
from online_readout import OnlineRLS, running_mean_mse
from streaming_tasks import gen_mackey_glass

# ========================== Fixed parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0

RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

KAPPA_NOMINAL = 25.0
KAPPA_MIN, KAPPA_MAX = 1.0, 60.0
LAMBDA_TARGET = -0.02       # slightly ordered: memory-rich side (S1 finding)
ETA_LAMBDA = 3.0            # homeostat gain (kappa units per unit lambda err)
FTLE_EVERY = 1000           # kappa update period (pulses)
FTLE_WINDOW = 400           # Benettin window for the lambda estimate
HOMEO_BLOCK = 200           # substrate trajectory block size
S_WINDOW = 200              # proxy window (Part 1 only)

TAU_DRIFT = 1.5
PRUNE_FRAC = 0.4
NOISE_SIG = 0.10

DISTURBANCES = ['none', 'tau_drift', 'edge_prune', 'noise']
PART1_KAPPAS = [15.0, 25.0, 40.0]
T_TOTAL = 21000
T_DISTURB = 10000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's6_chaos_regulator_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's6_chaos_regulator_v1.json')


# ========================== Substrate / disturbances ==========================

def build_csr(n_units=N_UNITS):
    return build_topology_csr('random_graph', n_units,
                              seed=TOPO_SEED, avg_degree=AVG_DEGREE)


def apply_disturbance(ip, idx, wt, tau, disturb, noise_std=None):
    """Return (new_ip, new_idx, new_wt, new_tau, noise_std) under disturbance.

    'none'      : unchanged
    'tau_drift' : tau scaled by TAU_DRIFT
    'edge_prune': 40% of edges dropped (CSR rebuilt)
    'noise'     : noise_std returned for readout feature noise
    """
    if disturb == 'none':
        return ip, idx, wt, tau, 0.0
    if disturb == 'tau_drift':
        return ip, idx, wt, tau * TAU_DRIFT, 0.0
    if disturb == 'edge_prune':
        # rebuild CSR keeping only a random 60% of edges
        n = N_UNITS
        rng = np.random.RandomState(TOPO_SEED + 1)
        degs = np.diff(ip)
        # collect edges
        src = []
        dst = []
        for i in range(n):
            for e in range(ip[i], ip[i + 1]):
                src.append(i)
                dst.append(int(idx[e]))
        src = np.array(src)
        dst = np.array(dst)
        # prune only one direction per undirected edge: keep edges with src<dst
        undir_mask = src < dst
        su, du = src[undir_mask], dst[undir_mask]
        n_edges = su.shape[0]
        keep_mask = rng.rand(n_edges) > PRUNE_FRAC
        su, du = su[keep_mask], du[keep_mask]
        src2 = np.concatenate([su, du])
        dst2 = np.concatenate([du, su])
        order = np.lexsort((dst2, src2))
        src2, dst2 = src2[order], dst2[order]
        ip2 = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            ip2[i + 1] = ip2[i] + int(np.sum(src2 == i))
        idx2 = dst2.astype(np.int64)
        wt2 = np.empty(len(idx2))
        for i in range(n):
            k = ip2[i + 1] - ip2[i]
            if k > 0:
                wt2[ip2[i]:ip2[i + 1]] = 1.0 / k
        return ip2, idx2, wt2, tau, 0.0
    if disturb == 'noise':
        return ip, idx, wt, tau, NOISE_SIG
    raise ValueError(disturb)


def activity_proxy(states, window=S_WINDOW):
    """Cheap criticality proxy: running std of the population-mean
    current ratio over a window. Returns (proxy_series (T,), nan-padded)."""
    obs = np.exp(gamma * states) / FEATURE_SCALE
    m = obs.mean(axis=1)                       # population mean per pulse
    T = m.shape[0]
    out = np.full(T, np.nan)
    if T >= window:
        for t in range(window - 1, T):
            out[t] = m[t - window + 1:t + 1].std()
    return out


def memory_capacity_heldout(obs, dt_seq, k_max=50, ridge_lambda=1.0):
    """Held-out Jaeger memory capacity (70/30 split, k_max buffer).
    Returns mc_total_test."""
    S, n = obs.shape
    X = obs[k_max:, :]
    T = X.shape[0]
    n_train = int(0.7 * T)
    mu = X[:n_train].mean(axis=0)
    sd = X[:n_train].std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    Y = np.empty((T, k_max + 1))
    for k in range(k_max + 1):
        Y[:, k] = dt_seq[k_max + np.arange(T) - k]
    ymu = Y[:n_train].mean(axis=0)
    Yc = Y - ymu
    Xtr, Xte = Xs[:n_train], Xs[n_train + k_max:]
    Ytr, Yte = Yc[:n_train], Yc[n_train + k_max:]
    A = Xtr.T @ Xtr + ridge_lambda * np.eye(n)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    Pte = Xte @ W
    pc = []
    for k in range(k_max + 1):
        a = Pte[:, k] - Pte[:, k].mean()
        b = Yte[:, k]
        denom = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
        pc.append(float((a * b).sum()) / denom if denom > 0 else 0.0)
    return float(np.sum(pc[1:]))


# ========================== Part 1: proxy validation ==========================

def run_part1(args):
    """(disturb, kappa, seed_idx) -> dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    disturb, kappa, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    ip, idx, wt, tau_d, noise_std = apply_disturbance(*build_csr(), tau, disturb)
    rng = np.random.RandomState(seed_idx * 31 + 7)
    dt_seq = rng.uniform(2e-6, 20e-6, 3000)
    states, _, _ = run_trajectory_nb(x0, tau_d, dt_seq, PW, ip, idx, wt,
                                     kappa, ALPHA0, ALPHA_MIN, ALPHA_MAX,
                                     gamma, COUPLING_CONTRAST_SELF, 500)
    obs = np.exp(gamma * states) / FEATURE_SCALE
    if noise_std > 0:
        obs = obs + rng.normal(0, noise_std * obs.std(axis=0), obs.shape)
    mc = memory_capacity_heldout(obs, dt_seq[500:])
    proxy = activity_proxy(states)
    proxy_mean = float(np.nanmean(proxy[1000:]))
    return {'part': 1, 'disturb': disturb, 'kappa': float(kappa),
            'seed_idx': seed_idx, 'mc_heldout': mc,
            'proxy_mean': proxy_mean, 'runtime_s': time.time() - t0}


# ========================== Part 2: online homeostat ==========================

def run_part2(args):
    """(disturb, arm, seed_idx, s_star) -> dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    disturb, arm, seed_idx, s_star = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    ip0, idx0, wt0, _, _ = apply_disturbance(*build_csr(), tau, 'none')
    dt_seq, target_seq = gen_mackey_glass(seed=seed_idx)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    # split trajectory: nominal segment [0, T_DISTURB), disturbed [T_DISTURB, T)
    ip_d, idx_d, wt_d, tau_d, noise_std = apply_disturbance(
        ip0, idx0, wt0, tau, disturb)

    rls = OnlineRLS(N_UNITS + 1, 1, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP, reg=RLS_REG)
    preds = np.empty(T)
    kappa = KAPPA_NOMINAL
    kappa_hist = np.empty(T)
    kappa_hist[:] = np.nan

    def feed_features(states_seg, rng, start):
        """Feed a states segment through the readout, return (preds_seg, mu_seg)."""
        obs = np.exp(gamma * states_seg) / FEATURE_SCALE
        if noise_std > 0:
            obs = obs + rng.normal(0, noise_std * obs.std(axis=0), obs.shape)
        F = np.hstack([obs, np.ones((obs.shape[0], 1))])
        seg_pred = np.empty(obs.shape[0])
        for j in range(obs.shape[0]):
            seg_pred[j] = rls.predict(F[j])[0]
            rls.update(F[j], target[start + j])
        return seg_pred, obs

    rng_noise = np.random.RandomState(seed_idx * 131 + 3)
    n_seg = T - T_DISTURB

    if arm == 'fixed':
        # nominal segment
        st1, _, _ = run_trajectory_nb(x0, tau, dt_seq[:T_DISTURB], PW,
                                      ip0, idx0, wt0, KAPPA_NOMINAL,
                                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                      COUPLING_CONTRAST_SELF, 0)
        p1, _ = feed_features(st1, rng_noise, 0)
        preds[:T_DISTURB] = p1
        # disturbed segment (same kappa)
        st2, _, _ = run_trajectory_nb(st1[-1], tau_d, dt_seq[T_DISTURB:], PW,
                                      ip_d, idx_d, wt_d, KAPPA_NOMINAL,
                                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                      COUPLING_CONTRAST_SELF, 0)
        p2, _ = feed_features(st2, rng_noise, T_DISTURB)
        preds[T_DISTURB:] = p2
        kappa_hist[:] = KAPPA_NOMINAL
    else:
        # regulated: per-block runs; every FTLE_EVERY pulses estimate the
        # finite-time Lyapunov exponent (Benettin pair, current kappa) and
        # step kappa toward lambda_target
        x_cur = x0.copy()
        tau_cur = tau
        ip_c, idx_c, wt_c = ip0, idx0, wt0
        disturbed_flag = False
        for blk_start in range(0, T, HOMEO_BLOCK):
            blk_end = min(blk_start + HOMEO_BLOCK, T)
            if blk_start >= T_DISTURB and not disturbed_flag:
                ip_c, idx_c, wt_c, tau_cur, noise_std = apply_disturbance(
                    ip_c, idx_c, wt_c, tau_cur, disturb)
                disturbed_flag = True
            if blk_start % FTLE_EVERY == 0:
                w = min(FTLE_WINDOW, T - blk_start)
                lam, _ = run_pair_ftle_nb(
                    x_cur, tau_cur, dt_seq[blk_start:blk_start + w], PW,
                    ip_c, idx_c, wt_c, kappa,
                    ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                    COUPLING_CONTRAST_SELF, 1e-8, 10)
                err = float(np.clip(LAMBDA_TARGET - lam, -1.0, 1.0))
                kappa = float(np.clip(kappa + ETA_LAMBDA * err,
                                      KAPPA_MIN, KAPPA_MAX))
            st_b, _, _ = run_trajectory_nb(x_cur, tau_cur,
                                           dt_seq[blk_start:blk_end], PW,
                                           ip_c, idx_c, wt_c, kappa,
                                           ALPHA0, ALPHA_MIN, ALPHA_MAX,
                                           gamma, COUPLING_CONTRAST_SELF, 0)
            p_b, obs_b = feed_features(st_b, rng_noise, blk_start)
            preds[blk_start:blk_end] = p_b
            kappa_hist[blk_start:blk_end] = kappa
            x_cur = st_b[-1].copy()

    # metrics: NMSE pre/post disturbance, recovery time, final NMSE
    var_full = float(target[T_DISTURB:].var())
    def nmse(seg_preds, seg_targets):
        return float(np.mean((seg_preds - seg_targets) ** 2)) / max(var_full, 1e-12)
    pre_nmse = nmse(preds[T_DISTURB - 2000:T_DISTURB],
                    target[T_DISTURB - 2000:T_DISTURB])
    post_nmse = nmse(preds[T_DISTURB + 3000:T_DISTURB + 5000],
                     target[T_DISTURB + 3000:T_DISTURB + 5000])
    final_nmse = nmse(preds[-2000:], target[-2000:])
    # recovery: first pulse after disturbance where running MSE (window 500)
    # drops below 2x the pre-disturbance NMSE level
    err2 = (preds - target) ** 2
    cum = np.cumsum(np.concatenate([[0.0], err2]))
    run = (cum[500:] - cum[:-500]) / 500.0
    thr = min(2.0 * pre_nmse * var_full, np.nanmean(err2[T_DISTURB + 3000:]))
    seg = run[T_DISTURB:]
    hits = np.where(seg <= thr)[0]
    recovery = float(hits[0]) if hits.size else np.nan
    # substrate-quality probe: post-disturbance held-out MC at the settled
    # kappa vs the nominal kappa (fresh i.i.d. drive, disturbed physics).
    # The readout-independent metric for "is the substrate at the
    # memory-rich operating point".
    ip_d2, idx_d2, wt_d2, tau_d2, ns2 = apply_disturbance(
        *build_csr(), tau, disturb)
    x0p = preprogram_vec(ALPHA0, tau_d2)
    rng2 = np.random.RandomState(seed_idx * 999 + 5)
    dt_probe = rng2.uniform(2e-6, 20e-6, 3000)
    mc_nominal = 0.0
    mc_settled = 0.0
    for kp, tag in [(KAPPA_NOMINAL, 'nominal'), (kappa, 'settled')]:
        st_p, _, _ = run_trajectory_nb(x0p, tau_d2, dt_probe, PW,
                                       ip_d2, idx_d2, wt_d2, kp,
                                       ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                                       COUPLING_CONTRAST_SELF, 500)
        obs_p = np.exp(gamma * st_p) / FEATURE_SCALE
        if ns2 > 0:
            obs_p = obs_p + rng2.normal(0, ns2 * obs_p.std(axis=0), obs_p.shape)
        mc_val = memory_capacity_heldout(obs_p, dt_probe[500:])
        if tag == 'nominal':
            mc_nominal = mc_val
        else:
            mc_settled = mc_val

    kappa_settled = float(np.nanmean(kappa_hist[-4000:]))

    return {'part': 2, 'disturb': disturb, 'arm': arm, 'seed_idx': seed_idx,
            'pre_nmse': pre_nmse, 'post_nmse': post_nmse,
            'final_nmse': final_nmse, 'recovery_pulses': recovery,
            'kappa_settled': kappa_settled,
            'mc_at_nominal': mc_nominal, 'mc_at_settled': mc_settled,
            'runtime_s': time.time() - t0}


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        if r['part'] == 1:
            groups.setdefault(('p1', r['disturb'], r['kappa'], 0), []).append(r)
        else:
            groups.setdefault(('p2', r['disturb'], r['arm'], 0), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        p, d, x, _ = key
        entry = {'part': p, 'disturb': d}
        if p == 'p1':
            entry.update({'kappa': x, 'n_runs': len(rs)})
            for f in ['mc_heldout', 'proxy_mean']:
                v = np.array([r[f] for r in rs], dtype=float)
                entry[f + '_mean'] = float(np.mean(v))
                entry[f + '_std'] = float(np.std(v))
        else:
            entry.update({'arm': x, 'n_runs': len(rs)})
            for f in ['pre_nmse', 'post_nmse', 'final_nmse', 'recovery_pulses',
                      'kappa_settled', 'mc_at_nominal', 'mc_at_settled']:
                v = np.array([r[f] for r in rs], dtype=float)
                entry[f + '_mean'] = float(np.nanmean(v))
                entry[f + '_std'] = float(np.nanstd(v))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 108)
    print("S6 RESULTS (mean over seeds)")
    print("=" * 108)
    print("\n[Part 1] proxy validation: MC_heldout and activity proxy vs kappa")
    for d in DISTURBANCES:
        rows = [a for a in agg if a['part'] == 'p1' and a['disturb'] == d]
        line = f"  {d:<12}: "
        for a in sorted(rows, key=lambda x: x['kappa']):
            line += f"k={a['kappa']:g}: MC={a['mc_heldout_mean']:.2f} S={a['proxy_mean_mean']:.4f} | "
        print(line)
    print("\n[Part 2] online homeostat: MG NMSE pre/post disturbance, recovery")
    for d in DISTURBANCES:
        rows = [a for a in agg if a['part'] == 'p2' and a['disturb'] == d]
        if not rows:
            continue
        print(f"  {d:<12}:")
        for a in sorted(rows, key=lambda x: x['arm']):
            print(f"    {a['arm']:<10} pre={a['pre_nmse_mean']:.4f} "
                  f"post={a['post_nmse_mean']:.4f} final={a['final_nmse_mean']:.4f} "
                  f"recovery={a['recovery_pulses_mean']:.0f}p "
                  f"kappa_settled={a['kappa_settled_mean']:.1f} "
                  f"MC(nom)={a['mc_at_nominal_mean']:.2f} "
                  f"MC(settled)={a['mc_at_settled_mean']:.2f}")


def dispatch(a):
    """Top-level dispatcher (picklable for Pool workers)."""
    if a[0] == 'p1':
        return run_part1(a[1:])
    return run_part2(a[1:])


# ========================== Main ==========================

def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S6 chaos regulator (quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_csr(16)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_NOMINAL,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    # S* = proxy at (kappa=25, nominal), seed 0
    tau_s = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=0)
    x0_s = preprogram_vec(ALPHA0, tau_s)
    ip_s, idx_s, wt_s, _, _ = apply_disturbance(*build_csr(), tau_s, 'none')
    dt_s = np.random.RandomState(0).uniform(2e-6, 20e-6, 2000)
    st_s, _, _ = run_trajectory_nb(x0_s, tau_s, dt_s, PW, ip_s, idx_s, wt_s,
                                   KAPPA_NOMINAL, ALPHA0, ALPHA_MIN, ALPHA_MAX,
                                   gamma, COUPLING_CONTRAST_SELF, 200)
    s_star = float(activity_proxy(st_s)[1000:].mean())
    print(f"S* (proxy at kappa=25 nominal) = {s_star:.4f}")

    n_seeds = 2 if quick else N_SEEDS
    all_args = []
    for d in DISTURBANCES:
        for k in PART1_KAPPAS:
            for s in range(n_seeds):
                all_args.append(('p1', d, k, s))
    if quick:
        dist2 = ['tau_drift', 'noise']
    else:
        dist2 = DISTURBANCES
    for d in dist2:
        for arm in ['fixed', 'regulated']:
            for s in range(n_seeds):
                all_args.append(('p2', d, arm, s, s_star))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seeds={n_seeds})")

    results = []
    with Pool(min(cpu_count(), max(1, n_runs))) as pool:
        done = 0
        for res in pool.imap_unordered(dispatch, all_args, chunksize=2):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['part', 'disturb', 'kappa', 'arm', 'seed_idx',
                  'mc_heldout', 'proxy_mean', 'pre_nmse', 'post_nmse',
                  'final_nmse', 'recovery_pulses', 'kappa_settled',
                  'mc_at_nominal', 'mc_at_settled', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'kappa_nominal': KAPPA_NOMINAL, 'kappa_range': [KAPPA_MIN, KAPPA_MAX],
        'lambda_target': LAMBDA_TARGET, 'eta_lambda': ETA_LAMBDA,
        'ftle_every': FTLE_EVERY, 'ftle_window': FTLE_WINDOW,
        'homeo_block': HOMEO_BLOCK, 's_window': S_WINDOW,
        'tau_drift': TAU_DRIFT, 'prune_frac': PRUNE_FRAC, 'noise_sig': NOISE_SIG,
        'disturbances': DISTURBANCES, 'part1_kappas': PART1_KAPPAS,
        't_total': T_TOTAL, 't_disturb': T_DISTURB,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
