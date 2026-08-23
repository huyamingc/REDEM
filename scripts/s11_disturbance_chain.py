#!/usr/bin/env python3
"""
Disturbance chain for the chaos homeostat (REDEM S11).
=============================================================================
Type:           PAPER
Experiment:     Three sequential disturbances (tau_drift -> edge_prune ->
                noise) applied at t=7k, 14k, 21k on a random_graph
                substrate with Mackey-Glass online task. Tests whether
                the homeostat recovers to the same kappa neighborhood
                across rounds and whether MC is stable under cumulative
                disturbance.

Disturbance chain (cumulative):
  Round 0 [0, 7k)     : nominal
  Round 1 [7k, 14k)   : tau_drift (tau *= 1.5)
  Round 2 [14k, 21k)  : edge_prune (40% edges removed, cumulative on top of drift)
  Round 3 [21k, 28k)  : readout noise (sigma=0.1, cumulative on top of drift+prune)

Arms:
  fixed      : kappa=25 throughout
  regulated  : homeostat adjusts kappa online (lambda_target=-0.02, eta=3)

Metrics per round: pre/post NMSE, kappa_settled, MC_heldout probe.
Cross-round: kappa return consistency |kappa_r2 - kappa_r1|, |kappa_r3 - kappa_r1|.

Output files:
  data/s11_disturbance_chain_v1.csv
  data/s11_disturbance_chain_v1.json

Usage: python s11_disturbance_chain.py [--quick]
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
    adjacency_to_csr, run_trajectory_nb, run_pair_ftle_nb)
from online_readout import OnlineRLS, memory_capacity_heldout
from streaming_tasks import gen_mackey_glass

# ========================== Fixed parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0

RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

KAPPA_NOMINAL = 25.0
KAPPA_MIN, KAPPA_MAX = 1.0, 60.0
LAMBDA_TARGET = -0.02
ETA_LAMBDA = 3.0
FTLE_EVERY = 1000
FTLE_WINDOW = 400
HOMEO_BLOCK = 200

TAU_DRIFT = 1.5
PRUNE_FRAC = 0.4
NOISE_SIG = 0.10

ROUND_LEN = 7000
T_TOTAL = 28000
DISTURB_TIMES = [7000, 14000, 21000]
DISTURB_TYPES = ['tau_drift', 'edge_prune', 'noise']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's11_disturbance_chain_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's11_disturbance_chain_v1.json')


def build_csr(n_units=N_UNITS):
    return build_topology_csr('random_graph', n_units,
                             seed=TOPO_SEED, avg_degree=AVG_DEGREE)


def prune_edges(ip, idx, wt, seed=TOPO_SEED + 1):
    """Remove PRUNE_FRAC of undirected edges from a CSR topology."""
    n = ip.shape[0] - 1
    rng = np.random.RandomState(seed)
    src = []
    dst = []
    for i in range(n):
        for e in range(ip[i], ip[i + 1]):
            src.append(i)
            dst.append(int(idx[e]))
    src = np.array(src)
    dst = np.array(dst)
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
    return ip2, idx2, wt2


def run_single(args):
    """(arm, seed_idx) -> dict with per-round metrics."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    ip0, idx0, wt0 = build_csr()
    dt_seq, target_seq = gen_mackey_glass(seed=seed_idx, n_points=T_TOTAL)
    T = min(dt_seq.shape[0], T_TOTAL)
    dt_seq = dt_seq[:T]
    target = target_seq[:T].astype(np.float64)

    rls = OnlineRLS(N_UNITS + 1, 1, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP, reg=RLS_REG)
    preds = np.empty(T)
    kappa = KAPPA_NOMINAL
    kappa_hist = np.empty(T)
    kappa_hist[:] = np.nan

    rng_noise = np.random.RandomState(seed_idx * 131 + 3)

    # Cumulative disturbance state
    tau_cur = tau.copy()
    ip_c, idx_c, wt_c = ip0, idx0, wt0
    noise_std = 0.0
    disturb_idx = 0

    def feed_features(states_seg, start):
        obs = np.exp(gamma * states_seg) / FEATURE_SCALE
        if noise_std > 0:
            obs = obs + rng_noise.normal(0, noise_std * obs.std(axis=0),
                                         obs.shape)
        F = np.hstack([obs, np.ones((obs.shape[0], 1))])
        seg_pred = np.empty(obs.shape[0])
        for j in range(obs.shape[0]):
            seg_pred[j] = rls.predict(F[j])[0]
            rls.update(F[j], target[start + j])
        return seg_pred

    if arm == 'fixed':
        x_cur = x0.copy()
        for blk_start in range(0, T, HOMEO_BLOCK):
            blk_end = min(blk_start + HOMEO_BLOCK, T)
            if disturb_idx < len(DISTURB_TIMES) and blk_start >= DISTURB_TIMES[disturb_idx]:
                d = DISTURB_TYPES[disturb_idx]
                if d == 'tau_drift':
                    tau_cur = tau * TAU_DRIFT
                elif d == 'edge_prune':
                    ip_c, idx_c, wt_c = prune_edges(ip_c, idx_c, wt_c)
                elif d == 'noise':
                    noise_std = NOISE_SIG
                disturb_idx += 1
            st_b, _, _ = run_trajectory_nb(
                x_cur, tau_cur, dt_seq[blk_start:blk_end], PW,
                ip_c, idx_c, wt_c, KAPPA_NOMINAL,
                ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                COUPLING_CONTRAST_SELF, 0)
            p_b = feed_features(st_b, blk_start)
            preds[blk_start:blk_end] = p_b
            kappa_hist[blk_start:blk_end] = KAPPA_NOMINAL
            x_cur = st_b[-1].copy()
    else:
        x_cur = x0.copy()
        for blk_start in range(0, T, HOMEO_BLOCK):
            blk_end = min(blk_start + HOMEO_BLOCK, T)
            if disturb_idx < len(DISTURB_TIMES) and blk_start >= DISTURB_TIMES[disturb_idx]:
                d = DISTURB_TYPES[disturb_idx]
                if d == 'tau_drift':
                    tau_cur = tau * TAU_DRIFT
                elif d == 'edge_prune':
                    ip_c, idx_c, wt_c = prune_edges(ip_c, idx_c, wt_c)
                elif d == 'noise':
                    noise_std = NOISE_SIG
                disturb_idx += 1
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
            st_b, _, _ = run_trajectory_nb(
                x_cur, tau_cur, dt_seq[blk_start:blk_end], PW,
                ip_c, idx_c, wt_c, kappa,
                ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                COUPLING_CONTRAST_SELF, 0)
            p_b = feed_features(st_b, blk_start)
            preds[blk_start:blk_end] = p_b
            kappa_hist[blk_start:blk_end] = kappa
            x_cur = st_b[-1].copy()

    # Per-round metrics
    var_full = float(target[DISTURB_TIMES[-1]:].var())
    def nmse(seg_p, seg_t):
        v = float(seg_t.var()) if len(seg_t) > 1 else var_full
        return float(np.mean((seg_p - seg_t) ** 2)) / max(v, 1e-12)

    results = {'arm': arm, 'seed_idx': seed_idx, 'runtime_s': time.time() - t0}
    round_boundaries = [0] + DISTURB_TIMES + [T]
    for r in range(4):
        lo = round_boundaries[r]
        hi = round_boundaries[r + 1]
        mid = (lo + hi) // 2
        if mid + 1000 < hi and mid - 1000 > lo:
            results[f'r{r}_nmse'] = nmse(preds[mid - 1000:mid + 1000],
                                         target[mid - 1000:mid + 1000])
        else:
            results[f'r{r}_nmse'] = float('nan')
        kh_seg = kappa_hist[lo + 1000:hi]
        results[f'r{r}_kappa'] = float(np.nanmean(kh_seg)) if np.any(~np.isnan(kh_seg)) else float('nan')

    # MC probe per round (at the kappa settled in that round)
    for r in range(1, 4):
        kp = results[f'r{r}_kappa']
        if arm == 'fixed':
            kp = KAPPA_NOMINAL
        if np.isnan(kp):
            results[f'r{r}_mc'] = float('nan')
            continue
        x0p = preprogram_vec(ALPHA0, tau_cur)
        rng2 = np.random.RandomState(seed_idx * 999 + r * 10)
        dt_probe = rng2.uniform(2e-6, 20e-6, 3000)
        st_p, _, _ = run_trajectory_nb(
            x0p, tau_cur, dt_probe, PW, ip_c, idx_c, wt_c, kp,
            ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
            COUPLING_CONTRAST_SELF, 500)
        obs_p = np.exp(gamma * st_p) / FEATURE_SCALE
        if noise_std > 0:
            obs_p = obs_p + rng2.normal(0, noise_std * obs_p.std(axis=0),
                                        obs_p.shape)
        results[f'r{r}_mc'] = memory_capacity_heldout(obs_p, dt_probe[500:])

    results['r0_mc'] = float('nan')
    # Round 0 MC: nominal physics, kappa=25
    x0n = preprogram_vec(ALPHA0, tau)
    rng_n = np.random.RandomState(seed_idx * 999)
    dt_n = rng_n.uniform(2e-6, 20e-6, 3000)
    st_n, _, _ = run_trajectory_nb(
        x0n, tau, dt_n, PW, ip0, idx0, wt0, KAPPA_NOMINAL,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
        COUPLING_CONTRAST_SELF, 500)
    obs_n = np.exp(gamma * st_n) / FEATURE_SCALE
    results['r0_mc'] = memory_capacity_heldout(obs_n, dt_n[500:])

    # Cross-round kappa consistency
    k1 = results.get('r1_kappa', float('nan'))
    k2 = results.get('r2_kappa', float('nan'))
    k3 = results.get('r3_kappa', float('nan'))
    if arm == 'fixed':
        k1 = k2 = k3 = KAPPA_NOMINAL
    results['kappa_drift_r2_r1'] = float(abs(k2 - k1)) if not (np.isnan(k1) or np.isnan(k2)) else float('nan')
    results['kappa_drift_r3_r1'] = float(abs(k3 - k1)) if not (np.isnan(k1) or np.isnan(k3)) else float('nan')

    return results


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(r['arm'], []).append(r)
    agg = []
    for arm, rs in sorted(groups.items()):
        entry = {'arm': arm, 'n_runs': len(rs)}
        for r_idx in range(4):
            for metric in ['nmse', 'kappa', 'mc']:
                key = f'r{r_idx}_{metric}'
                v = np.array([r[key] for r in rs], dtype=float)
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    entry[key + '_mean'] = float(np.mean(v))
                    entry[key + '_std'] = float(np.std(v))
                else:
                    entry[key + '_mean'] = float('nan')
                    entry[key + '_std'] = float('nan')
        for key in ['kappa_drift_r2_r1', 'kappa_drift_r3_r1']:
            v = np.array([r[key] for r in rs], dtype=float)
            v = v[~np.isnan(v)]
            entry[key + '_mean'] = float(np.mean(v)) if len(v) > 0 else float('nan')
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 110)
    print("S11 RESULTS (mean over seeds): disturbance chain")
    print("=" * 110)
    print(f" {'arm':>10} | {'r0_nmse':>8} {'r1_nmse':>8} {'r2_nmse':>8} {'r3_nmse':>8} | "
          f"{'r0_kap':>7} {'r1_kap':>7} {'r2_kap':>7} {'r3_kap':>7} | "
          f"{'r0_mc':>6} {'r1_mc':>6} {'r2_mc':>6} {'r3_mc':>6} | "
          f"{'d_r2':>5} {'d_r3':>5}")
    for a in agg:
        print(f" {a['arm']:>10} | "
              f"{a.get('r0_nmse_mean', float('nan')):>8.4f} "
              f"{a.get('r1_nmse_mean', float('nan')):>8.4f} "
              f"{a.get('r2_nmse_mean', float('nan')):>8.4f} "
              f"{a.get('r3_nmse_mean', float('nan')):>8.4f} | "
              f"{a.get('r0_kappa_mean', float('nan')):>7.1f} "
              f"{a.get('r1_kappa_mean', float('nan')):>7.1f} "
              f"{a.get('r2_kappa_mean', float('nan')):>7.1f} "
              f"{a.get('r3_kappa_mean', float('nan')):>7.1f} | "
              f"{a.get('r0_mc_mean', float('nan')):>6.2f} "
              f"{a.get('r1_mc_mean', float('nan')):>6.2f} "
              f"{a.get('r2_mc_mean', float('nan')):>6.2f} "
              f"{a.get('r3_mc_mean', float('nan')):>6.2f} | "
              f"{a.get('kappa_drift_r2_r1_mean', float('nan')):>5.1f} "
              f"{a.get('kappa_drift_r3_r1_mean', float('nan')):>5.1f}")


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S11 disturbance chain "
          f"(quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_csr(16)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_NOMINAL,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = 2 if quick else N_SEEDS
    all_args = [(arm, s) for arm in ['fixed', 'regulated']
                for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seeds={n_seeds})")

    results = []
    with Pool(min(cpu_count(), max(1, n_runs))) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=1):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 5) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed_idx',
                  'r0_nmse', 'r1_nmse', 'r2_nmse', 'r3_nmse',
                  'r0_kappa', 'r1_kappa', 'r2_kappa', 'r3_kappa',
                  'r0_mc', 'r1_mc', 'r2_mc', 'r3_mc',
                  'kappa_drift_r2_r1', 'kappa_drift_r3_r1', 'runtime_s']
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
        'homeo_block': HOMEO_BLOCK, 'tau_drift': TAU_DRIFT,
        'prune_frac': PRUNE_FRAC, 'noise_sig': NOISE_SIG,
        'round_len': ROUND_LEN, 't_total': T_TOTAL,
        'disturb_times': DISTURB_TIMES, 'disturb_types': DISTURB_TYPES,
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
