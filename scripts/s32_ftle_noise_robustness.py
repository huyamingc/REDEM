#!/usr/bin/env python3
"""
S32: Homeostat robustness to FTLE-estimation noise (Paper A follow-up).
=============================================================================
Type:           PAPER
Paper Section:  Paper A, Sec. 4 (lambda-homeostat) follow-up
Experiment:     The homeostat steps kappa from a finite-time Benettin
                estimate of lambda taken every 1000 pulses over a
                400-pulse window. That estimate is noisy; the audit
                flagged that the noise level is uncontrolled. This script
                injects controlled Gaussian noise into the lambda estimate
                and tests whether the homeostat still (a) settles at
                kappa ~ 26-27 and (b) retains its post-disturbance
                recovery, under the S11 sequential disturbance chain.

Arms (10 seeds): noise_std in {0.0, 0.01, 0.03, 0.05, 0.10} added to
every lambda estimate (lambda scale is ~0.05-0.1 per pulse; std 0.10 is
severe corruption). The 0.0 arm must reproduce the S11 regulated anchor
(r3 MC 8.47, settled kappa ~28.5).

Output files:
  data/s32_ftle_noise_v1.csv
  data/s32_ftle_noise_v1.json

Usage: python s32_ftle_noise_robustness.py [--quick] [--sequential]
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
from online_readout import OnlineRLS, memory_capacity_heldout
from streaming_tasks import gen_mackey_glass
from s11_disturbance_chain import (
    prune_edges, N_UNITS, CV_TAU, TOPO_SEED, AVG_DEGREE, FEATURE_SCALE,
    RLS_FORGETTING, RLS_INIT_COV, RLS_TRACE_CAP, RLS_REG,
    KAPPA_NOMINAL, KAPPA_MIN, KAPPA_MAX, LAMBDA_TARGET, ETA_LAMBDA,
    FTLE_EVERY, FTLE_WINDOW, HOMEO_BLOCK, TAU_DRIFT, PRUNE_FRAC, NOISE_SIG,
    T_TOTAL, DISTURB_TIMES, DISTURB_TYPES)

N_SEEDS = 10
NOISE_STDS = [0.0, 0.01, 0.03, 0.05, 0.10]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's32_ftle_noise_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's32_ftle_noise_v1.json')


def build_csr(n_units=N_UNITS):
    return build_topology_csr('random_graph', n_units,
                              seed=TOPO_SEED, avg_degree=AVG_DEGREE)


def run_single(args):
    """(noise_std, seed_idx) -> metrics (S11 regulated protocol + noise)."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    noise_std, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    ip0, idx0, wt0 = build_csr()
    dt_seq, target_seq = gen_mackey_glass(seed=seed_idx, n_points=T_TOTAL)
    T = min(dt_seq.shape[0], T_TOTAL)
    dt_seq = dt_seq[:T]
    target = target_seq[:T].astype(np.float64)

    rls = OnlineRLS(N_UNITS + 1, 1, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                    reg=RLS_REG)
    preds = np.empty(T)
    kappa = KAPPA_NOMINAL
    kappa_hist = np.empty(T)
    kappa_hist[:] = np.nan
    rng_noise = np.random.RandomState(seed_idx * 271 + 5)   # lambda noise
    rng_feat = np.random.RandomState(seed_idx * 131 + 3)    # readout noise

    tau_cur = tau.copy()
    ip_c, idx_c, wt_c = ip0, idx0, wt0
    noise_std_feat = 0.0
    disturb_idx = 0

    x_cur = x0.copy()
    for blk_start in range(0, T, HOMEO_BLOCK):
        blk_end = min(blk_start + HOMEO_BLOCK, T)
        if (disturb_idx < len(DISTURB_TIMES)
                and blk_start >= DISTURB_TIMES[disturb_idx]):
            d = DISTURB_TYPES[disturb_idx]
            if d == 'tau_drift':
                tau_cur = tau * TAU_DRIFT
            elif d == 'edge_prune':
                ip_c, idx_c, wt_c = prune_edges(ip_c, idx_c, wt_c)
            elif d == 'noise':
                noise_std_feat = NOISE_SIG
            disturb_idx += 1

        if blk_start % FTLE_EVERY == 0:
            w = min(FTLE_WINDOW, T - blk_start)
            lam, _ = run_pair_ftle_nb(
                x_cur, tau_cur, dt_seq[blk_start:blk_start + w], PW,
                ip_c, idx_c, wt_c, kappa,
                ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                COUPLING_CONTRAST_SELF, 1e-8, 10)
            if noise_std > 0:
                lam = lam + rng_noise.normal(0, noise_std)
            err = float(np.clip(LAMBDA_TARGET - lam, -1.0, 1.0))
            kappa = float(np.clip(kappa + ETA_LAMBDA * err,
                                  KAPPA_MIN, KAPPA_MAX))

        st_b, _, _ = run_trajectory_nb(
            x_cur, tau_cur, dt_seq[blk_start:blk_end], PW,
            ip_c, idx_c, wt_c, kappa,
            ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
            COUPLING_CONTRAST_SELF, 0)
        obs = np.exp(gamma * st_b) / FEATURE_SCALE
        if noise_std_feat > 0:
            obs = obs + rng_feat.normal(
                0, noise_std_feat * obs.std(axis=0), obs.shape)
        F = np.hstack([obs, np.ones((obs.shape[0], 1))])
        for j in range(obs.shape[0]):
            preds[blk_start + j] = rls.predict(F[j])[0]
            rls.update(F[j], target[blk_start + j])
        kappa_hist[blk_start:blk_end] = kappa
        x_cur = st_b[-1].copy()

    # Per-round metrics (S11 protocol)
    var_full = float(target[DISTURB_TIMES[-1]:].var())

    def nmse(seg_p, seg_t):
        v = float(seg_t.var()) if len(seg_t) > 1 else var_full
        return float(np.mean((seg_p - seg_t) ** 2)) / max(v, 1e-12)

    results = {'noise_std': float(noise_std), 'seed_idx': seed_idx,
               'runtime_s': time.time() - t0}
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
        kh = kappa_hist[lo + 1000:hi]
        results[f'r{r}_kappa'] = (float(np.nanmean(kh))
                                  if np.any(~np.isnan(kh)) else float('nan'))

    for r in range(1, 4):
        kp = results[f'r{r}_kappa']
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
        if noise_std_feat > 0:
            obs_p = obs_p + rng2.normal(
                0, noise_std_feat * obs_p.std(axis=0), obs_p.shape)
        results[f'r{r}_mc'] = memory_capacity_heldout(obs_p, dt_probe[500:])

    x0n = preprogram_vec(ALPHA0, tau)
    rng_n = np.random.RandomState(seed_idx * 999)
    dt_n = rng_n.uniform(2e-6, 20e-6, 3000)
    st_n, _, _ = run_trajectory_nb(
        x0n, tau, dt_n, PW, ip0, idx0, wt0, KAPPA_NOMINAL,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
        COUPLING_CONTRAST_SELF, 500)
    obs_n = np.exp(gamma * st_n) / FEATURE_SCALE
    results['r0_mc'] = memory_capacity_heldout(obs_n, dt_n[500:])
    return results


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S32 FTLE-noise robustness "
          f"(quick={quick}, sequential={sequential})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_csr(16)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w,
                      KAPPA_NOMINAL, ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    run_pair_ftle_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w,
                     KAPPA_NOMINAL, ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                     COUPLING_CONTRAST_SELF, 1e-8, 10)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = 2 if quick else N_SEEDS
    all_args = [(ns, s) for ns in NOISE_STDS for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} ({len(NOISE_STDS)} noise levels "
          f"x{n_seeds} seeds)")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_single(a))
            if (i + 1) % max(1, n_runs // 5) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 5) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['noise_std', 'seed_idx', 'r0_nmse', 'r1_nmse', 'r2_nmse',
                  'r3_nmse', 'r1_kappa', 'r2_kappa', 'r3_kappa',
                  'r0_mc', 'r1_mc', 'r2_mc', 'r3_mc', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 92)
    print("S32 RESULTS (Paper A follow-up): FTLE-noise robustness")
    print("=" * 92)
    print(f" {'noise':>6} | {'r3_mc':>6} | {'r1_kappa':>8} "
          f"{'r2_kappa':>8} {'r3_kappa':>8} | {'r3_nmse':>8}")
    for ns in NOISE_STDS:
        rs = [r for r in results if r['noise_std'] == ns]
        mc = np.mean([r['r3_mc'] for r in rs])
        print(f" {ns:>6.2f} | {mc:>6.2f} | "
              f"{np.mean([r['r1_kappa'] for r in rs]):>8.2f} "
              f"{np.mean([r['r2_kappa'] for r in rs]):>8.2f} "
              f"{np.mean([r['r3_kappa'] for r in rs]):>8.2f} | "
              f"{np.mean([r['r3_nmse'] for r in rs]):>8.4f}")
    print("-" * 92)
    print("S11 regulated anchor: r3 MC 8.47, kappa 26.15->28.51")

    params = {
        'experiment': 'homeostat robustness to FTLE-estimation noise',
        'protocol': 'S11 sequential disturbance chain, regulated arm; '
                    'Gaussian noise added to every lambda estimate',
        'noise_stds': NOISE_STDS,
        'lambda_scale_note': 'lambda magnitude ~0.05-0.1 per pulse; '
                             'std 0.10 is severe corruption',
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'rows': results}, f, indent=2)

    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
