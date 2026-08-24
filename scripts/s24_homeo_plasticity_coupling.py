#!/usr/bin/env python3
"""
Homeostat-plasticity coupling (M4-M5 loop, REDEM S24).
=============================================================================
Type:           PAPER
Experiment:     E1: does the M4-M5 coupling loop (homeostat's kappa set-point
                gating the rewiring rate) add value under a sequential
                disturbance chain? 4 arms x 10 seeds.

Protocol: identical to S11 (T=28000, Mackey-Glass online task, three
cumulative disturbances at t=7k/14k/21k: tau_drift, edge_prune 40%,
readout noise sigma=0.1). On top of the S11 substrate loop, structure
plasticity (M4) rewires the coupling graph every 2000 pulses with a churn
fraction that is either fixed (0.05) or gated by the homeostat's kappa
deviation from nominal (the M4-M5 coupling hypothesis: rewire more
aggressively while the homeostat is actively compensating a disturbance).

Arms:
  fixed_kappa_churn   : kappa=25 fixed, fixed churn 0.05 (no homeostat)
  homeo_no_plasticity : homeostat, no rewiring (S11 'regulated' anchor)
  homeo_fixed_churn   : homeostat, fixed churn 0.05 (S8-style 'full')
  coupled_churn       : homeostat, churn gated by |kappa - 25|
                        churn = clip(0.05 + 0.05*|kappa-25|, 0.02, 0.20)

Metrics per round (r0..r3): NMSE, kappa_settled, held-out MC probe.
Cross-round: kappa return consistency (as S11).

Output files:
  data/s24_homeo_plasticity_coupling_v1.csv
  data/s24_homeo_plasticity_coupling_v1.json

Usage: python s24_homeo_plasticity_coupling.py [--quick]
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
from structure_plasticity import evolve_mask

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

PLASTICITY_EVERY = 2000
CHURN_BASE = 0.05
CHURN_GAIN = 0.05          # churn per unit of |kappa - kappa_nominal|
CHURN_MIN, CHURN_MAX = 0.02, 0.20

TAU_DRIFT = 1.5
PRUNE_FRAC = 0.4
NOISE_SIG = 0.10

ROUND_LEN = 7000
T_TOTAL = 28000
DISTURB_TIMES = [7000, 14000, 21000]
DISTURB_TYPES = ['tau_drift', 'edge_prune', 'noise']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's24_homeo_plasticity_coupling_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's24_homeo_plasticity_coupling_v1.json')


def initial_mask():
    """random_graph mask (same topology seed as S11/S8)."""
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    mask = np.zeros((N_UNITS, N_UNITS), dtype=bool)
    for i in range(N_UNITS):
        for e in range(ip[i], ip[i + 1]):
            j = int(idx[e])
            mask[min(i, j), max(i, j)] = True
    return mask


def prune_mask(mask, rng):
    """Remove PRUNE_FRAC of undirected edges from the mask (same semantics
    as S11's edge_prune disturbance)."""
    edges = np.argwhere(np.triu(mask, 1))
    keep = rng.rand(edges.shape[0]) > PRUNE_FRAC
    out = mask.copy()
    for (i, j), kp in zip(edges, keep):
        if not kp:
            out[i, j] = False
    return out


def churn_for(arm, kappa):
    if arm == 'coupled_churn':
        return float(np.clip(CHURN_BASE + CHURN_GAIN * abs(kappa - KAPPA_NOMINAL),
                             CHURN_MIN, CHURN_MAX))
    return CHURN_BASE


def run_single(args):
    """(arm, seed_idx) -> dict with per-round metrics."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    dt_seq, target_seq = gen_mackey_glass(seed=seed_idx, n_points=T_TOTAL)
    T = min(dt_seq.shape[0], T_TOTAL)
    dt_seq = dt_seq[:T]
    target = target_seq[:T].astype(np.float64)

    rls = OnlineRLS(N_UNITS + 1, 1, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP, reg=RLS_REG)
    preds = np.empty(T)
    kappa_hist = np.empty(T)
    kappa_hist[:] = np.nan

    rng_noise = np.random.RandomState(seed_idx * 131 + 3)
    rng_prune = np.random.RandomState(TOPO_SEED + 1)

    # Cumulative disturbance state
    tau_cur = tau.copy()
    mask = initial_mask()
    ip_c, idx_c, wt_c = adjacency_to_csr(mask)
    noise_std = 0.0
    disturb_idx = 0

    use_homeo = arm in ('homeo_no_plasticity', 'homeo_fixed_churn',
                        'coupled_churn')
    use_plastic = arm in ('fixed_kappa_churn', 'homeo_fixed_churn',
                          'coupled_churn')

    kappa = KAPPA_NOMINAL

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

    x_cur = x0.copy()
    for blk_start in range(0, T, HOMEO_BLOCK):
        blk_end = min(blk_start + HOMEO_BLOCK, T)
        if disturb_idx < len(DISTURB_TIMES) and blk_start >= DISTURB_TIMES[disturb_idx]:
            d = DISTURB_TYPES[disturb_idx]
            if d == 'tau_drift':
                tau_cur = tau * TAU_DRIFT
            elif d == 'edge_prune':
                mask = prune_mask(mask, rng_prune)
                ip_c, idx_c, wt_c = adjacency_to_csr(mask)
            elif d == 'noise':
                noise_std = NOISE_SIG
            disturb_idx += 1
        if use_homeo and blk_start % FTLE_EVERY == 0:
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

        if use_plastic and blk_end % PLASTICITY_EVERY == 0 and blk_end < T:
            obs_b = np.exp(gamma * st_b) / FEATURE_SCALE
            z = (obs_b - obs_b.mean(axis=0)) / (obs_b.std(axis=0) + 1e-12)
            C = (z.T @ z) / obs_b.shape[0]
            churn = churn_for(arm, kappa)
            n_grow = max(1, int(churn * np.triu(mask, 1).sum()))
            mask = evolve_mask(mask, C, n_grow)
            ip_c, idx_c, wt_c = adjacency_to_csr(mask)

    # Per-round metrics
    var_full = float(target[DISTURB_TIMES[-1]:].var())

    def nmse(seg_p, seg_t):
        v = float(seg_t.var()) if len(seg_t) > 1 else var_full
        return float(np.mean((seg_p - seg_t) ** 2)) / max(v, 1e-12)

    results = {'arm': arm, 'seed_idx': seed_idx,
               'n_edges_final': int(mask.sum()),
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
        kh_seg = kappa_hist[lo + 1000:hi]
        results[f'r{r}_kappa'] = float(np.nanmean(kh_seg)) if np.any(~np.isnan(kh_seg)) else float('nan')

    # MC probe per round (at the kappa settled in that round)
    for r in range(1, 4):
        kp = results[f'r{r}_kappa']
        if arm == 'fixed_kappa_churn':
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

    # Round 0 MC: nominal physics, kappa=25, initial mask
    x0n = preprogram_vec(ALPHA0, tau)
    rng_n = np.random.RandomState(seed_idx * 999)
    dt_n = rng_n.uniform(2e-6, 20e-6, 3000)
    ip_n, idx_n, wt_n = adjacency_to_csr(initial_mask())
    st_n, _, _ = run_trajectory_nb(
        x0n, tau, dt_n, PW, ip_n, idx_n, wt_n, KAPPA_NOMINAL,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
        COUPLING_CONTRAST_SELF, 500)
    obs_n = np.exp(gamma * st_n) / FEATURE_SCALE
    results['r0_mc'] = memory_capacity_heldout(obs_n, dt_n[500:])

    # Cross-round kappa consistency
    k1 = results.get('r1_kappa', float('nan'))
    k2 = results.get('r2_kappa', float('nan'))
    k3 = results.get('r3_kappa', float('nan'))
    if arm == 'fixed_kappa_churn':
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
        entry['n_edges_final_mean'] = float(np.mean([r['n_edges_final'] for r in rs]))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 118)
    print("S24 RESULTS (mean over seeds): homeostat-plasticity coupling")
    print("=" * 118)
    print(f" {'arm':>20} | {'r1_mc':>6} {'r2_mc':>6} {'r3_mc':>6} | "
          f"{'r1_kap':>7} {'r3_kap':>7} | {'r3_nmse':>8} | {'n_edges':>7}")
    for a in agg:
        print(f" {a['arm']:>20} | "
              f"{a.get('r1_mc_mean', float('nan')):>6.2f} "
              f"{a.get('r2_mc_mean', float('nan')):>6.2f} "
              f"{a.get('r3_mc_mean', float('nan')):>6.2f} | "
              f"{a.get('r1_kappa_mean', float('nan')):>7.1f} "
              f"{a.get('r3_kappa_mean', float('nan')):>7.1f} | "
              f"{a.get('r3_nmse_mean', float('nan')):>8.4f} | "
              f"{a.get('n_edges_final_mean', float('nan')):>7.0f}")


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S24 homeostat-plasticity "
          f"coupling (quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16,
                                           seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_NOMINAL,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    arms = ['fixed_kappa_churn', 'homeo_no_plasticity', 'homeo_fixed_churn',
            'coupled_churn']
    n_seeds = 2 if quick else N_SEEDS
    all_args = [(arm, s) for arm in arms for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seeds={n_seeds}, arms={arms})")

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
                  'kappa_drift_r2_r1', 'kappa_drift_r3_r1',
                  'n_edges_final', 'runtime_s']
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
        'homeo_block': HOMEO_BLOCK,
        'plasticity_every': PLASTICITY_EVERY,
        'churn_base': CHURN_BASE, 'churn_gain': CHURN_GAIN,
        'churn_range': [CHURN_MIN, CHURN_MAX],
        'tau_drift': TAU_DRIFT, 'prune_frac': PRUNE_FRAC, 'noise_sig': NOISE_SIG,
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
