#!/usr/bin/env python3
"""
S28: Causal leak audit on the disturbance chain (Paper B follow-up).
=============================================================================
Type:           PAPER
Paper Section:  Paper B, Sec. 4.3 (causal audit) follow-up
Experiment:     The committed S13 causal audit ran on regime_switch, where
                every arm sits at ~0.996 accuracy (a ceiling): leak
                detection there is capped at <=0.02 pp and cannot
                discriminate "no leak" from "leak masked by the ceiling".
                This script re-runs the same leak injections on the S11
                sequential disturbance chain (T=28000, disturbances at
                7k/14k/21k, Mackey-Glass online task), where the
                mechanisms' substrate-level effects are visible through
                the readout-independent held-out MC probe and per-round
                NMSE.

Arms (10 seeds, 6 arms; leak semantics verbatim from S13):
  normal          : full system - RLS on [obs; 1; metadata-EMA] + homeostat
                    (FTLE kappa regulation) + plasticity (5% churn / 2k)
  leak_rls        : RLS target at t mixes 1% of y[t+50]
  leak_metadata   : metadata EMA input mixes 1% of f[t+50]
  leak_ftle       : homeostat lambda mixes 1% of lambda[t+400]
  leak_plasticity : plasticity C mixes 10% of the NEXT block's correlation
  no_plasticity   : M4 off (lower bound for the plasticity comparison)

Verdict rule (same as S13): a leak arm whose per-round MC or NMSE
improves over normal beyond a threshold is a detected leak. On the chain
the FTLE leak acts through kappa -> MC and the plasticity leak through
structure -> MC, so the audit can finally see the channels the ceilinged
regime_switch task could not.

Cross-check: the homeostat dynamics in 'normal' must reproduce the S11
regulated anchor (kappa drift 25.3 -> 28.5; r3 MC ~8.47 for the
homeostat-only arm - here with metadata+plasticity the MC may differ,
but kappa drift should match).

Output files:
  data/s28_causal_audit_chain_v1.csv
  data/s28_causal_audit_chain_v1.json

Usage: python s28_causal_audit_chain.py [--quick] [--sequential]
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
from s11_disturbance_chain import (
    prune_edges, N_UNITS, CV_TAU, TOPO_SEED, AVG_DEGREE, FEATURE_SCALE,
    RLS_FORGETTING, RLS_INIT_COV, RLS_TRACE_CAP, RLS_REG,
    KAPPA_NOMINAL, KAPPA_MIN, KAPPA_MAX, LAMBDA_TARGET, ETA_LAMBDA,
    FTLE_EVERY, FTLE_WINDOW, HOMEO_BLOCK, TAU_DRIFT, PRUNE_FRAC, NOISE_SIG,
    T_TOTAL, DISTURB_TIMES, DISTURB_TYPES)

# ========================== Fixed parameters ==========================
N_SEEDS = 10
TAU_SLOW = 500.0
PLASTICITY_EVERY = 2000
PLASTICITY_CHURN = 0.05
LEAK_FRAC = 0.01
LEAK_FRAC_PLASTICITY = 0.10
LEAK_K = 50
LEAK_FTLE_AHEAD = 400

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's28_causal_audit_chain_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's28_causal_audit_chain_v1.json')


def build_csr(n_units=N_UNITS):
    return build_topology_csr('random_graph', n_units,
                              seed=TOPO_SEED, avg_degree=AVG_DEGREE)


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

    # Full-system readout: [obs; 1; metadata]
    rls = OnlineRLS(N_UNITS + 1 + N_UNITS, 1, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                    reg=RLS_REG)
    preds = np.empty(T)
    kappa = KAPPA_NOMINAL
    kappa_hist = np.empty(T)
    kappa_hist[:] = np.nan

    # Metadata
    m_state = np.zeros(N_UNITS)

    # Plasticity mask (start from the random_graph topology)
    mask = np.zeros((N_UNITS, N_UNITS), dtype=bool)
    for i in range(N_UNITS):
        for e in range(ip0[i], ip0[i + 1]):
            j = int(idx0[e])
            mask[min(i, j), max(i, j)] = True
    n_edges = int(mask.sum())

    use_plasticity = arm != 'no_plasticity'
    rng_noise = np.random.RandomState(seed_idx * 131 + 3)

    # Cumulative disturbance state
    tau_cur = tau.copy()
    ip_c, idx_c, wt_c = ip0, idx0, wt0
    noise_std = 0.0
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
                noise_std = NOISE_SIG
            disturb_idx += 1

        # Homeostat: FTLE -> kappa (with optional future-lambda leak)
        if blk_start % FTLE_EVERY == 0:
            w = min(FTLE_WINDOW, T - blk_start)
            lam, _ = run_pair_ftle_nb(
                x_cur, tau_cur, dt_seq[blk_start:blk_start + w], PW,
                ip_c, idx_c, wt_c, kappa,
                ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                COUPLING_CONTRAST_SELF, 1e-8, 10)
            if (arm == 'leak_ftle'
                    and blk_start + LEAK_FTLE_AHEAD + w < T):
                lam_fut, _ = run_pair_ftle_nb(
                    x_cur, tau_cur,
                    dt_seq[blk_start + LEAK_FTLE_AHEAD:
                           blk_start + LEAK_FTLE_AHEAD + w], PW,
                    ip_c, idx_c, wt_c, kappa,
                    ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                    COUPLING_CONTRAST_SELF, 1e-8, 10)
                lam = (1.0 - LEAK_FRAC) * lam + LEAK_FRAC * lam_fut
            err = float(np.clip(LAMBDA_TARGET - lam, -1.0, 1.0))
            kappa = float(np.clip(kappa + ETA_LAMBDA * err,
                                  KAPPA_MIN, KAPPA_MAX))

        # Trajectory block on the current physics (and mask, if plasticity)
        ip_blk, idx_blk, wt_blk = (adjacency_to_csr(mask) if use_plasticity
                                   else (ip_c, idx_c, wt_c))
        st_b, _, _ = run_trajectory_nb(
            x_cur, tau_cur, dt_seq[blk_start:blk_end], PW,
            ip_blk, idx_blk, wt_blk, kappa,
            ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
            COUPLING_CONTRAST_SELF, 0)
        n_blk = st_b.shape[0]

        # Features: obs (+noise), metadata EMA (with optional leak)
        for j in range(n_blk):
            t = blk_start + j
            f = np.exp(gamma * st_b[j]) / FEATURE_SCALE
            if noise_std > 0:
                f = f + rng_noise.normal(
                    0, noise_std * f.std(axis=0), f.shape)
            if arm == 'leak_metadata' and j + LEAK_K < n_blk:
                f_fut = (np.exp(gamma * st_b[j + LEAK_K]) / FEATURE_SCALE)
                f = (1.0 - LEAK_FRAC) * f + LEAK_FRAC * f_fut
            m_state = (1.0 - 1.0 / TAU_SLOW) * m_state \
                + (1.0 / TAU_SLOW) * f
            feat = np.concatenate([f, [1.0], m_state])
            preds[t] = rls.predict(feat)[0]
            y = target[t]
            if arm == 'leak_rls' and t + LEAK_K < T:
                y = (1.0 - LEAK_FRAC) * y + LEAK_FRAC * target[t + LEAK_K]
            rls.update(feat, y)

        kappa_hist[blk_start:blk_end] = kappa
        x_cur = st_b[-1].copy()

        # Structure plasticity (with optional future-correlation leak)
        if use_plasticity and (blk_start + n_blk) % PLASTICITY_EVERY == 0:
            obs_block = np.exp(gamma * st_b) / FEATURE_SCALE
            z = (obs_block - obs_block.mean(axis=0)) / (
                obs_block.std(axis=0) + 1e-12)
            C = (z.T @ z) / obs_block.shape[0]
            if arm == 'leak_plasticity':
                t_fut0 = blk_start + n_blk
                n_fut = min(PLASTICITY_EVERY, T - t_fut0)
                if n_fut >= 2:
                    st_f, _, _ = run_trajectory_nb(
                        x_cur, tau_cur, dt_seq[t_fut0:t_fut0 + n_fut], PW,
                        ip_blk, idx_blk, wt_blk, kappa,
                        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                        COUPLING_CONTRAST_SELF, 0)
                    obs_f = np.exp(gamma * st_f) / FEATURE_SCALE
                    z_f = (obs_f - obs_f.mean(axis=0)) / (
                        obs_f.std(axis=0) + 1e-12)
                    C_fut = (z_f.T @ z_f) / obs_f.shape[0]
                    C = ((1.0 - LEAK_FRAC_PLASTICITY) * C
                         + LEAK_FRAC_PLASTICITY * C_fut)
            n_grow = max(1, int(PLASTICITY_CHURN * n_edges))
            mask = evolve_mask(mask, C, n_grow)
            n_edges = int(mask.sum())

    # Per-round metrics (S11 protocol)
    var_full = float(target[DISTURB_TIMES[-1]:].var())

    def nmse(seg_p, seg_t):
        v = float(seg_t.var()) if len(seg_t) > 1 else var_full
        return float(np.mean((seg_p - seg_t) ** 2)) / max(v, 1e-12)

    results = {'arm': arm, 'seed_idx': seed_idx,
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
        results[f'r{r}_kappa'] = (float(np.nanmean(kh_seg))
                                  if np.any(~np.isnan(kh_seg))
                                  else float('nan'))

    # MC probe per round (at the kappa settled in that round)
    for r in range(1, 4):
        kp = results[f'r{r}_kappa']
        if np.isnan(kp):
            results[f'r{r}_mc'] = float('nan')
            continue
        x0p = preprogram_vec(ALPHA0, tau_cur)
        rng2 = np.random.RandomState(seed_idx * 999 + r * 10)
        dt_probe = rng2.uniform(2e-6, 20e-6, 3000)
        st_p, _, _ = run_trajectory_nb(
            x0p, tau_cur, dt_probe, PW, ip_blk, idx_blk, wt_blk, kp,
            ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
            COUPLING_CONTRAST_SELF, 500)
        obs_p = np.exp(gamma * st_p) / FEATURE_SCALE
        if noise_std > 0:
            obs_p = obs_p + rng2.normal(0, noise_std * obs_p.std(axis=0),
                                        obs_p.shape)
        results[f'r{r}_mc'] = memory_capacity_heldout(obs_p, dt_probe[500:])

    # Round 0 MC: nominal physics, kappa=25 (S11 anchor)
    x0n = preprogram_vec(ALPHA0, tau)
    rng_n = np.random.RandomState(seed_idx * 999)
    dt_n = rng_n.uniform(2e-6, 20e-6, 3000)
    st_n, _, _ = run_trajectory_nb(
        x0n, tau, dt_n, PW, ip0, idx0, wt0, KAPPA_NOMINAL,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
        COUPLING_CONTRAST_SELF, 500)
    obs_n = np.exp(gamma * st_n) / FEATURE_SCALE
    results['r0_mc'] = memory_capacity_heldout(obs_n, dt_n[500:])

    results['n_edges'] = int(mask.sum())
    return results


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(r['arm'], []).append(r)
    agg = []
    for arm, rs in sorted(groups.items()):
        entry = {'arm': arm, 'n_runs': len(rs)}
        for f in ['r0_nmse', 'r1_nmse', 'r2_nmse', 'r3_nmse',
                  'r1_kappa', 'r2_kappa', 'r3_kappa',
                  'r0_mc', 'r1_mc', 'r2_mc', 'r3_mc']:
            v = np.array([r[f] for r in rs], dtype=float)
            v = v[~np.isnan(v)]
            entry[f + '_mean'] = float(np.mean(v)) if v.size else float('nan')
            entry[f + '_std'] = float(np.std(v)) if v.size else float('nan')
        entry['n_edges_mean'] = float(np.mean([r['n_edges'] for r in rs]))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 100)
    print("S28 RESULTS (Paper B follow-up): causal leak audit on the "
          "disturbance chain")
    print("=" * 100)
    print(f" {'arm':>18} | {'r0_mc':>6} {'r3_mc':>6} {'r3_nmse':>8} | "
          f"{'kappa_r3':>7} {'edges':>6}")
    normal = next((a for a in agg if a['arm'] == 'normal'), None)
    for a in agg:
        delta = ""
        if normal is not None and a['arm'] != 'normal':
            d = a['r3_mc_mean'] - normal['r3_mc_mean']
            dn = a['r3_nmse_mean'] - normal['r3_nmse_mean']
            delta = f" (dMC={d:+.3f}, dNMSE={dn:+.4f})"
        print(f" {a['arm']:>18} | {a['r0_mc_mean']:>6.2f} "
              f"{a['r3_mc_mean']:>6.2f} {a['r3_nmse_mean']:>8.4f} | "
              f"{a['r3_kappa_mean']:>7.1f} {a['n_edges_mean']:>6.0f}{delta}")

    if normal is not None:
        print("\n  Verdict (leak detected if a leak arm improves on normal):")
        for a in agg:
            if a['arm'].startswith('leak_'):
                d = a['r3_mc_mean'] - normal['r3_mc_mean']
                dn = normal['r3_nmse_mean'] - a['r3_nmse_mean']
                print(f"    {a['arm']:>18}: r3 MC delta {d:+.3f}, "
                      f"r3 NMSE delta {dn:+.4f}")


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S28 causal audit on chain "
          f"(quick={quick}, sequential={sequential})")

    # numba warmup (small cell, same code paths)
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

    arms = ['normal', 'leak_rls', 'leak_metadata', 'leak_ftle',
            'leak_plasticity', 'no_plasticity']
    if quick:
        arms = ['normal', 'leak_ftle', 'leak_plasticity']
    n_seeds = 2 if quick else N_SEEDS
    all_args = [(a, s) for a in arms for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seeds={n_seeds}, arms={len(arms)})")

    results = []
    if sequential:
        for i, args in enumerate(all_args):
            results.append(run_single(args))
            done = i + 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 10) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed_idx', 'r0_nmse', 'r1_nmse', 'r2_nmse',
                  'r3_nmse', 'r1_kappa', 'r2_kappa', 'r3_kappa',
                  'r0_mc', 'r1_mc', 'r2_mc', 'r3_mc', 'n_edges',
                  'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'experiment': 'causal leak audit on the S11 disturbance chain '
                      '(Paper B Sec 4.3 follow-up; S13 re-run off the '
                      'ceilinged regime_switch task)',
        'protocol': 'S11 chain verbatim (T=28000, disturbs 7k/14k/21k: '
                    'tau_drift/edge_prune/noise, Mackey-Glass online task)',
        'readout': 'OnlineRLS on [obs; 1; metadata-EMA(tau_slow=500)]',
        'homeostat': {'lambda_target': LAMBDA_TARGET, 'eta': ETA_LAMBDA,
                      'ftle_every': FTLE_EVERY, 'ftle_window': FTLE_WINDOW},
        'plasticity': {'every': PLASTICITY_EVERY, 'churn': PLASTICITY_CHURN},
        'leaks': {'leak_frac': LEAK_FRAC, 'leak_k': LEAK_K,
                  'leak_ftle_ahead': LEAK_FTLE_AHEAD,
                  'leak_frac_plasticity': LEAK_FRAC_PLASTICITY,
                  'semantics': 'verbatim from S13'},
        'arms': arms, 'n_seeds': n_seeds, 'quick': bool(quick),
        'verdict_rule': 'leak arm improving r3 MC or r3 NMSE over normal '
                        'beyond threshold = detected leak',
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
