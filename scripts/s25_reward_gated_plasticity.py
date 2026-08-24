#!/usr/bin/env python3
"""
Reward-gated structural plasticity (M4 novelty hypothesis, REDEM S25).
=============================================================================
Type:           PAPER
Experiment:     E2: can a novelty intrinsic reward (temporal activity
                variance) guide structure-level exploration as well as the
                functional-connectivity correlation used by M4? 4 arms x
                10 seeds, S7-style evolutionary MC protocol.

Background: S3/S4 established that reward/novelty signals fail at the
READOUT (no error channel); Paper B re-purposes them conceptually to the
STRUCTURE level (M4), but the implemented M4 is correlation-guided only.
This experiment tests the structure-level claim directly: with identical
churn (5% per round, the gentle regime), does novelty-reward edge selection
produce topologies with memory capacity comparable to correlation-guided
selection -- and does either beat random structural change?

Protocol (identical to S7): start from ring_bidir; for 5 rounds, drive the
substrate with 3000 i.i.d. intervals (kappa=25); per round, prune n_grow
edges / grow n_grow unconnected pairs (constant edge count) according to
the arm's selection rule; probe held-out MC (70/30) at the final topology.

Selection rules:
  evolve_corr    : prune lowest |corr|, grow highest |corr| (M4 as shipped)
  evolve_novelty : per-unit novelty r_i = std(feature_i) over the round;
                   edge score = mean(r_i, r_j); prune lowest, grow highest
  evolve_random  : random prune/grow (control for structural change per se)

Output files:
  data/s25_reward_gated_plasticity_v1.csv
  data/s25_reward_gated_plasticity_v1.json

Usage: python s25_reward_gated_plasticity.py [--quick]
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
    adjacency_to_csr, run_trajectory_nb)
from online_readout import memory_capacity_heldout
from structure_plasticity import evolve_mask

# ========================== Fixed parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
KAPPA = 25.0

N_ROUNDS = 5
ROUND_LEN = 3000
CHURN = 0.05                # gentle churn fraction per round
DT_LO, DT_HI = 2e-6, 20e-6

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's25_reward_gated_plasticity_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's25_reward_gated_plasticity_v1.json')


def ring_mask():
    ip, idx, wt = build_topology_csr('ring_bidir', N_UNITS)
    mask = np.zeros((N_UNITS, N_UNITS), dtype=bool)
    for i in range(N_UNITS):
        for e in range(ip[i], ip[i + 1]):
            j = int(idx[e])
            mask[min(i, j), max(i, j)] = True
    return mask


def novelty_evolve(mask, novelty, n_grow):
    """Novelty-reward-guided rewiring: prune existing edges with the LOWEST
    mean endpoint novelty, grow unconnected pairs with the HIGHEST mean
    endpoint novelty. novelty: (N,) per-unit intrinsic reward."""
    m = mask.copy()
    undir = np.triu(m, 1)
    edges = np.argwhere(undir)
    if len(edges) == 0:
        return m
    scores = np.array([0.5 * (novelty[int(i)] + novelty[int(j)])
                       for (i, j) in edges])
    order = np.argsort(scores)
    to_prune = edges[order[:min(n_grow, len(edges))]]
    for (i, j) in to_prune:
        m[i, j] = False
    free = np.argwhere(np.triu(~m, 1))
    if len(free) == 0:
        return m
    free_scores = np.array([0.5 * (novelty[int(i)] + novelty[int(j)])
                            for (i, j) in free])
    pick = np.argsort(free_scores)[::-1][:min(n_grow, len(free))]
    for idx in pick:
        i, j = free[idx]
        m[i, j] = True
    return m


def random_evolve(mask, rng, n_grow):
    """Random prune/grow control (constant edge count)."""
    m = mask.copy()
    edges = np.argwhere(np.triu(m, 1))
    if len(edges) > 0:
        pick = rng.choice(len(edges), size=min(n_grow, len(edges)),
                          replace=False)
        for idx in pick:
            i, j = edges[idx]
            m[i, j] = False
    free = np.argwhere(np.triu(~m, 1))
    if len(free) > 0:
        pick = rng.choice(len(free), size=min(n_grow, len(free)),
                          replace=False)
        for idx in pick:
            i, j = free[idx]
            m[i, j] = True
    return m


def run_single(args):
    """(arm, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    arm, seed_idx = args
    t0 = time.time()

    rng = np.random.RandomState(seed_idx * 71 + 3)
    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    mask = ring_mask()
    n_edges = int(mask.sum())

    if arm == 'fixed_ring':
        dt_probe = rng.uniform(DT_LO, DT_HI, 3000)
        ip, idx, wt = adjacency_to_csr(mask)
        states, _, _ = run_trajectory_nb(x0, tau, dt_probe, PW, ip, idx, wt,
                                         KAPPA, ALPHA0, ALPHA_MIN, ALPHA_MAX,
                                         gamma, COUPLING_CONTRAST_SELF, 500)
        obs = np.exp(gamma * states) / FEATURE_SCALE
        mc = memory_capacity_heldout(obs, dt_probe[500:])
        return {'arm': arm, 'seed_idx': seed_idx, 'n_edges': n_edges,
                'mc_final': mc, 'mc_initial': mc,
                'degree_mean': float(np.mean(np.diff(ip))),
                'runtime_s': time.time() - t0}

    mc_initial = 0.0
    for rnd in range(N_ROUNDS):
        dt = rng.uniform(DT_LO, DT_HI, ROUND_LEN)
        ip, idx, wt = adjacency_to_csr(mask)
        states, _, _ = run_trajectory_nb(x0, tau, dt, PW, ip, idx, wt,
                                         KAPPA, ALPHA0, ALPHA_MIN, ALPHA_MAX,
                                         gamma, COUPLING_CONTRAST_SELF, 200)
        obs = np.exp(gamma * states) / FEATURE_SCALE
        z = (obs - obs.mean(axis=0)) / (obs.std(axis=0) + 1e-12)
        C = (z.T @ z) / obs.shape[0]
        if rnd == 0:
            mc_initial = memory_capacity_heldout(obs, dt[200:])
        n_grow = max(1, int(CHURN * (np.triu(mask, 1).sum())))
        if arm == 'evolve_corr':
            mask = evolve_mask(mask, C, n_grow)
        elif arm == 'evolve_novelty':
            novelty = obs.std(axis=0)
            mask = novelty_evolve(mask, novelty, n_grow)
        elif arm == 'evolve_random':
            mask = random_evolve(mask, rng, n_grow)
        x0 = states[-1].copy()

    dt_probe = rng.uniform(DT_LO, DT_HI, 3000)
    ip, idx, wt = adjacency_to_csr(mask)
    states, _, _ = run_trajectory_nb(x0, tau, dt_probe, PW, ip, idx, wt,
                                     KAPPA, ALPHA0, ALPHA_MIN, ALPHA_MAX,
                                     gamma, COUPLING_CONTRAST_SELF, 500)
    obs = np.exp(gamma * states) / FEATURE_SCALE
    mc_final = memory_capacity_heldout(obs, dt_probe[500:])
    degs = np.diff(adjacency_to_csr(mask)[0])
    return {'arm': arm, 'seed_idx': seed_idx, 'n_edges': int(mask.sum()),
            'mc_initial': mc_initial, 'mc_final': mc_final,
            'degree_mean': float(degs.mean()), 'runtime_s': time.time() - t0}


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(r['arm'], []).append(r)
    agg = []
    for arm, rs in sorted(groups.items()):
        entry = {'arm': arm, 'n_runs': len(rs)}
        for f in ['mc_final', 'mc_initial', 'degree_mean']:
            v = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.mean(v))
            entry[f + '_std'] = float(np.std(v))
        entry['n_edges_mean'] = float(np.mean([r['n_edges'] for r in rs]))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 96)
    print("S25 RESULTS (mean over seeds): reward-gated vs correlation plasticity")
    print("=" * 96)
    print(f"  {'arm':<16} {'n_edges':>8} {'degree':>7} | {'MC_initial':>11} | "
          f"{'MC_final':>9}")
    order = ['fixed_ring', 'evolve_random', 'evolve_novelty', 'evolve_corr']
    for arm in order:
        a = next((x for x in agg if x['arm'] == arm), None)
        if a is None:
            continue
        print(f"  {a['arm']:<16} {a['n_edges_mean']:>8.0f} "
              f"{a['degree_mean_mean']:>7.2f} | "
              f"{a['mc_initial_mean']:>11.2f} | {a['mc_final_mean']:>9.2f}")


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S25 reward-gated plasticity "
          f"(quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('ring_bidir', 16)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    arms = ['fixed_ring', 'evolve_random', 'evolve_novelty', 'evolve_corr']
    if quick:
        arms = ['fixed_ring', 'evolve_novelty', 'evolve_corr']
    n_seeds = 2 if quick else N_SEEDS
    all_args = [(a, s) for a in arms for s in range(n_seeds)]
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
    fieldnames = ['arm', 'seed_idx', 'n_edges', 'degree_mean',
                  'mc_initial', 'mc_final', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'kappa': KAPPA, 'n_rounds': N_ROUNDS, 'round_len': ROUND_LEN,
        'churn': CHURN, 'novelty': 'per-unit std of feature over round',
        'selection': {'evolve_corr': 'prune/grow by |corr| (M4 as shipped)',
                      'evolve_novelty': 'prune/grow by mean endpoint novelty',
                      'evolve_random': 'random prune/grow control'},
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
