#!/usr/bin/env python3
"""
Structure plasticity benchmark (M4, REDEM S7).
=============================================================================
Type:           PAPER
Paper Section:  New-algorithm project Step S7
Experiment:     Does usage-driven structural rewiring (prune low-co-activity
                edges, grow random novel edges, constant edge count) evolve
                topologies whose memory capacity matches or beats fixed
                topologies -- and does it REPAIR pruned-damage?

Framing (from the S4 gate): the intrinsic/novelty signal's correct role is
STRUCTURE exploration (which edges to try), not readout credit assignment.
The random edge growth here is that novelty exploration; the co-activity
usage correlation is the structural Hebbian retention rule.

Protocol (readout-independent metric = held-out memory capacity):
  1. Start from a topology mask (ring_bidir, random_graph, or damaged).
  2. For 5 evolutionary rounds, each round:
       - drive the substrate with 3000 i.i.d. intervals (kappa=25, mode 1)
       - per-edge usage = Pearson correlation of the two endpoints'
         current-ratio features over the round
       - prune the 20% lowest-|corr| edges, grow 20% random unconnected
         pairs (constant edge count)
  3. After evolution, probe held-out MC (Jaeger, 70/30) at the final
     topology; compare across arms.

Arms (10 seeds):
  fixed_ring        : ring_bidir, no evolution (density 2)
  fixed_random_sp   : sparse Erdos-Renyi (avg degree 2), no evolution
  evolve_ring       : ring_bidir + evolution (constant density 2)
  fixed_random      : random_graph (avg degree 8), no evolution (reference)
  damaged_frozen    : random_graph with 40% edges pruned, no evolution
  repair_damaged    : damaged random_graph + evolution (constant density)

Output files:
  data/s7_structure_plasticity_v1.csv    (one row per run)
  data/s7_structure_plasticity_v1.json   (params + aggregates)

Usage: python structure_plasticity.py [--quick]
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

# ========================== Fixed parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
KAPPA = 25.0

N_ROUNDS = 5
ROUND_LEN = 3000          # i.i.d. drive pulses per evolutionary round
PRUNE_FRAC = 0.20
GROW_FRAC = 0.20
DT_LO, DT_HI = 2e-6, 20e-6

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's7_structure_plasticity_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's7_structure_plasticity_v1.json')


def initial_mask(arm, rng):
    """(mask, n_edges_target) for each arm."""
    if arm == 'fixed_ring' or arm == 'evolve_ring' or arm == 'evolve_ring_gentle':
        ip, idx, wt = build_topology_csr('ring_bidir', N_UNITS)
        mask = np.zeros((N_UNITS, N_UNITS), dtype=bool)
        for i in range(N_UNITS):
            for e in range(ip[i], ip[i + 1]):
                j = int(idx[e])
                mask[min(i, j), max(i, j)] = True
        return mask, int(mask.sum())
    if arm == 'fixed_random_sp':
        p = 2.0 / (N_UNITS - 1)
        m = np.triu(rng.rand(N_UNITS, N_UNITS) < p, 1)
        return m, int(m.sum())
    if arm == 'fixed_random' or arm == 'damaged_frozen' or arm == 'repair_damaged':
        ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                         seed=TOPO_SEED, avg_degree=AVG_DEGREE)
        mask = np.zeros((N_UNITS, N_UNITS), dtype=bool)
        for i in range(N_UNITS):
            for e in range(ip[i], ip[i + 1]):
                j = int(idx[e])
                mask[min(i, j), max(i, j)] = True
        if arm in ('damaged_frozen', 'repair_damaged'):
            # prune 40% of the undirected edges (upper-triangle convention)
            edges = np.argwhere(np.triu(mask, 1))
            keep = rng.rand(edges.shape[0]) > 0.4
            mask = np.zeros_like(mask)
            for (i, j), kp in zip(edges, keep):
                if kp:
                    mask[i, j] = True
        return mask, int(mask.sum())
    raise ValueError(arm)


def evolve_mask(mask, corr_matrix, n_grow):
    """Functional-connectivity-driven rewiring (Hebbian structural
    plasticity): prune the n_grow edges with the LOWEST |corr| and grow
    edges between the n_grow UNCONNECTED pairs with the HIGHEST |corr|
    ("fire together, wire together" for missing connections). Constant
    edge count. corr_matrix: (N, N) pairwise feature correlation.
    """
    m = mask.copy()
    undir = np.triu(m, 1)
    edges = np.argwhere(undir)
    if len(edges) == 0:
        return m
    usages = np.array([abs(corr_matrix[int(i), int(j)]) for (i, j) in edges])
    order = np.argsort(usages)
    to_prune = edges[order[:n_grow]]
    for (i, j) in to_prune:
        m[i, j] = False
    free = np.argwhere(np.triu(~m, 1))
    if len(free) == 0:
        return m
    free_corr = np.array([abs(corr_matrix[int(i), int(j)]) for (i, j) in free])
    pick_order = np.argsort(free_corr)[::-1][:min(n_grow, len(free))]
    for idx in pick_order:
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
    mask, n_edges = initial_mask(arm, rng)

    if arm in ('fixed_ring', 'fixed_random_sp', 'fixed_random', 'damaged_frozen'):
        # no evolution: run one 3000-pulse round for the probe drive
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

    # evolutionary arms
    mc_initial = 0.0
    for rnd in range(N_ROUNDS):
        dt = rng.uniform(DT_LO, DT_HI, ROUND_LEN)
        ip, idx, wt = adjacency_to_csr(mask)
        states, _, _ = run_trajectory_nb(x0, tau, dt, PW, ip, idx, wt,
                                         KAPPA, ALPHA0, ALPHA_MIN, ALPHA_MAX,
                                         gamma, COUPLING_CONTRAST_SELF, 200)
        # per-edge usage: Pearson corr of endpoint features over the round
        # (states is already post-washout, aligned with dt[200:])
        obs = np.exp(gamma * states) / FEATURE_SCALE
        z = (obs - obs.mean(axis=0)) / (obs.std(axis=0) + 1e-12)
        C = (z.T @ z) / obs.shape[0]
        if rnd == 0:
            mc_initial = memory_capacity_heldout(obs, dt[200:])
        churn = 0.05 if arm == 'evolve_ring_gentle' else GROW_FRAC
        n_grow = max(1, int(churn * (np.triu(mask, 1).sum())))
        mask = evolve_mask(mask, C, n_grow)
        x0 = states[-1].copy()

    # final probe
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
    print("\n" + "=" * 100)
    print("S7 RESULTS (mean over seeds): held-out MC")
    print("=" * 100)
    print(f"  {'arm':<16} {'n_edges':>8} {'degree':>7} | {'MC_initial':>11} | "
          f"{'MC_final':>9}")
    order = ['fixed_ring', 'fixed_random_sp', 'evolve_ring', 'evolve_ring_gentle',
             'fixed_random', 'damaged_frozen', 'repair_damaged']
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
    print(f"[{time.strftime('%H:%M:%S')}] START S7 structure plasticity "
          f"(quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16,
                                           seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    arms = ['fixed_ring', 'fixed_random_sp', 'evolve_ring', 'evolve_ring_gentle',
            'fixed_random', 'damaged_frozen', 'repair_damaged']
    if quick:
        arms = ['fixed_ring', 'evolve_ring', 'repair_damaged']
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
        'prune_frac': PRUNE_FRAC, 'grow_frac': GROW_FRAC,
        'avg_degree': AVG_DEGREE, 'n_seeds': n_seeds, 'quick': bool(quick),
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
