#!/usr/bin/env python3
"""
Task-level CV sweep (Paper A CV-as-design-knob claim, REDEM S10).
=============================================================================
Type:           PAPER
Paper Section:  Paper A Discussion: "the CV sweep is characterized at the
                kernel level only" -- now extended to the task level.
Experiment:     Held-out memory capacity (Jaeger, 70/30) of the substrate at
                the memory-optimal operating points, vs the spectrum width
                CV in {0.1, 0.2, 0.4}. Configs: parallel (uncoupled),
                random_graph at kappa=25 (near-edge optimum), ring_bidir at
                kappa=20. The forgetting-kernel theory predicts the 1/e
                horizon stays pinned while the tail weight grows with CV;
                at the task level this should translate to modest MC gains
                at higher CV (heavier tail -> more retained old lags).

Output files:
  data/s10_cv_sweep_v1.csv    (one row per run)
  data/s10_cv_sweep_v1.json   (params + aggregates)

Usage: python cv_sweep.py [--quick]
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
    COUPLING_NONE, COUPLING_CONTRAST_SELF,
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    run_trajectory_nb)
from online_readout import memory_capacity_heldout

N_UNITS = 256
N_SEEDS = 10
FEATURE_SCALE = 10.0
TOPO_SEED = 777
AVG_DEGREE = 8
T_TOTAL = 3000
WASHOUT = 500

# (config_name, topo_csr, mode, kappa)
CONFIGS = [
    ('parallel', 'parallel', COUPLING_NONE, 0.0),
    ('random_graph_k25', 'random_graph', COUPLING_CONTRAST_SELF, 25.0),
    ('ring_bidir_k20', 'ring_bidir', COUPLING_CONTRAST_SELF, 20.0),
]
CVS = [0.1, 0.2, 0.4]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's10_cv_sweep_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's10_cv_sweep_v1.json')


def run_single(args):
    """(config_name, topo, mode, kappa, cv, seed_idx) -> dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    config, topo, mode, kappa, cv, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, cv, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    ip, idx, wt = build_topology_csr(topo, N_UNITS, seed=TOPO_SEED,
                                     avg_degree=AVG_DEGREE)
    rng = np.random.RandomState(seed_idx * 31 + 7)
    dt_seq = rng.uniform(2e-6, 20e-6, T_TOTAL)
    states, _, _ = run_trajectory_nb(x0, tau, dt_seq, PW, ip, idx, wt,
                                     kappa, ALPHA0, ALPHA_MIN, ALPHA_MAX,
                                     gamma, mode, WASHOUT)
    obs = np.exp(gamma * states) / FEATURE_SCALE
    mc = memory_capacity_heldout(obs, dt_seq[WASHOUT:])
    return {'config': config, 'cv': float(cv), 'seed_idx': seed_idx,
            'mc_heldout': mc, 'runtime_s': time.time() - t0}


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['config'], r['cv']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        c, cv = key
        vals = np.array([r['mc_heldout'] for r in rs], dtype=float)
        agg.append({'config': c, 'cv': cv, 'n_runs': len(rs),
                    'mc_mean': float(vals.mean()), 'mc_std': float(vals.std())})
    return agg


def print_table(agg):
    print("\n" + "=" * 80)
    print("S10 CV sweep: held-out MC vs spectrum width")
    print("=" * 80)
    for c in ['parallel', 'random_graph_k25', 'ring_bidir_k20']:
        rows = [a for a in agg if a['config'] == c]
        line = f"  {c:<18}: " + " | ".join(
            f"CV={a['cv']}: MC={a['mc_mean']:.2f}±{a['mc_std']:.2f}"
            for a in sorted(rows, key=lambda x: x['cv']))
        print(line)


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S10 CV sweep (quick={quick})")

    n_seeds = 2 if quick else N_SEEDS
    cvs = [0.1, 0.4] if quick else CVS
    all_args = [(c, t, m, k, cv, s)
                for c, t, m, k in CONFIGS for cv in cvs
                for s in range(n_seeds)]
    print(f"total runs: {len(all_args)}")

    results = []
    with Pool(min(cpu_count(), max(1, len(all_args)))) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=2):
            results.append(res)
            done += 1
            if done % max(1, len(all_args) // 5) == 0 or done == len(all_args):
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{len(all_args)}",
                      flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['config', 'cv', 'seed_idx', 'mc_heldout', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {'n_units': N_UNITS, 'cv_list': CVS, 'configs': CONFIGS,
              't_total': T_TOTAL, 'washout': WASHOUT, 'n_seeds': n_seeds,
              'quick': bool(quick)}
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
