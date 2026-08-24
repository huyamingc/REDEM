#!/usr/bin/env python3
"""
S30: N=1024 integrated-system replication at 10 seeds (Paper B follow-up).
=============================================================================
Type:           PAPER
Paper Section:  Paper B, Sec. 4.4 (integration and baselines) follow-up
Experiment:     The committed S8 N=1024 confirmation ran baseline vs full at
                3 seeds only (docstring: "~2min/run, keep the confirmation
                lean"). This script raises the N=1024 confirmation to the
                full 10-seed paired discipline so that "persisting at
                N=1024" (abstract: 0.998 vs 0.976) supports a paired test.

Protocol: identical to S8 (regime_switch task, RLS fast+slow readout,
homeostat, plasticity; seed draws indexed by seed_idx so baseline/full are
paired per seed). run_single is imported verbatim from
integrated_benchmark.py; only the (arm, seed, n_units) grid differs.

Arms (10 seeds, N=1024):
  baseline : RLS fast-only, kappa=25, fixed random_graph (S2/S5 base)
  full     : RLS + dual metadata + lambda-homeostat + gentle plasticity

Output files:
  data/s30_integrated_1024_v1.csv
  data/s30_integrated_1024_v1.json

Usage: python s30_integrated_1024.py [--quick] [--sequential]
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
from integrated_benchmark import (
    run_single, aggregate, N_UNITS, CV_TAU, TOPO_SEED, KAPPA_NOMINAL,
    TAU_SLOW, LAMBDA_TARGET, PLASTICITY_EVERY, PLASTICITY_CHURN)
from shallow_trap_array_simulator import gamma, tau0, gen_tau_vec, preprogram_vec
from recurrent_substrate import (
    COUPLING_CONTRAST_SELF, PW, ALPHA0, ALPHA_MIN, ALPHA_MAX,
    build_topology_csr, run_trajectory_nb, run_pair_ftle_nb)

N_1024 = 1024
N_SEEDS = 10
ARMS_1024 = ['baseline', 'full']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's30_integrated_1024_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's30_integrated_1024_v1.json')


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    n_workers = None
    if '--workers' in sys.argv:
        i = sys.argv.index('--workers')
        n_workers = int(sys.argv[i + 1])
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S30 N=1024 replication "
          f"(quick={quick}, sequential={sequential}, "
          f"workers={n_workers or 'auto'})")

    # numba warmup (small cell)
    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16, seed=TOPO_SEED)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_NOMINAL,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = 2 if quick else N_SEEDS
    all_args = [(arm, s, N_1024) for arm in ARMS_1024 for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} ({len(ARMS_1024)} arms x{n_seeds} seeds, "
          f"N={N_1024})")

    results = []
    if sequential:
        for i, args in enumerate(all_args):
            results.append(run_single(args))
            if (i + 1) % max(1, n_runs // 5) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        n_proc = (min(n_workers, cpu_count())
                  if n_workers else min(cpu_count(), max(1, n_runs)))
        with Pool(n_proc) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 5) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed_idx', 'n_units', 'overall_acc', 'steady_acc',
                  'adapt_time', 'kappa_settled', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    print("\n" + "=" * 100)
    print("S30 RESULTS (Paper B follow-up): N=1024 at 10 seeds")
    print("=" * 100)
    for a in sorted(agg, key=lambda x: x['arm']):
        print(f"  {a['arm']:<14} | acc {a['overall_acc_mean']:.4f} "
              f"+/- {a['overall_acc_std']:.4f} | steady "
              f"{a['steady_acc_mean']:.4f} | adapt "
              f"{a['adapt_time_mean']:.1f} | kappa {a['kappa_settled_mean']:.1f}")

    # paired full vs baseline
    full = {r['seed_idx']: r for r in results if r['arm'] == 'full'}
    base = {r['seed_idx']: r for r in results if r['arm'] == 'baseline'}
    ds = [full[s]['overall_acc'] - base[s]['overall_acc']
          for s in range(n_seeds) if s in full and s in base]
    ds = np.array(ds)
    t_p = ds.mean() / (ds.std(ddof=1) / np.sqrt(ds.size)) if ds.std(ddof=1) > 0 else float('nan')
    print(f"\n  paired full-baseline acc: {ds.mean():+.4f} +/- "
          f"{ds.std(ddof=1):.4f}, n_pos={int(np.sum(ds > 0))}/{ds.size}, "
          f"t={t_p:+.2f}")
    print(f"  (S8 committed N=1024 at 3 seeds: baseline 0.9764, full 0.9977)")

    params = {
        'experiment': 'N=1024 integrated-system replication at 10 seeds '
                      '(S8 N=1024 confirmation upgrade)',
        'protocol': 'verbatim from S8 (regime_switch task, RLS fast+slow '
                    'readout, homeostat lambda_target=-0.02, plasticity '
                    '5% churn / 2k pulses); seeds paired per seed_idx',
        'n_units': N_1024, 'arms': ARMS_1024, 'n_seeds': n_seeds,
        's8_reference': 'N=1024 baseline 0.97637 (3 seeds), full 0.99770 '
                        '(3 seeds)',
        'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
