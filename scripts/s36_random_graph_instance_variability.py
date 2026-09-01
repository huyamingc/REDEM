#!/usr/bin/env python3
"""
S36: Random-graph instance variability (Paper A follow-up).
=============================================================================
Type:           PAPER
Paper Section:  Paper A, Sec. 4.1 / Table 1 follow-up
Experiment:     Table 1's random_graph peak (13.91 held-out MC at kappa=25,
                +53% over the 9.07 uncoupled baseline) was measured on ONE
                fixed Erdos-Renyi graph instance (TOPO_SEED=777). This scan
                repeats the random_graph mode-1 kappa sweep on 9 independent
                graph instances (8 fresh + the original 777) to quantify
                graph-to-graph variability of the peak held-out MC and of
                the peak location kappa*.

Protocol (per instance): the S1 v2 protocol verbatim -- random_graph,
COUPLING_CONTRAST_SELF, kappa in KAPPA_CONTRAST_MODE1, 10 paired substrate
seeds, N=256, T=1200, washout=200, k_max=50. Reuses
substrate_recurrence_characterization.run_single with the module-level
TOPO_SEED set per task in the worker process (Pool chunksize=1, so each
task sets its own instance immediately before the call).

Output files:
  data/s36_topo_instance_variability_v1.csv
  data/s36_topo_instance_variability_v1.json

Usage: python s36_random_graph_instance_variability.py [--quick]
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
import substrate_recurrence_characterization as src
from shallow_trap_array_simulator import gamma, tau0, gen_tau_vec, preprogram_vec
from recurrent_substrate import (
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    run_trajectory_nb, run_pair_ftle_nb, COUPLING_CONTRAST_SELF)

TOPO_SEEDS = [777, 101, 202, 303, 404, 505, 606, 707, 808]
N_SEEDS = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's36_topo_instance_variability_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's36_topo_instance_variability_v1.json')
PHASE_CSV = os.path.join(DATA_DIR, 'substrate_phase_diagram_v2.csv')


def run_wrapped(args):
    """(topo_seed, kappa, seed_idx) -> S1 run row with the graph instance set."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    topo_seed, kappa, seed_idx = args
    src.TOPO_SEED = topo_seed
    row = src.run_single(('random_graph', 'random_graph',
                          COUPLING_CONTRAST_SELF, kappa, seed_idx,
                          src.N_UNITS, src.T_TOTAL, src.N_WASHOUT,
                          src.K_MAX))
    row['topo_seed'] = int(topo_seed)
    return row


def uncoupled_baseline():
    """Uncoupled (parallel, kappa=0) held-out MC mean from the S1 v2 CSV."""
    vals = []
    with open(PHASE_CSV, 'r') as f:
        for r in csv.DictReader(f):
            if r['topology'] == 'parallel' and float(r['kappa']) == 0.0:
                vals.append(float(r['mc_total_test']))
    return float(np.mean(vals))


def main():
    quick = '--quick' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S36 random-graph instance "
          f"variability (quick={quick})")

    # numba warmup in the parent (populates the disk cache that spawned
    # Pool workers reuse)
    tau_w = gen_tau_vec(16, src.CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(8, 10e-6)
    ip_w, idx_w, w_w = build_topology_csr("random_graph", 16)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, w_w, 0.1,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    run_pair_ftle_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, w_w, 0.1,
                     ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                     COUPLING_CONTRAST_SELF, 1e-8, 4)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    topo_seeds = TOPO_SEEDS[:2] if quick else TOPO_SEEDS
    kappas = [20.0, 25.0, 30.0] if quick else src.KAPPA_CONTRAST_MODE1
    n_seeds = 2 if quick else N_SEEDS
    all_args = [(g, k, s) for g in topo_seeds for k in kappas
                for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} ({len(topo_seeds)} graph instances x "
          f"{len(kappas)} kappas x {n_seeds} seeds)")

    results = []
    with Pool(min(cpu_count(), max(1, n_runs))) as pool:
        done = 0
        for res in pool.imap_unordered(run_wrapped, all_args, chunksize=1):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['topo_seed', 'config_name', 'topology', 'coupling_mode',
                  'kappa', 'seed_idx', 'n_units', 'cv_tau', 't_total',
                  'n_washout', 'k_max', 'ftle_per_pulse', 'mc_total',
                  'mc_total_test', 'mc_k0', 'mc_k0_test', 'inter_rms',
                  'alpha_clip_frac', 'g_abs_mean', 'mean_state', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    # ---- per-instance analysis ----
    baseline = uncoupled_baseline() if not quick else float('nan')
    print("\n" + "=" * 100)
    print("S36 RESULTS (Paper A follow-up): random-graph instance variability")
    print("=" * 100)
    print(f"  uncoupled baseline (S1 v2 parallel, held-out MC): {baseline:.2f}")
    print(f" {'topo_seed':>9} | {'peak_mc_test':>12} {'kappa*':>7} | "
          f"{'enh_vs_base':>11}")
    peaks = []
    per_inst = {}
    for g in topo_seeds:
        rs = [r for r in results if r['topo_seed'] == g]
        curve = {}
        for k in kappas:
            vs = [r['mc_total_test'] for r in rs
                  if abs(r['kappa'] - k) < 1e-9]
            curve[k] = float(np.mean(vs)) if vs else float('nan')
        k_star = max(curve, key=lambda k: curve[k])
        peak = curve[k_star]
        peaks.append(peak)
        per_inst[str(g)] = {'peak_mc_test': peak, 'kappa_star': k_star,
                            'mc_test_by_kappa': curve}
        enh = (peak - baseline) / baseline * 100 if baseline == baseline else float('nan')
        print(f" {g:>9} | {peak:>12.2f} {k_star:>7g} | {enh:>+10.1f}%")

    peaks_arr = np.array(peaks)
    print(f"\n  across {len(peaks)} instances: peak held-out MC = "
          f"{peaks_arr.mean():.2f} +/- {peaks_arr.std(ddof=1):.2f} "
          f"(min {peaks_arr.min():.2f}, max {peaks_arr.max():.2f})")
    if baseline == baseline:
        enh_mean = (peaks_arr - baseline) / baseline * 100
        print(f"  enhancement over uncoupled: {enh_mean.mean():+.1f}% "
              f" +/- {enh_mean.std(ddof=1):.1f}%")

    params = {
        'experiment': 'random-graph instance variability on the S1 v2 '
                      'protocol (Paper A Table 1 follow-up)',
        'mechanism': 'src.run_single reused; module-level TOPO_SEED set '
                     'per task in the worker (chunksize=1)',
        'topo_seeds': topo_seeds, 'kappas': kappas, 'n_seeds': n_seeds,
        'n_units': src.N_UNITS, 't_total': src.T_TOTAL,
        'n_washout': src.N_WASHOUT, 'k_max': src.K_MAX,
        'uncoupled_baseline_mc_test': baseline, 'quick': bool(quick),
    }
    out_json = (JSON_PATH if not quick
                else JSON_PATH.replace('.json', '_quick.json'))
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'per_instance': per_inst,
                   'rows': results}, f, indent=2)

    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
