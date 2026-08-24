#!/usr/bin/env python3
"""
S27: Clip-range ablation and fine kappa grid for the order-chaos transition.
=============================================================================
Type:           PAPER
Paper Section:  Paper A, Sec. 3 (phase diagram) follow-up
Experiment:     Two open questions about the sharp order-chaos transition at
                kappa* in (25,30) for the negative-feedback (mode 1)
                topologies:

  Q1 (sharpness): where exactly is kappa*? The committed v2 grid is
      {...,20,25,30,...}; this script fills kappa = 21..29 at unit
      resolution (plus 15/40 anchors) so the crossing can be pinned and
      the transition width (seed spread) quantified.

  Q2 (mechanism): is the chaos driven by the TOPOLOGY COUPLING or by the
      physical alpha_eff CLIP (clip onset coincides with the transition,
      clip fraction 0.3-0.6)? We re-run the fine grid under three clip
      ranges: alpha_max in {0.1 (default), 0.2, 0.5} (alpha_min fixed).
      If kappa* shifts right as the clip widens -> clip-induced chaos
      (the paper's "physics knob" claim weakens); if kappa* is invariant
      -> coupling-driven chaos (claim strengthens).

  Protocol: identical to the v2 phase diagram (paired tau draws per seed,
  fixed topology structure TOPO_SEED, Benettin FTLE, held-out Jaeger MC);
  only the kappa grid and alpha_max differ. 10 seeds per cell.

Output files:
  data/s27_clip_kappa_fine_v1.csv    (one row per run)
  data/s27_clip_kappa_fine_v1.json   (params + per-config aggregates)

Usage: python s27_clip_kappa_fine.py [--quick] [--sequential]
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
    COUPLING_CONTRAST_SELF, PW, ALPHA0, ALPHA_MIN, ALPHA_MAX,
    build_topology_csr, run_trajectory_nb, run_pair_ftle_nb)
from substrate_recurrence_characterization import (
    memory_capacity, N_UNITS, CV_TAU, T_TOTAL, N_WASHOUT, K_MAX,
    RIDGE_LAMBDA, DT_LO, DT_HI, TOPO_SEED, LATERAL_RADIUS, AVG_DEGREE,
    EPS_BENETTIN, RENORM_EVERY)

# ========================== Grid ==========================
TOPOLOGIES = ['ring_bidir', 'lateral_ring', 'random_graph']
ALPHA_MAX_LIST = [0.1, 0.2, 0.5]     # default 0.1; widen to test clip role
KAPPA_FINE = [15.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
              28.0, 29.0, 30.0, 40.0]
N_SEEDS = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's27_clip_kappa_fine_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's27_clip_kappa_fine_v1.json')


# ========================== Single cell ==========================

def run_single(args):
    """One (topology, alpha_max, kappa, seed) cell."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    topo, alpha_max, kappa, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    rng_main = np.random.RandomState(seed_idx * 77 + 13)
    rng_alt = np.random.RandomState(seed_idx * 77 + 13 + 5001)
    dt_main = rng_main.uniform(DT_LO, DT_HI, T_TOTAL)
    dt_alt = rng_alt.uniform(DT_LO, DT_HI, T_TOTAL)
    indptr, indices, wts = build_topology_csr(
        topo, N_UNITS, seed=TOPO_SEED, lateral_radius=LATERAL_RADIUS,
        avg_degree=AVG_DEGREE)

    states, clip_frac, g_abs = run_trajectory_nb(
        x0, tau, dt_main, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, alpha_max, gamma, COUPLING_CONTRAST_SELF, N_WASHOUT)
    states_alt, _, _ = run_trajectory_nb(
        x0, tau, dt_alt, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, alpha_max, gamma, COUPLING_CONTRAST_SELF, N_WASHOUT)
    inter_rms = float(np.sqrt(np.mean((states_alt - states) ** 2)))
    ftle, _ = run_pair_ftle_nb(
        x0, tau, dt_main, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, alpha_max, gamma, COUPLING_CONTRAST_SELF,
        EPS_BENETTIN, RENORM_EVERY)
    obs = np.exp(gamma * states)
    mc = memory_capacity(obs, dt_main, N_WASHOUT, K_MAX, RIDGE_LAMBDA)

    return {
        'topology': topo,
        'alpha_max': float(alpha_max),
        'kappa': float(kappa),
        'seed_idx': int(seed_idx),
        'ftle_per_pulse': float(ftle),
        'mc_total': mc['mc_total_train'],
        'mc_total_test': mc['mc_total_test'],
        'mc_k0_test': mc['mc_k0_test'],
        'inter_rms': inter_rms,
        'alpha_clip_frac': float(clip_frac),
        'g_abs_mean': float(g_abs),
        'runtime_s': time.time() - t0,
    }


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        key = (r['topology'], r['alpha_max'], r['kappa'])
        groups.setdefault(key, []).append(r)
    agg = []
    for (topo, amax, kappa), rs in sorted(groups.items()):
        def ms(field):
            vals = np.array([r[field] for r in rs], dtype=float)
            return (float(vals.mean()),
                    float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        ftle_m, ftle_s = ms('ftle_per_pulse')
        mc_m, mc_s = ms('mc_total_test')
        agg.append({
            'topology': topo, 'alpha_max': amax, 'kappa': kappa,
            'n_runs': len(rs),
            'ftle_mean': ftle_m, 'ftle_std': ftle_s,
            'mc_total_test_mean': mc_m, 'mc_total_test_std': mc_s,
            'mc_k0_test_mean': ms('mc_k0_test')[0],
            'inter_rms_mean': ms('inter_rms')[0],
            'alpha_clip_frac_mean': ms('alpha_clip_frac')[0],
            'g_abs_mean_mean': ms('g_abs_mean')[0],
        })
    return agg


def kappa_star(agg, topo, amax):
    """Linear-interpolated crossing of mean FTLE through zero."""
    rows = sorted([a for a in agg if a['topology'] == topo
                   and a['alpha_max'] == amax], key=lambda a: a['kappa'])
    for i in range(len(rows) - 1):
        f0, f1 = rows[i]['ftle_mean'], rows[i + 1]['ftle_mean']
        if f0 <= 0.0 < f1:
            k0, k1 = rows[i]['kappa'], rows[i + 1]['kappa']
            return k0 + (k1 - k0) * (0.0 - f0) / (f1 - f0)
    return float('nan')


def print_table(agg):
    print("\n" + "=" * 108)
    print("S27 RESULTS (Paper A follow-up): clip-range ablation x fine kappa")
    print("=" * 108)
    for topo in TOPOLOGIES:
        print(f"\n--- {topo} ---")
        for amax in ALPHA_MAX_LIST:
            rows = sorted([a for a in agg if a['topology'] == topo
                           and a['alpha_max'] == amax], key=lambda a: a['kappa'])
            ks = kappa_star(agg, topo, amax)
            line = ' | '.join(
                'k={0:>4.0f} ftle={1:+.3f}±{2:.2f} mc={3:5.2f}±{4:4.2f} '
                'clip={5:.2f}'.format(a['kappa'], a['ftle_mean'],
                                      a['ftle_std'], a['mc_total_test_mean'],
                                      a['mc_total_test_std'],
                                      a['alpha_clip_frac_mean'])
                for a in rows)
            print(f"  amax={amax}: {line}")
            print(f"    -> kappa* (ftle cross 0): {ks:.1f}")
    print("-" * 108)
    print("Read: if kappa* is invariant to alpha_max -> coupling-driven "
          "chaos (claim strengthened); if it shifts right -> clip-driven.")


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S27 clip-kappa fine "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    all_args = [(topo, amax, kappa, s)
                for topo in TOPOLOGIES for amax in ALPHA_MAX_LIST
                for kappa in KAPPA_FINE for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} ({len(TOPOLOGIES)} topo x "
          f"{len(ALPHA_MAX_LIST)} alpha_max x {len(KAPPA_FINE)} kappa "
          f"x{n_seeds} seeds)")

    results = []
    if sequential:
        for i, args in enumerate(all_args):
            results.append(run_single(args))
            if (i + 1) % max(1, n_runs // 10) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
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

    agg = aggregate(results)
    print_table(agg)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['topology', 'alpha_max', 'kappa', 'seed_idx',
                  'ftle_per_pulse', 'mc_total', 'mc_total_test', 'mc_k0_test',
                  'inter_rms', 'alpha_clip_frac', 'g_abs_mean', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    ks_summary = {f'{t}_amax{amax}': kappa_star(agg, t, amax)
                  for t in TOPOLOGIES for amax in ALPHA_MAX_LIST}
    params = {
        'experiment': 'clip-range ablation x fine kappa grid (Paper A '
                      'phase-diagram follow-up)',
        'protocol': 'identical to v2 phase diagram (paired tau draws per '
                    'seed, TOPO_SEED fixed structure, Benettin FTLE, '
                    'held-out Jaeger MC); only kappa grid and alpha_max '
                    'differ',
        'topologies': TOPOLOGIES,
        'alpha_max_list': ALPHA_MAX_LIST,
        'alpha_min': ALPHA_MIN,
        'kappa_fine': KAPPA_FINE,
        'n_seeds': n_seeds,
        'kappa_star_ftle_crossing': ks_summary,
        'interpretation': 'kappa* invariant to alpha_max -> coupling-driven '
                          'chaos; shifting right -> clip-induced chaos',
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
