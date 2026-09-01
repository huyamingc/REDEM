#!/usr/bin/env python3
"""
S34: Causal-leak sensitivity scan (Paper B follow-up).
=============================================================================
Type:           PAPER
Paper Section:  Paper B, Sec. 4.3 (causal audit) follow-up
Experiment:     The S28 chain-protocol audit found no significant benefit
                from 1% future leaks (FTLE) / 10% future correlation
                (plasticity). The audit flagged that the leak magnitudes
                are small; this scan asks whether LARGER leaks change the
                verdict, i.e. how much future information the audit can
                actually resolve.

Arms (10 seeds; the S28 protocol verbatim, leaks parameterized):
  leak_ftle       : frac x FTLE-leak horizon in
                    {(0.01, 400) [S28 baseline], (0.10, 400),
                     (0.10, 200), (0.10, 50)}
                    run_wrapped sets s28.LEAK_FTLE_AHEAD = k for leak_ftle
                    arms, making the horizon a genuine scan dimension
                    (s28 hard-codes LEAK_FTLE_AHEAD=400).
  leak_plasticity : frac in {0.10 [S28 baseline], 0.30, 0.50}

Verdict rule: if r3 MC or r3 NMSE improves over the S28 'normal' arm
(bootstrap CI excludes 0) at any larger leak, the audit's sensitivity is
bounded and the "causally clean" claim is scoped to small leaks; if even
10-50x larger leaks show no benefit, the claim is robust.

Mechanism: reuses s28_causal_audit_chain.run_single by setting its
module-level leak parameters per task in the worker process (Pool
chunksize=1, so each task sets its own values immediately before the
call; the base 'normal' arm is unaffected and reproduced by default
constants).

Output files:
  data/s34_leak_sensitivity_v1.csv
  data/s34_leak_sensitivity_v1.json

Usage: python s34_leak_sensitivity.py [--quick] [--sequential]
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
import s28_causal_audit_chain as s28m
from s28_causal_audit_chain import run_single as s28_run_single

N_SEEDS = 10

# (arm, leak_frac, leak_k, leak_frac_plasticity)
# For leak_ftle arms leak_k IS the FTLE leak horizon (s28.LEAK_FTLE_AHEAD);
# for other arms it is the unused metadata/RLS lookahead placeholder.
ARMS = [
    ('leak_ftle', 0.01, 400, 0.10),    # S28 baseline (frac=0.01, horizon=400)
    ('leak_ftle', 0.10, 400, 0.10),    # 10x leak, S28 horizon
    ('leak_ftle', 0.10, 200, 0.10),    # 10x leak, shorter horizon
    ('leak_ftle', 0.10, 50, 0.10),     # 10x leak, shortest horizon
    ('leak_plasticity', 0.01, 50, 0.10),  # S28 baseline
    ('leak_plasticity', 0.01, 50, 0.30),
    ('leak_plasticity', 0.01, 50, 0.50),
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's34_leak_sensitivity_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's34_leak_sensitivity_v1.json')


def run_wrapped(args):
    """(arm, frac, k, pfrac, seed) -> S28 row with parameterized leaks."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, frac, k, pfrac, seed = args
    s28m.LEAK_FRAC = frac
    s28m.LEAK_K = k
    s28m.LEAK_FRAC_PLASTICITY = pfrac
    if arm == 'leak_ftle':
        s28m.LEAK_FTLE_AHEAD = k   # leak_k now drives the FTLE horizon
    row = s28_run_single((arm, seed))
    row['leak_frac'] = float(frac)
    row['leak_k'] = int(k)
    row['leak_frac_plasticity'] = float(pfrac)
    return row


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S34 leak sensitivity "
          f"(quick={quick}, sequential={sequential})")

    # numba warmup via s28's own warmup block
    tau_w = s28m.gen_tau_vec(16, s28m.CV_TAU, s28m.tau0, seed=0)
    x0_w = s28m.preprogram_vec(s28m.ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = s28m.build_csr(16)
    s28m.run_trajectory_nb(x0_w, tau_w, dt_w, s28m.PW, ip_w, idx_w, wt_w,
                           s28m.KAPPA_NOMINAL, s28m.ALPHA0, s28m.ALPHA_MIN,
                           s28m.ALPHA_MAX, s28m.gamma,
                           s28m.COUPLING_CONTRAST_SELF, 0)
    s28m.run_pair_ftle_nb(x0_w, tau_w, dt_w, s28m.PW, ip_w, idx_w, wt_w,
                          s28m.KAPPA_NOMINAL, s28m.ALPHA0, s28m.ALPHA_MIN,
                          s28m.ALPHA_MAX, s28m.gamma,
                          s28m.COUPLING_CONTRAST_SELF, 1e-8, 10)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = 2 if quick else N_SEEDS
    all_args = [(a, f, k, p, s) for (a, f, k, p) in ARMS
                for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} ({len(ARMS)} leak configs x{n_seeds} seeds)")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_wrapped(a))
            if (i + 1) % max(1, n_runs // 5) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_wrapped, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 5) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'leak_frac', 'leak_k', 'leak_frac_plasticity',
                  'seed_idx', 'r0_nmse', 'r1_nmse', 'r2_nmse', 'r3_nmse',
                  'r1_kappa', 'r2_kappa', 'r3_kappa', 'r0_mc', 'r1_mc',
                  'r2_mc', 'r3_mc', 'n_edges', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 100)
    print("S34 RESULTS (Paper B follow-up): leak sensitivity scan")
    print("=" * 100)
    print(f"  S28 normal arm reference: r3 MC 6.17, r3 NMSE 0.0642")
    print(f" {'arm':>15} {'frac':>6} {'k':>4} {'pfrac':>6} | "
          f"{'r3_mc':>7} {'dMC':>7} | {'r3_nmse':>8} {'dNMSE':>8}")
    for (a, f, k, p) in ARMS:
        rs = [r for r in results
              if r['arm'] == a and abs(r['leak_frac'] - f) < 1e-9
              and r['leak_k'] == k and abs(r['leak_frac_plasticity'] - p) < 1e-9]
        mc = np.mean([r['r3_mc'] for r in rs])
        nm = np.mean([r['r3_nmse'] for r in rs])
        print(f" {a:>15} {f:>6.2f} {k:>4} {p:>6.2f} | {mc:>7.2f} "
              f"{mc - 6.17:>+7.2f} | {nm:>8.4f} {nm - 0.0642:>+8.4f}")

    params = {
        'experiment': 'leak sensitivity scan on the S28 chain-protocol audit',
        'mechanism': 's28 run_single reused; module-level leak params set '
                     'per task in the worker (chunksize=1)',
        'arms': ARMS,
        's28_normal_reference': {'r3_mc': 6.17, 'r3_nmse': 0.0642},
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
