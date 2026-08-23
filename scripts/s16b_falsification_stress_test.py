#!/usr/bin/env python3
"""
Falsification stress test: MC probe protocol robustness (Paper C S16b).
=============================================================================
Type:           PAPER
Experiment:     Does the S14/S16 falsification - the slow trace does not
                transfer MC robustness to an ESN (esn_dual r3_mc <= esn_fast
                at every tau_m) - survive probe-protocol choices? Tests three
                MC probe variants at tau_m in {500, 2000}, 10 seeds:

  V0 raw-slow    : slow = EMA of raw states; noise added after slow
                   (the S14/S16 reference semantics)
  V1 std-slow    : slow = EMA of per-unit standardized states (mu/sd from
                   the probe's first 30%; the S10 preprocessing semantics)
  V2 noisy-slow  : noise injected into the states BEFORE computing the slow
                   trace (slow carries the disturbance; tests whether the
                   reference protocol's clean-slow construction is the
                   source of the result)

Arms (task loop identical to S16):
  esn_fast : ESN-256-hetero fast states + OnlineRLS (variant V0 only)
  esn_dual : ESN + slow trace (tau_m) + OnlineRLS (variants V0/V1/V2)

Judgment: for each (tau_m, variant), if the paired diff
r3_mc(dual) - r3_mc(fast) is negative in (nearly) all seeds, the
falsification is robust to the probe protocol.

Output files:
  data/s16b_falsification_stress_test_v1.csv
    (columns: tau_m, arm, variant, seed, r0_mc, r3_mc, r3_nmse)
  data/s16b_falsification_stress_test_v1.json

Usage: python s16b_falsification_stress_test.py [--quick] [--sequential]
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
from online_readout import OnlineRLS, memory_capacity_heldout
from streaming_tasks import gen_mackey_glass
from fair_esn_comparison import ESN
from s14_esn_disturbance_chain import (
    N_UNITS, CV_TAU, FEATURE_SCALE, RLS_FORGETTING, RLS_INIT_COV,
    RLS_TRACE_CAP, RLS_REG, ESN_SEED_OFFSET, HOMEO_BLOCK, TAU_DRIFT,
    PRUNE_FRAC, NOISE_SIG, T_TOTAL, DISTURB_TIMES, DISTURB_TYPES,
    esn_process_block, slow_ema_block, prune_esn_weights)

TAU_M_LIST = [500.0, 2000.0]
VARIANTS = ['V0', 'V1', 'V2']
N_SEEDS = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's16b_falsification_stress_test_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's16b_falsification_stress_test_v1.json')

FIELDNAMES = ['tau_m', 'arm', 'variant', 'seed', 'r0_mc', 'r3_mc', 'r3_nmse']


def esn_mc_probe_v(esn, W_res, lr, omlr, use_slow, noise_std, seed,
                   tau_m, variant):
    """MC heldout probe with protocol variant (see module docstring)."""
    rng = np.random.RandomState(seed)
    dt_probe = rng.uniform(2e-6, 20e-6, 3000)
    u_probe = (dt_probe - dt_probe.min()) / max(
        dt_probe.max() - dt_probe.min(), 1e-12)
    states = esn_process_block(
        np.zeros(esn.n_reservoir), u_probe[:, None],
        esn.W_in, W_res, esn.bias, lr, omlr)
    if not use_slow:
        obs = states
        if noise_std > 0:
            obs = obs + rng.normal(0, noise_std * obs.std(axis=0), obs.shape)
        return memory_capacity_heldout(obs[500:], dt_probe[500:])

    if variant == 'V1':
        n30 = max(1, int(0.3 * states.shape[0]))
        mu = states[:n30].mean(axis=0)
        sd = states[:n30].std(axis=0)
        sd[sd < 1e-9] = 1.0
        src = (states - mu) / sd
    else:
        src = states
    if variant == 'V2' and noise_std > 0:
        # State-level noise: the fast channel is corrupted, and the slow
        # trace is computed FROM the noisy states (EMA denoises it).
        noisy = states + rng.normal(0, noise_std * states.std(axis=0),
                                    states.shape)
        slow, _ = slow_ema_block(states[0].copy(), noisy, tau_m)
        obs = np.hstack([noisy, slow])
        return memory_capacity_heldout(obs[500:], dt_probe[500:])
    slow, _ = slow_ema_block(states[0].copy(), src, tau_m)
    obs = np.hstack([states, slow])
    if noise_std > 0:
        obs = obs + rng.normal(0, noise_std * obs.std(axis=0), obs.shape)
    return memory_capacity_heldout(obs[500:], dt_probe[500:])


def run_task_loop(arm, seed_idx, tau_m):
    """The S16 task loop; returns (rls-free) r3 NMSE and the disturbed
    ESN physics state for the probes."""
    use_slow = (arm == 'esn_dual')
    n_feat = 2 * N_UNITS + 1 if use_slow else N_UNITS + 1

    dt_seq, target_seq = gen_mackey_glass(seed=seed_idx, n_points=T_TOTAL)
    T = min(dt_seq.shape[0], T_TOTAL)
    dt_seq = dt_seq[:T]
    target = target_seq[:T].astype(np.float64)
    u_norm = (dt_seq - dt_seq.min()) / max(dt_seq.max() - dt_seq.min(), 1e-12)
    u_seq = u_norm[:, None]

    esn = ESN(n_input=1, n_reservoir=N_UNITS, spectral_radius=0.9,
              input_scaling=0.5, leaking_rate=0.2, hetero_lr=True,
              cv_lr=CV_TAU, seed=seed_idx + ESN_SEED_OFFSET)

    rls = OnlineRLS(n_feat, 1, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                    reg=RLS_REG)
    preds = np.empty(T)
    rng_noise = np.random.RandomState(seed_idx * 131 + 3)

    lr_cur = esn.lr.copy()
    omlr_cur = esn.one_minus_lr.copy()
    W_res_cur = esn.W_res.copy()
    noise_std = 0.0
    disturb_idx = 0

    r_cur = np.zeros(N_UNITS)
    slow_prev = None

    for blk_start in range(0, T, HOMEO_BLOCK):
        blk_end = min(blk_start + HOMEO_BLOCK, T)
        if disturb_idx < len(DISTURB_TIMES) and blk_start >= DISTURB_TIMES[disturb_idx]:
            d = DISTURB_TYPES[disturb_idx]
            if d == 'tau_drift':
                lr_cur = esn.lr / TAU_DRIFT
                omlr_cur = 1.0 - lr_cur
            elif d == 'edge_prune':
                W_res_cur = prune_esn_weights(W_res_cur)
            elif d == 'noise':
                noise_std = NOISE_SIG
            disturb_idx += 1

        st_b = esn_process_block(r_cur, u_seq[blk_start:blk_end],
                                 esn.W_in, W_res_cur, esn.bias,
                                 lr_cur, omlr_cur)
        obs = st_b
        if use_slow:
            if slow_prev is None:
                slow_prev = st_b[0].copy()
            slow_b, slow_prev = slow_ema_block(slow_prev, st_b, tau_m)
            obs = np.hstack([st_b, slow_b])
        if noise_std > 0:
            obs = obs + rng_noise.normal(
                0, noise_std * obs.std(axis=0), obs.shape)
        F = np.hstack([obs, np.ones((obs.shape[0], 1))])
        for j in range(obs.shape[0]):
            preds[blk_start + j] = rls.predict(F[j])[0]
            rls.update(F[j], target[blk_start + j])
        r_cur = st_b[-1].copy()

    var_full = float(target[DISTURB_TIMES[-1]:].var())
    lo, hi = DISTURB_TIMES[-1], T
    mid = (lo + hi) // 2
    if mid + 1000 < hi and mid - 1000 > lo:
        seg_p = preds[mid - 1000:mid + 1000]
        seg_t = target[mid - 1000:mid + 1000]
        v = float(seg_t.var())
        r3_nmse = float(np.mean((seg_p - seg_t) ** 2)) / max(v, 1e-12)
    else:
        r3_nmse = float('nan')

    return esn, lr_cur, omlr_cur, W_res_cur, noise_std, r3_nmse


def run_single(args):
    """(arm, seed_idx, tau_m) -> list of row dicts (one per variant)."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed_idx, tau_m = args
    t0 = time.time()
    use_slow = (arm == 'esn_dual')

    esn, lr_cur, omlr_cur, W_res_cur, noise_std, r3_nmse = run_task_loop(
        arm, seed_idx, tau_m)

    rows = []
    variants = VARIANTS if use_slow else ['V0']
    for v in variants:
        r0 = esn_mc_probe_v(esn, esn.W_res, esn.lr, esn.one_minus_lr,
                            use_slow, 0.0, seed_idx * 999, tau_m, v)
        r3 = esn_mc_probe_v(esn, W_res_cur, lr_cur, omlr_cur, use_slow,
                            noise_std, seed_idx * 999 + 30, tau_m, v)
        rows.append({'tau_m': float(tau_m), 'arm': arm, 'variant': v,
                     'seed': seed_idx, 'r0_mc': r0, 'r3_mc': r3,
                     'r3_nmse': r3_nmse,
                     'runtime_s': time.time() - t0})
    return rows


def aggregate(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['arm'], r['tau_m'], r['variant']), []).append(r)
    agg = []
    for (arm, tm, v), rs in sorted(groups.items()):
        entry = {'arm': arm, 'tau_m': tm, 'variant': v, 'n_runs': len(rs)}
        for m in ['r0_mc', 'r3_mc', 'r3_nmse']:
            vals = np.array([r[m] for r in rs], dtype=float)
            entry[m + '_mean'] = float(np.nanmean(vals))
            entry[m + '_std'] = float(np.nanstd(vals))
        agg.append(entry)
    return agg


def print_table(agg, rows):
    print("\n" + "=" * 96)
    print("S16b RESULTS: falsification stress test (10 seeds)")
    print("=" * 96)
    fast_r3 = {int(r['seed']): r['r3_mc'] for r in rows
               if r['arm'] == 'esn_fast'}
    seeds_present = sorted(fast_r3)
    print(f" {'tau_m':>6} {'variant':>7} | {'r0_mc':>6} {'r3_mc':>6} | "
          f"{'paired diff':>11} {'n_pos':>5} {'verdict':>10}")
    for a in sorted(agg, key=lambda x: (x['arm'], x['tau_m'], x['variant'])):
        if a['arm'] == 'esn_fast':
            print(f" {a['tau_m']:>6.0f} {a['variant']:>7} | "
                  f"{a['r0_mc_mean']:>6.2f} {a['r3_mc_mean']:>6.2f} | "
                  f"{'-':>11} {'-':>5} {'baseline':>10}")
            continue
        d = np.array([next(r['r3_mc'] for r in rows
                           if r['arm'] == 'esn_dual' and r['seed'] == s
                           and r['tau_m'] == a['tau_m'] and r['variant'] == a['variant'])
                      - fast_r3[s] for s in seeds_present])
        n_pos = int(np.sum(d > 0))
        verdict = 'robust' if n_pos <= 1 else 'VIOLATION'
        print(f" {a['tau_m']:>6.0f} {a['variant']:>7} | "
              f"{a['r0_mc_mean']:>6.2f} {a['r3_mc_mean']:>6.2f} | "
              f"{d.mean():>11.4f} {n_pos:>5d} {verdict:>10}")


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S16b falsification stress "
          f"test (quick={quick}, sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    all_args = ([('esn_fast', s, 500.0) for s in range(n_seeds)]
                + [('esn_dual', s, tm)
                   for tm in TAU_M_LIST for s in range(n_seeds)])
    n_runs = len(all_args)
    print(f"total task loops: {n_runs} (fast x{n_seeds}, "
          f"dual {len(TAU_M_LIST)} tau_m x{n_seeds})")

    rows = []
    if sequential:
        for i, a in enumerate(all_args):
            rows.extend(run_single(a))
            if (i + 1) % max(1, n_runs // 5) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                rows.extend(res)
                done += 1
                if done % max(1, n_runs // 5) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    agg = aggregate(rows)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'tau_m_list': TAU_M_LIST,
        'variants': VARIANTS,
        'variants_def': {
            'V0': 'slow = EMA(raw states); noise after slow (S14/S16 ref)',
            'V1': 'slow = EMA(standardized states, mu/sd from probe first 30%)',
            'V2': 'noise injected into states BEFORE slow computation'},
        'esn': 'hetero-lr (cv=0.20), spectral_radius=0.9, input_scaling=0.5, '
               'base leaking_rate=0.2', 'esn_seed_offset': ESN_SEED_OFFSET,
        'rls_forgetting': RLS_FORGETTING,
        'protocol': 'inherited from S16 (s16_tau_m_pressure_test.py)',
        'disturb_times': DISTURB_TIMES, 'disturb_types': DISTURB_TYPES,
        'judgment': 'paired r3_mc(dual)-r3_mc(fast); <=1 positive seed per '
                    '(tau_m, variant) -> falsification robust to probe protocol',
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg, rows)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
