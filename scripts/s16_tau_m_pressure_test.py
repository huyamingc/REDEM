#!/usr/bin/env python3
"""
tau_m pressure test for the metadata equalizer (Paper C S16).
=============================================================================
Type:           PAPER
Experiment:     Stress-test of the S14 falsification: does the slow-trace
                metadata's failure to transfer MC robustness to an ESN hold
                across metadata timescales? Sweeps tau_m in
                {200, 500, 1000, 2000} under the S14 disturbance chain,
                10 seeds.

Arms:
  esn_fast : ESN-256-hetero fast states + OnlineRLS (no metadata; tau_m=0
             sentinel, the tau_m-independent baseline, identical to the
             S14 esn_fast arm - reproducibility check)
  esn_dual : ESN + slow trace with tau_m in TAU_M_LIST + OnlineRLS

Protocol: identical to S14 (Mackey-Glass online task; cumulative
substrate-agnostic disturbances at t=7k/14k/21k: timescale drift, structure
prune, readout noise sigma=0.1; per-round Jaeger MC heldout probe with
features [fast] or [fast, slow]).

Judgment rule (Paper C decision gate):
  - all tau_m: r3_mc(dual) <= r3_mc(fast)  -> strong claim: metadata is
    non-transferable for MC robustness (adopt as is).
  - any tau_m: r3_mc(dual) > r3_mc(fast)   -> weaken claim to
    "non-transferable at typical timescales (200-1000)" and analyze the
    long-window coupling effect.

Output files:
  data/s16_tau_m_pressure_test_v1.csv    (columns: tau_m, arm, seed,
   r0_mc, r1_mc, r2_mc, r3_mc, r3_nmse)
  data/s16_tau_m_pressure_test_v1.json

Usage: python s16_tau_m_pressure_test.py [--quick] [--sequential]
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

TAU_M_LIST = [200.0, 500.0, 1000.0, 2000.0]
N_SEEDS = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's16_tau_m_pressure_test_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's16_tau_m_pressure_test_v1.json')

FIELDNAMES = ['tau_m', 'arm', 'seed',
              'r0_mc', 'r1_mc', 'r2_mc', 'r3_mc', 'r3_nmse']


def esn_mc_probe_tm(esn, W_res, lr, omlr, use_slow, noise_std, seed, tau_m):
    """Jaeger MC heldout probe at the current ESN physics (tau_m-aware
    copy of the S14 probe). Washout = 500, dt uniform [2,20]us,
    features [fast] or [fast, slow]."""
    rng = np.random.RandomState(seed)
    dt_probe = rng.uniform(2e-6, 20e-6, 3000)
    u_probe = (dt_probe - dt_probe.min()) / max(
        dt_probe.max() - dt_probe.min(), 1e-12)
    states = esn_process_block(
        np.zeros(esn.n_reservoir), u_probe[:, None],
        esn.W_in, W_res, esn.bias, lr, omlr)
    obs = states
    if use_slow:
        slow, _ = slow_ema_block(states[0].copy(), states, tau_m)
        obs = np.hstack([states, slow])
    if noise_std > 0:
        obs = obs + rng.normal(0, noise_std * obs.std(axis=0), obs.shape)
    return memory_capacity_heldout(obs[500:], dt_probe[500:])


def run_single(args):
    """(arm, seed_idx, tau_m) -> dict with per-round MC and r3 NMSE."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed_idx, tau_m = args
    t0 = time.time()

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

    # Per-round NMSE (same windows as S11/S14)
    var_full = float(target[DISTURB_TIMES[-1]:].var())

    def nmse(seg_p, seg_t):
        v = float(seg_t.var()) if len(seg_t) > 1 else var_full
        return float(np.mean((seg_p - seg_t) ** 2)) / max(v, 1e-12)

    round_boundaries = [0] + DISTURB_TIMES + [T]
    nmse_by_round = {}
    for r in range(4):
        lo = round_boundaries[r]
        hi = round_boundaries[r + 1]
        mid = (lo + hi) // 2
        if mid + 1000 < hi and mid - 1000 > lo:
            nmse_by_round[r] = nmse(preds[mid - 1000:mid + 1000],
                                    target[mid - 1000:mid + 1000])
        else:
            nmse_by_round[r] = float('nan')

    # MC probes: nominal (r0) at seed_idx*999, rounds 1-3 at seed_idx*999+r*10
    results = {'tau_m': float(tau_m), 'arm': arm, 'seed': seed_idx,
               'r0_mc': esn_mc_probe_tm(esn, esn.W_res, esn.lr,
                                        esn.one_minus_lr, use_slow, 0.0,
                                        seed_idx * 999, tau_m),
               'r3_nmse': nmse_by_round[3]}
    for r in range(1, 4):
        results[f'r{r}_mc'] = esn_mc_probe_tm(
            esn, W_res_cur, lr_cur, omlr_cur, use_slow, noise_std,
            seed_idx * 999 + r * 10, tau_m)
    results['runtime_s'] = time.time() - t0
    return results


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['arm'], r['tau_m']), []).append(r)
    agg = []
    for (arm, tm), rs in sorted(groups.items()):
        entry = {'arm': arm, 'tau_m': tm, 'n_runs': len(rs)}
        for m in ['r0_mc', 'r1_mc', 'r2_mc', 'r3_mc', 'r3_nmse']:
            v = np.array([r[m] for r in rs], dtype=float)
            entry[m + '_mean'] = float(np.nanmean(v))
            entry[m + '_std'] = float(np.nanstd(v))
        agg.append(entry)
    return agg


def print_table(agg, fast_rows, results):
    print("\n" + "=" * 100)
    print("S16 RESULTS (mean over seeds): tau_m pressure test")
    print("=" * 100)
    print(f" {'tau_m':>6} {'arm':>9} | {'r0_mc':>6} {'r1_mc':>6} "
          f"{'r2_mc':>6} {'r3_mc':>6} {'r3_nmse':>8} | {'d_r3(fast)':>10}")
    fast_mean = {f['tau_m']: f for f in fast_rows} if fast_rows else {}
    fast_r3 = np.array([r['r3_mc'] for r in results if r['arm'] == 'esn_fast'])
    fast_r3_mean = float(np.nanmean(fast_r3)) if fast_r3.size else float('nan')
    for a in sorted(agg, key=lambda x: (x['arm'], x['tau_m'])):
        d = ''
        if a['arm'] == 'esn_dual':
            fm = fast_r3_mean
            d = f"{a['r3_mc_mean'] - fm:>10.3f}"
        print(f" {a['tau_m']:>6.0f} {a['arm']:>9} | "
              f"{a['r0_mc_mean']:>6.2f} {a['r1_mc_mean']:>6.2f} "
              f"{a['r2_mc_mean']:>6.2f} {a['r3_mc_mean']:>6.2f} "
              f"{a['r3_nmse_mean']:>8.4f} | {d}")

    # Judgment rule
    duals = [a for a in agg if a['arm'] == 'esn_dual']
    viol = [a for a in duals if a['r3_mc_mean'] > fast_r3_mean]
    print("\nJudgment rule (r3_mc dual vs fast):")
    if not viol:
        print("  STRONG CLAIM holds: r3_mc(dual) <= r3_mc(fast) at ALL tau_m "
              "-> metadata is non-transferable for MC robustness.")
    else:
        for a in viol:
            print(f"  VIOLATION at tau_m={a['tau_m']:.0f}: r3_mc(dual) "
                  f"{a['r3_mc_mean']:.3f} > fast {fast_r3_mean:.3f}")
        print("  WEAKEN claim to 'non-transferable at typical timescales "
              "(200-1000)'; analyze the long-window coupling effect.")


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S16 tau_m pressure test "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    all_args = ([('esn_fast', s, 0.0) for s in range(n_seeds)]
                + [('esn_dual', s, tm)
                   for tm in TAU_M_LIST for s in range(n_seeds)])
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (fast baseline x{n_seeds}, "
          f"dual {len(TAU_M_LIST)} tau_m x{n_seeds})")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_single(a))
            if (i + 1) % max(1, n_runs // 5) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 5) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    fast_rows = [a for a in agg if a['arm'] == 'esn_fast']
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'tau_m_list': TAU_M_LIST,
        'tau_m_sentinel_fast': 0.0,
        'esn': 'hetero-lr (cv=0.20), spectral_radius=0.9, input_scaling=0.5, '
               'base leaking_rate=0.2', 'esn_seed_offset': ESN_SEED_OFFSET,
        'rls_forgetting': RLS_FORGETTING,
        'protocol': 'inherited from S14 (s14_esn_disturbance_chain.py)',
        'disturb_times': DISTURB_TIMES, 'disturb_types': DISTURB_TYPES,
        'disturb_defs': {
            'timescale_drift': 'substrate tau*=1.5; ESN leaking_rate/=1.5',
            'structure_prune': 'substrate 40% edges; ESN 40% weights',
            'readout_noise': 'sigma=0.1 on readout features (all arms)'},
        'judgment_rule': 'all tau_m: r3_mc(dual)<=r3_mc(fast) -> strong '
                         'claim; any violation -> weaken to typical '
                         'timescales (200-1000)',
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg, fast_rows, results)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
