#!/usr/bin/env python3
"""
ESN-with-metadata under the sequential disturbance chain (Paper C S14).
=============================================================================
Type:           PAPER
Experiment:     Paper C decisive transfer experiment: does the M3 slow-trace
                metadata provide sequential disturbance robustness on a
                generic ESN, as the homeostat does on the Si3N4 substrate?
                Mirrors the S11 disturbance-chain protocol with
                substrate-agnostic disturbance definitions.

Disturbance chain (cumulative), substrate-agnostic per-arm definitions:
  Round 0 [0, 7k)     : nominal
  Round 1 [7k, 14k)   : timescale drift  (substrate: tau *= 1.5;
                        ESN: leaking rate /= 1.5, state relaxation slowed
                        by the same factor)
  Round 2 [14k, 21k)  : structure prune  (40% of connections removed:
                        substrate: undirected random_graph edges;
                        ESN: reservoir weight entries)
  Round 3 [21k, 28k)  : readout noise (sigma=0.1 on readout features,
                        identical for all arms)

Arms (each 10 seeds, Mackey-Glass online prediction, S11 protocol):
  esn_fast  : ESN-256-hetero fast states + OnlineRLS (no metadata)
  esn_dual  : ESN-256-hetero + M3 slow trace (tau_slow=500) + OnlineRLS
  redem_reg : Si3N4 substrate + RLS + M5 homeostat (the S11 regulated arm,
              re-run via s11_disturbance_chain.run_single for exact
              reproducibility of the +32% anchor)

Metrics per round (same as S11): NMSE on the online task, kappa settled
(NaN for ESN arms), MC_heldout probe at the settled physics. For esn_dual
the MC probe uses [fast, slow] features, testing the Proposition-1 kernel
extension; for the other arms it uses fast features only.

Output files:
  data/s14_esn_disturbance_chain_v1.csv
  data/s14_esn_disturbance_chain_v1.json

Usage: python s14_esn_disturbance_chain.py [--quick] [--sequential]
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
    run_trajectory_nb, run_pair_ftle_nb)
from online_readout import OnlineRLS, memory_capacity_heldout
from streaming_tasks import gen_mackey_glass
from fair_esn_comparison import ESN
from s11_disturbance_chain import (
    run_single as s11_run_single,
    prune_edges as s11_prune_edges,
    KAPPA_NOMINAL, KAPPA_MIN, KAPPA_MAX, LAMBDA_TARGET, ETA_LAMBDA,
    FTLE_EVERY, FTLE_WINDOW, HOMEO_BLOCK, TAU_DRIFT, PRUNE_FRAC,
    NOISE_SIG, ROUND_LEN, T_TOTAL, DISTURB_TIMES, DISTURB_TYPES)

# ========================== Fixed parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0

RLS_FORGETTING = 0.999
RLS_INIT_COV = 1.0
RLS_TRACE_CAP = 1e8
RLS_REG = 1e-4

TAU_SLOW = 500.0
ESN_SEED_OFFSET = 999

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's14_esn_disturbance_chain_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's14_esn_disturbance_chain_v1.json')

ARMS = ['esn_fast', 'esn_dual', 'redem_reg']


def esn_process_block(r0, u_block, W_in, W_res, bias, lr, omlr):
    """Run one ESN block. Python loop (inherited from fair_esn_comparison,
    Type ML); small per-block cost, kept identical to the baseline ESN."""
    T = u_block.shape[0]
    r = r0.copy()
    states = np.empty((T, lr.shape[0]), dtype=np.float64)
    for t in range(T):
        r = omlr * r + lr * np.tanh(W_in @ u_block[t] + W_res @ r + bias)
        states[t] = r
    return states


def slow_ema_block(slow_prev, fast_block, tau_slow):
    """Blockwise slow EMA, same recurrence as integrated_benchmark.slow_ema."""
    lam = 1.0 / tau_slow
    out = np.empty_like(fast_block)
    m = slow_prev.copy()
    for t in range(fast_block.shape[0]):
        m = (1.0 - lam) * m + lam * fast_block[t]
        out[t] = m
    return out, m


def prune_esn_weights(W, seed=TOPO_SEED + 1, frac=PRUNE_FRAC):
    """Zero PRUNE_FRAC of reservoir weight entries (structure disturbance)."""
    rng = np.random.RandomState(seed)
    mask = rng.rand(*W.shape) > frac
    return W * mask


def esn_mc_probe(esn, W_res, lr, omlr, use_slow, noise_std, seed):
    """Jaeger MC heldout probe at the current ESN physics. Washout = 500,
    dt uniform over [2,20]us, features [fast] or [fast, slow]."""
    rng = np.random.RandomState(seed)
    dt_probe = rng.uniform(2e-6, 20e-6, 3000)
    u_probe = (dt_probe - dt_probe.min()) / max(
        dt_probe.max() - dt_probe.min(), 1e-12)
    states = esn_process_block(
        np.zeros(esn.n_reservoir), u_probe[:, None],
        esn.W_in, W_res, esn.bias, lr, omlr)
    obs = states
    if use_slow:
        slow, _ = slow_ema_block(states[0].copy(), states, TAU_SLOW)
        obs = np.hstack([states, slow])
    if noise_std > 0:
        obs = obs + rng.normal(0, noise_std * obs.std(axis=0), obs.shape)
    return memory_capacity_heldout(obs[500:], dt_probe[500:])


def run_single(args):
    """(arm, seed_idx) -> dict with per-round metrics."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed_idx = args
    t0 = time.time()

    if arm == 'redem_reg':
        # Exact reuse of the S11 regulated arm (10 seeds) -> byte-identical
        # anchor (r3_mc ~ 8.47, +32% over the S11 fixed arm).
        res = s11_run_single(('regulated', seed_idx))
        res['arm'] = 'redem_reg'
        return res

    # ---------------- ESN arms ----------------
    dt_seq, target_seq = gen_mackey_glass(seed=seed_idx, n_points=T_TOTAL)
    T = min(dt_seq.shape[0], T_TOTAL)
    dt_seq = dt_seq[:T]
    target = target_seq[:T].astype(np.float64)
    u_norm = (dt_seq - dt_seq.min()) / max(dt_seq.max() - dt_seq.min(), 1e-12)
    u_seq = u_norm[:, None]

    esn = ESN(n_input=1, n_reservoir=N_UNITS, spectral_radius=0.9,
              input_scaling=0.5, leaking_rate=0.2, hetero_lr=True,
              cv_lr=CV_TAU, seed=seed_idx + ESN_SEED_OFFSET)
    use_slow = (arm == 'esn_dual')
    n_feat = 2 * N_UNITS + 1 if use_slow else N_UNITS + 1

    rls = OnlineRLS(n_feat, 1, forgetting=RLS_FORGETTING,
                    init_cov=RLS_INIT_COV, trace_cap=RLS_TRACE_CAP,
                    reg=RLS_REG)
    preds = np.empty(T)
    kappa_hist = np.empty(T)
    kappa_hist[:] = np.nan

    rng_noise = np.random.RandomState(seed_idx * 131 + 3)

    # Cumulative disturbance state (ESN view)
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
            slow_b, slow_prev = slow_ema_block(slow_prev, st_b, TAU_SLOW)
            obs = np.hstack([st_b, slow_b])
        if noise_std > 0:
            obs = obs + rng_noise.normal(
                0, noise_std * obs.std(axis=0), obs.shape)
        F = np.hstack([obs, np.ones((obs.shape[0], 1))])
        for j in range(obs.shape[0]):
            preds[blk_start + j] = rls.predict(F[j])[0]
            rls.update(F[j], target[blk_start + j])
        r_cur = st_b[-1].copy()

    # Per-round metrics
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
        results[f'r{r}_kappa'] = float('nan')  # ESN has no kappa

    # MC probes: nominal (r0) at seed_idx*999, rounds 1-3 at seed_idx*999+r*10
    rng_n = np.random.RandomState(seed_idx * 999)
    results['r0_mc'] = esn_mc_probe(esn, esn.W_res, esn.lr,
                                    esn.one_minus_lr, use_slow, 0.0,
                                    seed_idx * 999)
    for r in range(1, 4):
        results[f'r{r}_mc'] = esn_mc_probe(
            esn, W_res_cur, lr_cur, omlr_cur, use_slow, noise_std,
            seed_idx * 999 + r * 10)

    results['kappa_drift_r2_r1'] = float('nan')
    results['kappa_drift_r3_r1'] = float('nan')
    return results


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(r['arm'], []).append(r)
    agg = []
    for arm, rs in sorted(groups.items()):
        entry = {'arm': arm, 'n_runs': len(rs)}
        for r_idx in range(4):
            for metric in ['nmse', 'kappa', 'mc']:
                key = f'r{r_idx}_{metric}'
                v = np.array([r[key] for r in rs], dtype=float)
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    entry[key + '_mean'] = float(np.mean(v))
                    entry[key + '_std'] = float(np.std(v))
                else:
                    entry[key + '_mean'] = float('nan')
                    entry[key + '_std'] = float('nan')
        for key in ['kappa_drift_r2_r1', 'kappa_drift_r3_r1']:
            v = np.array([r[key] for r in rs], dtype=float)
            v = v[~np.isnan(v)]
            entry[key + '_mean'] = float(np.mean(v)) if len(v) > 0 else float('nan')
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 110)
    print("S14 RESULTS (mean over seeds): ESN+metadata disturbance chain")
    print("=" * 110)
    print(f" {'arm':>10} | {'r0_nmse':>8} {'r1_nmse':>8} {'r2_nmse':>8} {'r3_nmse':>8} | "
          f"{'r0_mc':>6} {'r1_mc':>6} {'r2_mc':>6} {'r3_mc':>6} | "
          f"{'r3_vs_r1':>8}")
    for a in agg:
        r1 = a.get('r1_mc_mean', float('nan'))
        r3 = a.get('r3_mc_mean', float('nan'))
        rel = (r3 - r1) / r1 * 100.0 if r1 == r1 and r1 != 0 else float('nan')
        print(f" {a['arm']:>10} | "
              f"{a.get('r0_nmse_mean', float('nan')):>8.4f} "
              f"{a.get('r1_nmse_mean', float('nan')):>8.4f} "
              f"{a.get('r2_nmse_mean', float('nan')):>8.4f} "
              f"{a.get('r3_nmse_mean', float('nan')):>8.4f} | "
              f"{a.get('r0_mc_mean', float('nan')):>6.2f} "
              f"{a.get('r1_mc_mean', float('nan')):>6.2f} "
              f"{a.get('r2_mc_mean', float('nan')):>6.2f} "
              f"{a.get('r3_mc_mean', float('nan')):>6.2f} | "
              f"{rel:>8.1f}%")


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S14 ESN disturbance chain "
          f"(quick={quick}, sequential={sequential})")

    # numba warmup (small substrate run incl. FTLE for the redem_reg arm)
    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16,
                                           seed=TOPO_SEED, avg_degree=4)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_NOMINAL,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    run_pair_ftle_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_NOMINAL,
                     ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                     COUPLING_CONTRAST_SELF, 1e-8, 10)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = 2 if quick else N_SEEDS
    all_args = [(arm, s) for arm in ARMS for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (arms={len(ARMS)}, seeds={n_seeds})")

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
    fieldnames = ['arm', 'seed_idx',
                  'r0_nmse', 'r1_nmse', 'r2_nmse', 'r3_nmse',
                  'r0_kappa', 'r1_kappa', 'r2_kappa', 'r3_kappa',
                  'r0_mc', 'r1_mc', 'r2_mc', 'r3_mc',
                  'kappa_drift_r2_r1', 'kappa_drift_r3_r1', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'tau_slow': TAU_SLOW,
        'esn': 'hetero-lr (cv=0.20), spectral_radius=0.9, input_scaling=0.5, '
               'base leaking_rate=0.2', 'esn_seed_offset': ESN_SEED_OFFSET,
        'rls_forgetting': RLS_FORGETTING,
        'kappa_nominal': KAPPA_NOMINAL, 'kappa_range': [KAPPA_MIN, KAPPA_MAX],
        'lambda_target': LAMBDA_TARGET, 'eta_lambda': ETA_LAMBDA,
        'ftle_every': FTLE_EVERY, 'ftle_window': FTLE_WINDOW,
        'homeo_block': HOMEO_BLOCK, 'tau_drift': TAU_DRIFT,
        'prune_frac': PRUNE_FRAC, 'noise_sig': NOISE_SIG,
        'round_len': ROUND_LEN, 't_total': T_TOTAL,
        'disturb_times': DISTURB_TIMES, 'disturb_types': DISTURB_TYPES,
        'disturb_defs': {
            'timescale_drift': 'substrate tau*=1.5; ESN leaking_rate/=1.5',
            'structure_prune': 'substrate 40% edges; ESN 40% weights',
            'readout_noise': 'sigma=0.1 on readout features (all arms)'},
        'mc_probe_features': {'esn_fast': 'fast',
                              'esn_dual': 'fast+slow',
                              'redem_reg': 'fast (S11 protocol)'},
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
