#!/usr/bin/env python3
"""
Substrate recurrence characterization: kappa-MC-lambda phase diagram (S1).
=============================================================================
Type:           PAPER
Paper Section:  New-algorithm project Step S1
Experiment:     Phase diagram of the recurrent relaxation substrate:
                finite-time Lyapunov exponent (FTLE), memory capacity
                (MC, Jaeger definition), and input separation vs coupling
                kappa, across topologies and coupling families.

Protocol (fixed seeds, fully reproducible):
  * Drive: i.i.d. uniform pulse intervals dt ~ U[2us, 20us] (fast-drive
    regime; memory window N_eff = tau0 / mean(dt) ~ 17 pulses, consistent
    with the paper's operating point).
  * Per run, three trajectories sharing x0 and tau:
      (a) main : drive stream A          -> states for MC regression
      (b) alt  : drive stream B          -> input separation metric
      (c) twin : drive stream A with perturbed initial state (Benettin FTLE)
  * MC(k): squared Pearson correlation of ridge-decoded interval dt_{t-k}
    from current-ratio observables i_t = exp(gamma * x_t); k = 0..K_MAX.
    MC_total = sum over k = 1..K_MAX.
  * 10 independent seeds per configuration; tau draws are paired across
    kappa within a topology (same seed_idx -> same tau), so kappa acts as
    the dose and the substrate draw is the paired block.
  * The random-graph structure is FIXED across the whole sweep (TOPO_SEED),
    so topology is the treatment and kappa the dose.

Sweep grid (v2, refined after v1 bracketing):
  parallel                     : kappa = 0                  (baseline)
  ring_bidir / lateral_ring /
  random_graph   (mode 1)      : kappa in KAPPA_CONTRAST_MODE1 (bisects the
                                 v1 10->30 chaos bracket: 15/20/25)
  ring_unidir (mode 2)         : kappa in KAPPA_RING_UNIDIR (bisects v1's
                                 near-edge window around kappa ~ 1-3)
  hub_star (mode 2)            : kappa in KAPPA_HUB
  add_ring_bidir / add_random_graph (mode 3): kappa in KAPPA_ADDITIVE
                                 (bisects the saturation cliff at ~0.1-0.2)

MC protocol: ridge fit on the first 70% of collected rows, a k_max-row
leakage buffer, held-out corr^2 on the last 30% (mc_total_test). The
train-segment values follow the classic Jaeger definition.

Output files:
  data/substrate_phase_diagram_v2.csv   (one row per run)
  data/substrate_phase_diagram_v2.json  (all params + per-config aggregates)

Usage:
  python substrate_recurrence_characterization.py [--selftest] [--quick]
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
    COUPLING_NONE, COUPLING_CONTRAST_SELF, COUPLING_CONTRAST_NBR, COUPLING_ADDITIVE,
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    run_trajectory_nb, run_pair_ftle_nb, self_test as substrate_self_test)

# ========================== Fixed experiment parameters ==========================
N_UNITS = 256
CV_TAU = 0.20
T_TOTAL = 1200        # pulses per trajectory
N_WASHOUT = 200       # washout pulses discarded before collection
K_MAX = 50            # MC lags 1..K_MAX (k=0 reported separately)
N_SEEDS = 10
EPS_BENETTIN = 1e-8
RENORM_EVERY = 10
RIDGE_LAMBDA = 1.0
DT_LO, DT_HI = 2e-6, 20e-6
TOPO_SEED = 777       # fixed random-graph structure across the whole sweep
LATERAL_RADIUS = 4
AVG_DEGREE = 8

KAPPA_CONTRAST_MODE1 = [0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 100.0]
KAPPA_RING_UNIDIR = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0]
KAPPA_HUB = [0.3, 0.5, 1.0, 3.0]
KAPPA_ADDITIVE = [0.01, 0.03, 0.06, 0.08, 0.1, 0.15, 0.2]

# (config_name, csr_topology_name, coupling_mode, kappa_list)
CONFIGS = [("parallel", "parallel", COUPLING_NONE, [0.0])]
for _topo in ["ring_bidir", "lateral_ring", "random_graph"]:
    CONFIGS.append((_topo, _topo, COUPLING_CONTRAST_SELF, KAPPA_CONTRAST_MODE1))
CONFIGS.append(("ring_unidir", "ring_unidir", COUPLING_CONTRAST_NBR, KAPPA_RING_UNIDIR))
CONFIGS.append(("hub_star", "hub_star", COUPLING_CONTRAST_NBR, KAPPA_HUB))
for _topo in ["ring_bidir", "random_graph"]:
    CONFIGS.append((f"add_{_topo}", _topo, COUPLING_ADDITIVE, KAPPA_ADDITIVE))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 'substrate_phase_diagram_v2.csv')
JSON_PATH = os.path.join(DATA_DIR, 'substrate_phase_diagram_v2.json')


# ========================== Memory capacity (linear readout) ==========================

def memory_capacity(obs, dt_seq, n_washout, k_max, ridge_lambda, test_frac=0.3):
    """Jaeger memory capacity via multi-RHS ridge, with held-out validation.

    obs: (S, N) collected states (row t = state after pulse n_washout + t).
    The interval applied at that step is dt_seq[n_washout + t], so the lag-k
    target for row t is dt_seq[n_washout + t - k] (defined for t >= k).

    The collected rows are split chronologically: first (1-test_frac) for
    fitting, a k_max-row buffer (memory-window leakage guard), the rest for
    held-out evaluation. Returns a dict:
      mc_total_train / mc_total_test : sum of corr^2 over lags 1..k_max
      mc_k0_train / mc_k0_test       : corr^2 at lag 0
      mc_curve_train                 : corr^2 per lag (train segment), k=0..k_max
    """
    S, n = obs.shape
    X = obs[k_max:, :]
    T = X.shape[0]
    n_test = max(50, int(T * test_frac))
    n_train = T - n_test - k_max      # k_max-row leakage buffer between segments
    n_test = T - n_train - k_max
    mu = X[:n_train].mean(axis=0)
    sd = X[:n_train].std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    row_pulse = n_washout + k_max + np.arange(T)   # pulse index of each row
    Y = np.empty((T, k_max + 1))
    for k in range(k_max + 1):
        Y[:, k] = dt_seq[row_pulse - k]
    ymu = Y[:n_train].mean(axis=0)
    Yc = Y - ymu
    Xtr, Xte = Xs[:n_train], Xs[n_train + k_max:]
    Ytr, Yte = Yc[:n_train], Yc[n_train + k_max:]

    A = Xtr.T @ Xtr + ridge_lambda * np.eye(n)
    B = Xtr.T @ Ytr
    W = np.linalg.solve(A, B)
    Ptr = Xtr @ W
    Pte = Xte @ W

    def _corr2(P, Yc_seg):
        pc = np.empty(P.shape[1])
        for k in range(P.shape[1]):
            a = P[:, k] - P[:, k].mean()
            b = Yc_seg[:, k]
            denom = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
            pc[k] = float((a * b).sum()) / denom if denom > 0 else 0.0
        return pc

    pc_tr = _corr2(Ptr, Ytr)
    pc_te = _corr2(Pte, Yte)
    return {
        'mc_total_train': float(pc_tr[1:].sum()),
        'mc_total_test': float(pc_te[1:].sum()),
        'mc_k0_train': float(pc_tr[0]),
        'mc_k0_test': float(pc_te[0]),
        'mc_curve_train': pc_tr,
    }


# ========================== Single MC run (picklable top-level) ==========================

def run_single(args):
    """One Monte Carlo run of a single (config, kappa, seed) cell.

    args = (config_name, topo_name, mode, kappa, seed_idx,
            n_units, t_total, n_washout, k_max)
    Returns a flat metrics dict.
    """
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    (config_name, topo_name, mode, kappa, seed_idx,
     n_units, t_total, n_washout, k_max) = args
    t0 = time.time()

    # paired substrate draw: seed_idx alone determines tau and x0
    tau = gen_tau_vec(n_units, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)

    rng_main = np.random.RandomState(seed_idx * 77 + 13)
    rng_alt = np.random.RandomState(seed_idx * 77 + 13 + 5001)
    dt_main = rng_main.uniform(DT_LO, DT_HI, t_total)
    dt_alt = rng_alt.uniform(DT_LO, DT_HI, t_total)

    indptr, indices, wts = build_topology_csr(
        topo_name, n_units, seed=TOPO_SEED,
        lateral_radius=LATERAL_RADIUS, avg_degree=AVG_DEGREE)

    # (a) main trajectory: states for MC regression
    states, clip_frac, g_abs = run_trajectory_nb(
        x0, tau, dt_main, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, n_washout)

    # (b) alt trajectory: same init, different input -> separation
    states_alt, _, _ = run_trajectory_nb(
        x0, tau, dt_alt, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, n_washout)
    inter_rms = float(np.sqrt(np.mean((states_alt - states) ** 2)))

    # (c) Benettin FTLE on the main input stream
    ftle, _ = run_pair_ftle_nb(
        x0, tau, dt_main, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode,
        EPS_BENETTIN, RENORM_EVERY)

    # MC regression on current-ratio observables (physical readout quantity)
    obs = np.exp(gamma * states)
    mc = memory_capacity(obs, dt_main, n_washout, k_max, RIDGE_LAMBDA)

    return {
        'config_name': config_name,
        'topology': topo_name,
        'coupling_mode': int(mode),
        'kappa': float(kappa),
        'seed_idx': int(seed_idx),
        'n_units': int(n_units),
        'cv_tau': CV_TAU,
        't_total': int(t_total),
        'n_washout': int(n_washout),
        'k_max': int(k_max),
        'ftle_per_pulse': float(ftle),
        'mc_total': mc['mc_total_train'],
        'mc_total_test': mc['mc_total_test'],
        'mc_k0': mc['mc_k0_train'],
        'mc_k0_test': mc['mc_k0_test'],
        'inter_rms': inter_rms,
        'alpha_clip_frac': float(clip_frac),
        'g_abs_mean': float(g_abs),
        'mean_state': float(states.mean()),
        'mc_curve': [float(v) for v in mc['mc_curve_train']],
        'runtime_s': time.time() - t0,
    }


# ========================== Aggregation and reporting ==========================

def aggregate(results):
    """Group runs by (config_name, kappa); return per-config stats."""
    groups = {}
    for r in results:
        key = (r['config_name'], r['kappa'])
        groups.setdefault(key, []).append(r)
    agg = []
    for (cname, kappa), rs in sorted(groups.items(),
                                     key=lambda kv: (kv[0][0], kv[0][1])):
        def ms(field):
            vals = np.array([r[field] for r in rs], dtype=float)
            return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        ftle_m, ftle_s = ms('ftle_per_pulse')
        mc_m, mc_s = ms('mc_total')
        mctest_m, mctest_s = ms('mc_total_test')
        mck0_m, _ = ms('mc_k0')
        mck0t_m, _ = ms('mc_k0_test')
        inter_m, _ = ms('inter_rms')
        clip_m, _ = ms('alpha_clip_frac')
        gabs_m, _ = ms('g_abs_mean')
        curve = np.array([r['mc_curve'] for r in rs], dtype=float)
        agg.append({
            'config_name': cname, 'kappa': kappa, 'n_runs': len(rs),
            'ftle_mean': ftle_m, 'ftle_std': ftle_s,
            'mc_total_mean': mc_m, 'mc_total_std': mc_s,
            'mc_total_test_mean': mctest_m, 'mc_total_test_std': mctest_s,
            'mc_k0_mean': mck0_m, 'mc_k0_test_mean': mck0t_m,
            'inter_rms_mean': inter_m,
            'alpha_clip_frac_mean': clip_m,
            'g_abs_mean_mean': gabs_m,
            'mc_curve_mean': [float(v) for v in curve.mean(axis=0)],
        })
    return agg


def print_phase_table(agg):
    """Print the kappa-phase table and locate the critical kappa bracket."""
    print("\n" + "=" * 96)
    print("PHASE TABLE (mean over seeds): FTLE per pulse | MC_total (k=1..%d) | clip frac" % K_MAX)
    print("=" * 96)
    topologies = sorted({a['config_name'] for a in agg})
    for topo in topologies:
        rows = [a for a in agg if a['config_name'] == topo]
        print(f"\n--- {topo} ---")
        print(f"  {'kappa':>8} | {'FTLE':>9} | {'MC_tr':>9} | {'MC_te':>9} | "
              f"{'MCk0_te':>7} | {'inter':>7} | {'clip':>6} | {'|g|':>7} | regime")
        prev_ftle = None
        for a in rows:
            if a['ftle_mean'] > 0:
                regime = 'CHAOTIC'
            elif abs(a['ftle_mean']) <= 0.02:
                regime = 'near-edge'
            else:
                regime = 'ordered'
            flag = ''
            if prev_ftle is not None and prev_ftle < 0 <= a['ftle_mean']:
                flag = '  <-- kappa* bracket'
            print(f"  {a['kappa']:>8g} | {a['ftle_mean']:>9.4f} | {a['mc_total_mean']:>9.2f} | "
                  f"{a['mc_total_test_mean']:>9.2f} | {a['mc_k0_test_mean']:>7.2f} | "
                  f"{a['inter_rms_mean']:>7.4f} | "
                  f"{a['alpha_clip_frac_mean']:>6.2f} | {a['g_abs_mean_mean']:>7.2f} | "
                  f"{regime}{flag}")
            prev_ftle = a['ftle_mean']


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START substrate phase diagram v2 "
          f"(quick={quick})")

    # numba warmup in the parent process (populates the disk cache that
    # spawned Pool workers reuse)
    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(8, 10e-6)
    ip_w, idx_w, w_w = build_topology_csr("ring_bidir", 16)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, w_w, 0.1,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    run_pair_ftle_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, w_w, 0.1,
                     ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                     COUPLING_CONTRAST_SELF, 1e-8, 4)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    # build the run list
    n_seeds = N_SEEDS
    t_total, washout, k_max = T_TOTAL, N_WASHOUT, K_MAX
    configs = CONFIGS
    if quick:
        n_seeds = 2
        t_total, washout, k_max = 400, 100, 20
        configs = [("parallel", "parallel", COUPLING_NONE, [0.0]),
                   ("ring_bidir", "ring_bidir", COUPLING_CONTRAST_SELF,
                    [0.03, 1.0]),
                   ("add_random_graph", "random_graph", COUPLING_ADDITIVE, [0.4])]
    all_args = []
    for cname, tname, mode, klist in configs:
        for kappa in klist:
            for s in range(n_seeds):
                all_args.append((cname, tname, mode, kappa, s,
                                 N_UNITS, t_total, washout, k_max))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} "
          f"({len(configs)} config-groups x seeds, N={N_UNITS}, T={t_total})")

    results = []
    with Pool(min(cpu_count(), max(1, n_runs))) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=4):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    # ---- CSV output (batch write) ----
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['config_name', 'topology', 'coupling_mode', 'kappa', 'seed_idx',
                  'n_units', 'cv_tau', 't_total', 'n_washout', 'k_max',
                  'ftle_per_pulse', 'mc_total', 'mc_total_test', 'mc_k0', 'mc_k0_test',
                  'inter_rms', 'alpha_clip_frac', 'g_abs_mean', 'mean_state',
                  'runtime_s']
    with open(CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv'),
              'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    # ---- aggregate + JSON ----
    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'alpha_min': ALPHA_MIN, 'alpha_max': ALPHA_MAX, 'pw': PW,
        'gamma': float(gamma), 'tau0': float(tau0),
        'dt_lo': DT_LO, 'dt_hi': DT_HI,
        't_total': t_total, 'n_washout': washout, 'k_max': k_max,
        'n_seeds': n_seeds, 'eps_benettin': EPS_BENETTIN,
        'renorm_every': RENORM_EVERY, 'ridge_lambda': RIDGE_LAMBDA,
        'topo_seed': TOPO_SEED, 'lateral_radius': LATERAL_RADIUS,
        'avg_degree': AVG_DEGREE,
        'kappa_contrast_mode1': KAPPA_CONTRAST_MODE1,
        'kappa_ring_unidir': KAPPA_RING_UNIDIR,
        'kappa_hub': KAPPA_HUB,
        'kappa_additive': KAPPA_ADDITIVE,
        'mc_test_frac': 0.3,
        'configs': [[c[0], c[1], c[2], c[3]] for c in configs],
        'quick': bool(quick),
    }
    with open(JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json'),
              'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_phase_table(agg)
    print(f"\nCSV : {CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')}")
    print(f"JSON: {JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def main():
    if '--selftest' in sys.argv:
        print("=" * 64)
        print("substrate core self-test")
        print("=" * 64)
        all_ok = True
        for name, ok, detail in substrate_self_test():
            status = "PASS" if ok else "FAIL"
            all_ok = all_ok and ok
            print(f"  [{status}] {name}: {detail}")
        if not all_ok:
            sys.exit(1)
        print("core self-test passed")
        return
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
