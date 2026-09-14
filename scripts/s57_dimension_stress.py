#!/usr/bin/env python3
"""
Completing the corrected L3 stress-response curve: does 4D (hypersphere /
spherical shell) finally trigger representation-insufficiency on the COUPLED
substrate, where 2D/3D degraded but did not collapse? (REDEM S57)
=============================================================================
Type:           PAPER
Paper Section:  S55 follow-up (corrected L3 stress: dimension + nonlinearity)
Experiment:     s55 established input dimension as a CONTINUOUS L3 stress on
                the coupled substrate: well-conditioned (top-50 PCA) probe
                2D circle 0.800 -> 3D sphere 0.630 (rls 0.890 -> 0.733),
                degraded but NOT collapsed; parallel stays at chance. s57
                escalates to 4D with a NEW CORE 4-channel interleave encoding
                (gen_nonlinear4d) and asks whether dimension alone, or a
                STRONGER nonlinearity at 4D, finally pushes the coupled
                substrate to chance. The 2D/3D ladder is re-measured in the
                SAME script (same seeds, same probe convention) so the
                stress-response curve is internally consistent.

Arms (static boundary, block-vote scoring on the protocol decision):
  rls_raw      : RLS on raw features (can it exploit the substrate's
                 expansion?).
  delta_raw    : delta on raw features (first-order control).
  ridge probe  : well-conditioned (top-50 PCA) block-level ridge on raw
                 features -- the capacity ceiling, computed in main (seed 0).

Envs (boundary x substrate, with per-dim pulses/block):
  DIMENSION AXIS (pure quadratic ball):
  (2D circle,  random_graph_k25)  : ladder reference (s53/s55: 0.800).
  (2D circle,  parallel)          : ref control (chance).
  (3D sphere,  random_graph_k25)  : ladder reference (s55: 0.630).
  (3D sphere,  parallel)          : ref control (chance).
  (4D sphere,  random_graph_k25)  : 4D ladder point. NOTE: label balance at
                                    dim>=4 forces radius > 0.5 (no balanced
                                    fully-inside 4D ball exists), so the
                                    balanced 4D ball CLIPS the unit cube
                                    (r=0.570, P=0.502) -- the reading is
                                    confounded by the cube-face intersection.
  (4D sphere,  parallel)          : 4D control (expect chance).
  COMPLEXITY AXIS (annulus: TWO quadratic boundaries, balanced by
  construction at every dimension):
  (2D ring,    random_graph_k25)  : 2D annulus (0.30, 0.50) -> P=0.503.
  (2D ring,    parallel)          : control.
  (3D shell,   random_graph_k25)  : 3D annulus (0.179, 0.50) -> P=0.495,
                                    both boundaries fully inside (clean).
  (3D shell,   parallel)          : control.
  (4D shell,   random_graph_k25)  : 4D annulus (0.45, 0.640) -> P=0.501.
  (4D shell,   parallel)          : control.
  ENCODING CONTROL:
  (4D plane,   random_graph_k25)  : must stay easily decodable (validates
                                    the 4-channel interleave carries all
                                    four dims).

Predictions:
  P1 (4D encoding control): 4D plane probe >= ~0.75 on random_graph -- the
      4-channel interleave is a valid 4D encoding; any sphere/shell
      degradation is boundary-driven, not encoding-driven.
  P2 (dimension axis): 2D circle 0.800 -> 3D sphere 0.630 -> 4D sphere
      (balanced, cube-clipped): degradation is expected to continue, but
      the 4D point is CONFOUNDED by the label-balance clipping (flat
      cube-face pieces) and may read above the 3D point -- reported as an
      artifact of the balance constraint, not a reversal.
  P3 (complexity axis): the annulus is harder than the ball at the same
      dimension; 2D ring > 3D shell > 4D shell expected, with 4D shell
      predicted to approach or reach chance (the stronger nonlinearity
      finally triggers representation-insufficiency on the coupled
      substrate).
  P4 (online): rls_raw tracks the probe (structural gain exploits whatever
      the substrate provides, rls > probe gap as in s55); delta_raw stays
      at chance everywhere (first-order).
  P5 (parallel control): all boundaries at chance on parallel; delta
      first-order at chance everywhere.

Output files:
  data/s57_dimension_stress_v1.csv    (one row per run)
  data/s57_dimension_stress_v1.json   (params + per-cell aggregates + probes)

Usage: python s57_dimension_stress.py [--quick]
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper_e', 'deps'))
from shallow_trap_array_simulator import gamma, tau0, gen_tau_vec, preprogram_vec
from recurrent_substrate import (
    COUPLING_NONE, COUPLING_CONTRAST_SELF,
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    run_trajectory_nb)
from online_readout import OnlineRLS, ridge_fit, running_mean_accuracy
from streaming_tasks import (gen_nonlinear2d, gen_nonlinear3d,
                             gen_nonlinear4d, DB_K_PULSES, ROT3D_K_PULSES,
                             ROT4D_K_PULSES, ROT4D_SPHERE_R, ROT4D_SHELL_IN,
                             ROT4D_SHELL_OUT)

# ==================== Fixed parameters (S3/S39-S56-consistent) ====================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

DELTA_ETA = 0.05
W_MAX = 20.0
RLS_FORGETTING = 0.99
RLS_INIT_COV = 1.0

CAP_N_PC = 50          # PCA components for the well-conditioned probe

N_BLOCKS = 2000
# pulses per block come from the CORE interleave (3 channels x 7 / 4 x 6);
# imported instead of re-declared (dedup P1-95)
K3D = ROT3D_K_PULSES
K4D = ROT4D_K_PULSES

# annulus radii (balanced by construction; labels recomputed in gen_env)
RING_IN = 0.30         # 2D ring inner  -> P(y=1)=0.503 with outer 0.50
RING_OUT = 0.50
SHELL3_IN = 0.179      # 3D shell inner  -> P(y=1)=0.495 with outer 0.50
SHELL3_OUT = 0.50
SHELL4_IN = ROT4D_SHELL_IN    # 4D shell inner (CORE value, was literal 0.45)
SHELL4_OUT = ROT4D_SHELL_OUT  # 4D shell outer (CORE value, was 0.640) -> P=0.501

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's57_dimension_stress_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's57_dimension_stress_v1.json')

CURVE_WINDOW = 200

ARMS = ['rls_raw', 'delta_raw']

# env: (dim, boundary, substrate, k_pulses)
ENVS = [
    ('2D', 'circle', 'random_graph_k25', DB_K_PULSES),
    ('2D', 'circle', 'parallel', DB_K_PULSES),
    ('2D', 'ring', 'random_graph_k25', DB_K_PULSES),
    ('2D', 'ring', 'parallel', DB_K_PULSES),
    ('3D', 'sphere', 'random_graph_k25', K3D),
    ('3D', 'sphere', 'parallel', K3D),
    ('3D', 'shell', 'random_graph_k25', K3D),
    ('3D', 'shell', 'parallel', K3D),
    ('4D', 'plane', 'random_graph_k25', K4D),
    ('4D', 'sphere', 'random_graph_k25', K4D),
    ('4D', 'sphere', 'parallel', K4D),
    ('4D', 'shell', 'random_graph_k25', K4D),
    ('4D', 'shell', 'parallel', K4D),
]


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def acc_median_in(acc_run, b_lo, b_hi, k):
    lo, hi = b_lo * k, b_hi * k
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _cap(w, w_max):
    nw = float(np.linalg.norm(w))
    if nw > w_max:
        w = w * (w_max / nw)
    return w


def gen_env(dim, boundary, seed, k):
    # 2D/3D geometry uses the CORE defaults of the generators on purpose:
    # gen_nonlinear2d(radius=0.40) for the 2D circle and
    # gen_nonlinear3d(radius=0.50) for the 3D sphere. Both effective values are
    # recorded in the params JSON (dedup P1-95).
    if dim == '2D':
        dt, tgt, u0, u1 = gen_nonlinear2d(seed=seed, n_blocks=N_BLOCKS,
                                          k_pulses=k, boundary='circle')
        if boundary == 'ring':
            d2 = (u0 - 0.5) ** 2 + (u1 - 0.5) ** 2
            labels = ((d2 > RING_IN ** 2) & (d2 < RING_OUT ** 2)).astype(np.int64)
            tgt = np.repeat(labels, k)
        return dt, tgt
    if dim == '3D':
        dt, tgt, u0, u1, u2 = gen_nonlinear3d(seed=seed, n_blocks=N_BLOCKS,
                                              k_pulses=k, boundary='sphere')
        if boundary == 'shell':
            d2 = ((u0 - 0.5) ** 2 + (u1 - 0.5) ** 2 + (u2 - 0.5) ** 2)
            labels = ((d2 > SHELL3_IN ** 2) & (d2 < SHELL3_OUT ** 2)).astype(np.int64)
            tgt = np.repeat(labels, k)
        return dt, tgt
    # 4D: boundary in {'plane', 'sphere', 'shell'}; the geometry is passed
    # explicitly from the CORE names (identical to the CORE defaults, so the
    # numerical result is unchanged) instead of being re-declared here (P1-95)
    bd = 'linear' if boundary == 'plane' else boundary
    dt, tgt, _, _, _, _ = gen_nonlinear4d(seed=seed, n_blocks=N_BLOCKS,
                                          k_pulses=k, boundary=bd,
                                          radius=ROT4D_SPHERE_R,
                                          shell_in=SHELL4_IN,
                                          shell_out=SHELL4_OUT)
    return dt, tgt


def build_features(states, T):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    return np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])


def capacity_pca(F, tgt, b_lo, b_hi, k, n_pc=CAP_N_PC):
    """Well-conditioned capacity probe: held-out ridge on top-n_pc PCA of
    block-mean features (s55)."""
    Fb = np.array([F[b * k:(b + 1) * k].mean(axis=0) for b in range(b_lo, b_hi)])
    yb = tgt[b_lo * k:(b_hi * k):k].astype(np.float64)
    n = Fb.shape[0]
    n_tr = n // 2
    mu = Fb[:n_tr].mean(axis=0)
    sd = Fb[:n_tr].std(axis=0) + 1e-9
    Xs = (Fb - mu) / sd
    U, S, Vt = np.linalg.svd(Xs[:n_tr], full_matrices=False)
    kk = min(n_pc, Vt.shape[0])
    Xp = Xs @ Vt[:kk].T
    out = ridge_fit(Xp[:n_tr], yb[:n_tr], Xp[n_tr:], yb[n_tr:],
                    ridge_lambda=1.0)
    pred = out['pred_te']
    return float(np.mean((pred > 0.5).astype(np.float64) == yb[n_tr:]))


# ========================== Single run ==========================

def run_single(args):
    """(env_id, arm, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    env_id, arm, seed_idx = args
    dim, boundary, topo_name, k = ENVS[env_id]
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq = gen_env(dim, boundary, seed_idx, k)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F = build_features(states, T)
    d = F.shape[1]

    preds = np.empty(T)

    if arm == 'rls_raw':
        rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV)
        for t in range(T):
            preds[t] = float(rls.predict(F[t])[0])
            if (t + 1) % k == 0:
                seg = F[t - k + 1:t + 1]
                x_blk = seg.mean(axis=0)
                blk = preds[t - k + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                rls.update(x_blk, np.array([y_derived]))
                preds[t - k + 1:t + 1] = np.mean(blk)
    else:  # delta_raw
        rng_w = np.random.RandomState(seed_idx * 97 + 3)
        w = rng_w.uniform(-0.5, 0.5, d)
        acc_sF = np.zeros(d)
        acc_oF = np.zeros(d)
        for t in range(T):
            o = _sigmoid(float(w @ F[t]))
            preds[t] = o
            acc_sF += F[t]
            acc_oF += F[t] * o
            if (t + 1) % k == 0:
                blk = preds[t - k + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                w += DELTA_ETA * (y_derived * acc_sF - acc_oF)
                w = _cap(w, W_MAX)
                acc_sF[:] = 0.0
                acc_oF[:] = 0.0
                preds[t - k + 1:t + 1] = np.mean(blk)

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'dimension_stress', 'dim': dim, 'boundary': boundary,
           'substrate': topo_name, 'k_pulses': k, 'readout': arm,
           'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'pre_swap_acc': acc_median_in(acc_run, 800, 1000, k),
           'steady_acc': acc_median_in(acc_run, 1800, 2000, k),
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['dim'], r['boundary'], r['substrate'],
                           r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        dim, boundary, topo, read = key
        entry = {'dim': dim, 'boundary': boundary, 'substrate': topo,
                 'readout': read, 'n_runs': len(rs)}
        for f in ['mean_acc_all', 'pre_swap_acc', 'steady_acc']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        agg.append(entry)
    return agg


def print_table(agg, probes):
    print("\n" + "=" * 122)
    print("S57 DIMENSION STRESS: 4D ESCALATION OF THE CORRECTED L3 CURVE")
    print("=" * 122)
    print(f"\n  well-conditioned ridge probes (top-{CAP_N_PC} PCA, seed 0):")
    for k in sorted(probes.keys()):
        print(f"    {k}: {probes[k]:.3f}")
    print(f"\n  {'env':<30} | {'readout':<9} | {'pre':>6} | "
          f"{'steady':>6} | {'mean':>6}")
    for a in agg:
        env = f"{a['dim']} {a['boundary']} {a['substrate']}"
        print(f"  {env:<30} | {a['readout']:<9} | "
              f"{a['pre_swap_acc_mean']:>6.3f} | "
              f"{a['steady_acc_mean']:>6.3f} | "
              f"{a['mean_acc_all_mean']:>6.3f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S57 dimension stress "
          f"(quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16,
                                           seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_RANDOM,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = N_SEEDS if not quick else 2
    all_args = []
    for env_id in range(len(ENVS)):
        for arm in ARMS:
            for s in range(n_seeds):
                all_args.append((env_id, arm, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (envs={len(ENVS)}, arms={len(ARMS)}, "
          f"seeds={n_seeds})")

    results = []
    n_proc = min(cpu_count(), 8, max(1, n_runs))
    with Pool(n_proc) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=2):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                      flush=True)

    # well-conditioned capacity probes (seed 0, on the ladder + controls)
    probes = {}
    for env_id, (dim, boundary, topo_name, k) in enumerate(ENVS):
        tau_p = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=0)
        x0_p = preprogram_vec(ALPHA0, tau_p)
        ip_p, idx_p, wt_p, mode_p, kap_p = build_substrate(topo_name)
        dt_p, tgt_p = gen_env(dim, boundary, 0, k)
        st_p, _, _ = run_trajectory_nb(
            x0_p, tau_p, dt_p, PW, ip_p, idx_p, wt_p, kap_p,
            ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode_p, 0)
        F_p = build_features(st_p, dt_p.shape[0])
        key = f"{dim}|{boundary}|{topo_name}"
        probes[key] = capacity_pca(F_p, tgt_p, 800, 1000, k)
    for k, v in probes.items():
        print(f"  ridge probe {k}: {v:.3f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'dim', 'boundary', 'substrate', 'k_pulses',
                  'readout', 'seed_idx', 'n_units', 't_total', 'runtime_s',
                  'mean_acc_all', 'pre_swap_acc', 'steady_acc']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'topo_seed': TOPO_SEED, 'avg_degree': AVG_DEGREE,
        'kappa_random': KAPPA_RANDOM,
        'delta_eta': DELTA_ETA, 'w_max': W_MAX,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'cap_n_pc': CAP_N_PC,
        'n_blocks': N_BLOCKS, 'k3d': K3D, 'k4d': K4D,
        'ring2d_in': RING_IN, 'ring2d_out': RING_OUT,
        'shell3d_in': SHELL3_IN, 'shell3d_out': SHELL3_OUT,
        # 4D geometry now comes from the CORE exports (dedup P1-95); the 2D/3D
        # radii are the CORE generator defaults this script relies on by not
        # passing them.
        'sphere4d_radius': float(ROT4D_SPHERE_R), 'shell4d_in': SHELL4_IN,
        'shell4d_out': SHELL4_OUT,
        'circle2d_radius': 0.40,
        'sphere3d_radius': 0.50,
        'envs': [{'dim': e[0], 'boundary': e[1], 'substrate': e[2],
                  'k': e[3]} for e in ENVS],
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg, 'ridge_probe': probes},
                  f, indent=2)

    print_table(agg, probes)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
