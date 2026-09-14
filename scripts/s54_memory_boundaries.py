#!/usr/bin/env python3
"""
L4 memory boundaries: capacity (K slots), regime count (M vs K), and
forgetting (revisit interval). Does memory fail by EXHAUSTION or by TIME?
(REDEM S54)
=============================================================================
Type:           PAPER
Paper Section:  S52 follow-up (L4 memory capacity / forgetting boundaries)
Experiment:     s52 showed frozen-hypothesis memory + reward-rate selection
                beats continuous re-adaptation under A-B-A-B revisits
                (mean 0.888 vs 0.838). s54 maps WHERE that memory fails:

                  K-axis  : memory slots POP_K-1 in {1,3,7} (POP_K 2/4/8),
                            M=2 cycle. Prediction: 1 slot cannot hold both
                            hypotheses (it always holds the CURRENT regime's
                            snapshot, so a revisit finds a stale one) ->
                            K2 ~ rls_single; K4/K8 equal (2 distinct
                            hypotheses fit in >=2 slots).
                  M-axis  : M distinct boundaries in {2,3,4} with POP_K=4
                            (3 slots). Prediction: M=2,3 fit; M=4 SATURATES
                            the memory (one hypothesis is evicted, its
                            revisit is unmemorized) -> gain(M2) >= gain(M3)
                            > gain(M4) >= 0. Boundaries: rotated LINEAR
                            boundaries at theta_k = 0.6 + k*pi/4 (k=0..3,
                            all ~0.96 decodable) plus the hard CIRCLE
                            boundary (the s52 case where re-learning is
                            slow and memory matters most): M=2 = [lin0,
                            cir], M=3 = [lin0, cir, lin1], M=4 = [lin0,
                            cir, lin1, lin2].
                  S-axis  : revisit interval (segment length) in {250,500,
                            800} blocks, M=2, POP_K=4. Prediction: frozen
                            snapshots do NOT time-decay on a static stream
                            (same boundary = same solution); absence only
                            decays the SELECTION EMA, which re-climbs on
                            return. So interval alone does NOT kill memory
                            (steady gain persists) -- forgetting is
                            EVICTION-driven (capacity), not time-driven.

Environment: cyclic sequence of rotated-linear boundaries on the 2D
substrate (random_graph_k25, the decodable substrate), raw features (s53:
they already carry the decodable structure), K=20, T=40000 (2000 blocks).

Arms (block-vote scoring, protocol decision):
  rls_single : one RLS, continuous adaptation (baseline).
  pop_mem    : worker RLS + POP_K-1 frozen snapshots; functional-identity
               dedup (agreement > 0.85 on recent blocks), peak-EMA eviction,
               argmax-r_ema selection (the s52 v4 policy, parameterized by
               POP_K).

Predictions:
  P1 (K-axis): K2 ~ rls_single (1 slot cannot hold 2 hypotheses); K4 == K8
      (both hold 2 distinct hypotheses).
  P2 (M-axis): pop_mem gain over rls_single shrinks as M exceeds slot
      capacity: gain(M2) >= gain(M3) > gain(M4); M4 is the saturation
      boundary where the evicted hypothesis's revisit is unmemorized.
      [CAVEAT (2026-09-09, dedup P1-94): with N_BLOCKS=2000 and
      seg_blocks=500, M4 covers exactly ONE cycle (4 x 500 blocks) and the
      environment therefore contains NO revisit -- the saturation mechanism
      this prediction names is not observable in the M4 cell as configured.
      Reading the M4 cell as evidence for/against P2 is invalid; either
      raise N_BLOCKS for the M axis or report M4 as unobservable.]
  P3 (S-axis): interval alone does not degrade the steady gain (P2's gain
      persists at all segment lengths) -- memory fails by EXHAUSTION, not
      by elapsed time.

Output files:
  data/s54_memory_boundaries_v1.csv    (one row per run)
  data/s54_memory_boundaries_v1.json   (params + per-cell aggregates)

Usage: python s54_memory_boundaries.py [--quick]
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
from online_readout import OnlineRLS, running_mean_accuracy
from streaming_tasks import (gen_nonlinear2d, DB_K_PULSES, ROT2D_THETA0,
                             ROT2D_BAND0, ROT2D_BAND1)

# ==================== Fixed parameters (S3/S39-S53-consistent) ====================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10
FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

RLS_FORGETTING = 0.99
RLS_INIT_COV = 1.0

PROBE_EMA_ALPHA = 0.02
SNAP_EMA_MIN = 0.55
SNAP_HOLD = 20
AGREE_W = 50
AGREE_THRESH = 0.85

N_BLOCKS = 2000

# rotated-linear boundary repertoire (all ~0.96 decodable on random_graph);
# the base angle is the CORE rotation-task angle ROT2D_THETA0 (= 0.6 rad),
# imported instead of re-declared (dedup P1-95)
THETAS = [ROT2D_THETA0 + k * np.pi / 4 for k in range(4)]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's54_memory_boundaries_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's54_memory_boundaries_v1.json')

CURVE_WINDOW = 200

ARMS = ['rls_single', 'pop_mem']

# env grid: (axis, config_id, M, seg_blocks, pop_k)
ENVS = [
    ('K', 'K2', 2, 500, 2),
    ('K', 'K4', 2, 500, 4),
    ('K', 'K8', 2, 500, 8),
    ('M', 'M2', 2, 500, 4),
    ('M', 'M3', 3, 500, 4),
    ('M', 'M4', 4, 500, 4),
    ('S', 'S250', 2, 250, 4),
    ('S', 'S500', 2, 500, 4),
    ('S', 'S800', 2, 800, 4),
]


def build_substrate(topo_name):
    if topo_name == 'parallel':
        ip, idx, wt = build_topology_csr('parallel', N_UNITS)
        return ip, idx, wt, COUPLING_NONE, 0.0
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def acc_median_in(acc_run, b_lo, b_hi):
    lo, hi = b_lo * DB_K_PULSES, b_hi * DB_K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


def gen_linear_theta(seed, n_blocks, theta):
    """Dual-band encoding of a linear boundary at arbitrary theta (same
    encoding as gen_nonlinear2d 'linear' but theta is a parameter)."""
    rng = np.random.RandomState(seed)
    u0 = rng.uniform(0.0, 1.0, n_blocks)
    u1 = rng.uniform(0.0, 1.0, n_blocks)
    # boundary centering tau_b = 0.5*(cos+sin) is the CORE 2D convention
    # (gen_rotation2d / gen_nonlinear2d, streaming_tasks)
    tau_b = 0.5 * (np.cos(theta) + np.sin(theta))
    labels = (np.cos(theta) * u0 + np.sin(theta) * u1 > tau_b).astype(np.int64)
    # CORE 2D dual-band encoding, imported instead of re-declared (dedup P1-95)
    b0 = ROT2D_BAND0
    b1 = ROT2D_BAND1
    lo0, hi0 = b0[0], b0[1]
    lo1, hi1 = b1[0], b1[1]
    dt = np.empty(n_blocks * DB_K_PULSES, dtype=np.float64)
    for b in range(n_blocks):
        seg = np.empty(DB_K_PULSES, dtype=np.float64)
        seg[0::2] = lo0 + u0[b] * (hi0 - lo0)
        seg[1::2] = lo1 + u1[b] * (hi1 - lo1)
        dt[b * DB_K_PULSES:(b + 1) * DB_K_PULSES] = seg
    return dt, np.repeat(labels, DB_K_PULSES).astype(np.int64)


# boundary sequence types per regime index: 'lin{k}' = rotated linear at
# THETAS[k], 'cir' = circle (the hard boundary where RLS re-learns slowly)
def boundary_types(m):
    """Return a list of m boundary type strings for the M-axis."""
    if m == 2:
        return ['lin0', 'cir']
    if m == 3:
        return ['lin0', 'cir', 'lin1']
    return ['lin0', 'cir', 'lin1', 'lin2']


def gen_cycle(seed, m, seg_blocks):
    """Cyclic sequence of m boundaries (rotated lines + the hard circle).

    Per-block sub-seeds are derived from the (seed, block) PAIR through
    RandomState's array-seed mixing. The previous formula `seed * 1009 + b`
    collided linearly: (s, b) and (s+1, b-1009) produced the SAME sub-seed
    (and ~98% of the time the same boundary type), so ~990 of 2000 blocks were
    shared between adjacent seeds and the effective sample size was inflated
    (bug fix 2026-09-09, dedup P1-94).
    """
    types = boundary_types(m)
    dt_parts = []
    tgt_parts = []
    for b in range(N_BLOCKS):
        btype = types[b // seg_blocks % m]
        s = int(np.random.RandomState([int(seed), int(b)]).randint(
            0, 2**31 - 1))
        if btype == 'cir':
            dt, tgt, _, _ = gen_nonlinear2d(seed=s, n_blocks=1,
                                            k_pulses=DB_K_PULSES,
                                            boundary='circle')
            dt_parts.append(dt)
            tgt_parts.append(tgt)
        else:
            k = int(btype[3:])
            dt, tgt = gen_linear_theta(s, 1, THETAS[k])
            dt_parts.append(dt)
            tgt_parts.append(tgt)
    return np.concatenate(dt_parts), np.concatenate(tgt_parts).astype(np.int64)


def build_features(states, T):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    return np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])


# ========================== Single run ==========================

def run_single(args):
    """(env_id, arm, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    env_id, arm, seed_idx = args
    axis, cfg, m, seg_blocks, pop_k = next(e for e in ENVS if e[1] == env_id)
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate('random_graph_k25')

    dt_seq, target_seq = gen_cycle(seed_idx, m, seg_blocks)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F = build_features(states, T)
    d = F.shape[1]

    preds = np.empty(T)
    n_snaps = 0
    active_mem_blocks = 0

    if arm == 'rls_single':
        rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV)
        for t in range(T):
            preds[t] = float(rls.predict(F[t])[0])
            if (t + 1) % DB_K_PULSES == 0:
                seg = F[t - DB_K_PULSES + 1:t + 1]
                x_blk = seg.mean(axis=0)
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                rls.update(x_blk, np.array([y_derived]))
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
    else:  # pop_mem (s52 v4 policy)
        K = pop_k
        worker = OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                           init_cov=RLS_INIT_COV)
        mem_w = []
        mem_ema = []
        peak_ema = []
        r_ema_w = 0.0
        hold_cnt = 0
        active_kind = 0
        active_idx = 0
        recent_blk = []
        for t in range(T):
            if active_kind == 0:
                preds[t] = float(worker.predict(F[t])[0])
            else:
                preds[t] = float(mem_w[active_idx] @ F[t])
            if (t + 1) % DB_K_PULSES == 0:
                seg = F[t - DB_K_PULSES + 1:t + 1]
                x_blk = seg.mean(axis=0)
                recent_blk.append(x_blk)
                if len(recent_blk) > AGREE_W:
                    recent_blk.pop(0)
                wv = float(worker.predict(x_blk)[0])
                r_w = 1.0 if (wv > 0.5) == target[t] else -1.0
                r_ema_w = (1.0 - PROBE_EMA_ALPHA) * r_ema_w \
                    + PROBE_EMA_ALPHA * r_w
                for i in range(len(mem_w)):
                    mv = float(mem_w[i] @ x_blk)
                    r_m = 1.0 if (mv > 0.5) == target[t] else -1.0
                    mem_ema[i] = (1.0 - PROBE_EMA_ALPHA) * mem_ema[i] \
                        + PROBE_EMA_ALPHA * r_m
                v_w = float(wv > 0.5)
                r_w2 = 1.0 if v_w == target[t] else -1.0
                y_d = v_w if r_w2 > 0.0 else (1.0 - v_w)
                worker.update(x_blk, np.array([y_d]))
                # snapshot (functional-identity dedup + peak-EMA eviction)
                if r_ema_w > SNAP_EMA_MIN:
                    hold_cnt += 1
                    if hold_cnt >= SNAP_HOLD:
                        snap = worker.W[:, 0].copy()
                        if recent_blk:
                            wv_agree = (snap @ np.array(recent_blk).T > 0.5)
                            agree = []
                            for mw in mem_w:
                                m_agree = (mw @ np.array(recent_blk).T > 0.5)
                                agree.append(float(np.mean(
                                    wv_agree == m_agree)))
                            best_agr = max(agree) if agree else 0.0
                            i_best = int(np.argmax(agree)) if agree else 0
                        else:
                            best_agr, i_best = 0.0, 0
                            agree = []
                        if agree and best_agr > AGREE_THRESH:
                            mem_w[i_best] = snap
                            mem_ema[i_best] = r_ema_w
                            peak_ema[i_best] = max(peak_ema[i_best], r_ema_w)
                            n_snaps += 1
                        else:
                            if len(mem_w) < K - 1:
                                mem_w.append(snap)
                                mem_ema.append(r_ema_w)
                                peak_ema.append(r_ema_w)
                                n_snaps += 1
                            elif agree:
                                i_low = int(np.argmin(peak_ema))
                                mem_w[i_low] = snap
                                mem_ema[i_low] = r_ema_w
                                peak_ema[i_low] = r_ema_w
                                n_snaps += 1
                        hold_cnt = 0
                else:
                    hold_cnt = 0
                scores = [r_ema_w] + list(mem_ema)
                best = int(np.argmax(scores))
                if best == 0:
                    active_kind = 0
                    preds[t - DB_K_PULSES + 1:t + 1] = wv
                else:
                    active_kind = 1
                    active_idx = best - 1
                    active_mem_blocks += 1
                    preds[t - DB_K_PULSES + 1:t + 1] = \
                        float(mem_w[active_idx] @ x_blk)

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    # per-boundary-type accuracy (mean over that type's segments)
    types = boundary_types(m)
    per_type = {}
    for b in range(N_BLOCKS):
        bt = types[b // seg_blocks % m]
        lo, hi = b * DB_K_PULSES, (b + 1) * DB_K_PULSES
        seg = acc_run[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            per_type.setdefault(bt, []).append(float(np.nanmedian(seg)))
    type_acc = {f"bt{idx}_{bt}": float(np.mean(per_type.get(bt, [float('nan')])))
                for idx, bt in enumerate(types[:m])}

    res = {'task': 'memory_boundaries', 'axis': axis, 'env': env_id,
           'm_regimes': m, 'seg_blocks': seg_blocks, 'pop_k': pop_k,
           'substrate': 'random_graph_k25', 'readout': arm,
           'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'steady_acc': acc_median_in(acc_run, 1800, 2000),
           'n_snaps': n_snaps, 'active_mem_blocks': active_mem_blocks,
           'runtime_s': time.time() - t0}
    res.update(type_acc)
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['env'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        env, read = key
        entry = {'env': env, 'axis': rs[0]['axis'], 'm_regimes': rs[0]['m_regimes'],
                 'seg_blocks': rs[0]['seg_blocks'], 'pop_k': rs[0]['pop_k'],
                 'readout': read, 'n_runs': len(rs)}
        for f in ['mean_acc_all', 'steady_acc']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        entry['n_snaps_mean'] = float(np.nanmean(
            np.array([r['n_snaps'] for r in rs], dtype=float)))
        entry['active_mem_frac'] = float(np.nanmean(
            np.array([r['active_mem_blocks'] for r in rs], dtype=float)
            / (N_BLOCKS / 1)))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 118)
    print("S54 L4 MEMORY BOUNDARIES (K capacity / M saturation / S interval)")
    print("=" * 118)
    for axis in ('K', 'M', 'S'):
        print(f"\n--- axis {axis} ---")
        print(f"  {'env':<6} {'M':>2} {'seg':>5} {'K':>2} | "
              f"{'rls mean':>8} {'pop mean':>8} {'gain':>7} | "
              f"{'snaps':>5} {'mem%':>5}")
        for env in sorted(set(a['env'] for a in agg if a['axis'] == axis)):
            rls = next((a for a in agg if a['env'] == env
                        and a['readout'] == 'rls_single'), None)
            pop = next((a for a in agg if a['env'] == env
                        and a['readout'] == 'pop_mem'), None)
            if rls and pop:
                gain = pop['mean_acc_all_mean'] - rls['mean_acc_all_mean']
                print(f"  {env:<6} {pop['m_regimes']:>2} "
                      f"{pop['seg_blocks']:>5} {pop['pop_k']:>2} | "
                      f"{rls['mean_acc_all_mean']:>8.3f} "
                      f"{pop['mean_acc_all_mean']:>8.3f} "
                      f"{gain:>+7.3f} | {pop['n_snaps_mean']:>5.1f} "
                      f"{pop['active_mem_frac']:>5.2f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S54 memory boundaries "
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
    for _, env_id, _, _, _ in ENVS:
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

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'axis', 'env', 'm_regimes', 'seg_blocks', 'pop_k',
                  'substrate', 'readout', 'seed_idx', 'n_units', 't_total',
                  'runtime_s', 'mean_acc_all', 'steady_acc', 'n_snaps',
                  'active_mem_blocks', 'bt0_lin0', 'bt1_cir', 'bt2_lin1',
                  'bt3_lin2']
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
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'probe_ema_alpha': PROBE_EMA_ALPHA, 'snap_ema_min': SNAP_EMA_MIN,
        'snap_hold': SNAP_HOLD, 'agree_w': AGREE_W,
        'agree_thresh': AGREE_THRESH,
        'n_blocks': N_BLOCKS, 'thetas': [float(t) for t in THETAS],
        # encoding traceability (dedup P1-95): the 2D dual-band limits and the
        # base boundary angle above come from streaming_tasks; the circle
        # radius is the CORE default of gen_nonlinear2d(radius=0.40) that this
        # script relies on by not passing it.
        'rot2d_band0': [float(ROT2D_BAND0[0]), float(ROT2D_BAND0[1])],
        'rot2d_band1': [float(ROT2D_BAND1[0]), float(ROT2D_BAND1[1])],
        'rot2d_theta0': float(ROT2D_THETA0),
        'circle2d_radius': 0.40,
        'envs': [{'env': e[1], 'axis': e[0], 'm': e[2], 'seg': e[3],
                  'pop_k': e[4]} for e in ENVS],
        'n_seeds': n_seeds, 'quick': bool(quick),
        'subseed_scheme': ('RandomState([seed, block]) array-seed mixing '
                           '(fixes the seed*1009+b linear collisions, '
                           'dedup P1-94)'),
        'm4_revisit_observable': False,
        'm4_revisit_note': ('M4 = 4 x 500 blocks == N_BLOCKS: exactly one '
                            'cycle, no revisit; P2 not observable in this '
                            'cell as configured'),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
