#!/usr/bin/env python3
"""
Self-tuned snapshot gate: can the memory layer learn WHEN to keep experience
without a hand-set threshold? (REDEM G1 = s59)
=============================================================================
Type:           PAPER
Paper Section:  S41 value-learning generalised to the memory layer
Experiment:     The only remaining hand-set threshold in the L4 memory
                system is the snapshot gate SNAP_EMA_MIN=0.55 (s52 v4): the
                worker is snapshotted only when its reward-rate EMA exceeds
                it. s41 showed a meta-layer can learn its OWN flip policy
                (value v) from the reward-rate change, converging to +0.504
                in a swapping environment and ~0.150 in a stable one. G1 asks
                the same question for the MEMORY layer: can the system learn
                WHEN it is worth snapshotting, i.e. learn the snapshot gate
                v_snap online, so that no hand-set SNAP threshold exists?

Environment (both on random_graph_k25, the s52 substrate):
  dyn    : A-B-A-B [linear, circle, linear, circle] x 500 blocks
           (T=2000 blocks, T_pulses=40000) -- the s52 environment where
           frozen snapshots pay off at revisits (seg3 linear, seg4 circle).
  static : single 'linear' regime x 4000 blocks (T_pulses=80000) -- no
           revisits; snapshots are functionally useless (duplicates).

Arms (s52 v4 memory machinery; only the snapshot GATE differs):
  rls_single : worker RLS only, no memory (no-experience baseline).
  pop_mem    : s52 v4 with the FIXED gate SNAP_EMA_MIN=0.55 (fixed-threshold
               experience; dyn = s52 replication).
  mem_vlearn : s52 v4 with a LEARNED gate v_snap (no hand-set SNAP
               threshold): v_snap init 0.30 (optimistic-low -> snapshots
               eagerly), bounded [0.20, 0.95], re-evaluated at every block b
               with b >= last_eval_b + W_EVAL (W_EVAL=400 blocks, block-
               anchored) by the realized usefulness of memory. Usefulness is
               NOT "any memory slot won the argmax selection in the window":
               the counter is `better_blocks`, the number of blocks in the
               window where memory was SELECTED (the argmax of
               [r_ema_w] + mem_ema picked a slot) AND the selected slot
               STRICTLY outperformed the worker on the same block
               (r_m > r_w, r_m = correctness of the selected slot's block
               vote, r_w = correctness of the worker's). If
               better_blocks >= SEL_SUST (SEL_SUST=8) the gate is lowered by
               V_GAMMA (v_snap = max(V_SNAP_LO, v_snap - V_GAMMA), "keep
               remembering"); otherwise it is raised by V_GAMMA
               (v_snap = min(V_SNAP_HI, v_snap + V_GAMMA), "require higher
               quality, i.e. stop wasting capacity"); the counter then
               resets to 0. A mere selection without an improvement is
               therefore NOT enough to keep the gate low.

Value signal (mirrors s41's Delta-r attribution): the strict improvement
r_m > r_w on the selected blocks is the system's own ground truth for "this
hypothesis is useful NOW" (a tie is not evidence). In the dynamic env memory
wins at every revisit (seg3/seg4) and beats the re-learning worker, so the
gate stays low; in the static env the selected frozen copies only tie the
worker on most blocks (measured selected-block fraction 0.118, see P2), so
better_blocks stays below SEL_SUST, the gate rises and snapshotting
self-suppresses.

Predictions (measured values are the 10-seed aggregates in
data/s59_self_tuned_gate_v1.json):
  P1 (dyn, no degradation from self-tuning): mem_vlearn mean_acc ~= pop_mem
      (within noise), both > rls_single; v_snap stays low enough that
      snapshots keep occurring (n_snaps comparable); memory is selected a
      comparable fraction of blocks.
      [data (dyn): mean_acc 0.876 mem_vlearn / 0.885 pop_mem / 0.842
      rls_single (P1 holds); n_snaps 88.3 vs 68.7; selected-block fraction
      0.227 vs 0.308; v_snap_final 0.252, v_snap_q1 0.380.]
  P2 (static, no harm): mem_vlearn mean_acc = rls_single (self-tuning never
      hurts the worker); v_snap RISES (learns snapshots are worthless);
      selection fraction ~ 0; n_snaps reduced toward the end.
      [data (static): mean_acc 0.955 vs 0.946 rls_single (P2 holds);
      v_snap_final 0.922 (near the V_SNAP_HI=0.95 bound) and v_snap_q1 0.420,
      so the gate does rise; but the selected-block fraction is 0.118, i.e.
      small yet NOT ~0, and total n_snaps is 161.1 vs 196.3 for the fixed
      gate. The per-run CSV does not store a within-run snap trajectory, so
      "reduced toward the end" is a mechanism statement, not a measured one.]
  P3 (environment readout, the s41 parallel): v_snap_final(dyn) <
      v_snap_final(static) across seeds -- the learned gate reflects the
      environment (memory valuable vs worthless), just like s41's v
      (+0.504 vs 0.150).
      [data: v_snap_final 0.252 (dyn) < 0.922 (static), P3 holds.]

Output files:
  data/s59_self_tuned_gate_v1.csv    (one row per run)
  data/s59_self_tuned_gate_v1.json   (params + per-cell aggregates)

Usage: python s59_self_tuned_gate.py [--quick]
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
    COUPLING_CONTRAST_SELF, PW, ALPHA0, ALPHA_MIN, ALPHA_MAX,
    build_topology_csr, run_trajectory_nb)
from online_readout import OnlineRLS, running_mean_accuracy
from streaming_tasks import gen_nonlinear2d, DB_K_PULSES

# ==================== Fixed parameters (S3/S39-S52-consistent) ====================
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

# s52 memory policy (v4)
PROBE_EMA_ALPHA = 0.02
SNAP_EMA_MIN = 0.55        # the FIXED gate (pop_mem baseline)
SNAP_HOLD = 20
POP_K = 5
AGREE_W = 50
AGREE_THRESH = 0.85

# G1 self-tuning of the snapshot gate
V_SNAP_INIT = 0.30
V_SNAP_LO = 0.20
V_SNAP_HI = 0.95
V_GAMMA = 0.08
W_EVAL = 400               # usefulness window (blocks)
SEL_SUST = 8               # sustained selection run that counts as useful

N_SEGMENTS = 4
SEG_BLOCKS = 500
SEGMENTS = ['linear', 'circle', 'linear', 'circle']
STATIC_BLOCKS = 4000
STATIC_BOUNDARY = 'linear'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's59_self_tuned_gate_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's59_self_tuned_gate_v1.json')

CURVE_WINDOW = 200

ARMS = ['rls_single', 'pop_mem', 'mem_vlearn']
ENVS = ['dyn', 'static']


def acc_median_in(acc_run, b_lo, b_hi):
    lo, hi = b_lo * DB_K_PULSES, b_hi * DB_K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


def gen_env(env, seed):
    rng = np.random.RandomState(seed)
    if env == 'dyn':
        dt_parts, tgt_parts = [], []
        for bd in SEGMENTS:
            s = int(rng.randint(0, 2**31 - 1))
            dt, tgt, _, _ = gen_nonlinear2d(seed=s, n_blocks=SEG_BLOCKS,
                                            k_pulses=DB_K_PULSES,
                                            boundary=bd)
            dt_parts.append(dt)
            tgt_parts.append(tgt)
        return (np.concatenate(dt_parts),
                np.concatenate(tgt_parts).astype(np.int64))
    # static: single linear regime
    dt, tgt, _, _ = gen_nonlinear2d(seed=seed, n_blocks=STATIC_BLOCKS,
                                    k_pulses=DB_K_PULSES,
                                    boundary=STATIC_BOUNDARY)
    return dt, tgt.astype(np.int64)


def build_features(states, T):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    return np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])


# ========================== Single run ==========================

def run_single(args):
    """(arm, env, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    arm, env, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts = build_topology_csr('random_graph', N_UNITS,
                                              seed=TOPO_SEED,
                                              avg_degree=AVG_DEGREE)

    dt_seq, target_seq = gen_env(env, seed_idx)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)
    n_blocks = T // DB_K_PULSES

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, KAPPA_RANDOM,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, COUPLING_CONTRAST_SELF, 0)
    F = build_features(states, T)
    d = F.shape[1]

    preds = np.empty(T)
    rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV)

    if arm == 'rls_single':
        for t in range(T):
            preds[t] = float(rls.predict(F[t])[0])
            if (t + 1) % DB_K_PULSES == 0:
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                x_blk = F[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
                rls.update(x_blk, np.array([y_derived]))
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
    else:
        vlearn = (arm == 'mem_vlearn')
        v_snap = V_SNAP_INIT if vlearn else SNAP_EMA_MIN
        v_snap_traj = []          # (block, v_snap)
        mem_w = []
        mem_ema = []
        peak_ema = []
        r_ema_w = 0.0
        hold_cnt = 0
        active_kind = 0
        active_idx = 0
        n_snaps = 0
        n_sel_blocks = 0          # blocks where memory won selection (all mem arms)
        better_blocks = 0         # mem-active blocks where r_m > r_w (vlearn)
        recent_blk = []
        last_eval_b = -W_EVAL
        for t in range(T):
            if active_kind == 0:
                preds[t] = float(rls.predict(F[t])[0])
            else:
                preds[t] = float(mem_w[active_idx] @ F[t])
            if (t + 1) % DB_K_PULSES == 0:
                b = (t + 1) // DB_K_PULSES - 1
                seg = F[t - DB_K_PULSES + 1:t + 1]
                x_blk = seg.mean(axis=0)
                recent_blk.append(x_blk)
                if len(recent_blk) > AGREE_W:
                    recent_blk.pop(0)
                wv = float(rls.predict(x_blk)[0])
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
                rls.update(x_blk, np.array([y_d]))
                # snapshot gate = v_snap (learned or fixed)
                if r_ema_w > v_snap:
                    hold_cnt += 1
                    if hold_cnt >= SNAP_HOLD:
                        snap = rls.W[:, 0].copy()
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
                            if len(mem_w) < POP_K - 1:
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
                # G1: usefulness evaluation (time-anchored). Usefulness =
                # memory, when selected, STRICTLY OUTPERFORMS the worker on
                # the same blocks (r_m > r_w for >= SEL_SUST blocks in the
                # window): on a stationary stream the frozen copy equals the
                # worker block-for-block (r_m == r_w, redundant memory -> not
                # useful -> gate rises); at a dynamic revisit the frozen slot
                # beats the re-learning worker (r_m > r_w for many blocks ->
                # useful -> gate stays low).
                if vlearn:
                    if b >= last_eval_b + W_EVAL:
                        if better_blocks >= SEL_SUST:
                            v_snap = max(V_SNAP_LO, v_snap - V_GAMMA)
                        else:
                            v_snap = min(V_SNAP_HI, v_snap + V_GAMMA)
                        v_snap_traj.append((b, v_snap))
                        last_eval_b = b
                        better_blocks = 0
                # select active
                scores = [r_ema_w] + list(mem_ema)
                best = int(np.argmax(scores))
                if best == 0:
                    active_kind = 0
                    preds[t - DB_K_PULSES + 1:t + 1] = wv
                else:
                    active_kind = 1
                    active_idx = best - 1
                    preds[t - DB_K_PULSES + 1:t + 1] = \
                        float(mem_w[active_idx] @ x_blk)
                    n_sel_blocks += 1
                    if vlearn:
                        mv = float(mem_w[active_idx] @ x_blk)
                        r_m = 1.0 if (mv > 0.5) == target[t] else -1.0
                        if r_m > r_w:
                            better_blocks += 1

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    res = {'task': 'self_tuned_gate', 'env': env, 'readout': arm,
           'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T),
           'runtime_s': time.time() - t0,
           'mean_acc_all': float(np.nanmean(acc_run)),
           'n_snaps': n_snaps if arm != 'rls_single' else -1,
           'sel_frac': (n_sel_blocks / n_blocks) if arm != 'rls_single'
                       else float('nan')}
    if arm == 'mem_vlearn':
        res['v_snap_final'] = v_snap
        if v_snap_traj:
            # quartile means of the trajectory
            bs = np.array([x[0] for x in v_snap_traj])
            vs = np.array([x[1] for x in v_snap_traj])
            res['v_snap_last4'] = float(np.mean(vs[-4:])) if len(vs) >= 4 \
                else float(np.mean(vs))
            res['v_snap_q1'] = float(np.mean(vs[:max(1, len(vs)//4)]))
        else:
            res['v_snap_final'] = v_snap
            res['v_snap_last4'] = v_snap
            res['v_snap_q1'] = v_snap
    else:
        res['v_snap_final'] = float('nan')
        res['v_snap_last4'] = float('nan')
        res['v_snap_q1'] = float('nan')
    if env == 'dyn':
        seg_acc = []
        for s in range(N_SEGMENTS):
            seg_acc.append(acc_median_in(acc_run, s * SEG_BLOCKS,
                                         (s + 1) * SEG_BLOCKS))
        for s in range(N_SEGMENTS):
            res[f'seg{s + 1}_acc'] = seg_acc[s]
    return res


# ========================== Aggregation ==========================

def agg_list(rs, field):
    vals = np.array([r[field] for r in rs], dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if vals.size else float('nan')


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['env'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        env, read = key
        entry = {'env': env, 'readout': read, 'n_runs': len(rs)}
        fields = (['mean_acc_all', 'n_snaps', 'sel_frac'] +
                  ['v_snap_final', 'v_snap_last4', 'v_snap_q1'])
        if env == 'dyn':
            fields += ['seg%d_acc' % s for s in range(1, 5)]
        for f in fields:
            entry[f + '_mean'] = agg_list(rs, f)
        vf = [r['v_snap_final'] for r in rs if np.isfinite(r['v_snap_final'])]
        entry['v_snap_final_std'] = float(np.nanstd(vf)) if vf \
            else float('nan')
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 118)
    print("S59 SELF-TUNED SNAPSHOT GATE (G1: learn WHEN to keep experience)")
    print("=" * 118)
    for env in ENVS:
        print(f"\n--- env {env} ---")
        print(f"  {'readout':<12} | {'mean_acc':>8} | {'n_snaps':>7} | "
              f"{'sel_frac':>8} | {'v_final':>7} | {'v_last4':>7} | "
              f"{'v_q1':>6}" + (" | s1/s2/s3/s4" if env == 'dyn' else ""))
        for a in [x for x in agg if x['env'] == env]:
            row = (f"  {a['readout']:<12} | {a['mean_acc_all_mean']:>8.3f} | "
                   f"{a['n_snaps_mean']:>7.1f} | {a['sel_frac_mean']:>8.3f} | "
                   f"{a['v_snap_final_mean']:>7.3f} | "
                   f"{a['v_snap_last4_mean']:>7.3f} | "
                   f"{a['v_snap_q1_mean']:>6.3f}")
            if env == 'dyn':
                row += " | " + "/".join(
                    f"{a['seg%d_acc_mean' % s]:.3f}" for s in range(1, 5))
            print(row)


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S59 self-tuned gate "
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
    for env in ENVS:
        for arm in ARMS:
            for s in range(n_seeds):
                all_args.append((arm, env, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (envs={ENVS}, arms={ARMS}, seeds={n_seeds})")

    results = []
    n_proc = min(cpu_count(), 8, max(1, n_runs))
    with Pool(n_proc) as pool:
        done = 0
        for res in pool.imap_unordered(run_single, all_args, chunksize=2):
            results.append(res)
            done += 1
            if done % max(1, n_runs // 10) == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress "
                      f"{done}/{n_runs}", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'env', 'readout', 'seed_idx', 'n_units',
                  't_total', 'runtime_s', 'mean_acc_all', 'n_snaps',
                  'sel_frac', 'v_snap_final', 'v_snap_last4', 'v_snap_q1',
                  'seg1_acc', 'seg2_acc', 'seg3_acc', 'seg4_acc']
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
        'snap_hold': SNAP_HOLD, 'pop_k': POP_K, 'agree_w': AGREE_W,
        'agree_thresh': AGREE_THRESH,
        'v_snap_init': V_SNAP_INIT, 'v_snap_lo': V_SNAP_LO,
        'v_snap_hi': V_SNAP_HI, 'v_gamma': V_GAMMA, 'w_eval': W_EVAL,
        'sel_sust': SEL_SUST,
        'segments': SEGMENTS, 'seg_blocks': SEG_BLOCKS,
        'static_blocks': STATIC_BLOCKS, 'static_boundary': STATIC_BOUNDARY,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    payload = _nan_to_null({'params': params, 'aggregates': agg,
                            'null_reason': ('non-finite aggregate values are '
                                            'written as null (undefined, e.g. an '
                                            'arm that never sustained a snapshot '
                                            'selection, so v_snap_final, '
                                            'v_snap_last4, v_snap_q1 and '
                                            'sel_frac have no measurement)')})
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total "
          f"{time.time() - t_start:.1f}s")


def _nan_to_null(obj):
    """Recursively map non-finite floats to None (strict-JSON safe).

    Bug fix 2026-09-09 (dedup P1-90): json.dump wrote bare NaN literals for
    undefined aggregates (v_snap_final / v_snap_last4 / v_snap_q1 / sel_frac
    of an arm that never sustained a snapshot selection), which is not valid
    JSON and is rejected by strict parsers.
    """
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_null(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
