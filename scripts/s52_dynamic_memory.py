#!/usr/bin/env python3
"""
L4 applicability: does hypothesis MEMORY + reward-rate selection beat
continuous re-adaptation under dynamic regime REVISITS? (REDEM S52)
=============================================================================
Type:           PAPER
Paper Section:  S51 follow-up (L4 selection under dynamic multi-regime
                stress)
Experiment:     s51 showed population selection is redundant on a STATIC
                capacity wall (circle): the wall is fixed, one enriched
                readout reaches it, and a population of first-order
                specialists cannot rescue an optimizer failure. s52 tests
                the predicted L4 value domain: DYNAMIC multi-regime stress
                with REGIME REVISITS. A single adaptive readout must re-learn
                a returning regime from (nearly) scratch (RLS forgetting
                erases the old solution over a long absence). A population
                with hypothesis MEMORY -- frozen snapshots of past solutions,
                selected by per-specialist reward-rate EMA -- should
                reactivate the returning solution immediately.

Environment: 4 x 500-block segments on the 2D substrate (random_graph_k25,
the substrate where the circle is decodable): linear -> circle -> linear ->
circle (A-B-A-B revisit). Quadratic features (the s51 basis) so both regimes
are within the hypothesis class. K=20, T=40000.

Arms (block-vote scoring, protocol decision):
  rls_single  : one block-level RLS on quadratic features, continuous
                adaptation (s51 winner; baseline).
  pop_mem     : worker RLS + up to 4 FROZEN memory snapshots (taken when the
                worker is stable); active = argmax r_ema across all; worker
                learns; memory slots reactivate the returning regime.
  pop_plain   : K=5 RLS specialists, ALL continuously adapting, active =
                argmax r_ema (selection WITHOUT memory -- control: does
                diversity alone help?). Each specialist starts from its own
                RandomState-drawn weight vector (std = POP_PLAIN_INIT_SCALE
                / sqrt(d)); before 2026-09-09 this arm was degenerate (all
                K specialists started at W=0 and stayed bit-identical, so
                the control never tested diversity -- dedup P1-13).

Predictions:
  P1 (baseline): rls_single dips after each switch and re-learns; its second
      circle segment recovers from scratch (no memory of the first).
  P2 (memory): pop_mem recovers the RETURNING regimes faster than rls_single
      (seg3 linear / seg4 circle higher mean), because the frozen snapshot
      is immediately correct -- mean_acc_all(pop_mem) > mean_acc_all(rls).
  P3 (no-memory control): pop_plain ~ rls_single -- homogeneous specialists
      all re-learn the same way, selection without memory adds nothing.

Memory-policy development notes (v1->v4, each a falsifiable mechanism fix):
  v1 LRU-by-current-EMA: evicts the revisiting hypothesis when its EMA
      decays during the other regime (circle memory lost during linear) ->
      no seg4 gain.
  v2 refuse-to-evict-distinct: cosine identity in the 513-dim quadratic
      space is ~0.1-0.2 even for the SAME boundary, so every snapshot looks
      "new" and the circle never enters the full linear-filled memory.
  v3 evict-lowest-PEAK-EMA: still loses the circle because its peak (0.68)
      < linear's peak (0.9); peak is the wrong eviction key.
  v4 FUNCTIONAL identity (agreement on recent blocks) + peak-EMA eviction:
      same boundary refreshes its slot; a genuinely new regime hypothesis is
      added and kept. This is the working policy and the mechanistic
      finding: hypothesis identity must be judged functionally (does it
      make the same predictions?), not geometrically (weight similarity).

Output files:
  data/s52_dynamic_memory_v1.csv    (one row per run)
  data/s52_dynamic_memory_v1.json   (params + per-cell aggregates + per-seed
                                     stats/paired tests + pop_plain sub-seeds)

Usage: python s52_dynamic_memory.py [--quick]
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
from streaming_tasks import gen_nonlinear2d, DB_K_PULSES

# ==================== Fixed parameters (S3/S39-S51-consistent) ====================
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

N_SEGMENTS = 4          # A-B-A-B
SEG_BLOCKS = 500        # blocks per segment
SEGMENTS = ['linear', 'circle', 'linear', 'circle']

PROBE_EMA_ALPHA = 0.02
SNAP_EMA_MIN = 0.55     # worker r_ema threshold for snapshot (must be low
                        # enough to capture the circle solution: its steady
                        # acc ~0.85 -> r_ema ~0.70)
SNAP_HOLD = 20          # consecutive blocks above threshold
POP_K = 5               # total specialists (1 worker + 4 memory)

# pop_plain diversity (bug fix 2026-09-09, dedup P1-13): OnlineRLS starts at
# W = 0 with no RNG, so all K control specialists were bit-identical and the
# "does diversity alone help?" control never tested diversity. Each specialist
# now receives its own RandomState-derived initial weight vector with
# std = POP_PLAIN_INIT_SCALE / sqrt(d); the per-specialist sub-seeds are
# written to the output JSON.
POP_PLAIN_INIT_SCALE = 1.0
POP_PLAIN_SEED_BASE = 90001
POP_PLAIN_SEED_STRIDE = 1009

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's52_dynamic_memory_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's52_dynamic_memory_v1.json')

CURVE_WINDOW = 200

ARMS = ['rls_single', 'pop_mem', 'pop_plain']


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


def gen_alternating(seed):
    """Concatenate A-B-A-B static-boundary segments on the 2D substrate."""
    rng = np.random.RandomState(seed)
    dt_parts = []
    tgt_parts = []
    for bd in SEGMENTS:
        s = int(rng.randint(0, 2**31 - 1))
        dt, tgt, _, _ = gen_nonlinear2d(seed=s, n_blocks=SEG_BLOCKS,
                                        k_pulses=DB_K_PULSES, boundary=bd)
        dt_parts.append(dt)
        tgt_parts.append(tgt)
    return (np.concatenate(dt_parts), np.concatenate(tgt_parts).astype(np.int64))


def build_features(states, T):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F = np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])
    d = F.shape[1]
    return np.hstack([F, F[:, :d - 1] ** 2])   # quadratic basis (s51)


# ========================== Single run ==========================

def run_single(args):
    """(arm, seed_idx) -> metrics dict. Substrate fixed: random_graph_k25."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    arm, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate('random_graph_k25')

    dt_seq, target_seq = gen_alternating(seed_idx)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F = build_features(states, T)
    d = F.shape[1]

    preds = np.empty(T)
    spec_seeds = None          # pop_plain per-specialist init sub-seeds
    pop_plain_init_std = float('nan')

    if arm == 'rls_single':
        rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV)
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
    elif arm == 'pop_plain':
        K = POP_K
        rls_list = [OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                              init_cov=RLS_INIT_COV) for _ in range(K)]
        # per-specialist diversity (see POP_PLAIN_* above; dedup P1-13)
        spec_seeds = [int(seed_idx) * POP_PLAIN_SEED_STRIDE
                      + POP_PLAIN_SEED_BASE + k for k in range(K)]
        init_std = POP_PLAIN_INIT_SCALE / np.sqrt(float(d))
        pop_plain_init_std = init_std
        for k in range(K):
            rng_k = np.random.RandomState(spec_seeds[k])
            rls_list[k].W[:, 0] = rng_k.normal(0.0, init_std, d)
        r_ema = np.zeros(K)
        active = 0
        for t in range(T):
            preds[t] = float(rls_list[active].predict(F[t])[0])
            if (t + 1) % DB_K_PULSES == 0:
                seg = F[t - DB_K_PULSES + 1:t + 1]
                blk_o = np.array([float(rls_list[k].predict(x)[0])
                                  for k in range(K) for x in [seg.mean(axis=0)]])
                r_k = np.where((blk_o > 0.5) == target[t], 1.0, -1.0)
                r_ema = (1.0 - PROBE_EMA_ALPHA) * r_ema + PROBE_EMA_ALPHA * r_k
                for k in range(K):
                    vk = float(blk_o[k] > 0.5)
                    rk = 1.0 if vk == target[t] else -1.0
                    y_derived = vk if rk > 0.0 else (1.0 - vk)
                    rls_list[k].update(seg.mean(axis=0), np.array([y_derived]))
                active = int(np.argmax(r_ema))
                preds[t - DB_K_PULSES + 1:t + 1] = blk_o[active]
    else:  # pop_mem
        worker = OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                           init_cov=RLS_INIT_COV)
        mem_w = []                 # frozen weight snapshots (d,) each
        mem_ema = []               # per-snapshot reward-rate EMA
        peak_ema = []              # per-snapshot PEAK reward-rate EMA
        r_ema_w = 0.0
        hold_cnt = 0
        active_kind = 0            # 0 = worker, 1.. = memory slot idx
        active_idx = 0
        n_snaps = 0
        recent_blk = []            # last AGREE_W block-mean features
        AGREE_W = 50
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
                # worker reward-rate EMA
                wv = float(worker.predict(x_blk)[0])
                r_w = 1.0 if (wv > 0.5) == target[t] else -1.0
                r_ema_w = (1.0 - PROBE_EMA_ALPHA) * r_ema_w + PROBE_EMA_ALPHA * r_w
                # memory slot EMAs (each evaluated on the current block)
                for i in range(len(mem_w)):
                    mv = float(mem_w[i] @ x_blk)
                    r_m = 1.0 if (mv > 0.5) == target[t] else -1.0
                    mem_ema[i] = (1.0 - PROBE_EMA_ALPHA) * mem_ema[i] \
                        + PROBE_EMA_ALPHA * r_m
                # worker learns (only the worker is adaptive)
                v_w = float(wv > 0.5)
                r_w2 = 1.0 if v_w == target[t] else -1.0
                y_d = v_w if r_w2 > 0.0 else (1.0 - v_w)
                worker.update(x_blk, np.array([y_d]))
                # snapshot when worker is stable. Memory policy (v4):
                # identity is FUNCTIONAL (agreement on the last AGREE_W
                # blocks), not geometric -- in the 513-dim quadratic space
                # weight vectors are near-orthogonal (cos ~0.1-0.2) even for
                # the SAME boundary, so a cosine dedup treats every snapshot
                # as new and evicts the revisiting-relevant hypothesis.
                #   * a candidate that AGREES > 0.85 with a slot on recent
                #     blocks IS that hypothesis -> refresh the slot;
                #   * otherwise it is a NEW hypothesis: add if room, else
                #     evict the slot with the lowest PEAK EMA (a hypothesis
                #     that was never good). Peak EMA is right: a solution
                #     for a regime that has rotated away has a decaying
                #     CURRENT EMA but is exactly the one to KEEP.
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
                        if agree and best_agr > 0.85:
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
                # select active: max over worker + memory EMAs
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

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    seg_acc = []
    for s in range(N_SEGMENTS):
        seg_acc.append(acc_median_in(acc_run, s * SEG_BLOCKS,
                                     (s + 1) * SEG_BLOCKS))
    res = {'task': 'alternating2d', 'substrate': 'random_graph_k25',
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'seg1_acc': seg_acc[0], 'seg2_acc': seg_acc[1],
           'seg3_acc': seg_acc[2], 'seg4_acc': seg_acc[3],
           'n_snaps': n_snaps if arm == 'pop_mem' else -1,
           'spec_seeds': spec_seeds,
           'pop_plain_init_std': pop_plain_init_std,
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(r['readout'], []).append(r)
    agg = []
    for read, rs in sorted(groups.items()):
        entry = {'readout': read, 'n_runs': len(rs)}
        for f in ['mean_acc_all', 'seg1_acc', 'seg2_acc', 'seg3_acc',
                  'seg4_acc']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        entry['n_snaps_mean'] = float(np.nanmean(
            np.array([r['n_snaps'] for r in rs], dtype=float)))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 118)
    print("S52 DYNAMIC REGIME REVISITS: HYPOTHESIS MEMORY + SELECTION")
    print("=" * 118)
    print(f"\n  segments: {SEGMENTS} (A-B-A-B, 500 blocks each)")
    print(f"  {'readout':<10} | {'seg1 lin':>8} | {'seg2 cir':>8} | "
          f"{'seg3 lin':>8} | {'seg4 cir':>8} | {'mean':>6} | {'snaps':>5}")
    for a in agg:
        print(f"  {a['readout']:<10} | {a['seg1_acc_mean']:>8.3f} | "
              f"{a['seg2_acc_mean']:>8.3f} | {a['seg3_acc_mean']:>8.3f} | "
              f"{a['seg4_acc_mean']:>8.3f} | {a['mean_acc_all_mean']:>6.3f} | "
              f"{a['n_snaps_mean']:>5.1f}")


def _t_crit_975(df):
    """Two-sided 95% Student-t critical value (small-df table, 1.96 fallback)."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
             18: 2.101, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045,
             30: 2.042}
    if df <= 0:
        return float('nan')
    if df in table:
        return table[df]
    return 1.96 if df > 30 else table[max(k for k in table if k <= df)]


def paired_stats(results, field='mean_acc_all'):
    """Per-seed statistics and a paired arm comparison (dedup P0-18).

    Every arm runs on the SAME 10 task realizations (seeds), so arms are
    paired by seed_idx and a paired t-test on the seed-wise differences is
    the appropriate comparison (the paper quoted mean +- SD with no test).
    Returns per-arm n / mean / SD / SEM / 95% CI (Student-t, n-1 dof) plus
    the paired difference vs rls_single; the paired seed list is included so
    the pairing is auditable.
    """
    by_arm = {}
    for r in results:
        by_arm.setdefault(r['readout'], {})[r['seed_idx']] = r[field]
    base = by_arm.get('rls_single', {})
    try:
        from scipy import stats as _sps
        has_scipy = True
    except ImportError:                     # optional; table CI still valid
        has_scipy = False
    out = {}
    for arm, d in sorted(by_arm.items()):
        vals = np.array([d[k] for k in sorted(d)], dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        m = float(np.mean(vals)) if n else float('nan')
        sd = float(np.std(vals, ddof=1)) if n > 1 else float('nan')
        sem = sd / np.sqrt(n) if n > 1 else float('nan')
        tcrit = _t_crit_975(n - 1)
        entry = {'n': n, 'mean': m, 'sd': sd, 'sem': sem,
                 'ci95_lo': m - tcrit * sem, 'ci95_hi': m + tcrit * sem}
        common = sorted(set(d) & set(base))
        if arm != 'rls_single' and len(common) > 1:
            a = np.array([d[k] for k in common], dtype=float)
            b = np.array([base[k] for k in common], dtype=float)
            diff = a - b
            sdd = float(np.std(diff, ddof=1))
            tval = (float(np.mean(diff) / (sdd / np.sqrt(diff.size)))
                    if sdd > 0 else float('nan'))
            pval = float('nan')
            if has_scipy and np.isfinite(tval):
                pval = float(2.0 * _sps.t.sf(abs(tval), diff.size - 1))
            entry['paired_vs_rls_single'] = {
                'n_pairs': int(diff.size), 'seeds': common,
                'mean_diff': float(np.mean(diff)), 'sd_diff': sdd,
                't': tval, 'p_two_sided':
                    pval if has_scipy else None,
                't_crit_975': _t_crit_975(diff.size - 1)}
        out[arm] = entry
    return {'field': field,
            'pairing': 'arms paired by seed_idx (same task realization)',
            'ci_method': 'two-sided 95% Student-t, n-1 dof',
            'test': 'paired t-test vs rls_single (p from scipy)'
                    if has_scipy else 'paired t vs rls_single (t only)'}


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S52 dynamic memory "
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
    for arm in ARMS:
        for s in range(n_seeds):
            all_args.append((arm, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (arms={len(ARMS)}, seeds={n_seeds})")

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
    fieldnames = ['task', 'substrate', 'readout', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'mean_acc_all',
                  'seg1_acc', 'seg2_acc', 'seg3_acc', 'seg4_acc', 'n_snaps']
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
        'n_segments': N_SEGMENTS, 'seg_blocks': SEG_BLOCKS,
        'segments': SEGMENTS,
        'probe_ema_alpha': PROBE_EMA_ALPHA, 'snap_ema_min': SNAP_EMA_MIN,
        'snap_hold': SNAP_HOLD, 'pop_k': POP_K,
        'n_seeds': n_seeds, 'quick': bool(quick),
        'pop_plain_init_scale': POP_PLAIN_INIT_SCALE,
        'pop_plain_seed_base': POP_PLAIN_SEED_BASE,
        'pop_plain_seed_stride': POP_PLAIN_SEED_STRIDE,
        'pop_plain_init_note': ('per-specialist RandomState initial weights, '
                                'std = scale/sqrt(d); fixes the bit-identical '
                                'specialists of the pre-2026-09-09 control '
                                'arm, dedup P1-13'),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    stats = paired_stats(results)
    spec = {r['seed_idx']: r['spec_seeds'] for r in results
            if r['readout'] == 'pop_plain' and r['spec_seeds']}
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg, 'stats': stats,
                   'pop_plain_specialist_seeds': spec}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
