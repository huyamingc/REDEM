#!/usr/bin/env python3
"""
Joint self-evolution: do hypothesis MEMORY and AUTONOMOUS BASIS EXPANSION
compose under dynamic regime revisits + representation pressure? (REDEM S56)
=============================================================================
Type:           PAPER
Paper Section:  S52/S53/S54 composition (the full self-evolution loop)
Experiment:     s52 showed hypothesis memory + reward-rate selection beats
                continuous re-adaptation under A-B-A-B regime revisits (on
                the coupled substrate where both regimes are decodable).
                s53 showed a well-conditioned capacity sense correctly IDLES
                on the coupled substrate (representation sufficient) and
                triggers a basis expansion (linear -> quadratic) on the
                uncoupled (parallel) substrate, partially recovering an
                otherwise undecodable circle. s56 runs the SAME A-B-A-B
                revisit structure on BOTH substrates with a 2x2 of
                mechanisms: memory (s52 policy v4) x autonomous expansion
                (s53 sense). The two mechanisms live on orthogonal axes --
                memory preserves hypotheses for REVISITED regimes, expansion
                repairs REPRESENTATION insufficiency -- and s56 asks whether
                they compose into one self-evolving system, and whether an
                upgrade invalidates remembered hypotheses.

Environment: 4 x 500-block segments [linear, circle, linear, circle]
             (A-B-A-B) on BOTH substrates; K=20, T=40000. The circle is
             decodable on random_graph_k25 (representation sufficient) and
             NOT on parallel (representation insufficient; s53/s55).

Arms (block-vote scoring, protocol decision):
  rls_single     : single RLS on linear features (baseline: no memory, no
                   expansion).
  rls_quad_start : single RLS on quadratic features from block 0 (oracle
                   upper bound; no sense needed).
  pop_mem        : population memory (s52 v4), linear features only
                   (memory axis alone).
  rls_auto       : single RLS + capacity sense; expands to quadratic when
                   capacity stays below threshold (expansion axis alone,
                   s53).
  pop_auto       : population memory + capacity sense (THE JOINT SYSTEM).

Design decisions:
  (a) Memory across upgrade: the quadratic basis CONTAINS the linear one
      (F_quad = [F_lin, F_lin^2]), so a remembered linear-basis hypothesis
      remains exactly valid after expansion (padded with zeros). pop_auto
      therefore KEEPS its memory across the upgrade: representation repair
      costs nothing for prior hypotheses. Only the worker RLS is reset (its
      weights must be re-learned in the larger space) and the sense buffers
      are cleared.
  (b) Capacity sense v2 (RELATIVE, dual-basis): the s53 absolute sense
      (probe < 0.55) is unreliable in a DYNAMIC environment -- a 240-block
      window straddling a regime switch reads ~0.5 on BOTH substrates even
      when both tasks are individually decodable (boundary-mixing artifact),
      so it false-triggers on the coupled substrate and its readings hover
      at threshold on the insufficient one (near-chance trigger timing).
      The v2 sense evaluates the CANDIDATE basis against the CURRENT basis:
      at each refit it probes BOTH linear and quadratic block-mean features
      against the system's own (r,vote)-reconstructed labels and expands
      iff quad - lin >= REL_MARGIN (the upgrade reads better) AND quad >=
      QUAD_FLOOR (the upgrade is usable), sustained over REL_HOLD refits.
      Boundary mixing moves both readings together (delta ~ 0) so the
      relative test is immune to it; genuine insufficiency (circle on
      parallel) shows a sustained +0.10..+0.21 delta.

Predictions:
  P1 (no false trigger): on random_graph delta ~ 0 at every refit (both
      bases decode both regimes; boundary dips cancel) -> auto arms never
      expand; rls_auto ~ rls_single, pop_auto ~ pop_mem (expansion idles
      at zero cost in the dynamic setting).
  P2 (correct trigger): on parallel delta >= +0.10 sustained inside the
      first circle segment -> auto arms expand inside segment 2 and recover
      the circle, while rls_single/pop_mem stay at chance on both circle
      segments.
  P3 (memory domain, s52 replication): on random_graph pop arms beat rls
      arms on the REVISIT segments (seg3 linear, seg4 circle).
  P4 (composition): on parallel, pop_auto >= max(pop_mem, rls_auto) mean:
      expansion recovers the circle while memory preserves the linear
      solution across the upgrade (seg3 linear revisit is immediate for
      pop_auto because the frozen linear slot survives, re-learned by
      rls_auto).
  P5 (oracle gap): rls_quad_start is the ceiling; auto arms approach it
      inside segment 2 (partial recovery, s53: 0.53 -> 0.60 mean vs oracle).


Output files:
  data/s56_joint_self_evolution_v1.csv   (one row per run)
  data/s56_joint_self_evolution_v1.json  (params + per-cell aggregates)

Usage: python s56_joint_self_evolution.py [--quick]
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
from streaming_tasks import gen_nonlinear2d, DB_K_PULSES

# ==================== Fixed parameters (S3/S39-S55-consistent) ====================
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
SNAP_EMA_MIN = 0.55
SNAP_HOLD = 20
POP_K = 5               # total specialists (1 worker + 4 memory)
AGREE_W = 50
AGREE_THRESH = 0.85

# s56 relative (dual-basis) capacity sense: expands iff the candidate
# quadratic basis reads better than the current linear basis on the
# system's own (r,vote)-reconstructed labels, sustained over a hold.
WIN_BLOCKS = 240
REFIT_EVERY = 20
CAP_N_PC = 50
QUAD_FLOOR = 0.60       # candidate basis must be usable, not just less-bad
REL_MARGIN = 0.10       # quad_cap - lin_cap must exceed this
REL_HOLD = 3            # consecutive refits above the margin before trigger
WARMUP_BLOCKS = 300

N_SEGMENTS = 4          # A-B-A-B
SEG_BLOCKS = 500        # blocks per segment
SEGMENTS = ['linear', 'circle', 'linear', 'circle']
SUBSTRATES = ['random_graph_k25', 'parallel']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's56_joint_self_evolution_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's56_joint_self_evolution_v1.json')

CURVE_WINDOW = 200

ARMS = ['rls_single', 'rls_quad_start', 'pop_mem', 'rls_auto', 'pop_auto']


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
    return (np.concatenate(dt_parts),
            np.concatenate(tgt_parts).astype(np.int64))


def build_features(states, T, quadratic):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F = np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])
    if quadratic:
        d = F.shape[1]
        F = np.hstack([F, F[:, :d - 1] ** 2])
    return F


def capacity_pca50(Xw, yw):
    """Well-conditioned capacity probe: held-out ridge accuracy on the
    top-50 PCA components of a block-mean feature window (s53)."""
    Xa = np.asarray(Xw, dtype=np.float64)
    ya = np.asarray(yw, dtype=np.float64)
    n = Xa.shape[0]
    n_tr = max(20, n // 2)
    mu = Xa[:n_tr].mean(axis=0)
    sd = Xa[:n_tr].std(axis=0) + 1e-9
    Xs = (Xa - mu) / sd
    U, S, Vt = np.linalg.svd(Xs[:n_tr], full_matrices=False)
    k = min(CAP_N_PC, Vt.shape[0])
    Xp = Xs @ Vt[:k].T
    out = ridge_fit(Xp[:n_tr], ya[:n_tr], Xp[n_tr:], ya[n_tr:],
                    ridge_lambda=1.0)
    pred = out['pred_te']
    return float(np.mean((pred > 0.5).astype(np.float64) == ya[n_tr:]))


# ========================== Single run ==========================

def run_single(args):
    """(substrate, arm, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, seed_idx = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq = gen_alternating(seed_idx)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F_lin = build_features(states, T, quadratic=False)
    F_quad = build_features(states, T, quadratic=True)

    quad_start = (arm == 'rls_quad_start')
    F = F_quad if quad_start else F_lin
    d = F.shape[1]

    preds = np.empty(T)
    switch_block = -1
    detect_cap = float('nan')
    detect_delta = float('nan')
    n_expansions = 0
    n_snaps = 0
    cap_readings = []          # (block_idx, lin_cap, quad_cap, delta)

    if arm in ('rls_single', 'rls_quad_start', 'rls_auto'):
        auto = (arm == 'rls_auto')
        rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                        init_cov=RLS_INIT_COV)
        Xw = []
        Xwq = []
        yw = []
        rel_hold = 0
        for t in range(T):
            preds[t] = float(rls.predict(F[t])[0])
            if (t + 1) % DB_K_PULSES == 0:
                b = (t + 1) // DB_K_PULSES - 1
                seg = F[t - DB_K_PULSES + 1:t + 1]
                x_blk = seg.mean(axis=0)
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                rls.update(x_blk, np.array([y_derived]))
                preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
                if auto:
                    x_blk_q = F_quad[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
                    Xw.append(x_blk)
                    Xwq.append(x_blk_q)
                    yw.append(y_derived)
                    if len(Xw) > WIN_BLOCKS:
                        Xw.pop(0)
                        Xwq.pop(0)
                        yw.pop(0)
                    if (len(Xw) >= 40 and b >= WARMUP_BLOCKS
                            and (b + 1) % REFIT_EVERY == 0
                            and switch_block < 0):
                        cl = capacity_pca50(Xw, yw)
                        cq = capacity_pca50(Xwq, yw)
                        delta = cq - cl
                        cap_readings.append((b, cl, cq, delta))
                        if cq >= QUAD_FLOOR and delta >= REL_MARGIN:
                            rel_hold += 1
                            if rel_hold >= REL_HOLD:
                                switch_block = b
                                detect_cap = float(cl)
                                detect_delta = float(delta)
                                F = F_quad
                                d = F.shape[1]
                                rls = OnlineRLS(d, 1,
                                                forgetting=RLS_FORGETTING,
                                                init_cov=RLS_INIT_COV)
                                Xw = []
                                Xwq = []
                                yw = []
                                rel_hold = 0
                                n_expansions += 1
                        else:
                            rel_hold = 0
    else:  # pop_mem / pop_auto
        auto = (arm == 'pop_auto')
        worker = OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                           init_cov=RLS_INIT_COV)
        mem_w = []                 # frozen weight snapshots (d,) each
        mem_ema = []               # per-snapshot reward-rate EMA
        peak_ema = []              # per-snapshot PEAK reward-rate EMA
        r_ema_w = 0.0
        hold_cnt = 0
        active_kind = 0            # 0 = worker, 1.. = memory slot idx
        active_idx = 0
        recent_blk = []            # last AGREE_W block-mean features
        Xw = []
        Xwq = []
        yw = []
        rel_hold = 0
        for t in range(T):
            if active_kind == 0:
                preds[t] = float(worker.predict(F[t])[0])
            else:
                preds[t] = float(mem_w[active_idx] @ F[t])
            if (t + 1) % DB_K_PULSES == 0:
                b = (t + 1) // DB_K_PULSES - 1
                seg = F[t - DB_K_PULSES + 1:t + 1]
                x_blk = seg.mean(axis=0)
                recent_blk.append(x_blk)
                if len(recent_blk) > AGREE_W:
                    recent_blk.pop(0)
                blk = preds[t - DB_K_PULSES + 1:t + 1]
                vote = float(np.mean(blk) > 0.5)
                r = 1.0 if vote == target[t] else -1.0
                y_derived = vote if r > 0.0 else (1.0 - vote)
                # worker reward-rate EMA (worker evaluated on current block)
                wv = float(worker.predict(x_blk)[0])
                r_w = 1.0 if (wv > 0.5) == target[t] else -1.0
                r_ema_w = (1.0 - PROBE_EMA_ALPHA) * r_ema_w \
                    + PROBE_EMA_ALPHA * r_w
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
                # snapshot when worker is stable (s52 memory policy v4:
                # FUNCTIONAL identity via agreement on recent blocks, not
                # geometric; evict the slot with the lowest PEAK EMA).
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
                # capacity sense (pop_auto only). RELATIVE dual-basis probe:
                # both bases judged against the system's own
                # (r,vote)-reconstructed labels (exact regardless of the
                # active predictor, s46 identity). Boundary mixing moves
                # both readings together (delta ~ 0), so the relative test
                # is immune to the v1 absolute sense's false triggers.
                if auto:
                    x_blk_q = F_quad[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
                    Xw.append(x_blk)
                    Xwq.append(x_blk_q)
                    yw.append(y_derived)
                    if len(Xw) > WIN_BLOCKS:
                        Xw.pop(0)
                        Xwq.pop(0)
                        yw.pop(0)
                    if (len(Xw) >= 40 and b >= WARMUP_BLOCKS
                            and (b + 1) % REFIT_EVERY == 0
                            and switch_block < 0):
                        cl = capacity_pca50(Xw, yw)
                        cq = capacity_pca50(Xwq, yw)
                        delta = cq - cl
                        cap_readings.append((b, cl, cq, delta))
                        if cq >= QUAD_FLOOR and delta >= REL_MARGIN:
                            rel_hold += 1
                            if rel_hold >= REL_HOLD:
                                switch_block = b
                                detect_cap = float(cl)
                                detect_delta = float(delta)
                                # expand basis. The quadratic basis CONTAINS
                                # the linear one, so frozen memory snapshots
                                # stay exactly valid (pad with zeros) -- the
                                # upgrade costs nothing for prior hypotheses.
                                # Only the worker is reset (re-learned in the
                                # larger space) and the probe buffers clear.
                                d_new = F_quad.shape[1]
                                pad = d_new - F.shape[1]
                                for i in range(len(mem_w)):
                                    mem_w[i] = np.concatenate(
                                        [mem_w[i], np.zeros(pad)])
                                F = F_quad
                                worker = OnlineRLS(d_new, 1,
                                                   forgetting=RLS_FORGETTING,
                                                   init_cov=RLS_INIT_COV)
                                r_ema_w = 0.0
                                hold_cnt = 0
                                recent_blk = []
                                Xw = []
                                Xwq = []
                                yw = []
                                rel_hold = 0
                                n_expansions += 1
                                # recompute the current block-mean feature in
                                # the NEW basis so the selection step below
                                # stays dimension-consistent (memory slots
                                # were just padded to d_new).
                                seg = F[t - DB_K_PULSES + 1:t + 1]
                                x_blk = seg.mean(axis=0)
                        else:
                            rel_hold = 0
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

    def _cap_in(b_lo, b_hi, idx):
        vals = [c[idx] for c in cap_readings if b_lo <= c[0] < b_hi]
        return float(np.mean(vals)) if vals else float('nan')

    res = {'task': 'joint_self_evo', 'substrate': topo_name,
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'mean_acc_all': float(np.nanmean(acc_run)),
           'seg1_acc': seg_acc[0], 'seg2_acc': seg_acc[1],
           'seg3_acc': seg_acc[2], 'seg4_acc': seg_acc[3],
           'seg3_early': acc_median_in(acc_run, 2 * SEG_BLOCKS,
                                       2 * SEG_BLOCKS + 250),
           'seg4_early': acc_median_in(acc_run, 3 * SEG_BLOCKS,
                                       3 * SEG_BLOCKS + 250),
           'switch_block': switch_block, 'detect_cap': detect_cap,
           'detect_delta': detect_delta,
           'n_expansions': n_expansions, 'n_snaps': n_snaps,
           'seg1_cap': _cap_in(0, SEG_BLOCKS, 1),
           'seg2_cap': _cap_in(SEG_BLOCKS, 2 * SEG_BLOCKS, 1),
           'seg1_delta': _cap_in(0, SEG_BLOCKS, 3),
           'seg2_delta': _cap_in(SEG_BLOCKS, 2 * SEG_BLOCKS, 3),
           'runtime_s': time.time() - t0}
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['substrate'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        topo, read = key
        entry = {'substrate': topo, 'readout': read, 'n_runs': len(rs)}
        for f in ['mean_acc_all', 'seg1_acc', 'seg2_acc', 'seg3_acc',
                  'seg4_acc', 'seg3_early', 'seg4_early']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        sw = np.array([r['switch_block'] for r in rs], dtype=float)
        entry['switch_block_mean'] = float(np.nanmean(sw))
        entry['switch_block_min'] = float(np.nanmin(sw))
        entry['switch_block_max'] = float(np.nanmax(sw))
        entry['n_detected'] = int(np.sum(sw >= 0))
        entry['detect_cap_mean'] = float(np.nanmean(
            np.array([r['detect_cap'] for r in rs], dtype=float)))
        entry['n_expansions_mean'] = float(np.nanmean(
            np.array([r['n_expansions'] for r in rs], dtype=float)))
        entry['n_snaps_mean'] = float(np.nanmean(
            np.array([r['n_snaps'] for r in rs], dtype=float)))
        for f in ['seg1_cap', 'seg2_cap', 'seg1_delta', 'seg2_delta']:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
        entry['detect_delta_mean'] = float(np.nanmean(
            np.array([r['detect_delta'] for r in rs], dtype=float)))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 128)
    print("S56 JOINT SELF-EVOLUTION: MEMORY x AUTONOMOUS BASIS (A-B-A-B)")
    print("=" * 128)
    print(f"\n  segments: {SEGMENTS} (A-B-A-B, 500 blocks each)")
    for topo in SUBSTRATES:
        print(f"\n--- {topo} ---")
        print(f"  {'readout':<14} | {'seg1 lin':>7} | {'seg2 cir':>7} | "
              f"{'seg3 lin':>7} | {'seg4 cir':>7} | {'mean':>6} | "
              f"{'s3_early':>8} | {'s4_early':>8} | {'switch':>7} | "
              f"{'d2':>6} | {'det':>3} | {'snaps':>5}")
        for a in [x for x in agg if x['substrate'] == topo]:
            sw = a['switch_block_mean'] if a['n_detected'] else float('nan')
            d2 = a['seg2_delta_mean'] if np.isfinite(a['seg2_delta_mean']) \
                else float('nan')
            print(f"  {a['readout']:<14} | {a['seg1_acc_mean']:>7.3f} | "
                  f"{a['seg2_acc_mean']:>7.3f} | {a['seg3_acc_mean']:>7.3f} | "
                  f"{a['seg4_acc_mean']:>7.3f} | {a['mean_acc_all_mean']:>6.3f} | "
                  f"{a['seg3_early_mean']:>8.3f} | "
                  f"{a['seg4_early_mean']:>8.3f} | {sw:>7.1f} | "
                  f"{d2:>6.3f} | "
                  f"{a['n_detected']:>3} | {a['n_snaps_mean']:>5.1f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S56 joint self-evolution "
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
    for topo in SUBSTRATES:
        for arm in ARMS:
            for s in range(n_seeds):
                all_args.append((topo, arm, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (substrates={len(SUBSTRATES)}, "
          f"arms={len(ARMS)}, seeds={n_seeds})")

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
                  'seg1_acc', 'seg2_acc', 'seg3_acc', 'seg4_acc',
                  'seg3_early', 'seg4_early', 'switch_block', 'detect_cap',
                  'detect_delta', 'n_expansions', 'n_snaps', 'seg1_cap',
                  'seg2_cap', 'seg1_delta', 'seg2_delta']
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
        'win_blocks': WIN_BLOCKS, 'refit_every': REFIT_EVERY,
        'cap_n_pc': CAP_N_PC, 'quad_floor': QUAD_FLOOR,
        'rel_margin': REL_MARGIN, 'rel_hold': REL_HOLD,
        'warmup_blocks': WARMUP_BLOCKS,
        'n_segments': N_SEGMENTS, 'seg_blocks': SEG_BLOCKS,
        'segments': SEGMENTS, 'substrates': SUBSTRATES,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    payload = _nan_to_null({'params': params, 'aggregates': agg,
                            'null_reason': ('non-finite aggregate values are '
                                            'written as null (undefined, e.g. a '
                                            'cell where no expansion was '
                                            'triggered, so detect_cap, seg1_cap, '
                                            'seg2_cap, seg1_delta, seg2_delta and '
                                            'detect_delta have no measurement)')})
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


def _nan_to_null(obj):
    """Recursively map non-finite floats to None (strict-JSON safe).

    Bug fix 2026-09-09 (dedup P1-90): json.dump wrote bare NaN literals for
    undefined aggregates (detect_cap / seg1_cap / seg2_cap / seg1_delta /
    seg2_delta / detect_delta of a cell where no expansion was triggered),
    which is not valid JSON and is rejected by strict parsers.
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
