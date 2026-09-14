#!/usr/bin/env python3
"""
Relative capacity sense under regime-switch stress (REDEM S58B)
=============================================================================
Type:           PAPER
Paper Section:  S56 follow-up (capacity-sense reliability boundary)
Experiment:     s56's relative (dual-basis) capacity sense triggered 9/10 on
                parallel (7/10 inside the first circle segment b~739-999,
                2/10 delayed to the seg-4 revisit b~1739/1799) and 1/10 never
                (seed 4, seg-2 delta ~ 0.041, no 3 consecutive refits >=
                REL_MARGIN). s58b STRESSES the sense: does trigger reliability
                survive shorter segments (denser regime switches) and a
                denser 3-regime sequence? It also separates the two trigger
                gates (QUAD_FLOOR=0.60 on the candidate basis vs REL_MARGIN=
                0.10 on the delta) per run to attribute every miss
                (floor-gate vs margin-gate vs hold).

Environment: static 2D boundary segments concatenated into a regime
             sequence, on BOTH substrates:
               ABAB   : [linear, circle, linear, circle]  x SEG_BLOCKS
               ABCABC : [linear, circle, ring, linear, circle, ring] x SEG
                        (ring = 2D annulus (0.30, 0.50), P=0.503, s57;
                        labels recomputed from returned u-vectors)
             SEG_BLOCKS in {500, 250, 125, 60} for ABAB and {500, 250}
             for ABCABC. K=20, T = n_segments * SEG * K.

Arms (block-vote scoring; only the capacity-sense users, s56 params fixed):
  rls_auto : single RLS + relative capacity sense (expansion axis).
  pop_auto : population memory (s52 v4) + relative capacity sense (joint).
  Both arms run the SAME relative capacity sense (WIN_BLOCKS/REFIT_EVERY/
  QUAD_FLOOR/REL_MARGIN/REL_HOLD fixed); they differ ONLY in the readout
  policy: rls_auto is one continuously adapting RLS, pop_auto adds the s52-v4
  frozen-snapshot memory with argmax-EMA selection. This branching is a bug
  fix (2026-09-09, dedup P0-27): the earlier version branched on a dead local
  `auto = True`, so the two arms were byte-identical pseudo-replicates.

Design decisions:
  (a) WARMUP scales with run length (WARMUP = max(60, int(0.15*total_blocks)))
      so the sense is active over the same FRACTION of the run at every
      segment length. SEG=500 ABAB reproduces s56 exactly (WARMUP=300) -- the
      baseline cell doubles as a reproduction check.
  (b) One-shot sense kept (s56 `switch_block < 0` gate): the multi-level
      question (R58C) is a separate experiment.
  (c) Per-run gate attribution: from the refit trace, count refits passing
      each gate (delta >= REL_MARGIN / cq >= QUAD_FLOOR / both) and classify
      a MISS as 'floor' (candidate never usable), 'margin' (delta never
      strong), 'hold' (both gates passed at least once but never REL_HOLD
      consecutive), or 'mixed'. This separates the seed-4 mechanism question.
  (d) C = 2D ring (annulus, TWO quadratic surfaces) instead of circle: a
      harder-but-still-decodable regime on random_graph (s57: probe 0.650)
      and insufficient on parallel, doubling the switch frequency per block.

Predictions:
  P1 (reliability cliff): trigger reliability
      (1 - false_rate_random) * (correct_rate_parallel) holds near 1 at
      SEG=500, degrades through SEG in {250, 125}, and collapses at SEG=60
      (a 240-block window spans >= 3 regimes there; the relative test's
      boundary-mixing immunity is expected to break when mixing is
      permanent).
  P2 (mechanism): at the cliff, random_graph false triggers appear first
      (delta noise grows with mixing) and are 'margin'- or 'hold'-gated,
      NOT 'floor' (both bases read well on random_graph); parallel misses
      are 'margin'-gated (delta weak) more often than 'floor'.
  P3 (seed 4): at SEG=500 the seed-4 miss attributes to the delta gate
      (seg-2 best delta < 0.10 sustained) rather than QUAD_FLOOR (cq in
      seg 2 reads >= 0.60).
  P4 (ABCABC): denser 3-regime sequences degrade reliability at a LARGER
      segment length than ABAB (more boundary crossings per window).

Output files:
  data/s58b_relative_sense_stress_v1.csv   (one row per run)
  data/s58b_relative_sense_stress_v1.json  (params + aggregates + reliability)
  data/s58b_trace_seed4.json               (--trace only: refit traces)

Usage: python s58b_relative_sense_stress.py [--quick] [--trace]
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

# ==================== Fixed parameters (S3/S39-S56-consistent) ====================
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
POP_K = 5
AGREE_W = 50
AGREE_THRESH = 0.85

# s56 relative (dual-basis) capacity sense -- params FIXED (R58B stress axis
# is the regime-switch structure, not the sense internals).
WIN_BLOCKS = 240
REFIT_EVERY = 20
CAP_N_PC = 50
QUAD_FLOOR = 0.60
REL_MARGIN = 0.10
REL_HOLD = 3
WARMUP_FRACTION = 0.15       # sense active from this fraction of the run
WARMUP_FLOOR = 60            # ... but never earlier than this many blocks

# s57 2D ring (annulus) boundary for the ABCABC sequence (P=0.503)
RING_IN = 0.30
RING_OUT = 0.50

# regime-sequence configs
SEG_LENS_ABAB = [500, 250, 125, 60]
SEG_LENS_ABCABC = [500, 250]
SEQ_ABAB = ['linear', 'circle', 'linear', 'circle']
SEQ_ABCABC = ['linear', 'circle', 'ring', 'linear', 'circle', 'ring']
SUBSTRATES = ['random_graph_k25', 'parallel']
ARMS = ['rls_auto', 'pop_auto']
MAX_SEG = 6                  # pad per-segment fields to this many columns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's58b_relative_sense_stress_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's58b_relative_sense_stress_v1.json')
TRACE_PATH = os.path.join(DATA_DIR, 's58b_trace_seed4.json')

CURVE_WINDOW = 200


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


def gen_alternating(seed, segments, seg_blocks):
    """Concatenate static-boundary segments on the 2D substrate. 'ring' is
    the 2D annulus of s57, relabelled from the returned u-vectors."""
    rng = np.random.RandomState(seed)
    dt_parts = []
    tgt_parts = []
    for bd in segments:
        s = int(rng.randint(0, 2**31 - 1))
        if bd == 'ring':
            dt, _, u0, u1 = gen_nonlinear2d(seed=s, n_blocks=seg_blocks,
                                            k_pulses=DB_K_PULSES,
                                            boundary='linear')
            d2 = (u0 - 0.5) ** 2 + (u1 - 0.5) ** 2
            lab = ((d2 > RING_IN ** 2) & (d2 < RING_OUT ** 2)).astype(np.int64)
            tgt = np.repeat(lab, DB_K_PULSES).astype(np.int64)
        else:
            dt, tgt, _, _ = gen_nonlinear2d(seed=s, n_blocks=seg_blocks,
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


def gate_attribution(cap_readings):
    """Classify a MISS by which trigger gate blocked it. Returns
    (blocker, best_delta, b_best, cq_best, n_above_margin, n_above_floor,
     n_above_both). 'hold' means both gates passed at least once but never
     REL_HOLD consecutive refits."""
    if not cap_readings:
        return ('no_refits', float('nan'), -1, float('nan'),
                0, 0, 0)
    deltas = np.array([c[3] for c in cap_readings], dtype=float)
    cqs = np.array([c[2] for c in cap_readings], dtype=float)
    best_i = int(np.argmax(deltas))
    best_delta = float(deltas[best_i])
    b_best = int(cap_readings[best_i][0])
    cq_best = float(cqs[best_i])
    n_above_margin = int(np.sum(deltas >= REL_MARGIN))
    n_above_floor = int(np.sum(cqs >= QUAD_FLOOR))
    n_above_both = int(np.sum((deltas >= REL_MARGIN) & (cqs >= QUAD_FLOOR)))
    if n_above_both > 0:
        blocker = 'hold'
    elif n_above_floor == 0:
        blocker = 'floor'
    elif n_above_margin == 0:
        blocker = 'margin'
    else:
        blocker = 'mixed'
    return (blocker, best_delta, b_best, cq_best,
            n_above_margin, n_above_floor, n_above_both)


# ========================== Single run ==========================

def run_single(args):
    """(substrate, arm, seg_len, seq_kind, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, seg_len, seq_kind, seed_idx = args
    t0 = time.time()

    segments = SEQ_ABAB if seq_kind == 'ABAB' else SEQ_ABCABC
    n_seg = len(segments)
    total_blocks = n_seg * seg_len
    warmup = max(WARMUP_FLOOR, int(WARMUP_FRACTION * total_blocks))

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq = gen_alternating(seed_idx, segments, seg_len)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F_lin = build_features(states, T, quadratic=False)
    F_quad = build_features(states, T, quadratic=True)
    F = F_lin
    d = F.shape[1]

    preds = np.empty(T)
    switch_block = -1
    detect_cap = float('nan')
    detect_cq = float('nan')
    detect_delta = float('nan')
    n_expansions = 0
    n_snaps = 0
    cap_readings = []          # (block_idx, lin_cap, quad_cap, delta)

    # ARM BRANCHING (bug fix 2026-09-09, F-P0-3 / dedup P0-27): both arms are
    # sense users, but ONLY pop_auto carries the s52-v4 population memory
    # (snapshot + argmax-EMA selection). rls_auto is a single continuously
    # adapting RLS + the same relative capacity sense. The previous version
    # branched on a dead local `auto = True`, so both arms were byte-identical
    # (240 CSV rows were 120 real runs).
    use_memory = (arm == 'pop_auto')
    worker = OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                       init_cov=RLS_INIT_COV)
    mem_w = []                 # frozen weight snapshots (d,) each
    mem_ema = []
    peak_ema = []
    r_ema_w = 0.0
    hold_cnt = 0
    active_kind = 0            # 0 = worker, 1.. = memory slot idx
    active_idx = 0
    recent_blk = []
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
            if use_memory and r_ema_w > SNAP_EMA_MIN:
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
            # relative dual-basis capacity sense (both arms)
            x_blk_q = F_quad[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
            Xw.append(x_blk)
            Xwq.append(x_blk_q)
            yw.append(y_derived)
            if len(Xw) > WIN_BLOCKS:
                Xw.pop(0)
                Xwq.pop(0)
                yw.pop(0)
            if (len(Xw) >= 40 and b >= warmup
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
                        detect_cq = float(cq)
                        detect_delta = float(delta)
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
    seg_acc = [acc_median_in(acc_run, s * seg_len,
                             (s + 1) * seg_len) for s in range(n_seg)]
    seg_acc += [float('nan')] * (MAX_SEG - n_seg)

    def _cap_in(b_lo, b_hi, idx):
        vals = [c[idx] for c in cap_readings if b_lo <= c[0] < b_hi]
        return float(np.mean(vals)) if vals else float('nan')

    seg_cap = [_cap_in(s * seg_len, (s + 1) * seg_len, 2)
               for s in range(n_seg)]
    seg_cap += [float('nan')] * (MAX_SEG - n_seg)
    seg_delta = [_cap_in(s * seg_len, (s + 1) * seg_len, 3)
                 for s in range(n_seg)]
    seg_delta += [float('nan')] * (MAX_SEG - n_seg)

    blocker, best_delta, b_best, cq_best, n_m, n_f, n_b = \
        gate_attribution(cap_readings)
    if switch_block >= 0:
        blocker = 'triggered'

    res = {'task': 'relative_sense_stress', 'substrate': topo_name,
           'readout': arm, 'seg_len': int(seg_len), 'seq_kind': seq_kind,
           'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T),
           'runtime_s': time.time() - t0,
           'mean_acc_all': float(np.nanmean(acc_run)),
           'switch_block': switch_block, 'detect_cap': detect_cap,
           'detect_cq': detect_cq, 'detect_delta': detect_delta,
           'n_expansions': n_expansions, 'n_snaps': n_snaps,
           'best_delta': best_delta, 'best_delta_b': b_best,
           'cq_at_best_delta': cq_best, 'n_above_margin': n_m,
           'n_above_floor': n_f, 'n_above_both': n_b,
           'gate_blocker': blocker,
           'cap_trace': cap_readings}
    for s in range(MAX_SEG):
        res[f'seg{s + 1}_acc'] = seg_acc[s]
    for s in range(MAX_SEG):
        res[f'seg{s + 1}_cap'] = seg_cap[s]
    for s in range(MAX_SEG):
        res[f'seg{s + 1}_delta'] = seg_delta[s]
    return res


# ========================== Aggregation ==========================

def agg_list(rs, field):
    vals = np.array([r[field] for r in rs], dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def aggregate(results):
    groups = {}
    for r in results:
        key = (r['substrate'], r['readout'], r['seg_len'], r['seq_kind'])
        groups.setdefault(key, []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        topo, read, seg_len, seq_kind = key
        entry = {'substrate': topo, 'readout': read, 'seg_len': seg_len,
                 'seq_kind': seq_kind, 'n_runs': len(rs)}
        for f in (['mean_acc_all'] +
                  [f'seg{s}_acc' for s in range(1, MAX_SEG + 1)]):
            m, sd = agg_list(rs, f)
            entry[f + '_mean'] = m
            entry[f + '_std'] = sd
        sw = np.array([r['switch_block'] for r in rs], dtype=float)
        entry['switch_block_mean'] = float(np.nanmean(sw))
        entry['switch_block_min'] = float(np.nanmin(sw))
        entry['switch_block_max'] = float(np.nanmax(sw))
        entry['n_detected'] = int(np.sum(sw >= 0))
        entry['detect_cap_mean'] = float(np.nanmean(
            np.array([r['detect_cap'] for r in rs], dtype=float)))
        entry['detect_cq_mean'] = float(np.nanmean(
            np.array([r['detect_cq'] for r in rs], dtype=float)))
        entry['detect_delta_mean'] = float(np.nanmean(
            np.array([r['detect_delta'] for r in rs], dtype=float)))
        entry['best_delta_mean'] = float(np.nanmean(
            np.array([r['best_delta'] for r in rs], dtype=float)))
        entry['n_expansions_mean'] = float(np.nanmean(
            np.array([r['n_expansions'] for r in rs], dtype=float)))
        for g in ['floor', 'margin', 'hold', 'mixed', 'no_refits',
                  'triggered']:
            entry['blk_' + g] = int(
                sum(1 for r in rs if r['gate_blocker'] == g))
        for f in (['seg1_cap', 'seg2_cap', 'seg1_delta', 'seg2_delta'] +
                  [f'seg{s}_cap' for s in range(1, MAX_SEG + 1)] +
                  [f'seg{s}_delta' for s in range(1, MAX_SEG + 1)]):
            if any(f in r for r in rs):
                m, _ = agg_list(rs, f)
                entry[f + '_mean'] = m
        agg.append(entry)
    return agg


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (k of n).

    Used for the trigger-rate factors (dedup P0-8: the reliability product
    had no uncertainty attached). Returns (lo, hi); (nan, nan) for n == 0.
    """
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability_table(agg, n_seeds):
    """(1 - false_rate_random) * (correct_rate_parallel) per config.

    GROUND TRUTH for the two factors (dedup P0-8): the substrate/regime design
    table fixes which substrate is under-represented -- circle/annulus are
    decodable on random_graph_k25 but NOT on parallel (s57/s58a capacity
    survey, an offline measurement independent of the online capacity probe).
    A trigger on parallel is therefore a TRUE POSITIVE ('correct trigger'),
    a trigger on random_graph_k25 a FALSE POSITIVE. The truth is thus supplied
    by the substrate table, not by the probe being evaluated (no circularity);
    the two factors and their denominators are reported separately, per arm
    and pooled, with Wilson 95% CIs. Reliability is reported as a product of
    two binomial rates measured on DISJOINT run sets (different substrates),
    so the product is valid without an independence assumption on a shared
    sample.
    """
    out = []
    for seq_kind in ['ABAB', 'ABCABC']:
        for seg_len in sorted({e['seg_len'] for e in agg
                               if e['seq_kind'] == seq_kind}):
            rnd = [e for e in agg if e['seq_kind'] == seq_kind
                   and e['seg_len'] == seg_len
                   and e['substrate'] == 'random_graph_k25']
            par = [e for e in agg if e['seq_kind'] == seq_kind
                   and e['seg_len'] == seg_len
                   and e['substrate'] == 'parallel']
            # aggregate both arms' n_detected
            false_n = sum(e['n_detected'] for e in rnd)
            false_tot = sum(e['n_runs'] for e in rnd)
            corr_n = sum(e['n_detected'] for e in par)
            corr_tot = sum(e['n_runs'] for e in par)
            false_rate = false_n / false_tot if false_tot else float('nan')
            corr_rate = corr_n / corr_tot if corr_tot else float('nan')
            rel = (1.0 - false_rate) * corr_rate
            f_lo, f_hi = wilson_ci(false_n, false_tot)
            c_lo, c_hi = wilson_ci(corr_n, corr_tot)
            entry = {'seq_kind': seq_kind, 'seg_len': seg_len,
                     'false_rate_random': false_rate,
                     'correct_rate_parallel': corr_rate,
                     'reliability': rel,
                     # sample units and denominators (dedup P0-8)
                     'unit': 'run (one seed x one arm x one substrate)',
                     'false_n_random': false_n, 'false_tot_random': false_tot,
                     'correct_n_parallel': corr_n,
                     'correct_tot_parallel': corr_tot,
                     'false_rate_ci95': [f_lo, f_hi],
                     'correct_rate_ci95': [c_lo, c_hi],
                     # CI on the product: independent disjoint binomials
                     'reliability_ci95_lo': (1.0 - f_hi) * c_lo,
                     'reliability_ci95_hi': (1.0 - f_lo) * c_hi,
                     'arms': {}}
            for arm in ARMS:
                ar = [e for e in rnd if e['readout'] == arm]
                ap = [e for e in par if e['readout'] == arm]
                fn = sum(e['n_detected'] for e in ar)
                ft = sum(e['n_runs'] for e in ar)
                cn = sum(e['n_detected'] for e in ap)
                ct = sum(e['n_runs'] for e in ap)
                entry['arms'][arm] = {
                    'false_n_random': fn, 'false_tot_random': ft,
                    'correct_n_parallel': cn, 'correct_tot_parallel': ct,
                    'false_rate_random': fn / ft if ft else float('nan'),
                    'correct_rate_parallel': cn / ct if ct else float('nan'),
                    'reliability': ((1.0 - fn / ft) * (cn / ct)
                                    if ft and ct else float('nan')),
                }
            out.append(entry)
    return out


def print_table(agg, rel, n_seeds):
    print("\n" + "=" * 132)
    print("S58B RELATIVE CAPACITY SENSE UNDER REGIME-SWITCH STRESS")
    print("=" * 132)
    for seq_kind in ['ABAB', 'ABCABC']:
        segs = SEQ_ABAB if seq_kind == 'ABAB' else SEQ_ABCABC
        print(f"\n=== seq {seq_kind}: {segs} ===")
        for topo in SUBSTRATES:
            print(f"\n--- {topo} ---")
            hdr = f"  {'arm':<9} {'seg':>4} | " + "".join(
                f"{'s%d' % (i + 1):>6}" for i in range(len(segs)))
            hdr += f" | {'mean':>6} | {'sw':>5} | {'det':>3} | {'d_best':>6}"
            print(hdr)
            for seg_len in sorted({e['seg_len'] for e in agg
                                   if e['seq_kind'] == seq_kind}):
                for arm in ARMS:
                    rs = [e for e in agg if e['seq_kind'] == seq_kind
                          and e['seg_len'] == seg_len
                          and e['substrate'] == topo and e['readout'] == arm]
                    if not rs:
                        continue
                    e = rs[0]
                    sw = e['switch_block_mean'] if e['n_detected'] \
                        else float('nan')
                    row = f"  {arm:<9} {seg_len:>4} | " + "".join(
                        f"{e['seg%d_acc_mean' % (i + 1)]:>6.3f}"
                        for i in range(len(segs)))
                    row += f" | {e['mean_acc_all_mean']:>6.3f} | " \
                           f"{sw:>5.0f} | {e['n_detected']:>3} | " \
                           f"{e['best_delta_mean']:>6.3f}"
                    print(row)
    print("\n--- trigger reliability (arms pooled; Wilson 95% CI) ---")
    print(f"  {'seq':>7} {'seg_len':>7} | {'false/rand':>14} | "
          f"{'correct/par':>14} | {'reliability':>18}")
    for r in rel:
        print(f"  {r['seq_kind']:>7} {r['seg_len']:>7} | "
              f"{r['false_rate_random']:>6.3f} "
              f"({r['false_n_random']:>2d}/{r['false_tot_random']:<2d}) | "
              f"{r['correct_rate_parallel']:>6.3f} "
              f"({r['correct_n_parallel']:>2d}/{r['correct_tot_parallel']:<2d})"
              f" | {r['reliability']:>6.3f} "
              f"[{r['reliability_ci95_lo']:.3f},"
              f"{r['reliability_ci95_hi']:.3f}]")
    print("\n--- trigger reliability per arm ---")
    for r in rel:
        for arm in ARMS:
            a = r['arms'][arm]
            print(f"  {r['seq_kind']:>7} {r['seg_len']:>7} {arm:<9} | "
                  f"false {a['false_n_random']}/{a['false_tot_random']} | "
                  f"correct {a['correct_n_parallel']}/"
                  f"{a['correct_tot_parallel']} | "
                  f"reliability {a['reliability']:.3f}")
    print("\n--- miss-gate attribution on parallel (non-triggered runs) ---")
    for seq_kind in ['ABAB', 'ABCABC']:
        for seg_len in sorted({e['seg_len'] for e in agg
                               if e['seq_kind'] == seq_kind}):
            par = [e for e in agg if e['seq_kind'] == seq_kind
                   and e['seg_len'] == seg_len
                   and e['substrate'] == 'parallel']
            tot_n = sum(e['n_runs'] for e in par)
            det_n = sum(e['n_detected'] for e in par)
            counts = {g: sum(e['blk_' + g] for e in par)
                      for g in ['floor', 'margin', 'hold', 'mixed',
                                'no_refits']}
            print(f"  {seq_kind} seg={seg_len}: detected {det_n}/{tot_n}"
                  + "".join(f", {g}={counts[g]}" for g in
                            ['floor', 'margin', 'hold', 'mixed',
                             'no_refits']))


# ========================== Main ==========================

def run_sweep(quick=False, trace=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S58B relative sense stress "
          f"(quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16,
                                           seed=TOPO_SEED,
                                           avg_degree=AVG_DEGREE)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w,
                      KAPPA_RANDOM, ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    n_seeds = N_SEEDS if not quick else 2
    all_args = []
    for seq_kind, seg_lens in [('ABAB', SEG_LENS_ABAB),
                               ('ABCABC', SEG_LENS_ABCABC)]:
        for seg_len in seg_lens:
            for topo in SUBSTRATES:
                for arm in ARMS:
                    for s in range(n_seeds):
                        all_args.append((topo, arm, seg_len, seq_kind, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (configs: "
          f"ABAB x {len(SEG_LENS_ABAB)} + ABCABC x "
          f"{len(SEG_LENS_ABCABC)} seg lens, 2 substrates x 2 arms x "
          f"{n_seeds} seeds)")

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
    csv_rows = [{k: v for k, v in r.items() if k != 'cap_trace'}
                for r in results]
    fieldnames = ['task', 'substrate', 'readout', 'seg_len', 'seq_kind',
                  'seed_idx', 'n_units', 't_total', 'runtime_s',
                  'mean_acc_all', 'switch_block', 'detect_cap', 'detect_cq',
                  'detect_delta', 'n_expansions', 'n_snaps', 'best_delta',
                  'best_delta_b', 'cq_at_best_delta', 'n_above_margin',
                  'n_above_floor', 'n_above_both', 'gate_blocker']
    for s in range(1, MAX_SEG + 1):
        fieldnames += [f'seg{s}_acc', f'seg{s}_cap', f'seg{s}_delta']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(csv_rows)

    agg = aggregate(results)
    rel = reliability_table(agg, n_seeds)
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
        'warmup_fraction': WARMUP_FRACTION, 'warmup_floor': WARMUP_FLOOR,
        'ring_in': RING_IN, 'ring_out': RING_OUT,
        'seg_lens_abab': SEG_LENS_ABAB, 'seg_lens_abcabc': SEG_LENS_ABCABC,
        'seq_abab': SEQ_ABAB, 'seq_abcabc': SEQ_ABCABC,
        'substrates': SUBSTRATES, 'arms': ARMS,
        'arm_branching': {'rls_auto': 'single OnlineRLS + relative sense',
                          'pop_auto': 'OnlineRLS + s52-v4 memory slots '
                                      '+ relative sense'},
        'reliability_truth_source': (
            'substrate/regime design table (offline capacity survey s57/s58a): '
            'trigger on parallel = true positive, trigger on '
            'random_graph_k25 = false positive; truth independent of the '
            'online probe'),
        'reliability_unit': 'run (seed x arm x substrate)',
        'reliability_ci': 'Wilson 95% interval on each factor; product CI '
                          'from the two disjoint factor CIs',
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    payload = _nan_to_null({
        'params': params, 'aggregates': agg, 'reliability': rel,
        'null_reason': ('non-finite aggregate values are written as null '
                        '(undefined, e.g. a stress segment the run length '
                        'never reached, or a detector that never fired)')})
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print_table(agg, rel, n_seeds)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")

    if trace:
        # seed-4 mechanism trace: (parallel, both arms, SEG=500, ABAB)
        tr = {}
        for arm in ARMS:
            r = run_single(('parallel', arm, 500, 'ABAB', 4))
            tr[arm] = {k: r[k] for k in
                       ['switch_block', 'detect_cap', 'detect_cq',
                        'detect_delta', 'best_delta', 'best_delta_b',
                        'cq_at_best_delta', 'n_above_margin',
                        'n_above_floor', 'n_above_both', 'gate_blocker']}
            tr[arm]['trace'] = r['cap_trace']
            print(f"[{time.strftime('%H:%M:%S')}] trace {arm}: "
                  f"switch={r['switch_block']}, "
                  f"best_delta={r['best_delta']:.3f} at b={r['best_delta_b']}, "
                  f"cq={r['cq_at_best_delta']:.3f}, "
                  f"blocker={r['gate_blocker']}")
        with open(TRACE_PATH, 'w') as f:
            json.dump(_nan_to_null(tr), f, indent=2, allow_nan=False)
        print(f"TRACE: {TRACE_PATH}")

    print(f"[{time.strftime('%H:%M:%S')}] DONE, total "
          f"{time.time() - t_start:.1f}s")


def _nan_to_null(obj):
    """Recursively map non-finite floats to None (strict-JSON safe).

    Bug fix 2026-09-09 (dedup P1-90): json.dump wrote bare NaN literals for
    undefined aggregates (e.g. gain_mean of an arm with no reward blocks),
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
    trace = '--trace' in sys.argv
    run_sweep(quick=quick, trace=trace)


if __name__ == '__main__':
    main()
