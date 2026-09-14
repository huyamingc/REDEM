#!/usr/bin/env python3
"""
Adaptive sense window: can the capacity sense self-scale its integration
window to the regime dwell, structurally repairing the s58b reliability
cliff? (REDEM G2 = s60)
=============================================================================
Type:           PAPER
Paper Section:  S58b/S58f follow-up (the structural fix)
Experiment:     s58b showed the relative (dual-basis) capacity sense has a
                reliability cliff at window/segment ratio >= ~2: on the
                insufficient (parallel) substrate the 240-block probe
                window straddles short regimes and DILUTES the genuine
                quad-vs-linear delta signal (d_best collapses 0.19 -> 0.04
                as SEG 500 -> 125). s58f proved the cliff cannot be fixed by
                tuning REL_MARGIN (noise floor and signal overlap) and
                concluded the fix must be STRUCTURAL: windows that track the
                regime dwell (WIN ~ 0.5 x dwell keep windows clean). G2
                implements exactly that fix and tests it.

Adaptive window: the sense uses a LONG window (240) in easy regimes and
switches IMMEDIATELY to a SHORT window (WIN_SHORT=100) the moment it detects
hard content -- the worker reward-rate EMA r_ema_det dropping below DET_LO
(0.2); the adaptation starts at block DET_START=100 (cold-start skip) and
returns to the long window once r_ema_det recovers above REC_HI (0.6). On the
parallel substrate this fires ~60 blocks into each circle (undecodable
content) and clears at the next linear segment. A 100-block probe window fits
inside a 250-block circle and keeps the block-mean window free of regime
mixing, which s58b showed is what dilutes the genuine quad-vs-linear delta
signal. The detector does NOT stay silent on the coupled substrate: the
measured hard-content fraction on random_graph at SEG=250 is 0.127 with
win_mean 222.2 for the no-memory adaptive arm rls_auto_adapt, and 0.159 with
win_mean 217.7 for the memory arm pop_auto_adapt. The measured win_mean on
the parallel substrate is 173.6 at SEG=250 and 163.3 at SEG=500 for
rls_auto_adapt (176.1 and 164.0 for pop_auto_adapt). The circle content is
partly undecodable on the coupled substrate too, so the window is shortened
part of the time on BOTH substrates; the two adaptive arms are independent
runs (the arm-branching repair removed a duplication, so they no longer share
a trajectory). The sense buffers (Xw/Xwq/yw) are trimmed when the window
shortens.

Environment: s58b ABAB [linear, circle, linear, circle] x SEG_BLOCKS,
SEG_BLOCKS in {500, 250}, K=20, both substrates. Arms:
  rls_auto      : single RLS + sense, fixed WIN=240 (s58b replication).
  pop_auto      : memory + sense, fixed WIN=240 (s58b replication).
  rls_auto_adapt: rls_auto with the adaptive window.
  pop_auto_adapt: pop_auto with the adaptive window.
Every other sense parameter is fixed at the s56/s58b values.

Predictions (falsifiable):
  P1 (parallel, the cliff): at SEG=250 the adaptive arms' correct-trigger
      rate exceeds the fixed arms' (0.30 in s58b), because the shortened
      window (WIN ~ 125-140) fits inside the 250-block regimes and the
      delta signal survives; reliability rises toward the SEG=500 level.
      [REFUTED by own data: see data/s60_adaptive_window_v1.csv] The
      measured win_mean is 173.6, not 125-140, and the parallel
      correct-trigger count at SEG=250 is 3/10 for rls_auto_adapt and 3/10
      for pop_auto_adapt -- identical to the fixed arms (3/10 each). The
      pooled reliability is therefore unchanged at 0.270 (false 2/20 on
      random_graph, correct 6/20 on parallel). The shortened window does
      raise the sense margin (d_best 0.174 -> 0.198) but that does not convert
      into more detections.
  P2 (random, no false-trigger inflation): on random_graph the hard-regime
      detector never fires (r_ema stays high), so WIN stays at 240 and the
      false-trigger rate matches the fixed arms (low).
      [data: the false-trigger rate does match (1/10 per adaptive arm at
      SEG=250, same as the fixed arms), but the detector DOES fire on the
      coupled substrate -- hard_frac 0.127 with win_mean 222.2 at SEG=250
      (0.0995 with win_mean 226.1 at SEG=500) for rls_auto_adapt, against
      0.159 / 217.7 and 0.006 / 239.1 for pop_auto_adapt -- so the "never
      fires" part of this prediction is refuted as a mechanism statement.]
  P3 (long segments unchanged): at SEG=500 both adaptive and fixed arms
      match (reliability ~0.9); the adaptation is a no-op where the fixed
      window was already clean.
      [REFUTED by own data: see data/s60_adaptive_window_v1.csv] At SEG=500
      the parallel correct-trigger count falls from 9/10 per fixed arm to
      5/10 per adaptive arm, so the pooled reliability drops from 0.900 to
      0.475 (pop_auto_adapt alone 0.500, rls_auto_adapt 0.450 -- the latter's
      SEG=500 false trigger, 0/10 -> 1/10 on random_graph, appears only after
      the arm-branching repair) -- the adaptation is not a no-op at long
      segments.]

Output files:
  data/s60_adaptive_window_v1.csv    (one row per run)
  data/s60_adaptive_window_v1.json   (params + per-cell aggregates)

Usage: python s60_adaptive_window.py [--quick]
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

# s56 relative sense -- fixed except the window (the adapted axis)
WIN_BLOCKS = 240
REFIT_EVERY = 20
CAP_N_PC = 50
QUAD_FLOOR = 0.60
REL_MARGIN = 0.10
REL_HOLD = 3
WARMUP_FRACTION = 0.15
WARMUP_FLOOR = 60

# G2 adaptive window
WIN_SHORT = 100          # probe window while a hard regime is detected
WIN_MIN = WIN_SHORT
WIN_MAX = WIN_BLOCKS
DET_LO = 0.2            # r_ema below this = "hard regime" (short window)
REC_HI = 0.6            # r_ema above this = "recovered" (long window)
DET_START = 100         # skip the cold-start adaptation (b >= this)

SEG_LENS = [500, 250]
SEGMENTS = ['linear', 'circle', 'linear', 'circle']
SUBSTRATES = ['random_graph_k25', 'parallel']
ARMS = ['rls_auto', 'pop_auto', 'rls_auto_adapt', 'pop_auto_adapt']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's60_adaptive_window_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's60_adaptive_window_v1.json')

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


def gen_alternating(seed, seg_blocks):
    rng = np.random.RandomState(seed)
    dt_parts = []
    tgt_parts = []
    for bd in SEGMENTS:
        s = int(rng.randint(0, 2**31 - 1))
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
    """(substrate, arm, seg_len, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, seg_len, seed_idx = args
    t0 = time.time()

    adapt = arm.endswith('_adapt')
    n_seg = len(SEGMENTS)
    total_blocks = n_seg * seg_len
    warmup = max(WARMUP_FLOOR, int(WARMUP_FRACTION * total_blocks))

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq = gen_alternating(seed_idx, seg_len)
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
    detect_delta = float('nan')
    n_expansions = 0
    cap_readings = []

    win = WIN_BLOCKS                     # current sense window
    hard = False                         # detected "in a hard regime"
    win_samples = []                     # win after warmup (for win_mean)
    r_ema_det = 0.0

    # ARM BRANCHING (bug fix 2026-09-10, dedup P1-104): all four arms are
    # sense users, but ONLY the pop_auto family carries the s52-v4 population
    # memory (snapshot + argmax-EMA selection). The rls_auto family is a single
    # continuously adapting RLS + the same relative capacity sense. The
    # previous version had no branch at all, so the memory layer ran inside
    # every arm and rls_auto was a byte-identical pseudo-replicate of pop_auto
    # (both fixed arms) and rls_auto_adapt of pop_auto_adapt (both adaptive
    # arms) -- 160 CSV rows were 80 real runs. Same defect, same fix as s58b
    # (F-P0-3 / dedup P0-27).
    use_memory = arm.startswith('pop_')
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
                    else:
                        if len(mem_w) < POP_K - 1:
                            mem_w.append(snap)
                            mem_ema.append(r_ema_w)
                            peak_ema.append(r_ema_w)
                        elif agree:
                            i_low = int(np.argmin(peak_ema))
                            mem_w[i_low] = snap
                            mem_ema[i_low] = r_ema_w
                            peak_ema[i_low] = r_ema_w
                    hold_cnt = 0
            else:
                hold_cnt = 0
            # G2: adaptive window -- IMMEDIATE response to detected hard
            # content. The sense keeps a LONG window in easy regimes (the
            # probe is well-conditioned) and switches to a SHORT window the
            # moment the system detects it has entered a hard regime
            # (r_ema_det < DET_LO; on the parallel substrate this fires
            # ~60 blocks into each circle, whose content is undecodable, and
            # clears on recovery above REC_HI at the next linear segment).
            # A short window (WIN_SHORT=100) fits inside a 250-block circle
            # and keeps the probe's block-mean window FREE of regime mixing,
            # which s58b showed is what dilutes the genuine quad-vs-linear
            # delta signal. On the coupled substrate r_ema_det never stays
            # below DET_LO (both boundaries decodable), so the window stays
            # long -- no behaviour change where the fixed window was clean.
            if adapt:
                r_ema_det = (1.0 - PROBE_EMA_ALPHA) * r_ema_det \
                    + PROBE_EMA_ALPHA * r
                if b >= DET_START:
                    if r_ema_det < DET_LO:
                        hard = True
                    elif r_ema_det >= REC_HI:
                        hard = False
                win = WIN_SHORT if hard else WIN_BLOCKS
                if len(Xw) > win:
                    while len(Xw) > win:
                        Xw.pop(0)
                        Xwq.pop(0)
                        yw.pop(0)
                if b >= warmup:
                    win_samples.append(win)
            # relative dual-basis sense
            x_blk_q = F_quad[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
            Xw.append(x_blk)
            Xwq.append(x_blk_q)
            yw.append(y_derived)
            if len(Xw) > win:
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
                        r_ema_det = 0.0
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

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    seg_acc = [acc_median_in(acc_run, s * seg_len,
                             (s + 1) * seg_len) for s in range(n_seg)]

    deltas = np.array([c[3] for c in cap_readings], dtype=float)
    best_delta = float(np.max(deltas)) if deltas.size else float('nan')

    res = {'task': 'adaptive_window', 'substrate': topo_name,
           'readout': arm, 'seg_len': int(seg_len), 'seed_idx': seed_idx,
           'n_units': N_UNITS, 't_total': int(T),
           'runtime_s': time.time() - t0,
           'mean_acc_all': float(np.nanmean(acc_run)),
           'switch_block': switch_block, 'detect_delta': detect_delta,
           'n_expansions': n_expansions, 'best_delta': best_delta,
           'win_mean': float(np.mean(win_samples)) if win_samples
                       else float(WIN_BLOCKS),
           'hard_frac': float(np.mean([1.0 if w < WIN_BLOCKS else 0.0
                                       for w in win_samples]))
                       if win_samples else 0.0}
    for s in range(n_seg):
        res[f'seg{s + 1}_acc'] = seg_acc[s]
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        key = (r['substrate'], r['readout'], r['seg_len'])
        groups.setdefault(key, []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        topo, read, seg_len = key
        entry = {'substrate': topo, 'readout': read, 'seg_len': seg_len,
                 'n_runs': len(rs)}
        for f in (['mean_acc_all', 'best_delta', 'win_mean', 'hard_frac'] +
                  [f'seg{s}_acc' for s in range(1, 5)]):
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        sw = np.array([r['switch_block'] for r in rs], dtype=float)
        entry['n_detected'] = int(np.sum(sw >= 0))
        agg.append(entry)
    return agg


def agg_cell(agg, substrate, readout, seg_len, field):
    """Value of `field` for one (substrate, readout, seg_len) cell of agg.

    Returns NaN when the cell is absent (an arm or segment length not run in
    this sweep), so the params JSON never carries a fabricated number.
    """
    for e in agg:
        if (e['substrate'] == substrate and e['readout'] == readout
                and e['seg_len'] == seg_len):
            return float(e[field])
    return float('nan')


def print_table(agg):
    print("\n" + "=" * 124)
    print("S60 ADAPTIVE WINDOW (G2: window tracks regime dwell)")
    print("=" * 124)
    for seg_len in SEG_LENS:
        print(f"\n=== SEG={seg_len} ===")
        print(f"  {'substrate':<16} | {'arm':<15} | {'mean':>6} | "
              f"{'det':>3} | {'win':>4} | {'hard%':>5} | {'d_best':>6}")
        for topo in SUBSTRATES:
            for arm in ARMS:
                rs = [e for e in agg if e['substrate'] == topo
                      and e['readout'] == arm and e['seg_len'] == seg_len]
                if not rs:
                    continue
                e = rs[0]
                print(f"  {topo:<16} | {arm:<15} | "
                      f"{e['mean_acc_all_mean']:>6.3f} | "
                      f"{e['n_detected']:>3} | {e['win_mean_mean']:>4.0f} | "
                      f"{100.0 * e['hard_frac_mean']:>5.0f} | "
                      f"{e['best_delta_mean']:>6.3f}")
    print("\n--- reliability (false on random x correct on parallel, both arms pooled) ---")
    for seg_len in SEG_LENS:
        for tag, arms in [('fixed', ['rls_auto', 'pop_auto']),
                          ('adapt', ['rls_auto_adapt', 'pop_auto_adapt'])]:
            rnd = [e for e in agg if e['seg_len'] == seg_len
                   and e['readout'] in arms
                   and e['substrate'] == 'random_graph_k25']
            par = [e for e in agg if e['seg_len'] == seg_len
                   and e['readout'] in arms and e['substrate'] == 'parallel']
            fn = sum(e['n_detected'] for e in rnd)
            ft = sum(e['n_runs'] for e in rnd)
            cn = sum(e['n_detected'] for e in par)
            ct = sum(e['n_runs'] for e in par)
            rel = (1.0 - fn / ft) * (cn / ct) if ft and ct else float('nan')
            print(f"  SEG={seg_len} {tag:<6}: false {fn}/{ft}, "
                  f"correct {cn}/{ct}, reliability {rel:.3f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S60 adaptive window "
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
    for seg_len in SEG_LENS:
        for topo in SUBSTRATES:
            for arm in ARMS:
                for s in range(n_seeds):
                    all_args.append((topo, arm, seg_len, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seg_lens={SEG_LENS}, "
          f"substrates={len(SUBSTRATES)}, arms={len(ARMS)}, "
          f"seeds={n_seeds})")

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
    fieldnames = ['task', 'substrate', 'readout', 'seg_len', 'seed_idx',
                  'n_units', 't_total', 'runtime_s', 'mean_acc_all',
                  'switch_block', 'detect_delta', 'n_expansions',
                  'best_delta', 'win_mean', 'hard_frac',
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
        'win_blocks': WIN_BLOCKS, 'refit_every': REFIT_EVERY,
        'cap_n_pc': CAP_N_PC, 'quad_floor': QUAD_FLOOR,
        'rel_margin': REL_MARGIN, 'rel_hold': REL_HOLD,
        'warmup_fraction': WARMUP_FRACTION, 'warmup_floor': WARMUP_FLOOR,
        'win_min': WIN_MIN, 'win_max': WIN_MAX, 'win_short': WIN_SHORT,
        'det_lo': DET_LO, 'rec_hi': REC_HI, 'det_start': DET_START,
        'seg_lens': SEG_LENS, 'segments': SEGMENTS,
        'substrates': SUBSTRATES, 'arms': ARMS,
        # measured G2 quantities quoted by the docstring (dedup P1-98): the
        # hard-content fraction and the window means this sweep actually
        # computed. The per-cell values are in 'aggregates'; these entries make
        # the docstring's numbers traceable from the params block itself.
        'hard_frac': agg_cell(agg, 'random_graph_k25', 'rls_auto_adapt', 250,
                              'hard_frac_mean'),
        'hard_frac_pooled': float(np.nanmean(
            [e['hard_frac_mean'] for e in agg])),
        'win_mean_parallel_seg250': agg_cell(agg, 'parallel',
                                             'rls_auto_adapt', 250,
                                             'win_mean_mean'),
        'win_mean_parallel_seg500': agg_cell(agg, 'parallel',
                                             'rls_auto_adapt', 500,
                                             'win_mean_mean'),
        'win_mean_random_seg250': agg_cell(agg, 'random_graph_k25',
                                           'rls_auto_adapt', 250,
                                           'win_mean_mean'),
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total "
          f"{time.time() - t_start:.1f}s")


def main():
    quick = '--quick' in sys.argv
    run_sweep(quick=quick)


if __name__ == '__main__':
    main()
