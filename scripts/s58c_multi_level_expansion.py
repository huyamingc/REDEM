#!/usr/bin/env python3
"""
Multi-level expansion chain: does autonomous basis expansion CONTINUE after
the first upgrade (cooldown instead of one-shot gate)? (REDEM S58C)
=============================================================================
Type:           PAPER
Paper Section:  S53/S56 follow-up (multi-level self-evolution)
Experiment:     s53 (static) and s56 (dynamic) capacity senses trigger a
                SINGLE linear->quadratic expansion and then stop sensing
                (s56 `switch_block < 0` one-shot gate). A real self-evolving
                system should keep sensing and expand AGAIN when the next
                task still exceeds the current representation. s58c removes
                the one-shot gate, adds a cooldown (one full window after
                each expansion), and lets the relative dual-basis sense run
                across the basis ladder linear -> quadratic -> cubic.

                The second-level question is empirical on the parallel
                substrate: s53 showed quadratic features only weakly repair
                the circle there (0.48 -> 0.57), and the substrate caps the
                whole family near chance for hard boundaries (s58a: the
                coupled substrate saturates, parallel collapses). So the
                falsifiable claim is really about the DECISION, not the
                recovery: after L1 expands on circle, does the sense (a)
                trigger L2 on a boundary that genuinely needs a higher basis,
                or (b) CORRECTLY REFUSE when the candidate basis cannot
                out-read the current one (no over-expansion)?

Environment: dynamic A-B-C-A-B-C on the 2D encoding, both substrates:
             [linear, circle, ring, linear, circle, ring] x 500 blocks,
             K=20, T=60000. ring = 2D annulus (0.30, 0.50), P=0.503 (s57);
             a boundary the quadratic basis cannot express as one surface
             (band on r^2 needs quartic terms) -- the L2 stress.

Basis ladder (degree k, coordinate-power convention of s53/s56):
             level 0: linear  F0 = [z, bias]                  (257)
             level 1: quad    [F0, z^2]                       (513)
             level 2: cubic   [F0, z^2, z^3]                  (769)
             where z = standardized exp(gamma*state)/FEATURE_SCALE.

Arms (block-vote scoring, protocol decision):
  rls_auto_multi  : single RLS + multi-level sense (no memory).
  pop_auto_multi  : population memory (s52 v4) + multi-level sense; each
                    expansion pads frozen snapshots with zeros, so a
                    remembered linear hypothesis stays exactly valid across
                    TWO upgrades (double zero-padding; s56 (a) generalised).
  rls_quad_start  : quadratic from block 0 (reference for the circle).
  rls_cubic_start : cubic from block 0 (oracle ceiling for the ring).

Sense (relative dual-basis, s56 v2, params fixed): at level k, probe BOTH
the current (level k) and candidate (level k+1) block-mean features against
the (r,vote)-reconstructed labels; expand iff candidate >= QUAD_FLOOR AND
candidate - current >= REL_MARGIN, sustained REL_HOLD refits, with a
COOLDOWN of one full window after each expansion (sense blind while the
probe buffers rebuild). Max level = 2 (cubic): no candidate beyond it.

Predictions:
  P1 (L1, s56 replication): on parallel, level-0 vs level-1 sense triggers
      inside the first circle segment (b ~ 739-999); on random_graph it
      never triggers (substrate provides the expansion; delta ~ 0).
  P2 (L2 decision, the new claim): after the cooldown, the level-1 vs
      level-2 sense on parallel either (a) triggers on a ring window
      (cubic out-reads quadratic: candidate genuinely helps) or (b) refuses
      (quadratic == cubic near chance on parallel: the substrate cannot
      support the upgrade -- "correct rejection of an invalid upgrade").
      Both are positive results; the ring is the probe.
  P3 (no over-expansion): on random_graph the sense stays at level 0 the
      whole run (0 expansions); on parallel it stops at the highest level
      that helps -- no runaway basis growth.
  P4 (memory across levels): pop_auto_multi keeps its linear-hypothesis
      snapshots across the upgrades (double padding) and still wins the
      linear REVISIT segments (seg4) over rls_auto_multi after both
      expansions.

Output files:
  data/s58c_multi_level_expansion_v1.csv   (one row per run)
  data/s58c_multi_level_expansion_v1.json  (params + per-cell aggregates)

Usage: python s58c_multi_level_expansion.py [--quick]
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

# s56 relative (dual-basis) capacity sense -- params fixed
WIN_BLOCKS = 240
REFIT_EVERY = 20
CAP_N_PC = 50
QUAD_FLOOR = 0.60
REL_MARGIN = 0.10
REL_HOLD = 3
WARMUP_FRACTION = 0.15
WARMUP_FLOOR = 60
COOLDOWN_BLOCKS = WIN_BLOCKS   # sense blind for one full window after expansion
MAX_LEVEL = 2                  # 0=linear, 1=quadratic, 2=cubic (cap)

# s57 2D ring (annulus) boundary (P=0.503)
RING_IN = 0.30
RING_OUT = 0.50

SEG_BLOCKS = 500
SEQ = ['linear', 'circle', 'ring', 'linear', 'circle', 'ring']
N_SEG = len(SEQ)
SUBSTRATES = ['random_graph_k25', 'parallel']
ARMS = ['rls_auto_multi', 'pop_auto_multi', 'rls_quad_start',
        'rls_cubic_start']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's58c_multi_level_expansion_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's58c_multi_level_expansion_v1.json')

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


def gen_alternating(seed):
    """[linear, circle, ring, linear, circle, ring] on the 2D substrate."""
    rng = np.random.RandomState(seed)
    dt_parts = []
    tgt_parts = []
    for bd in SEQ:
        s = int(rng.randint(0, 2**31 - 1))
        if bd == 'ring':
            dt, _, u0, u1 = gen_nonlinear2d(seed=s, n_blocks=SEG_BLOCKS,
                                            k_pulses=DB_K_PULSES,
                                            boundary='linear')
            d2 = (u0 - 0.5) ** 2 + (u1 - 0.5) ** 2
            lab = ((d2 > RING_IN ** 2) & (d2 < RING_OUT ** 2)).astype(np.int64)
            tgt = np.repeat(lab, DB_K_PULSES).astype(np.int64)
        else:
            dt, tgt, _, _ = gen_nonlinear2d(seed=s, n_blocks=SEG_BLOCKS,
                                            k_pulses=DB_K_PULSES, boundary=bd)
        dt_parts.append(dt)
        tgt_parts.append(tgt)
    return (np.concatenate(dt_parts),
            np.concatenate(tgt_parts).astype(np.int64))


def build_features_deg(states, T, degree):
    """Basis of coordinate powers: degree 1 = [z, bias], degree 2 adds z^2,
    degree 3 adds z^3 (z = standardized exp(gamma*state)/FEATURE_SCALE)."""
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F0 = (obs_raw - mu) / sd
    F = np.hstack([F0, np.full((T, 1), BIAS)])
    for k in range(2, degree + 1):
        F = np.hstack([F, F0 ** k])
    return F


def capacity_pca50(Xw, yw):
    """Well-conditioned capacity probe (s53): held-out ridge accuracy on the
    top-50 PCA components of a block-mean feature window."""
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

    total_blocks = N_SEG * SEG_BLOCKS
    warmup = max(WARMUP_FLOOR, int(WARMUP_FRACTION * total_blocks))

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate(topo_name)

    dt_seq, target_seq = gen_alternating(seed_idx)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    # basis ladder (built once; levels are nested so zero-padding preserves
    # frozen hypotheses exactly across upgrades)
    F_levels = [build_features_deg(states, T, deg)
                for deg in (1, 2, 3)]
    F_dims = [F.shape[1] for F in F_levels]

    preds = np.empty(T)
    n_expansions = 0
    level = 0
    last_expand_b = -COOLDOWN_BLOCKS
    n_snaps = 0
    cap_readings = []          # (level, block_idx, cur_cap, cand_cap, delta)
    exp_blocks = []            # per-expansion trigger blocks
    exp_deltas = []            # per-expansion detect delta
    exp_cqs = []               # per-expansion candidate reading

    if arm in ('rls_quad_start', 'rls_cubic_start'):
        start_deg = 2 if arm == 'rls_quad_start' else 3
        F = F_levels[start_deg - 1]
        d = F.shape[1]
        rls = OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                        init_cov=RLS_INIT_COV)
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
    else:
        # rls_auto_multi / pop_auto_multi: shared multi-level sense
        auto = True
        pop = (arm == 'pop_auto_multi')
        Fcur = F_levels[0]
        d = Fcur.shape[1]
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
        Xw = []                    # current-level block means
        Xc = []                    # candidate-level block means
        yw = []
        rel_hold = 0

        def expand(b, cl, cc, delta):
            nonlocal level, last_expand_b, Fcur, worker, r_ema_w, hold_cnt
            nonlocal recent_blk, Xw, Xc, yw, rel_hold, n_expansions
            n_expansions += 1
            level += 1
            last_expand_b = b
            exp_blocks.append(b)
            exp_deltas.append(float(delta))
            exp_cqs.append(float(cc))
            d_new = F_dims[level]
            if pop:
                pad = d_new - Fcur.shape[1]
                for i in range(len(mem_w)):
                    mem_w[i] = np.concatenate(
                        [mem_w[i], np.zeros(pad)])
            Fcur = F_levels[level]
            worker = OnlineRLS(d_new, 1, forgetting=RLS_FORGETTING,
                               init_cov=RLS_INIT_COV)
            r_ema_w = 0.0
            hold_cnt = 0
            recent_blk = []
            Xw = []
            Xc = []
            yw = []
            rel_hold = 0

        for t in range(T):
            if active_kind == 0:
                preds[t] = float(worker.predict(Fcur[t])[0])
            else:
                preds[t] = float(mem_w[active_idx] @ Fcur[t])
            if (t + 1) % DB_K_PULSES == 0:
                b = (t + 1) // DB_K_PULSES - 1
                seg = Fcur[t - DB_K_PULSES + 1:t + 1]
                x_blk = seg.mean(axis=0)
                if pop:
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
                if pop:
                    for i in range(len(mem_w)):
                        mv = float(mem_w[i] @ x_blk)
                        r_m = 1.0 if (mv > 0.5) == target[t] else -1.0
                        mem_ema[i] = (1.0 - PROBE_EMA_ALPHA) * mem_ema[i] \
                            + PROBE_EMA_ALPHA * r_m
                v_w = float(wv > 0.5)
                r_w2 = 1.0 if v_w == target[t] else -1.0
                y_d = v_w if r_w2 > 0.0 else (1.0 - v_w)
                worker.update(x_blk, np.array([y_d]))
                if pop:
                    if r_ema_w > SNAP_EMA_MIN:
                        hold_cnt += 1
                        if hold_cnt >= SNAP_HOLD:
                            snap = worker.W[:, 0].copy()
                            if recent_blk:
                                wv_agree = (snap @ np.array(recent_blk).T
                                            > 0.5)
                                agree = []
                                for mw in mem_w:
                                    m_agree = (mw @ np.array(recent_blk).T
                                               > 0.5)
                                    agree.append(float(np.mean(
                                        wv_agree == m_agree)))
                                best_agr = max(agree) if agree else 0.0
                                i_best = int(np.argmax(agree)) if agree \
                                    else 0
                            else:
                                best_agr, i_best = 0.0, 0
                                agree = []
                            if agree and best_agr > AGREE_THRESH:
                                mem_w[i_best] = snap
                                mem_ema[i_best] = r_ema_w
                                peak_ema[i_best] = max(peak_ema[i_best],
                                                       r_ema_w)
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
                # multi-level relative sense
                if level < MAX_LEVEL:
                    xb_cur = Fcur[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
                    xb_cand = F_levels[level + 1][
                        t - DB_K_PULSES + 1:t + 1].mean(axis=0)
                    Xw.append(xb_cur)
                    Xc.append(xb_cand)
                    yw.append(y_derived)
                    if len(Xw) > WIN_BLOCKS:
                        Xw.pop(0)
                        Xc.pop(0)
                        yw.pop(0)
                    if (len(Xw) >= 40 and b >= warmup
                            and (b + 1) % REFIT_EVERY == 0
                            and b >= last_expand_b + COOLDOWN_BLOCKS):
                        cl = capacity_pca50(Xw, yw)
                        cc = capacity_pca50(Xc, yw)
                        delta = cc - cl
                        cap_readings.append((level, b, cl, cc, delta))
                        if cc >= QUAD_FLOOR and delta >= REL_MARGIN:
                            rel_hold += 1
                            if rel_hold >= REL_HOLD:
                                expand(b, cl, cc, delta)
                                seg = Fcur[t - DB_K_PULSES + 1:t + 1]
                                x_blk = seg.mean(axis=0)
                        else:
                            rel_hold = 0
                # select active (pop only; rls_auto has no memory slots)
                if pop:
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
    seg_acc = [acc_median_in(acc_run, s * SEG_BLOCKS,
                             (s + 1) * SEG_BLOCKS) for s in range(N_SEG)]

    def _best_delta_at(lvl):
        vals = [c[3] for c in cap_readings if c[0] == lvl]
        return float(np.max(vals)) if vals else float('nan')

    # divergence diagnostics: the worker RLS can wind up (weight blow-up)
    # on boundaries it cannot fit online (s58c finding: the memoryless arm
    # diverges on the ring; memory selection protects the pop arm).
    w_norm_end = float('nan')
    if arm in ('rls_auto_multi', 'pop_auto_multi'):
        w_norm_end = float(np.linalg.norm(worker.W))

    res = {'task': 'multi_level_expansion', 'substrate': topo_name,
           'readout': arm, 'seed_idx': seed_idx, 'n_units': N_UNITS,
           't_total': int(T), 'runtime_s': time.time() - t0,
           'mean_acc_all': float(np.nanmean(acc_run)),
           'n_expansions': n_expansions, 'n_snaps': n_snaps,
           'best_delta_l0': _best_delta_at(0),
           'best_delta_l1': _best_delta_at(1),
           'w_norm_end': w_norm_end,
           'max_abs_pred': float(np.max(np.abs(preds)))}
    for s in range(N_SEG):
        res[f'seg{s + 1}_acc'] = seg_acc[s]
    for k in range(2):                       # exp1/exp2 columns
        res[f'exp{k + 1}_block'] = exp_blocks[k] if k < len(exp_blocks) \
            else -1
        res[f'exp{k + 1}_delta'] = exp_deltas[k] if k < len(exp_deltas) \
            else float('nan')
        res[f'exp{k + 1}_cq'] = exp_cqs[k] if k < len(exp_cqs) \
            else float('nan')
    return res


# ========================== Aggregation ==========================

def agg_list(rs, field):
    vals = np.array([r[field] for r in rs], dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['substrate'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        topo, read = key
        entry = {'substrate': topo, 'readout': read, 'n_runs': len(rs)}
        for f in (['mean_acc_all'] +
                  [f'seg{s}_acc' for s in range(1, N_SEG + 1)]):
            m, sd = agg_list(rs, f)
            entry[f + '_mean'] = m
            entry[f + '_std'] = sd
        entry['n_expansions_mean'] = float(np.nanmean(
            np.array([r['n_expansions'] for r in rs], dtype=float)))
        entry['n_expansions_max'] = int(np.max(
            [r['n_expansions'] for r in rs]))
        entry['n_l2'] = int(sum(1 for r in rs if r['n_expansions'] >= 2))
        entry['n_l1'] = int(sum(1 for r in rs if r['n_expansions'] >= 1))
        entry['n_snaps_mean'] = float(np.nanmean(
            np.array([r['n_snaps'] for r in rs], dtype=float)))
        entry['best_delta_l0_mean'] = float(np.nanmean(
            np.array([r['best_delta_l0'] for r in rs], dtype=float)))
        entry['best_delta_l1_mean'] = float(np.nanmean(
            np.array([r['best_delta_l1'] for r in rs], dtype=float)))
        wn = np.array([r['w_norm_end'] for r in rs], dtype=float)
        wn = wn[np.isfinite(wn)]
        entry['w_norm_end_mean'] = float(np.nanmean(wn)) if wn.size \
            else float('nan')
        entry['w_norm_end_max'] = float(np.nanmax(wn)) if wn.size \
            else float('nan')
        mp = np.array([r['max_abs_pred'] for r in rs], dtype=float)
        entry['max_abs_pred_mean'] = float(np.nanmean(mp))
        for k in (1, 2):
            vals_b = np.array([r[f'exp{k}_block'] for r in rs], dtype=float)
            vals_d = np.array([r[f'exp{k}_delta'] for r in rs], dtype=float)
            vals_q = np.array([r[f'exp{k}_cq'] for r in rs], dtype=float)
            entry[f'exp{k}_block_mean'] = float(np.nanmean(
                vals_b[vals_b >= 0])) if np.any(vals_b >= 0) else float('nan')
            entry[f'exp{k}_delta_mean'] = float(np.nanmean(vals_d))
            entry[f'exp{k}_cq_mean'] = float(np.nanmean(vals_q))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 128)
    print("S58C MULTI-LEVEL EXPANSION CHAIN (linear -> quad -> cubic)")
    print("=" * 128)
    print(f"\n  sequence: {SEQ} x {SEG_BLOCKS} blocks")
    for topo in SUBSTRATES:
        print(f"\n--- {topo} ---")
        print(f"  {'readout':<16} | " + "".join(
            f"{'s%d' % (i + 1):>6}" for i in range(N_SEG)) +
            f" | {'mean':>6} | {'exp':>3} | {'nL1':>3} | {'nL2':>3} | "
            f"{'bL1':>5} | {'bL2':>5} | {'dL0':>5} | {'dL1':>5} | "
            f"{'wNorm':>7} | {'snaps':>5}")
        for a in [x for x in agg if x['substrate'] == topo]:
            b1 = a['exp1_block_mean'] if np.isfinite(a['exp1_block_mean']) \
                else float('nan')
            b2 = a['exp2_block_mean'] if np.isfinite(a['exp2_block_mean']) \
                else float('nan')
            wn = a['w_norm_end_mean'] if np.isfinite(a['w_norm_end_mean']) \
                else float('nan')
            print(f"  {a['readout']:<16} | " + "".join(
                f"{a['seg%d_acc_mean' % (i + 1)]:>6.3f}"
                for i in range(N_SEG)) +
                f" | {a['mean_acc_all_mean']:>6.3f} | "
                f"{a['n_expansions_mean']:>3.1f} | {a['n_l1']:>3} | "
                f"{a['n_l2']:>3} | {b1:>5.0f} | {b2:>5.0f} | "
                f"{a['best_delta_l0_mean']:>5.3f} | "
                f"{a['best_delta_l1_mean']:>5.3f} | "
                f"{wn:>7.0f} | "
                f"{a['n_snaps_mean']:>5.1f}")


def _nan_to_null(obj):
    """Recursively map non-finite floats to None (strict-JSON safe).

    Bug fix 2026-09-09 (dedup P1-90): json.dump wrote bare NaN literals for
    undefined aggregates (e.g. w_norm_end of an arm that never expanded),
    which is not valid JSON and is rejected by strict parsers.
    """
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_null(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S58C multi-level expansion "
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
                print(f"[{time.strftime('%H:%M:%S')}] progress "
                      f"{done}/{n_runs}", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'substrate', 'readout', 'seed_idx', 'n_units',
                  't_total', 'runtime_s', 'mean_acc_all', 'n_expansions',
                  'n_snaps', 'best_delta_l0', 'best_delta_l1',
                  'exp1_block', 'exp1_delta', 'exp1_cq',
                  'exp2_block', 'exp2_delta', 'exp2_cq',
                  'w_norm_end', 'max_abs_pred']
    for s in range(1, N_SEG + 1):
        fieldnames.append(f'seg{s}_acc')
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
        'cooldown_blocks': COOLDOWN_BLOCKS, 'max_level': MAX_LEVEL,
        'ring_in': RING_IN, 'ring_out': RING_OUT,
        'seg_blocks': SEG_BLOCKS, 'seq': SEQ,
        'substrates': SUBSTRATES, 'arms': ARMS,
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    payload = _nan_to_null({'params': params, 'aggregates': agg,
                            'null_reason': ('non-finite aggregate values are '
                                            'written as null (undefined, e.g. '
                                            'an arm that never expanded)')})
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, allow_nan=False)

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
