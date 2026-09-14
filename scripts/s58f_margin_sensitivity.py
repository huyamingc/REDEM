#!/usr/bin/env python3
"""
Capacity-sense parameter sensitivity: can a larger REL_MARGIN kill the
random_graph false trigger without killing parallel triggers? (REDEM S58F)
=============================================================================
Type:           PAPER
Paper Section:  S58b follow-up (sense-parameter calibration; R58G motivated
                item)
Experiment:     s58b showed the relative (dual-basis) capacity sense has a
                probe-noise floor: on random_graph the best single-refit
                delta averages ~0.13 (max 0.38), and at SEG=250 one seed
                false-triggered (b=219, pure-linear window, delta 0.182
                sustained 3 refits) -- REL_MARGIN=0.10 sits INSIDE the
                noise floor. s58f sweeps REL_MARGIN in {0.10, 0.15, 0.20}
                on the S58b ABAB environment (SEG in {500, 250}) and asks
                whether a higher margin trades away the genuine parallel
                trigger signal.

Environment: A-B-A-B [linear, circle, linear, circle] x SEG_BLOCKS,
             SEG_BLOCKS in {500, 250}, K=20, both substrates. Arms
             rls_auto / pop_auto (the relative-sense users; every other
             sense parameter fixed at the s56/s58b values).

Prediction (falsifiable):
  P1: raising REL_MARGIN to 0.15/0.20 removes the random_graph false
      trigger with little cost to the parallel correct-trigger rate at
      SEG=500 (genuine circle delta is 0.10-0.25, so 0.15 should keep most
      triggers, 0.20 may cost some).
      [PARTIALLY REFUTED by own data. The counts below are read from
      data/s58f_margin_sensitivity_v1.csv, stated per arm (each arm has 10
      seeds; the print_table totals pool both arms, e.g. 18/20 = 9/10 per
      arm). At SEG=250 the random_graph (false) trigger count is 1/10 at
      margin 0.10, still 1/10 at margin 0.15, and 0/10 only at margin 0.20;
      at SEG=500 random_graph is already 0/10 at every margin. So 0.20 --
      NOT 0.15 -- is the margin at which the random false triggers vanish,
      which is the corrected form of this prediction's threshold claim. The
      cost on the insufficient (parallel) substrate is larger than
      predicted: at SEG=500 the genuine (correct) trigger count falls from
      9/10 at margin 0.10 to 6/10 at margin 0.15 and 1/10 at margin 0.20.
      Corrected 2026-09-09, dedup P1-97.]
  P2: at SEG=250 (window mixing dilutes the parallel signal to d_best
      ~0.17 mean), the margin increase costs MORE -- parallel correct rate
      is already 0.30 at margin 0.10.
      [data: SEG=250 parallel per arm 3/10 at margin 0.10, 1/10 at 0.15,
      0/10 at 0.20 -- the cost is indeed larger than at SEG=500.]
  The two curves (false rate vs margin on random, correct rate vs margin
  on parallel) define the operating point of the sense: the margin should
  sit above the noise floor (random) but below the genuine-signal band
  (parallel).

Output files:
  data/s58f_margin_sensitivity_v1.csv   (one row per run)
  data/s58f_margin_sensitivity_v1.json  (params + per-cell aggregates)

Usage: python s58f_margin_sensitivity.py [--quick]
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

# s56 relative sense -- fixed except REL_MARGIN (the swept axis)
WIN_BLOCKS = 240
REFIT_EVERY = 20
CAP_N_PC = 50
QUAD_FLOOR = 0.60
REL_HOLD = 3
WARMUP_FRACTION = 0.15
WARMUP_FLOOR = 60

MARGINS = [0.10, 0.15, 0.20]      # swept axis
SEG_LENS = [500, 250]             # s58b reliability cliff neighborhood
SEGMENTS = ['linear', 'circle', 'linear', 'circle']
SUBSTRATES = ['random_graph_k25', 'parallel']
ARMS = ['rls_auto', 'pop_auto']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's58f_margin_sensitivity_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's58f_margin_sensitivity_v1.json')

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
    """(substrate, arm, seg_len, margin, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    topo_name, arm, seg_len, margin, seed_idx = args
    t0 = time.time()

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

    worker = OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                       init_cov=RLS_INIT_COV)
    # ARM BRANCHING (bug fix 2026-09-09, dedup P0-27): only pop_auto carries
    # the s52-v4 snapshot memory; rls_auto is a single adapting RLS. The
    # previous version branched on a dead local `auto = True`, so the two arms
    # were byte-identical pseudo-replicates.
    use_memory = (arm == 'pop_auto')
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
            # relative dual-basis sense with the config's margin
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
                if cq >= QUAD_FLOOR and delta >= margin:
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
    cqs = np.array([c[2] for c in cap_readings], dtype=float)
    best_delta = float(np.max(deltas)) if deltas.size else float('nan')
    n_above_margin = int(np.sum(deltas >= margin)) if deltas.size else 0
    n_above_both = int(np.sum((deltas >= margin)
                              & (cqs >= QUAD_FLOOR))) if deltas.size else 0

    res = {'task': 'margin_sensitivity', 'substrate': topo_name,
           'readout': arm, 'seg_len': int(seg_len), 'rel_margin': margin,
           'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T),
           'runtime_s': time.time() - t0,
           'mean_acc_all': float(np.nanmean(acc_run)),
           'switch_block': switch_block, 'detect_delta': detect_delta,
           'n_expansions': n_expansions, 'best_delta': best_delta,
           'n_above_margin': n_above_margin, 'n_above_both': n_above_both}
    for s in range(n_seg):
        res[f'seg{s + 1}_acc'] = seg_acc[s]
    return res


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        key = (r['substrate'], r['readout'], r['seg_len'], r['rel_margin'])
        groups.setdefault(key, []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        topo, read, seg_len, margin = key
        entry = {'substrate': topo, 'readout': read, 'seg_len': seg_len,
                 'rel_margin': margin, 'n_runs': len(rs)}
        for f in (['mean_acc_all', 'best_delta'] +
                  [f'seg{s}_acc' for s in range(1, 5)]):
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        sw = np.array([r['switch_block'] for r in rs], dtype=float)
        entry['n_detected'] = int(np.sum(sw >= 0))
        entry['switch_block_mean'] = float(np.nanmean(sw))
        entry['n_above_margin_mean'] = float(np.nanmean(
            np.array([r['n_above_margin'] for r in rs], dtype=float)))
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 120)
    print("S58F REL_MARGIN SENSITIVITY (false rate on random vs correct on parallel)")
    print("=" * 120)
    for seg_len in SEG_LENS:
        print(f"\n=== SEG={seg_len} ===")
        print(f"  {'margin':>6} | {'false/20 (random)':>17} | "
              f"{'correct/20 (parallel)':>20} | "
              f"{'reliability':>11} | {'d_best rand':>11} | "
              f"{'d_best par':>10}")
        for margin in MARGINS:
            rnd = [e for e in agg if e['seg_len'] == seg_len
                   and e['rel_margin'] == margin
                   and e['substrate'] == 'random_graph_k25']
            par = [e for e in agg if e['seg_len'] == seg_len
                   and e['rel_margin'] == margin
                   and e['substrate'] == 'parallel']
            fn = sum(e['n_detected'] for e in rnd)
            ft = sum(e['n_runs'] for e in rnd)
            cn = sum(e['n_detected'] for e in par)
            ct = sum(e['n_runs'] for e in par)
            fr = fn / ft if ft else float('nan')
            cr = cn / ct if ct else float('nan')
            rel = (1.0 - fr) * cr
            dr = np.nanmean([e['best_delta_mean'] for e in rnd]) \
                if rnd else float('nan')
            dp = np.nanmean([e['best_delta_mean'] for e in par]) \
                if par else float('nan')
            print(f"  {margin:>6.2f} | {fn:>7}/{ft:<9} | "
                  f"{cn:>7}/{ct:<11} | {rel:>11.3f} | "
                  f"{dr:>11.3f} | {dp:>10.3f}")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S58F margin sensitivity "
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
        for margin in MARGINS:
            for topo in SUBSTRATES:
                for arm in ARMS:
                    for s in range(n_seeds):
                        all_args.append((topo, arm, seg_len, margin, s))
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (seg_lens={SEG_LENS}, margins={MARGINS}, "
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
    fieldnames = ['task', 'substrate', 'readout', 'seg_len', 'rel_margin',
                  'seed_idx', 'n_units', 't_total', 'runtime_s',
                  'mean_acc_all', 'switch_block', 'detect_delta',
                  'n_expansions', 'best_delta', 'n_above_margin',
                  'n_above_both', 'seg1_acc', 'seg2_acc', 'seg3_acc',
                  'seg4_acc']
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
        'rel_hold': REL_HOLD, 'warmup_fraction': WARMUP_FRACTION,
        'warmup_floor': WARMUP_FLOOR,
        'margins': MARGINS, 'seg_lens': SEG_LENS,
        'segments': SEGMENTS, 'substrates': SUBSTRATES, 'arms': ARMS,
        'arm_branching': {'rls_auto': 'single OnlineRLS + relative sense',
                          'pop_auto': 'OnlineRLS + s52-v4 memory slots '
                                      '+ relative sense'},
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
