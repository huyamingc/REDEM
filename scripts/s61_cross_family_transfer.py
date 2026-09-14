#!/usr/bin/env python3
"""
Cross-family experience transfer: does a frozen hypothesis help on a
DIFFERENT boundary family, or does the memory correctly refuse to reuse it?
(REDEM G3 = s61)
=============================================================================
Type:           PAPER
Paper Section:  S52/S54 follow-up (the generalization radius of experience)
Experiment:     s52/s54 demonstrated experience reuse under REVISITS of the
                SAME boundary family (linear/circle come back and the frozen
                hypothesis is reactivated). The everyday reading of "similar
                experience" is broader: would a hypothesis transfer ACROSS
                boundary families? s61 tests this with a 5-segment sequence
                [linear, circle, linear, ring, linear] where the circle
                hypothesis is exposed to the RING (a different radial 2D
                boundary: disk vs annulus) and the linear hypothesis to a
                second linear revisit.

Environment: A-B-A-C-A [linear, circle, linear, ring, linear] x 500 blocks
             on BOTH substrates (K=20; ring = 2D annulus (0.30,0.50), s57
             relabel). Arms:
  rls_single : worker RLS only (no memory baseline).
  pop_mem    : s52 v4 functional-identity memory (fixed SNAP_EMA_MIN=0.55).

Question: does the FROZEN circle hypothesis transfer to the ring segment
(seg4)? We measure its accuracy on ring content explicitly (transfer
diagnostic): >0.5 = partial transfer, ~0.5 = no transfer, <0.5 =
anti-correlated (the memory protects against reusing it). Selection is
r_ema-argmax, so a non-transferring slot simply loses selection -- the
worker re-learns and no harm occurs.

Predictions (falsifiable):
  P1 (same-family revisit, s52 replication): pop_mem beats rls_single on the
      seg3 linear revisit (frozen linear slot reactivated).
  P2 (no cross-family transfer on random_graph): the circle slot's accuracy
      on ring content is ~chance or below (disk vs annulus anti-correlate),
      so its selection fraction on seg4 is ~0 and pop_mem seg4 ~= rls_single
      seg4 (no transfer gain, no harm from the foreign segment).
  P3 (memory not poisoned): the seg5 linear revisit still works (the foreign
      ring segment did not corrupt the linear slot) -- pop_mem seg5 >
      rls_single seg5.

Output files:
  data/s61_cross_family_v1.csv     (one row per run)
  data/s61_cross_family_v1.json    (params + per-cell aggregates)

Usage: python s61_cross_family_transfer.py [--quick]
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
SNAP_EMA_MIN = 0.55
SNAP_HOLD = 20
POP_K = 5
AGREE_W = 50
AGREE_THRESH = 0.85

# s57 2D ring (annulus) boundary
RING_IN = 0.30
RING_OUT = 0.50

N_SEGMENTS = 5
SEG_BLOCKS = 500
SEGMENTS = ['linear', 'circle', 'linear', 'ring', 'linear']
SUBSTRATES = ['random_graph_k25', 'parallel']
ARMS = ['rls_single', 'pop_mem']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's61_cross_family_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's61_cross_family_v1.json')

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
    rng = np.random.RandomState(seed)
    dt_parts = []
    tgt_parts = []
    for bd in SEGMENTS:
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


def build_features(states, T):
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    return np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])


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
    else:  # pop_mem (s52 v4)
        worker = OnlineRLS(d, 1, forgetting=RLS_FORGETTING,
                           init_cov=RLS_INIT_COV)
        mem_w = []
        mem_ema = []
        peak_ema = []
        r_ema_w = 0.0
        hold_cnt = 0
        active_kind = 0
        active_idx = 0
        n_snaps = 0
        recent_blk = []
        slot_sel = np.zeros(N_SEGMENTS)   # blocks each segment with memory active
        slot_votes = np.zeros(N_SEGMENTS)  # per-segment memory-vote accuracy
        slot_counts = np.zeros(N_SEGMENTS)
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
                # selection
                scores = [r_ema_w] + list(mem_ema)
                best = int(np.argmax(scores))
                seg_idx = min(b // SEG_BLOCKS, N_SEGMENTS - 1)
                if best == 0:
                    active_kind = 0
                    preds[t - DB_K_PULSES + 1:t + 1] = wv
                else:
                    active_kind = 1
                    active_idx = best - 1
                    mv = float(mem_w[active_idx] @ x_blk)
                    preds[t - DB_K_PULSES + 1:t + 1] = \
                        float(mem_w[active_idx] @ x_blk)
                    slot_sel[seg_idx] += 1
                    slot_votes[seg_idx] += 1.0 if (mv > 0.5) == target[t] \
                        else 0.0
                    slot_counts[seg_idx] += 1

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    seg_acc = [acc_median_in(acc_run, s * SEG_BLOCKS,
                             (s + 1) * SEG_BLOCKS) for s in range(N_SEGMENTS)]

    res = {'task': 'cross_family', 'substrate': topo_name, 'readout': arm,
           'seed_idx': seed_idx, 'n_units': N_UNITS, 't_total': int(T),
           'runtime_s': time.time() - t0,
           'mean_acc_all': float(np.nanmean(acc_run)),
           'n_snaps': n_snaps if arm == 'pop_mem' else -1}
    for s in range(N_SEGMENTS):
        res[f'seg{s + 1}_acc'] = seg_acc[s]
        if arm == 'pop_mem':
            res[f'mem_sel{s + 1}'] = float(slot_sel[s])
            res[f'mem_acc{s + 1}'] = float(slot_votes[s] / slot_counts[s]) \
                if slot_counts[s] > 0 else float('nan')
    return res


# ========================== Aggregation ==========================

def agg_list(rs, field):
    vals = np.array([r[field] for r in rs], dtype=float)
    return float(np.nanmean(vals)) if vals.size else float('nan')


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['substrate'], r['readout']), []).append(r)
    agg = []
    for key, rs in sorted(groups.items()):
        topo, read = key
        entry = {'substrate': topo, 'readout': read, 'n_runs': len(rs)}
        fields = ['mean_acc_all', 'n_snaps'] + \
            [f'seg{s}_acc' for s in range(1, 6)]
        if read == 'pop_mem':
            fields += ['mem_sel%d' % s for s in range(1, 6)] + \
                      ['mem_acc%d' % s for s in range(1, 6)]
        for f in fields:
            entry[f + '_mean'] = agg_list(rs, f)
        agg.append(entry)
    return agg


def print_table(agg):
    print("\n" + "=" * 130)
    print("S61 CROSS-FAMILY TRANSFER (A-B-A-C-A: linear,circle,linear,ring,linear)")
    print("=" * 130)
    for topo in SUBSTRATES:
        print(f"\n--- {topo} ---")
        print(f"  {'readout':<10} | {'s1 lin':>6} | {'s2 cir':>6} | "
              f"{'s3 lin':>6} | {'s4 ring':>6} | {'s5 lin':>6} | "
              f"{'mean':>6} | {'snaps':>5}")
        for a in [x for x in agg if x['substrate'] == topo]:
            print(f"  {a['readout']:<10} | "
                  + " | ".join(f"{a['seg%d_acc_mean' % s]:>6.3f}"
                               for s in range(1, 6))
                  + f" | {a['mean_acc_all_mean']:>6.3f} | "
                  + (f"{a['n_snaps_mean']:>5.1f}" if a['readout'] == 'pop_mem'
                     else '  n/a'))
        # memory diagnostics on pop_mem
        pm = [x for x in agg if x['substrate'] == topo
              and x['readout'] == 'pop_mem'][0]
        print(f"  [pop_mem diagnostics: seg4 ring: mem_acc="
              f"{pm['mem_acc4_mean']:.3f} (frozen-slot accuracy on ring), "
              f"mem_sel={pm['mem_sel4_mean']:.0f} blocks; "
              f"seg5 linear revisit: mem_sel={pm['mem_sel5_mean']:.0f} "
              f"blocks]")


# ========================== Main ==========================

def run_sweep(quick=False):
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S61 cross-family transfer "
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
                print(f"[{time.strftime('%H:%M:%S')}] progress "
                      f"{done}/{n_runs}", flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'substrate', 'readout', 'seed_idx', 'n_units',
                  't_total', 'runtime_s', 'mean_acc_all', 'n_snaps']
    for s in range(1, 6):
        fieldnames += [f'seg{s}_acc', f'mem_sel{s}', f'mem_acc{s}']
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
        'ring_in': RING_IN, 'ring_out': RING_OUT,
        'segments': SEGMENTS, 'seg_blocks': SEG_BLOCKS,
        'substrates': SUBSTRATES, 'arms': ARMS,
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
