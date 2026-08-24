#!/usr/bin/env python3
"""
S33: M5 state-norm homeostat in the P4 benchmark (Paper D follow-up).
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S33 (Paper D, P4 follow-up): S22's full stack is
                M1+M3+M4 (M5 excluded there: "state dynamics, P3"). The
                audit asked whether M5's state-norm regulation helps on
                the longer, irregular four-domain stream. This script
                adds the P3 M5 controller (E2, verbatim from S21) to the
                SSM-REDEM arm and compares against the committed S22 arm
                on the SAME protocol and seeds.

Arms (10 seeds, paired):
  SSM-REDEM   : M1 + M3 + M4 soft routing (must reproduce S22's
                committed 13.18 stream / 8.93 forgetting - cross-check)
  SSM-REDEM+M5: same + M5: Delta_t = clip(Delta_t + eta*(||h^w||-U), 1,
                dt_max); h_t = A^{Delta_t} h_{t-1} + B e (S21 E2
                verbatim; U=sqrt(128), eta=0.05, dt_max=10)

Metrics identical to S22 (stream/forgetting ppl, T_adapt). M5's expected
value on this task is modest (the fast-channel metadata is already
stationary), so the result is reported honestly either way.

Output files:
  data/s33_ssm_p4_m5_v1.csv
  data/s33_ssm_p4_m5_v1.json

Usage: python s33_ssm_p4_m5.py [--quick] [--sequential]
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
import torch
from multiprocessing import Pool, cpu_count

from s18_llm_drift_gate import (gen_stream, VOCAB, HOLDOUT_LEN,
                                T_ADAPT_WINDOW, STEADY_WINDOW,
                                T_ADAPT_RATIO)
from s19_ssm_rls_readout import (sample_substrate, whiten_scale,
                                 N_STATE, SEED_SCALE, SEED_OFF)
from s20_ssm_m3_routing import fast_mask_of, REF_LEN
from s21_ssm_m4_m5 import make_readout, rls_update, E2_U, E2_ETA, E2_DT_MAX
from s22_ssm_p4_benchmark import (gen_multi_drift_stream, N_DOMAINS,
                                  SHIFTS, BIAS_SETS, refs_fast_multi,
                                  soft_weights)

torch.set_num_threads(1)

N_SEEDS = 10
TAU_M = 500.0
ARMS = ['SSM-REDEM', 'SSM-REDEM+M5']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's33_ssm_p4_m5_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's33_ssm_p4_m5_v1.json')


def run_ssm(args):
    """(arm, seed) -> SSM metrics (S22 protocol, optional M5)."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    t0 = time.time()
    use_m5 = arm.endswith('+M5')

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    fm = fast_mask_of(A)
    stream, domains, switch_times, seg_lens = gen_multi_drift_stream(seed)
    T = stream.shape[0]
    F = N_STATE + 1

    refs = refs_fast_multi(seed, A, B, scale)
    pair_d = [float(torch.norm(refs[i] - refs[j]))
              for i in range(N_DOMAINS) for j in range(i + 1, N_DOMAINS)]
    kappa = 0.5 * float(np.median(pair_d))
    Ws = [make_readout(F) for _ in range(N_DOMAINS)]
    slow = refs[0].clone()
    W, P = Ws[0]

    ce = np.empty(T, dtype=np.float64)
    ce[:] = np.nan
    nneg = 0
    h = torch.zeros(N_STATE, dtype=torch.float64)
    dt = 1.0                     # M5 effective time step (starts at 1)
    dt_hist = np.empty(T)
    dt_hist[:] = np.nan

    for t in range(1, T):
        if use_m5:
            dt = min(max(dt + E2_ETA * (float(torch.norm(h * scale)) - E2_U),
                         1.0), E2_DT_MAX)
            dt_hist[t] = dt
            h = torch.pow(A, dt) * h + B[:, stream[t - 1]]
        else:
            h = A * h + B[:, stream[t - 1]]
        hw = h * scale
        phi = torch.cat([B[:, stream[t - 1]],
                         torch.ones(1, dtype=torch.float64)])
        lam = 1.0 / TAU_M
        slow = (1.0 - lam) * slow + lam * hw[fm]
        dists = [torch.norm(slow - r) for r in refs]
        w = soft_weights(dists, kappa)
        y_hat = sum(w[i] * (Ws[i][0] @ phi) for i in range(N_DOMAINS))
        p_t = float(y_hat[stream[t]].clamp(min=1e-12, max=1.0))
        if float(y_hat[stream[t]]) <= 0.0:
            nneg += 1
        ce[t] = -np.log(p_t)
        target = torch.zeros(VOCAB, dtype=torch.float64)
        target[stream[t]] = 1.0
        for i in range(N_DOMAINS):
            Wi, Pi = Ws[i]
            Ws[i] = rls_update(Wi, Pi, phi, target, float(w[i]))

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    neg_frac = float(nneg) / (T - 1)

    t_adapts = []
    for si, t_s in enumerate(switch_times):
        seg_end = (switch_times[si + 1] if si + 1 < len(switch_times) else T)
        seg_ce = ce[t_s:seg_end]
        if seg_ce.size == 0:
            continue
        steady = float(np.exp(np.mean(seg_ce[-STEADY_WINDOW:])))
        thr = steady * T_ADAPT_RATIO
        cum = np.concatenate([[0.0], seg_ce])
        found = None
        for j in range(T_ADAPT_WINDOW, seg_ce.shape[0] + 1):
            wppl = np.exp((cum[j] - cum[j - T_ADAPT_WINDOW])
                          / T_ADAPT_WINDOW)
            if wppl <= thr:
                found = j
                break
        t_adapts.append(float(found) if found is not None else float('nan'))

    forgets = []
    for si, t_s in enumerate(switch_times):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3, SHIFTS[prev_dom],
                          HOLDOUT_LEN, BIAS_SETS[prev_dom])
        Wf = Ws[prev_dom][0]
        hh = torch.zeros(N_STATE, dtype=torch.float64)
        dth = 1.0
        ces = []
        for t in range(1, HOLDOUT_LEN):
            if use_m5:
                dth = min(max(dth + E2_ETA
                              * (float(torch.norm(hh * scale)) - E2_U),
                              1.0), E2_DT_MAX)
                hh = torch.pow(A, dth) * hh + B[:, hold[t - 1]]
            else:
                hh = A * hh + B[:, hold[t - 1]]
            phi = torch.cat([B[:, hold[t - 1]],
                             torch.ones(1, dtype=torch.float64)])
            p_t = float((Wf @ phi)[hold[t]].clamp(min=1e-12, max=1.0))
            ces.append(-np.log(p_t))
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))

    return {'arm': arm, 'seed': seed, 'stream_ppl': stream_ppl,
            'neg_frac': neg_frac,
            't_adapt_mean': float(np.nanmean(t_adapts))
            if t_adapts else float('nan'),
            'forgetting_ppl': forgetting_ppl,
            'dt_mean': float(np.nanmean(dt_hist)) if use_m5 else float('nan'),
            'dt_max': float(np.nanmax(dt_hist)) if use_m5 else float('nan'),
            'runtime_s': time.time() - t0}


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S33 M5-in-P4 "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    all_args = [(arm, s) for arm in ARMS for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} ({len(ARMS)} arms x{n_seeds} seeds)")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_ssm(a))
            if (i + 1) % max(1, n_runs // 5) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_ssm, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 5) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed', 'stream_ppl', 'neg_frac', 't_adapt_mean',
                  'forgetting_ppl', 'dt_mean', 'dt_max', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 92)
    print("S33 RESULTS (Paper D follow-up): M5 in the P4 benchmark")
    print("=" * 92)
    for arm in ARMS:
        rs = [r for r in results if r['arm'] == arm]
        dtm = [r['dt_mean'] for r in rs if r['dt_mean'] == r['dt_mean']]
        dtx = [r['dt_max'] for r in rs if r['dt_max'] == r['dt_max']]
        dtm_s = f"{np.mean(dtm):5.2f}" if dtm else "  nan"
        dtx_s = f"{np.max(dtx):5.2f}" if dtx else "  nan"
        print(f"  {arm:>12}: stream {np.mean([r['stream_ppl'] for r in rs]):7.3f}"
              f" | forget {np.mean([r['forgetting_ppl'] for r in rs]):7.3f}"
              f" | dt_mean {dtm_s} dt_max {dtx_s}")
    print("-" * 92)
    print("S22 committed: SSM-REDEM stream 13.18, forgetting 8.93 "
          "(cross-check for the SSM-REDEM arm)")

    def paired(metric):
        a = {r['seed']: r for r in results if r['arm'] == ARMS[0]}
        b = {r['seed']: r for r in results if r['arm'] == ARMS[1]}
        ds = [a[s][metric] - b[s][metric] for s in range(n_seeds)
              if s in a and s in b]
        return np.array(ds)

    for m in ['stream_ppl', 'forgetting_ppl']:
        d = paired(m)
        if d.size:
            t_p = d.mean() / (d.std(ddof=1) / np.sqrt(d.size)) \
                if d.std(ddof=1) > 0 else float('nan')
            print(f"  {ARMS[0]} vs {ARMS[1]} [{m}]: {d.mean():+.3f} "
                  f"(negative = M5 improves), n_pos "
                  f"{int(np.sum(d < 0))}/{d.size}, t={t_p:+.2f}")

    params = {
        'experiment': 'M5 state-norm homeostat in the P4 benchmark '
                      '(S22 had M5 excluded)',
        'protocol': 'S22 verbatim (4-domain irregular switches, 10 seeds, '
                    'seed rules); M5 controller verbatim from S21 E2 '
                    '(U=sqrt(128), eta=0.05, dt_max=10)',
        'arms': {'SSM-REDEM': 'M1+M3+M4 (cross-check vs S22)',
                 'SSM-REDEM+M5': 'M1+M3+M4+M5'},
        's22_reference': 'SSM-REDEM stream 13.18, forgetting 8.93',
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'rows': results}, f, indent=2)

    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
