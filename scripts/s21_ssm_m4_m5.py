#!/usr/bin/env python3
"""
S21: REDEM-SSM P3 - M4 gentle routing + M5 state-norm homeostat.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S21 (Paper D P3): two mechanism tests on the diagonal-SSM
                host, both within the RLS-only constraint (no gradient):

E1 (M4 "gentle wins"): P2 showed A3 routing (abrupt specialist switch)
improves forgetting but PAYS a stream cost (+0.38..+4.08 ppl - the
specialization-vs-lag tradeoff). E1 tests whether SOFT routing - a
continuous softmax weighting of the two specialists by their EMA
distances (no hysteresis flip) - smooths the transition and reduces the
stream penalty while keeping the forgetting gain:
  A1        : bare single RLS readout (replication baseline)
  A3-abrupt : P2's routing (hysteresis flip, P2 replication)
  A3-soft   : softmax-distance weighted routing (gentle)
tau_m = 500 (P2's sweet spot). Task/seed rules/metrics verbatim from s18.

E2 (M5 state-norm homeostat): P2 found the FULL whitened-state EMA is a
non-stationary domain statistic (slow channels, tau up to 3000, accumulate
over the stream; the detector never flips - fixed by masking to fast
channels). E2 tests the homeostat alternative: a closed-loop controller
monitors the whitened-state norm and adjusts the effective time step
(Delta_t) to hold the norm in a target band around sqrt(N) - the
whitening's unit-variance intent:
  h_t = A^{Delta_t} h_{t-1} + B e_{t-1}
  Delta_t = clip(Delta_{t-1} + eta (||h_w|| - U), 1, Delta_max)
Arms: BARE vs REG, clean vs disturbed (a state spike injected at the
first switch). Metrics (mechanism-level, honest scope): full-state-EMA
detector flips (5/5 known switches), norm mean/max/final, recovery tokens
to return to the band after the spike, Delta_t activity.

Predictions (Paper D, P3/P4):
  P3: gradual M4 beats abrupt on stream ppl (soft < abrupt, n>=6/10),
      forgetting not degraded.
  P4: M5 regulation keeps the state norm in band (and restores the
      full-state EMA detector to 5/5); bare drifts out of band and does
      not recover from the disturbance.
Falsified iff the predicted direction reverses (reported honestly).

Output files:
  data/s21_ssm_m4_m5_v1.csv
  data/s21_ssm_m4_m5_v1.json

Usage: python s21_ssm_m4_m5.py [--quick] [--sequential]
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

from s19_ssm_rls_readout import (gen_drift_stream, gen_stream,
                                 sample_substrate, whiten_scale,
                                 VOCAB, N_STATE, SEG_LEN, N_SEGMENTS,
                                 HOLDOUT_LEN, BIAS_A, BIAS_B,
                                 RLS_LAMBDA, RLS_DELTA, SEED_SCALE,
                                 SEED_OFF)
from s20_ssm_m3_routing import (fast_mask_of, gate_estimate,
                                GATE_MARGIN, T_ADAPT_WINDOW, STEADY_WINDOW,
                                T_ADAPT_RATIO)

torch.set_num_threads(1)   # per-worker; Pool gives process-level parallelism

# ========================== Fixed parameters ==========================
TAU_M = 500.0              # E1 metadata timescale (P2 sweet spot)
N_SEEDS = 10
REF_LEN = 1500

# E2 homeostat
E2_U = float(np.sqrt(N_STATE))     # target whitened-state norm (~sqrt(N))
E2_ETA = 0.05
E2_DT_MAX = 10.0
E2_BAND = 1.05             # recovery threshold: norm <= U*BAND after spike
E2_SPIKE = 10.0            # disturbance magnitude at the first switch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's21_ssm_m4_m5_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's21_ssm_m4_m5_v1.json')


# ========================== E1: readout machinery (s20 reuse) ==========================

def make_readout(F):
    W = torch.zeros(VOCAB, F, dtype=torch.float64)
    W[:, -1] = 1.0 / VOCAB
    P = torch.eye(F, dtype=torch.float64) / RLS_DELTA
    return W, P


def rls_update(W, P, phi, target, err_scale):
    e = err_scale * (target - (W @ phi))
    g = P @ phi
    k = g / (RLS_LAMBDA + float(phi @ g))
    W = W + torch.outer(e, k)
    P = (P - torch.outer(k, phi) @ P) / RLS_LAMBDA
    return W, P


def refs_fast(seed, A, B, scale):
    """Fast-channel metadata references (P2 host)."""
    fm = fast_mask_of(A)
    refs = []
    for d in range(2):
        rs = gen_stream(seed * 31 + d * 131 + 7, 1 if d == 0 else 7,
                        REF_LEN, BIAS_A if d == 0 else BIAS_B)
        h = torch.zeros(N_STATE, dtype=torch.float64)
        acc = torch.zeros(N_STATE, dtype=torch.float64)
        for t in range(REF_LEN):
            h = A * h + B[:, rs[t]]
            acc = acc + h * scale
        refs.append((acc / REF_LEN)[fm])
    return refs


def run_e1(args):
    """(arm, tau_m, seed) -> E1 metrics. Top-level; unbuffered."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, tau_m, seed = args
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    fm = fast_mask_of(A)
    stream, domains = gen_drift_stream(seed)
    T = stream.shape[0]
    F = N_STATE + 1

    refs = refs_fast(seed, A, B, scale)
    sep = float(torch.norm(refs[0] - refs[1]))
    kappa = 0.5 * max(sep, 1e-6)      # softmax temperature (gentle scale)

    if arm == 'A1':
        W, P = make_readout(F)
        W0 = W1 = P0 = P1 = None
    else:
        W0, P0 = make_readout(F)
        W1, P1 = make_readout(F)
        W = P = None

    slow = refs[0].clone()
    prev_est = 0
    last_switch_t = -10 ** 9

    ce = np.empty(T, dtype=np.float64)
    ce[:] = np.nan
    nneg = 0
    h = torch.zeros(N_STATE, dtype=torch.float64)

    for t in range(1, T):
        h = A * h + B[:, stream[t - 1]]
        hw = h * scale

        if arm == 'A1':
            y_hat = (W @ torch.cat([B[:, stream[t - 1]],
                                    torch.ones(1, dtype=torch.float64)]))
            p_t = float(y_hat[stream[t]].clamp(min=1e-12, max=1.0))
            if float(y_hat[stream[t]]) <= 0.0:
                nneg += 1
            ce[t] = -np.log(p_t)
            target = torch.zeros(VOCAB, dtype=torch.float64)
            target[stream[t]] = 1.0
            W, P = rls_update(W, P,
                              torch.cat([B[:, stream[t - 1]],
                                         torch.ones(1,
                                                    dtype=torch.float64)]),
                              target, 1.0)
        else:
            lam = 1.0 / tau_m
            slow = (1.0 - lam) * slow + lam * hw[fm]
            if arm == 'A3-abrupt':
                est = gate_estimate(slow, refs, prev_est, GATE_MARGIN)
                if est != prev_est:
                    last_switch_t = t
                    prev_est = est
                w0 = 1.0 if est == 0 else 0.0
                w1 = 1.0 - w0
            else:  # A3-soft: continuous softmax weights (no flip)
                d0 = float(torch.norm(slow - refs[0]))
                d1 = float(torch.norm(slow - refs[1]))
                w1 = 1.0 / (1.0 + np.exp((d1 - d0) / kappa))
                w0 = 1.0 - w1

            phi = torch.cat([B[:, stream[t - 1]],
                             torch.ones(1, dtype=torch.float64)])
            y_hat = w0 * (W0 @ phi) + w1 * (W1 @ phi)
            p_t = float(y_hat[stream[t]].clamp(min=1e-12, max=1.0))
            if float(y_hat[stream[t]]) <= 0.0:
                nneg += 1
            ce[t] = -np.log(p_t)
            target = torch.zeros(VOCAB, dtype=torch.float64)
            target[stream[t]] = 1.0
            W0, P0 = rls_update(W0, P0, phi, target, w0)
            W1, P1 = rls_update(W1, P1, phi, target, w1)

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    neg_frac = float(nneg) / (T - 1)

    # T_adapt (s18 protocol, known switches)
    t_adapts = []
    switch_times = [SEG_LEN * s for s in range(1, N_SEGMENTS)]
    for si, t_s in enumerate(switch_times):
        seg_end = SEG_LEN * (si + 2) if si + 2 <= N_SEGMENTS else T
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

    # Forgetting: domain-matched specialist (A3) or single readout (A1)
    forgets = []
    for si, t_s in enumerate(switch_times):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3,
                          1 if prev_dom == 0 else 7, HOLDOUT_LEN,
                          BIAS_A if prev_dom == 0 else BIAS_B)
        hh = torch.zeros(N_STATE, dtype=torch.float64)
        if arm == 'A1':
            Wf = W
        else:
            Wf = W0 if prev_dom == 0 else W1
        ces = []
        for t in range(1, HOLDOUT_LEN):
            hh = A * hh + B[:, hold[t - 1]]
            phi = torch.cat([B[:, hold[t - 1]],
                             torch.ones(1, dtype=torch.float64)])
            p_t = float((Wf @ phi)[hold[t]].clamp(min=1e-12, max=1.0))
            ces.append(-np.log(p_t))
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))

    return {'exp': 'E1', 'arm': arm, 'tau_m': float(tau_m), 'seed': seed,
            'stream_ppl': stream_ppl, 'neg_frac': neg_frac,
            't_adapt_mean': float(np.nanmean(t_adapts))
            if t_adapts else float('nan'),
            'forgetting_ppl': forgetting_ppl,
            'runtime_s': time.time() - t0}


# ========================== E2: M5 state-norm homeostat ==========================

def refs_full_reg(seed, A, B, scale, regulated):
    """Full-state metadata references, optionally under regulation."""
    refs = []
    for d in range(2):
        rs = gen_stream(seed * 31 + d * 131 + 7, 1 if d == 0 else 7,
                        REF_LEN, BIAS_A if d == 0 else BIAS_B)
        h = torch.zeros(N_STATE, dtype=torch.float64)
        acc = torch.zeros(N_STATE, dtype=torch.float64)
        dt = 1.0
        for t in range(REF_LEN):
            if regulated:
                h = torch.pow(A, dt) * h + B[:, rs[t]]
            else:
                h = A * h + B[:, rs[t]]
            hw = h * scale
            acc = acc + hw
            if regulated:
                n = float(torch.norm(hw))
                dt = min(max(dt + E2_ETA * (n - E2_U), 1.0), E2_DT_MAX)
        refs.append(acc / REF_LEN)
    return refs


def run_e2(args):
    """(reg, dist, seed) -> E2 metrics. Mechanism-level, honest scope."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    reg, dist, seed = args
    arm = f"{'REG' if reg else 'BARE'}-{'dist' if dist else 'clean'}"
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    stream, domains = gen_drift_stream(seed)
    T = stream.shape[0]

    refs = refs_full_reg(seed, A, B, scale, reg)
    slow = refs[0].clone()
    prev_est = 0
    flips = 0
    h = torch.zeros(N_STATE, dtype=torch.float64)
    dt = 1.0
    norms = np.empty(T, dtype=np.float64)
    dts = []
    spike_t = SEG_LEN
    recovery = None
    rng = np.random.RandomState(seed * 17 + 5)
    u = torch.from_numpy(rng.randn(N_STATE) / np.sqrt(N_STATE))

    for t in range(1, T):
        if reg:
            h = torch.pow(A, dt) * h + B[:, stream[t - 1]]
        else:
            h = A * h + B[:, stream[t - 1]]
        if dist and t == spike_t:
            h = h + E2_SPIKE * u
        hw = h * scale
        norms[t] = float(torch.norm(hw))
        if reg:
            dt = min(max(dt + E2_ETA * (norms[t] - E2_U), 1.0), E2_DT_MAX)
            dts.append(dt)
        # full-state EMA detector
        lam = 1.0 / TAU_M
        slow = (1.0 - lam) * slow + lam * hw
        est = gate_estimate(slow, refs, prev_est, GATE_MARGIN)
        if est != prev_est:
            flips += 1
            prev_est = est
        # recovery: first t after the spike with norm back in band
        if dist and recovery is None and t > spike_t and \
                norms[t] <= E2_U * E2_BAND:
            recovery = t - spike_t

    norm_mean = float(np.nanmean(norms[1:]))
    norm_max = float(np.nanmax(norms[1:]))
    norm_final = float(norms[-1])
    return {'exp': 'E2', 'arm': arm, 'reg': bool(reg), 'dist': bool(dist),
            'seed': seed, 'flips': int(flips), 'norm_mean': norm_mean,
            'norm_max': norm_max, 'norm_final': norm_final,
            'recovery_tokens': float(recovery) if recovery is not None
            else float('nan'),
            'dt_mean': float(np.mean(dts)) if dts else float('nan'),
            'dt_max': float(np.max(dts)) if dts else float('nan'),
            'runtime_s': time.time() - t0}


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['exp'], r['arm']), []).append(r)
    agg = []
    for (exp, arm), rs in sorted(groups.items()):
        entry = {'exp': exp, 'arm': arm, 'n_runs': len(rs)}
        for m in rs[0]:
            if m in ('exp', 'arm', 'seed', 'runtime_s', 'reg', 'dist',
                     'tau_m'):
                continue
            v = np.array([r[m] for r in rs], dtype=float)
            v = v[~np.isnan(v)]
            entry[m + '_mean'] = float(np.mean(v)) if v.size else float('nan')
        if 'flips_mean' in entry:
            entry['flips_mean'] = float(np.mean([r['flips'] for r in rs]))
        agg.append(entry)
    return agg


def print_table(results):
    print("\n" + "=" * 110)
    print("S21 RESULTS (Paper D P3): M4 gentle routing (E1) + M5 "
          "state-norm homeostat (E2), 10 seeds")
    print("=" * 110)
    e1 = [r for r in results if r['exp'] == 'E1']
    e2 = [r for r in results if r['exp'] == 'E2']
    print("\nE1 (M4, tau_m=500):")
    print(f" {'arm':>10} | {'stream':>9} {'d_ppl':>9} {'n_imp':>5} | "
          f"{'forget':>9} {'d_forget':>9} {'n_imp':>5} | {'T_adapt':>8} "
          f"{'neg%':>5}")
    base = {r['seed']: r for r in e1 if r['arm'] == 'A1'}
    for arm in ['A1', 'A3-abrupt', 'A3-soft']:
        rs = [r for r in e1 if r['arm'] == arm]
        if not rs:
            continue
        ms = float(np.mean([r['stream_ppl'] for r in rs]))
        mf = float(np.mean([r['forgetting_ppl'] for r in rs]))
        if arm == 'A1':
            print(f" {arm:>10} | {ms:>9.3f} {'-':>9} {'-':>5} | "
                  f"{mf:>9.3f} {'-':>9} {'-':>5} | "
                  f"{np.nanmean([r['t_adapt_mean'] for r in rs]):>8.0f} "
                  f"{np.mean([r['neg_frac'] for r in rs]) * 100:>5.1f}")
            continue
        ds = np.array([r['stream_ppl'] - base[r['seed']]['stream_ppl']
                       for r in rs])
        df = np.array([r['forgetting_ppl']
                       - base[r['seed']]['forgetting_ppl'] for r in rs])
        print(f" {arm:>10} | {ms:>9.3f} {ds.mean():>+9.3f} "
              f"{int(np.sum(ds < 0)):>5}/10 | {mf:>9.3f} "
              f"{df.mean():>+9.3f} {int(np.sum(df < 0)):>5}/10 | "
              f"{np.nanmean([r['t_adapt_mean'] for r in rs]):>8.0f} "
              f"{np.mean([r['neg_frac'] for r in rs]) * 100:>5.1f}")
    # soft vs abrupt (the P3 prediction)
    sa = {r['seed']: r for r in e1 if r['arm'] == 'A3-abrupt'}
    so = {r['seed']: r for r in e1 if r['arm'] == 'A3-soft'}
    if sa and so:
        d = np.array([so[s]['stream_ppl'] - sa[s]['stream_ppl']
                      for s in so if s in sa])
        print(f" soft vs abrupt stream: {d.mean():+.3f}, soft better "
              f"{int(np.sum(d < 0))}/10")
        d = np.array([so[s]['forgetting_ppl'] - sa[s]['forgetting_ppl']
                      for s in so if s in sa])
        print(f" soft vs abrupt forget: {d.mean():+.3f}, soft better "
              f"{int(np.sum(d < 0))}/10")

    print("\nE2 (M5, full-state EMA detector; U=sqrt(N), eta=0.05, "
          "dt_max=10):")
    print(f" {'arm':>12} | {'flips':>6} | {'norm_mean':>9} "
          f"{'norm_max':>9} {'norm_final':>10} | {'recovery':>9} "
          f"{'dt_mean':>8} {'dt_max':>7}")
    for arm in ['BARE-clean', 'BARE-dist', 'REG-clean', 'REG-dist']:
        rs = [r for r in e2 if r['arm'] == arm]
        if not rs:
            continue
        print(f" {arm:>12} | "
              f"{np.mean([r['flips'] for r in rs]):>6.2f} | "
              f"{np.mean([r['norm_mean'] for r in rs]):>9.1f} "
              f"{np.mean([r['norm_max'] for r in rs]):>9.1f} "
              f"{np.mean([r['norm_final'] for r in rs]):>10.1f} | "
              f"{np.nanmean([r['recovery_tokens'] for r in rs]):>9.0f} "
              f"{np.mean([r['dt_mean'] for r in rs]):>8.2f} "
              f"{np.mean([r['dt_max'] for r in rs]):>7.1f}")


def dispatch(task):
    """Top-level Pool worker: (exp, args) -> run_e1 or run_e2."""
    exp, a = task
    return run_e1(a) if exp == 'E1' else run_e2(a)


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S21 M4+M5 (quick={quick}, "
          f"sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    e1_args = [(arm, TAU_M, s) for arm in ['A1', 'A3-abrupt', 'A3-soft']
               for s in range(n_seeds)]
    e2_args = [(reg, dist, s) for reg in [False, True]
               for dist in [False, True] for s in range(n_seeds)]
    all_args = [('E1', a) for a in e1_args] + [('E2', a) for a in e2_args]
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (E1 {len(e1_args)}, E2 {len(e2_args)})")

    results = []
    if sequential:
        for i, (exp, a) in enumerate(all_args):
            results.append(run_e1(a) if exp == 'E1' else run_e2(a))
            if (i + 1) % max(1, n_runs // 10) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(dispatch, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 10) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    print_table(results)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['exp', 'arm', 'tau_m', 'seed', 'stream_ppl', 'neg_frac',
                  't_adapt_mean', 'forgetting_ppl', 'flips', 'norm_mean',
                  'norm_max', 'norm_final', 'recovery_tokens', 'dt_mean',
                  'dt_max', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    params = {
        'host': 'diagonal linear SSM (N=128), CV spectrum, whitened; '
                'readout = RLS on [B e; 1] (P1/P3a host)',
        'e1_m4': {'tau_m': TAU_M,
                  'arms': {'A1': 'bare single RLS readout',
                           'A3-abrupt': 'P2 routing replication '
                                        '(hysteresis flip)',
                           'A3-soft': 'softmax-distance weighted routing, '
                                      'kappa = 0.5*||ref0-ref1|| (gentle)'},
                  'prediction': 'soft < abrupt on stream ppl, forgetting '
                                'not degraded (P3: gradual beats abrupt)'},
        'e2_m5': {'controller': 'Delta_t = clip(Delta_t + eta*(||h_w||-U), '
                                '1, dt_max); h_t = A^{Delta_t} h_{t-1} + B e',
                  'U': E2_U, 'eta': E2_ETA, 'dt_max': E2_DT_MAX,
                  'band': E2_BAND, 'spike': E2_SPIKE,
                  'prediction': 'REG keeps norm in band and restores the '
                                'full-state EMA detector (5/5); BARE drifts '
                                '(P4)'},
        'task': 'identical to s18: two biased-bigram generators, known '
                'switches, 6x3000; seed rules verbatim',
        'metrics': {'E1': 'stream ppl / forgetting ppl / T_adapt / neg_frac '
                         '(s18 protocol)',
                    'E2': 'full-state EMA detector flips, norm mean/max/final, '
                          'recovery tokens, dt activity (mechanism-level)'},
        'discipline': '10 seeds, paired sign consistency',
        'env': {'torch': torch.__version__, 'numpy': np.__version__,
                'cpu': 'torch CPU only, no mamba-ssm dependency'},
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': aggregate(results)}, f,
                  indent=2)

    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
