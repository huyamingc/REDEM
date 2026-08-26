#!/usr/bin/env python3
"""
S20: REDEM-SSM P2 - M3 metadata EMA + drift detection + routing on the
diagonal SSM host.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S20 (Paper D P2): does the M3 slow-trace metadata transfer
                to the diagonal-SSM host (host-invariance of the Paper C
                Section 6 policy lesson: routing transfers, gating does
                not)?

Host (per P1/P3a evidence, S19): diagonal linear SSM (N=128), CV spectrum
(log-uniform tau in [1,3000]), spectrum-whitened state h_w. The readout
uses the ADDITIVE CURRENT-TOKEN INPUT PATH (phi = [B e_{t-1}; 1]) - the
P1/P3a result that the state alone is not a readout substrate. The state's
role here is the M3 METADATA: a per-token EMA second state
m_t = (1-1/tau_m) m_{t-1} + (1/tau_m) h_w,t[fast] - the FAST-CHANNEL
(tau <= 8) whitened state only - estimates the current domain (nearest
fixed per-domain reference with hysteresis margin), selecting WHICH
readout is active (A3 routing) or WHEN updates happen (A2 gating).

Metadata feature note: the FULL whitened-state EMA is NOT a stationary
domain statistic - the slow channels (tau up to 3000) accumulate over the
whole stream, so the EMA drifts away from any fixed per-domain reference
(the state-EMA detector never flips). The
fast channels (tau <= 8) converge in a few tokens, are stationary, and
their stationary mean carries the domain marginal (separation
||ref0-ref1|| ~ 1.2 vs ~0.07 for the input projection alone).

Arms (s18 protocol verbatim, adapted to RLS readouts):
  A1 bare       : single RLS readout on [B e; 1], updates every token
  A2 gate-only  : same readout; slow-trace estimate; RLS error scaled 1.0
                  for tau_m tokens after a DETECTED switch, 0.10 within a
                  domain (the "when to adapt" claim)
  A3 routing    : TWO RLS readouts (one per domain); the slow-trace
                  estimate selects the active one for prediction AND
                  update; inactive readout is fully frozen (W and P)
                  (the "which readout to adapt" claim)

Task/seed rules/metrics are verbatim from s18 (Paper C Sec 6): two
biased-bigram Markov generators (A: shift=1 bias {0..7}; B: shift=7 bias
{24..31}), known switches, 6 x 3000 tokens. Paired per-seed comparison vs
A1 with sign consistency (10 seeds). CE metric: linear CE on the MMSE
readout (CE = -ln(clip(y_hat[target], eps, 1)), no softmax), with
neg_frac reported as diagnostic.

P2 predictions (Paper D, P2):
  - routing (A3) improves forgetting on the SSM host at every tau_m
    (like s18 A3: 9-10/10 seeds);
  - gating-only (A2) is falsified on the SSM host (like s18 A2: stream ppl
    worse than A1 at every tau_m, 0/10 improved).
Falsified iff routing forgetting diffs are never negative (0/10 improved
at some tau_m) OR gating stream diffs are never negative (0/10 improved at
every tau_m). Reported honestly either way.

Output files:
  data/s20_ssm_m3_routing_v1.csv
  data/s20_ssm_m3_routing_v1.json

Usage: python s20_ssm_m3_routing.py [--quick] [--sequential]
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

torch.set_num_threads(1)   # per-worker; Pool gives process-level parallelism

# ========================== Fixed parameters ==========================
TAU_M_LIST = [200.0, 500.0, 1000.0, 2000.0]
N_SEEDS = 10
REF_LEN = 1500              # tokens per domain for the metadata references
GATE_MARGIN = 1.15          # hysteresis band (s18)
GATE_LOW_FRAC = 0.10        # in-domain RLS error scale for A2 (s18)

T_ADAPT_WINDOW = 20
STEADY_WINDOW = 400
T_ADAPT_RATIO = 1.5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's20_ssm_m3_routing_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's20_ssm_m3_routing_v1.json')
S19_CSV = os.path.join(DATA_DIR, 's19_ssm_rls_readout_v1.csv')

ARMS = ['A1', 'A2', 'A3']
SPECTRUM = 'CV'             # log-uniform tau in [1,3000] (the P1/P3a host)


# ========================== M3 metadata on the SSM host ==========================

def fast_mask_of(A):
    """Fast channels (tau <= 8 tokens): stationary, bounded state channels
    carrying the recent-token statistics (the M3 metadata feature)."""
    tau = -1.0 / torch.log(A)
    return tau <= 8.0


def reference_features(seed, A, B, scale):
    """Fixed per-domain metadata references: mean FAST-CHANNEL whitened
    state over a reference stream per domain (s18 seed rule: seed*31 +
    d*131 + 7). The fast channels converge in a few tokens, so the mean
    over 1500 reference tokens is the stationary domain level."""
    fm = fast_mask_of(A)
    refs = []
    for d in range(2):
        ref_stream = gen_stream(seed * 31 + d * 131 + 7,
                                1 if d == 0 else 7, REF_LEN,
                                BIAS_A if d == 0 else BIAS_B)
        h = torch.zeros(N_STATE, dtype=torch.float64)
        acc = torch.zeros(N_STATE, dtype=torch.float64)
        for t in range(REF_LEN):
            h = A * h + B[:, ref_stream[t]]
            acc = acc + (h * scale)
        refs.append((acc / REF_LEN)[fm])
    return refs


def gate_estimate(slow, refs, prev=0, margin=GATE_MARGIN):
    """Domain 0/1 by nearest reference with a hysteresis band (s18): flip
    only when one distance is clearly (< 1/margin) smaller."""
    d0 = float(torch.norm(slow - refs[0]))
    d1 = float(torch.norm(slow - refs[1]))
    if d0 * margin < d1:
        return 0
    if d1 * margin < d0:
        return 1
    return prev


# ========================== Per-run experiment ==========================

def make_readout(F):
    """Fresh RLS readout: uniform-prior bias init."""
    W = torch.zeros(VOCAB, F, dtype=torch.float64)
    W[:, -1] = 1.0 / VOCAB
    P = torch.eye(F, dtype=torch.float64) / RLS_DELTA
    return W, P


def rls_update(W, P, phi, target, err_scale):
    """One RLS step (predict-before-update callers already computed y_hat).
    err_scale implements A2's update gating (1.0 post-switch, low within)."""
    e = err_scale * (target - (W @ phi))
    g = P @ phi
    k = g / (RLS_LAMBDA + float(phi @ g))
    W = W + torch.outer(e, k)
    P = (P - torch.outer(k, phi) @ P) / RLS_LAMBDA
    return W, P


def run_single(args):
    """(arm, tau_m, seed_idx) -> metrics dict. Top-level; unbuffered."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, tau_m, seed = args
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, SPECTRUM)
    scale = whiten_scale(A)
    fm = fast_mask_of(A)
    stream, domains = gen_drift_stream(seed)
    T = stream.shape[0]
    F = N_STATE + 1

    refs = reference_features(seed, A, B, scale)

    if arm == 'A3':
        W0, P0 = make_readout(F)
        W1, P1 = make_readout(F)
    else:
        W, P = make_readout(F)

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

        if arm in ('A2', 'A3'):
            lam = 1.0 / tau_m
            slow = (1.0 - lam) * slow + lam * hw[fm]
            est = gate_estimate(slow, refs, prev_est)
            if est != prev_est:
                last_switch_t = t
                prev_est = est
        else:
            est = 0

        phi = torch.cat([B[:, stream[t - 1]],
                         torch.ones(1, dtype=torch.float64)])
        if arm == 'A3':
            Wa, Pa = (W0, P0) if est == 0 else (W1, P1)
        else:
            Wa, Pa = W, P
        y_hat = Wa @ phi
        p_target = float(y_hat[stream[t]].clamp(min=1e-12, max=1.0))
        if float(y_hat[stream[t]]) <= 0.0:
            nneg += 1
        ce[t] = -np.log(p_target)

        target = torch.zeros(VOCAB, dtype=torch.float64)
        target[stream[t]] = 1.0
        if arm == 'A2':
            scale_err = 1.0 if (t - last_switch_t) < tau_m else GATE_LOW_FRAC
        else:
            scale_err = 1.0
        if arm == 'A3':
            if est == 0:
                W0, P0 = rls_update(W0, P0, phi, target, 1.0)
            else:
                W1, P1 = rls_update(W1, P1, phi, target, 1.0)
        else:
            W, P = rls_update(W, P, phi, target, scale_err)

    # ---- Metrics (s18 definitions) ----
    valid = ce[1:]
    stream_ppl = float(np.exp(np.nanmean(valid)))
    neg_frac = float(nneg) / (T - 1)

    # T_adapt: post-switch running-window ppl (known switches)
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

    # Forgetting: held-out previous domain, current readout(s). A3 uses the
    # domain-matched specialist (s18 semantics).
    forgets = []
    for si, t_s in enumerate(switch_times):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3,
                          1 if prev_dom == 0 else 7, HOLDOUT_LEN,
                          BIAS_A if prev_dom == 0 else BIAS_B)
        hh = torch.zeros(N_STATE, dtype=torch.float64)
        if arm == 'A3':
            Wf, _ = (W0, P0) if prev_dom == 0 else (W1, P1)
        else:
            Wf = W
        ces = []
        for t in range(1, HOLDOUT_LEN):
            hh = A * hh + B[:, hold[t - 1]]
            phi = torch.cat([B[:, hold[t - 1]],
                             torch.ones(1, dtype=torch.float64)])
            p_target = float((Wf @ phi)[hold[t]].clamp(min=1e-12, max=1.0))
            ces.append(-np.log(p_target))
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))

    return {'arm': arm, 'tau_m': float(tau_m), 'seed': seed,
            'stream_ppl': stream_ppl, 'neg_frac': neg_frac,
            't_adapt_mean': float(np.nanmean(t_adapts))
            if t_adapts else float('nan'),
            'forgetting_ppl': forgetting_ppl,
            'runtime_s': time.time() - t0}


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['arm'], r['tau_m']), []).append(r)
    agg = []
    for (arm, tm), rs in sorted(groups.items()):
        entry = {'arm': arm, 'tau_m': tm, 'n_runs': len(rs)}
        for m in ['stream_ppl', 't_adapt_mean', 'forgetting_ppl']:
            v = np.array([r[m] for r in rs], dtype=float)
            v = v[~np.isnan(v)]
            entry[m + '_mean'] = float(np.mean(v)) if v.size else float('nan')
            entry[m + '_std'] = float(np.std(v)) if v.size else float('nan')
        entry['neg_frac_mean'] = float(np.mean(
            [r['neg_frac'] for r in rs]))
        agg.append(entry)
    return agg


def paired_diff(results, arm, tm, metric, base='A1'):
    d = []
    for s in range(N_SEEDS):
        a = next((r[metric] for r in results
                  if r['arm'] == arm and r['tau_m'] == tm and r['seed'] == s),
                 None)
        b = next((r[metric] for r in results
                  if r['arm'] == base and r['tau_m'] == 0.0
                  and r['seed'] == s), None)
        if a is not None and b is not None:
            d.append(float(a) - float(b))
    return np.array(d)


def print_table(agg, results, s19_ref):
    print("\n" + "=" * 120)
    print("S20 RESULTS (Paper D P2): M3 metadata EMA + routing on the "
          "diagonal SSM host (10 seeds)")
    print("=" * 120)
    print(f" {'arm':>3} {'tau_m':>6} | {'stream':>9} {'d_ppl':>9} "
          f"{'n_imp':>5} | {'forget':>9} {'d_forget':>9} {'n_imp':>5} | "
          f"{'T_adapt':>8} {'neg%':>5} | {'oracle-ref':>10}")
    for a in sorted(agg, key=lambda x: (x['arm'], x['tau_m'])):
        if a['arm'] == 'A1':
            print(f" {a['arm']:>3} {'-':>6} | {a['stream_ppl_mean']:>9.3f} "
                  f"{'-':>9} {'-':>5} | {a['forgetting_ppl_mean']:>9.3f} "
                  f"{'-':>9} {'-':>5} | {a['t_adapt_mean_mean']:>8.0f} "
                  f"{a['neg_frac_mean'] * 100:>5.1f} | "
                  f"{s19_ref['stream']:>10.3f}")
            continue
        dp = paired_diff(results, a['arm'], a['tau_m'], 'stream_ppl')
        df = paired_diff(results, a['arm'], a['tau_m'], 'forgetting_ppl')
        sgn = lambda d: (f"{d.mean():+.3f} ({int(np.sum(d < 0))}/10)"
                         if d.size else '-')
        print(f" {a['arm']:>3} {a['tau_m']:>6.0f} | "
              f"{a['stream_ppl_mean']:>9.3f} {sgn(dp):>9} | "
              f"{a['forgetting_ppl_mean']:>9.3f} {sgn(df):>9} | "
              f"{a['t_adapt_mean_mean']:>8.0f} "
              f"{a['neg_frac_mean'] * 100:>5.1f} | {'-':>10}")


def load_s19_bproj():
    """A1 cross-check reference: s19 B-proj means (the same readout)."""
    vals_s, vals_f = [], []
    with open(S19_CSV, 'r') as f:
        for row in csv.DictReader(f):
            if row['arm'] == 'B-proj':
                vals_s.append(float(row['stream_ppl']))
                vals_f.append(float(row['forgetting_ppl']))
    return {'stream': float(np.mean(vals_s)),
            'forgetting': float(np.mean(vals_f))}


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S20 M3 routing on SSM host "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    all_args = ([('A1', 0.0, s) for s in range(n_seeds)]
                + [('A2', tm, s) for tm in TAU_M_LIST for s in range(n_seeds)]
                + [('A3', tm, s) for tm in TAU_M_LIST for s in range(n_seeds)])
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (A1 x{n_seeds}, A2/A3 x{len(TAU_M_LIST)} "
          f"tau_m x{n_seeds}; spectrum={SPECTRUM})")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_single(a))
            if (i + 1) % max(1, n_runs // 10) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 10) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    agg = aggregate(results)
    s19_ref = load_s19_bproj()
    print_table(agg, results, s19_ref)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'tau_m', 'seed', 'stream_ppl', 'neg_frac',
                  't_adapt_mean', 'forgetting_ppl', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    paired = {}
    for arm in ['A2', 'A3']:
        for tm in TAU_M_LIST:
            dp = paired_diff(results, arm, tm, 'stream_ppl')
            df = paired_diff(results, arm, tm, 'forgetting_ppl')
            paired[f'{arm}-{tm:g}'] = {
                'stream_diff_mean': float(dp.mean()) if dp.size else None,
                'stream_improved_n': int(np.sum(dp < 0)) if dp.size else None,
                'forgetting_diff_mean': float(df.mean()) if df.size else None,
                'forgetting_improved_n': int(np.sum(df < 0)) if df.size else None,
            }

    params = {
        'host': {'type': 'diagonal linear SSM (hand-rolled, torch CPU)',
                 'state_dim': N_STATE,
                 'spectrum': 'log-uniform tau in [1,3000] (CV, P1/P3a host)',
                 'whitening': 'h_t * sqrt(N*(1-A_i^2))'},
        'readout_m1': {'type': 'per-token RLS on the input path '
                              'phi = [B e_{t-1}; 1] (P1/P3a evidence: the '
                              'current token must be an additive feature)',
                       'feature_dim': N_STATE + 1,
                       'lambda': RLS_LAMBDA, 'delta': RLS_DELTA,
                       'metric': 'CE = -ln(clip(y_hat[target], 1e-12, 1)), '
                                 'no softmax'},
        'metadata_m3': {'type': 'per-token EMA second state m_t = '
                                '(1-1/tau_m) m_{t-1} + (1/tau_m) h_w,t '
                                '[fast channels, tau<=8]',
                        'note': 'full whitened-state EMA is NOT a '
                                'stationary domain statistic - slow '
                                'channels accumulate over the stream '
                                '(detector never flips); fast channels are stationary '
                                'and carry the domain marginal '
                                '(separation ~1.2)',
                        'references': 'mean fast-channel whitened state '
                                      'over 1500-token reference streams '
                                      'per domain (s18 seed rule)',
                        'gate': 'nearest reference + 1.15 hysteresis margin',
                        'tau_m_list': TAU_M_LIST},
        'arms': {'A1': 'bare: single RLS readout, updates every token',
                 'A2': 'gate-only: RLS error scaled 1.0 for tau_m tokens '
                       'after a detected switch, 0.10 within a domain',
                 'A3': 'routing: two RLS readouts (one per domain); '
                       'slow-trace estimate selects active for prediction '
                       'and update; inactive fully frozen'},
        'task': {'desc': 'identical to s18 (Paper C Sec 6): two biased-'
                         'bigram Markov generators (A: shift=1 bias {0..7}; '
                         'B: shift=7 bias {24..31}), known switches, 6x3000',
                 'seg_len': SEG_LEN, 'n_segments': N_SEGMENTS,
                 't_total': SEG_LEN * N_SEGMENTS,
                 'seed_rules': 'verbatim from s18'},
        'metrics': {'stream_ppl': 'exp(mean CE) over the stream, '
                                  'predict-before-update',
                    'neg_frac': 'fraction of clipped (<=0) predictions',
                    't_adapt_mean': 'post-switch tokens to window ppl <= '
                                    'steady*1.5 (known switches)',
                    'forgetting_ppl': 'held-out ppl of the previous domain '
                                      '(domain-matched readout for A3)'},
        'cross_check': {'s19_B-proj_reference': s19_ref,
                        'note': 'A1 must reproduce s19 B-proj (same '
                                'readout/task/seeds)'},
        'paired_vs_a1': paired,
        'predictions_p2': 'routing (A3) improves forgetting at every tau_m '
                          '(s18 A3: 9-10/10); gating-only (A2) falsified '
                          'at every tau_m (s18 A2: 0/10)',
        'discipline': 'known switch instants (s15), 10 seeds, paired sign '
                      'consistency (s16)',
        'env': {'torch': torch.__version__, 'numpy': np.__version__,
                'cpu': 'torch CPU only, no mamba-ssm dependency'},
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
