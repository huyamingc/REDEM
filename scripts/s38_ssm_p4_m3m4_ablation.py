#!/usr/bin/env python3
"""
S38: Paper D P4 M3/M4 factor ablation on the four-domain host.
=============================================================================
Type:           ML (torch CPU; Pool around independent trials)
Experiment:     S38: separate the benchmark contributions of M3 (fast-channel
                metadata / nearest-reference assignment) and M4 (soft
                routing over specialists). s22 only compared the endpoints
                (M1 vs M1+M3+M4); this script adds two intermediate arms.

Host:           diagonal SSM, CV spectrum, whitened state; readout uses the
                additive current-token input path (M1), same as s22.

Arms (10 seeds, paired):
  SSM-bare     : M1 only — single RLS on [B e; 1]  (endpoint, same as s22)
  SSM-uniform  : M1 + four specialists, soft weights FIXED at 1/4
                 (M4 structure without M3 assignment)
  SSM-hard     : M1 + M3 fast-channel EMA + hard nearest-reference routing
                 (binary one-hot assignment; inactive specialists frozen)
                 (M3 assignment without soft mixtures)
  SSM-REDEM    : M1 + M3 + M4 soft routing  (full stack; same as s22)

Protocol, seeds, metrics, CE clipping: verbatim from s22_ssm_p4_benchmark.

Output files:
  data/s38_ssm_p4_m3m4_ablation_v1.csv
  data/s38_ssm_p4_m3m4_ablation_v1.json

Usage: python s38_ssm_p4_m3m4_ablation.py [--quick] [--sequential]
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

from s18_llm_drift_gate import gen_stream
from s19_ssm_rls_readout import (sample_substrate, whiten_scale,
                                 N_STATE, RLS_DELTA)
from s20_ssm_m3_routing import fast_mask_of, REF_LEN
from s21_ssm_m4_m5 import make_readout, rls_update

torch.set_num_threads(1)

N_DOMAINS = 4
N_SEGMENTS = 8
SEG_LO = 2500
SEG_HI = 3500
N_SEEDS = 10
TAU_M = 500.0
SHIFTS = [1, 3, 7, 11]
BIAS_SETS = [list(range(0, 4)), list(range(8, 12)),
             list(range(16, 20)), list(range(24, 28))]
HOLDOUT_LEN = 500
VOCAB = 32
SEED_SCALE = 101
SEED_OFF = 17

ARMS = ['SSM-bare', 'SSM-uniform', 'SSM-hard', 'SSM-REDEM']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's38_ssm_p4_m3m4_ablation_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's38_ssm_p4_m3m4_ablation_v1.json')


def gen_multi_drift_stream(seed, n_domains=N_DOMAINS,
                           n_segments=N_SEGMENTS, seg_lo=SEG_LO,
                           seg_hi=SEG_HI):
    rng = np.random.RandomState(seed * 7 + 11)
    parts, domains, seg_lens = [], [], []
    for s in range(n_segments):
        dom = s % n_domains
        seg_len = int(rng.uniform(seg_lo, seg_hi))
        dom_seed = seed * 31 + s * 131 + 7
        parts.append(gen_stream(dom_seed, SHIFTS[dom], seg_len,
                                BIAS_SETS[dom]))
        domains.append(dom)
        seg_lens.append(seg_len)
    stream = np.concatenate(parts)
    switch_times = np.cumsum(seg_lens)[:-1].tolist()
    return stream, np.repeat(np.array(domains), seg_lens), \
        switch_times, seg_lens


def refs_fast_multi(seed, A, B, scale):
    fm = fast_mask_of(A)
    refs = []
    for d in range(N_DOMAINS):
        rs = gen_stream(seed * 31 + d * 131 + 7, SHIFTS[d], REF_LEN,
                        BIAS_SETS[d])
        h = torch.zeros(N_STATE, dtype=torch.float64)
        acc = torch.zeros(N_STATE, dtype=torch.float64)
        for t in range(REF_LEN):
            h = A * h + B[:, rs[t]]
            acc = acc + h * scale
        refs.append((acc / REF_LEN)[fm])
    return refs


def soft_weights(distances, kappa):
    z = -torch.stack(distances) / kappa
    z = z - z.max()
    w = torch.exp(z)
    return w / w.sum()


def run_ssm(args):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    fm = fast_mask_of(A)
    stream, domains, switch_times, _ = gen_multi_drift_stream(seed)
    T = stream.shape[0]
    F = N_STATE + 1

    multi = arm in ('SSM-uniform', 'SSM-hard', 'SSM-REDEM')
    refs = refs_fast_multi(seed, A, B, scale)
    pair_d = [float(torch.norm(refs[i] - refs[j]))
              for i in range(N_DOMAINS) for j in range(i + 1, N_DOMAINS)]
    kappa = 0.5 * float(np.median(pair_d))

    if multi:
        Ws = [make_readout(F) for _ in range(N_DOMAINS)]
    else:
        Ws = [make_readout(F)]
    slow = refs[0].clone()
    W, P = Ws[0]

    ce = np.full(T, np.nan, dtype=np.float64)
    ce_unc = np.full(T, np.nan, dtype=np.float64)
    nneg = nclip = n_unc_undef = 0
    h = torch.zeros(N_STATE, dtype=torch.float64)

    for t in range(1, T):
        h = A * h + B[:, stream[t - 1]]
        hw = h * scale
        phi = torch.cat([B[:, stream[t - 1]],
                         torch.ones(1, dtype=torch.float64)])

        if arm == 'SSM-bare':
            y_hat = W @ phi
            err_scale = 1.0
        elif arm == 'SSM-uniform':
            y_hat = sum(Ws[i][0] @ phi for i in range(N_DOMAINS)) / N_DOMAINS
            err_scale = 1.0 / N_DOMAINS
        elif arm == 'SSM-hard':
            lam = 1.0 / TAU_M
            slow = (1.0 - lam) * slow + lam * hw[fm]
            dists = [torch.norm(slow - r) for r in refs]
            idx = int(torch.argmin(torch.stack(dists)))
            y_hat = Ws[idx][0] @ phi
            err_scale = 1.0
        else:  # SSM-REDEM
            lam = 1.0 / TAU_M
            slow = (1.0 - lam) * slow + lam * hw[fm]
            dists = [torch.norm(slow - r) for r in refs]
            w = soft_weights(dists, kappa)
            y_hat = sum(w[i] * (Ws[i][0] @ phi) for i in range(N_DOMAINS))
            err_scale = None  # per-specialist

        y_tok = y_hat[stream[t]]
        p_t = float(y_tok.clamp(min=1e-12, max=1.0))
        yt = float(y_tok)
        if yt <= 0.0:
            nneg += 1
        if yt <= 1e-12:
            nclip += 1
        if yt > 0.0:
            ce_unc[t] = -np.log(yt)
        else:
            n_unc_undef += 1
        ce[t] = -np.log(p_t)

        target = torch.zeros(VOCAB, dtype=torch.float64)
        target[stream[t]] = 1.0

        if arm == 'SSM-bare':
            W, P = rls_update(W, P, phi, target, 1.0)
        elif arm == 'SSM-uniform':
            for i in range(N_DOMAINS):
                Wi, Pi = Ws[i]
                Ws[i] = rls_update(Wi, Pi, phi, target, err_scale)
        elif arm == 'SSM-hard':
            Wi, Pi = Ws[idx]
            Ws[idx] = rls_update(Wi, Pi, phi, target, 1.0)
        else:
            for i in range(N_DOMAINS):
                Wi, Pi = Ws[i]
                Ws[i] = rls_update(Wi, Pi, phi, target, float(w[i]))

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    stream_ppl_unclipped = (float(np.exp(np.nanmean(ce_unc[1:])))
                            if n_unc_undef < (T - 1) else float('nan'))

    # T_adapt (s22 protocol)
    t_adapts = []
    for si, t_s in enumerate(switch_times):
        seg_end = (switch_times[si + 1] if si + 1 < len(switch_times) else T)
        seg_ce = ce[t_s:seg_end]
        if seg_ce.size == 0:
            continue
        steady = float(np.exp(np.mean(seg_ce[-400:])))
        thr = steady * 1.5
        cum = np.concatenate([[0.0], seg_ce])
        found = None
        for j in range(20, seg_ce.shape[0] + 1):
            wppl = np.exp((cum[j] - cum[j - 20]) / 20)
            if wppl <= thr:
                found = j
                break
        t_adapts.append(float(found) if found is not None else float('nan'))

    # Forgetting
    forgets = []
    forgets_unc = []
    fneg = fclip = n_f_undef = 0
    for si, t_s in enumerate(switch_times):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3, SHIFTS[prev_dom],
                          HOLDOUT_LEN, BIAS_SETS[prev_dom])
        if arm == 'SSM-bare':
            Wf = W
        else:
            Wf = Ws[prev_dom][0]
        hh = torch.zeros(N_STATE, dtype=torch.float64)
        ces = []
        ces_unc = []
        for t in range(1, HOLDOUT_LEN):
            hh = A * hh + B[:, hold[t - 1]]
            phi = torch.cat([B[:, hold[t - 1]],
                             torch.ones(1, dtype=torch.float64)])
            y_tok = (Wf @ phi)[hold[t]]
            p_t = float(y_tok.clamp(min=1e-12, max=1.0))
            yt = float(y_tok)
            if yt <= 0.0:
                fneg += 1
            if yt <= 1e-12:
                fclip += 1
            if yt > 0.0:
                ces_unc.append(-np.log(yt))
            else:
                n_f_undef += 1
            ces.append(-np.log(p_t))
        forgets.append(float(np.exp(np.mean(ces))))
        forgets_unc.append(float(np.exp(np.mean(ces_unc)))
                           if ces_unc else float('nan'))

    forgetting_ppl = float(np.mean(forgets))
    forgetting_ppl_unclipped = (float(np.nanmean(forgets_unc))
                                if forgets_unc else float('nan'))
    n_hold = (HOLDOUT_LEN - 1) * max(1, len(forgets))

    return {
        'arm': arm, 'seed': seed,
        'stream_ppl': stream_ppl,
        'neg_frac': float(nneg) / (T - 1),
        'clip_frac': float(nclip) / (T - 1),
        't_adapt_mean': float(np.nanmean(t_adapts)) if t_adapts else float('nan'),
        'forgetting_ppl': forgetting_ppl,
        'stream_ppl_unclipped': stream_ppl_unclipped,
        'stream_n_unc_undefined': int(n_unc_undef),
        'forget_neg_frac': float(fneg) / n_hold if n_hold else float('nan'),
        'forget_clip_frac': float(fclip) / n_hold if n_hold else float('nan'),
        'forgetting_ppl_unclipped': forgetting_ppl_unclipped,
        'forget_n_unc_undefined': int(n_f_undef),
        'runtime_s': time.time() - t0,
    }


def paired(results, arm_a, arm_b, metric):
    da = {r['seed']: r[metric] for r in results if r['arm'] == arm_a}
    db = {r['seed']: r[metric] for r in results if r['arm'] == arm_b}
    seeds = sorted(set(da) & set(db))
    return np.array([da[s] - db[s] for s in seeds], dtype=np.float64)


def print_table(results):
    print('\narm means (reported metric):')
    for arm in ARMS:
        rows = [r for r in results if r['arm'] == arm]
        if not rows:
            continue
        sp = np.mean([r['stream_ppl'] for r in rows])
        fp = np.mean([r['forgetting_ppl'] for r in rows])
        su = np.nanmean([r['stream_ppl_unclipped'] for r in rows])
        fu = np.nanmean([r['forgetting_ppl_unclipped'] for r in rows])
        print(f'  {arm:12s}  stream {sp:7.3f}  forget {fp:7.3f}  '
              f'unclipped {su:7.3f}/{fu:7.3f}')
    print('\npaired (a - b):')
    pairs = [('SSM-uniform', 'SSM-bare'), ('SSM-hard', 'SSM-bare'),
             ('SSM-REDEM', 'SSM-bare'), ('SSM-hard', 'SSM-uniform'),
             ('SSM-REDEM', 'SSM-hard'), ('SSM-REDEM', 'SSM-uniform')]
    for a, b in pairs:
        for label, metric in [('stream', 'stream_ppl'),
                              ('forget', 'forgetting_ppl')]:
            d = paired(results, a, b, metric)
            if d.size == 0:
                continue
            print(f'  {a} vs {b} [{label}]: {d.mean():+.3f}, '
                  f'{a} better {int(np.sum(d < 0))}/{d.size}')


def dispatch(task):
    (kind, arm), s = task
    return run_ssm((arm, s))


def _nan_to_null(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_null(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S38 M3/M4 ablation "
          f"(quick={quick}, sequential={sequential})")
    n_seeds = 1 if quick else N_SEEDS
    all_args = [(('SSM', arm), s) for arm in ARMS for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f'total runs: {n_runs} ({ARMS} x{n_seeds})')

    results = []
    if sequential:
        for i, task in enumerate(all_args):
            results.append(dispatch(task))
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

    results.sort(key=lambda r: (ARMS.index(r['arm']), r['seed']))
    print_table(results)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed', 'stream_ppl', 'neg_frac', 'clip_frac',
                  't_adapt_mean', 'forgetting_ppl', 'stream_ppl_unclipped',
                  'stream_n_unc_undefined', 'forget_neg_frac',
                  'forget_clip_frac', 'forgetting_ppl_unclipped',
                  'forget_n_unc_undefined', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    comp = {}
    pairs = [('SSM-uniform', 'SSM-bare'), ('SSM-hard', 'SSM-bare'),
             ('SSM-REDEM', 'SSM-bare'), ('SSM-hard', 'SSM-uniform'),
             ('SSM-REDEM', 'SSM-hard'), ('SSM-REDEM', 'SSM-uniform')]
    for a, b in pairs:
        key = f'{a}_vs_{b}'
        comp[key] = {
            'stream_diff_mean': float(paired(results, a, b, 'stream_ppl').mean()),
            'stream_improved_n': int(np.sum(paired(results, a, b, 'stream_ppl') < 0)),
            'forgetting_diff_mean': float(paired(results, a, b, 'forgetting_ppl').mean()),
            'forgetting_improved_n': int(np.sum(paired(results, a, b, 'forgetting_ppl') < 0)),
            'stream_unclipped_diff_mean': float(paired(
                results, a, b, 'stream_ppl_unclipped').mean()),
            'forgetting_unclipped_diff_mean': float(paired(
                results, a, b, 'forgetting_ppl_unclipped').mean()),
        }

    params = {
        'task': 'same as s22: 4 domains, irregular switch Uniform(2500,3500), 8 segments',
        'arms': {
            'SSM-bare': 'M1 only',
            'SSM-uniform': 'M1 + 4 specialists, soft weights = 1/4 (M4 without M3)',
            'SSM-hard': 'M1 + M3 EMA + hard nearest-ref routing (M3 without soft M4)',
            'SSM-REDEM': 'M1 + M3 + M4 soft (full stack, s22 protocol)',
        },
        'comparisons': comp,
        'discipline': '10 seeds, paired sign consistency',
        'tau_m': TAU_M,
        'n_seeds': n_seeds, 'quick': bool(quick),
        'env': {'torch': torch.__version__, 'numpy': np.__version__},
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump(_nan_to_null({'params': params, 'rows': results}), f,
                  indent=2, allow_nan=False)
    print(f'\nCSV : {out_csv}\nJSON: {out_json}')
    print(f'[{time.strftime("%H:%M:%S")}] DONE, total {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
