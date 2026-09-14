#!/usr/bin/env python3
"""
S50: Paper F pilot P-F0 - nonlinear readout maps vs clipped-linear RLS.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S50 (Paper F, C1/C2/C3): does a trained softmax (C1) or
                probability-simplex (C2) readout eliminate Paper D's clip
                floor on the same SSM host / task / seed rules; C3 adds
                dimension/optimizer controls under the clean metric.
                Paper C-numbering: C1=trained softmax, C2=post-hoc simplex,
                C3=state under clean metric (this script implements all three).

Paper D (s19) used a squared-loss RLS readout evaluated as
CE = -ln clip(y_hat[target], 1e-12, 1). Negative or near-zero mass hits a
fixed floor; several stream claims were floor-sensitive. Paper F C1/C2 test
whether changing only the *readout map* (not the host) removes that floor:

  lin_clip   : y = W phi; CE = -ln clip(y[e], 1e-12, 1); train RLS (D)
  simplex    : p = project_simplex(y); CE = -ln p[e]; train RLS (same W)
  softmax_sgd: p = softmax(W phi / T); CE = -ln p[e]; train online CE SGD

Features (paired to s19):
  proj : B e_{t-1} + bias                 (s19 B-proj)
  skip : h^w + B e_{t-1} + bias           (s19 CV-skip)

Host / task / seed rules: VERBATIM s19 (whitened log-uniform tau in
[1, 3000], biased-bigram A/B drift, seed*SEED_SCALE+SEED_OFF). This enables
paired comparison against data/s19_ssm_rls_readout_v1.csv when present.

C1/C2 success gate (pre-registered): nonlinear arms achieve clip_frac/floor_frac
< 1e-3; paired stream/forgetting vs Skip-lin-clip reported honestly either way.
No claim that softmax "beats" D — the claim is metric stability + floor removal.

Output files:
  data/s50_paper_f_pilot_nl_readout_v1.csv
  data/s50_paper_f_pilot_nl_readout_v1.json

Usage: python s50_paper_f_pilot_nl_readout.py [--quick] [--sequential]
                 [--only B-noise-softmax,Skip-sub-softmax]

P-F0b dimension controls (added after review):
  B-noise-softmax  : [B e; N(0,1)^128; 1] + softmax AdaGrad
                     capacity match, zero state information
  Skip-sub-softmax : [h^w[:32]; B e; 1] + softmax AdaGrad
                     equal-width state vs input path
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_token_io import save_stream_and_holdout
from s19_ssm_rls_readout import (
    VOCAB, N_STATE, BIAS_A, BIAS_B, SEG_LEN, N_SEGMENTS, HOLDOUT_LEN,
    N_SEEDS, SEED_SCALE, SEED_OFF, RLS_LAMBDA, RLS_DELTA,
    sample_substrate, whiten_scale, gen_drift_stream, gen_stream,
)

torch.set_num_threads(1)

# ========================== Fixed parameters ==========================
SOFTMAX_T = 1.0
SOFTMAX_LR = 0.05          # base AdaGrad step for softmax_sgd
SOFTMAX_EPS = 1e-8
GRAD_CLIP = 5.0            # clip ||g||_2 of (p - e) before W update
FLOOR_EPS = 1e-12

ARMS = {
    'B-lin-clip':      {'feat': 'proj', 'map': 'lin_clip'},
    'B-simplex':       {'feat': 'proj', 'map': 'simplex'},
    'B-softmax-sgd':   {'feat': 'proj', 'map': 'softmax_sgd'},
    'Skip-lin-clip':   {'feat': 'skip', 'map': 'lin_clip'},
    'Skip-simplex':    {'feat': 'skip', 'map': 'simplex'},
    'Skip-softmax-sgd': {'feat': 'skip', 'map': 'softmax_sgd'},
    # P-F0b dimension controls (softmax only)
    'B-noise-softmax': {'feat': 'noise', 'map': 'softmax_sgd'},
    'Skip-sub-softmax': {'feat': 'sub', 'map': 'softmax_sgd'},
}
NOISE_DIM = 128
SUBSTATE_DIM = 32
NOISE_SEED_OFF = 9

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's50_paper_f_pilot_nl_readout_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's50_paper_f_pilot_nl_readout_v1.json')
S19_CSV = os.path.join(DATA_DIR, 's19_ssm_rls_readout_v1.csv')


def project_simplex(y, eps=FLOOR_EPS):
    """Euclidean projection of y onto the probability simplex (approximate
    via clip-negatives + renormalize). All-nonpositive -> uniform."""
    y_pos = np.maximum(y, 0.0)
    s = float(y_pos.sum())
    if s <= eps:
        return np.full_like(y_pos, 1.0 / y_pos.size)
    p = y_pos / s
    # guard residual numerical zeros on the target later via floor_frac
    return p


def softmax_np(logits, t=SOFTMAX_T):
    z = logits / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def ce_from_p(p, target):
    return -np.log(max(float(p[target]), FLOOR_EPS))


def run_single(args):
    """(arm, seed) -> metrics dict. Top-level (picklable); unbuffered."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    cfg = ARMS[arm]
    t0 = time.time()
    rmap = cfg['map']

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    stream, domains = gen_drift_stream(seed)
    T = stream.shape[0]

    noise_rng = np.random.RandomState(
        seed * SEED_SCALE + SEED_OFF + NOISE_SEED_OFF)

    def build_phi(hw, prev_tok, noise=None):
        x = B[:, prev_tok]
        feat = cfg['feat']
        if feat == 'proj':
            parts = [x]
        elif feat == 'skip':
            parts = [hw, x]
        elif feat == 'noise':
            parts = [x, torch.as_tensor(noise, dtype=torch.float64)]
        elif feat == 'sub':
            parts = [hw[:SUBSTATE_DIM], x]
        else:
            raise ValueError(feat)
        parts.append(torch.ones(1, dtype=torch.float64))
        return torch.cat(parts)

    z_state = torch.zeros(N_STATE, dtype=torch.float64)
    if cfg['feat'] == 'noise':
        z_noise = np.zeros(NOISE_DIM, dtype=np.float64)
    else:
        z_noise = None
    F = len(build_phi(z_state, 0, z_noise))
    W = torch.zeros(VOCAB, F, dtype=torch.float64)
    W[:, -1] = 1.0 / VOCAB
    P = torch.eye(F, dtype=torch.float64) / RLS_DELTA
    h = torch.zeros(N_STATE, dtype=torch.float64)

    use_rls = rmap in ('lin_clip', 'simplex')
    ce = np.empty(T, dtype=np.float64)
    ce[:] = np.nan
    nclip = 0
    nneg = 0
    nfloor = 0
    hs = np.empty((T - 1, F), dtype=np.float64)
    # AdaGrad accumulators for softmax_sgd (stability on skip features)
    G = torch.zeros_like(W)

    for t in range(1, T):
        h = A * h + B[:, stream[t - 1]]
        noise_t = None
        if cfg['feat'] == 'noise':
            noise_t = noise_rng.randn(NOISE_DIM)
        phi = build_phi(h * scale, stream[t - 1], noise_t)
        hs[t - 1] = phi.numpy()
        tgt = int(stream[t])

        if use_rls:
            y = (W @ phi).numpy()
            if rmap == 'lin_clip':
                yt = float(y[tgt])
                if yt <= 0.0:
                    nneg += 1
                if yt <= FLOOR_EPS:
                    nclip += 1
                ce[t] = -np.log(max(yt, FLOOR_EPS))
                p_eval = None
            else:  # simplex
                p_eval = project_simplex(y)
                pt = float(p_eval[tgt])
                if pt <= FLOOR_EPS:
                    nfloor += 1
                ce[t] = ce_from_p(p_eval, tgt)
            # RLS update on raw y (identical to D)
            target = torch.zeros(VOCAB, dtype=torch.float64)
            target[tgt] = 1.0
            g = P @ phi
            k = g / (RLS_LAMBDA + float(phi @ g))
            e = target - (W @ phi)
            W = W + torch.outer(e, k)
            P = (P - torch.outer(k, phi) @ P) / RLS_LAMBDA
        else:  # softmax_sgd
            logits = (W @ phi).numpy()
            p = softmax_np(logits)
            pt = float(p[tgt])
            if pt <= FLOOR_EPS:
                nfloor += 1
            ce[t] = ce_from_p(p, tgt)
            grad = p.copy()
            grad[tgt] -= 1.0
            gn = float(np.linalg.norm(grad))
            if gn > GRAD_CLIP:
                grad *= GRAD_CLIP / gn
            g_t = torch.outer(torch.tensor(grad, dtype=torch.float64), phi)
            G = G + g_t * g_t
            W = W - SOFTMAX_LR * g_t / (torch.sqrt(G) + SOFTMAX_EPS)

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    clip_frac = float(nclip) / (T - 1)
    floor_frac = float(nfloor) / (T - 1)
    neg_frac = float(nneg) / (T - 1) if use_rls and rmap == 'lin_clip' else 0.0

    # Pooled ridge oracle on the same features (linear bound; diagnostic)
    Y = np.zeros((T - 1, VOCAB))
    Y[np.arange(T - 1), stream[1:]] = 1.0
    X = hs
    XtX = X.T @ X + RLS_DELTA * np.eye(F)
    Wr = Y.T @ X @ np.linalg.inv(XtX)
    yhat = X @ Wr.T
    p_or = np.clip(
        np.take_along_axis(yhat, stream[1:][:, None], axis=1).ravel(),
        FLOOR_EPS, 1.0)
    oracle_ppl = float(np.exp(np.mean(-np.log(p_or))))

    # Forgetting holdout (s18/s19 metric)
    forgets = []
    n_hold_floor = 0
    n_hold = 0
    hold_ce_all = []
    for si, t_s in enumerate([SEG_LEN * s for s in range(1, N_SEGMENTS)]):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3,
                          1 if prev_dom == 0 else 7, HOLDOUT_LEN,
                          BIAS_A if prev_dom == 0 else BIAS_B)
        hh = torch.zeros(N_STATE, dtype=torch.float64)
        ces = []
        for t in range(1, HOLDOUT_LEN):
            hh = A * hh + B[:, hold[t - 1]]
            noise_h = None
            if cfg['feat'] == 'noise':
                noise_h = noise_rng.randn(NOISE_DIM)
            phi = build_phi(hh * scale, hold[t - 1], noise_h)
            tgt = int(hold[t])
            n_hold += 1
            if use_rls:
                y = (W @ phi).numpy()
                if rmap == 'lin_clip':
                    yt = float(y[tgt])
                    if yt <= FLOOR_EPS:
                        n_hold_floor += 1
                    c = -np.log(max(yt, FLOOR_EPS))
                else:
                    p = project_simplex(y)
                    pt = float(p[tgt])
                    if pt <= FLOOR_EPS:
                        n_hold_floor += 1
                    c = ce_from_p(p, tgt)
            else:
                p = softmax_np((W @ phi).numpy())
                pt = float(p[tgt])
                if pt <= FLOOR_EPS:
                    n_hold_floor += 1
                c = ce_from_p(p, tgt)
            ces.append(c)
            hold_ce_all.append(c)
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))
    forget_floor_frac = float(n_hold_floor) / n_hold if n_hold else float('nan')

    save_stream_and_holdout(
        's50', arm, seed, 'x', ce, ce,
        np.asarray(hold_ce_all, dtype=np.float64),
        np.asarray(hold_ce_all, dtype=np.float64))

    return {
        'arm': arm,
        'seed': seed,
        'feat': cfg['feat'],
        'readout_map': rmap,
        'stream_ppl': stream_ppl,
        'clip_frac': clip_frac,
        'floor_frac': floor_frac,
        'neg_frac': neg_frac,
        'forgetting_ppl': forgetting_ppl,
        'forget_floor_frac': forget_floor_frac,
        'oracle_ppl': oracle_ppl,
        'softmax_lr': SOFTMAX_LR if rmap == 'softmax_sgd' else float('nan'),
        'runtime_s': time.time() - t0,
    }


def load_s19_ref():
    """Optional paired refs from committed s19 rows."""
    ref = {}
    if not os.path.exists(S19_CSV):
        return ref
    want = {'B-proj': 'B-lin-clip', 'CV-skip': 'Skip-lin-clip'}
    with open(S19_CSV, 'r') as f:
        for row in csv.DictReader(f):
            if row['arm'] in want:
                key = (want[row['arm']], int(row['seed']))
                ref[key] = {
                    'stream_ppl': float(row['stream_ppl']),
                    'forgetting_ppl': float(row['forgetting_ppl']),
                    'clip_frac': float(row['clip_frac']),
                }
    return ref


def print_table(results, ref):
    print('\n' + '=' * 110)
    print('S50 RESULTS (Paper F P-F0): nonlinear readout vs clipped-linear '
          'RLS on the s19 host (10 seeds)')
    print('=' * 110)
    print(f" {'arm':>16} {'seed':>4} | {'stream':>9} {'forget':>9} "
          f"{'clip%':>7} {'floor%':>7} | {'oracle':>8} | {'d_stream':>9} "
          f"{'d_forget':>9} {'d_clip%':>8}")
    by_arm = {}
    for r in results:
        by_arm.setdefault(r['arm'], []).append(r)
    for arm in ARMS:
        rs = sorted(by_arm.get(arm, []), key=lambda x: x['seed'])
        for r in rs:
            key = (arm, r['seed'])
            if key in ref:
                ds = r['stream_ppl'] - ref[key]['stream_ppl']
                df = r['forgetting_ppl'] - ref[key]['forgetting_ppl']
                dc = (r['clip_frac'] + r['floor_frac']
                      - ref[key]['clip_frac']) * 100
            else:
                ds = df = dc = float('nan')
            print(f" {arm:>16} {r['seed']:>4} | {r['stream_ppl']:>9.3f} "
                  f"{r['forgetting_ppl']:>9.3f} "
                  f"{r['clip_frac'] * 100:>6.2f}% {r['floor_frac'] * 100:>6.2f}% | "
                  f"{r['oracle_ppl']:>8.3f} | {ds:>+9.3f} {df:>+9.3f} {dc:>+8.2f}")
        if not rs:
            continue
        mean_s = np.mean([r['stream_ppl'] for r in rs])
        mean_f = np.mean([r['forgetting_ppl'] for r in rs])
        mean_c = np.mean([(r['clip_frac'] + r['floor_frac']) for r in rs])
        n_under = int(np.sum([(r['clip_frac'] + r['floor_frac']) < 1e-3
                              for r in rs]))
        print(f" {'':>16} {'':>4} | MEAN stream {mean_s:.3f} forget {mean_f:.3f} "
              f"floor+clip {mean_c * 100:.3f}% | under_1e-3 {n_under}/{len(rs)}")
        print('-' * 110)
    return by_arm


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S50 Paper F C1/C2/C3 pilot "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    only = None
    if '--only' in sys.argv:
        i = sys.argv.index('--only')
        only = set(sys.argv[i + 1].split(','))
        unknown = only - set(ARMS)
        if unknown:
            raise SystemExit(f'unknown arms: {sorted(unknown)}')
    arm_list = [a for a in ARMS if only is None or a in only]
    all_args = [(arm, s) for arm in arm_list for s in range(n_seeds)]
    print(f"total runs: {len(all_args)} (arms {arm_list}, {n_seeds} seeds, "
          f"softmax_lr={SOFTMAX_LR}, T={SOFTMAX_T}, N={N_STATE})")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_single(a))
            print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{len(all_args)}",
                  flush=True)
    else:
        with Pool(min(cpu_count(), max(1, len(all_args)))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{len(all_args)}",
                      flush=True)

    ref = load_s19_ref()
    print_table(results, ref)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed', 'feat', 'readout_map', 'stream_ppl',
                  'clip_frac', 'floor_frac', 'neg_frac', 'forgetting_ppl',
                  'forget_floor_frac', 'oracle_ppl', 'softmax_lr', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    rows = list(results)
    if only is not None and not quick and os.path.exists(out_csv):
        keep = {(r['arm'], int(r['seed'])) for r in results}
        with open(out_csv, 'r', newline='') as f:
            for old in csv.DictReader(f):
                key = (old['arm'], int(old['seed']))
                if key not in keep:
                    old['seed'] = int(old['seed'])
                    for k in ('stream_ppl', 'clip_frac', 'floor_frac', 'neg_frac',
                              'forgetting_ppl', 'forget_floor_frac', 'oracle_ppl',
                              'softmax_lr', 'runtime_s'):
                        if old.get(k) not in (None, ''):
                            old[k] = float(old[k])
                    rows.append(old)
        rows.sort(key=lambda r: (r['arm'], int(r['seed'])))
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    meta = {
        'script': 's50_paper_f_pilot_nl_readout',
        'paper': 'Paper F',
        'claim': 'C1/C2/C3',
        'n_seeds': n_seeds,
        'quick': quick,
        'softmax_lr': SOFTMAX_LR,
        'softmax_T': SOFTMAX_T,
        'softmax_eps': SOFTMAX_EPS,
        'grad_clip': GRAD_CLIP,
        'floor_eps': FLOOR_EPS,
        'arms': ARMS,
        'ran_arms': arm_list,
        'noise_dim': NOISE_DIM,
        'substate_dim': SUBSTATE_DIM,
        'noise_seed_off': NOISE_SEED_OFF,
        'host': 's19 CV-whiten (log-uniform tau [1,3000], whitened)',
        'task': 's18/s19 two-domain biased-bigram drift',
        'seed_rule': f'seed*{SEED_SCALE}+{SEED_OFF}',
        'elapsed_s': time.time() - t_start,
        'results': rows,
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] DONE S50 -> {out_csv}")
    print(f"elapsed: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
