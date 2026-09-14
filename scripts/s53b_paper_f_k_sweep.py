#!/usr/bin/env python3
"""
S53b: Paper F - top-k sweep to diagnose N=512 Gate-C-topk failure.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S53b: at N=512, does the Gate-C-topk stream/forget collapse
                come from a bad k rule (k=max(8,N//4)=128) rather than from
                the mechanism itself?

Arms: Gate-C-topk-softmax with k in {16, 32, 64, 128, 256} at N=512,
and k in {16, 32, 64} at N=128 (reference). B-softmax-sgd at both N as
the no-gate baseline. 5 seeds, same protocol as s53.

Output:
  data/s53b_paper_f_k_sweep_v1.csv
  data/s53b_paper_f_k_sweep_v1.json

Usage: python s53b_paper_f_k_sweep.py [--quick] [--sequential]
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
from s19_ssm_rls_readout import (
    VOCAB, BIAS_A, BIAS_B, SEG_LEN, N_SEGMENTS, HOLDOUT_LEN,
    SEED_SCALE, SEED_OFF, gen_drift_stream, gen_stream,
)
from s50_paper_f_pilot_nl_readout import (
    SOFTMAX_LR, SOFTMAX_EPS, GRAD_CLIP, FLOOR_EPS,
    softmax_np, ce_from_p,
)
from s53_paper_f_scaling import (
    sample_substrate_n, whiten_scale_n, TAU_M,
)

torch.set_num_threads(1)

# (N, k) pairs; B-softmax uses k=0
CONFIGS = [
    (128, 0),    # baseline
    (128, 16),
    (128, 32),
    (128, 64),
    (512, 0),    # baseline
    (512, 32),
    (512, 64),
    (512, 128),  # s53 default rule at 512
    (512, 256),
]
N_SEEDS = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's53b_paper_f_k_sweep_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's53b_paper_f_k_sweep_v1.json')


def run_one(args):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    n, k, seed = args
    n = int(n)
    k = int(k)
    t0 = time.time()
    use_gate = k > 0
    n_exp = 1

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate_n(seed, n)
    scale = whiten_scale_n(A, n)
    stream, _domains = gen_drift_stream(seed)
    T = stream.shape[0]

    if use_gate:
        gstore = {
            'C': torch.randn(n, n, dtype=torch.float64) / np.sqrt(n),
            'b': torch.zeros(n, dtype=torch.float64),
        }
        GG = {key: torch.zeros_like(gstore[key]) for key in gstore}
        F = 2 * n + 1
    else:
        gstore, GG = {}, {}
        F = n + 1

    W = torch.zeros(VOCAB, F, dtype=torch.float64)
    GW = torch.zeros_like(W)
    W[:, -1] = 1.0 / VOCAB

    def build_phi(hw, prev_tok):
        x = B[:, prev_tok]
        if not use_gate:
            return torch.cat([x, torch.ones(1, dtype=torch.float64)])
        g = torch.sigmoid(gstore['C'] @ x + gstore['b'])
        idx = torch.topk(g, k=k).indices
        mask = torch.zeros_like(g)
        mask[idx] = 1.0
        return torch.cat([hw * (g * mask), x, torch.ones(1, dtype=torch.float64)])

    h = torch.zeros(n, dtype=torch.float64)
    ce = np.empty(T, dtype=np.float64)
    ce[:] = np.nan
    target_oh = np.zeros(VOCAB)

    for t in range(1, T):
        h = A * h + B[:, stream[t - 1]]
        hw = h * scale
        x = B[:, stream[t - 1]]
        phi = build_phi(hw, stream[t - 1])
        tgt = int(stream[t])
        target_oh[:] = 0.0
        target_oh[tgt] = 1.0

        p = softmax_np((W @ phi).numpy())
        ce[t] = ce_from_p(p, tgt)
        gvec = p - target_oh
        gn = float(np.linalg.norm(gvec))
        if gn > GRAD_CLIP:
            gvec = gvec * (GRAD_CLIP / gn)
        g_t = torch.outer(torch.tensor(gvec, dtype=torch.float64), phi)
        GW = GW + g_t * g_t
        W = W - SOFTMAX_LR * g_t / (torch.sqrt(GW) + SOFTMAX_EPS)
        dphi = torch.tensor(gvec, dtype=torch.float64) @ W

        if use_gate:
            pre = gstore['C'] @ x + gstore['b']
            g2 = torch.sigmoid(pre)
            idx = torch.topk(g2, k=k).indices
            mask = torch.zeros_like(g2)
            mask[idx] = 1.0
            dpre = dphi[:n] * hw * g2 * (1 - g2) * mask
            grads = {'C': torch.outer(dpre, x), 'b': dpre}
            for key in ('C', 'b'):
                gr = grads[key]
                GG[key] = GG[key] + gr * gr
                gstore[key] = gstore[key] - SOFTMAX_LR * gr / (
                    torch.sqrt(GG[key]) + SOFTMAX_EPS)

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))

    forgets = []
    for si, t_s in enumerate([SEG_LEN * s for s in range(1, N_SEGMENTS)]):
        prev_dom = int(_domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3,
                          1 if prev_dom == 0 else 7, HOLDOUT_LEN,
                          BIAS_A if prev_dom == 0 else BIAS_B)
        hh = torch.zeros(n, dtype=torch.float64)
        ces = []
        for t in range(1, HOLDOUT_LEN):
            hh = A * hh + B[:, hold[t - 1]]
            phi = build_phi(hh * scale, hold[t - 1])
            ces.append(ce_from_p(softmax_np((W @ phi).numpy()), int(hold[t])))
        forgets.append(float(np.exp(np.mean(ces))))

    return {
        'N': n, 'k': k, 'seed': seed, 'arm': 'B-softmax' if k == 0 else 'Gate-C-topk',
        'stream_ppl': stream_ppl,
        'forgetting_ppl': float(np.mean(forgets)),
        'runtime_s': time.time() - t0,
    }


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t0 = time.time()
    n_seeds = 1 if quick else N_SEEDS
    cfgs = CONFIGS[:4] if quick else CONFIGS
    all_args = [(n, k, s) for (n, k) in cfgs for s in range(n_seeds)]
    print(f"[{time.strftime('%H:%M:%S')}] START S53b k-sweep "
          f"(quick={quick}, sequential={sequential}, configs={cfgs})")
    print(f"total runs: {len(all_args)}")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_one(a))
            print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{len(all_args)}",
                  flush=True)
    else:
        n_proc = min(3, cpu_count())
        with Pool(min(n_proc, max(1, len(all_args)))) as pool:
            done = 0
            for res in pool.imap_unordered(run_one, all_args, chunksize=1):
                results.append(res)
                done += 1
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{len(all_args)}",
                      flush=True)

    print('\n' + '=' * 80)
    print('S53b RESULTS: Gate-C-topk k-sweep')
    print('=' * 80)
    by = {}
    for r in results:
        by.setdefault((r['N'], r['k']), []).append(r)
    for key in sorted(by):
        rs = by[key]
        ms = np.mean([r['stream_ppl'] for r in rs])
        mf = np.mean([r['forgetting_ppl'] for r in rs])
        label = 'B-softmax' if key[1] == 0 else f'Gate k={key[1]}'
        print(f" N={key[0]:4d}  {label:>12}  n={len(rs)}  "
              f"stream {ms:8.3f}  forget {mf:8.3f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['N', 'k', 'seed', 'arm', 'stream_ppl', 'forgetting_ppl',
                  'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    meta = {
        'script': 's53b_paper_f_k_sweep',
        'CONFIGS': cfgs,
        'N_SEEDS': n_seeds,
        'elapsed_s': time.time() - t0,
        'results': results,
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] DONE S53b -> {out_csv} "
          f"({time.time() - t0:.1f}s)")


if __name__ == '__main__':
    main()
