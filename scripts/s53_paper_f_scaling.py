#!/usr/bin/env python3
"""
S53: Paper F C6/scaling - main arms across host width N.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S53 (Paper F, scaling support): do C1 (B-softmax), C4b
                (Gate-C-topk), and C5 (Soft-topk) keep their ordering as
                the diagonal-SSM width N grows?

N_STATE in s19/s51/s52 is a module global (128). This script re-samples
the substrate at variable N under the same seed rule (seed*101+17) and
the same log-uniform tau in [1,3000] + whitening. Task/stream seeds are
unchanged (s18 protocol), so rows at N=128 should approximately reproduce
s51/s52 means (optimizer is sequential AdaGrad; expect close, not bitwise).

Arms x N:
  B-softmax-sgd     : input path only (C1 baseline)
  Gate-C-topk-softmax : sparse learned gate, k = max(8, N//4) (C4b)
  Soft-topk         : two experts + soft routing, same gate (C5)

N_LIST default: 128, 256, 512. Seeds default: 5 (paired across N).

Output:
  data/s53_paper_f_scaling_v1.csv
  data/s53_paper_f_scaling_v1.json

Usage: python s53_paper_f_scaling.py [--quick] [--sequential]
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
    SEED_SCALE, SEED_OFF, TAU_MIN, TAU_MAX, B_GAIN,
    gen_drift_stream, gen_stream,
)
from s50_paper_f_pilot_nl_readout import (
    SOFTMAX_LR, SOFTMAX_EPS, GRAD_CLIP, FLOOR_EPS,
    softmax_np, ce_from_p,
)

torch.set_num_threads(1)

N_LIST = [128, 256, 512]
N_SEEDS = 5
ARMS = ['B-softmax-sgd', 'Gate-C-topk-softmax', 'Soft-topk']
TAU_M = 500.0
SOFT_TAU_R = 1.0
GATE_MARGIN = 1.15
FAST_TAU = 8.0
REF_LEN = 1500

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's53_paper_f_scaling_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's53_paper_f_scaling_v1.json')


def sample_substrate_n(seed, n):
    """CV spectrum + random B at width n (s19 seed rule)."""
    rng = np.random.RandomState(seed * SEED_SCALE + SEED_OFF + 1)
    tau = np.exp(rng.uniform(np.log(TAU_MIN), np.log(TAU_MAX), n))
    A = torch.tensor(np.exp(-1.0 / tau), dtype=torch.float64)
    g = torch.Generator()
    g.manual_seed(seed * SEED_SCALE + SEED_OFF)
    B = (torch.randn(n, VOCAB, dtype=torch.float64, generator=g)
         * B_GAIN / np.sqrt(n))
    return A, B


def whiten_scale_n(A, n):
    return torch.sqrt((1.0 - A * A) * n)


def fast_mask_of(A):
    tau = -1.0 / torch.log(A)
    return tau <= FAST_TAU


def reference_features_n(seed, A, B, scale, n):
    fm = fast_mask_of(A)
    refs = []
    for d in range(2):
        ref_stream = gen_stream(seed * 31 + d * 131 + 7,
                                1 if d == 0 else 7, REF_LEN,
                                BIAS_A if d == 0 else BIAS_B)
        h = torch.zeros(n, dtype=torch.float64)
        acc = torch.zeros(n, dtype=torch.float64)
        for t in range(REF_LEN):
            h = A * h + B[:, ref_stream[t]]
            acc = acc + (h * scale)
        refs.append((acc / REF_LEN)[fm])
    return refs


def gate_estimate(slow, refs, prev=0, margin=GATE_MARGIN):
    d0 = float(torch.norm(slow - refs[0]))
    d1 = float(torch.norm(slow - refs[1]))
    if d0 * margin < d1:
        return 0
    if d1 * margin < d0:
        return 1
    return prev


def topk_of(n):
    return max(8, n // 4)


def run_single(args):
    """(arm, n, seed) -> metrics."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, n, seed = args
    t0 = time.time()
    n = int(n)
    k = topk_of(n)
    n_exp = 1 if arm == 'B-softmax-sgd' else (1 if arm == 'Gate-C-topk-softmax' else 2)

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate_n(seed, n)
    scale = whiten_scale_n(A, n)
    fm = fast_mask_of(A)
    stream, domains = gen_drift_stream(seed)
    T = stream.shape[0]
    refs = reference_features_n(seed, A, B, scale, n)

    use_gate = arm != 'B-softmax-sgd'
    if use_gate:
        gC = torch.randn(n, n, dtype=torch.float64) / np.sqrt(n)
        gb = torch.zeros(n, dtype=torch.float64)
        gstore = {'C': gC, 'b': gb}
        GG = {key: torch.zeros_like(gstore[key]) for key in gstore}
        F = 2 * n + 1
    else:
        gstore = {}
        GG = {}
        F = n + 1

    Ws = [torch.zeros(VOCAB, F, dtype=torch.float64) for _ in range(n_exp)]
    GWs = [torch.zeros_like(W) for W in Ws]
    for W in Ws:
        W[:, -1] = 1.0 / VOCAB

    def build_phi(hw, prev_tok):
        x = B[:, prev_tok]
        if not use_gate:
            return torch.cat([x, torch.ones(1, dtype=torch.float64)])
        g = torch.sigmoid(gstore['C'] @ x + gstore['b'])
        idx = torch.topk(g, k=k).indices
        mask = torch.zeros_like(g)
        mask[idx] = 1.0
        g = g * mask
        return torch.cat([hw * g, x, torch.ones(1, dtype=torch.float64)])

    def adagrad_W(W, GW, phi, grad_vec):
        gn = float(np.linalg.norm(grad_vec))
        if gn > GRAD_CLIP:
            grad_vec = grad_vec * (GRAD_CLIP / gn)
        g_t = torch.outer(torch.tensor(grad_vec, dtype=torch.float64), phi)
        GW = GW + g_t * g_t
        W = W - SOFTMAX_LR * g_t / (torch.sqrt(GW) + SOFTMAX_EPS)
        return W, GW

    def gate_grads(dpre, x):
        grads = {'C': torch.outer(dpre, x), 'b': dpre}
        for key in ('C', 'b'):
            gr = grads[key]
            GG[key] = GG[key] + gr * gr
            gstore[key] = gstore[key] - SOFTMAX_LR * gr / (
                torch.sqrt(GG[key]) + SOFTMAX_EPS)

    slow = refs[0].clone()
    h = torch.zeros(n, dtype=torch.float64)
    ce = np.empty(T, dtype=np.float64)
    ce[:] = np.nan
    target_oh = np.zeros(VOCAB)

    for t in range(1, T):
        h = A * h + B[:, stream[t - 1]]
        hw = h * scale
        x = B[:, stream[t - 1]]
        lam = 1.0 / TAU_M
        slow = (1.0 - lam) * slow + lam * hw[fm]
        phi = build_phi(hw, stream[t - 1])
        tgt = int(stream[t])
        target_oh[:] = 0.0
        target_oh[tgt] = 1.0

        if arm == 'Soft-topk':
            d0 = float(torch.norm(slow - refs[0]))
            d1 = float(torch.norm(slow - refs[1]))
            logits_pi = np.array([-d0, -d1]) / SOFT_TAU_R
            logits_pi = logits_pi - logits_pi.max()
            e = np.exp(logits_pi)
            pis = e / e.sum()
            ps = [softmax_np((W @ phi).numpy()) for W in Ws]
            p = pis[0] * ps[0] + pis[1] * ps[1]
            pt = float(p[tgt])
            ce[t] = ce_from_p(p, tgt)
            p_tgt = max(pt, FLOOR_EPS)
            dphi = torch.zeros(F, dtype=torch.float64)
            for j in range(2):
                resp = float(pis[j]) * float(ps[j][tgt]) / p_tgt
                gvec = (ps[j] - target_oh) * resp
                Ws[j], GWs[j] = adagrad_W(Ws[j], GWs[j], phi, gvec)
                dphi = dphi + resp * (
                    torch.tensor(ps[j] - target_oh, dtype=torch.float64) @ Ws[j])
        else:
            p = softmax_np((Ws[0] @ phi).numpy())
            ce[t] = ce_from_p(p, tgt)
            gvec = p - target_oh
            Ws[0], GWs[0] = adagrad_W(Ws[0], GWs[0], phi, gvec)
            dphi = torch.tensor(gvec, dtype=torch.float64) @ Ws[0]

        if use_gate:
            d_hwg = dphi[:n]
            pre = gstore['C'] @ x + gstore['b']
            g2 = torch.sigmoid(pre)
            idx = torch.topk(g2, k=k).indices
            mask = torch.zeros_like(g2)
            mask[idx] = 1.0
            dpre = d_hwg * hw * g2 * (1 - g2) * mask
            gate_grads(dpre, x)

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))

    forgets = []
    for si, t_s in enumerate([SEG_LEN * s for s in range(1, N_SEGMENTS)]):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3,
                          1 if prev_dom == 0 else 7, HOLDOUT_LEN,
                          BIAS_A if prev_dom == 0 else BIAS_B)
        hh = torch.zeros(n, dtype=torch.float64)
        ces = []
        for t in range(1, HOLDOUT_LEN):
            hh = A * hh + B[:, hold[t - 1]]
            phi = build_phi(hh * scale, hold[t - 1])
            tgt = int(hold[t])
            if arm == 'Soft-topk':
                slow_h = (1 - 1 / TAU_M) * refs[prev_dom] + (1 / TAU_M) * (
                    hh * scale)[fm]
                d0 = float(torch.norm(slow_h - refs[0]))
                d1 = float(torch.norm(slow_h - refs[1]))
                logits_pi = np.array([-d0, -d1]) / SOFT_TAU_R
                logits_pi = logits_pi - logits_pi.max()
                e = np.exp(logits_pi)
                pi = e / e.sum()
                p = (pi[0] * softmax_np((Ws[0] @ phi).numpy())
                     + pi[1] * softmax_np((Ws[1] @ phi).numpy()))
            else:
                p = softmax_np((Ws[0] @ phi).numpy())
            ces.append(ce_from_p(p, tgt))
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))

    return {
        'arm': arm, 'N': n, 'seed': seed, 'topk_k': k if use_gate else 0,
        'n_experts': n_exp if arm == 'Soft-topk' else 1,
        'stream_ppl': stream_ppl,
        'forgetting_ppl': forgetting_ppl,
        'forgetting_ppl_med': float(np.median(forgets)),
        'runtime_s': time.time() - t0,
    }


def print_table(results):
    print('\n' + '=' * 88)
    print('S53 RESULTS (Paper F scaling): main arms vs N')
    print('=' * 88)
    by = {}
    for r in results:
        by.setdefault((r['arm'], r['N']), []).append(r)
    for arm in ARMS:
        for n in sorted({int(r['N']) for r in results}):
            rs = by.get((arm, n), [])
            if not rs:
                continue
            ms = np.mean([r['stream_ppl'] for r in rs])
            mf = np.mean([r['forgetting_ppl'] for r in rs])
            print(f" {arm:>22} N={n:4d} n={len(rs)}  "
                  f"stream {ms:8.3f}  forget {mf:8.3f}")
        print('-' * 88)
    return by


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t0 = time.time()
    n_seeds = 1 if quick else N_SEEDS
    n_list = [128, 256] if quick else list(N_LIST)
    if '--n' in sys.argv:
        n_list = [int(x) for x in sys.argv[sys.argv.index('--n') + 1].split(',')]
    if '--seeds' in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index('--seeds') + 1])
    n_proc = min(4, cpu_count())
    if '--nproc' in sys.argv:
        n_proc = int(sys.argv[sys.argv.index('--nproc') + 1])
    all_args = [(a, n, s) for a in ARMS for n in n_list for s in range(n_seeds)]
    print(f"[{time.strftime('%H:%M:%S')}] START S53 scaling "
          f"(quick={quick}, sequential={sequential}, N={n_list}, "
          f"seeds={n_seeds}, nproc={n_proc})")
    print(f"total runs: {len(all_args)}")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_single(a))
            print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{len(all_args)}",
                  flush=True)
    else:
        with Pool(min(n_proc, max(1, len(all_args)))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{len(all_args)}",
                      flush=True)

    print_table(results)
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'N', 'seed', 'topk_k', 'n_experts', 'stream_ppl',
                  'forgetting_ppl', 'forgetting_ppl_med', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    rows = list(results)
    if '--n' in sys.argv and os.path.exists(out_csv):
        keep = {(r['arm'], int(r['N']), int(r['seed'])) for r in results}
        with open(out_csv, 'r', newline='') as f:
            for old in csv.DictReader(f):
                key = (old['arm'], int(old['N']), int(old['seed']))
                if key not in keep:
                    old['N'] = int(old['N'])
                    old['seed'] = int(old['seed'])
                    for k in ('topk_k', 'n_experts', 'stream_ppl',
                              'forgetting_ppl', 'forgetting_ppl_med', 'runtime_s'):
                        if old.get(k) not in (None, ''):
                            old[k] = float(old[k])
                    rows.append(old)
        rows.sort(key=lambda r: (r['arm'], int(r['N']), int(r['seed'])))
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    meta = {
        'script': 's53_paper_f_scaling',
        'claim': 'C6/scaling',
        'N_LIST': n_list,
        'N_SEEDS': n_seeds,
        'ARMS': ARMS,
        'topk_rule': 'max(8, N//4)',
        'tau_m': TAU_M,
        'elapsed_s': time.time() - t0,
        'results': rows,
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] DONE S53 -> {out_csv} "
          f"({time.time() - t0:.1f}s)")


if __name__ == '__main__':
    main()
