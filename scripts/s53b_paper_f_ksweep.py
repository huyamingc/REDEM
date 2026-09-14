#!/usr/bin/env python3
"""
S53b: Paper F k-sweep at N=512 (why Gate-C-topk / Soft-topk failed to scale).
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S53b: at N=512, does a k other than N//4=128 restore the
                N=128 ordering (Gate-C-topk / Soft-topk better than B-softmax)?

Arms: Gate-C-topk and Soft-topk only (B-softmax from s53 N=512 is the
reference). k in {16, 32, 64, 128}. Seeds: 3 (paired).

Output:
  data/s53b_paper_f_ksweep_v1.csv
  data/s53b_paper_f_ksweep_v1.json

Usage: python s53b_paper_f_ksweep.py [--quick] [--sequential]
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
from s53_paper_f_scaling import (
    run_single as _unused,  # ensure module import side effects
    sample_substrate_n, whiten_scale_n, fast_mask_of,
    reference_features_n, TAU_M, SOFT_TAU_R, GATE_MARGIN,
)
# Re-implement run with explicit k (s53 hardcodes k=max(8,N//4))
from s19_ssm_rls_readout import (
    VOCAB, BIAS_A, BIAS_B, SEG_LEN, N_SEGMENTS, HOLDOUT_LEN,
    SEED_SCALE, SEED_OFF, gen_drift_stream, gen_stream,
)
from s50_paper_f_pilot_nl_readout import (
    SOFTMAX_LR, SOFTMAX_EPS, GRAD_CLIP, FLOOR_EPS,
    softmax_np, ce_from_p,
)

torch.set_num_threads(1)

N = 512
K_LIST = [16, 32, 64, 128]
N_SEEDS = 3
ARMS = ['Gate-C-topk-softmax', 'Soft-topk']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's53b_paper_f_ksweep_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's53b_paper_f_ksweep_v1.json')


def run_one(args):
    arm, k, seed = args
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    t0 = time.time()
    k = int(k)
    n = N
    n_exp = 1 if arm == 'Gate-C-topk-softmax' else 2

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate_n(seed, n)
    scale = whiten_scale_n(A, n)
    fm = fast_mask_of(A)
    stream, domains = gen_drift_stream(seed)
    T = stream.shape[0]
    refs = reference_features_n(seed, A, B, scale, n)

    gC = torch.randn(n, n, dtype=torch.float64) / np.sqrt(n)
    gb = torch.zeros(n, dtype=torch.float64)
    gstore = {'C': gC, 'b': gb}
    GG = {key: torch.zeros_like(gstore[key]) for key in gstore}
    F = 2 * n + 1
    Ws = [torch.zeros(VOCAB, F, dtype=torch.float64) for _ in range(n_exp)]
    GWs = [torch.zeros_like(W) for W in Ws]
    for W in Ws:
        W[:, -1] = 1.0 / VOCAB

    def build_phi(hw, prev_tok):
        x = B[:, prev_tok]
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
            ce[t] = ce_from_p(p, tgt)
            p_tgt = max(float(p[tgt]), FLOOR_EPS)
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

        d_hwg = dphi[:n]
        pre = gstore['C'] @ x + gstore['b']
        g2 = torch.sigmoid(pre)
        idx = torch.topk(g2, k=k).indices
        mask = torch.zeros_like(g2)
        mask[idx] = 1.0
        dpre = d_hwg * hw * g2 * (1 - g2) * mask
        grads = {'C': torch.outer(dpre, x), 'b': dpre}
        for key in ('C', 'b'):
            gr = grads[key]
            GG[key] = GG[key] + gr * gr
            gstore[key] = gstore[key] - SOFTMAX_LR * gr / (
                torch.sqrt(GG[key]) + SOFTMAX_EPS)

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

    return {
        'arm': arm, 'N': n, 'k': k, 'seed': seed,
        'stream_ppl': stream_ppl,
        'forgetting_ppl': float(np.mean(forgets)),
        'runtime_s': time.time() - t0,
    }


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t0 = time.time()
    n_seeds = 1 if quick else N_SEEDS
    k_list = [32, 128] if quick else list(K_LIST)
    all_args = [(a, k, s) for a in ARMS for k in k_list for s in range(n_seeds)]
    print(f"[{time.strftime('%H:%M:%S')}] START S53b k-sweep N={N} "
          f"k={k_list} seeds={n_seeds}")
    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_one(a))
            print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(all_args)}", flush=True)
    else:
        with Pool(min(2, cpu_count(), max(1, len(all_args)))) as pool:
            done = 0
            for res in pool.imap_unordered(run_one, all_args, chunksize=1):
                results.append(res)
                done += 1
                print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(all_args)}",
                      flush=True)

    print('\n' + '=' * 80)
    by = {}
    for r in results:
        by.setdefault((r['arm'], r['k']), []).append(r)
    for arm in ARMS:
        for k in k_list:
            rs = by.get((arm, k), [])
            if not rs:
                continue
            print(f" {arm:>22} k={k:4d} n={len(rs)}  "
                  f"stream {np.mean([x['stream_ppl'] for x in rs]):8.3f}  "
                  f"forget {np.mean([x['forgetting_ppl'] for x in rs]):8.3f}")
        print('-' * 80)
    print('B-softmax N=512 reference (s53): stream 7.248  forget 11.093')

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'N', 'k', 'seed', 'stream_ppl', 'forgetting_ppl',
                  'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'N': N, 'k_list': k_list, 'n_seeds': n_seeds,
                   'elapsed_s': time.time() - t0, 'results': results}, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] DONE S53b -> {out_csv}")


if __name__ == '__main__':
    main()
