#!/usr/bin/env python3
"""
S54: Paper F std-benchmark - Mackey-Glass next-bin CE on the main arms.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S54: do C1/C4b/C5 ordering transfer from the biased-bigram
                drift stream to a classical continuous dynamical system
                (Mackey-Glass), under the same diagonal host family and
                trained-softmax readout?

Task: Mackey-Glass tau=17, dt=10, after transient. Series min-max mapped
to VOCAB=32 integer bins; predict bin(t) from bin history via
phi built from the SSM (input path = B e_{t-1} on the previous bin).
Protocol: 18k train tokens (match s19 length), then held-out 2k from a
later segment for 'forgetting-style' generalization (not domain switch).

Arms (N=128, 5 seeds):
  B-softmax-sgd
  Gate-C-topk-softmax  (k=32)
  Soft-topk            (two experts; metadata = fast EMA; refs from
                       first/second half of a reference stream)

Output:
  data/s54_paper_f_mackey_glass_v1.csv
  data/s54_paper_f_mackey_glass_v1.json

Usage: python s54_paper_f_mackey_glass.py [--quick] [--sequential]
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
    VOCAB, N_STATE, SEED_SCALE, SEED_OFF, sample_substrate, whiten_scale,
)
from s50_paper_f_pilot_nl_readout import (
    SOFTMAX_LR, SOFTMAX_EPS, GRAD_CLIP, FLOOR_EPS, SUBSTATE_DIM,
    softmax_np, ce_from_p,
)
from s20_ssm_m3_routing import fast_mask_of

torch.set_num_threads(1)

ARMS = ['B-softmax-sgd', 'Gate-C-topk-softmax', 'Soft-topk']
TOPK_K = SUBSTATE_DIM
TAU_M = 500.0
SOFT_TAU_R = 1.0
N_TRAIN = 18000
N_HOLD = 2000
N_SEEDS = 5
MG_BETA, MG_GAMMA, MG_TAU, MG_DT = 0.2, 0.1, 17.0, 10.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's54_paper_f_mackey_glass_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's54_paper_f_mackey_glass_v1.json')


def mackey_glass(n_total, seed):
    """Euler MG; return exactly n_total samples after a transient."""
    rng = np.random.RandomState(seed * 17 + 3)
    delay = max(int(round(MG_TAU / MG_DT)), 1)
    hist_len = delay + 1
    transient = 200
    n_sim = n_total + transient
    x = np.zeros(n_sim + hist_len)
    x[:hist_len] = 0.5 + 0.1 * rng.rand(hist_len)
    p = 10.0
    for t in range(hist_len - 1, n_sim + hist_len - 1):
        xd = x[t - delay]
        x[t + 1] = x[t] + MG_DT * (MG_BETA * xd / (1.0 + xd ** p) - MG_GAMMA * x[t])
    return x[hist_len + transient: hist_len + transient + n_total]


def bin_series(x, n_bins=VOCAB):
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.int64)
    z = (x - lo) / (hi - lo)
    b = np.clip((z * n_bins).astype(np.int64), 0, n_bins - 1)
    return b


def make_refs_mg(seed, A, B, scale, n_ref=1500):
    """Two references from early vs late MG bins (pseudo-domains)."""
    series = bin_series(mackey_glass(n_ref * 2 + 500, seed + 99))
    fm = fast_mask_of(A)
    refs = []
    for half, sl in enumerate((slice(0, n_ref), slice(n_ref, 2 * n_ref))):
        stream = series[sl]
        h = torch.zeros(N_STATE, dtype=torch.float64)
        acc = torch.zeros(N_STATE, dtype=torch.float64)
        for t in range(n_ref):
            h = A * h + B[:, stream[t]]
            acc = acc + (h * scale)
        refs.append((acc / n_ref)[fm])
    return refs


def run_single(args):
    arm, seed = args
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    t0 = time.time()
    n_exp = 1 if arm == 'B-softmax-sgd' else (1 if arm == 'Gate-C-topk-softmax' else 2)
    use_gate = arm != 'B-softmax-sgd'

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    fm = fast_mask_of(A)

    series = bin_series(mackey_glass(N_TRAIN + N_HOLD + 10, seed))
    train = series[:N_TRAIN + 1]
    hold = series[N_TRAIN: N_TRAIN + N_HOLD + 1]
    refs = make_refs_mg(seed, A, B, scale)

    if use_gate:
        gC = torch.randn(N_STATE, N_STATE, dtype=torch.float64) / np.sqrt(N_STATE)
        gb = torch.zeros(N_STATE, dtype=torch.float64)
        gstore = {'C': gC, 'b': gb}
        GG = {k: torch.zeros_like(gstore[k]) for k in gstore}
        F = 2 * N_STATE + 1
    else:
        gstore, GG, F = {}, {}, N_STATE + 1

    Ws = [torch.zeros(VOCAB, F, dtype=torch.float64) for _ in range(n_exp)]
    GWs = [torch.zeros_like(W) for W in Ws]
    for W in Ws:
        W[:, -1] = 1.0 / VOCAB

    def build_phi(hw, prev_tok):
        x = B[:, prev_tok]
        if not use_gate:
            return torch.cat([x, torch.ones(1, dtype=torch.float64)])
        g = torch.sigmoid(gstore['C'] @ x + gstore['b'])
        idx = torch.topk(g, k=TOPK_K).indices
        mask = torch.zeros_like(g)
        mask[idx] = 1.0
        g = g * mask
        return torch.cat([hw * g, x, torch.ones(1, dtype=torch.float64)])

    def adagrad_W(W, GW, phi, gv):
        gn = float(np.linalg.norm(gv))
        if gn > GRAD_CLIP:
            gv = gv * (GRAD_CLIP / gn)
        g_t = torch.outer(torch.tensor(gv, dtype=torch.float64), phi)
        GW = GW + g_t * g_t
        W = W - SOFTMAX_LR * g_t / (torch.sqrt(GW) + SOFTMAX_EPS)
        return W, GW

    slow = refs[0].clone()
    h = torch.zeros(N_STATE, dtype=torch.float64)
    ce = np.empty(N_TRAIN, dtype=np.float64)
    ce[:] = np.nan
    target_oh = np.zeros(VOCAB)

    for t in range(1, N_TRAIN):
        h = A * h + B[:, train[t - 1]]
        hw = h * scale
        x = B[:, train[t - 1]]
        slow = (1 - 1 / TAU_M) * slow + (1 / TAU_M) * hw[fm]
        phi = build_phi(hw, train[t - 1])
        tgt = int(train[t])
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

        if use_gate:
            d_hwg = dphi[:N_STATE]
            pre = gstore['C'] @ x + gstore['b']
            g2 = torch.sigmoid(pre)
            idx = torch.topk(g2, k=TOPK_K).indices
            mask = torch.zeros_like(g2)
            mask[idx] = 1.0
            dpre = d_hwg * hw * g2 * (1 - g2) * mask
            grads = {'C': torch.outer(dpre, x), 'b': dpre}
            for key in ('C', 'b'):
                gr = grads[key]
                GG[key] = GG[key] + gr * gr
                gstore[key] = gstore[key] - SOFTMAX_LR * gr / (
                    torch.sqrt(GG[key]) + SOFTMAX_EPS)

    train_ppl = float(np.exp(np.nanmean(ce[1:])))

    # holdout: frozen params, no update
    h = torch.zeros(N_STATE, dtype=torch.float64)
    ces = []
    for t in range(1, N_HOLD):
        h = A * h + B[:, hold[t - 1]]
        phi = build_phi(h * scale, hold[t - 1])
        tgt = int(hold[t])
        if arm == 'Soft-topk':
            d0 = float(torch.norm(slow - refs[0]))
            d1 = float(torch.norm(slow - refs[1]))
            logits_pi = np.array([-d0, -d1]) / SOFT_TAU_R
            logits_pi = logits_pi - logits_pi.max()
            e = np.exp(logits_pi)
            pi = e / e.sum()
            p = (pi[0] * softmax_np((Ws[0] @ phi).numpy())
                 + pi[1] * softmax_np((Ws[1] @ phi).numpy()))
        else:
            p = softmax_np((Ws[0] @ phi).numpy())
        ces.append(ce_from_p(p, tgt))
    hold_ppl = float(np.exp(np.mean(ces)))

    return {
        'arm': arm, 'seed': seed,
        'train_ppl': train_ppl,
        'holdout_ppl': hold_ppl,
        'runtime_s': time.time() - t0,
    }


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t0 = time.time()
    n_seeds = 1 if quick else N_SEEDS
    all_args = [(a, s) for a in ARMS for s in range(n_seeds)]
    print(f"[{time.strftime('%H:%M:%S')}] START S54 Mackey-Glass "
          f"(quick={quick}, sequential={sequential})")
    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_single(a))
            print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(all_args)}", flush=True)
    else:
        with Pool(min(3, cpu_count(), max(1, len(all_args)))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(all_args)}",
                      flush=True)

    print('\n' + '=' * 80)
    by = {}
    for r in results:
        by.setdefault(r['arm'], []).append(r)
    for arm in ARMS:
        rs = by.get(arm, [])
        if not rs:
            continue
        print(f" {arm:>22} n={len(rs)}  train {np.mean([x['train_ppl'] for x in rs]):8.3f}"
              f"  holdout {np.mean([x['holdout_ppl'] for x in rs]):8.3f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed', 'train_ppl', 'holdout_ppl', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'task': 'mackey-glass-bin32', 'n_train': N_TRAIN,
                   'n_hold': N_HOLD, 'n_seeds': n_seeds,
                   'elapsed_s': time.time() - t0, 'results': results}, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] DONE S54 -> {out_csv}")


if __name__ == '__main__':
    main()
