#!/usr/bin/env python3
"""
S52: Paper F C5 - soft routing over online softmax experts on a Gate-C-topk host.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S52 (Paper F, C5): under the C1 calibrated softmax readout and
                the C4b sparse learned gate (top-k), does M3-style soft
                routing over two online experts improve retention over a
                single specialist without giving up the stream gain?

Paper D (s20 A3) showed hard routing of RLS readouts improves forgetting.
Paper F changes three things under the paradigm rules:
  (1) readout = trained softmax (C1), not clipped-linear RLS;
  (2) features = Gate-C-topk (C4b), not bare input path;
  (3) routing = soft mixture over experts (primary) with hard as control.

Arms (same host/task/seeds as s19/s51):
  Single-topk  : one Gate-C-topk + one softmax expert (C5 control)
  Hard-topk    : two experts; nearest fast-EMA reference with hysteresis
                 selects active expert (s20 A3 semantics, softmax CE)
  Soft-topk    : two experts; pi = softmax(-||m-ref||/T_r); p = sum pi_k p_k;
                 each expert AdaGrad-updated with weight pi_k

C5 pre-registration: Soft-topk improves forgetting_ppl over Single-topk on
>=7/10 seeds AND stream_ppl is not worse than Single-topk on >=7/10 seeds.
Falsified if forgetting never improves (0/10) or stream always worsens.

Feedback budget (R4): per-token labels (same as s50/s51). Not an E-port.

Output files:
  data/s52_paper_f_soft_route_experts_v1.csv
  data/s52_paper_f_soft_route_experts_v1.json

Usage: python s52_paper_f_soft_route_experts.py [--quick] [--sequential]
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
    N_SEEDS, SEED_SCALE, SEED_OFF,
    sample_substrate, whiten_scale, gen_drift_stream, gen_stream,
)
from s50_paper_f_pilot_nl_readout import (
    SOFTMAX_LR, SOFTMAX_EPS, GRAD_CLIP, FLOOR_EPS, SUBSTATE_DIM,
    softmax_np, ce_from_p,
)
from s20_ssm_m3_routing import fast_mask_of, reference_features, gate_estimate

torch.set_num_threads(1)

ARMS = ['Single-topk', 'Hard-topk', 'Soft-topk']
TOPK_K = SUBSTATE_DIM
TAU_M = 500.0
SOFT_TAU_R = 1.0            # temperature on EMA distances
GATE_MARGIN = 1.15

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's52_paper_f_soft_route_experts_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's52_paper_f_soft_route_experts_v1.json')


def run_single(args):
    """(arm, seed) -> metrics. Top-level; unbuffered."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    t0 = time.time()
    n_exp = 1 if arm == 'Single-topk' else 2

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    fm = fast_mask_of(A)
    stream, domains = gen_drift_stream(seed)
    T = stream.shape[0]
    refs = reference_features(seed, A, B, scale)

    # Shared learned gate (C4b): C, b trained jointly with experts
    gC = torch.randn(N_STATE, N_STATE, dtype=torch.float64) / np.sqrt(N_STATE)
    gb = torch.zeros(N_STATE, dtype=torch.float64)
    gstore = {'C': gC, 'b': gb}
    GG = {k: torch.zeros_like(gstore[k]) for k in gstore}

    F = 2 * N_STATE + 1
    Ws = [torch.zeros(VOCAB, F, dtype=torch.float64) for _ in range(n_exp)]
    GWs = [torch.zeros_like(W) for W in Ws]
    for W in Ws:
        W[:, -1] = 1.0 / VOCAB

    def build_phi(hw, prev_tok):
        x = B[:, prev_tok]
        g = torch.sigmoid(gstore['C'] @ x + gstore['b'])
        idx = torch.topk(g, k=TOPK_K).indices
        mask = torch.zeros_like(g)
        mask[idx] = 1.0
        g = g * mask
        return torch.cat([hw * g, x, torch.ones(1, dtype=torch.float64)]), g

    def adagrad_W(W, GW, phi, grad_vec):
        gn = float(np.linalg.norm(grad_vec))
        if gn > GRAD_CLIP:
            grad_vec = grad_vec * (GRAD_CLIP / gn)
        g_t = torch.outer(torch.tensor(grad_vec, dtype=torch.float64), phi)
        GW = GW + g_t * g_t
        W = W - SOFTMAX_LR * g_t / (torch.sqrt(GW) + SOFTMAX_EPS)
        return W, GW

    def adagrad_gate(dpre, x, hw_unused=None):
        grads = {
            'C': torch.outer(dpre, x),
            'b': dpre,
        }
        for k in ('C', 'b'):
            gr = grads[k]
            GG[k] = GG[k] + gr * gr
            gstore[k] = gstore[k] - SOFTMAX_LR * gr / (
                torch.sqrt(GG[k]) + SOFTMAX_EPS)

    def gate_grads(phi_grad_hwg, hw, x):
        """phi_grad_hwg = dL/d (hw*g) from expert mix."""
        pre = gstore['C'] @ x + gstore['b']
        g2 = torch.sigmoid(pre)
        idx = torch.topk(g2, k=TOPK_K).indices
        mask = torch.zeros_like(g2)
        mask[idx] = 1.0
        dpre = phi_grad_hwg * hw * g2 * (1 - g2) * mask
        adagrad_gate(dpre, x)

    slow = refs[0].clone()
    prev_est = 0
    h = torch.zeros(N_STATE, dtype=torch.float64)
    ce = np.empty(T, dtype=np.float64)
    ce[:] = np.nan
    nfloor = 0
    n_route_flip = 0

    for t in range(1, T):
        h = A * h + B[:, stream[t - 1]]
        hw = h * scale
        x = B[:, stream[t - 1]]
        lam = 1.0 / TAU_M
        slow = (1.0 - lam) * slow + lam * hw[fm]
        d0 = float(torch.norm(slow - refs[0]))
        d1 = float(torch.norm(slow - refs[1]))

        phi, _g = build_phi(hw, stream[t - 1])
        tgt = int(stream[t])

        if arm == 'Single-topk':
            pis = np.array([1.0])
            ps = [softmax_np((Ws[0] @ phi).numpy())]
        elif arm == 'Hard-topk':
            est = gate_estimate(slow, refs, prev_est, GATE_MARGIN)
            if est != prev_est:
                n_route_flip += 1
                prev_est = est
            pis = np.zeros(2)
            pis[est] = 1.0
            ps = [softmax_np((Ws[est] @ phi).numpy())]
        else:  # Soft-topk
            dists = np.array([d0, d1], dtype=np.float64)
            logits_pi = -dists / SOFT_TAU_R
            logits_pi = logits_pi - logits_pi.max()
            e = np.exp(logits_pi)
            pis = e / e.sum()
            ps = [softmax_np((W @ phi).numpy()) for W in Ws]

        if arm == 'Soft-topk':
            p = pis[0] * ps[0] + pis[1] * ps[1]
        else:
            p = ps[0]

        pt = float(p[tgt])
        if pt <= FLOOR_EPS:
            nfloor += 1
        ce[t] = ce_from_p(p, tgt)

        # Expert updates: each expert sees (p_expert - e) * pi_k
        # Gate sees dL/dphi through the mixture.
        target_oh = np.zeros(VOCAB)
        target_oh[tgt] = 1.0
        dphi = torch.zeros(F, dtype=torch.float64)

        if arm == 'Single-topk':
            gvec = ps[0] - target_oh
            Ws[0], GWs[0] = adagrad_W(Ws[0], GWs[0], phi, gvec)
            dphi = torch.tensor(gvec, dtype=torch.float64) @ Ws[0]
        elif arm == 'Hard-topk':
            k = int(np.argmax(pis))
            gvec = ps[0] - target_oh
            Ws[k], GWs[k] = adagrad_W(Ws[k], GWs[k], phi, gvec)
            dphi = torch.tensor(gvec, dtype=torch.float64) @ Ws[k]
        else:
            p_tgt = max(float(p[tgt]), FLOOR_EPS)
            for k in range(2):
                # exact dCE/dlogits_k for p = sum_k pi_k p_k
                resp = float(pis[k]) * float(ps[k][tgt]) / p_tgt
                gvec = (ps[k] - target_oh) * resp
                Ws[k], GWs[k] = adagrad_W(Ws[k], GWs[k], phi, gvec)
                dphi = dphi + resp * (
                    torch.tensor(ps[k] - target_oh, dtype=torch.float64) @ Ws[k])

        d_hwg = dphi[:N_STATE]
        gate_grads(d_hwg, hw, x)

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    floor_frac = float(nfloor) / (T - 1)

    # Forgetting: domain-matched expert (or mixture) on held-out previous domain
    forgets = []
    hold_ce = []
    for si, t_s in enumerate([SEG_LEN * s for s in range(1, N_SEGMENTS)]):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3,
                          1 if prev_dom == 0 else 7, HOLDOUT_LEN,
                          BIAS_A if prev_dom == 0 else BIAS_B)
        hh = torch.zeros(N_STATE, dtype=torch.float64)
        ces = []
        for t in range(1, HOLDOUT_LEN):
            hh = A * hh + B[:, hold[t - 1]]
            hwh = hh * scale
            phi, _ = build_phi(hwh, hold[t - 1])
            tgt = int(hold[t])
            if arm == 'Single-topk':
                p = softmax_np((Ws[0] @ phi).numpy())
            elif arm == 'Hard-topk':
                # s20 semantics: domain-matched specialist
                p = softmax_np((Ws[prev_dom] @ phi).numpy())
            else:
                # soft mixture using holdout EMA toward matching ref
                slow_h = (1 - 1 / TAU_M) * refs[prev_dom] + (1 / TAU_M) * hwh[fm]
                d0 = float(torch.norm(slow_h - refs[0]))
                d1 = float(torch.norm(slow_h - refs[1]))
                logits_pi = np.array([-d0, -d1]) / SOFT_TAU_R
                logits_pi = logits_pi - logits_pi.max()
                e = np.exp(logits_pi)
                pi = e / e.sum()
                p = (pi[0] * softmax_np((Ws[0] @ phi).numpy())
                     + pi[1] * softmax_np((Ws[1] @ phi).numpy()))
            c = ce_from_p(p, tgt)
            ces.append(c)
            hold_ce.append(c)
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))
    forgetting_med = float(np.median(forgets))

    save_stream_and_holdout(
        's52', arm, seed, 'x', ce, ce,
        np.asarray(hold_ce, dtype=np.float64),
        np.asarray(hold_ce, dtype=np.float64))

    return {
        'arm': arm, 'seed': seed,
        'stream_ppl': stream_ppl,
        'floor_frac': floor_frac,
        'forgetting_ppl': forgetting_ppl,
        'forgetting_ppl_med': forgetting_med,
        'n_route_flip': n_route_flip,
        'tau_m': TAU_M,
        'soft_tau_r': SOFT_TAU_R if arm == 'Soft-topk' else float('nan'),
        'runtime_s': time.time() - t0,
    }


def print_table(results):
    print('\n' + '=' * 96)
    print('S52 RESULTS (Paper F C5): soft/hard routing on Gate-C-topk experts')
    print('=' * 96)
    by = {}
    for r in results:
        by.setdefault(r['arm'], []).append(r)
    for arm in ARMS:
        rs = sorted(by.get(arm, []), key=lambda x: x['seed'])
        if not rs:
            continue
        ms = np.mean([r['stream_ppl'] for r in rs])
        mf = np.mean([r['forgetting_ppl'] for r in rs])
        md = np.median([r['forgetting_ppl'] for r in rs])
        print(f" {arm:>14} n={len(rs):2d}  stream {ms:8.3f}  "
              f"forget mean/med {mf:8.3f}/{md:7.3f}  "
              f"flips {np.mean([r['n_route_flip'] for r in rs]):.1f}")
    return by


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t0 = time.time()
    n_seeds = 1 if quick else N_SEEDS
    all_args = [(a, s) for a in ARMS for s in range(n_seeds)]
    print(f"[{time.strftime('%H:%M:%S')}] START S52 "
          f"(quick={quick}, sequential={sequential}, tau_m={TAU_M})")
    print(f"total runs: {len(all_args)}")

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

    print_table(results)
    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed', 'stream_ppl', 'floor_frac', 'forgetting_ppl',
                  'forgetting_ppl_med', 'n_route_flip', 'tau_m', 'soft_tau_r',
                  'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    meta = {
        'script': 's52_paper_f_soft_route_experts',
        'claim': 'C5',
        'arms': ARMS,
        'topk_k': TOPK_K,
        'tau_m': TAU_M,
        'soft_tau_r': SOFT_TAU_R,
        'softmax_lr': SOFTMAX_LR,
        'n_seeds': n_seeds,
        'quick': quick,
        'elapsed_s': time.time() - t0,
        'results': results,
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] DONE S52 -> {out_csv} "
          f"({time.time() - t0:.1f}s)")


if __name__ == '__main__':
    main()
