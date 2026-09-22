#!/usr/bin/env python3
"""
S51: Paper F C4/C4b - learned selectivity gate under calibrated softmax readout.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S51 (Paper F, C4/C4b): does a LEARNED feature-level gate make
                the SSM state net-useful under the C1 softmax readout, and is
                hard top-k (C4b) the missing sparsity pressure?

Paper D (s19 CV-gate) falsified a FIXED random input gate (0/10). Paper F
crosses that flagged boundary by training the gate. s50 P-F0/P-F0b showed:
  - trained softmax removes the clip floor (C1);
  - full ungated state is stream-neutral and retention-harmful;
  - 32-d substate is stream-positive and retention-safe;
  - 128-d pure noise does not collapse retention (not a dimension artifact).

C4/C4b therefore requires a three-way comparison against calibrated baselines
(not against Skip-lin-clip alone):
  B-softmax-sgd      : input path only (from s50, recomputed here for pairing)
  Skip-sub-softmax   : h^w[:32] + input (best cheap state from s50)
  B-noise-softmax    : input + 128-d i.i.d. noise (capacity control)
  Gate-C-softmax     : learn C,b; g = sigmoid(C x + b); phi = [h^w ⊙ g; x; 1]
  Gate-static-softmax: learn a; g = sigmoid(a) (static channel mask)
  Gate-elem-softmax  : learn γ,b; g = sigmoid(γ ⊙ h^w + b) (state-elem gate)

Falsification (pre-registered): C4/C4b fails if no gate arm beats Skip-sub on
stream same-token-set CE on >=8/10 seeds, or if every gate arm is worse than
B-softmax on forgetting on >=8/10 seeds.

Output files:
  data/s51_paper_f_learned_gate_v2.csv (v2: adds per-channel top-k
                 selection histograms; v1 files untouched)
  data/s51_paper_f_learned_gate_v2.json

Usage: python s51_paper_f_learned_gate.py [--quick] [--sequential]
                 [--only Gate-C-topk-softmax,Gate-C-L1-softmax]

C4 (unconstrained gates) and C4b/M2b (sparsity-pressured):
  Gate-C-topk-softmax : g = sigmoid(Cx+b), keep top-k=32 channels
  Gate-C-L1-softmax   : same + λ||g||_1 on gate grads (λ=0.01)
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
    SOFTMAX_T, SOFTMAX_LR, SOFTMAX_EPS, GRAD_CLIP, FLOOR_EPS,
    NOISE_DIM, SUBSTATE_DIM, NOISE_SEED_OFF,
    softmax_np, ce_from_p,
)

torch.set_num_threads(1)

ARMS = {
    'B-softmax-sgd':      {'feat': 'proj'},
    'Skip-sub-softmax':   {'feat': 'sub'},
    'B-noise-softmax':    {'feat': 'noise'},
    'Gate-C-softmax':     {'feat': 'gate_c'},
    'Gate-static-softmax': {'feat': 'gate_static'},
    'Gate-elem-softmax':  {'feat': 'gate_elem'},
    # M2b / C4b: sparsity-pressured learned selectivity
    'Gate-C-topk-softmax': {'feat': 'gate_c_topk'},
    'Gate-C-L1-softmax':   {'feat': 'gate_c_l1'},
}
TOPK_K = SUBSTATE_DIM          # match Skip-sub width
L1_LAMBDA = 0.01               # on gate pre-activation grads

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's51_paper_f_learned_gate_v2.csv')
JSON_PATH = os.path.join(DATA_DIR, 's51_paper_f_learned_gate_v2.json')


def run_single(args):
    """(arm, seed) -> metrics. Top-level; unbuffered."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    feat = ARMS[arm]['feat']
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    stream, domains = gen_drift_stream(seed)
    T = stream.shape[0]
    noise_rng = np.random.RandomState(
        seed * SEED_SCALE + SEED_OFF + NOISE_SEED_OFF)

    # Gate / extra params (learned for gate_* arms); store is mutated in-place
    gstore = {}
    if feat in ('gate_c', 'gate_c_topk', 'gate_c_l1'):
        gstore['C'] = torch.randn(N_STATE, N_STATE, dtype=torch.float64) / np.sqrt(N_STATE)
        gstore['b'] = torch.zeros(N_STATE, dtype=torch.float64)
    elif feat == 'gate_static':
        gstore['a'] = torch.zeros(N_STATE, dtype=torch.float64)
    elif feat == 'gate_elem':
        gstore['gam'] = torch.zeros(N_STATE, dtype=torch.float64)
        gstore['b'] = torch.zeros(N_STATE, dtype=torch.float64)
    g_keys = list(gstore.keys())

    def feat_dim():
        # x = B[:, tok] is N_STATE-dimensional (input projection column)
        if feat == 'proj':
            return N_STATE + 1
        if feat == 'noise':
            return N_STATE + NOISE_DIM + 1
        if feat == 'sub':
            return SUBSTATE_DIM + N_STATE + 1
        return 2 * N_STATE + 1  # gate_*: [hw*g; x; 1]

    F = feat_dim()
    W = torch.zeros(VOCAB, F, dtype=torch.float64)
    W[:, -1] = 1.0 / VOCAB
    GW = torch.zeros_like(W)
    GG = {k: torch.zeros_like(gstore[k]) for k in g_keys}

    def build_phi(hw, prev_tok, noise=None):
        x = B[:, prev_tok]
        if feat == 'proj':
            parts = [x]
        elif feat == 'noise':
            parts = [x, torch.as_tensor(noise, dtype=torch.float64)]
        elif feat == 'sub':
            parts = [hw[:SUBSTATE_DIM], x]
        elif feat in ('gate_c', 'gate_c_topk', 'gate_c_l1'):
            g = torch.sigmoid(gstore['C'] @ x + gstore['b'])
            if feat == 'gate_c_topk':
                # hard cardinality: keep top-k gate values
                idx = torch.topk(g, k=TOPK_K).indices
                mask = torch.zeros_like(g)
                mask[idx] = 1.0
                g = g * mask
            parts = [hw * g, x]
        elif feat == 'gate_static':
            g = torch.sigmoid(gstore['a'])
            parts = [hw * g, x]
        else:  # gate_elem
            g = torch.sigmoid(gstore['gam'] * hw + gstore['b'])
            parts = [hw * g, x]
        parts.append(torch.ones(1, dtype=torch.float64))
        return torch.cat(parts)

    h = torch.zeros(N_STATE, dtype=torch.float64)
    ce = np.empty(T, dtype=np.float64)
    ce[:] = np.nan
    nfloor = 0
    hs = np.empty((T - 1, F), dtype=np.float64)

    # per-channel top-k selection histogram (channel-utilization evidence;
    # v2 addition: counts which state channels the top-k gate picks over time)
    chan_count = np.zeros(N_STATE, dtype=np.int64) if feat == 'gate_c_topk' else None
    chan_tokens = 0

    for t in range(1, T):
        h = A * h + B[:, stream[t - 1]]
        hw = h * scale
        noise_t = None
        if feat == 'noise':
            noise_t = noise_rng.randn(NOISE_DIM)
        phi = build_phi(hw, stream[t - 1], noise_t)
        hs[t - 1] = phi.detach().numpy()
        tgt = int(stream[t])

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
        GW = GW + g_t * g_t
        W = W - SOFTMAX_LR * g_t / (torch.sqrt(GW) + SOFTMAX_EPS)

        # Gate parameter grads: dL/dphi * dphi/dgate params
        if g_keys:
            x = B[:, stream[t - 1]]
            gw = torch.tensor(grad, dtype=torch.float64)
            d_hwg = W[:, :N_STATE].T @ gw
            grads = {}
            if feat in ('gate_c', 'gate_c_topk', 'gate_c_l1'):
                pre = gstore['C'] @ x + gstore['b']
                g2 = torch.sigmoid(pre)
                dpre = d_hwg * hw * g2 * (1 - g2)
                if feat == 'gate_c_topk':
                    # straight-through style: zero grads on non-topk channels
                    idx = torch.topk(g2, k=TOPK_K).indices
                    mask = torch.zeros_like(g2)
                    mask[idx] = 1.0
                    dpre = dpre * mask
                    if chan_count is not None:
                        chan_count[idx.numpy()] += 1
                        chan_tokens += 1
                elif feat == 'gate_c_l1':
                    # λ ||g||_1 on the gate values
                    dpre = dpre + L1_LAMBDA * torch.sign(g2) * g2 * (1 - g2)
                grads['C'] = torch.outer(dpre, x)
                grads['b'] = dpre
            elif feat == 'gate_static':
                g2 = torch.sigmoid(gstore['a'])
                grads['a'] = d_hwg * hw * g2 * (1 - g2)
            else:
                g2 = torch.sigmoid(gstore['gam'] * hw + gstore['b'])
                dpre = d_hwg * hw * g2 * (1 - g2)
                grads['gam'] = dpre * hw
                grads['b'] = dpre
            for k in g_keys:
                gr = grads[k]
                GG[k] = GG[k] + gr * gr
                gstore[k] = gstore[k] - SOFTMAX_LR * gr / (
                    torch.sqrt(GG[k]) + SOFTMAX_EPS)

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    floor_frac = float(nfloor) / (T - 1)

    # pooled ridge diagnostic (not a bound)
    Y = np.zeros((T - 1, VOCAB))
    Y[np.arange(T - 1), stream[1:]] = 1.0
    X = hs
    XtX = X.T @ X + np.eye(F)
    Wr = Y.T @ X @ np.linalg.inv(XtX)
    yhat = X @ Wr.T
    p_or = np.clip(
        np.take_along_axis(yhat, stream[1:][:, None], axis=1).ravel(),
        FLOOR_EPS, 1.0)
    oracle_ppl = float(np.exp(np.mean(-np.log(p_or))))

    # Forgetting holdout (frozen final gate/W)
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
            if feat == 'noise':
                noise_h = noise_rng.randn(NOISE_DIM)
            phi = build_phi(hh * scale, hold[t - 1], noise_h)
            tgt = int(hold[t])
            n_hold += 1
            p = softmax_np((W @ phi).numpy())
            pt = float(p[tgt])
            if pt <= FLOOR_EPS:
                n_hold_floor += 1
            c = ce_from_p(p, tgt)
            ces.append(c)
            hold_ce_all.append(c)
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))
    forgetting_med = float(np.median(forgets))
    forget_floor_frac = float(n_hold_floor) / n_hold if n_hold else float('nan')

    # gate stats for gate_* arms
    gate_l1 = float('nan')
    gate_active = float('nan')
    if feat in ('gate_c', 'gate_static', 'gate_elem', 'gate_c_topk', 'gate_c_l1'):
        key = 'a' if feat == 'gate_static' else 'b'
        g0 = torch.sigmoid(gstore[key]).numpy()
        gate_l1 = float(np.mean(g0))
        if feat == 'gate_c_topk':
            gate_active = TOPK_K / float(N_STATE)
        else:
            gate_active = float((g0 > 0.5).mean())

    # channel-utilization summary (v2): per-channel share of top-k
    # selections over the stream, plus the never-selected count.
    chan_util_min = float('nan')
    chan_util_gini = float('nan')
    chan_never_selected = -1
    if chan_count is not None and chan_tokens > 0:
        share = chan_count / float(chan_tokens)          # sums to k/N = 0.25
        chan_util_min = float(share.min())
        sorted_share = np.sort(share)
        n_ch = len(share)
        cum = np.cumsum(sorted_share)
        chan_util_gini = float((n_ch + 1 - 2 * (cum / cum[-1]).sum()) / n_ch)
        chan_never_selected = int((chan_count == 0).sum())

    save_stream_and_holdout(
        's51', arm, seed, 'x', ce, ce,
        np.asarray(hold_ce_all, dtype=np.float64),
        np.asarray(hold_ce_all, dtype=np.float64))

    return {
        'arm': arm, 'seed': seed, 'feat': feat,
        'stream_ppl': stream_ppl,
        'floor_frac': floor_frac,
        'forgetting_ppl': forgetting_ppl,
        'forgetting_ppl_med': forgetting_med,
        'forget_floor_frac': forget_floor_frac,
        'oracle_ppl': oracle_ppl,
        'chan_util_min': chan_util_min,
        'chan_util_gini': chan_util_gini,
        'chan_never_selected': chan_never_selected,
        'gate_mean': gate_l1,
        'gate_frac_gt_half': gate_active,
        'n_params_gate': int(sum(gstore[k].numel() for k in g_keys)),
        'runtime_s': time.time() - t0,
    }


def print_table(results):
    print('\n' + '=' * 100)
    print('S51 RESULTS (Paper F C4/C4b): learned gate vs calibrated baselines')
    print('=' * 100)
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
        fl = np.mean([r['floor_frac'] for r in rs])
        gm = np.mean([r['gate_mean'] for r in rs
                      if np.isfinite(r['gate_mean'])])
        print(f" {arm:>22} n={len(rs):2d}  stream {ms:8.3f}  "
              f"forget mean/med {mf:8.3f}/{md:7.3f}  floor {fl * 100:.3f}%"
              + (f"  gate_mean {gm:.3f}" if np.isfinite(gm) else ''))
    return by


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t0 = time.time()
    only = None
    if '--only' in sys.argv:
        i = sys.argv.index('--only')
        only = set(sys.argv[i + 1].split(','))
        unknown = only - set(ARMS)
        if unknown:
            raise SystemExit(f'unknown arms: {sorted(unknown)}')
    arm_list = [a for a in ARMS if only is None or a in only]
    n_seeds = 1 if quick else N_SEEDS
    all_args = [(a, s) for a in arm_list for s in range(n_seeds)]
    print(f"[{time.strftime('%H:%M:%S')}] START S51 "
          f"(quick={quick}, sequential={sequential}, arms={arm_list})")
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
    fieldnames = ['arm', 'seed', 'feat', 'stream_ppl', 'floor_frac',
                  'forgetting_ppl', 'forgetting_ppl_med', 'forget_floor_frac', 'chan_util_min', 'chan_util_gini',
                  'chan_never_selected',
                  'oracle_ppl', 'gate_mean', 'gate_frac_gt_half',
                  'n_params_gate', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    rows = list(results)
    if only is not None and not quick and os.path.exists(out_csv):
        keep = {(r['arm'], int(r['seed'])) for r in results}
        with open(out_csv, 'r', newline='') as f:
            for old in csv.DictReader(f):
                key = (old['arm'], int(old['seed']))
                if key not in keep:
                    old['seed'] = int(old['seed'])
                    for k in ('stream_ppl', 'floor_frac', 'forgetting_ppl',
                              'forgetting_ppl_med', 'forget_floor_frac',
                              'oracle_ppl', 'gate_mean', 'gate_frac_gt_half',
                              'n_params_gate', 'runtime_s'):
                        if old.get(k) not in (None, ''):
                            try:
                                old[k] = float(old[k])
                            except ValueError:
                                pass
                    rows.append(old)
        rows.sort(key=lambda r: (r['arm'], int(r['seed'])))
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    meta = {
        'script': 's51_paper_f_learned_gate',
        'claim': 'C4/C4b',
        'softmax_lr': SOFTMAX_LR,
        'grad_clip': GRAD_CLIP,
        'topk_k': TOPK_K,
        'l1_lambda': L1_LAMBDA,
        'n_seeds': n_seeds,
        'quick': quick,
        'arms': ARMS,
        'elapsed_s': time.time() - t0,
        'results': rows,
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] DONE S51 -> {out_csv} "
          f"({time.time() - t0:.1f}s)")


if __name__ == '__main__':
    main()
