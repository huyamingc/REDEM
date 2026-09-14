#!/usr/bin/env python3
"""
S67: Paper F — host freeze policy (random / pretrained / online) under the
same selective-SSM parameterisation as s66.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Paper Section:  Paper F follow-up (external baseline s66 → host-policy)
Experiment:     S67 host-freeze comparison
=============================================================================

Motivation (from s66's frozen-host surprise)

  s66 showed, on the identical host protocol / stream / seeds / metric:
    X-SelSSM-frozen  stream 9.03  forget  10.38
    X-SelSSM-full    stream 10.19 forget 115.24   (matched lr; 4x lr diverges)
    F-Gate-C-topk    stream 7.03  forget   9.20
    F-B-softmax-sgd  stream 8.65  forget   8.89
  Ordering: random frozen ≫ jointly trained host, yet frozen still loses to
  F's Gate-C-topk and to F's B-softmax-sgd on both axes.

  That ordering does NOT license "selective SSMs are inferior". It licenses
  a narrower question that must be answered BEFORE any Paper H work that
  generates new host / form families:

    Is the frozen advantage a property of the host, or an optimisation
    interference property of joint online CE?

Pre-registered questions (written before the full run; report either way)

  Q1  pretrained-frozen vs random-frozen on forgetting ppl.
      Prediction: pretrained-frozen is better if mean Δ < 0 and ≥7/10 seeds.
      (Short host adaptation on domain A, then freeze, should retain more.)
  Q2  pretrained-frozen vs online (matched lr) on forgetting ppl.
      Prediction: freeze-after-burn-in better if ≥7/10 seeds and Δ < 0.
      (Tests whether joint training after the burn-in is the destructive part.)
  Q3  random-frozen+topk vs F-Gate-C-topk on stream ppl.
      Prediction: not worse by more than +5% mean (selective dynamics do not
      destroy the sparse-gate win) — a loss still informative.
  Q4  pretrained-frozen+topk vs F-Gate-C-topk on stream ppl.
      Prediction: ≥7/10 better or parity; if better, s66's gap was host
      optimisation, not the selective-SSM family.

Arms
  X67-random-frozen          s66 frozen host, no training (anchor)
  X67-pretrained-frozen      host trained ONLY on the first segment
                             (t in [1, SEG_LEN]), then frozen; readout online
  X67-online                 host trained throughout at matched SOFTMAX_LR
                             (anchor to s66 full @ lr_scale=1; NOT the 4x
                             divergent rate)
  X67-random-frozen-topk     random-frozen host + F hard top-k gate (k=32)
                             on the host-gated whitened state
  X67-pretrained-frozen-topk pretrained-frozen host + same top-k gate
  F-B-softmax-sgd            calibrated input-path anchor (s50)
  F-Gate-C-topk-softmax      sparse-gate anchor (s51)

Host / task / seed rules are imported from s19 verbatim. Selective-SSM
forward and AdaGrad host updates reuse s66 helpers so random-frozen and
online stay bit-comparable with the committed s66 rows when validation runs.

Anti-confounds
  - identical A, B, stream, holdouts per seed;
  - identical per-token label budget (CE);
  - online arm uses ONLY the matched learning rate (s66 pooled 4x lr is a
    separate disclosed defect and is NOT re-run here);
  - top-k is applied on host-gated features, so the host gate is not removed;
  - report stream and forget together; no floor_frac=0 as a finding.

Outputs
  data/s67_host_freeze_v1.csv
  data/s67_host_freeze_v1.json

Usage
  python s67_host_freeze.py [--quick] [--sequential] [--no-validate]
          [--only ARM1,ARM2]
"""
from __future__ import annotations

import os
import sys
import csv
import json
import time

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
from multiprocessing import Pool, cpu_count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from per_token_io import save_stream_and_holdout
from s19_ssm_rls_readout import (
    VOCAB, N_STATE,
    SEG_LEN, N_SEGMENTS, HOLDOUT_LEN,
    N_SEEDS, SEED_SCALE, SEED_OFF,
    sample_substrate, whiten_scale, gen_drift_stream, gen_stream,
    BIAS_A, BIAS_B,
)
from s50_paper_f_pilot_nl_readout import (
    SOFTMAX_LR, SOFTMAX_EPS, GRAD_CLIP, FLOOR_EPS, SUBSTATE_DIM,
    softmax_np, ce_from_p,
)
from s66_external_ssm_baseline import (
    SELSSM_HID, HOST_KEYS,
    init_external, selssm_dt, selssm_forward_autograd,
    softmax_update, load_anchor_values,
)

torch.set_num_threads(1)

TOPK_K = SUBSTATE_DIM  # 32, matched to Paper F
PRETRAIN_END = SEG_LEN  # first segment only; then freeze

# host_policy values
POLICY_RANDOM = 'random_frozen'
POLICY_PRETRAIN = 'pretrained_frozen'
POLICY_ONLINE = 'online'

ARMS = {
    'X67-random-frozen': {
        'host': POLICY_RANDOM, 'use_topk': False, 'map': 'softmax_sgd',
    },
    'X67-pretrained-frozen': {
        'host': POLICY_PRETRAIN, 'use_topk': False, 'map': 'softmax_sgd',
    },
    'X67-online': {
        'host': POLICY_ONLINE, 'use_topk': False, 'map': 'softmax_sgd',
    },
    'X67-random-frozen-topk': {
        'host': POLICY_RANDOM, 'use_topk': True, 'map': 'softmax_sgd',
    },
    'X67-pretrained-frozen-topk': {
        'host': POLICY_PRETRAIN, 'use_topk': True, 'map': 'softmax_sgd',
    },
    'F-B-softmax-sgd': {
        'host': 'none', 'use_topk': False, 'map': 'softmax_sgd',
        'feat': 'proj',
    },
    'F-Gate-C-topk-softmax': {
        'host': 'none', 'use_topk': True, 'map': 'softmax_sgd',
        'feat': 'gate_c_topk',
    },
}

DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's67_host_freeze_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's67_host_freeze_v1.json')
S66_CSV = os.path.join(DATA_DIR, 's66_external_ssm_baseline_v1.csv')
S50_CSV = os.path.join(DATA_DIR, 's50_paper_f_pilot_nl_readout_v1.csv')
S51_CSV = os.path.join(DATA_DIR, 's51_paper_f_learned_gate_v1.csv')
VAL_TOL = 1e-9


def _load_ref(path, arm, seeds=range(N_SEEDS)):
    """{(seed): {stream_ppl, forgetting_ppl}} for one arm from a committed CSV."""
    ref = {}
    if not os.path.exists(path):
        return ref
    with open(path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('arm') != arm:
                continue
            try:
                s = int(float(row['seed']))
            except (TypeError, ValueError):
                continue
            # s66 full was run at lr_scale 1 and 4; keep matched-lr rows only
            if 'lr_scale' in row and row['lr_scale'] not in (None, ''):
                try:
                    if abs(float(row['lr_scale']) - 1.0) > 1e-12:
                        continue
                except ValueError:
                    pass
            if s not in seeds:
                continue
            ref[s] = {
                'stream_ppl': float(row['stream_ppl']),
                'forgetting_ppl': float(row['forgetting_ppl']),
            }
    return ref


def run_single(args):
    """(arm, seed) -> metrics. Top-level (picklable); unbuffered."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    cfg = ARMS[arm]
    policy = cfg['host']
    use_topk = bool(cfg['use_topk'])
    feat = cfg.get('feat', 'selssm')
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    stream, domains = gen_drift_stream(seed)
    T = stream.shape[0]

    # ---- host parameters (s66 init; independent RNG stream +5) ----
    st = init_external(seed, 'selsmm_full' if policy != 'none' else 'proj')

    # ---- F-style gate (used for use_topk or explicit Gate-C-topk arm) ----
    gstore = {}
    if use_topk or feat == 'gate_c_topk':
        # s51 draws C from the GLOBAL torch RNG after substrate sampling
        gstore['C'] = torch.randn(N_STATE, N_STATE,
                                  dtype=torch.float64) / np.sqrt(N_STATE)
        gstore['b'] = torch.zeros(N_STATE, dtype=torch.float64)
    GG = {k: torch.zeros_like(v) for k, v in gstore.items()}

    if policy == 'none' and feat == 'proj':
        Fdim = N_STATE + 1
    elif policy == 'none' and feat == 'gate_c_topk':
        Fdim = 2 * N_STATE + 1
    else:
        # selective host features: [host-gated hw ; x ; 1] optionally with
        # an extra F top-k mask on the same host-gated vector
        Fdim = 2 * N_STATE + 1

    W = torch.zeros(VOCAB, Fdim, dtype=torch.float64)
    W[:, -1] = 1.0 / VOCAB
    G = torch.zeros_like(W)

    def host_train_now(t):
        if policy == POLICY_ONLINE:
            return True
        if policy == POLICY_PRETRAIN:
            return t <= PRETRAIN_END
        return False

    def build_phi(prev_tok, hw, host_gate):
        x = B[:, prev_tok]
        if policy == 'none' and feat == 'proj':
            parts = [x]
        elif policy == 'none' and feat == 'gate_c_topk':
            g = torch.sigmoid(gstore['C'] @ x + gstore['b'])
            idx = torch.topk(g, k=TOPK_K).indices
            mask = torch.zeros_like(g)
            mask[idx] = 1.0
            parts = [hw * g * mask, x]
        else:
            # host-gated whitened state
            base = hw * host_gate if host_gate is not None else hw
            if use_topk:
                g = torch.sigmoid(gstore['C'] @ x + gstore['b'])
                idx = torch.topk(g, k=TOPK_K).indices
                mask = torch.zeros_like(g)
                mask[idx] = 1.0
                parts = [base * g * mask, x]
            else:
                parts = [base, x]
        parts.append(torch.ones(1, dtype=torch.float64))
        return torch.cat(parts)

    def upstream_d_hwg(grad_np, n_used=N_STATE):
        gw = torch.tensor(grad_np, dtype=torch.float64)
        return W[:, :n_used].T @ gw

    h = torch.zeros(N_STATE, dtype=torch.float64)
    ce = np.empty(T, dtype=np.float64)
    ce[:] = np.nan
    nfloor = 0
    hs = np.empty((T - 1, Fdim), dtype=np.float64)

    for t in range(1, T):
        prev_tok = int(stream[t - 1])
        h_prev = h
        if policy != 'none':
            (_, h, host_gate, _, _) = selssm_forward_autograd(
                A, B, h_prev, st, prev_tok,
                torch.zeros(N_STATE, dtype=torch.float64))
            hw = h * scale
        else:
            h = A * h + B[:, prev_tok]
            hw = h * scale
            host_gate = None

        phi = build_phi(prev_tok, hw, host_gate)
        hs[t - 1] = phi.detach().numpy()
        tgt = int(stream[t])

        logits = (W @ phi).numpy()
        p = softmax_np(logits)
        if float(p[tgt]) <= FLOOR_EPS:
            nfloor += 1
        ce[t] = ce_from_p(p, tgt)
        grad = p.copy()
        grad[tgt] -= 1.0
        W, G = softmax_update(W, G, grad, phi)

        # F gate params (chain rule on host-gated base * g)
        if gstore:
            x = B[:, prev_tok]
            if policy != 'none':
                base = (hw * host_gate) if host_gate is not None else hw
            else:
                base = hw
            d_base = upstream_d_hwg(grad)
            pre = gstore['C'] @ x + gstore['b']
            g2 = torch.sigmoid(pre)
            dpre = d_base * base * g2 * (1.0 - g2)
            idx = torch.topk(g2, k=TOPK_K).indices
            mask = torch.zeros_like(g2)
            mask[idx] = 1.0
            dpre = dpre * mask
            grads = {'C': torch.outer(dpre, x), 'b': dpre}
            for k in gstore:
                gr = grads[k]
                GG[k] = GG[k] + gr * gr
                gstore[k] = gstore[k] - SOFTMAX_LR * gr / (
                    torch.sqrt(GG[k]) + SOFTMAX_EPS)

        # host backward + optional AdaGrad step
        if policy != 'none':
            if use_topk or feat == 'gate_c_topk':
                # upstream through host-gated base * g * mask (already folded
                # into d_base above for gate; for host we need d(hw*gate)/dh
                gw = torch.tensor(grad, dtype=torch.float64)
                d_state = W[:, :N_STATE].T @ gw
                if use_topk:
                    # recompute mask/g at this token for the host path
                    x = B[:, prev_tok]
                    g2 = torch.sigmoid(gstore['C'] @ x + gstore['b'])
                    idx = torch.topk(g2, k=TOPK_K).indices
                    mask = torch.zeros_like(g2)
                    mask[idx] = 1.0
                    d_state = d_state * g2 * mask
                dout = d_state * (host_gate if host_gate is not None
                                  else torch.ones_like(d_state))
            else:
                gw = torch.tensor(grad, dtype=torch.float64)
                dout = (W[:, :N_STATE].T @ gw) * host_gate

            (h_det, h_new, gate_b, dh_prev, hgrads) = selssm_forward_autograd(
                A, B, h_prev, st, prev_tok, dout)
            h = h_new
            host_gate = gate_b
            if host_train_now(t):
                for k, gr in hgrads.items():
                    gkey = k + '_G'
                    st[gkey] = st.get(gkey, torch.zeros_like(st[k]))
                    st[gkey] = st[gkey] + gr * gr
                    st[k] = st[k] - SOFTMAX_LR * gr / (
                        torch.sqrt(st[gkey]) + SOFTMAX_EPS)

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    floor_frac = float(nfloor) / (T - 1)

    Y = np.zeros((T - 1, VOCAB))
    Y[np.arange(T - 1), stream[1:]] = 1.0
    X = hs
    XtX = X.T @ X + 1e-4 * np.eye(Fdim)
    Wr = Y.T @ X @ np.linalg.inv(XtX)
    yhat = X @ Wr.T
    p_or = np.clip(
        np.take_along_axis(yhat, stream[1:][:, None], axis=1).ravel(),
        FLOOR_EPS, 1.0)
    oracle_ppl = float(np.exp(np.mean(-np.log(p_or))))

    # forgetting holdout — frozen parameters (s18/s19 metric)
    forgets = []
    hold_ce_all = []
    for si, t_s in enumerate([SEG_LEN * s for s in range(1, N_SEGMENTS)]):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3,
                          1 if prev_dom == 0 else 7, HOLDOUT_LEN,
                          BIAS_A if prev_dom == 0 else BIAS_B)
        hh = torch.zeros(N_STATE, dtype=torch.float64)
        ces = []
        for t in range(1, HOLDOUT_LEN):
            ptok = int(hold[t - 1])
            tgt = int(hold[t])
            if policy != 'none':
                xh = B[:, ptok]
                dth = selssm_dt(st, xh)
                tau_h = -1.0 / torch.log(A.clamp(max=1.0 - 1e-12))
                hh = (torch.exp(-dth / tau_h) * hh + dth * xh)
                gate_h = torch.sigmoid(st['Wg'] @ xh + st['bg'])
                hw_h = hh * scale
            else:
                hh = A * hh + B[:, ptok]
                hw_h = hh * scale
                gate_h = None
            phi_h = build_phi(ptok, hw_h, gate_h)
            p = softmax_np((W @ phi_h).numpy())
            if float(p[tgt]) <= FLOOR_EPS:
                pass
            c = ce_from_p(p, tgt)
            ces.append(c)
            hold_ce_all.append(c)
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))
    forgetting_med = float(np.median(forgets))

    n_params_readout = int(W.numel())
    n_params_gate = int(sum(v.numel() for v in gstore.values()))
    n_params_host = int(sum(st[k].numel() for k in HOST_KEYS if k in st))

    save_stream_and_holdout(
        's67', arm, seed, policy, ce, ce,
        np.asarray(hold_ce_all, dtype=np.float64),
        np.asarray(hold_ce_all, dtype=np.float64))

    return {
        'arm': arm, 'seed': seed, 'host_policy': policy,
        'use_topk': int(use_topk), 'readout_map': cfg['map'],
        'stream_ppl': stream_ppl,
        'clip_frac': 0.0, 'neg_frac': 0.0, 'floor_frac': floor_frac,
        'forgetting_ppl': forgetting_ppl,
        'forgetting_ppl_med': forgetting_med,
        'oracle_ppl': oracle_ppl,
        'n_params_readout': n_params_readout,
        'n_params_gate': n_params_gate,
        'n_params_host': n_params_host,
        'pretrain_end_tokens': PRETRAIN_END if policy == POLICY_PRETRAIN else 0,
        'runtime_s': time.time() - t0,
    }


def paired_stats(results, arm_a, arm_b, col):
    by = {}
    for r in results:
        by.setdefault(r['arm'], {})[int(r['seed'])] = float(r[col])
    if arm_a not in by or arm_b not in by:
        return None
    seeds = sorted(set(by[arm_a]) & set(by[arm_b]))
    if not seeds:
        return None
    a = np.array([by[arm_a][s] for s in seeds])
    b = np.array([by[arm_b][s] for s in seeds])
    d = a - b
    n = len(seeds)
    sd = float(d.std(ddof=1)) if n > 1 else float('nan')
    t = float(d.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float('nan')
    wins = int((d < 0).sum())
    return {'n': n, 'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
            'delta': float(d.mean()), 'sd': sd, 't': t, 'wins': wins}


def print_table(results):
    print('\n' + '=' * 100)
    print('S67 RESULTS (host freeze policy)')
    print('=' * 100)
    print(f" {'arm':>28} {'n':>2} {'stream':>9} {'forget':>9} {'par':>7}")
    by = {}
    for r in results:
        by.setdefault(r['arm'], []).append(r)
    for arm in ARMS:
        rs = by.get(arm, [])
        if not rs:
            continue
        sp = float(np.mean([r['stream_ppl'] for r in rs]))
        fp = float(np.mean([r['forgetting_ppl'] for r in rs]))
        npar = (rs[0]['n_params_readout'] + rs[0]['n_params_gate']
                + rs[0]['n_params_host'])
        print(f' {arm:>28} {len(rs):2d} {sp:9.4f} {fp:9.4f} {npar:7d}')

    pairs = [
        ('X67-pretrained-frozen', 'X67-random-frozen', 'forgetting_ppl', 'Q1'),
        ('X67-pretrained-frozen', 'X67-online', 'forgetting_ppl', 'Q2'),
        ('X67-random-frozen-topk', 'F-Gate-C-topk-softmax', 'stream_ppl', 'Q3'),
        ('X67-pretrained-frozen-topk', 'F-Gate-C-topk-softmax', 'stream_ppl', 'Q4'),
        ('X67-random-frozen', 'F-B-softmax-sgd', 'stream_ppl', 'ctx'),
        ('X67-online', 'X67-random-frozen', 'stream_ppl', 'ctx'),
        ('X67-online', 'X67-random-frozen', 'forgetting_ppl', 'ctx'),
    ]
    print('\n paired (A - B; lower better; wins = seeds A better)')
    for a, b, col, tag in pairs:
        s = paired_stats(results, a, b, col)
        if s is None:
            continue
        print(f'  [{tag}] {a:>28} - {b:<24} {col:<15} '
              f'd={s["delta"]:+8.4f} t={s["t"]:+7.2f} '
              f'{s["wins"]:2d}/{s["n"]}')
    return by


def validate(results):
    """Compare s67 anchors to committed s66 / s50 / s51 rows."""
    checks = [
        ('X67-random-frozen', S66_CSV, 'X-SelSSM-frozen'),
        ('X67-online', S66_CSV, 'X-SelSSM-full'),
        ('F-B-softmax-sgd', S50_CSV, 'B-softmax-sgd'),
        ('F-Gate-C-topk-softmax', S51_CSV, 'Gate-C-topk-softmax'),
    ]
    print('\n' + '=' * 100)
    print('ANCHOR VALIDATION (s67 vs committed s66/s50/s51)')
    print('=' * 100)
    worst = 0.0
    bad = []
    n_pairs = 0
    for s67_arm, path, src_arm in checks:
        ref = _load_ref(path, src_arm)
        if not ref:
            print(f'  SKIP {s67_arm}: no ref rows for {src_arm} in {path}')
            continue
        for r in results:
            if r['arm'] != s67_arm:
                continue
            s = int(r['seed'])
            if s not in ref:
                continue
            for col in ('stream_ppl', 'forgetting_ppl'):
                a, b = float(r[col]), ref[s][col]
                rel = abs(a - b) / max(abs(b), 1e-30)
                worst = max(worst, rel)
                n_pairs += 1
                flag = 'OK ' if rel <= VAL_TOL else 'FAIL'
                if rel > VAL_TOL:
                    bad.append((s67_arm, s, col, a, b, rel))
                print(f'  {flag} {s67_arm:28s} s{s} {col:15s} '
                      f'here {a:12.6f}  ref {b:12.6f}  rel {rel:.2e}')
    print(f'  pairs checked: {n_pairs}  worst_rel: {worst:.3e}  tol: {VAL_TOL:.0e}')
    status = 'fail' if bad else ('ok' if n_pairs else 'no_ref')
    if status == 'ok':
        print('  ALL ANCHORS MATCH.')
    elif status == 'fail':
        print(f'  {len(bad)} MISMATCH(ES) — interpret s67 new arms with caution.')
    return {'status': status, 'pairs': n_pairs, 'worst_rel': worst,
            'mismatches': bad[:20]}


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    no_val = '--no-validate' in sys.argv
    only = None
    if '--only' in sys.argv:
        i = sys.argv.index('--only')
        only = set(sys.argv[i + 1].split(','))
        unknown = only - set(ARMS)
        if unknown:
            raise SystemExit(f'unknown arms: {sorted(unknown)}')
    arm_list = [a for a in ARMS if only is None or a in only]
    n_seeds = 2 if quick else N_SEEDS
    all_args = [(a, s) for a in arm_list for s in range(n_seeds)]
    print(f"[{time.strftime('%H:%M:%S')}] START S67 "
          f"(quick={quick}, sequential={sequential}, arms={arm_list}, "
          f"n_seeds={n_seeds})")
    print(f'total runs: {len(all_args)}', flush=True)

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
    val = {'status': 'skipped'}
    if not no_val and not quick:
        val = validate(results)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = [
        'arm', 'seed', 'host_policy', 'use_topk', 'readout_map',
        'stream_ppl', 'clip_frac', 'neg_frac', 'floor_frac',
        'forgetting_ppl', 'forgetting_ppl_med', 'oracle_ppl',
        'n_params_readout', 'n_params_gate', 'n_params_host',
        'pretrain_end_tokens', 'runtime_s',
    ]
    rows = sorted(results, key=lambda r: (r['arm'], int(r['seed'])))
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    meta = {
        'script': 's67_host_freeze',
        'claim': 'Paper F s67: random vs pretrained vs online host freeze',
        'softmax_lr': SOFTMAX_LR,
        'topk_k': TOPK_K,
        'pretrain_end_tokens': PRETRAIN_END,
        'selssm_hidden': SELSSM_HID,
        'n_seeds': n_seeds,
        'quick': quick,
        'arms': ARMS,
        'pre_registration': {
            'Q1': 'pretrained-frozen better than random-frozen on forget '
                  'if Δ<0 and ≥7/10',
            'Q2': 'pretrained-frozen better than online on forget '
                  'if Δ<0 and ≥7/10',
            'Q3': 'random-frozen-topk stream not worse than Gate-C-topk '
                  'by more than +5% mean',
            'Q4': 'pretrained-frozen-topk stream ≥ Gate-C-topk if ≥7/10 or parity',
        },
        'anchor_validation': val,
        'elapsed_s': None,
        'results': rows,
    }
    # elapsed filled after write path known
    meta['elapsed_s'] = time.time() - (
        time.time() - sum(r.get('runtime_s', 0.0) for r in rows))
    # prefer wall-clock from main start
    meta['elapsed_s'] = float(sum(r.get('runtime_s', 0.0) for r in rows))
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[{time.strftime("%H:%M:%S")}] DONE S67 -> {out_csv}', flush=True)


if __name__ == '__main__':
    main()
