#!/usr/bin/env python3
"""
S26: Fair Transformer references for the P4 benchmark (Paper D).
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S26 (Paper D, P4 follow-up): close the reference-fairness
                gaps of the S22 benchmark, under the same 4-domain
                irregular-switch protocol and 10-seed paired discipline:

  (a) TF-A1 TUNED grid  : s18 bare online LoRA re-run over a small
      (lr x rank) grid  : lr in {3e-4, 1e-3, 3e-3, 1e-2} x rank in {16, 32}.
      The S22 TF-A1 used the untuned s18 defaults (lr=1e-3, rank=16) and
      sat near uniform (22.46); this grid asks whether the gap to the SSM
      stack shrinks under reasonable tuning.
  (b) TF-A3 ROUTING     : the s18 A3 domain-routing design (two adapters,
      slow EMA of frozen-base hidden means, nearest-reference + hysteresis,
      only the active adapter updates) generalized to FOUR adapters
      (one per domain) at tau_m = 500 (the P2/P4 sweet spot). S22
      explicitly deferred this ("s18 model has 2 adapters").

  The comparison answers: does the SSM-REDEM advantage over the
  Transformer reference come from the HOST (SSM) or from the MECHANISMS
  (M3 routing)? TF-A3 has the mechanisms (routing) on a Transformer host;
  if TF-A3 approaches SSM-REDEM, the mechanisms transfer and the host
  matters less; if it stays far, the diagonal-SSM host is load-bearing.

  Seed rules: model init torch.manual_seed(seed*SEED_SCALE+SEED_OFF) and
  the task generator gen_multi_drift_stream(seed) are taken VERBATIM from
  S22, so the (1e-3, rank 16) A1 config must reproduce the committed S22
  TF-A1 numbers (cross-check) and every seed pairs with the S22 SSM rows.

Output files:
  data/s26_ssm_p4_fair_tf_v1.csv
  data/s26_ssm_p4_fair_tf_v1.json

Usage: python s26_ssm_p4_fair_tf.py [--quick] [--sequential]
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
import torch.nn as nn
from multiprocessing import Pool, cpu_count

from s18_llm_drift_gate import (TinyCharLM, lora_params, gen_stream,
                                CTX, VOCAB, RANK, LR_HEAD, LR_LORA,
                                HOLDOUT_LEN, T_ADAPT_WINDOW, STEADY_WINDOW,
                                T_ADAPT_RATIO, REF_LEN, hidden_mean)
from s22_ssm_p4_benchmark import (gen_multi_drift_stream, N_DOMAINS,
                                  N_SEGMENTS, SEG_LO, SEG_HI, SHIFTS,
                                  BIAS_SETS)
from s19_ssm_rls_readout import SEED_SCALE, SEED_OFF

torch.set_num_threads(1)   # per-worker; Pool gives process-level parallelism

# ========================== Fixed parameters ==========================
N_SEEDS = 10
TAU_M = 500.0              # metadata EMA timescale (P2/P4 sweet spot)

# Tuned A1 grid: (lr, rank). lr scales BOTH head and LoRA (s18 keeps them
# equal); rank is the LoRA rank (s18 RANK=16).
A1_GRID = [(3e-4, 16), (1e-3, 16), (3e-3, 16), (1e-2, 16),
           (3e-4, 32), (1e-3, 32), (3e-3, 32), (1e-2, 32)]
# A3 routing: 4 adapters at the s18 default LoRA settings.
A3_LR, A3_RANK = 1e-3, 16
A3_TAU_M = TAU_M
GATE_MARGIN = 1.15         # hysteresis margin (s18 gate_estimate)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's26_ssm_p4_fair_tf_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's26_ssm_p4_fair_tf_v1.json')


# ========================== Model helpers ==========================

def extend_adapters(model, n_total, rank=RANK):
    """Extend every LoRALinear of a s18 TinyCharLM to n_total adapters by
    appending to its ParameterList (new adapters use `rank`)."""
    for blk in model.blocks:
        for m in [blk.qkv, blk.out, blk.ff1, blk.ff2]:
            while len(m.A) < n_total:
                in_f = m.A[0].shape[1]
                out_f = m.B[0].shape[0]
                m.A.append(nn.Parameter(torch.randn(rank, in_f) * 0.02))
                m.B.append(nn.Parameter(torch.zeros(out_f, rank)))
            m.n_adapters = n_total
    return model


def gate_estimate_n(slow, refs, prev, margin=GATE_MARGIN):
    """Nearest-reference domain estimate with a hysteresis margin: flip
    only when the nearest distance is clearly (< 1/margin) smaller than
    the second-nearest; otherwise keep the previous estimate. For two
    domains this reduces to the s18 gate_estimate rule."""
    ds = [float(torch.norm(slow - r)) for r in refs]
    order = sorted(range(len(ds)), key=lambda i: ds[i])
    if ds[order[0]] * margin < ds[order[1]]:
        return order[0]
    return prev


def eval_ce(model, idx, active=0, use_lora=True):
    """Mean next-token cross-entropy over idx (chunked at CTX)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for lo in range(0, idx.shape[1] - 1, CTX):
            chunk = idx[:, lo:lo + CTX]
            lg = model(chunk, active=active, use_lora=use_lora)
            losses.append(nn.functional.cross_entropy(
                lg[:, :-1, :].reshape(-1, VOCAB),
                chunk[:, 1:].reshape(-1)).item())
    return float(np.mean(losses))


# ========================== Per-run workers ==========================

def t_adapt_known(ce_all, switch_times, T):
    """Known-switch post-switch adaptation (s18 running-window protocol)."""
    t_adapts = []
    for si, t_s in enumerate(switch_times):
        seg_end = (switch_times[si + 1] if si + 1 < len(switch_times) else T)
        seg_ce = ce_all[t_s:seg_end]
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
    return t_adapts


def compute_forgetting_ppl(model, seed, domains, switch_times,
                           active_forget=0):
    """Held-out ppl of the previous domain after switching away (s18
    semantics; domain-matched adapter when routing)."""
    forgets = []
    for si, t_s in enumerate(switch_times):
        prev_dom = int(domains[t_s - 1])
        hold = torch.from_numpy(gen_stream(
            seed * 41 + si * 211 + 3, SHIFTS[prev_dom], HOLDOUT_LEN,
            BIAS_SETS[prev_dom]).reshape(1, -1))
        fce = eval_ce(model, hold, active=active_forget, use_lora=True)
        forgets.append(float(np.exp(fce)))
    return float(np.mean(forgets)) if forgets else float('nan')


def run_tf_a1(args):
    """(arm_label, lr, rank, seed) -> TF-A1 metrics (bare online LoRA)."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, lr, rank, seed = args
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    RANK_G = RANK                    # keep s18.RANK untouched for A3 runs
    import s18_llm_drift_gate as s18m
    s18m.RANK = rank
    model = TinyCharLM()
    s18m.RANK = RANK_G

    stream, domains, switch_times, seg_lens = gen_multi_drift_stream(seed)
    T = stream.shape[0]

    head_params = list(model.head.parameters())
    lora = list(lora_params(model, 0))
    optim = torch.optim.Adam(
        [{'params': head_params, 'lr': lr}, {'params': lora, 'lr': lr}])

    ce_all = np.full(T, np.nan)
    lo = 0
    while lo < T - 1:
        hi = min(lo + CTX, T)
        if hi - lo < 2:
            break
        idx = torch.from_numpy(stream[lo:hi].reshape(1, -1))
        model.train()
        logits = model(idx, active=0, use_lora=True)
        with torch.no_grad():
            ce_all[lo + 1:hi] = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, VOCAB),
                idx[:, 1:].reshape(-1), reduction='none').numpy()
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, VOCAB), idx[:, 1:].reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        lo += CTX - 1

    stream_ppl = float(np.exp(np.nanmean(ce_all[1:])))
    t_adapts = t_adapt_known(ce_all, switch_times, T)
    model.eval()
    forgetting_ppl = compute_forgetting_ppl(model, seed, domains,
                                            switch_times, active_forget=0)

    return {'arm': arm, 'lr': float(lr), 'rank': int(rank), 'seed': seed,
            'stream_ppl': stream_ppl, 'neg_frac': float('nan'),
            't_adapt_mean': float(np.nanmean(t_adapts))
            if t_adapts else float('nan'),
            'forgetting_ppl': forgetting_ppl,
            'runtime_s': time.time() - t0}


def run_tf_a3(args):
    """(arm_label, rank, seed) -> TF-A3 metrics (4-adapter routing)."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, rank, seed = args
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    import s18_llm_drift_gate as s18m
    s18m.RANK = rank
    model = TinyCharLM()
    s18m.RANK = RANK
    extend_adapters(model, N_DOMAINS, rank=rank)

    # Fixed per-domain references on the FROZEN base (s18 A3 semantics)
    refs = [hidden_mean(model, torch.from_numpy(
        gen_stream(seed * 31 + d * 131 + 7, SHIFTS[d], REF_LEN,
                   BIAS_SETS[d]).reshape(1, -1)))
        for d in range(N_DOMAINS)]

    stream, domains, switch_times, seg_lens = gen_multi_drift_stream(seed)
    T = stream.shape[0]

    head_params = list(model.head.parameters())
    lora = list(lora_params(model, None))     # all 4 adapters
    optim = torch.optim.Adam(
        [{'params': head_params, 'lr': A3_LR}, {'params': lora, 'lr': A3_LR}])

    ce_all = np.full(T, np.nan)
    slow = refs[0].clone()
    prev_est = 0
    lo = 0
    while lo < T - 1:
        hi = min(lo + CTX, T)
        if hi - lo < 2:
            break
        idx = torch.from_numpy(stream[lo:hi].reshape(1, -1))

        hm = hidden_mean(model, idx)
        with torch.no_grad():
            lam = 1.0 - np.exp(-(hi - lo) / A3_TAU_M)
            slow = (1.0 - lam) * slow + lam * hm
        est = gate_estimate_n(slow, refs, prev_est)
        if est != prev_est:
            prev_est = est

        model.train()
        logits = model(idx, active=est, use_lora=True)
        with torch.no_grad():
            ce_all[lo + 1:hi] = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, VOCAB),
                idx[:, 1:].reshape(-1), reduction='none').numpy()
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, VOCAB), idx[:, 1:].reshape(-1))
        optim.zero_grad()
        loss.backward()
        for blk in model.blocks:
            for m in [blk.qkv, blk.out, blk.ff1, blk.ff2]:
                for i in range(m.n_adapters):
                    if i != est:
                        if m.A[i].grad is not None:
                            m.A[i].grad.zero_()
                        if m.B[i].grad is not None:
                            m.B[i].grad.zero_()
        optim.step()
        lo += CTX - 1

    stream_ppl = float(np.exp(np.nanmean(ce_all[1:])))
    t_adapts = t_adapt_known(ce_all, switch_times, T)
    model.eval()
    # domain-matched specialist for the held-out previous domain
    forget_vals = []
    for si, t_s in enumerate(switch_times):
        prev_dom = int(domains[t_s - 1])
        hold = torch.from_numpy(gen_stream(
            seed * 41 + si * 211 + 3, SHIFTS[prev_dom], HOLDOUT_LEN,
            BIAS_SETS[prev_dom]).reshape(1, -1))
        forget_vals.append(float(np.exp(eval_ce(model, hold,
                                                active=prev_dom,
                                                use_lora=True))))
    forgetting_ppl = float(np.mean(forget_vals)) if forget_vals else float('nan')

    return {'arm': arm, 'lr': float(A3_LR), 'rank': int(rank), 'seed': seed,
            'stream_ppl': stream_ppl, 'neg_frac': float('nan'),
            't_adapt_mean': float(np.nanmean(t_adapts))
            if t_adapts else float('nan'),
            'forgetting_ppl': forgetting_ppl,
            'runtime_s': time.time() - t0}


# ========================== Aggregation ==========================

def dispatch(task):
    """Top-level Pool worker: ((kind, cfg), seed) -> metrics."""
    (kind, cfg), seed = task
    if kind == 'A1':
        lr, rank = cfg
        return run_tf_a1((f'A1-lr{lr:.0e}-r{rank}', lr, rank, seed))
    lr, rank = cfg
    return run_tf_a3((f'A3-lr{lr:.0e}-r{rank}', rank, seed))


def build_args(n_seeds):
    args = [(('A1', cfg), s) for cfg in A1_GRID for s in range(n_seeds)]
    args += [(('A3', (A3_LR, A3_RANK)), s) for s in range(n_seeds)]
    return args


def print_table(results):
    print("\n" + "=" * 104)
    print("S26 RESULTS (Paper D P4 fair-TF follow-up): 4-domain "
          "irregular-switch, 10 seeds")
    print("=" * 104)
    arms = sorted({r['arm'] for r in results})
    print(f" {'arm':>18} | {'stream':>9} | {'forget':>9} | {'T_adapt':>8}")
    means = {}
    for arm in arms:
        rs = [r for r in results if r['arm'] == arm]
        means[arm] = (float(np.mean([r['stream_ppl'] for r in rs])),
                      float(np.mean([r['forgetting_ppl'] for r in rs])))
        print(f" {arm:>18} | {means[arm][0]:>9.3f} | {means[arm][1]:>9.3f} "
              f"| {np.nanmean([r['t_adapt_mean'] for r in rs]):>8.0f}")
    print("-" * 104)
    print("(paired comparisons vs the committed S22 SSM rows are computed "
          "in the analysis script; A1-lr1e-3-r16 must reproduce S22 TF-A1 "
          "22.46 stream / 19.01 forgetting)")
    print(f"\nS22 committed reference: TF-A1 stream 22.46, forgetting "
          f"19.01; SSM-bare 15.43/13.41; SSM-REDEM 13.18/8.93")


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S26 fair-TF "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    all_args = build_args(n_seeds)
    n_runs = len(all_args)
    print(f"total runs: {n_runs} ({len(A1_GRID)} A1 configs + 1 A3 config "
          f"x{n_seeds} seeds)")

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

    print_table(results)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'lr', 'rank', 'seed', 'stream_ppl', 'neg_frac',
                  't_adapt_mean', 'forgetting_ppl', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    params = {
        'task': {'desc': '4 biased-bigram domains (shifts 1/3/7/11, disjoint '
                         'bias sets of 4), cycling schedule, IRREGULAR switch '
                         'intervals Uniform(2500,3500), 8 segments, ~24k '
                         'tokens; s16 protocol generalized - VERBATIM from '
                         'S22 so seeds pair with the committed S22 rows',
                 'seed_rules': 'model init torch.manual_seed(seed*'
                               f'{SEED_SCALE}+{SEED_OFF}) and task '
                               'gen_multi_drift_stream(seed) verbatim from '
                               'S22'},
        'arms': {'A1_grid': [f'lr={lr:.0e}, rank={rank}'
                             for (lr, rank) in A1_GRID],
                 'A3': f'4-adapter routing (s18 A3 generalized), lr={A3_LR}, '
                       f'rank={A3_RANK}, tau_m={A3_TAU_M}, margin='
                       f'{GATE_MARGIN}'},
        'metadata_m3': {'tau_m': TAU_M,
                        'feature': 'slow EMA of frozen-base hidden means '
                                   '(s18 semantics), 4 fixed per-domain '
                                   'references'},
        'cross_check': 'A1-lr1e-3-r16 must reproduce S22 TF-A1 '
                       '(stream 22.46, forgetting 19.01)',
        'honest_scope': 'single tau_m (500) for A3; A3 rank not swept; '
                        'no scaling claims; pairing with S22 SSM rows '
                        'relies on identical seed rules',
        'discipline': '10 seeds, paired sign consistency',
        'env': {'torch': torch.__version__, 'numpy': np.__version__,
                'cpu': 'torch CPU only, no mamba-ssm dependency'},
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'rows': results}, f, indent=2)

    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
