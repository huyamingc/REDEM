#!/usr/bin/env python3
"""
S22: REDEM-SSM P4 benchmark - multi-domain, irregular-switch streaming.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S22 (Paper D P4): does the full REDEM-SSM stack beat (a)
                the bare SSM host (M1 only) and (b) the s18 Transformer
                + LoRA reference (A1: bare online LoRA), on a HARDER
                streaming protocol than s18?

Task (s16 protocol generalized): FOUR biased-bigram Markov domains
(shifts {1,3,7,11}, disjoint bias sets of 4 symbols each) in a cycling
schedule with IRREGULAR switch intervals (uniform 2500-3500 tokens), 8
segments, ~24k tokens. 1st-order structure (the honest scope of the
input-path readout); seed rules follow s18 (segment seed = seed*31 +
s*131 + 7).

Arms (10 seeds, paired analysis):
  SSM-bare : single RLS readout on [B e_{t-1}; 1] (M1 only; the P1 host)
  SSM-REDEM: full stack - M1 (RLS input-path readout) + M3 (fast-channel
             whitened-state EMA metadata, 4 per-domain references) + M4
             (SOFT routing: softmax-distance weights over 4 domain
             specialists; dormant-P refresh per P3) - M5 excluded (state
             dynamics mechanism, tested in P3; the metadata uses
             stationary fast channels)
  TF-A1    : s18 TinyCharLM + LoRA, bare online training (the committed
             s18 reference model, retrained on this task)

Metrics (s18 protocol): stream ppl (predict-before-update), forgetting
ppl (held-out previous domain, domain-matched specialist for REDEM),
T_adapt (known switches, 20-token window), neg_frac (SSM arms).

Benchmark hypotheses (falsifiable, 10-seed sign consistency):
  H1: SSM-REDEM beats SSM-bare on stream ppl and forgetting.
  H2: SSM-REDEM matches or beats TF-A1 on stream ppl.
Falsified iff the paired diff reverses on 0/10 seeds (reported honestly).

Honest scope: no WikiText-103, no scaling claims; the real-text corpus
benchmark is deferred (no external text in the repo - user decision
needed); the transformer routing reference (TF-A3) is limited to 2
adapters in the s18 model and is not run on this 4-domain task.

Output files:
  data/s22_ssm_p4_benchmark_v1.csv
  data/s22_ssm_p4_benchmark_v1.json

Usage: python s22_ssm_p4_benchmark.py [--quick] [--sequential]
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
                                CTX, VOCAB, D_MODEL, LR_HEAD, LR_LORA,
                                HOLDOUT_LEN, T_ADAPT_WINDOW, STEADY_WINDOW,
                                T_ADAPT_RATIO)
from s19_ssm_rls_readout import (sample_substrate, whiten_scale,
                                 N_STATE, RLS_LAMBDA,
                                 RLS_DELTA, SEED_SCALE, SEED_OFF)
from s20_ssm_m3_routing import fast_mask_of, REF_LEN
from s21_ssm_m4_m5 import make_readout, rls_update

torch.set_num_threads(1)   # per-worker; Pool gives process-level parallelism

# ========================== Fixed parameters ==========================
N_DOMAINS = 4
N_SEGMENTS = 8
SEG_LO = 2500
SEG_HI = 3500
N_SEEDS = 10
TAU_M = 500.0              # metadata EMA timescale (P2/P3 sweet spot)
SHIFTS = [1, 3, 7, 11]
BIAS_SETS = [list(range(0, 4)), list(range(8, 12)),
             list(range(16, 20)), list(range(24, 28))]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's22_ssm_p4_benchmark_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's22_ssm_p4_benchmark_v1.json')


# ========================== Task: 4-domain irregular switches ==========================

def gen_multi_drift_stream(seed, n_domains=N_DOMAINS,
                           n_segments=N_SEGMENTS, seg_lo=SEG_LO,
                           seg_hi=SEG_HI):
    """Cycling multi-domain stream with IRREGULAR switch intervals.
    Segment s uses domain s % n_domains; interval ~ Uniform(seg_lo,
    seg_hi). Returns (stream, domains, switch_times, seg_lens)."""
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


# ========================== SSM arms ==========================

def refs_fast_multi(seed, A, B, scale):
    """Fast-channel metadata references, one per domain."""
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
    """Softmax-distance routing weights over the specialists."""
    z = -torch.stack(distances) / kappa
    z = z - z.max()
    w = torch.exp(z)
    return w / w.sum()


def run_ssm(args):
    """(arm, seed) -> SSM metrics. Top-level; unbuffered."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    fm = fast_mask_of(A)
    stream, domains, switch_times, seg_lens = gen_multi_drift_stream(seed)
    T = stream.shape[0]
    F = N_STATE + 1

    refs = refs_fast_multi(seed, A, B, scale)
    if arm == 'SSM-REDEM':
        pair_d = [float(torch.norm(refs[i] - refs[j]))
                  for i in range(N_DOMAINS) for j in range(i + 1,
                                                          N_DOMAINS)]
        kappa = 0.5 * float(np.median(pair_d))
        Ws = [make_readout(F) for _ in range(N_DOMAINS)]
        slow = refs[0].clone()
    else:
        Ws = [make_readout(F)]
        kappa = None
        slow = None
    W, P = Ws[0]

    ce = np.empty(T, dtype=np.float64)
    ce[:] = np.nan
    nneg = 0
    h = torch.zeros(N_STATE, dtype=torch.float64)

    for t in range(1, T):
        h = A * h + B[:, stream[t - 1]]
        hw = h * scale
        phi = torch.cat([B[:, stream[t - 1]],
                         torch.ones(1, dtype=torch.float64)])
        if arm == 'SSM-REDEM':
            lam = 1.0 / TAU_M
            slow = (1.0 - lam) * slow + lam * hw[fm]
            dists = [torch.norm(slow - r) for r in refs]
            w = soft_weights(dists, kappa)
            y_hat = sum(w[i] * (Ws[i][0] @ phi) for i in range(N_DOMAINS))
        else:
            y_hat = W @ phi
        p_t = float(y_hat[stream[t]].clamp(min=1e-12, max=1.0))
        if float(y_hat[stream[t]]) <= 0.0:
            nneg += 1
        ce[t] = -np.log(p_t)
        target = torch.zeros(VOCAB, dtype=torch.float64)
        target[stream[t]] = 1.0
        if arm == 'SSM-REDEM':
            for i in range(N_DOMAINS):
                Wi, Pi = Ws[i]
                Ws[i] = rls_update(Wi, Pi, phi, target, float(w[i]))
        else:
            W, P = rls_update(W, P, phi, target, 1.0)

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    neg_frac = float(nneg) / (T - 1)

    # T_adapt (known switches, s18 protocol)
    t_adapts = []
    for si, t_s in enumerate(switch_times):
        seg_end = (switch_times[si + 1] if si + 1 < len(switch_times)
                   else T)
        seg_ce = ce[t_s:seg_end]
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

    # Forgetting: held-out previous domain; domain-matched specialist
    forgets = []
    for si, t_s in enumerate(switch_times):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3, SHIFTS[prev_dom],
                          HOLDOUT_LEN, BIAS_SETS[prev_dom])
        if arm == 'SSM-REDEM':
            Wf = Ws[prev_dom][0]
        else:
            Wf = W
        hh = torch.zeros(N_STATE, dtype=torch.float64)
        ces = []
        for t in range(1, HOLDOUT_LEN):
            hh = A * hh + B[:, hold[t - 1]]
            phi = torch.cat([B[:, hold[t - 1]],
                             torch.ones(1, dtype=torch.float64)])
            p_t = float((Wf @ phi)[hold[t]].clamp(min=1e-12, max=1.0))
            ces.append(-np.log(p_t))
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))

    return {'arm': arm, 'seed': seed, 'stream_ppl': stream_ppl,
            'neg_frac': neg_frac,
            't_adapt_mean': float(np.nanmean(t_adapts))
            if t_adapts else float('nan'),
            'forgetting_ppl': forgetting_ppl,
            'runtime_s': time.time() - t0}


# ========================== TF-A1 arm (s18 model, replicated loop) ==========================

def run_tf(args):
    """(arm, seed) -> TF metrics. A1: bare online LoRA (s18 replication)."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    model = TinyCharLM()
    stream, domains, switch_times, seg_lens = gen_multi_drift_stream(seed)
    T = stream.shape[0]

    head_params = list(model.head.parameters())
    lora = list(lora_params(model, 0))
    optim = torch.optim.Adam(
        [{'params': head_params, 'lr': LR_HEAD},
         {'params': lora, 'lr': LR_LORA}])

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

    # T_adapt (known switches)
    t_adapts = []
    for si, t_s in enumerate(switch_times):
        seg_end = (switch_times[si + 1] if si + 1 < len(switch_times)
                   else T)
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

    # Forgetting: held-out previous domain (single adapter, s18 semantics)
    model.eval()
    forgets = []
    for si, t_s in enumerate(switch_times):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3, SHIFTS[prev_dom],
                          HOLDOUT_LEN, BIAS_SETS[prev_dom])
        hold_t = torch.from_numpy(hold.reshape(1, -1))
        with torch.no_grad():
            ces = []
            for lo2 in range(0, HOLDOUT_LEN - 1, CTX):
                chunk = hold_t[:, lo2:lo2 + CTX]
                lg = model(chunk, active=0, use_lora=True)
                ces.append(nn.functional.cross_entropy(
                    lg[:, :-1, :].reshape(-1, VOCAB),
                    chunk[:, 1:].reshape(-1)).item())
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))

    return {'arm': arm, 'seed': seed, 'stream_ppl': stream_ppl,
            'neg_frac': float('nan'),
            't_adapt_mean': float(np.nanmean(t_adapts))
            if t_adapts else float('nan'),
            'forgetting_ppl': forgetting_ppl,
            'runtime_s': time.time() - t0}


# ========================== Aggregation ==========================

def paired(results, arm_a, arm_b, metric):
    d = []
    for s in range(N_SEEDS):
        a = next((r[metric] for r in results
                  if r['arm'] == arm_a and r['seed'] == s), None)
        b = next((r[metric] for r in results
                  if r['arm'] == arm_b and r['seed'] == s), None)
        if a is not None and b is not None:
            d.append(float(a) - float(b))
    return np.array(d)


def print_table(results):
    print("\n" + "=" * 104)
    print("S22 RESULTS (Paper D P4): 4-domain irregular-switch benchmark, "
          "10 seeds")
    print("=" * 104)
    arms = ['SSM-bare', 'SSM-REDEM', 'TF-A1']
    print(f" {'arm':>10} | {'stream':>9} | {'forget':>9} | "
          f"{'T_adapt':>8} {'neg%':>6}")
    means = {}
    for arm in arms:
        rs = [r for r in results if r['arm'] == arm]
        if not rs:
            continue
        means[arm] = (float(np.mean([r['stream_ppl'] for r in rs])),
                      float(np.mean([r['forgetting_ppl'] for r in rs])))
        negs = [r['neg_frac'] for r in rs]
        neg = (np.nanmean(negs) if any(not np.isnan(x) for x in negs)
               else float('nan'))
        neg_s = f"{neg * 100:>6.1f}" if not np.isnan(neg) else f"{'-':>6}"
        print(f" {arm:>10} | {means[arm][0]:>9.3f} | {means[arm][1]:>9.3f} "
              f"| {np.nanmean([r['t_adapt_mean'] for r in rs]):>8.0f} "
              f"{neg_s}")
    print("-" * 104)
    for a, b in [('SSM-REDEM', 'SSM-bare'), ('SSM-REDEM', 'TF-A1'),
                 ('SSM-bare', 'TF-A1')]:
        for m, label in [('stream_ppl', 'stream'), ('forgetting_ppl',
                                                    'forget')]:
            d = paired(results, a, b, m)
            if d.size:
                print(f" {a} vs {b} [{label}]: {d.mean():+.3f}, "
                      f"{a} better {int(np.sum(d < 0))}/10")


def dispatch(task):
    """Top-level Pool worker: (kind, arm), seed -> run_ssm or run_tf."""
    (kind, arm), s = task
    return run_ssm((arm, s)) if kind == 'SSM' else run_tf((arm, s))


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S22 P4 benchmark "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    all_args = ([(('SSM', arm), s) for arm in ['SSM-bare', 'SSM-REDEM']
                 for s in range(n_seeds)]
                + [(('TF', 'TF-A1'), s) for s in range(n_seeds)])
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (SSM-bare, SSM-REDEM, TF-A1 x{n_seeds}; "
          f"4 domains, irregular switches)")

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
    fieldnames = ['arm', 'seed', 'stream_ppl', 'neg_frac', 't_adapt_mean',
                  'forgetting_ppl', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    comp = {}
    for a, b in [('SSM-REDEM', 'SSM-bare'), ('SSM-REDEM', 'TF-A1'),
                 ('SSM-bare', 'TF-A1')]:
        comp[f'{a}_vs_{b}'] = {
            'stream_diff_mean': float(paired(results, a, b,
                                             'stream_ppl').mean()),
            'stream_improved_n': int(np.sum(paired(results, a, b,
                                                   'stream_ppl') < 0)),
            'forgetting_diff_mean': float(paired(results, a, b,
                                                 'forgetting_ppl').mean()),
            'forgetting_improved_n': int(np.sum(paired(results, a, b,
                                                       'forgetting_ppl') < 0)),
        }

    params = {
        'task': {'desc': '4 biased-bigram domains (shifts 1/3/7/11, disjoint '
                         'bias sets of 4), cycling schedule, IRREGULAR '
                         'switch intervals Uniform(2500,3500), 8 segments, '
                         '~24k tokens; s16 protocol generalized',
                 'seed_rules': 's18 pattern (segment seed = seed*31+s*131+7)'},
        'arms': {'SSM-bare': 'single RLS readout [B e; 1] (M1 only)',
                 'SSM-REDEM': 'M1 + M3 (fast-channel EMA, 4 refs) + M4 soft '
                              'routing (softmax-distance weights, kappa = '
                              '0.5*median pairwise ref distance; dormant-P '
                              'refresh); M5 excluded (state dynamics, P3)',
                 'TF-A1': 's18 TinyCharLM + LoRA, bare online training'},
        'metadata_m3': {'tau_m': TAU_M,
                        'feature': 'fast-channel (tau<=8) whitened state'},
        'hypotheses': {'H1': 'SSM-REDEM beats SSM-bare on stream/forgetting',
                       'H2': 'SSM-REDEM matches or beats TF-A1 on stream'},
        'comparisons': comp,
        'honest_scope': 'no WikiText-103, no scaling claims; real-text '
                        'corpus benchmark deferred (no external text in '
                        'repo); TF-A3 4-domain routing not run (s18 model '
                        'has 2 adapters)',
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
