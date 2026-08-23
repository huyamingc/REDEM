#!/usr/bin/env python3
"""
S23: REDEM-SSM real-text benchmark - two public-domain sources.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S23 (Paper D P4 extension, real-text corpus): does the full
                REDEM-SSM stack transfer to REAL-TEXT domains (two
                public-domain books) under the s18 switching protocol?

Corpus (data/corpora/README.md): Alice's Adventures in Wonderland
(alice.txt) vs A Tale of Two Cities (dickens.txt), Project Gutenberg,
public domain, bodies stripped. Character-level, case-folded, 32-symbol
vocabulary (31 most frequent chars + UNK).

Task: 6 alternating segments of 3000 chars each from the two sources
(per-seed random window positions, KNOWN switch instants), 18k chars.
The input-path RLS readout is a first-order bigram model; the honest
ceiling on this task is the char-bigram conditional entropy, and
higher-order structure (exploitable by the transformer within its
context) is out of the readout's reach - reported, not hidden.

Arms (10 seeds, paired):
  SSM-bare : single RLS readout [B e_{t-1}; 1] (M1 only)
  SSM-REDEM: M1 + M3 (fast-channel EMA, 2 per-source references) + M4
             soft routing (2 specialists, dormant-P refresh)
  TF-A1    : s18 TinyCharLM + LoRA, bare online training

Metrics (s18 protocol): stream ppl, forgetting ppl (held-out 500-char
window from the previous source, domain-matched specialist for REDEM),
T_adapt, neg_frac (SSM arms).

Hypotheses (falsifiable, 10-seed sign consistency):
  H1: SSM-REDEM beats SSM-bare on stream and forgetting.
  H2: SSM-REDEM matches or beats TF-A1 on stream.
Falsified iff the paired diff reverses on 0/10 seeds (reported honestly).

Output files:
  data/s23_ssm_p4_realtext_v1.csv
  data/s23_ssm_p4_realtext_v1.json

Usage: python s23_ssm_p4_realtext.py [--quick] [--sequential]
"""
import os
import sys
import time
import csv
import json
from collections import Counter

os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from multiprocessing import Pool, cpu_count

from s18_llm_drift_gate import (TinyCharLM, lora_params, CTX, VOCAB,
                                LR_HEAD, LR_LORA, HOLDOUT_LEN,
                                T_ADAPT_WINDOW, STEADY_WINDOW,
                                T_ADAPT_RATIO)
from s19_ssm_rls_readout import (sample_substrate, whiten_scale,
                                 N_STATE, RLS_LAMBDA, RLS_DELTA,
                                 SEED_SCALE, SEED_OFF)
from s20_ssm_m3_routing import fast_mask_of, REF_LEN
from s21_ssm_m4_m5 import make_readout, rls_update

torch.set_num_threads(1)   # per-worker; Pool gives process-level parallelism

# ========================== Fixed parameters ==========================
N_SEEDS = 10
SEG_LEN = 3000
N_SEGMENTS = 6
TAU_M = 500.0
V = 32                       # 31 frequent chars + UNK (matches VOCAB=32)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CORPUS_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'corpora')
CSV_PATH = os.path.join(DATA_DIR, 's23_ssm_p4_realtext_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's23_ssm_p4_realtext_v1.json')

# ========================== Corpus ==========================

def load_corpus():
    """Returns (alice_encoded, dickens_encoded, mapping, unk_idx)."""
    with open(os.path.join(CORPUS_DIR, 'alice.txt'), encoding='utf-8') as f:
        alice = f.read()
    with open(os.path.join(CORPUS_DIR, 'dickens.txt'), encoding='utf-8') as f:
        dickens = f.read()
    counts = Counter((alice + dickens).lower())
    top = [ch for ch, _ in counts.most_common(V - 1)]
    mapping = {ch: i for i, ch in enumerate(top)}
    unk = V - 1

    def enc(text):
        return np.array([mapping.get(ch, unk) for ch in text.lower()],
                        dtype=np.int64)

    return enc(alice), enc(dickens), mapping, unk


def gen_real_stream(seed, sources, seg_len=SEG_LEN, n_segments=N_SEGMENTS):
    """Alternating two-source stream with KNOWN switch instants; per-seed
    random window positions."""
    rng = np.random.RandomState(seed * 7 + 11)
    parts, domains, starts = [], [], []
    for s in range(n_segments):
        dom = s % 2
        src = sources[dom]
        start = int(rng.randint(0, len(src) - seg_len))
        parts.append(src[start:start + seg_len])
        domains.append(dom)
        starts.append(start)
    stream = np.concatenate(parts)
    return stream, np.repeat(np.array(domains), seg_len), starts


def ref_windows(seed, sources, n=REF_LEN):
    """Per-source reference windows (per-seed positions)."""
    rng = np.random.RandomState(seed * 31 + 7)
    out = []
    for d in range(2):
        start = int(rng.randint(0, len(sources[d]) - n))
        out.append(sources[d][start:start + n])
    return out


# ========================== SSM arms ==========================

def refs_fast_real(seed, A, B, scale, windows):
    fm = fast_mask_of(A)
    refs = []
    for w in windows:
        h = torch.zeros(N_STATE, dtype=torch.float64)
        acc = torch.zeros(N_STATE, dtype=torch.float64)
        for t in range(len(w)):
            h = A * h + B[:, w[t]]
            acc = acc + h * scale
        refs.append((acc / len(w))[fm])
    return refs


def soft_weights(distances, kappa):
    z = -torch.stack(distances) / kappa
    z = z - z.max()
    w = torch.exp(z)
    return w / w.sum()


def run_ssm(args):
    """(arm, seed) -> SSM metrics on the real-text task."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, 'CV')
    scale = whiten_scale(A)
    fm = fast_mask_of(A)
    sources = load_corpus()[:2]
    stream, domains, starts = gen_real_stream(seed, sources)
    T = stream.shape[0]
    F = N_STATE + 1
    windows = ref_windows(seed, sources)
    refs = refs_fast_real(seed, A, B, scale, windows)

    if arm == 'SSM-REDEM':
        sep = float(torch.norm(refs[0] - refs[1]))
        kappa = 0.5 * max(sep, 1e-6)
        Ws = [make_readout(F) for _ in range(2)]
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
            y_hat = sum(w[i] * (Ws[i][0] @ phi) for i in range(2))
        else:
            y_hat = W @ phi
        p_t = float(y_hat[stream[t]].clamp(min=1e-12, max=1.0))
        if float(y_hat[stream[t]]) <= 0.0:
            nneg += 1
        ce[t] = -np.log(p_t)
        target = torch.zeros(VOCAB, dtype=torch.float64)
        target[stream[t]] = 1.0
        if arm == 'SSM-REDEM':
            for i in range(2):
                Wi, Pi = Ws[i]
                Ws[i] = rls_update(Wi, Pi, phi, target, float(w[i]))
        else:
            W, P = rls_update(W, P, phi, target, 1.0)

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    neg_frac = float(nneg) / (T - 1)

    # T_adapt (known switches)
    t_adapts = []
    for si in range(1, N_SEGMENTS):
        t_s = si * SEG_LEN
        seg_ce = ce[t_s:t_s + SEG_LEN]
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

    # Forgetting: held-out 500-char window from the previous source
    rng = np.random.RandomState(seed * 41 + 3)
    forgets = []
    for si in range(1, N_SEGMENTS):
        prev_dom = int(domains[si * SEG_LEN - 1])
        src = sources[prev_dom]
        start = int(rng.randint(0, len(src) - HOLDOUT_LEN))
        hold = src[start:start + HOLDOUT_LEN]
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


# ========================== TF-A1 arm ==========================

def run_tf(args):
    """(arm, seed) -> TF metrics on the real-text task."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    model = TinyCharLM()
    sources = load_corpus()[:2]
    stream, domains, starts = gen_real_stream(seed, sources)
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

    model.eval()
    rng = np.random.RandomState(seed * 41 + 3)
    forgets = []
    for si in range(1, N_SEGMENTS):
        prev_dom = int(domains[si * SEG_LEN - 1])
        src = sources[prev_dom]
        start = int(rng.randint(0, len(src) - HOLDOUT_LEN))
        hold = torch.from_numpy(src[start:start + HOLDOUT_LEN].reshape(1, -1))
        with torch.no_grad():
            ces = []
            for lo2 in range(0, HOLDOUT_LEN - 1, CTX):
                chunk = hold[:, lo2:lo2 + CTX]
                lg = model(chunk, active=0, use_lora=True)
                ces.append(nn.functional.cross_entropy(
                    lg[:, :-1, :].reshape(-1, VOCAB),
                    chunk[:, 1:].reshape(-1)).item())
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))

    return {'arm': arm, 'seed': seed, 'stream_ppl': stream_ppl,
            'neg_frac': float('nan'),
            't_adapt_mean': float('nan'),
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
    print("S23 RESULTS (Paper D real-text benchmark): Alice vs Dickens, "
          "10 seeds")
    print("=" * 104)
    arms = ['SSM-bare', 'SSM-REDEM', 'TF-A1']
    print(f" {'arm':>10} | {'stream':>9} | {'forget':>9} | {'neg%':>6}")
    for arm in arms:
        rs = [r for r in results if r['arm'] == arm]
        if not rs:
            continue
        negs = [r['neg_frac'] for r in rs]
        neg = (np.nanmean(negs) if any(not np.isnan(x) for x in negs)
               else float('nan'))
        neg_s = f"{neg * 100:>6.1f}" if not np.isnan(neg) else f"{'-':>6}"
        print(f" {arm:>10} | "
              f"{np.mean([r['stream_ppl'] for r in rs]):>9.3f} | "
              f"{np.mean([r['forgetting_ppl'] for r in rs]):>9.3f} | "
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
    (kind, arm), s = task
    return run_ssm((arm, s)) if kind == 'SSM' else run_tf((arm, s))


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S23 real-text benchmark "
          f"(quick={quick}, sequential={sequential})")

    alice, dickens, mapping, unk = load_corpus()
    print(f"corpus: alice {len(alice)} chars, dickens {len(dickens)} chars, "
          f"vocab {len(mapping)}+UNK")
    print(f"top-5 chars: {list(mapping)[:5]}")

    n_seeds = 1 if quick else N_SEEDS
    all_args = ([(('SSM', arm), s) for arm in ['SSM-bare', 'SSM-REDEM']
                 for s in range(n_seeds)]
                + [(('TF', 'TF-A1'), s) for s in range(n_seeds)])
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (SSM-bare, SSM-REDEM, TF-A1 x{n_seeds})")

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
        'task': {'desc': 'two public-domain books (Alice vs Dickens), '
                         'char-level, case-folded, 32-symbol vocab (31 + '
                         'UNK), 6 x 3000 alternating segments, known '
                         'switch instants, per-seed window positions',
                 'corpus': 'data/corpora/{alice,dickens}.txt (Gutenberg, '
                           'public domain)'},
        'arms': {'SSM-bare': 'single RLS readout [B e; 1] (M1 only)',
                 'SSM-REDEM': 'M1 + M3 (fast-channel EMA, 2 refs) + M4 soft '
                              'routing (2 specialists, dormant-P refresh)',
                 'TF-A1': 's18 TinyCharLM + LoRA, bare online training'},
        'metadata_m3': {'tau_m': TAU_M,
                        'feature': 'fast-channel (tau<=8) whitened state'},
        'hypotheses': {'H1': 'SSM-REDEM beats SSM-bare on stream/forgetting',
                       'H2': 'SSM-REDEM matches or beats TF-A1 on stream'},
        'comparisons': comp,
        'honest_scope': 'input-path readout is first-order (char-bigram '
                        'ceiling); higher-order text structure out of '
                        'reach; no scaling claims',
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
