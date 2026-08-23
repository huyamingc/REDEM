#!/usr/bin/env python3
"""
LLM drift-gate proof of concept (Paper C S18).
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     Does the M3 slow-trace metadata transfer to a Transformer
                substrate? Tiny char-level LM with hand-rolled LoRA on a
                streaming domain-drift task (two bigram generators with
                KNOWN switch instants - the s15 lesson). Arms (the "when to
                adapt" / "which adapter" claims):

  A1 bare          : LoRA adapters update every chunk (fixed LR)
  A2 drift gate    : slow EMA of frozen-base hidden means estimates the
                     domain; adapter updates are boosted for tau_m tokens
                     after a detected switch and suppressed (10%) within
                     a domain
  A3 domain routing: two LoRA adapters; the slow-trace estimate selects
                     the active adapter; only the active one updates

The output head (readout analog) is always online-trained in every arm.
The slow trace is computed on the FROZEN base representations (the
substrate analog), with fixed per-domain references - exactly how REDEM
computes metadata on a frozen substrate.

tau_m in {200, 500, 1000, 2000}; 10 seeds. Metrics: stream perplexity,
post-switch T_adapt (known switches, running-window protocol of s15),
forgetting perplexity (held-out sample of the previous domain - the LLM
metric Paper C does not have); paired analysis vs A1 with sign
consistency (the s16 discipline).

Output files:
  data/s18_llm_drift_gate_v1.csv
  data/s18_llm_drift_gate_v1.json

Usage: python s18_llm_drift_gate.py [--quick] [--sequential]
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

torch.set_num_threads(1)   # per-worker; Pool gives process-level parallelism

# ========================== Fixed parameters ==========================
VOCAB = 32
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 2
CTX = 128
RANK = 16
N_ADAPTERS = 2              # A3 uses both; A1/A2 use adapter 0 only

LR_HEAD = 1e-3               # readout (head) learning rate (Adam)
LR_LORA = 1e-3               # base LoRA learning rate (Adam)
GATE_LOW_FRAC = 0.10         # in-domain LR multiplier for A2

SEG_LEN = 3000              # tokens per domain segment (known switch instants)
N_SEGMENTS = 6              # total segments; segment 0 is always domain A
T_TOTAL = SEG_LEN * N_SEGMENTS

TAU_M_LIST = [200.0, 500.0, 1000.0, 2000.0]
N_SEEDS = 10

T_ADAPT_WINDOW = 20
STEADY_WINDOW = 400
T_ADAPT_RATIO = 1.5         # threshold: window ppl <= steady * ratio
HOLDOUT_LEN = 500
REF_LEN = 1500              # tokens per generator for the reference features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's18_llm_drift_gate_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's18_llm_drift_gate_v1.json')

ARMS = ['A1', 'A2', 'A3']


# ========================== Task: two bigram generators ==========================

BIAS_A = list(range(0, 8))        # domain A marginal bias set
BIAS_B = list(range(24, 32))      # domain B marginal bias set
P_SHIFT = 0.7                     # deterministic transition probability
BIAS_WEIGHT = 3.0                 # marginal bias strength


def gen_transition(seed, shift, bias=None, p_shift=P_SHIFT,
                   bias_weight=BIAS_WEIGHT):
    """P(next | prev) = p_shift * onehot((prev+shift) mod VOCAB)
    + (1-p_shift) * q, where q is uniform over VOCAB with `bias` chars
    weighted bias_weight extra. The shift gives strong learnable bigram
    structure (ceiling ppl ~ e^1); the bias makes the MARGINAL differ
    between domains, so the mean hidden state (the slow-trace feature)
    separates them - mirroring the regime_switch design of Paper C."""
    q = np.ones(VOCAB)
    if bias is not None:
        q[bias] *= bias_weight
    q /= q.sum()
    M = np.zeros((VOCAB, VOCAB))
    for c in range(VOCAB):
        M[c, (c + shift) % VOCAB] += p_shift
        M[c] += (1.0 - p_shift) * q
    return M


def gen_stream(seed, shift, n_tokens, bias=None):
    """Sample a char stream from the bigram Markov chain (domain = shift,
    marginal bias = bias)."""
    rng = np.random.RandomState(seed)
    M = gen_transition(seed, shift, bias)
    out = np.empty(n_tokens, dtype=np.int64)
    c = int(rng.randint(VOCAB))
    for t in range(n_tokens):
        out[t] = c
        c = int(rng.choice(VOCAB, p=M[c]))
    return out


def gen_drift_stream(seed, seg_len=SEG_LEN, n_segments=N_SEGMENTS):
    """Alternating A/B stream with KNOWN switch instants. Segment 0 = A."""
    rng = np.random.RandomState(seed * 7 + 11)
    parts = []
    domains = []
    for s in range(n_segments):
        dom = s % 2
        dom_seed = seed * 31 + s * 131 + 7
        parts.append(gen_stream(dom_seed, 1 if dom == 0 else 7, seg_len,
                                BIAS_A if dom == 0 else BIAS_B))
        domains.append(dom)
    return np.concatenate(parts), np.repeat(np.array(domains), seg_len)


# ========================== Model: tiny char LM with hand-rolled LoRA ==========================

class LoRALinear(nn.Module):
    """Frozen base linear + trainable low-rank deltas W = B_a @ A_a.
    n_adapters=2 supports domain routing (A3); use_lora=False gives the
    frozen-base forward (the metadata channel)."""
    def __init__(self, in_f, out_f, n_adapters=N_ADAPTERS):
        super().__init__()
        self.base = nn.Linear(in_f, out_f)
        self.base.requires_grad_(False)
        self.n_adapters = n_adapters
        self.A = nn.ParameterList([
            nn.Parameter(torch.randn(RANK, in_f) * 0.02)
            for _ in range(n_adapters)])
        self.B = nn.ParameterList([
            nn.Parameter(torch.zeros(out_f, RANK))
            for _ in range(n_adapters)])

    def forward(self, x, active=0, use_lora=True):
        if not use_lora:
            return self.base(x)
        return self.base(x) + (x @ self.A[active].t()) @ self.B[active].t()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = LoRALinear(D_MODEL, 3 * D_MODEL)
        self.out = LoRALinear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff1 = LoRALinear(D_MODEL, 2 * D_MODEL)
        self.ff2 = LoRALinear(2 * D_MODEL, D_MODEL)

    def forward(self, x, active=0, use_lora=True):
        h = self.ln1(x)
        qkv = self.qkv(h, active, use_lora)
        q, k, v = qkv.chunk(3, dim=-1)
        B, T, D = q.shape
        hd = D // N_HEADS
        q = q.view(B, T, N_HEADS, hd).transpose(1, 2)
        k = k.view(B, T, N_HEADS, hd).transpose(1, 2)
        v = v.view(B, T, N_HEADS, hd).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (hd ** 0.5)
        mask = torch.triu(torch.full((T, T), float('-inf')),
                          diagonal=1).to(x.device)
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        o = attn @ v
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out(o, active, use_lora)
        r = self.ln2(x)
        x = x + self.ff2(torch.relu(self.ff1(r, active, use_lora)),
                         active, use_lora)
        return x


class TinyCharLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.embed.requires_grad_(False)
        self.pos = nn.Parameter(torch.randn(CTX, D_MODEL) * 0.02)
        self.pos.requires_grad_(False)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.ln.requires_grad_(False)
        self.head = nn.Linear(D_MODEL, VOCAB)   # trainable readout

    def forward(self, idx, active=0, use_lora=True):
        B, T = idx.shape
        x = self.embed(idx) + self.pos[:T]
        for blk in self.blocks:
            x = blk(x, active, use_lora)
        return self.head(self.ln(x))


# ========================== Per-run experiment ==========================

def lora_params(model, active_only=None):
    """Trainable LoRA params. active_only: int -> only that adapter's A/B;
    None -> all adapters (A3 routing selects by zeroing grads)."""
    for blk in model.blocks:
        for m in [blk.qkv, blk.out, blk.ff1, blk.ff2]:
            idxs = ([active_only] if active_only is not None
                    else range(m.n_adapters))
            for i in idxs:
                yield m.A[i]
                yield m.B[i]


def hidden_mean(model, idx):
    """Frozen-base forward; returns mean last hidden vector (D,) over the
    whole sequence (chunked at CTX to fit the positional table)."""
    with torch.no_grad():
        model.eval()
        means = []
        for lo in range(0, idx.shape[1], CTX):
            chunk = idx[:, lo:lo + CTX]
            B, T = chunk.shape
            x = model.embed(chunk) + model.pos[:T]
            for blk in model.blocks:
                x = blk(x, 0, use_lora=False)
            h = model.ln(x)
            means.append(h.mean(dim=1))
        h = torch.stack(means).mean(dim=0)
        return h.squeeze(0).detach()   # (D,)


def gate_estimate(slow, refs, prev=0, margin=1.15):
    """Domain 0/1 by nearest reference with a hysteresis band: flip only
    when one distance is clearly (< 1/margin) smaller; otherwise keep the
    previous estimate (suppresses oscillation near the boundary)."""
    d0 = float(torch.norm(slow - refs[0]))
    d1 = float(torch.norm(slow - refs[1]))
    if d0 * margin < d1:
        return 0
    if d1 * margin < d0:
        return 1
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


def run_single(args):
    """(arm, tau_m, seed_idx) -> metrics dict."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, tau_m, seed_idx = args
    t0 = time.time()

    torch.manual_seed(seed_idx * 101 + 17)
    model = TinyCharLM()
    stream, domains = gen_drift_stream(seed_idx)
    T = stream.shape[0]

    # Fixed per-domain references on the frozen base (metadata statistics)
    refs = [hidden_mean(model, torch.from_numpy(
        gen_stream(seed_idx * 31 + d * 131 + 7, 1 if d == 0 else 7,
                   REF_LEN, BIAS_A if d == 0 else BIAS_B)
        .reshape(1, -1))) for d in range(2)]

    head_params = list(model.head.parameters())
    if arm == 'A3':
        lora = list(lora_params(model, None))
    else:
        lora = list(lora_params(model, 0))
    optim = torch.optim.Adam(
        [{'params': head_params, 'lr': LR_HEAD},
         {'params': lora, 'lr': LR_LORA}])

    ce_all = np.full(T, np.nan)
    slow = refs[0].clone()
    last_switch_t = -10 ** 9
    prev_est = 0

    lo = 0
    while lo < T - 1:
        hi = min(lo + CTX, T)
        if hi - lo < 2:
            break
        idx = torch.from_numpy(stream[lo:hi].reshape(1, -1))

        if arm in ('A2', 'A3'):
            hm = hidden_mean(model, idx)
            with torch.no_grad():
                # per-token-equivalent EMA decay over the chunk length
                lam = 1.0 - np.exp(-(hi - lo) / tau_m)
                slow = (1.0 - lam) * slow + lam * hm
            est = gate_estimate(slow, refs, prev_est)
            if est != prev_est:
                last_switch_t = hi
                prev_est = est

        # Online train step (predict-before-update on the chunk)
        model.train()
        logits = model(idx, active=(est if arm == 'A3' else 0),
                       use_lora=True)
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, VOCAB),
            idx[:, 1:].reshape(-1))
        with torch.no_grad():
            ce_all[lo + 1:hi] = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, VOCAB),
                idx[:, 1:].reshape(-1), reduction='none').numpy()

        optim.zero_grad()
        loss.backward()
        if arm == 'A2':
            scale = 1.0 if (hi - last_switch_t) < tau_m else GATE_LOW_FRAC
            optim.param_groups[1]['lr'] = LR_LORA * scale
        else:
            optim.param_groups[1]['lr'] = LR_LORA
        if arm == 'A3':
            # zero grads of the inactive adapter
            active = est
            for blk in model.blocks:
                for m in [blk.qkv, blk.out, blk.ff1, blk.ff2]:
                    for i in range(m.n_adapters):
                        if i != active:
                            if m.A[i].grad is not None:
                                m.A[i].grad.zero_()
                            if m.B[i].grad is not None:
                                m.B[i].grad.zero_()
        optim.step()
        lo += CTX - 1

    # ---- Metrics ----
    valid = ce_all[1:]
    stream_ppl = float(np.exp(np.nanmean(valid)))

    # Per-segment steady ppl and post-switch T_adapt (known switches)
    t_adapts = []
    forgets = []
    switch_times = [SEG_LEN * s for s in range(1, N_SEGMENTS)]
    for si, t_s in enumerate(switch_times):
        seg_end = SEG_LEN * (si + 2) if si + 2 <= N_SEGMENTS else T
        seg_ce = ce_all[t_s:seg_end]
        if seg_ce.size == 0:
            continue
        steady = float(np.exp(np.mean(seg_ce[-STEADY_WINDOW:])))
        thr = steady * T_ADAPT_RATIO
        # running-window ppl over the segment (window fully post-switch)
        cum = np.concatenate([[0.0], seg_ce])
        found = None
        for j in range(T_ADAPT_WINDOW, seg_ce.shape[0] + 1):
            wppl = np.exp((cum[j] - cum[j - T_ADAPT_WINDOW])
                          / T_ADAPT_WINDOW)
            if wppl <= thr:
                found = j
                break
        t_adapts.append(float(found) if found is not None else float('nan'))

        # Forgetting: held-out sample of the PREVIOUS domain, current model.
        # A3 evaluates with the adapter MATCHING the holdout's domain (each
        # adapter's retention); A1/A2 use the single adapter.
        prev_dom = int(domains[t_s - 1])
        hold = torch.from_numpy(gen_stream(
            seed_idx * 41 + si * 211 + 3, 1 if prev_dom == 0 else 7,
            HOLDOUT_LEN, BIAS_A if prev_dom == 0 else BIAS_B)
            .reshape(1, -1))
        active_forget = (prev_dom if arm == 'A3' else 0)
        fce = eval_ce(model, hold, active=active_forget, use_lora=True)
        forgets.append(float(np.exp(fce)))

    return {'arm': arm, 'tau_m': float(tau_m), 'seed': seed_idx,
            'stream_ppl': stream_ppl,
            't_adapt_mean': float(np.nanmean(t_adapts))
            if t_adapts else float('nan'),
            'forgetting_ppl': float(np.mean(forgets))
            if forgets else float('nan'),
            'runtime_s': time.time() - t0}


# ========================== Aggregation ==========================

def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault((r['arm'], r['tau_m']), []).append(r)
    agg = []
    for (arm, tm), rs in sorted(groups.items()):
        entry = {'arm': arm, 'tau_m': tm, 'n_runs': len(rs)}
        for m in ['stream_ppl', 't_adapt_mean', 'forgetting_ppl']:
            v = np.array([r[m] for r in rs], dtype=float)
            v = v[~np.isnan(v)]
            entry[m + '_mean'] = float(np.mean(v)) if v.size else float('nan')
            entry[m + '_std'] = float(np.std(v)) if v.size else float('nan')
        agg.append(entry)
    return agg


def paired_diff(results, arm, tm, metric, base='A1'):
    d = []
    for s in range(N_SEEDS):
        a = next((r[metric] for r in results
                  if r['arm'] == arm and r['tau_m'] == tm and r['seed'] == s),
                 None)
        b = next((r[metric] for r in results
                  if r['arm'] == base and r['tau_m'] == 0.0
                  and r['seed'] == s), None)
        if a is not None and b is not None:
            d.append(float(a) - float(b))
    d = np.array(d)
    return d


def print_table(agg, results):
    print("\n" + "=" * 108)
    print("S18 RESULTS (mean over seeds): LLM drift gate (tiny transformer "
          "+ LoRA)")
    print("=" * 108)
    print(f" {'arm':>3} {'tau_m':>6} | {'stream_ppl':>10} | "
          f"{'t_adapt':>8} | {'forget':>8} | "
          f"{'d_ppl':>8} {'d_tadapt':>9} {'d_forget':>9}")
    for a in sorted(agg, key=lambda x: (x['arm'], x['tau_m'])):
        if a['arm'] == 'A1':
            print(f" {a['arm']:>3} {'-':>6} | "
                  f"{a['stream_ppl_mean']:>10.3f} | "
                  f"{a['t_adapt_mean_mean']:>8.0f} | "
                  f"{a['forgetting_ppl_mean']:>8.3f} | "
                  f"{'-':>8} {'-':>9} {'-':>9}")
            continue
        dp = paired_diff(results, a['arm'], a['tau_m'], 'stream_ppl')
        dt = paired_diff(results, a['arm'], a['tau_m'], 't_adapt_mean')
        df = paired_diff(results, a['arm'], a['tau_m'], 'forgetting_ppl')
        sgn = lambda d: (f"{d.mean():+.3f} ({int(np.sum(d < 0))}/10)"
                         if d.size else '-')
        print(f" {a['arm']:>3} {a['tau_m']:>6.0f} | "
              f"{a['stream_ppl_mean']:>10.3f} | "
              f"{a['t_adapt_mean_mean']:>8.0f} | "
              f"{a['forgetting_ppl_mean']:>8.3f} | "
              f"{sgn(dp):>8} {sgn(dt):>9} {sgn(df):>9}")


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S18 LLM drift gate "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    all_args = ([('A1', 0.0, s) for s in range(n_seeds)]
                + [('A2', tm, s) for tm in TAU_M_LIST for s in range(n_seeds)]
                + [('A3', tm, s) for tm in TAU_M_LIST for s in range(n_seeds)])
    n_runs = len(all_args)
    print(f"total runs: {n_runs} (A1 x{n_seeds}, A2/A3 x{len(TAU_M_LIST)} "
          f"tau_m x{n_seeds})")

    results = []
    if sequential:
        for i, a in enumerate(all_args):
            results.append(run_single(a))
            if (i + 1) % max(1, n_runs // 10) == 0 or (i + 1) == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] progress {i + 1}/{n_runs}",
                      flush=True)
    else:
        with Pool(min(cpu_count(), max(1, n_runs))) as pool:
            done = 0
            for res in pool.imap_unordered(run_single, all_args, chunksize=1):
                results.append(res)
                done += 1
                if done % max(1, n_runs // 10) == 0 or done == n_runs:
                    print(f"[{time.strftime('%H:%M:%S')}] progress {done}/{n_runs}",
                          flush=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'tau_m', 'seed', 'stream_ppl', 't_adapt_mean',
                  'forgetting_ppl', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    agg = aggregate(results)
    params = {
        'model': f'tiny char LM: d_model={D_MODEL}, layers={N_LAYERS}, '
                 f'heads={N_HEADS}, ctx={CTX}, vocab={VOCAB}',
        'lora_rank': RANK, 'n_adapters': N_ADAPTERS,
        'lr_head': LR_HEAD, 'lr_lora': LR_LORA,
        'gate_low_frac': GATE_LOW_FRAC,
        'task': {'desc': 'two bigram Markov generators (vowel vs consonant '
                         'biased) with known switches',
                 'seg_len': SEG_LEN,
                 'n_segments': N_SEGMENTS, 't_total': T_TOTAL},
        'arms': {'A1': 'bare online LoRA (update every chunk)',
                 'A2': 'drift gate: slow EMA of frozen-base hidden means, '
                       'boost for tau_m after detected switch, 10% within',
                 'A3': 'domain routing: two adapters, slow-trace selects '
                       'active, only active updates'},
        'tau_m_list': TAU_M_LIST,
        'metrics': {'stream_ppl': 'exp(mean CE) over the stream',
                    't_adapt_mean': 'post-switch tokens to window ppl <= '
                                    'steady*1.5 (known switches)',
                    'forgetting_ppl': 'held-out ppl of the previous domain '
                                      'after switching away'},
        'discipline': 'known switch instants (s15), tau_m sweep (s16), '
                      'paired sign consistency (s16)',
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params, 'aggregates': agg}, f, indent=2)

    print_table(agg, results)
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
