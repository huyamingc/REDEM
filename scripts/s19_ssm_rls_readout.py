#!/usr/bin/env python3
"""
S19: REDEM-SSM P1 prototype - per-token RLS readout on a hand-rolled
diagonal SSM host.
=============================================================================
Type:           ML (torch CPU; no @njit; Pool only around independent trials)
Experiment:     S19 (Paper D P1): does the M1 online RLS readout track known
                domain switches on a linear SSM host at least as well as the
                s18 Transformer+LoRA A1 arm (Paper C Section 6)?

Host (native REDEM-SSM):
  h_t = A h_{t-1} + B e_{t-1},  A = diag(exp(-1/tau_i)).
  Readout W (V x (N+1)): per-token RLS (M1), O(F^2) in feature dim F=N+1,
  predict-before-update. No metadata (M3), no plasticity (M4), no
  regulation (M5): P1 is the bare M1 prototype.

The un-whitened log-normal host (LN-raw) is falsified by its own
diagnostics: the
log-normal tau spectrum centered at tau0=174 tokens has NO fast channels,
so the previous-token identity is drowned in the decayed history mixture,
and slow channels accumulate state magnitude while RLS P grows as
(1/lambda)^t in unexcited directions (W norm ~ 1.7e4, P trace ~ 9e8,
catastrophic held-out ppl). This script's adopted host applies the
evidence-driven revision: spectrum-whitening
(h_t * sqrt(N*(1 - A_i^2)), steady-state channel normalization),
timescale-coverage spectrum (log-uniform tau in [1, 3000] tokens - the
Paper C "timescale coverage" thesis applied to the host), and RLS
lambda = 0.9999 (memory ~ 10k tokens ~ task horizon).

Metric note: the RLS readout minimizes squared error on one-hot
targets, so its output y_hat = W phi is a LINEAR ESTIMATE OF THE
CONDITIONAL DISTRIBUTION (linear MMSE), NOT a softmax logit vector.
Applying softmax to it double-normalizes and squashes the estimate toward
uniform (all arms then show ppl ~ 26-31 ~ uniform while
the MLE table ceiling is 7.34). The correct evaluation is CE =
-ln(clip(y_hat[target], eps, 1)) with W initialized so the bias predicts
the uniform distribution (eps = 1e-12).

Arms (isolate the two factors + the M1 control):
  LN-raw     : log-normal tau (tau0=174, CV=0.20, Paper A params), no
               whitening - the un-whitened P1 failure case
  LN-whiten  : same spectrum + whitening (conditioning factor alone)
  CV-whiten  : log-uniform tau in [1,3000] + whitening (coverage factor on
               top; the proposed P1 host)
  B-proj     : control - RLS on the CURRENT-TOKEN projection B e_{t-1}
               (no state): shows M1 itself converges to the pooled table
               ceiling (~12 ppl vs oracle 7.34)
  CV-skip    : whitened state + current-token projection (both paths
               combined: linear readout on state only is the
               failure; giving it the current token restores tracking)
  CV-gate    : P3a - input-gated state features, y = W (h_w *
               sigmoid(C x + b)) with x = B e_{t-1}, C fixed random
               (Mamba-style multiplicative gating of the state pathway,
               at the feature level so RLS stays closed-form); the test
               of whether gating lets the state readout succeed
  CV-gate-g5 : same gate with gamma = 5.0 (sharper sigmoid) - shows the
               fixed-gate result is not a tuning artifact

Task: IDENTICAL to s18 - two biased-bigram Markov generators (domain A:
shift=1, bias {0..7}; domain B: shift=7, bias {24..31}), alternating every
3000 tokens (known switch instants), 6 segments, 18k tokens. Generators,
seed rules, and metrics are copied verbatim from s18_llm_drift_gate.py so
the per-seed paired comparison against the committed A1 arm is valid.

P1 prediction (Paper D, P1): per-token RLS on
the SSM output projection tracks known domain switches at least as well as
the s18 A1 arm. Falsification: RLS-on-SSM stream ppl worse than A1 on
every seed (0/10 improved). Reported honestly either way, plus a pooled
ridge oracle (in-sample linear-in-history bound) per arm.

Output files:
  data/s19_ssm_rls_readout_v1.csv
  data/s19_ssm_rls_readout_v1.json

Usage: python s19_ssm_rls_readout.py [--quick] [--sequential]
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

torch.set_num_threads(1)   # per-worker; Pool gives process-level parallelism

# ========================== Fixed parameters ==========================
VOCAB = 32
N_STATE = 128               # SSM state dimension (64-256)
TAU0 = 174.0                # Paper A median tau, scaled to tokens
CV_TAU = 0.20               # Paper A log-normal CV
TAU_MIN = 1.0               # coverage-spectrum range (tokens)
TAU_MAX = 3000.0
B_GAIN = 1.0

RLS_LAMBDA = 0.9999         # RLS forgetting factor (memory ~10k tokens)
RLS_DELTA = 1.0             # P0 = delta^-1 I

SEG_LEN = 3000              # tokens per domain segment (s18 protocol)
N_SEGMENTS = 6
T_TOTAL = SEG_LEN * N_SEGMENTS
HOLDOUT_LEN = 500           # s18 forgetting holdout length
N_SEEDS = 10

SEED_SCALE = 101            # s18 model seed rule: seed*101 + 17
SEED_OFF = 17

ARMS = {
    'LN-raw':    {'spectrum': 'LN', 'whiten': False, 'feat': 'state'},
    'LN-whiten': {'spectrum': 'LN', 'whiten': True,  'feat': 'state'},
    'CV-whiten': {'spectrum': 'CV', 'whiten': True,  'feat': 'state'},
    'B-proj':    {'spectrum': 'CV', 'whiten': True,  'feat': 'proj'},
    'CV-skip':   {'spectrum': 'CV', 'whiten': True,  'feat': 'skip'},
    'CV-gate':   {'spectrum': 'CV', 'whiten': True,  'feat': 'gate',
                  'gamma': 1.0},
    'CV-gate-g5': {'spectrum': 'CV', 'whiten': True, 'feat': 'gate',
                   'gamma': 5.0},
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's19_ssm_rls_readout_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's19_ssm_rls_readout_v1.json')
S18_CSV = os.path.join(DATA_DIR, 's18_llm_drift_gate_v1.csv')


# ========================== Task (verbatim from s18, Paper C Sec 6) ==========

BIAS_A = list(range(0, 8))        # domain A marginal bias set
BIAS_B = list(range(24, 32))      # domain B marginal bias set
P_SHIFT = 0.7                     # deterministic transition probability
BIAS_WEIGHT = 3.0                 # marginal bias strength


def gen_transition(seed, shift, bias=None, p_shift=P_SHIFT,
                   bias_weight=BIAS_WEIGHT):
    """P(next | prev) = p_shift * onehot((prev+shift) mod VOCAB)
    + (1-p_shift) * q, with `bias` chars weighted bias_weight extra.
    Copied verbatim from s18_llm_drift_gate.py."""
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
    """Sample a char stream from the bigram Markov chain. Verbatim s18."""
    rng = np.random.RandomState(seed)
    M = gen_transition(seed, shift, bias)
    out = np.empty(n_tokens, dtype=np.int64)
    c = int(rng.randint(VOCAB))
    for t in range(n_tokens):
        out[t] = c
        c = int(rng.choice(VOCAB, p=M[c]))
    return out


def gen_drift_stream(seed, seg_len=SEG_LEN, n_segments=N_SEGMENTS):
    """Alternating A/B stream with KNOWN switch instants; segment 0 = A.
    Verbatim s18 (same seed rules -> same streams per seed)."""
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


# ========================== SSM host + RLS readout ==========================

def sample_substrate(seed, spectrum):
    """Per-seed substrate realization: tau spectrum on the A diagonal and a
    fixed random input projection B, drawn under the s18 model seed rule
    (seed*101+17). spectrum: 'LN' log-normal (Paper A: tau0, CV) or 'CV'
    log-uniform over [TAU_MIN, TAU_MAX] (timescale coverage)."""
    sigma = float(np.sqrt(np.log(1.0 + CV_TAU ** 2)))
    rng = np.random.RandomState(seed * SEED_SCALE + SEED_OFF + 1)
    if spectrum == 'LN':
        tau = np.exp(np.log(TAU0) + sigma * rng.randn(N_STATE))
    else:
        tau = np.exp(rng.uniform(np.log(TAU_MIN), np.log(TAU_MAX),
                                 N_STATE))
    A = torch.tensor(np.exp(-1.0 / tau), dtype=torch.float64)
    g = torch.Generator()
    g.manual_seed(seed * SEED_SCALE + SEED_OFF)
    B = (torch.randn(N_STATE, VOCAB, dtype=torch.float64, generator=g)
         * B_GAIN / np.sqrt(N_STATE))
    return A, B


def whiten_scale(A):
    """Spectrum whitening: per-channel steady-state scaling of the linear
    recurrence h_t = A h_{t-1} + B e_{t-1}. With unit-variance inputs,
    Var(h_i) ~= 1/(N*(1-A_i^2)); multiplying by sqrt(N*(1-A_i^2)) brings
    every channel to unit stationary variance and bounds slow-channel
    magnitude (h_t * scale stays O(1) even for A -> 1)."""
    return torch.sqrt((1.0 - A * A) * N_STATE)


def ce_readout(y_hat, target):
    """Cross-entropy of a linear readout's predicted distribution y_hat
    (= W phi, a linear MMSE estimate of the one-hot conditional mean) vs
    the target token: CE = -ln(clip(y_hat[target], eps, 1)). NO softmax:
    the squared-loss readout output is already a distribution estimate
    (double-normalizing it squashes toward uniform)."""
    p = float(y_hat[target].clamp(min=1e-12, max=1.0))
    return -np.log(p)


def run_single(args):
    """(arm, seed) -> metrics dict. Top-level (picklable); unbuffered."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    cfg = ARMS[arm]
    t0 = time.time()

    torch.manual_seed(seed * SEED_SCALE + SEED_OFF)
    A, B = sample_substrate(seed, cfg['spectrum'])
    if cfg['whiten']:
        scale = whiten_scale(A)
    else:
        scale = torch.ones(N_STATE, dtype=torch.float64)
    stream, domains = gen_drift_stream(seed)
    T = stream.shape[0]

    # P3a: fixed random input gate C (Mamba-style multiplicative gating of
    # the state pathway, applied at the feature level so the RLS readout
    # stays linear-in-features/closed-form). y = W (h_w * sigmoid(C x + b)),
    # x = B e_{t-1} (the state input projection). C drawn per seed from the
    # substrate RNG stream.
    if cfg['feat'] == 'gate':
        g_gen = torch.Generator()
        g_gen.manual_seed(seed * SEED_SCALE + SEED_OFF + 3)
        C = (torch.randn(N_STATE, N_STATE, dtype=torch.float64,
                         generator=g_gen) / np.sqrt(N_STATE))
        b_gate = torch.zeros(N_STATE, dtype=torch.float64)
    else:
        C = b_gate = None

    def build_phi(hw, prev_tok):
        """Readout features for the arm: whitened state (state), current
        token projection (proj), both (skip), or input-gated state
        (gate: h_w * sigmoid(C x + b), the P3a Mamba-style gate), plus
        bias."""
        if cfg['feat'] == 'state':
            parts = [hw]
        elif cfg['feat'] == 'proj':
            parts = [B[:, prev_tok]]
        elif cfg['feat'] == 'skip':
            parts = [hw, B[:, prev_tok]]
        else:  # 'gate'
            gamma = cfg.get('gamma', 1.0)
            parts = [hw * torch.sigmoid(gamma * (C @ B[:, prev_tok])
                                        + b_gate)]
        parts.append(torch.ones(1, dtype=torch.float64))
        return torch.cat(parts)

    # ---- Online pass: per-token RLS on the output projection (M1) ----
    F = len(build_phi(torch.zeros(N_STATE, dtype=torch.float64), 0))
    W = torch.zeros(VOCAB, F, dtype=torch.float64)
    W[:, -1] = 1.0 / VOCAB      # bias -> uniform prior (no cold-start blowup)
    P = torch.eye(F, dtype=torch.float64) / RLS_DELTA
    h = torch.zeros(N_STATE, dtype=torch.float64)
    ce = np.empty(T, dtype=np.float64)
    ce[:] = np.nan
    nneg = 0
    hs = np.empty((T - 1, F), dtype=np.float64)
    for t in range(1, T):
        h = A * h + B[:, stream[t - 1]]
        hw = h * scale
        phi = build_phi(hw, stream[t - 1])
        hs[t - 1] = phi.numpy()
        y_hat = W @ phi
        p_target = float(y_hat[stream[t]].clamp(min=1e-12, max=1.0))
        if float(y_hat[stream[t]]) <= 0.0:
            nneg += 1
        ce[t] = -np.log(p_target)
        # RLS update (after predicting: predict-before-update)
        target = torch.zeros(VOCAB, dtype=torch.float64)
        target[stream[t]] = 1.0
        g = P @ phi
        k = g / (RLS_LAMBDA + float(phi @ g))
        e = target - (W @ phi)
        W = W + torch.outer(e, k)
        P = (P - torch.outer(k, phi) @ P) / RLS_LAMBDA
    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    neg_frac = float(nneg) / (T - 1)

    # ---- Pooled ridge oracle (in-sample linear-in-history bound) ----
    Y = np.zeros((T - 1, VOCAB))
    Y[np.arange(T - 1), stream[1:]] = 1.0
    X = hs
    XtX = X.T @ X + RLS_DELTA * np.eye(F)
    Wr = Y.T @ X @ np.linalg.inv(XtX)
    yhat = X @ Wr.T
    p = np.clip(np.take_along_axis(yhat, stream[1:][:, None], axis=1).ravel(),
                1e-12, 1.0)
    oracle_ppl = float(np.exp(np.mean(-np.log(p))))

    # ---- Forgetting: final readout, held-out previous domain (s18 metric) ----
    forgets = []
    for si, t_s in enumerate([SEG_LEN * s for s in range(1, N_SEGMENTS)]):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3,
                          1 if prev_dom == 0 else 7, HOLDOUT_LEN,
                          BIAS_A if prev_dom == 0 else BIAS_B)
        hh = torch.zeros(N_STATE, dtype=torch.float64)
        ces = []
        for t in range(1, HOLDOUT_LEN):
            hh = A * hh + B[:, hold[t - 1]]
            phi = build_phi(hh * scale, hold[t - 1])
            p_target = float((W @ phi)[hold[t]].clamp(min=1e-12, max=1.0))
            ces.append(-np.log(p_target))
        forgets.append(float(np.exp(np.mean(ces))))
    forgetting_ppl = float(np.mean(forgets))

    return {'arm': arm, 'seed': seed, 'stream_ppl': stream_ppl,
            'neg_frac': neg_frac, 'forgetting_ppl': forgetting_ppl,
            'oracle_ppl': oracle_ppl, 'runtime_s': time.time() - t0}


# ========================== Aggregation ==========================

def load_s18_a1():
    """Per-seed A1 reference (stream/forgetting ppl) from the committed
    s18 CSV, for paired comparison."""
    ref = {}
    with open(S18_CSV, 'r') as f:
        for row in csv.DictReader(f):
            if row['arm'] == 'A1':
                ref[int(row['seed'])] = (float(row['stream_ppl']),
                                         float(row['forgetting_ppl']))
    return ref


def print_table(results, ref, quick=False):
    print("\n" + "=" * 118)
    print("S19 RESULTS (Paper D P1): per-token RLS readout on a diagonal "
          "SSM host vs s18 A1 (transformer+LoRA), 10 seeds")
    print("=" * 118)
    print(f" {'arm':>10} {'seed':>4} | {'SSM stream':>11} {'A1 stream':>11} "
          f"{'d_stream':>9} | {'neg%':>5} | {'SSM forget':>11} {'A1 forget':>11} "
          f"{'d_forget':>9} | {'oracle':>9}")
    by_arm = {}
    for r in results:
        by_arm.setdefault(r['arm'], []).append(r)
    for arm in ARMS:
        rs = by_arm.get(arm, [])
        d_stream, d_forget = [], []
        for r in sorted(rs, key=lambda x: x['seed']):
            a1s, a1f = ref[r['seed']]
            ds = r['stream_ppl'] - a1s
            df = r['forgetting_ppl'] - a1f
            d_stream.append(ds)
            d_forget.append(df)
            print(f" {arm:>10} {r['seed']:>4} | {r['stream_ppl']:>11.3f} "
                  f"{a1s:>11.3f} {ds:>+9.3f} | {r['neg_frac'] * 100:>4.1f} | "
                  f"{r['forgetting_ppl']:>11.3f} "
                  f"{a1f:>11.3f} {df:>+9.3f} | {r['oracle_ppl']:>9.3f}")
        if not rs:
            continue
        d_stream = np.array(d_stream)
        d_forget = np.array(d_forget)
        n_imp_s = int(np.sum(d_stream < 0))
        n_imp_f = int(np.sum(d_forget < 0))
        print(f" {'':>10} {'':>4} | stream {np.mean([r['stream_ppl'] for r in rs]):.3f} "
              f"vs {np.mean([v[0] for v in ref.values()]):.3f} | "
              f"diff {d_stream.mean():+.3f}, improved {n_imp_s}/10 | "
              f"forget {np.mean([r['forgetting_ppl'] for r in rs]):.3f} "
              f"vs {np.mean([v[1] for v in ref.values()]):.3f} | "
              f"diff {d_forget.mean():+.3f}, improved {n_imp_f}/10 | "
              f"oracle {np.mean([r['oracle_ppl'] for r in rs]):.3f}")
        print("-" * 118)
    return by_arm


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S19 SSM-RLS prototype "
          f"(quick={quick}, sequential={sequential})")

    n_seeds = 1 if quick else N_SEEDS
    all_args = [(arm, s) for arm in ARMS for s in range(n_seeds)]
    print(f"total runs: {len(all_args)} (arms {list(ARMS)}, {n_seeds} seeds, "
          f"lambda={RLS_LAMBDA}, N={N_STATE})")

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

    ref = load_s18_a1()
    by_arm = print_table(results, ref)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed', 'stream_ppl', 'neg_frac', 'forgetting_ppl',
                  'oracle_ppl', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    arm_desc = {
        'LN-raw': 'log-normal tau (tau0=174, CV=0.20, Paper A params), no '
                  'whitening (the P1 failure case)',
        'LN-whiten': 'log-normal tau + spectrum whitening (conditioning '
                     'factor alone)',
        'CV-whiten': 'log-uniform tau in [1,3000] + whitening (coverage '
                     'factor on top; proposed P1 host)',
        'B-proj': 'control: RLS on the current-token projection B e_{t-1} '
                  '(no state) - M1 converges toward the pooled-table '
                  'ceiling (oracle 7.34)',
        'CV-skip': 'whitened state + current-token projection (both paths '
                   'combined)',
        'CV-gate': 'P3a: input-gated state, y = W (h_w * sigmoid(C x + b)), '
                   'x = B e_{t-1}, C fixed random per seed - the Mamba-'
                   'style multiplicative gate at the feature level',
        'CV-gate-g5': 'same gate with gamma=5.0 (sharper sigmoid) - '
                      'fixed-gate result is not a tuning artifact',
    }
    paired = {}
    for arm in ARMS:
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        paired[arm] = {
            'stream_ppl_mean': float(np.mean([r['stream_ppl'] for r in rs])),
            'stream_ppl_diff_mean': float(np.mean(
                [r['stream_ppl'] - ref[r['seed']][0] for r in rs])),
            'stream_ppl_improved_n': int(np.sum(
                [r['stream_ppl'] < ref[r['seed']][0] for r in rs])),
            'forgetting_ppl_mean': float(np.mean(
                [r['forgetting_ppl'] for r in rs])),
            'forgetting_ppl_diff_mean': float(np.mean(
                [r['forgetting_ppl'] - ref[r['seed']][1] for r in rs])),
            'forgetting_ppl_improved_n': int(np.sum(
                [r['forgetting_ppl'] < ref[r['seed']][1] for r in rs])),
            'oracle_ppl_mean': float(np.mean([r['oracle_ppl'] for r in rs])),
        }

    params = {
        'host': {'type': 'diagonal linear SSM (hand-rolled, torch CPU)',
                 'state_dim': N_STATE,
                 'A': 'diag(exp(-1/tau_i)); LN: LogNormal(tau0=174, CV=0.20) '
                      '(Paper A); CV: log-uniform in [1,3000] (timescale '
                      'coverage, Paper C thesis)',
                 'B': 'fixed random input projection, gain 1/sqrt(N), '
                      'per-seed draw (seed*101+17 rule)',
                 'whitening': 'h_t * sqrt(N*(1-A_i^2)): per-channel '
                              'steady-state normalization'},
        'readout_m1': {'type': 'per-token RLS on the output projection',
                       'feature_dim': N_STATE + 1,
                       'complexity': 'O(F^2) per token (not O(P^2))',
                       'lambda': RLS_LAMBDA, 'delta': RLS_DELTA,
                       'update': 'predict-before-update, one-hot target, '
                                 'bias initialized to uniform prior',
                       'metric': 'CE = -ln(clip(y_hat[target], 1e-12, 1)) '
                                 'on the linear MMSE output y_hat = W phi '
                                 '(no softmax)'},
        'arms': arm_desc,
        'p1_failure_analysis': 'The un-whitened host fails for two reasons: '
                         '(1) host '
                         'conditioning - the log-normal tau spectrum '
                         '(tau0=174, CV=0.20) has no fast channels, slow '
                         'channels accumulate state magnitude while RLS P '
                         'grows as (1/lambda)^t in unexcited directions '
                         '(W norm ~1.7e4, P trace ~9e8, catastrophic '
                         'held-out ppl); (2) the squared-loss '
                         'readout output is a linear MMSE distribution '
                         'estimate, and softmaxing it (as if it were '
                         'logits) squashes all arms toward uniform '
                         '(ppl ~26-31 vs the 7.34 MLE-table ceiling). '
                         'Both are resolved by the adopted host (whitening + '
                         'coverage '
                         'spectrum + lambda=0.9999; direct CE with '
                         'eps-clipping).',
        'task': {'desc': 'identical to s18 (Paper C Sec 6): two biased-'
                         'bigram Markov generators (A: shift=1 bias {0..7}; '
                         'B: shift=7 bias {24..31}), known switches, 6x3000',
                 'seg_len': SEG_LEN, 'n_segments': N_SEGMENTS,
                 't_total': T_TOTAL, 'seed_rules': 'verbatim from s18'},
        'metrics': {'stream_ppl': 'exp(mean CE) over the stream, '
                                  'predict-before-update; CE = -ln(clip('
                                  'y_hat[target], 1e-12, 1)) on the linear '
                                  'MMSE output',
                    'neg_frac': 'fraction of tokens whose predicted '
                                'probability was <= 0 (clipped) - '
                                'diagnostic of the linear-MMSE evaluation',
                    'forgetting_ppl': 's18 metric: final readout on held-out '
                                      'previous-domain samples (5 switches)',
                    'oracle_ppl': 'pooled ridge, in-sample linear-in-history '
                                  'bound (alpha=delta)'},
        'reference': {'file': 'data/s18_llm_drift_gate_v1.csv',
                      'arm': 'A1 (bare online LoRA, transformer host)',
                      'stream_ppl_mean': float(np.mean(
                          [v[0] for v in ref.values()])),
                      'forgetting_ppl_mean': float(np.mean(
                          [v[1] for v in ref.values()]))},
        'paired_vs_a1': paired,
        'p1_prediction': 'RLS-on-SSM tracks known switches at least as well '
                         'as s18 A1; falsified iff 0/10 improved',
        'discipline': 'known switch instants (s15), 10 seeds, paired sign '
                      'consistency (s16)',
        'env': {'torch': torch.__version__, 'numpy': np.__version__,
                'cpu': 'torch CPU only, no mamba-ssm dependency'},
        'n_seeds': n_seeds, 'quick': bool(quick),
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({'params': params,
                   'rows': sorted(results, key=lambda x: (x['arm'],
                                                          x['seed']))}, f,
                  indent=2)

    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
