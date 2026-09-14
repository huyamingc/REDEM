#!/usr/bin/env python3
"""
S40: Paper D — classical ESN + online RLS baseline on the P4 protocol.
=============================================================================
Type:           ML (torch/numpy CPU)
Experiment:     S40: fair classical reservoir (ESN) + online RLS readout on
                the SAME four-domain irregular-switch stream, seeds, and
                metrics as s22 SSM-bare / SSM-REDEM. Closes the Limitations
                gap "no separately wired random reservoir (ESN) baseline".

Host (classical ESN, Jaeger-style):
  r_t = (1 - a) r_{t-1} + a * tanh( W_in e_{t-1} + W_res r_{t-1} + b )
  W_res: sparse random, scaled to spectral radius SR
  W_in : random dense (V -> N_res), input scaling IS
  leak : a
  Readout: online RLS on phi = [r_t; 1] (same RLS lambda/delta as SSM arms)

Arms (10 seeds each):
  ESN-lin        : one-hot of previous token + bias (no reservoir control)
  ESN-128        : classical ESN N_res=128, SR=0.9, leak=0.5, IS=1.0
                   readout on [r_t; 1] only  (separately wired reservoir)
  ESN-128-sr1.0  : classical ESN N_res=128, SR=1.0 (edge of stability)
  ESN-256        : classical ESN N_res=256, SR=0.9, readout on [r_t; 1]

Protocol/metrics: verbatim s22 (stream predict-before-update CE, held-out
forgetting on previous domain, same seed rules). RLS recurrence identical
to s21 (lambda forgetting included).

Output files:
  data/s40_esn_rls_p4_baseline_v1.csv
  data/s40_esn_rls_p4_baseline_v1.json

Usage: python s40_esn_rls_p4_baseline.py [--quick] [--sequential]
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
from multiprocessing import Pool, cpu_count

from s18_llm_drift_gate import gen_stream
from s19_ssm_rls_readout import RLS_LAMBDA, RLS_DELTA

N_DOMAINS = 4
N_SEGMENTS = 8
SEG_LO = 2500
SEG_HI = 3500
N_SEEDS = 10
HOLDOUT_LEN = 500
VOCAB = 32
SEED_SCALE = 101
SEED_OFF = 17
T_ADAPT_WINDOW = 20
STEADY_WINDOW = 400
T_ADAPT_RATIO = 1.5
SHIFTS = [1, 3, 7, 11]
BIAS_SETS = [list(range(0, 4)), list(range(8, 12)),
             list(range(16, 20)), list(range(24, 28))]

# ESN grid: (name, N_res, spectral_radius, leak, input_scale, use_skip)
# Classical ESN readout is on reservoir state only (Jaeger). A skip arm was
# prototyped and dropped: concatenating raw reservoir noise next to the
# one-hot path destabilises the shared online RLS (ppl ≫ one-hot control),
# so it would not be a fair classical-ESN statement.
ARMS = [
    ('ESN-lin', 0, 0.0, 0.0, 0.0, False),
    ('ESN-128', 128, 0.9, 0.5, 1.0, False),
    ('ESN-128-sr1.0', 128, 1.0, 0.5, 1.0, False),
    ('ESN-256', 256, 0.9, 0.5, 1.0, False),
]
ARM_NAMES = [a[0] for a in ARMS]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's40_esn_rls_p4_baseline_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's40_esn_rls_p4_baseline_v1.json')


def gen_multi_drift_stream(seed):
    rng = np.random.RandomState(seed * 7 + 11)
    parts, domains, seg_lens = [], [], []
    for s in range(N_SEGMENTS):
        dom = s % N_DOMAINS
        seg_len = int(rng.uniform(SEG_LO, SEG_HI))
        dom_seed = seed * 31 + s * 131 + 7
        parts.append(gen_stream(dom_seed, SHIFTS[dom], seg_len,
                                BIAS_SETS[dom]))
        domains.append(dom)
        seg_lens.append(seg_len)
    stream = np.concatenate(parts)
    switch_times = np.cumsum(seg_lens)[:-1].tolist()
    return stream, np.repeat(np.array(domains), seg_lens), switch_times


class ESN:
    """Classic echo-state network (fixed random reservoir)."""

    def __init__(self, n_res, spectral_radius, leak, input_scale, seed):
        self.n_res = n_res
        self.leak = leak
        rng = np.random.RandomState(seed * SEED_SCALE + SEED_OFF + 7)
        # sparse W_res: ~10% connectivity, denser than classic 1% for small N
        mask = (rng.rand(n_res, n_res) < 0.1).astype(np.float64)
        W = rng.randn(n_res, n_res) * mask
        # spectral radius scaling
        rho = float(np.max(np.abs(np.linalg.eigvals(W)))) if n_res <= 256 else 1.0
        if rho < 1e-12:
            # fallback: power iteration on W @ W.T (cheaper for larger N)
            x = rng.randn(n_res)
            for _ in range(50):
                x = W @ x
                nrm = np.linalg.norm(x)
                if nrm < 1e-15:
                    break
                x = x / nrm
            rho = float(np.linalg.norm(W @ x))
        self.W_res = W * (spectral_radius / max(rho, 1e-12))
        self.W_in = rng.uniform(-input_scale, input_scale, (n_res, VOCAB))
        self.bias = rng.uniform(-0.1, 0.1, n_res)
        self.state = np.zeros(n_res, dtype=np.float64)

    def step(self, token):
        u = np.zeros(self.n_res, dtype=np.float64)
        u += self.W_in[:, token]
        u += self.W_res @ self.state
        u += self.bias
        self.state = (1.0 - self.leak) * self.state + self.leak * np.tanh(u)
        return self.state


def rls_step(W, P, phi, target, lam=RLS_LAMBDA, delta=None):
    """Standard RLS (same recurrence as s21_rls_update); W: (V, F), P: (F, F)."""
    g = P @ phi
    denom = lam + float(phi @ g)
    if not np.isfinite(denom) or denom <= 0:
        return W, P
    k = g / denom
    e = target - (W @ phi)
    W = W + np.outer(e, k)
    P = (P - np.outer(k, phi) @ P) / lam
    return W, P


def make_linear_readout(F, seed):
    rng = np.random.RandomState(seed * SEED_SCALE + SEED_OFF + 3)
    W = np.zeros((VOCAB, F), dtype=np.float64)
    # bias column predicts uniform prior
    W[:, -1] = 1.0 / VOCAB
    P = np.eye(F, dtype=np.float64) * float(RLS_DELTA)
    return W, P


def run_esn(args):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    arm, seed = args
    t0 = time.time()
    arm_cfg = {a[0]: a for a in ARMS}[arm]
    _, n_res, sr, leak, iscale, use_skip = arm_cfg

    stream, domains, switch_times = gen_multi_drift_stream(seed)
    T = stream.shape[0]

    def onehot(tok):
        v = np.zeros(VOCAB, dtype=np.float64)
        v[int(tok)] = 1.0
        return v

    if arm == 'ESN-lin':
        F = VOCAB + 1
        W, P = make_linear_readout(F, seed)
        esn = None
    else:
        esn = ESN(n_res, sr, leak, iscale, seed)
        F = n_res + (VOCAB if use_skip else 0) + 1
        W, P = make_linear_readout(F, seed)

    ce = np.full(T, np.nan, dtype=np.float64)
    ce_unc = np.full(T, np.nan, dtype=np.float64)
    nneg = nclip = n_unc_undef = 0

    for t in range(1, T):
        e_prev = onehot(stream[t - 1])
        if esn is None:
            phi = np.concatenate([e_prev, [1.0]])
        else:
            r = esn.step(int(stream[t - 1]))
            r = r / max(float(np.linalg.norm(r)), 1e-12) * np.sqrt(n_res)
            if use_skip:
                phi = np.concatenate([r, e_prev, [1.0]])
            else:
                phi = np.concatenate([r, [1.0]])

        y_hat = W @ phi
        y_tok = y_hat[stream[t]]
        p_t = float(min(max(y_tok, 1e-12), 1.0))
        yt = float(y_tok)
        if yt <= 0.0:
            nneg += 1
        if yt <= 1e-12:
            nclip += 1
        if yt > 0.0:
            ce_unc[t] = -np.log(yt)
        else:
            n_unc_undef += 1
        ce[t] = -np.log(p_t)

        target = np.zeros(VOCAB, dtype=np.float64)
        target[stream[t]] = 1.0
        W, P = rls_step(W, P, phi, target)

    stream_ppl = float(np.exp(np.nanmean(ce[1:])))
    stream_ppl_unclipped = (float(np.exp(np.nanmean(ce_unc[1:])))
                            if n_unc_undef < (T - 1) else float('nan'))

    t_adapts = []
    for si, t_s in enumerate(switch_times):
        seg_end = (switch_times[si + 1] if si + 1 < len(switch_times) else T)
        seg_ce = ce[t_s:seg_end]
        if seg_ce.size == 0:
            continue
        steady = float(np.exp(np.mean(seg_ce[-STEADY_WINDOW:])))
        thr = steady * T_ADAPT_RATIO
        cum = np.concatenate([[0.0], seg_ce])
        found = None
        for j in range(T_ADAPT_WINDOW, seg_ce.shape[0] + 1):
            wppl = np.exp((cum[j] - cum[j - T_ADAPT_WINDOW]) / T_ADAPT_WINDOW)
            if wppl <= thr:
                found = j
                break
        t_adapts.append(float(found) if found is not None else float('nan'))

    # Forgetting: evaluate frozen readout on previous domain (no update)
    forgets = []
    forgets_unc = []
    fneg = fclip = n_f_undef = 0
    for si, t_s in enumerate(switch_times):
        prev_dom = int(domains[t_s - 1])
        hold = gen_stream(seed * 41 + si * 211 + 3, SHIFTS[prev_dom],
                          HOLDOUT_LEN, BIAS_SETS[prev_dom])
        # fresh reservoir state for holdout (cold start)
        if esn is not None:
            esn_h = ESN(n_res, sr, leak, iscale, seed)
        ces, ces_unc = [], []
        for t in range(1, HOLDOUT_LEN):
            e_prev = onehot(hold[t - 1])
            if esn is None:
                phi = np.concatenate([e_prev, [1.0]])
            else:
                r = esn_h.step(int(hold[t - 1]))
                r = r / max(float(np.linalg.norm(r)), 1e-12) * np.sqrt(n_res)
                if use_skip:
                    phi = np.concatenate([r, e_prev, [1.0]])
                else:
                    phi = np.concatenate([r, [1.0]])
            y_tok = W @ phi
            y_tok = y_tok[hold[t]]
            p_t = float(min(max(y_tok, 1e-12), 1.0))
            yt = float(y_tok)
            if yt <= 0.0:
                fneg += 1
            if yt <= 1e-12:
                fclip += 1
            if yt > 0.0:
                ces_unc.append(-np.log(yt))
            else:
                n_f_undef += 1
            ces.append(-np.log(p_t))
        forgets.append(float(np.exp(np.mean(ces))))
        forgets_unc.append(float(np.exp(np.mean(ces_unc)))
                           if ces_unc else float('nan'))

    n_hold = (HOLDOUT_LEN - 1) * max(1, len(forgets))
    return {
        'arm': arm, 'seed': seed,
        'n_res': int(n_res),
        'spectral_radius': float(sr),
        'leak': float(leak),
        'stream_ppl': stream_ppl,
        'neg_frac': float(nneg) / (T - 1),
        'clip_frac': float(nclip) / (T - 1),
        't_adapt_mean': float(np.nanmean(t_adapts)) if t_adapts else float('nan'),
        'forgetting_ppl': float(np.mean(forgets)),
        'stream_ppl_unclipped': stream_ppl_unclipped,
        'stream_n_unc_undefined': int(n_unc_undef),
        'forget_neg_frac': float(fneg) / n_hold if n_hold else float('nan'),
        'forget_clip_frac': float(fclip) / n_hold if n_hold else float('nan'),
        'forgetting_ppl_unclipped': float(np.nanmean(forgets_unc))
        if forgets_unc else float('nan'),
        'forget_n_unc_undefined': int(n_f_undef),
        'runtime_s': time.time() - t0,
    }


def paired(results, arm_a, arm_b, metric):
    da = {r['seed']: r[metric] for r in results if r['arm'] == arm_a}
    db = {r['seed']: r[metric] for r in results if r['arm'] == arm_b}
    seeds = sorted(set(da) & set(db))
    return np.array([da[s] - db[s] for s in seeds], dtype=np.float64)


def print_table(results):
    print('\nESN arm means:')
    for arm in ARM_NAMES:
        rows = [r for r in results if r['arm'] == arm]
        if not rows:
            continue
        sp = np.mean([r['stream_ppl'] for r in rows])
        fp = np.mean([r['forgetting_ppl'] for r in rows])
        print(f'  {arm:14s}  stream {sp:7.3f}  forget {fp:7.3f}')
    print('\npaired ESN vs linear control:')
    for arm in ARM_NAMES:
        if arm == 'ESN-lin':
            continue
        for label, metric in [('stream', 'stream_ppl'),
                              ('forget', 'forgetting_ppl')]:
            d = paired(results, arm, 'ESN-lin', metric)
            if d.size:
                print(f'  {arm} vs ESN-lin [{label}]: {d.mean():+.3f}, '
                      f'{arm} better {int(np.sum(d < 0))}/{d.size}')


def dispatch(task):
    (kind, arm), s = task
    return run_esn((arm, s))


def _nan_to_null(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_null(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main():
    quick = '--quick' in sys.argv
    sequential = '--sequential' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S40 ESN+RLS P4 baseline "
          f"(quick={quick}, sequential={sequential})")
    n_seeds = 1 if quick else N_SEEDS
    all_args = [(('ESN', arm), s) for arm in ARM_NAMES for s in range(n_seeds)]
    n_runs = len(all_args)
    print(f'total runs: {n_runs} ({ARM_NAMES} x{n_seeds})')

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

    results.sort(key=lambda r: (ARM_NAMES.index(r['arm']), r['seed']))
    print_table(results)

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['arm', 'seed', 'n_res', 'spectral_radius', 'leak',
                  'stream_ppl', 'neg_frac', 'clip_frac', 't_adapt_mean',
                  'forgetting_ppl', 'stream_ppl_unclipped',
                  'stream_n_unc_undefined', 'forget_neg_frac',
                  'forget_clip_frac', 'forgetting_ppl_unclipped',
                  'forget_n_unc_undefined', 'runtime_s']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    # load s22 anchors if present for cross-script comparison
    s22_csv = os.path.join(DATA_DIR, 's22_ssm_p4_benchmark_v1.csv')
    s22_arms = {}
    if os.path.exists(s22_csv):
        with open(s22_csv, newline='') as f:
            for row in csv.DictReader(f):
                s22_arms.setdefault(row['arm'], []).append(row)

    comp = {}
    for arm in ARM_NAMES:
        if arm == 'ESN-lin':
            continue
        d_sp = paired(results, arm, 'ESN-lin', 'stream_ppl')
        d_fp = paired(results, arm, 'ESN-lin', 'forgetting_ppl')
        comp[f'{arm}_vs_ESN-lin'] = {
            'stream_diff_mean': float(d_sp.mean()),
            'stream_improved_n': int(np.sum(d_sp < 0)),
            'forgetting_diff_mean': float(d_fp.mean()),
            'forgetting_improved_n': int(np.sum(d_fp < 0)),
        }

    # cross-script: best ESN vs SSM-bare / SSM-REDEM from s22 (seed-aligned)
    for s22_arm in ['SSM-bare', 'SSM-REDEM']:
        if s22_arm not in s22_arms:
            continue
        for esn_arm in ['ESN-128', 'ESN-256']:
            da = {int(r['seed']): r for r in results if r['arm'] == esn_arm}
            db = {int(r['seed']): r for r in s22_arms[s22_arm]}
            seeds = sorted(set(da) & set(db))
            if not seeds:
                continue
            d_sp = np.array([float(da[s]['stream_ppl']) - float(db[s]['stream_ppl'])
                             for s in seeds])
            d_fp = np.array([float(da[s]['forgetting_ppl']) -
                             float(db[s]['forgetting_ppl']) for s in seeds])
            comp[f'{esn_arm}_vs_{s22_arm}'] = {
                'stream_diff_mean': float(d_sp.mean()),
                'stream_improved_n': int(np.sum(d_sp < 0)),
                'forgetting_diff_mean': float(d_fp.mean()),
                'forgetting_improved_n': int(np.sum(d_fp < 0)),
                'note': 'cross-script seed-aligned vs data/s22_ssm_p4_benchmark_v1.csv',
            }

    params = {
        'host': 'classic ESN: r=(1-a)r + a*tanh(W_in e + W_res r + b); '
                'W_res sparse 10% edges, spectral-radius scaled; online RLS on [r;1]',
        'arms': {a[0]: {'n_res': a[1], 'sr': a[2], 'leak': a[3], 'is': a[4]}
                 for a in ARMS},
        'protocol': 'identical to s22 (4 domains, irregular 2500-3500, 8 segs, '
                    'same seed rules and CE metrics)',
        'comparisons': comp,
        'discipline': '10 seeds, paired sign consistency',
        'n_seeds': n_seeds, 'quick': bool(quick),
        'env': {'numpy': np.__version__},
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump(_nan_to_null({'params': params, 'rows': results}), f,
                  indent=2, allow_nan=False)
    print(f'\nCSV : {out_csv}\nJSON: {out_json}')
    print(f'[{time.strftime("%H:%M:%S")}] DONE, total {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
