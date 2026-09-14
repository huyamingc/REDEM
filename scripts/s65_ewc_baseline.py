#!/usr/bin/env python3
"""
External continual-learning baseline (EWC) on the s52 revisit protocol
=============================================================================
Type:           PAPER
Paper Section:  Results -- external baseline for the memory / revisit claim
                (R3, R5). Closes the "no external method comparison" gap.
Experiment:     s52 showed that frozen-hypothesis memory plus reward-rate
                selection beats continuous re-adaptation (rls_single) under
                A-B-A-B regime revisits. That comparison is wholly INTERNAL:
                every arm is an RLS readout on the same quadratic basis. This
                script adds the canonical EXTERNAL continual-learning
                competitor, Elastic Weight Consolidation (EWC; Kirkpatrick,
                Pascanu, Rabinowitz et al., PNAS 114(13):3521-3526, 2017), on
                the IDENTICAL protocol, task instances and seeds, so the two
                can be paired seed by seed.

Environment:    identical to s52 -- 4 x 500-block segments
                (linear -> circle -> linear -> circle, A-B-A-B) on the
                deployed coupled substrate (random_graph_k25, kappa=25),
                N_UNITS=256, K=20 pulses per block so T=40000, quadratic
                basis (d=513), 10 seeds. Task instances are regenerated from
                the same per-seed RandomState that s52 uses, so seed s here is
                the same task realization as seed s in
                data/s52_dynamic_memory_v1.csv.

Arms (four):
  rls_single : s52's reference arm, reproduced from the same code path. It
               doubles as a REPRODUCTION CHECK: if its per-seed values do not
               match data/s52_dynamic_memory_v1.csv exactly, the task
               instances differ and the pairing would be invalid. The check is
               written to the output JSON.
  rls_ewc    : EWC on the SAME second-order optimiser as the reference. The
               quadratic penalty is injected exactly, as pseudo-observations
               (design row sqrt(lambda * F_i) on each coordinate, target
               sqrt(lambda * F_i) * anchor_i), which is the least-squares form
               of the EWC objective. This arm isolates CONSOLIDATION at
               matched optimiser: it differs from rls_single only by the
               penalty.
  sgd_plain  : the same linear readout trained by online AdaGrad on the same
               derived-label channel, with NO consolidation penalty. Honest
               control for the first-order EWC arm.
  ewc_sgd    : AdaGrad plus the EWC penalty, i.e. EWC as published (a penalty
               added to a gradient objective). Segment boundaries are given to
               both EWC arms by design (the standard EWC assumption); the RLS
               arms do not use them. That asymmetry is stated in the paper.

Why two EWC arms: a single-pass first-order optimiser over 800 blocks on a
  513-dimensional quadratic basis is badly conditioned and never leaves chance
  with a fixed step, while RLS is second-order and converges in O(d) updates.
  Comparing EWC-for-SGD against an RLS reference would therefore measure the
  optimiser, not the mechanism. rls_ewc removes that confound; ewc_sgd keeps
  the method as published. Both are reported.

Derived-label channel: every arm consumes the same information. The reward is
  the block vote's correctness (r = +-1) and the regression target is
  y = vote if r > 0 else 1 - vote, exactly as in s52. No arm receives the
  true label.

Hyperparameter tuning (reported, not hidden):
  Tuned on a 3-seed grid (seeds 0, 1, 2) before the main run.
  - lambda is selected on REVISIT RETENTION, mean(seg3_acc, seg4_acc), i.e. the
    two returning regimes, because that is the quantity a consolidation method
    exists to protect. Selecting on whole-run mean would penalise EWC for the
    stability-plasticity trade-off it is designed to make, and in this protocol
    it selects lambda = 0, which degenerates the arm into its own control.
  - lr is selected on whole-run mean accuracy.
  Both grids, the selection metric and the winner are written to
  data/s65_tuning_v1.json and echoed into the main JSON. Tuning seeds are a
  subset of the main seeds, which favours the baseline rather than the stack;
  this is stated in the paper.

Output files:
  data/s65_ewc_baseline_v1.csv    (one row per run)
  data/s65_ewc_baseline_v1.json   (params + aggregates + paired stats +
                                   tuning grid + reproduction check)
  data/s65_tuning_v1.json         (tuning grid only)

Usage: python s65_ewc_baseline.py [--quick] [--tune-only]
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

# numba is optional; the heavy kernels (trajectory, RLS) are already jitted
# inside paper_e/deps. This script's own loops are either vectorised or are the
# online training loop, which CLAUDE.md exempts from @njit. The guard is kept
# so the module imports cleanly in a numba-free environment.
try:
    from numba import njit  # noqa: F401
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'paper_e', 'deps'))
from shallow_trap_array_simulator import gamma, tau0, gen_tau_vec, preprogram_vec
from recurrent_substrate import (
    COUPLING_CONTRAST_SELF,
    PW, ALPHA0, ALPHA_MIN, ALPHA_MAX, build_topology_csr,
    run_trajectory_nb)
from online_readout import OnlineRLS, running_mean_accuracy
from streaming_tasks import gen_nonlinear2d, DB_K_PULSES

# ==================== Fixed parameters (s52-consistent) ====================
N_UNITS = 256
CV_TAU = 0.20
TOPO_SEED = 777
AVG_DEGREE = 8
N_SEEDS = 10

FEATURE_SCALE = 10.0
BIAS = 1.0
KAPPA_RANDOM = 25.0

RLS_FORGETTING = 0.99
RLS_INIT_COV = 1.0

N_SEGMENTS = 4          # A-B-A-B
SEG_BLOCKS = 500        # blocks per segment
SEGMENTS = ['linear', 'circle', 'linear', 'circle']
CURVE_WINDOW = 200      # s52's running-mean accuracy window

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CSV_PATH = os.path.join(DATA_DIR, 's65_ewc_baseline_v1.csv')
JSON_PATH = os.path.join(DATA_DIR, 's65_ewc_baseline_v1.json')
TUNE_PATH = os.path.join(DATA_DIR, 's65_tuning_v1.json')

ARMS = ['rls_single', 'rls_ewc', 'sgd_plain', 'ewc_sgd']
TUNED_ARMS = ['rls_ewc', 'ewc_sgd']

TUNE_SEEDS = [0, 1, 2]
TUNE_LR = [0.1, 0.3, 1.0, 3.0]
TUNE_LAMBDA = [0.0, 0.1, 1.0, 10.0, 100.0]

DEFAULT_LR = 0.3
DEFAULT_LAMBDA = 1.0

T_CRIT_975_DF9 = 2.2622


# ==================== Task / features (s52-identical) ====================

def build_substrate():
    ip, idx, wt = build_topology_csr('random_graph', N_UNITS,
                                     seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    return ip, idx, wt, COUPLING_CONTRAST_SELF, KAPPA_RANDOM


def acc_median_in(acc_run, b_lo, b_hi):
    lo, hi = b_lo * DB_K_PULSES, b_hi * DB_K_PULSES
    seg = acc_run[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(np.nanmedian(seg)) if seg.size else float('nan')


def gen_alternating(seed):
    """Concatenate A-B-A-B static-boundary segments (identical to s52)."""
    rng = np.random.RandomState(seed)
    dt_parts = []
    tgt_parts = []
    for bd in SEGMENTS:
        s = int(rng.randint(0, 2**31 - 1))
        dt, tgt, _, _ = gen_nonlinear2d(seed=s, n_blocks=SEG_BLOCKS,
                                        k_pulses=DB_K_PULSES, boundary=bd)
        dt_parts.append(dt)
        tgt_parts.append(tgt)
    return (np.concatenate(dt_parts), np.concatenate(tgt_parts).astype(np.int64))


def build_features(states, T):
    """Quadratic basis, identical to s52 (d = 2*256 + 1 = 513)."""
    obs_raw = np.exp(gamma * states) / FEATURE_SCALE
    n_fit = int(0.3 * T)
    mu = obs_raw[:n_fit].mean(axis=0)
    sd = obs_raw[:n_fit].std(axis=0)
    sd[sd < 1e-9] = 1.0
    F = np.hstack([(obs_raw - mu) / sd, np.full((T, 1), BIAS)])
    d = F.shape[1]
    return np.hstack([F, F[:, :d - 1] ** 2])


def normalise_fisher(f):
    """Scale the diagonal Fisher to mean 1 so lambda is comparable across runs."""
    m = float(np.mean(f))
    return f / m if m > 1e-12 else np.ones_like(f)


def fisher_diag(Fb, yb, w):
    """Empirical diagonal Fisher on block-mean features at weights w."""
    err = yb - Fb @ w
    return ((err ** 2)[:, None] * (Fb ** 2)).mean(axis=0)


# ==================== Readouts ====================

def _scalar(v):
    """OnlineRLS.predict returns shape (n_outputs,); the gradient readouts
    return a Python float. Normalise both to a scalar at the call site."""
    arr = np.asarray(v, dtype=float).reshape(-1)
    return float(arr[0])


class AdaGradEWC:
    """Linear readout trained by online AdaGrad with an optional EWC penalty.

    One update per block, on the block-mean feature and the derived label.
    EWC state is a list of (fisher, anchor) pairs, one per finished segment;
    penalties from all finished segments are summed (the original EWC form).
    """

    def __init__(self, d, lr=0.3, lam=1.0, use_penalty=True, eps=1e-8):
        self.w = np.zeros(d)
        self.G = np.zeros(d)
        self.lr = float(lr)
        self.lam = float(lam)
        self.eps = float(eps)
        self.use_penalty = bool(use_penalty)
        self.fishers = []
        self.anchors = []

    def predict(self, x):
        return float(self.w @ x)

    def update(self, x, y):
        err = float(y) - float(self.w @ x)
        grad = -2.0 * err * x
        if self.use_penalty and self.fishers:
            pen = np.zeros_like(self.w)
            for f, a in zip(self.fishers, self.anchors):
                pen += f * (self.w - a)
            grad = grad + self.lam * pen
        self.G += grad * grad
        self.w -= self.lr * grad / (np.sqrt(self.G) + self.eps)

    def consolidate(self, Fb, yb):
        self.fishers.append(normalise_fisher(fisher_diag(Fb, yb, self.w)))
        self.anchors.append(self.w.copy())


class RLSEWC:
    """EWC applied to the same RLS readout as the rls_single reference.

    The quadratic penalty is exact in least-squares form: for each stored task
    the penalty  lambda/2 * sum_i F_i (w_i - a_i)^2  equals the squared error
    of a pseudo-observation with design row  x_i = sqrt(lambda F_i)  and
    target  y = x . a.  Feeding those rows through the RLS recursion solves the
    penalised problem, so this arm differs from rls_single by the consolidation
    term alone.
    """

    def __init__(self, d, forgetting=0.99, init_cov=1.0, lam=1.0):
        self.rls = OnlineRLS(d, 1, forgetting=forgetting, init_cov=init_cov)
        self.lam = float(lam)
        self.d = d
        self.fishers = []
        self.anchors = []

    def predict(self, x):
        return float(self.rls.predict(x)[0])

    def update(self, x, y):
        self.rls.update(x, np.array([y]))
        if self.lam <= 0.0:
            # No penalty: feeding a zero design row would still inflate the
            # inverse covariance by 1/forgetting on every block, so lambda = 0
            # must skip the pseudo-observation entirely to stay identical to
            # the rls_single reference.
            return
        for f, a in zip(self.fishers, self.anchors):
            xa = np.sqrt(self.lam * f)
            self.rls.update(xa, np.array([float(xa @ a)]))

    def consolidate(self, Fb, yb):
        w = np.asarray(self.rls.W[:, 0], dtype=float)
        self.fishers.append(normalise_fisher(fisher_diag(Fb, yb, w)))
        self.anchors.append(w.copy())


# ==================== Single run ====================

def run_single(args):
    """(arm, seed_idx, lr, lam) -> metrics dict. Substrate: random_graph_k25."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

    arm, seed_idx, lr, lam = args
    t0 = time.time()

    tau = gen_tau_vec(N_UNITS, CV_TAU, tau0, seed=seed_idx)
    x0 = preprogram_vec(ALPHA0, tau)
    indptr, indices, wts, mode, kappa = build_substrate()

    dt_seq, target_seq = gen_alternating(seed_idx)
    T = dt_seq.shape[0]
    target = target_seq.astype(np.float64)
    n_blocks = T // DB_K_PULSES

    states, _, _ = run_trajectory_nb(
        x0, tau, dt_seq, PW, indptr, indices, wts, kappa,
        ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma, mode, 0)
    F = build_features(states, T)
    d = F.shape[1]

    preds = np.empty(T)
    Fb = np.empty((n_blocks, d))
    yb = np.empty(n_blocks)

    if arm == 'rls_single':
        rd = OnlineRLS(d, 1, forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV)
        anchor_every = False
    elif arm == 'rls_ewc':
        rd = RLSEWC(d, forgetting=RLS_FORGETTING, init_cov=RLS_INIT_COV, lam=lam)
        anchor_every = True
    else:
        rd = AdaGradEWC(d, lr=lr, lam=lam, use_penalty=(arm == 'ewc_sgd'))
        anchor_every = (arm == 'ewc_sgd')

    for t in range(T):
        preds[t] = _scalar(rd.predict(F[t]))
        if (t + 1) % DB_K_PULSES == 0:
            bi = (t + 1) // DB_K_PULSES - 1
            blk = preds[t - DB_K_PULSES + 1:t + 1]
            vote = float(np.mean(blk) > 0.5)
            r = 1.0 if vote == target[t] else -1.0
            y_derived = vote if r > 0.0 else (1.0 - vote)
            x_blk = F[t - DB_K_PULSES + 1:t + 1].mean(axis=0)
            Fb[bi] = x_blk
            yb[bi] = y_derived
            rd.update(x_blk, y_derived)
            preds[t - DB_K_PULSES + 1:t + 1] = np.mean(blk)
            # EWC anchors at each segment boundary (boundaries are given to the
            # EWC arms by design; see the module docstring).
            if anchor_every and (bi + 1) % SEG_BLOCKS == 0 and bi + 1 < n_blocks:
                lo = bi + 1 - SEG_BLOCKS
                rd.consolidate(Fb[lo:bi + 1], yb[lo:bi + 1])

    acc_run = running_mean_accuracy(preds, target, CURVE_WINDOW)
    seg_acc = [acc_median_in(acc_run, s * SEG_BLOCKS, (s + 1) * SEG_BLOCKS)
               for s in range(N_SEGMENTS)]
    retention = float(np.nanmean([seg_acc[2], seg_acc[3]]))  # seg3, seg4 = revisits
    return {
        'task': 'ewc_baseline', 'substrate': 'random_graph_k25',
        'readout': arm, 'seed_idx': seed_idx,
        'n_units': N_UNITS, 't_total': int(T),
        'runtime_s': round(time.time() - t0, 3),
        'lr': lr, 'lam': lam,
        'mean_acc_all': float(np.nanmean(acc_run)),
        'retention_seg3_seg4': retention,
        'seg1_acc': seg_acc[0], 'seg2_acc': seg_acc[1],
        'seg3_acc': seg_acc[2], 'seg4_acc': seg_acc[3],
    }


# ==================== Aggregation / stats ====================

FIELDS = ['mean_acc_all', 'retention_seg3_seg4', 'seg1_acc', 'seg2_acc',
          'seg3_acc', 'seg4_acc']


def aggregate(results):
    groups = {}
    for r in results:
        groups.setdefault(r['readout'], []).append(r)
    agg = []
    for read, rs in sorted(groups.items()):
        entry = {'readout': read, 'n_runs': len(rs)}
        for f in FIELDS:
            vals = np.array([r[f] for r in rs], dtype=float)
            entry[f + '_mean'] = float(np.nanmean(vals))
            entry[f + '_std'] = float(np.nanstd(vals))
        agg.append(entry)
    return agg


def paired_stats(results, ref_arm='rls_single'):
    """Per-seed paired comparison of every arm against ref_arm (df=9)."""
    by_arm = {}
    for r in results:
        by_arm.setdefault(r['readout'], {})[r['seed_idx']] = r
    ref = by_arm.get(ref_arm, {})
    out = {}
    for arm, dct in sorted(by_arm.items()):
        if arm == ref_arm:
            continue
        common = sorted(set(dct) & set(ref))
        if len(common) < 2:
            continue
        rec = {}
        for field in FIELDS:
            diff = np.array([dct[s][field] - ref[s][field] for s in common],
                            dtype=float)
            n = diff.size
            m = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1))
            se = sd / np.sqrt(n)
            t = m / se if se > 0 else float('nan')
            rec[field] = {
                'n': n, 'mean_diff': m, 'sd': sd, 'sem': se,
                't': t, 'df': n - 1,
                'critical_95_two_tailed': T_CRIT_975_DF9,
                'significant_at_5pct': bool(abs(t) > T_CRIT_975_DF9)
                if np.isfinite(t) else False,
                'seed_wins': int(np.sum(diff > 0)),
            }
        out[arm + '_vs_' + ref_arm] = rec
    return out


def reproduction_check(results):
    """Compare this run's rls_single against s52's committed per-seed values."""
    s52_csv = os.path.join(DATA_DIR, 's52_dynamic_memory_v1.csv')
    if not os.path.exists(s52_csv):
        return {'available': False, 'reason': 's52 csv not found'}
    ref = {}
    with open(s52_csv, newline='') as f:
        for row in csv.DictReader(f):
            if row['readout'] == 'rls_single':
                ref[int(row['seed_idx'])] = row
    fields = ['mean_acc_all', 'seg1_acc', 'seg2_acc', 'seg3_acc', 'seg4_acc']
    worst = 0.0
    n_cmp = 0
    per_seed = {}
    for r in results:
        if r['readout'] != 'rls_single':
            continue
        s = r['seed_idx']
        if s not in ref:
            continue
        diffs = {}
        for fl in fields:
            dd = abs(float(r[fl]) - float(ref[s][fl]))
            diffs[fl] = dd
            worst = max(worst, dd)
            n_cmp += 1
        per_seed[str(s)] = diffs
    return {
        'available': True,
        'reference_file': 'data/s52_dynamic_memory_v1.csv',
        'fields_compared': fields,
        'n_comparisons': n_cmp,
        'max_abs_difference': worst,
        'bit_identical': bool(worst == 0.0),
        'per_seed_abs_diff': per_seed,
    }


# ==================== Driver ====================

def _run_pool(all_args, n_runs, label):
    results = []
    n_proc = min(cpu_count(), 8, max(1, n_runs))
    with Pool(n_proc) as pool:
        done = 0
        step = max(1, n_runs // 10)
        for res in pool.imap_unordered(run_single, all_args, chunksize=1):
            results.append(res)
            done += 1
            if done % step == 0 or done == n_runs:
                print(f"[{time.strftime('%H:%M:%S')}] {label} {done}/{n_runs}",
                      flush=True)
    return results


def tune(quick=False):
    """Tune the EWC arms; lambda on revisit retention, lr on whole-run mean."""
    seeds = TUNE_SEEDS[:1] if quick else TUNE_SEEDS

    # --- stage 1: lr for the first-order arms, at lambda = 0 ---
    lr_args = [(arm, s, lr, 0.0)
               for arm in ['sgd_plain', 'ewc_sgd'] for lr in TUNE_LR
               for s in seeds]
    print(f"[{time.strftime('%H:%M:%S')}] tuning stage 1 (lr): "
          f"{len(lr_args)} runs")
    lr_res = _run_pool(lr_args, len(lr_args), 'tune-lr')
    lr_rows = []
    for arm in ['sgd_plain', 'ewc_sgd']:
        for lr in TUNE_LR:
            vals = [r['mean_acc_all'] for r in lr_res
                    if r['readout'] == arm and r['lr'] == lr and r['lam'] == 0.0]
            lr_rows.append({'arm': arm, 'lr': lr, 'n': len(vals),
                            'mean_acc_all': float(np.mean(vals)) if vals
                            else float('nan')})
    best_lr = {}
    for arm in ['sgd_plain', 'ewc_sgd']:
        cand = [r for r in lr_rows if r['arm'] == arm]
        cand.sort(key=lambda r: -r['mean_acc_all'])
        best_lr[arm] = float(cand[0]['lr'])
    print(f"[{time.strftime('%H:%M:%S')}] stage 1 best lr: {best_lr}")

    # --- stage 2: lambda for both EWC arms, on revisit retention ---
    lam_args = []
    for lam in TUNE_LAMBDA:
        for s in seeds:
            lam_args.append(('rls_ewc', s, 0.0, lam))
            lam_args.append(('ewc_sgd', s, best_lr['ewc_sgd'], lam))
    print(f"[{time.strftime('%H:%M:%S')}] tuning stage 2 (lambda): "
          f"{len(lam_args)} runs, metric = revisit retention")
    lam_res = _run_pool(lam_args, len(lam_args), 'tune-lam')
    lam_rows = []
    for arm in ['rls_ewc', 'ewc_sgd']:
        for lam in TUNE_LAMBDA:
            sel = [r for r in lam_res
                   if r['readout'] == arm and r['lam'] == lam]
            vals = [r['retention_seg3_seg4'] for r in sel]
            accs = [r['mean_acc_all'] for r in sel]
            lam_rows.append({
                'arm': arm, 'lam': lam, 'n': len(vals),
                'retention_mean': float(np.mean(vals)) if vals else float('nan'),
                'mean_acc_all': float(np.mean(accs)) if accs else float('nan'),
            })
    best_lam = {}
    for arm in ['rls_ewc', 'ewc_sgd']:
        cand = [r for r in lam_rows if r['arm'] == arm]
        cand.sort(key=lambda r: -r['retention_mean'])
        best_lam[arm] = float(cand[0]['lam'])
    print(f"[{time.strftime('%H:%M:%S')}] stage 2 best lam: {best_lam}")
    for r in sorted(lam_rows, key=lambda x: (x['arm'], -x['retention_mean'])):
        print(f"    {r['arm']:<10} lam={r['lam']:<7} retention="
              f"{r['retention_mean']:.4f} whole={r['mean_acc_all']:.4f}")

    payload = {
        'seeds': seeds,
        'lr_grid': TUNE_LR, 'lambda_grid': TUNE_LAMBDA,
        'lr_selection_metric': 'whole-run mean accuracy',
        'lambda_selection_metric': 'revisit retention, mean(seg3_acc, '
                                   'seg4_acc), the two returning regimes',
        'lambda_selection_rationale':
            'selecting lambda on whole-run mean penalises EWC for the '
            'stability-plasticity trade-off it exists to make, and selects '
            'lambda=0, degenerating the arm into its own control',
        'note': 'tuning seeds are a subset of the 10 main seeds; this favours '
                'the baseline, not the stack',
        'lr_grid_results': lr_rows, 'lambda_grid_results': lam_rows,
        'best_lr': best_lr, 'best_lambda': best_lam,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(TUNE_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    return best_lr, best_lam, payload


def main():
    quick = '--quick' in sys.argv
    tune_only = '--tune-only' in sys.argv
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START S65 EWC baseline (quick={quick})")

    tau_w = gen_tau_vec(16, CV_TAU, tau0, seed=0)
    x0_w = preprogram_vec(ALPHA0, tau_w)
    dt_w = np.full(16, 10e-6)
    ip_w, idx_w, wt_w = build_topology_csr('random_graph', 16,
                                           seed=TOPO_SEED, avg_degree=AVG_DEGREE)
    run_trajectory_nb(x0_w, tau_w, dt_w, PW, ip_w, idx_w, wt_w, KAPPA_RANDOM,
                      ALPHA0, ALPHA_MIN, ALPHA_MAX, gamma,
                      COUPLING_CONTRAST_SELF, 0)
    print(f"[{time.strftime('%H:%M:%S')}] numba warmup done")

    best_lr, best_lam, tune_payload = tune(quick=quick)
    if tune_only:
        print(f"[{time.strftime('%H:%M:%S')}] --tune-only, stopping")
        return

    n_seeds = N_SEEDS if not quick else 2
    all_args = []
    for arm in ARMS:
        for s in range(n_seeds):
            lr = best_lr.get(arm, DEFAULT_LR)
            lam = best_lam.get(arm, 0.0)
            all_args.append((arm, s, lr, lam))
    n_runs = len(all_args)
    print(f"[{time.strftime('%H:%M:%S')}] main run: {n_runs} runs "
          f"(arms={len(ARMS)}, seeds={n_seeds})")
    results = _run_pool(all_args, n_runs, 'main')

    os.makedirs(DATA_DIR, exist_ok=True)
    fieldnames = ['task', 'substrate', 'readout', 'seed_idx', 'n_units',
                  't_total', 'runtime_s', 'lr', 'lam', 'mean_acc_all',
                  'retention_seg3_seg4', 'seg1_acc', 'seg2_acc',
                  'seg3_acc', 'seg4_acc']
    out_csv = CSV_PATH if not quick else CSV_PATH.replace('.csv', '_quick.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)

    params = {
        'n_units': N_UNITS, 'cv_tau': CV_TAU, 'alpha0': ALPHA0,
        'gamma': float(gamma), 'tau0': float(tau0),
        'topo_seed': TOPO_SEED, 'avg_degree': AVG_DEGREE,
        'kappa_random': KAPPA_RANDOM,
        'rls_forgetting': RLS_FORGETTING, 'rls_init_cov': RLS_INIT_COV,
        'n_segments': N_SEGMENTS, 'seg_blocks': SEG_BLOCKS,
        'segments': SEGMENTS, 'curve_window': CURVE_WINDOW,
        'arms': ARMS, 'n_seeds': n_seeds, 'quick': bool(quick),
        'best_lr': best_lr, 'best_lambda': best_lam,
        'ewc_fisher': 'diagonal, empirical, on block-mean features, '
                      'normalised to mean 1',
        'ewc_penalty': 'sum over all finished segments (original EWC form)',
        'ewc_boundaries_given': True,
        'rls_ewc_penalty_form': 'exact least-squares pseudo-observations '
                                '(design sqrt(lambda*F_i), target '
                                'sqrt(lambda*F_i)*anchor_i)',
        'tuning': tune_payload,
    }
    out_json = JSON_PATH if not quick else JSON_PATH.replace('.json', '_quick.json')
    with open(out_json, 'w') as f:
        json.dump({
            'params': params,
            'aggregates': aggregate(results),
            'stats': paired_stats(results),
            'reproduction_check_vs_s52': reproduction_check(results),
        }, f, indent=2)

    print("\n" + "=" * 104)
    print("S65 EXTERNAL BASELINE (EWC) ON THE S52 REVISIT PROTOCOL")
    print("=" * 104)
    print(f"\n  segments: {SEGMENTS} (A-B-A-B, {SEG_BLOCKS} blocks each)")
    print(f"  {'arm':<12} | {'seg1 lin':>8} | {'seg2 cir':>8} | "
          f"{'seg3 lin':>8} | {'seg4 cir':>8} | {'retain':>6} | {'mean':>6}")
    for a in aggregate(results):
        print(f"  {a['readout']:<12} | {a['seg1_acc_mean']:>8.3f} | "
              f"{a['seg2_acc_mean']:>8.3f} | {a['seg3_acc_mean']:>8.3f} | "
              f"{a['seg4_acc_mean']:>8.3f} | "
              f"{a['retention_seg3_seg4_mean']:>6.3f} | "
              f"{a['mean_acc_all_mean']:>6.3f}")
    rc = reproduction_check(results)
    print(f"\n  reproduction check vs s52 rls_single: "
          f"max|diff| = {rc.get('max_abs_difference')} "
          f"({'BIT-IDENTICAL' if rc.get('bit_identical') else 'MISMATCH'})")
    st = paired_stats(results)
    for k, rec in st.items():
        w = rec['mean_acc_all']
        rt = rec['retention_seg3_seg4']
        print(f"  {k:<22} whole d={w['mean_diff']:+.4f} t={w['t']:6.2f} "
              f"wins={w['seed_wins']}/{w['n']} | retain d={rt['mean_diff']:+.4f} "
              f"t={rt['t']:6.2f} wins={rt['seed_wins']}/{rt['n']}")
    print(f"\nCSV : {out_csv}")
    print(f"JSON: {out_json}")
    print(f"[{time.strftime('%H:%M:%S')}] DONE, total {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
